"""
Weighted-MLM Defense
핵심 철학: Claim-bearing span(팩트 변조 대상)에 가중치 부여.

가중치 소스 (hybrid):
- NER span (spaCy: PERSON, ORG, GPE, DATE, CARDINAL, MONEY 등)
- regex 숫자/연도/퍼센트/통화/서수
- 질문 타입 조건부 (who→PERSON, when→DATE 등)
- 질문 키워드 포함 문장 내 span → 추가 가중치
- NER/regex 없을 때만 heuristic fallback
"""
import torch
import numpy as np
import re
from typing import Dict, List, Tuple, Set, Optional
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    from claim_bearing_span import build_span_weights
    _CLAIM_BEARING_AVAILABLE = True
except ImportError:
    _CLAIM_BEARING_AVAILABLE = False


class WeightedMLMDefense:
    """
    Weighted-MLM Defense
    
    핵심 아이디어:
    - 모든 토큰을 마스킹하여 표본 수 확보 (안정성)
    - Entity/숫자/Query 키워드 토큰이 틀리면 벌점 2배 (Novelty)
    - 가중 평균 PLL 계산
    """
    def __init__(self, model_name="bert-base-cased", device="auto"):
        """
        model_name: MLM 백본. bert-base-cased(110M) | roberta-large(355M) | microsoft/deberta-v3-large
        논문 디펜스: "모델-Agnostic. 백본 언어 이해력 ↑ → 시빌 방어율 선형 증가"
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        self.device = device
        # RoBERTa 계열: pad_token_id 미설정 시 eos_token_id 사용
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def extract_question_keywords(self, question: str) -> Set[str]:
        """질문에서 키워드 추출"""
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'who', 
                    'where', 'when', 'why', 'how', 'which', 'to', 'of', 'in', 'on', 
                    'at', 'for', 'with', 'by', 'from', 'as', 'be', 'been', 'have', 
                    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could'}
        
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = {w for w in words if w not in stopwords and len(w) > 2}
        return keywords
    
    def is_entity_token(self, token_text: str) -> bool:
        """토큰이 Entity(고유명사)인지 확인"""
        # 대문자로 시작하는 단어 (고유명사)
        if token_text and token_text[0].isupper() and len(token_text) > 2:
            # 일반적인 단어 제외
            if token_text.lower() not in ['the', 'this', 'that', 'these', 'those', 'there', 'here']:
                return True
        return False
    
    def is_number_token(self, token_text: str) -> bool:
        """토큰이 숫자인지 확인"""
        # 숫자 패턴
        if re.match(r'^\d+[\d,]*\.?\d*$', token_text):
            return True
        return False
    
    def normalize_token(self, token_text: str) -> str:
        """토큰 정규화 (TF 계산용)"""
        token = token_text.lower().replace('##', '').replace('Ġ', ' ').strip()
        token = re.sub(r'[^a-z0-9]', '', token)
        return token

    def build_token_tf(self, token_texts: List[str], valid_indices: np.ndarray) -> Dict[str, int]:
        """문서 내 토큰 빈도 계산 (TF)"""
        tf: Dict[str, int] = {}
        for idx in valid_indices:
            tok = self.normalize_token(token_texts[idx])
            if not tok:
                continue
            tf[tok] = tf.get(tok, 0) + 1
        return tf
    
    def find_sentence_boundaries(self, document: str) -> List[Tuple[int, int]]:
        """문장 경계 찾기 (시작 위치, 끝 위치)"""
        sentences = []
        current_start = 0
        for match in re.finditer(r'[.!?]\s+', document):
            sentences.append((current_start, match.end()))
            current_start = match.end()
        if current_start < len(document):
            sentences.append((current_start, len(document)))
        return sentences
    
    def compute_weighted_mlm_score(
        self,
        question: str,
        document: str,
        max_length: int = 512,
        use_query_entity_intersection: bool = True,
        use_importance_weighting: bool = False,
        importance_alpha: float = 0.6,
        importance_cap: float = 2.5,
        importance_scope: str = "entity_query",
        entity_weight: float = 1.5,
        query_weight: float = 1.3,
        intersection_weight: float = 3.5,
        mask_batch_size: int = 32,
        use_selective_masking: bool = True,
        use_nonlinear_penalty: bool = False,
        nonlinear_threshold: float = -2.0,
        nonlinear_mode: str = "square",
        use_claim_bearing_spans: bool = True,
    ) -> dict:
        """
        Weighted-MLM 점수 계산
        
        use_selective_masking=True: Entity/숫자/Query 토큰만 마스킹 (팩트 검증 집중, 5~10배 단축)
        use_selective_masking=False: 모든 토큰 마스킹 (기존 방식)
        
        Returns:
            dict with keys:
                - score: 최종 가중 평균 PLL (높을수록 자연스러움)
                - unweighted_pll: 가중치 없는 평균 PLL
                - weighted_pll: 가중 평균 PLL
                - entity_count: Entity/숫자 토큰 개수
                - query_count: Query 키워드 토큰 개수
        """
        if not document or len(document.strip()) == 0:
            return {
                'score': -999.0,
                'unweighted_pll': -999.0,
                'weighted_pll': -999.0,
                'entity_count': 0,
                'query_count': 0
            }
        
        # 1. 질문 키워드 추출 (string 기반만 사용, token-id equality 제거)
        question_keywords = self.extract_question_keywords(question)

        # 2. 토큰화
        inputs = self.tokenizer(
            document,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
            return_offsets_mapping=True
        ).to(self.device)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_ids = input_ids[0].cpu().numpy()
        
        # 3. 유효한 토큰 인덱스 (PAD, CLS, SEP 제외)
        valid_mask = (token_ids != self.tokenizer.pad_token_id) & \
                     (token_ids != self.tokenizer.cls_token_id) & \
                     (token_ids != self.tokenizer.sep_token_id)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return {
                'score': -999.0,
                'unweighted_pll': -999.0,
                'weighted_pll': -999.0,
                'entity_count': 0,
                'query_count': 0
            }
        
        # 4. 토큰 텍스트 가져오기
        token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
        token_tf = self.build_token_tf(token_texts, valid_indices) if use_importance_weighting else {}
        max_tf = max(token_tf.values()) if token_tf else 1
        
        # 5. 각 토큰의 가중치 계산
        # 개선: Query-Entity Intersection 적용
        token_weights = {}
        entity_count = 0
        query_count = 0
        intersection_count = 0
        
        # Query-Entity Intersection: 질문 키워드가 포함된 문장
        query_relevant_sentences = set()
        if use_query_entity_intersection and question_keywords:
            sentences = self.find_sentence_boundaries(document)
            for sent_start, sent_end in sentences:
                sentence_text = document[sent_start:sent_end].lower()
                if any(kw in sentence_text for kw in question_keywords):
                    query_relevant_sentences.add((sent_start, sent_end))

        offsets_raw = inputs.get("offset_mapping", None)
        if offsets_raw is None:
            offsets_list = [(0, 0)] * len(token_ids)
        else:
            o0 = offsets_raw[0]
            if hasattr(o0, "cpu"):
                raw_list = o0.cpu().tolist()
            else:
                raw_list = list(o0)

            def _to_int(v):
                return int(v.item()) if hasattr(v, "item") else int(v)

            offsets_list = [(_to_int(x[0]), _to_int(x[1])) for x in raw_list]

        # Claim-bearing span 가중치 (NER + regex + 질문타입). 문장 boost는 build_span_weights 내부에서만 적용
        span_weights = {}
        if use_claim_bearing_spans and _CLAIM_BEARING_AVAILABLE:
            span_weights = build_span_weights(
                document, question, offsets_list,
                question_keywords=question_keywords,
                use_ner=True, use_numeric_regex=True, use_heuristic_fallback=True,
            )

        use_legacy_fallback = (not use_claim_bearing_spans) or (len(span_weights) == 0)

        for idx in valid_indices:
            token_text = token_texts[idx].replace('##', '').replace('Ġ', ' ').strip()
            token_norm = self.normalize_token(token_text)
            is_query = token_norm in question_keywords
            is_entity = False
            is_intersection = False

            weight = 1.0
            if idx in span_weights:
                weight = span_weights[idx]
                is_entity = True
                entity_count += 1
                token_start = offsets_list[idx][0] if idx < len(offsets_list) else 0
                if use_query_entity_intersection and query_relevant_sentences and any(s <= token_start < e for s, e in query_relevant_sentences):
                    is_intersection = True
                    intersection_count += 1
            elif use_legacy_fallback:
                is_entity = self.is_entity_token(token_text) or self.is_number_token(token_text)
                if use_query_entity_intersection and is_entity and query_relevant_sentences:
                    token_start = offsets_list[idx][0] if idx < len(offsets_list) else 0
                    if any(s <= token_start < e for s, e in query_relevant_sentences):
                        is_intersection = True
                        weight = intersection_weight
                        intersection_count += 1
                        entity_count += 1
                    else:
                        weight = entity_weight
                        entity_count += 1
                elif is_entity:
                    weight = entity_weight
                    entity_count += 1

            if is_query:
                weight = max(weight, query_weight)
                query_count += 1

            # 중요 단어 가중치 (TF 기반 희소 단어 강조)
            if use_importance_weighting:
                token_norm = self.normalize_token(token_text)
                tf = token_tf.get(token_norm, 0)
                if tf > 0:
                    # 희소 단어일수록 큰 값이 되도록 조정
                    # importance >= 1.0
                    importance = (np.log1p(max_tf) - np.log1p(tf)) + 1.0
                    multiplier = 1.0 + importance_alpha * (importance - 1.0)
                    multiplier = min(importance_cap, max(1.0, multiplier))
                    if importance_scope == "all":
                        weight *= multiplier
                    elif importance_scope == "entity_query":
                        if is_entity or is_query:
                            weight *= multiplier
                    elif importance_scope == "intersection":
                        if is_intersection:
                            weight *= multiplier
            
            token_weights[idx] = weight
        
        # 6. Selective Masking: Entity/숫자/Query 토큰만 마스킹 (형식이 아닌 팩트 검증에 집중)
        #    use_selective_masking=True 시 가중치>1.0인 토큰만 마스킹 → 5~10배 추론 단축
        if use_selective_masking:
            mask_indices_list = [idx for idx in valid_indices if token_weights.get(idx, 1.0) > 1.0]
            if len(mask_indices_list) < 3:
                mask_indices_list = valid_indices.tolist()
        else:
            mask_indices_list = valid_indices.tolist()
        
        log_probs = []
        weights = []
        
        with torch.no_grad():
            for start in range(0, len(mask_indices_list), mask_batch_size):
                batch_indices = mask_indices_list[start:start + mask_batch_size]
                batch_size = len(batch_indices)
                batch_input_ids = input_ids.repeat(batch_size, 1)
                batch_attention = attention_mask.repeat(batch_size, 1)
                for row, idx in enumerate(batch_indices):
                    batch_input_ids[row, idx] = self.tokenizer.mask_token_id
                outputs = self.model(batch_input_ids, attention_mask=batch_attention)
                logits = outputs.logits
                for row, idx in enumerate(batch_indices):
                    original_token_id = token_ids[idx]
                    log_prob = torch.log_softmax(logits[row, idx], dim=0)[original_token_id].item()
                    log_probs.append(log_prob)
                    weights.append(token_weights.get(idx, 1.0))
        
        if len(log_probs) == 0:
            return {
                'score': -999.0,
                'unweighted_pll': -999.0,
                'weighted_pll': -999.0,
                'entity_count': 0,
                'query_count': 0
            }
        
        # 7. 가중 평균 PLL 계산 (+ 비선형 페널티: log P < threshold & weight>1 → 가중치^2 또는 exp)
        unweighted_pll = np.mean(log_probs)
        eff_weights = weights
        if use_nonlinear_penalty:
            eff_weights = []
            for lp, w in zip(log_probs, weights):
                if lp < nonlinear_threshold and w > 1.0:
                    eff_weights.append(w ** 2 if nonlinear_mode == "square" else float(np.exp(min(w, 10))))
                else:
                    eff_weights.append(w)
        weighted_pll = np.average(log_probs, weights=eff_weights)
        
        # 최종 점수는 가중 평균 PLL
        # 낮은 PLL (틀린 토큰)에 더 큰 벌점을 주기 위해 가중치 적용
        # Entity/Query 토큰이 틀리면 (낮은 PLL) 가중치 2.0이 적용되어 전체 점수에 더 큰 영향
        final_score = weighted_pll
        
        return {
            'score': final_score,
            'unweighted_pll': unweighted_pll,
            'weighted_pll': weighted_pll,
            'entity_count': entity_count,
            'query_count': query_count,
            'intersection_count': intersection_count,
        }

    def _compute_query_log_probs(self, question: str, max_length: int = 512) -> torch.Tensor:
        """
        log P(t) for each vocab t. Unconditioned prior (정보 이론적 기준선).
        [CLS] [MASK] [SEP]만 사용하여 '질문 다음 단어'가 아닌 순수 토큰 등장 확률을 구함.
        log P(t|D) - log P(t) 형태의 PMI가 수학적으로 타당함.
        """
        mask_id = self.tokenizer.mask_token_id
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        input_ids = torch.tensor(
            [[cls_id, mask_id, sep_id]],
            device=self.device,
            dtype=torch.long
        )
        with torch.no_grad():
            logits = self.model(input_ids).logits
        log_probs = torch.log_softmax(logits[0, 1, :], dim=-1)  # MASK 위치 = index 1
        return log_probs

    def compute_batch_weighted_mlm_score(
        self,
        question: str,
        documents: List[str],
        max_length: int = 512,
        use_query_entity_intersection: bool = True,
        use_importance_weighting: bool = False,
        importance_alpha: float = 0.6,
        importance_cap: float = 2.5,
        importance_scope: str = "entity_query",
        entity_weight: float = 1.5,
        query_weight: float = 1.3,
        intersection_weight: float = 3.5,
        mask_batch_size: int = 128,
        use_selective_masking: bool = True,
        use_nonlinear_penalty: bool = False,
        nonlinear_threshold: float = -2.0,
        nonlinear_mode: str = "square",
        use_contrastive_mlm: bool = False,
        use_claim_bearing_spans: bool = True,
    ) -> List[dict]:
        """다수의 문서를 한 번에 텐서 단위로 처리하여 연산 속도 극대화 (CUDA 동기화 제거)"""
        results = [
            {'score': -999.0, 'unweighted_pll': -999.0, 'weighted_pll': -999.0,
             'entity_count': 0, 'query_count': 0, 'intersection_count': 0}
            for _ in documents
        ]
        if not documents:
            return results

        question_keywords = self.extract_question_keywords(question)

        inputs = self.tokenizer(
            documents, return_tensors="pt", truncation=True, max_length=max_length,
            padding=True, return_offsets_mapping=True
        )
        input_ids_tensor = inputs["input_ids"]
        attention_mask_tensor = inputs["attention_mask"]
        offset_mapping = inputs.get("offset_mapping", None)

        input_ids_np = input_ids_tensor.cpu().numpy()
        batch_tasks = []
        doc_stats = {i: {'entity_count': 0, 'query_count': 0, 'intersection_count': 0} for i in range(len(documents))}

        for i, doc in enumerate(documents):
            if not doc or not doc.strip():
                continue
            token_ids = input_ids_np[i]
            token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
            offsets_raw = offset_mapping[i] if offset_mapping is not None else None
            if offsets_raw is not None:
                if hasattr(offsets_raw, 'cpu'):
                    offsets = offsets_raw.cpu().numpy()
                elif isinstance(offsets_raw, (list, tuple)):
                    def _to_int(x):
                        return int(x.item()) if hasattr(x, 'item') else int(x)
                    offsets = np.array([(_to_int(o[0]), _to_int(o[1])) for o in offsets_raw], dtype=np.int64)
                else:
                    offsets = np.array(offsets_raw)
            else:
                offsets = np.zeros((len(token_ids), 2), dtype=np.int64)

            valid_mask = (token_ids != self.tokenizer.pad_token_id) & \
                         (token_ids != self.tokenizer.cls_token_id) & \
                         (token_ids != self.tokenizer.sep_token_id)
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue

            token_tf = self.build_token_tf(token_texts, valid_indices) if use_importance_weighting else {}
            max_tf = max(token_tf.values()) if token_tf else 1

            query_relevant_sentences = set()
            if use_query_entity_intersection and question_keywords:
                for sent_start, sent_end in self.find_sentence_boundaries(doc):
                    if any(kw in doc[sent_start:sent_end].lower() for kw in question_keywords):
                        query_relevant_sentences.add((sent_start, sent_end))

            # Claim-bearing span 가중치 (batch). 문장 boost는 build_span_weights 내부에서만
            span_weights = {}
            if use_claim_bearing_spans and _CLAIM_BEARING_AVAILABLE:
                if hasattr(offsets, "tolist"):
                    offsets_list = [(int(o[0]), int(o[1])) for o in offsets.tolist()]
                else:
                    offsets_list = [(int(o[0]) if hasattr(o[0], "item") else int(o[0]),
                                    int(o[1]) if hasattr(o[1], "item") else int(o[1]))
                                   for o in offsets]
                span_weights = build_span_weights(
                    doc, question, offsets_list,
                    question_keywords=question_keywords,
                    use_ner=True, use_numeric_regex=True, use_heuristic_fallback=True,
                )

            use_legacy_fallback = (not use_claim_bearing_spans) or (len(span_weights) == 0)

            token_weights = {}
            for idx in valid_indices:
                token_text = token_texts[idx].replace('##', '').replace('Ġ', ' ').strip()
                token_norm = self.normalize_token(token_text)
                is_query = token_norm in question_keywords
                is_entity = False
                is_intersection = False

                weight = 1.0
                if idx in span_weights:
                    weight = span_weights[idx]
                    is_entity = True
                    doc_stats[i]['entity_count'] += 1
                    o0 = offsets[idx][0] if idx < len(offsets) else 0
                    token_start = int(o0.item()) if hasattr(o0, "item") else int(o0)
                    if use_query_entity_intersection and query_relevant_sentences and any(s <= token_start < e for s, e in query_relevant_sentences):
                        is_intersection = True
                        doc_stats[i]['intersection_count'] += 1
                elif use_legacy_fallback:
                    is_entity = self.is_entity_token(token_text) or self.is_number_token(token_text)
                    if use_query_entity_intersection and is_entity and query_relevant_sentences:
                        o0 = offsets[idx][0] if idx < len(offsets) else 0
                        token_start = int(o0.item()) if hasattr(o0, "item") else int(o0)
                        if any(s <= token_start < e for s, e in query_relevant_sentences):
                            is_intersection = True
                            weight = intersection_weight
                            doc_stats[i]['intersection_count'] += 1
                            doc_stats[i]['entity_count'] += 1
                        else:
                            weight = entity_weight
                            doc_stats[i]['entity_count'] += 1
                    elif is_entity:
                        weight = entity_weight
                        doc_stats[i]['entity_count'] += 1

                if is_query:
                    weight = max(weight, query_weight)
                    doc_stats[i]['query_count'] += 1

                if use_importance_weighting and token_tf.get(token_norm, 0) > 0:
                    tf = token_tf[token_norm]
                    importance = (np.log1p(max_tf) - np.log1p(tf)) + 1.0
                    multiplier = min(importance_cap, max(1.0, 1.0 + importance_alpha * (importance - 1.0)))
                    if importance_scope == "all" or (importance_scope == "entity_query" and (is_entity or is_query)) or (importance_scope == "intersection" and is_intersection):
                        weight *= multiplier

                token_weights[idx] = weight

            if use_selective_masking:
                mask_indices = [idx for idx in valid_indices if token_weights.get(idx, 1.0) > 1.0]
                if len(mask_indices) < 3:
                    mask_indices = valid_indices.tolist()
            else:
                mask_indices = valid_indices.tolist()

            for idx in mask_indices:
                batch_tasks.append({'doc_idx': i, 'token_idx': int(idx), 'orig_id': int(token_ids[idx]), 'weight': token_weights.get(idx, 1.0)})

        log_probs_per_doc = {i: [] for i in range(len(documents))}
        weights_per_doc = {i: [] for i in range(len(documents))}

        query_log_probs = None
        if use_contrastive_mlm:
            query_log_probs = self._compute_query_log_probs(question, max_length)

        with torch.no_grad():
            for start in range(0, len(batch_tasks), mask_batch_size):
                chunk = batch_tasks[start:start + mask_batch_size]
                doc_indices = [t['doc_idx'] for t in chunk]
                chunk_input_ids = input_ids_tensor[doc_indices].clone().to(self.device)
                chunk_attention = attention_mask_tensor[doc_indices].clone().to(self.device)

                for row, t in enumerate(chunk):
                    chunk_input_ids[row, t['token_idx']] = self.tokenizer.mask_token_id

                logits = self.model(chunk_input_ids, attention_mask=chunk_attention).logits

                row_indices = torch.arange(len(chunk), device=self.device)
                token_indices = torch.tensor([t['token_idx'] for t in chunk], device=self.device, dtype=torch.long)
                target_ids = torch.tensor([t['orig_id'] for t in chunk], device=self.device, dtype=torch.long)

                masked_logits = logits[row_indices, token_indices, :]
                log_probs = torch.log_softmax(masked_logits, dim=-1)
                target_log_probs = log_probs[row_indices, target_ids].cpu().numpy()

                if use_contrastive_mlm and query_log_probs is not None:
                    log_p_q = query_log_probs[target_ids].cpu().numpy()
                    log_p_q = np.clip(log_p_q, -20.0, 0.0)
                    contrastive = target_log_probs - log_p_q
                else:
                    contrastive = target_log_probs

                for row, t in enumerate(chunk):
                    doc_idx = t['doc_idx']
                    log_probs_per_doc[doc_idx].append(float(contrastive[row]))
                    weights_per_doc[doc_idx].append(t['weight'])

        for i in range(len(documents)):
            lp, w = log_probs_per_doc[i], weights_per_doc[i]
            if len(lp) > 0:
                eff_w = w
                if use_nonlinear_penalty:
                    eff_w = [
                        (x ** 2 if nonlinear_mode == "square" else float(np.exp(min(x, 10))))
                        if lp[j] < nonlinear_threshold and x > 1.0 else x
                        for j, x in enumerate(w)
                    ]
                results[i] = {
                    'score': float(np.average(lp, weights=eff_w)),
                    'unweighted_pll': float(np.mean(lp)),
                    'weighted_pll': float(np.average(lp, weights=eff_w)),
                    'entity_count': doc_stats[i]['entity_count'],
                    'query_count': doc_stats[i]['query_count'],
                    'intersection_count': doc_stats[i]['intersection_count']
                }
        return results

