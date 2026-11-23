"""Read-only RapidOCR 3.10.0 detail extraction; no gold or detector candidates.

The installed implementation remains unchanged. Hooks record native CTC and filter
evidence without changing predictions. A public character box is usable only after
the original fragment and every intervening row are proved to be the same occurrence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import socket
import sys
import threading
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OCR_ROOT = HERE.parent / 'ocr-adapters-v0'
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))
from adapters.rapid import Adapter, Capture, empty_status, plain, validate_parallel
from common import polygon_valid, validate_output

IDENTITY_KEYS = ('image_id', 'ocr_run_id', 'file_sha256', 'rgb_sha256',
                 'width', 'height', 'ocr_sha256')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


@contextmanager
def network_guard():
    """Python socket guard only; not an operating-system isolation claim."""
    old = socket.socket.connect, socket.socket.connect_ex, socket.create_connection
    def blocked(*args, **kwargs):
        raise RuntimeError('network access prohibited in local coordinate extraction')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = blocked
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = old


def bind_identity(rgb, ocr, identity):
    """Reject cross-image/run/source JSON mixing before starting OCR."""
    validate_output(ocr)
    ident = {k: copy.deepcopy(identity[k]) for k in IDENTITY_KEYS}
    if (ocr['image_id'] != ident['image_id'] or ocr['run_id'] != ident['ocr_run_id']
            or ocr['input']['sha256'] != ident['file_sha256']
            or digest(ocr) != ident['ocr_sha256']):
        raise ValueError('Detail extraction image/run/OCR identity mismatch')
    if (rgb.mode != 'RGB' or rgb.size != (ident['width'], ident['height'])
            or rgb.size != (ocr['input']['width'], ocr['input']['height'])
            or hashlib.sha256(rgb.tobytes()).hexdigest() != ident['rgb_sha256']):
        raise ValueError('Detail extraction RGB identity mismatch')
    for i, u in enumerate(ocr['units']):
        if u['id'] != f"{ident['ocr_run_id']}:{ident['image_id']}:u{i:04d}":
            raise ValueError('Detail extraction unit occurrence mismatch')
    return ident


def exact_nonspace_alignment(text, token_texts):
    """Whole-stream alignment, not a substring/name search or a simple zip.

    Native single-character output omits whitespace. Non-whitespace code points
    must match completely and in order. Multi-codepoint tokens are retained as
    indivisible spans only if their original exact slice has no omitted whitespace.
    """
    positions = [i for i, ch in enumerate(text) if not ch.isspace()]
    if any(not isinstance(t, str) or not t or any(c.isspace() for c in t)
           for t in token_texts):
        raise ValueError('Invalid/whitespace detail token')
    if ''.join(token_texts) != ''.join(text[i] for i in positions):
        raise ValueError('Detail text is not a complete exact nonspace projection')
    spans, cursor = [], 0
    for token in token_texts:
        selected = positions[cursor:cursor + len(token)]
        if not selected or text[selected[0]:selected[-1] + 1] != token:
            raise ValueError('Indivisible detail token crosses omitted whitespace')
        spans.append((selected[0], selected[-1] + 1))
        cursor += len(token)
    return spans


def filtered_row_indices(texts, scores, threshold=0.5):
    if len(texts) != len(scores):
        raise ValueError('Text/score filtering length mismatch')
    nonblank = [i for i, txt in enumerate(texts) if txt.strip()]
    keep = [i for i in nonblank if scores[i] >= threshold]
    return nonblank, keep


def audit_crop_transform(raw_quad, restored_quad, raw_crop_shape, mapped_crop_shape,
                         operation_record, width, height):
    """Expose native forward/inverse frame inconsistencies without repairing boxes.

    This is a structural coordinate check, never a glyph-coverage guarantee.
    """
    before = np.array(raw_quad, dtype=float)
    after = np.array(restored_quad, dtype=float)
    def dimensions(quad):
        edges = [float(np.linalg.norm(quad[(i+1) % 4] - quad[i])) for i in range(4)]
        return edges, int(max(edges[0], edges[2])), int(max(edges[1], edges[3]))
    raw_edges, fw, fh = dimensions(before)
    final_edges, _, _ = dimensions(after)
    forward_rotated = fh / fw >= 1.5 if fw > 0 else None
    inverse_rotated = (round(max(final_edges[1], final_edges[3]) /
                            max(final_edges[0], final_edges[2]), 2) >= 1.5
                       if max(final_edges[0], final_edges[2]) > 0 else True)
    iw, ih = int(final_edges[0]), int(final_edges[3])
    expected_raw = [fw, fh] if forward_rotated else [fh, fw]
    expected_mapped = [iw, ih] if inverse_rotated else [ih, iw]
    unclipped = before.copy()
    for key, op in reversed(list(operation_record.items())):
        if 'padding' in key:
            unclipped[:, 0] -= op['left']; unclipped[:, 1] -= op['top']
        elif 'preprocess' in key:
            unclipped[:, 0] *= op['ratio_w']; unclipped[:, 1] *= op['ratio_h']
    was_clipped = bool(np.any(unclipped[:, 0] < 0) or np.any(unclipped[:, 0] > width)
                       or np.any(unclipped[:, 1] < 0) or np.any(unclipped[:, 1] > height))
    reasons = []
    if raw_crop_shape[:2] != expected_raw:
        reasons.append('native_forward_crop_shape_mismatch')
    if forward_rotated != inverse_rotated:
        reasons.append('native_forward_inverse_rot90_disagreement')
    if mapped_crop_shape[:2] != expected_mapped:
        reasons.append('native_inverse_frame_differs_from_actual_restored_crop')
    if was_clipped:
        reasons.append('parent_clipping_changes_native_inverse_reference')
    return dict(status='uncertain' if reasons else 'ok', reasons=reasons,
        forward_quad=plain(before), restored_quad=plain(after),
        forward_crop_dimensions=[fw, fh], inverse_frame_dimensions=[iw, ih],
        raw_crop_shape=raw_crop_shape, mapped_crop_shape=mapped_crop_shape,
        forward_rot90=forward_rotated, inverse_rot90=inverse_rotated,
        parent_was_clipped=was_clipped,
        inverse_integerization='native int() truncation toward zero, retained unchanged')


def validate_filter_trace(output, trace):
    """Prove no empty-line/score filtering shifted public word-results rows."""
    before = trace['recognition']
    texts, scores = before['texts'], before['scores']
    nonblank, keep = filtered_row_indices(texts, scores)
    pre = trace.get('before_score_filter')
    calc = trace.get('calculation')
    if pre is None or calc is None:
        raise ValueError('Missing detail calculation/filter evidence')
    if (calc['texts'] != [texts[i] for i in nonblank]
            or pre['texts'] != calc['texts']
            or pre['scores'] != [scores[i] for i in nonblank]
            or pre['boxes'] != calc['boxes']
            or len(calc['raw_rows']) != len(nonblank)
            or len(calc['returned_rows']) != len(nonblank)
            or len(pre['word_results']) != len(nonblank)):
        raise ValueError('Blank-line/detail calculation alignment mismatch')
    if pre['word_results'] != calc['returned_rows']:
        raise ValueError('Detail rows changed between calculation and filtering')
    if any(not row for row in calc['returned_rows']):
        raise ValueError('Empty detail rows cannot be matched through native filtering')
    expected_indices = [j for j, i in enumerate(nonblank) if i in keep]
    expected_rows = [pre['word_results'][j] for j in expected_indices]
    if (output['texts'] != [texts[i] for i in keep]
            or output['word_results'] != expected_rows
            or output['boxes'] != [pre['boxes'][j] for j in expected_indices]
            or output['scores'] != [pre['scores'][j] for j in expected_indices]
            or len(output['boxes']) != len(keep)
            or len(output['scores']) != len(keep)
            or len(output['word_results']) != len(keep)):
        raise ValueError('Text-score/detail row alignment mismatch')
    return keep


class DecoderTrace:
    """Observe the actual selected CTC dictionary tokens, retaining object routing."""
    def __init__(self, inner, owner):
        self.inner, self.owner = inner, owner
    def __getattr__(self, key):
        return getattr(self.inner, key)
    def __call__(self, preds, return_word_box=False, **kwargs):
        result = self.inner(preds, return_word_box, **kwargs)
        if not return_word_box:
            return result
        lines, infos = result
        if len(lines) != len(infos):
            raise ValueError('CTC line/WordInfo count mismatch')
        indices, probabilities = preds.argmax(axis=2), preds.max(axis=2)
        for b, info in enumerate(infos):
            ids = indices[b]
            selected = np.ones(len(ids), dtype=bool)
            selected[1:] = ids[1:] != ids[:-1]
            selected &= ids != 0
            columns = np.where(selected)[0].tolist()
            tokens = [self.inner.character[int(ids[col])] for col in columns]
            confidences = [round(float(probabilities[b, col]), 5) for col in columns]
            self.owner._ctc_by_object[id(info)] = dict(
                text=lines[b][0], tokens=tokens, columns=columns,
                confidences=confidences, timestep_count=int(len(ids)),
                wh_ratio=float(kwargs['wh_ratio_list'][b]),
                max_wh_ratio=float(kwargs['max_wh_ratio']))
        return result


class DetailExtractor:
    """Synchronous, single-instance serialized extraction using existing local assets."""
    def __init__(self, asset_root):
        started = time.perf_counter()
        with network_guard():
            self.adapter = Adapter(asset_root)
        self.load_seconds = time.perf_counter() - started
        self.metadata = copy.deepcopy(self.adapter.metadata)
        self.engine = self.adapter.engine
        self._lock = threading.Lock()
        self.trace, self._ctc_by_object = {}, {}
        owner = self
        rec = self.engine._load_rec_model()
        rec.postprocess_op = DecoderTrace(rec.postprocess_op, self)
        recognize = self.engine.recognize_txt
        calculate = self.engine.calc_word_boxes
        score_filter = self.engine.filter_by_text_score

        def traced_rec(engine, images):
            result = recognize(images)
            infos = result.word_results
            owner.trace['recognition'] = dict(
                texts=plain(result.txts), scores=plain(result.scores),
                ctc=[copy.deepcopy(owner._ctc_by_object.get(id(info))) for info in infos],
                word_info=[word_info_plain(info) for info in infos])
            return result

        def traced_calculate(engine, images, boxes, result):
            owner.trace['calculation'] = dict(texts=plain(result.txts),
                boxes=plain(boxes), crop_shapes=[list(im.shape) for im in images],
                word_info=[word_info_plain(info) for info in result.word_results])
            rows = calculate(images, boxes, result)
            owner.trace['calculation']['raw_rows'] = plain(result.word_results)
            owner.trace['calculation']['returned_rows'] = plain(rows)
            return rows

        def traced_filter(engine, result):
            owner.trace['before_score_filter'] = public_plain(result)
            return score_filter(result)

        self.engine.recognize_txt = types.MethodType(traced_rec, self.engine)
        self.engine.calc_word_boxes = types.MethodType(traced_calculate, self.engine)
        self.engine.filter_by_text_score = types.MethodType(traced_filter, self.engine)

    def _reset(self):
        self.adapter.det_postprocess.box_thresh = 0.5
        self.adapter.det_postprocess.unclip_ratio = 1.6
        self.engine.update_params(use_det=True, use_cls=False, use_rec=True,
            return_word_box=False, return_single_char_box=False, text_score=0.5,
            box_thresh=0.5, unclip_ratio=1.6)

    def extract(self, rgb, original_ocr, identity):
        ident = bind_identity(rgb, original_ocr, identity)
        if original_ocr['engine'] != self.metadata:
            raise ValueError('Original OCR model/settings metadata differs from pinned adapter')
        with self._lock, network_guard():
            return self._extract(rgb, original_ocr, ident)

    def _extract(self, rgb, original_ocr, ident):
        from rapidocr.utils.log import logger
        from rapidocr.utils.output import RapidOCROutput
        started = time.perf_counter()
        self.trace, self._ctc_by_object = {}, {}
        self.adapter.evidence = {}
        capture = Capture()
        logger.addHandler(capture)
        rows, reasons, public, status = [], [], {}, 'failure'
        try:
            self._reset()
            output = self.engine(np.asarray(rgb)[:, :, ::-1].copy(), use_det=True,
                use_cls=False, use_rec=True, return_word_box=True,
                return_single_char_box=True, text_score=0.5,
                box_thresh=0.5, unclip_ratio=1.6)
            public = public_plain(output)
            if not isinstance(output, RapidOCROutput):
                raise ValueError('Expected full RapidOCROutput')
            if output.txts is None:
                status = empty_status(self.adapter.evidence, capture.messages)
                if original_ocr['units']:
                    raise ValueError('Detail rerun empty but original has fragments')
                if status != 'ok':
                    reasons.append('empty_without_witnessed_normal_filter_or_detection')
            else:
                validate_parallel(output.boxes, output.txts, output.scores)
                originals = original_ocr['units']
                if len(originals) != len(output.txts):
                    raise ValueError('Original/detail fragment count mismatch')
                # The whole ordered stream must agree. Never match by repeated text.
                for i, unit in enumerate(originals):
                    if (unit['text'] != output.txts[i]
                            or unit['polygons'] != [plain(output.boxes[i])]
                            or unit['confidence'] != float(output.scores[i])):
                        raise ValueError('Original/detail ordered fragment transcription/box/score mismatch')
                keep = validate_filter_trace(public, self.trace)
                for i, (unit, pre_index) in enumerate(zip(originals, keep)):
                    rows.append(self._unit(unit, public['word_results'][i], pre_index, ident))
                status = 'ok' if all(u['status'] == 'ok' for u in rows) else 'partial_failure'
            if output.txts is not None and capture.messages:
                if status == 'ok':
                    status = 'partial_failure'
                reasons.append('native_warning; inspect evidence.warnings')
        except Exception as ex:
            status = 'failure'
            reasons.append(type(ex).__name__ + ': ' + str(ex))
            rows = []
        finally:
            self._reset()
            logger.removeHandler(capture)
        if not rows and original_ocr['units']:
            rows = [dict(unit_id=u['id'], text=u['text'], polygons=copy.deepcopy(u['polygons']),
                tokens=[], status='unavailable', reasons=list(reasons)) for u in original_ocr['units']]
        return dict(schema_version='ocr-coordinate-details-v0', identity=ident,
            status=status, reasons=reasons, units=rows,
            method='native CTC single-character boxes; original fragment identity verified',
            settings=dict(return_word_box=True, return_single_char_box=True, use_cls=False,
                          baseline_model_settings_digest=digest(self.metadata)),
            transforms=dict(adapter=plain(self.adapter.evidence),
                calculation=self.trace.get('calculation'),
                coordinate_system='original RGB pixel boundaries; native restored integer boxes',
                crop_path='detector crop perspective; possible rot90; recognition resize/pad; '
                          'CTC columns; native average width; crop inverse and int truncation'),
            evidence=dict(public=public, trace=copy.deepcopy(self.trace), warnings=capture.messages,
                options_after_call=dict(return_word_box=self.engine.return_word_box,
                    return_single_char_box=self.engine.return_single_char_box,
                    use_cls=self.engine.use_cls)),
            timing=dict(detail_extract_seconds=time.perf_counter()-started,
                        adapter_prepare_seconds=self.load_seconds))

    def _unit(self, unit, native_tokens, pre_index, ident):
        row = dict(unit_id=unit['id'], text=unit['text'], polygons=copy.deepcopy(unit['polygons']),
                   status='ok', reasons=[], tokens=[], original_order_index=pre_index)
        try:
            nonblank, _ = filtered_row_indices(self.trace['recognition']['texts'],
                                              self.trace['recognition']['scores'])
            calc_index = nonblank.index(pre_index)
            native = self.adapter.evidence
            transform = audit_crop_transform(native['pre_filter']['boxes'][pre_index],
                unit['polygons'][0], native['crop_shapes'][pre_index],
                self.trace['calculation']['crop_shapes'][calc_index], native['op_record'],
                ident['width'], ident['height'])
            row['transform'] = transform
            if transform['status'] != 'ok':
                raise ValueError('; '.join(transform['reasons']))
            ctc = self.trace['recognition']['ctc'][pre_index]
            info = self.trace['recognition']['word_info'][pre_index]
            if ctc is None or ''.join(ctc['tokens']) != unit['text']:
                raise ValueError('CTC token provenance missing or text mismatch')
            # Installed get_word_info indexes Unicode code points with CTC positions.
            # A multi-codepoint vocabulary token makes that native mapping unproven.
            if any(len(t) != 1 for t in ctc['tokens']):
                raise ValueError('Native mapping unsupported for multi-codepoint CTC token')
            if (len(ctc['tokens']) != len(ctc['confidences'])
                    or len(ctc['tokens']) != len(ctc['columns'])
                    or info['confs'] != ctc['confidences']):
                raise ValueError('CTC confidence/column/token correspondence mismatch')
            text_tokens = [t[0] for t in native_tokens]
            spans = exact_nonspace_alignment(unit['text'], text_tokens)
            flattened = [c for word in info['words'] for c in word]
            columns = [c for word in info['word_cols'] for c in word]
            if flattened != text_tokens or len(columns) != len(spans):
                raise ValueError('Native WordInfo tokens/column count mismatch')
            for j, ((text, native_score, polygon), (start, end)) in enumerate(zip(native_tokens, spans)):
                if (end-start != 1 or columns[j] != ctc['columns'][start]
                        or text != ctc['tokens'][start]):
                    raise ValueError('Native detail CTC column/code-point mismatch')
                if not polygon_valid(polygon, ident['width'], ident['height']):
                    raise ValueError('Invalid/missing/outside detail polygon')
                score = ctc['confidences'][start]
                if not math.isfinite(score):
                    raise ValueError('Nonfinite CTC score')
                row['tokens'].append(dict(text=text, start=start, end=end,
                    polygon=copy.deepcopy(polygon), score=score, native_score=native_score,
                    score_status=('native_aligned' if j == start else 'raw_ctc_reassociated_after_whitespace'),
                    ctc_token_index=start, ctc_column=columns[j],
                    geometry_source='RapidOCR.CalRecBoxes native CTC width estimate',
                    alignment='whole nonspace stream exact; CTC token and column verified',
                    score_meaning='recognition confidence; not location accuracy probability'))
            row['evidence'] = dict(ctc=ctc, original_text_sha256=hashlib.sha256(
                unit['text'].encode('utf-8')).hexdigest(),
                omitted_whitespace_offsets=[i for i,c in enumerate(unit['text']) if c.isspace()])
        except (ValueError, KeyError, TypeError, IndexError) as ex:
            row.update(status='uncertain', reasons=[type(ex).__name__ + ': ' + str(ex)], tokens=[])
        return row


def word_info_plain(info):
    if info is None:
        return None
    return dict(words=plain(info.words), word_cols=plain(info.word_cols),
        word_types=[v.value for v in info.word_types], line_txt_len=float(info.line_txt_len),
        confs=plain(info.confs))


def public_plain(output):
    return dict(return_type=type(output).__name__, boxes=plain(getattr(output, 'boxes', None)),
        texts=plain(getattr(output, 'txts', None)), scores=plain(getattr(output, 'scores', None)),
        word_results=plain(getattr(output, 'word_results', None)))
