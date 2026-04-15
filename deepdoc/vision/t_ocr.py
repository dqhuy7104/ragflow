#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import asyncio
import importlib.metadata
import logging
import os
import re
import sys


from common.misc_utils import thread_pool_exec

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../')))

from deepdoc.vision.seeit import draw_box
from deepdoc.vision import OCR, init_in_out
import argparse
import numpy as np
from PIL import Image


class VietOCRAdapter:
    def __init__(
        self,
        config_name: str = "vgg_transformer",
        weights: str | None = None,
        use_gpu: bool = False,
        model_dir: str | None = None,
        model_repo_id: str | None = None,
    ):
        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("vietocr"):
                raise RuntimeError(
                    "VietOCR backend requested but the `vietocr` package is not installed. Install it with `pip install vietocr`."
                ) from exc
            raise
        except Exception as exc:
            torch_version = _package_version("torch")
            torchvision_version = _package_version("torchvision")
            raise RuntimeError(
                "Failed to import VietOCR dependencies. This usually means the local `torch` and `torchvision` "
                f"packages are incompatible. Current versions: torch={torch_version}, torchvision={torchvision_version}. "
                f"Original error: {exc}"
            ) from exc

        self.detector = OCR(
            model_dir=model_dir,
            model_repo_id=model_repo_id,
            recognition_backend="deepdoc",
        )
        self.use_gpu = use_gpu
        self.config_name = config_name

        config = Cfg.load_config_from_name(config_name)
        if weights:
            config["weights"] = weights
        if "cnn" in config and isinstance(config["cnn"], dict):
            config["cnn"]["pretrained"] = False
        config["device"] = "cuda:0" if use_gpu else "cpu"

        try:
            self.predictor = Predictor(config)
        except Exception as exc:
            torch_version = _package_version("torch")
            torchvision_version = _package_version("torchvision")
            raise RuntimeError(
                "Failed to initialize VietOCR. This usually means the local `torch` and `torchvision` packages "
                f"are incompatible. Current versions: torch={torch_version}, torchvision={torchvision_version}. "
                f"Original error: {exc}"
            ) from exc
        logging.info(
            "Loaded VietOCR backend with config=%s, use_gpu=%s, weights=%s",
            config_name,
            use_gpu,
            weights or "<package default>",
        )

    def __call__(self, img, device_id=0):
        if img is None:
            return []

        dt_boxes, _ = self.detector.text_detector[0](img)
        if dt_boxes is None:
            return []

        dt_boxes = self.detector.sorted_boxes(dt_boxes)
        results = []
        for box in dt_boxes:
            crop = self.detector.get_rotate_crop_image(img.copy(), box.copy())
            text = self.predictor.predict(Image.fromarray(crop))
            if text:
                results.append((box.tolist(), (text, 1.0)))
        return results


class PaddleOCRAdapter:
    def __init__(self, lang: str = "vi", use_gpu: bool = False):
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR backend requested but the `paddleocr` package is not installed."
            ) from exc

        self.lang = lang
        self.use_gpu = use_gpu
        device = "gpu" if use_gpu else "cpu"
        try:
            self.engine = PaddleOCR(lang=lang, device=device, use_textline_orientation=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize PaddleOCR. If you see an import error from `paddle`, `snapy`, or `kintera`, "
                "your environment likely has the wrong `paddle` package installed. Uninstall `paddle` and install "
                "the official `paddlepaddle` (CPU) or `paddlepaddle-gpu` (CUDA) build that matches your system."
            ) from exc
        logging.info(f"Loaded PaddleOCR backend with lang={lang}, use_gpu={use_gpu}")

    def __call__(self, img, device_id=0):
        if hasattr(self.engine, "predict"):
            result = self.engine.predict(img)
        else:
            result = self.engine.ocr(img, cls=True)
        return self._normalize_result(result)

    @staticmethod
    def _normalize_result(result):
        if not result:
            return []

        first = result[0] if isinstance(result, list) and result else result
        if isinstance(first, dict) and "rec_texts" in first:
            polys = first.get("rec_polys") or first.get("dt_polys") or []
            texts = first.get("rec_texts") or []
            scores = first.get("rec_scores") or []
            normalized = []
            for idx, text in enumerate(texts):
                if idx >= len(polys):
                    break
                score = scores[idx] if idx < len(scores) else 1.0
                box = polys[idx].tolist() if hasattr(polys[idx], "tolist") else polys[idx]
                normalized.append((box, (text, score)))
            return normalized

        if len(result) == 1 and isinstance(result[0], list) and result[0] and isinstance(result[0][0], (list, tuple)):
            # PaddleOCR 2.x commonly returns a single-item list for one input image.
            if len(result[0]) == 2 and isinstance(result[0][0], (list, tuple)) and isinstance(result[0][1], (list, tuple)):
                return result[0]
            return result[0]
        return result


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main(args):
    import torch.cuda

    cuda_devices = torch.cuda.device_count()
    limiter = [asyncio.Semaphore(1) for _ in range(cuda_devices)] if cuda_devices > 1 else None
    if args.backend == "paddleocr":
        ocr = PaddleOCRAdapter(lang=args.paddleocr_lang, use_gpu=args.paddleocr_use_gpu)
    elif args.backend == "vietocr":
        ocr = VietOCRAdapter(
            config_name=args.vietocr_config,
            weights=args.vietocr_weights,
            use_gpu=args.vietocr_use_gpu,
            model_dir=args.model_dir,
            model_repo_id=args.model_repo_id,
        )
    else:
        ocr = OCR(model_dir=args.model_dir, model_repo_id=args.model_repo_id)
    images, outputs = init_in_out(args)

    def _group_outputs():
        groups = []
        grouped = {}

        for img, out_path in zip(images, outputs):
            base_name = os.path.basename(out_path)
            match = re.match(r"^(.*\.pdf)_\d+\.jpg$", base_name, re.IGNORECASE)
            if match:
                group_key = match.group(1)
                output_name = os.path.splitext(group_key)[0]
            else:
                group_key = base_name
                output_name = os.path.splitext(base_name)[0]

            bucket = grouped.setdefault(group_key, {"output_name": output_name, "pages": []})
            bucket["pages"].append((img, out_path))

        for key, value in grouped.items():
            groups.append((key, value["output_name"], value["pages"]))
        return groups

    def _stack_images(page_images):
        widths = [img.size[0] for img in page_images]
        heights = [img.size[1] for img in page_images]
        canvas = Image.new("RGB", (max(widths), sum(heights)), (255, 255, 255))
        offset = 0
        for img in page_images:
            canvas.paste(img, (0, offset))
            offset += img.size[1]
        return canvas

    def __ocr(i, id, img):
        print("Task {} start".format(i))
        bxs = ocr(np.array(img), id)
        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = [{
            "text": t,
            "bbox": [b[0][0], b[0][1], b[1][0], b[-1][1]],
            "type": "ocr",
            "score": 1} for b, t in bxs if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]]
        img = draw_box(images[i], bxs, ["ocr"], 1.)
        img.save(outputs[i], quality=95)
        with open(outputs[i] + ".txt", "w+", encoding='utf-8') as f:
            f.write("\n".join([o["text"] for o in bxs]))

        print("Task {} done".format(i))

    async def __ocr_thread(i, id, img, limiter = None):
        if limiter:
            async with limiter:
                print(f"Task {i} use device {id}")
                await thread_pool_exec(__ocr, i, id, img)
        else:
            await thread_pool_exec(__ocr, i, id, img)


    async def __ocr_launcher():
        tasks = []
        for i, img in enumerate(images):
            dev_id = i % cuda_devices if cuda_devices > 1 else 0
            semaphore = limiter[dev_id] if limiter else None
            tasks.append(asyncio.create_task(__ocr_thread(i, dev_id, img, semaphore)))

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error("OCR tasks failed: {}".format(e))
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    if args.whole_file:
        for _, output_name, page_items in _group_outputs():
            page_texts = []
            annotated_pages = []
            for page_idx, (img, _) in enumerate(page_items):
                bxs = ocr(np.array(img), 0)
                bxs = [(line[0], line[1][0]) for line in bxs]
                bxs = [{
                    "text": t,
                    "bbox": [b[0][0], b[0][1], b[1][0], b[-1][1]],
                    "type": "ocr",
                    "score": 1} for b, t in bxs if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]]
                # annotated_pages.append(draw_box(img.copy(), bxs, ["ocr"], 1.))
                page_texts.append("\n".join([o["text"] for o in bxs]))

            # merged_image = _stack_images(annotated_pages)
            # merged_image.save(os.path.join(args.output_dir, output_name + "2" +  ".jpg"), quality=95)
            with open(os.path.join(args.output_dir, output_name +  "2" + ".txt"), "w+", encoding="utf-8") as f:
                f.write("\n\n".join(page_texts))
    else:
        asyncio.run(__ocr_launcher())

    print("OCR tasks are all done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Directory where to store images or PDFs, or a file path to a single image or PDF",
                        required=True)
    parser.add_argument('--output_dir', help="Directory where to store the output images. Default: './ocr_outputs'",
                        default="./ocr_outputs")
    parser.add_argument(
        '--model_dir',
        default=os.getenv("DEEPDOC_OCR_MODEL_DIR"),
        help="Path to an OCR model bundle containing det.onnx, rec.onnx, and ocr.res. Use this for a Vietnamese-trained bundle or any custom OCR base.",
    )
    parser.add_argument(
        '--model_repo_id',
        default=os.getenv("DEEPDOC_OCR_REPO_ID", "InfiniFlow/deepdoc"),
        help="Hugging Face repo id to download the OCR bundle from when model_dir is not provided.",
    )
    parser.add_argument(
        '--whole_file',
        action='store_true',
        help="Merge PDF pages into one annotated output image and one combined text file instead of per-page outputs.",
    )
    parser.add_argument(
        '--backend',
        choices=["deepdoc", "paddleocr", "vietocr"],
        default=os.getenv("DEEPDOC_OCR_BACKEND", "deepdoc"),
        help="OCR backend to use. `deepdoc` uses the built-in ONNX models, `paddleocr` uses PaddleOCR directly, and `vietocr` keeps DeepDoc detection while replacing only the recognition layer.",
    )
    parser.add_argument(
        '--paddleocr_lang',
        default=os.getenv("PADDLEOCR_LANGUAGE", "vi"),
        help="PaddleOCR language code to use when --backend=paddleocr. Default: vi.",
    )
    parser.add_argument(
        '--paddleocr_use_gpu',
        action='store_true',
        default=os.getenv("PADDLEOCR_USE_GPU", "false").lower() in ("1", "true", "yes"),
        help="Use GPU for PaddleOCR backend.",
    )
    parser.add_argument(
        '--vietocr_config',
        default=os.getenv("VIETOCR_CONFIG", "vgg_transformer"),
        help="VietOCR config name to use when --backend=vietocr. Default: vgg_transformer.",
    )
    parser.add_argument(
        '--vietocr_weights',
        default=os.getenv("VIETOCR_WEIGHTS"),
        help="Optional VietOCR weights path or URL override when --backend=vietocr.",
    )
    parser.add_argument(
        '--vietocr_use_gpu',
        action='store_true',
        default=os.getenv("VIETOCR_USE_GPU", "false").lower() in ("1", "true", "yes"),
        help="Use GPU for VietOCR backend.",
    )
    args = parser.parse_args()
    main(args)
