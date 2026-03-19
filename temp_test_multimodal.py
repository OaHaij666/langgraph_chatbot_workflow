from pathlib import Path
from config import set_config, get_config
from app.llm import get_llm_client


def main():
    project_root = Path(__file__).resolve().parent
    image_path = project_root / "test.png"
    if not image_path.exists():
        raise FileNotFoundError(f"未找到测试图片: {image_path}")

    set_config("llm.enable_multimodal", True)
    client = get_llm_client()

    prompt = "请识别图片中的主要动物，只回答动物名称。"
    multimodal_inputs = [
        {
            "type": "image",
            "source": "path",
            "value": str(image_path),
        }
    ]

    print("provider:", get_config("llm.provider"))
    print("multimodal_enabled:", get_config("llm.enable_multimodal"))
    print("image:", image_path)
    print("---- 模型输出 ----")
    result = client.chat(prompt, multimodal_inputs=multimodal_inputs)
    print(result)


if __name__ == "__main__":
    main()
