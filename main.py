from encoder import LinguisticEncoder

if __name__ == "__main__":
    print("Initializing Linguistic Encoder...")
    encoder = LinguisticEncoder(ids_path="data/ids.txt")

    source_text = "机器翻译"  # "Machine Translation"
    print(f"\nSource: {source_text}")
    print("-" * 60)
    print(f"{'STRATEGY':<25} | {'REPRESENTATION'}")
    print("-" * 60)

    strategies = [
        "baseline",
        "pinyin_no_tone",
        "pinyin_tone",
        "zhuyin",
        "morphological",
        "simplified_traditional",
        "radical_flat",
        "radical_structural",
        "wubi",
        "stroke_sequence",
    ]

    for strategy in strategies:
        try:
            result = encoder.encode(source_text, strategy)
            display = " ".join(str(x) for x in result)
            if len(display) > 80:
                display = display[:77] + "..."
            print(f"{strategy:<25} | {display}")
        except Exception as e:
            print(f"{strategy:<25} | Error: {e}")
