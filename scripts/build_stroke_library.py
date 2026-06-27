from rs_words.config import PATCH_BANK_DIR
from rs_words.stroke_library import build_stroke_patch_bank

if __name__ == "__main__":
    build_stroke_patch_bank(PATCH_BANK_DIR)
