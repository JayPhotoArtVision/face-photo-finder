import os
import subprocess
import sys

# ============================================================
# CONFIG
# ============================================================

TEST_DIR = "test_images"
SEARCH_SCRIPT = "face_search.py"

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)


# ============================================================
# FIND TEST PHOTOS
# ============================================================

test_photos = sorted(
    [
        f
        for f in os.listdir(TEST_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AI FACE SEARCH - 9 PHOTO BATCH TEST")
print("=" * 70)

print(
    f"Test photos: {len(test_photos)}"
)

print()


# ============================================================
# CHECK
# ============================================================

if len(test_photos) == 0:

    print("ERROR: No test photos found.")

    raise SystemExit


# ============================================================
# TEST EACH PHOTO
# ============================================================

results = []


for filename in test_photos:

    image_path = os.path.join(
        TEST_DIR,
        filename
    )

    print("=" * 70)
    print(f"TESTING: {filename}")
    print("=" * 70)

    try:

        result = subprocess.run(

            [
                sys.executable,
                SEARCH_SCRIPT,
                image_path
            ],

            text=True
        )

        if result.returncode == 0:

            results.append({
                "filename": filename,
                "status": "SUCCESS"
            })

        else:

            results.append({
                "filename": filename,
                "status": "ERROR"
            })

            print()
            print(
                f"ERROR while testing: {filename}"
            )

    except Exception as e:

        results.append({
            "filename": filename,
            "status": "ERROR"
        })

        print()
        print(
            f"ERROR while testing: {filename}"
        )

        print(e)

    print()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("BATCH TEST SUMMARY")
print("=" * 70)

success_count = sum(
    1
    for r in results
    if r["status"] == "SUCCESS"
)

error_count = len(results) - success_count


for result in results:

    print(
        f"{result['filename']:20} "
        f"{result['status']}"
    )


print()
print(
    f"Successful : {success_count}"
)

print(
    f"Errors     : {error_count}"
)

print()
print("=" * 70)
print("BATCH TEST COMPLETED")
print("=" * 70)