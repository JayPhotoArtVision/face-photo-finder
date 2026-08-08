
import numpy as np
import psycopg2


# ============================================================
# CONFIG
# ============================================================

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}

VALID_LABELS = ("A", "B", "C", "D")


# ============================================================
# HELPERS
# ============================================================

def percentile(values, p):

    if len(values) == 0:
        return 0.0

    return float(
        np.percentile(
            values,
            p
        )
    )



def print_statistics(name, values):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    if not values:

        print("No pairs found.")
        return

    values = np.array(
        values,
        dtype=np.float32
    )

    print(
        f"Pair count : {len(values)}"
    )

    print(
        f"Minimum    : {np.min(values):.4f}"
    )

    print(
        f"Maximum    : {np.max(values):.4f}"
    )

    print(
        f"Average    : {np.mean(values):.4f}"
    )

    print(
        f"Median     : {np.median(values):.4f}"
    )

    print(
        f"P05        : {percentile(values, 5):.4f}"
    )

    print(
        f"P10        : {percentile(values, 10):.4f}"
    )

    print(
        f"P25        : {percentile(values, 25):.4f}"
    )

    print(
        f"P50        : {percentile(values, 50):.4f}"
    )

    print(
        f"P75        : {percentile(values, 75):.4f}"
    )

    print(
        f"P90        : {percentile(values, 90):.4f}"
    )

    print(
        f"P95        : {percentile(values, 95):.4f}"
    )


# ============================================================
# CONNECT DATABASE
# ============================================================

print("=" * 70)
print("FACE SIMILARITY ANALYSIS")
print("=" * 70)

print()
print("Connecting to PostgreSQL...")

try:

    conn = psycopg2.connect(**DB_CONFIG)

    cursor = conn.cursor()

    print("PostgreSQL connected.")

except Exception as e:

    print("Database connection failed:")
    print(e)

    raise SystemExit


# ============================================================
# LOAD LABELED EMBEDDINGS
# ============================================================

cursor.execute("""
    SELECT
        fe.id,
        fe.photo_id,
        fe.face_index,
        fe.embedding,
        p.filename,
        fl.person_label
    FROM face_embeddings fe

    JOIN photos p
        ON p.id = fe.photo_id

    JOIN face_labels fl
        ON fl.embedding_id = fe.id

    WHERE fl.person_label IN ('A', 'B', 'C', 'D')

    ORDER BY fe.id;
""")

rows = cursor.fetchall()


print()
print(
    "Labeled embeddings:",
    len(rows)
)


if len(rows) < 2:

    print()
    print(
        "ERROR: At least 2 labeled embeddings are required."
    )

    cursor.close()
    conn.close()

    raise SystemExit


# ============================================================
# PREPARE EMBEDDINGS
# ============================================================

faces = []


for row in rows:

    embedding_id = row[0]
    photo_id = row[1]
    face_index = row[2]
    embedding_bytes = row[3]
    filename = row[4]
    person_label = row[5]


    embedding = np.frombuffer(
        embedding_bytes,
        dtype=np.float32
    ).copy()


    norm = np.linalg.norm(
        embedding
    )


    if norm == 0:

        print(
            "WARNING: Zero embedding:",
            embedding_id
        )

        continue


    embedding = (
        embedding / norm
    )


    faces.append({

        "id": embedding_id,

        "photo_id": photo_id,

        "face_index": face_index,

        "filename": filename,

        "person": person_label,

        "embedding": embedding
    })


print(
    "Usable embeddings:",
    len(faces)
)


# ============================================================
# COUNT LABELS
# ============================================================

label_counts = {}

for label in VALID_LABELS:

    count = sum(
        1
        for face in faces
        if face["person"] == label
    )

    label_counts[label] = count


print()
print("=" * 70)
print("LABELED FACE COUNTS")
print("=" * 70)

for label in VALID_LABELS:

    print(
        f"Person {label}: "
        f"{label_counts[label]}"
    )


# ============================================================
# CALCULATE PAIRWISE SIMILARITY
# ============================================================

same_pairs = []

different_pairs = []

pair_details = []


for i in range(len(faces)):

    face_a = faces[i]


    for j in range(i + 1, len(faces)):

        face_b = faces[j]


        # ----------------------------------------------------
        # Cosine similarity
        # ----------------------------------------------------

        similarity = float(
            np.dot(
                face_a["embedding"],
                face_b["embedding"]
            )
        )


        if face_a["person"] == face_b["person"]:

            pair_type = "SAME"

            same_pairs.append(
                similarity
            )

        else:

            pair_type = "DIFFERENT"

            different_pairs.append(
                similarity
            )


        pair_details.append({

            "type": pair_type,

            "similarity": similarity,

            "person_a": face_a["person"],

            "person_b": face_b["person"],

            "file_a": face_a["filename"],

            "file_b": face_b["filename"],

            "id_a": face_a["id"],

            "id_b": face_b["id"]
        })


# ============================================================
# BASIC SUMMARY
# ============================================================

total_pairs = len(pair_details)


print()
print("=" * 70)
print("PAIR SUMMARY")
print("=" * 70)

print(
    f"Total pairs     : {total_pairs}"
)

print(
    f"SAME pairs      : {len(same_pairs)}"
)

print(
    f"DIFFERENT pairs : {len(different_pairs)}"
)


# ============================================================
# STATISTICS
# ============================================================

print_statistics(
    "SAME-PERSON SIMILARITY",
    same_pairs
)

print_statistics(
    "DIFFERENT-PERSON SIMILARITY",
    different_pairs
)


# ============================================================
# OVERLAP ANALYSIS
# ============================================================

print()
print("=" * 70)
print("OVERLAP ANALYSIS")
print("=" * 70)


if same_pairs and different_pairs:

    same_min = min(same_pairs)

    same_max = max(same_pairs)

    diff_min = min(different_pairs)

    diff_max = max(different_pairs)


    print(
        f"SAME minimum       : "
        f"{same_min:.4f}"
    )

    print(
        f"SAME maximum       : "
        f"{same_max:.4f}"
    )

    print(
        f"DIFFERENT minimum  : "
        f"{diff_min:.4f}"
    )

    print(
        f"DIFFERENT maximum  : "
        f"{diff_max:.4f}"
    )


    overlap_low = max(
        same_min,
        diff_min
    )

    overlap_high = min(
        same_max,
        diff_max
    )


    if overlap_low <= overlap_high:

        print()
        print(
            "WARNING: Similarity ranges overlap."
        )

        print(
            f"Overlap range: "
            f"{overlap_low:.4f}"
            f" - "
            f"{overlap_high:.4f}"
        )

        print()
        print(
            "A single threshold will NOT perfectly"
        )

        print(
            "separate SAME and DIFFERENT persons."
        )

    else:

        print()
        print(
            "No range overlap detected."
        )

        print(
            "This is a good sign, but the dataset"
        )

        print(
            "is still small."
        )


# ============================================================
# MOST DANGEROUS DIFFERENT-PERSON PAIRS
# ============================================================

different_pairs_sorted = sorted(
    [
        pair
        for pair in pair_details
        if pair["type"] == "DIFFERENT"
    ],
    key=lambda x: x["similarity"],
    reverse=True
)


print()
print("=" * 70)
print("TOP 10 DIFFERENT-PERSON PAIRS")
print("=" * 70)


for rank, pair in enumerate(
    different_pairs_sorted[:10],
    start=1
):

    print()
    print(
        f"#{rank} "
        f"Similarity: "
        f"{pair['similarity']:.6f}"
    )

    print(
        f"   {pair['person_a']} | "
        f"{pair['file_a']} | "
        f"Face ID={pair['id_a']}"
    )

    print(
        f"   {pair['person_b']} | "
        f"{pair['file_b']} | "
        f"Face ID={pair['id_b']}"
    )


# ============================================================
# MOST DANGEROUS SAME-PERSON PAIRS
# ============================================================

same_pairs_sorted = sorted(
    [
        pair
        for pair in pair_details
        if pair["type"] == "SAME"
    ],
    key=lambda x: x["similarity"]
)


print()
print("=" * 70)
print("BOTTOM 10 SAME-PERSON PAIRS")
print("=" * 70)


for rank, pair in enumerate(
    same_pairs_sorted[:10],
    start=1
):

    print()
    print(
        f"#{rank} "
        f"Similarity: "
        f"{pair['similarity']:.6f}"
    )

    print(
        f"   Person {pair['person_a']}"
    )

    print(
        f"   {pair['file_a']} | "
        f"Face ID={pair['id_a']}"
    )

    print(
        f"   {pair['file_b']} | "
        f"Face ID={pair['id_b']}"
    )


# ============================================================
# CLOSE DATABASE
# ============================================================

cursor.close()

conn.close()


print()
print("=" * 70)
print("PostgreSQL connection closed.")
print("Analysis completed.")
print("=" * 70)

