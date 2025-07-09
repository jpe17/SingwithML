mkdir -p 04_extracted_notes
for f in 03_data_preprocessing/voice/*.wav; do
  out="04_extracted_notes/$(basename "${f%.wav}.csv")"
  python 04_extract_notes.py "$f" "$out"
done