from datasets import load_dataset

dataset = load_dataset(
    "changelinglab/librispeech-segment",
    split="train.clean.100",
    streaming=True,
)

# Don't decode audio yet
dataset = dataset.decode(False)

example = next(iter(dataset))

print("Fields:")
print(example.keys())

print("\nText:")
print(example["text"])

print("\nAudio:")
print("Path:", example["audio"]["path"])
print("Bytes:", len(example["audio"]["bytes"]))

print("\nPhones:")
print(example["phones"])

print("\nPhone starts:")
print(example["phone_starts"])

print("\nPhone ends:")
print(example["phone_ends"])

print("\nDuration:")
print(example["duration"])

print("\nSpeaker:")
print(example["speaker_id"])