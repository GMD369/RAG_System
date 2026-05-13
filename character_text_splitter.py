from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""

print("=" * 60)
print("1. CHARACTER TEXT SPLITTER — fails on long paragraphs")
print("=" * 60)

splitter1 = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=0)
chunks1 = splitter1.split_text(tesla_text)
for i, chunk in enumerate(chunks1, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()

print("\n" + "=" * 60)
print("2. RECURSIVE CHARACTER TEXT SPLITTER — handles it correctly")
print("=" * 60)

splitter2 = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=100,
    chunk_overlap=0,
)
chunks2 = splitter2.split_text(tesla_text)
for i, chunk in enumerate(chunks2, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()
