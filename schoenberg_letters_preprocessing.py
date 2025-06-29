import pandas as pd
import re
from pathlib import Path

def process_correspondence_csv(input_file, output_file='schoenberg_letters_chunks.csv', max_chunk_size=800):
    """
    Process correspondence CSV file for RAG implementation
    
    Args:
        input_file: Path to input CSV file
        output_file: Path for output CSV file
        max_chunk_size: Maximum characters per chunk
    
    Returns:
        DataFrame with processed chunks
    """
    
    # Load and clean data
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file, header=None, names=['letter_id', 'text'])
    
    # Remove header row if exists
    if df.iloc[0]['letter_id'] == 'Letter ID':
        df = df.iloc[1:].reset_index(drop=True)
    
    # Clean data
    df = df.dropna(subset=['letter_id', 'text'])
    df = df.drop_duplicates(subset=['letter_id'], keep='first')
    print(f"Loaded {len(df)} letters")
    
    # Clean text
    def clean_text(text):
        if pd.isna(text):
            return text
        
        # Remove excessive whitespace
        text = re.sub(r' {3,}', ' ', text)
        
        # Fix fragmented words
        text = re.sub(r'\b(\w) ([a-z]{2,})\b', r'\1\2', text)
        
        # Common German OCR fixes
        ocr_fixes = {
            r'\bge meldet\b': 'gemeldet',
            r'\bver rechnet\b': 'verrechnet', 
            r'\bDurch füh rung\b': 'Durchführung',
            r'\bEr ledigung\b': 'Erledigung'
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # Clean spacing and line breaks
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    df['text_cleaned'] = df['text'].apply(clean_text)
    
    # Create chunks
    print("Creating chunks...")
    chunks = []
    
    for _, row in df.iterrows():
        letter_id = row['letter_id']
        text = row['text_cleaned']
        
        if pd.isna(text) or len(text.strip()) == 0:
            continue
        
        if len(text) <= max_chunk_size:
            # Keep short letters as single chunk
            chunks.append({
                'letter_id': letter_id,
                'chunk_id': f"{letter_id}_001",
                'text': text,
                'chunk_type': 'full_letter',
                'chunk_index': 1,
                'total_chunks': 1
            })
        else:
            # Split long letters by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = ""
            chunk_index = 1
            letter_chunks = []
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                    letter_chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    chunk_index += 1
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            if current_chunk.strip():
                letter_chunks.append(current_chunk.strip())
            
            # Create chunk records
            for i, chunk_text in enumerate(letter_chunks, 1):
                chunks.append({
                    'letter_id': letter_id,
                    'chunk_id': f"{letter_id}_{i:03d}",
                    'text': chunk_text,
                    'chunk_type': 'partial_letter',
                    'chunk_index': i,
                    'total_chunks': len(letter_chunks)
                })
    
    # Create final DataFrame
    chunks_df = pd.DataFrame(chunks)
    chunks_df['char_count'] = chunks_df['text'].str.len()
    chunks_df['word_count'] = chunks_df['text'].str.split().str.len()
    
    # Save results
    chunks_df.to_csv(output_file, index=False)
    
    print(f"Processing complete!")
    print(f"  Total chunks: {len(chunks_df)}")
    print(f"  Average chunk length: {chunks_df['char_count'].mean():.0f} characters")
    print(f"  Saved to: {output_file}")
    
    return chunks_df

# Simple usage
if __name__ == "__main__":
    # Process the file
    result = process_correspondence_csv('letters_extract.csv')
    
    # Show sample
    print("\n=== SAMPLE CHUNKS ===")
    for i in range(min(3, len(result))):
        chunk = result.iloc[i]
        print(f"\nChunk {i+1}: {chunk['chunk_id']}")
        print(f"  Type: {chunk['chunk_type']}")
        print(f"  Length: {chunk['char_count']} chars")
        print(f"  Preview: {chunk['text'][:150]}...")