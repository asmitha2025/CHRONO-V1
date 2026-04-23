import PyPDF2
import sys
sys.stdout.reconfigure(encoding='utf-8')

reader = PyPDF2.PdfReader(r'c:\Users\harih\OneDrive\Desktop\CHRONO-V1\CHRONO_Full_Technical_Document.pdf')
print(f'Total Pages: {len(reader.pages)}')
print()

full_text = ""
for i in range(len(reader.pages)):
    text = reader.pages[i].extract_text()
    full_text += f'=== PAGE {i+1} ===\n{text}\n\n'

with open(r'c:\Users\harih\OneDrive\Desktop\CHRONO-V1\extracted_text.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)

print(f"Extracted {len(reader.pages)} pages to extracted_text.txt")
