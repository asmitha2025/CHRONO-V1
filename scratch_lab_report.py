from PIL import Image, ImageDraw, ImageFont
import os

def create_lab_report_image():
    # The updated 2024 TRIDENT FIRES timepoint data
    markers = {
        "LDH": (214.0, "U/L", "140 - 280"),
        "RDW": (14.2, "%", "11.5 - 14.5"),
        "Glucose (Fasting)": (99.5, "mg/dL", "70 - 100"),
        "Albumin": (3.95, "g/dL", "3.5 - 5.5"),
        "Creatinine": (0.89, "mg/dL", "0.6 - 1.2"),
        "C-Reactive Protein": (2.9, "mg/L", "0.0 - 3.0"),
        "ALP": (81.0, "U/L", "44 - 147"),
        "MCV": (91.0, "fL", "80 - 100"),
        "WBC": (7.8, "10^3/uL", "4.5 - 11.0"),
        "Lymphocyte %": (26.5, "%", "20.0 - 40.0"),
        "Neutrophils": (5.4, "10^3/uL", "1.8 - 7.7"),
        "Lymphocytes": (1.73, "10^3/uL", "1.0 - 4.8"),
        "Platelets": (290.0, "10^3/uL", "150 - 400"),
        "Haemoglobin": (13.5, "g/dL", "12.0 - 17.5")
    }

    width, height = 800, 1000
    img = Image.new('RGB', (width, height), color='white')
    d = ImageDraw.Draw(img)

    try:
        # Try to use standard font
        font_title = ImageFont.truetype("arial.ttf", 36)
        font_header = ImageFont.truetype("arial.ttf", 22)
        font_body = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        # Fallback to default if arial is not available
        font_title = ImageFont.load_default()
        font_header = ImageFont.load_default()
        font_body = ImageFont.load_default()

    # Header
    d.text((50, 50), "APOLLO DIAGNOSTICS - CHENNAI", fill="black", font=font_title)
    d.text((50, 120), "Patient Name: Priya Demo", fill="black", font=font_header)
    d.text((50, 155), "Patient ID: priya_demo_001", fill="black", font=font_header)
    d.text((50, 190), "Date: 2024-01-18", fill="black", font=font_header)

    d.line([(50, 240), (750, 240)], fill="black", width=3)
    
    # Table Headers
    d.text((50, 260), "TEST NAME", fill="black", font=font_header)
    d.text((350, 260), "RESULT", fill="black", font=font_header)
    d.text((500, 260), "UNIT", fill="black", font=font_header)
    d.text((620, 260), "REF RANGE", fill="black", font=font_header)
    
    d.line([(50, 300), (750, 300)], fill="black", width=1)

    # Table Body
    y_pos = 320
    for name, (val, unit, ref) in markers.items():
        d.text((50, y_pos), name, fill="black", font=font_body)
        
        # Highlight if it were out of bounds, but it's not!
        val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
        d.text((350, y_pos), val_str, fill="black", font=font_body)
        
        d.text((500, y_pos), unit, fill="black", font=font_body)
        d.text((620, y_pos), ref, fill="black", font=font_body)
        y_pos += 40

    d.line([(50, y_pos), (750, y_pos)], fill="black", width=2)
    
    # Footer statement
    d.text((50, y_pos + 30), "INTERPRETATION:", fill="black", font=font_header)
    d.text((50, y_pos + 60), "All values fall within the normal population reference range.", fill="black", font=font_body)
    
    # Ensure directory exists
    os.makedirs("c:/Users/harih/OneDrive/Desktop/CHRONO-V1/data/sample_patients", exist_ok=True)
    out_path = "c:/Users/harih/OneDrive/Desktop/CHRONO-V1/data/sample_patients/priya_lab_report_2024.png"
    img.save(out_path)
    print(f"Mock lab report saved to: {out_path}")

if __name__ == "__main__":
    create_lab_report_image()
