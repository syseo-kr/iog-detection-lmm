import os
import base64
import csv
import time
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configuration variables
BATCH_SIZE = 10  # Number of images to process at once
IMAGE_FOLDER = "img"  # Path to folder containing images
OUTPUT_CSV = "gambling_results.csv"  # Path to CSV file for saving results

# LLM model settings
OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
GEMINI_MODEL = "models/gemini-2.5-flash-preview-04-17"

# Prompt settings
PROMPT = f"""
You will be given a set of image files, each representing a screenshot of a website's landing page. Your task is to determine whether each website appears to be operating as an illegal online gambling site, based solely on the visual content of the image.

Instructions:
• 'O' → The image shows signs of an illegal online gambling website.
• 'X' → The image does not appear to be an illegal online gambling website.
• Use only the visual information in the image for classification.

IMPORTANT: You will be provided with {BATCH_SIZE} images. You MUST analyze ALL {BATCH_SIZE} images provided and include a judgment for EACH one. Do not skip any images.
Make sure you include EVERY filename in your response with its corresponding judgment.

Output Format: 
For each image, provide ONLY the filename and your judgment (O or X) in this exact format:
filename: [exact original filename] judgment: [O or X]

CRITICAL INSTRUCTIONS:
• DO NOT output any additional information other than the output format presented.
• DO NOT change or mix up the name of the image file when providing your judgment.
• DO NOT describe the images or explain your reasoning.
• DO NOT write any descriptions or analysis of what you see in the images.
• ONLY provide the filename and judgment in the exact format specified.

You must analyze exactly these files and include ALL of them in your response:
"""

# API key setup
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def encode_image_to_base64(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_to_bytes(image_path):
    """Encode image to bytes"""
    with open(image_path, "rb") as image_file:
        return image_file.read()

def get_all_image_files():
    """Get all image file paths from the image folder"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    
    for file in os.listdir(IMAGE_FOLDER):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(IMAGE_FOLDER, file))
    
    return image_files

def process_images_with_gpt(image_batch, max_retries=3):
    """Process images with GPT model"""
    last_result = ""  # Variable to store the last response
    
    for attempt in range(max_retries):
        try:
            # Include image file list first
            filenames = [os.path.basename(img) for img in image_batch]
            file_list_text = "\n" + "\n".join(filenames)
            
            messages = [{"role": "system", "content": PROMPT + "\n\n" + file_list_text}]
            
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                base64_image = encode_image_to_base64(img_path)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Image with filename: {filename}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                })
            
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            last_result = result  # Store result
            
            # Print response result
            print("\n----- GPT Response Result (Attempt {}/{}) -----".format(attempt+1, max_retries))
            print(result)
            print("----------------------------")
            
            # Response validation: Check if judgment exists for each file
            valid_response = True
            missing_files = []
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                if filename not in result and f"filename: {filename}" not in result:
                    valid_response = False
                    missing_files.append(filename)
            
            if not valid_response:
                print(f"Could not find judgment for the following files in GPT response: {', '.join(missing_files)}. Retrying... (Attempt {attempt+1}/{max_retries})")
            else:
                return result
            
            # Wait briefly before retry
            time.sleep(2)
            
        except Exception as e:
            print(f"GPT processing error: {e}. Retrying... (Attempt {attempt+1}/{max_retries})")
            time.sleep(2)
    
    # If maximum retry count exceeded
    print(f"GPT processing failed: Maximum retry limit ({max_retries}) exceeded.")
    return last_result  # Return last response even if invalid

def process_images_with_claude(image_batch, max_retries=3):
    """Process images with Claude model"""
    last_result = ""  # Variable to store last response
    
    for attempt in range(max_retries):
        try:
            # Include image file list first
            filenames = [os.path.basename(img) for img in image_batch]
            file_list_text = "\n" + "\n".join(filenames)
            
            messages = [{"role": "user", "content": [
                {"type": "text", "text": PROMPT + "\n\n" + file_list_text},
            ]}]
            
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                with open(img_path, "rb") as img_file:
                    img_data = img_file.read()
                    
                    # Detect image format
                    media_type = "image/jpeg"  # Default
                    file_ext = os.path.splitext(img_path)[1].lower()
                    if file_ext == ".png":
                        media_type = "image/png"
                    elif file_ext == ".gif":
                        media_type = "image/gif"
                    elif file_ext in [".jpeg", ".jpg"]:
                        media_type = "image/jpeg"
                    elif file_ext == ".webp":
                        media_type = "image/webp"
                    
                    messages[0]["content"].append(
                        {"type": "text", "text": f"Image with filename: {filename}"}
                    )
                    messages[0]["content"].append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(img_data).decode('utf-8')
                        }
                    })
            
            response = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1000,
                messages=messages
            )
            
            result = response.content[0].text
            last_result = result  # Store result
            
            # Print response result (only once)
            print("\n----- Claude Response Result (Attempt {}/{}) -----".format(attempt+1, max_retries))
            print(result)
            print("----------------------------")
            
            # Response validation: Check if judgment exists for each file
            valid_response = True
            missing_files = []
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                if filename not in result and f"filename: {filename}" not in result:
                    valid_response = False
                    missing_files.append(filename)
            
            if not valid_response:
                print(f"Could not find judgment for the following files in Claude response: {', '.join(missing_files)}. Retrying... (Attempt {attempt+1}/{max_retries})")
            else:
                return result  # Return immediately if valid response (no additional output)
            
            # Wait briefly before retry
            time.sleep(2)
            
        except Exception as e:
            print(f"Claude processing error: {e}. Retrying... (Attempt {attempt+1}/{max_retries})")
            time.sleep(2)
    
    # If maximum retry count exceeded
    print(f"Claude processing failed: Maximum retry limit ({max_retries}) exceeded.")
    return last_result  # Return last response even if invalid

def process_images_with_gemini(image_batch, max_retries=3):
    """Process images with Gemini model"""
    last_result = ""  # Variable to store last response
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            # Include image file list first
            filenames = [os.path.basename(img) for img in image_batch]
            file_list_text = "\n" + "\n".join(filenames)
            
            contents = [PROMPT + "\n\n" + file_list_text]
            
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                contents.append(f"Image with filename: {filename}")
                img = Image.open(img_path)
                contents.append(img)
            
            response = model.generate_content(contents)
            
            result = response.text
            last_result = result  # Store result
            
            # Print response result
            print("\n----- Gemini Response Result (Attempt {}/{}) -----".format(attempt+1, max_retries))
            print(result)
            print("----------------------------")
            
            # Response validation: Check if judgment exists for each file
            valid_response = True
            missing_files = []
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                if filename not in result and f"filename: {filename}" not in result:
                    valid_response = False
                    missing_files.append(filename)
            
            if not valid_response:
                print(f"Could not find judgment for the following files in Gemini response: {', '.join(missing_files)}. Retrying... (Attempt {attempt+1}/{max_retries})")
            else:
                return result
            
            # Wait briefly before retry
            time.sleep(2)
            
        except Exception as e:
            print(f"Gemini processing error: {e}. Retrying... (Attempt {attempt+1}/{max_retries})")
            time.sleep(2)
    
    # If maximum retry count exceeded
    print(f"Gemini processing failed: Maximum retry limit ({max_retries}) exceeded.")
    return last_result  # Return last response even if invalid

def parse_results(text, filenames):
    """Parse filename and judgment results from model response - supports various formats"""
    results = {}
    base_filenames = [os.path.basename(f) for f in filenames]
    
    if not text or text.strip() == "":
        print("Response is empty.")
        return results
    
    print("\n=== Text being parsed ===")
    print(text)
    print("========================\n")
    
    # 1. Handle standard format (filename: xxx judgment: O/X)
    lines = text.strip().split('\n')
    for line in lines:
        if 'judgment:' in line.lower() or 'judgment :' in line.lower():
            parts = line.split('judgment:') if 'judgment:' in line.lower() else line.split('judgment :')
            filename_part = parts[0].strip()
            judgment_part = parts[1].strip() if len(parts) > 1 else ""
            
            # Extract filename
            filename = ""
            if "filename:" in filename_part.lower():
                filename = filename_part.split("filename:")[1].strip()
            elif "filename :" in filename_part.lower():
                filename = filename_part.split("filename :")[1].strip()
            else:
                # If filename is not clearly indicated, match in order
                for f in base_filenames:
                    if f in filename_part:
                        filename = f
                        break
            
            # Remove brackets or quotes from filename
            filename = filename.strip("[]'\"\t ")
            
            # Extract judgment (O or X)
            judgment = ""
            if "O" in judgment_part or "o" in judgment_part:
                judgment = "O"
            elif "X" in judgment_part or "x" in judgment_part:
                judgment = "X"
            
            if filename and judgment:
                results[filename] = judgment
                print(f"Parsing result (standard format): {filename} -> {judgment}")
    
    # 2. Handle simple format (filename.png: O/X)
    if len(results) < len(base_filenames):
        for line in lines:
            if ':' in line and ('O' in line or 'X' in line or 'o' in line or 'x' in line):
                parts = line.split(':')
                if len(parts) >= 2:
                    filename_part = parts[0].strip()
                    judgment_part = parts[1].strip()
                    
                    # Match filename
                    matched_filename = ""
                    for f in base_filenames:
                        if f in filename_part or filename_part in f:
                            matched_filename = f
                            break
                    
                    if not matched_filename:
                        continue
                    
                    # Extract judgment
                    judgment = ""
                    if "O" in judgment_part or "o" in judgment_part:
                        judgment = "O"
                    elif "X" in judgment_part or "x" in judgment_part:
                        judgment = "X"
                    
                    if matched_filename and judgment and matched_filename not in results:
                        results[matched_filename] = judgment
                        print(f"Parsing result (simple format): {matched_filename} -> {judgment}")
    
    # 3. Handle table format (markdown table or other formats)
    if len(results) < len(base_filenames):
        for line in lines:
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                for part in parts:
                    if not part:
                        continue
                    
                    # Find filename and judgment in each cell
                    file_match = None
                    for f in base_filenames:
                        if f in part:
                            file_match = f
                            break
                    
                    if not file_match:
                        continue
                    
                    # Extract judgment
                    judgment = ""
                    if "O" in part or "o" in part:
                        judgment = "O"
                    elif "X" in part or "x" in part:
                        judgment = "X"
                    
                    if file_match and judgment and file_match not in results:
                        results[file_match] = judgment
                        print(f"Parsing result (table format): {file_match} -> {judgment}")
    
    # 4. Handle list format (filename and judgment are close together)
    if len(results) < len(base_filenames):
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            # Find filename in current line
            file_match = None
            for f in base_filenames:
                if f in line:
                    file_match = f
                    break
            
            if not file_match or file_match in results:
                continue
            
            # Find judgment in current line or next line
            judgment = ""
            search_lines = [line]
            if i + 1 < len(lines):
                search_lines.append(lines[i + 1])
            
            for search_line in search_lines:
                if "O" in search_line or "o" in search_line:
                    judgment = "O"
                    break
                elif "X" in search_line or "x" in search_line:
                    judgment = "X"
                    break
            
            if file_match and judgment:
                results[file_match] = judgment
                print(f"Parsing result (list format): {file_match} -> {judgment}")
    
    # 5. Last attempt - infer judgment from entire text for all remaining files
    if len(results) < len(base_filenames):
        for f in base_filenames:
            if f not in results:
                # Find context around filename
                context = ""
                for line in lines:
                    if f in line:
                        context = line
                        break
                
                if not context:
                    continue
                
                # Find O/X around filename
                judgment = ""
                if "O" in context or "o" in context:
                    judgment = "O"
                elif "X" in context or "x" in context:
                    judgment = "X"
                
                if judgment:
                    results[f] = judgment
                    print(f"Parsing result (context inference): {f} -> {judgment}")
    
    # 6. Handle completely different special format (filename.png: X format)
    if len(results) < len(base_filenames):
        for line in lines:
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    filename_part = parts[0].strip()
                    value_part = parts[1].strip()
                    
                    # Check filename
                    if filename_part.endswith('.png') or filename_part.endswith('.jpg') or filename_part.endswith('.jpeg'):
                        for f in base_filenames:
                            if filename_part == f:
                                # Extract judgment
                                judgment = ""
                                if value_part.strip() == 'O' or value_part.strip() == 'o':
                                    judgment = "O"
                                elif value_part.strip() == 'X' or value_part.strip() == 'x':
                                    judgment = "X"
                                
                                if judgment and f not in results:
                                    results[f] = judgment
                                    print(f"Parsing result (special format): {f} -> {judgment}")
    
    print(f"Final parsing result ({len(results)}/{len(base_filenames)} files processed): {results}")
    return results

def create_log_directory():
    """Create log directory"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def create_log_files(timestamp):
    """Create log directory and files"""
    log_dir = create_log_directory()
    
    # Create one consolidated log file
    combined_log_file = f"{log_dir}/combined_log_{timestamp}.log"
    
    with open(combined_log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Illegal Gambling Site Detection Log ({timestamp}) ===\n\n")
    
    return log_dir, combined_log_file

def append_to_log(log_file, batch_number, gpt_result, claude_result, gemini_result, gpt_parsed, claude_parsed, gemini_parsed, current_filenames):
    """Append results to log file"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    with open(log_file, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n\n{'='*50}\n")
        log_file.write(f"=== Batch {batch_number} Log ({timestamp}) ===\n")
        log_file.write(f"{'='*50}\n\n")
        log_file.write(f"Processing images: {', '.join(current_filenames)}\n\n")
        
        log_file.write("=== GPT Model Response ===\n")
        log_file.write(gpt_result if gpt_result else "No response")
        log_file.write("\n\n")
        
        log_file.write("=== Claude Model Response ===\n")
        log_file.write(claude_result if claude_result else "No response")
        log_file.write("\n\n")
        
        log_file.write("=== Gemini Model Response ===\n")
        log_file.write(gemini_result if gemini_result else "No response")
        log_file.write("\n\n")
        
        log_file.write("=== Parsing Results Summary ===\n")
        log_file.write("Filename, GPT, Claude, Gemini\n")
        for filename in current_filenames:
            gpt_judgment = gpt_parsed.get(filename, "")
            claude_judgment = claude_parsed.get(filename, "")
            gemini_judgment = gemini_parsed.get(filename, "")
            log_file.write(f"{filename}, {gpt_judgment}, {claude_judgment}, {gemini_judgment}\n")
        
        # Check if there are unparsed responses
        missing_gpt = [f for f in current_filenames if f not in gpt_parsed]
        missing_claude = [f for f in current_filenames if f not in claude_parsed]
        missing_gemini = [f for f in current_filenames if f not in gemini_parsed]
        
        if missing_gpt or missing_claude or missing_gemini:
            log_file.write("\n=== Missing Judgment Results ===\n")
            if missing_gpt:
                log_file.write(f"Files missing from GPT: {', '.join(missing_gpt)}\n")
            if missing_claude:
                log_file.write(f"Files missing from Claude: {', '.join(missing_claude)}\n")
            if missing_gemini:
                log_file.write(f"Files missing from Gemini: {', '.join(missing_gemini)}\n")
    
    print(f"Log has been added to the consolidated log file.")

def append_error_to_log(log_file, batch_number, error, current_filenames):
    """Add error information to log file"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n{'='*50}\n")
        f.write(f"=== Batch {batch_number} Error Log ({timestamp}) ===\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Error occurred during processing batch {batch_number}: {error}\n")
        f.write(f"Processing images: {', '.join(current_filenames)}\n")
    
    print(f"Error information has been added to the consolidated log file.")

def main():
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = f"gambling_results_{timestamp}.csv"  # Add timestamp to filename
    
    # Create log directory and files
    log_dir, combined_log_file = create_log_files(timestamp)
    
    # Get all image files
    all_images = get_all_image_files()
    total_images = len(all_images)
    
    if total_images == 0:
        print(f"No images found in '{IMAGE_FOLDER}' folder.")
        return
    
    print(f"Total of {total_images} images can be processed.")
    
    # Get starting index input
    start_index_input = input(f"Which image number would you like to start from? (1-{total_images}, default: 1): ").strip()
    start_index = 1
    
    try:
        if start_index_input:
            start_index = int(start_index_input)
            if start_index < 1:
                print("Cannot enter value less than 1. Starting from 1.")
                start_index = 1
            elif start_index > total_images:
                print(f"Cannot enter value greater than {total_images}. Starting from 1.")
                start_index = 1
    except ValueError:
        print("Not a valid number. Starting from 1.")
        start_index = 1
    
    # Convert starting index to batch index
    batch_start_index = ((start_index - 1) // BATCH_SIZE) * BATCH_SIZE
    
    # Display if already processed images exist
    if start_index > 1:
        print(f"Starting processing from image {start_index}. (Images 1-{start_index-1} will be skipped.)")
        
        # Check existing CSV file
        continue_csv = input("If you have an existing CSV file, would you like to continue writing to it? (y/n, default: n): ").strip().lower()
        
        if continue_csv == 'y':
            existing_csv = input("Enter existing CSV file path: ").strip()
            if os.path.exists(existing_csv):
                output_csv = existing_csv
                print(f"Results will be appended to '{output_csv}' file.")
            else:
                print(f"Could not find '{existing_csv}' file. Results will be saved to new file '{output_csv}'.")
                # Initialize CSV file
                with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['filename', 'gpt', 'claude', 'gemini'])
        else:
            # Initialize CSV file
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['filename', 'gpt', 'claude', 'gemini'])
    else:
        # Initialize CSV file
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['filename', 'gpt', 'claude', 'gemini'])
    
    print(f"Results will be saved to '{output_csv}' file.")
    print(f"Logs will be saved to '{combined_log_file}' file.")
    
    # Process in batches
    batch_index = batch_start_index
    while batch_index < total_images:
        # Select current batch images
        end_index = min(batch_index + BATCH_SIZE, total_images)
        current_batch = all_images[batch_index:end_index]
        current_filenames = [os.path.basename(img) for img in current_batch]
        batch_number = batch_index//BATCH_SIZE + 1
        
        print(f"\nProcessing batch {batch_number}... ({batch_index+1}-{end_index}/{total_images})")
        print("Currently processing images:", ", ".join(current_filenames))
        
        try:
            # GPT model processing
            print("\n===== Processing with GPT model... =====")
            gpt_result = ""
            gpt_retry_count = 0
            max_model_retries = 3
            
            while gpt_retry_count < max_model_retries:
                try:
                    gpt_result = process_images_with_gpt(current_batch)
                    
                    # Parse results (don't print response results again)
                    print("\n===== Parsing GPT results... =====")
                    gpt_parsed = parse_results(gpt_result, current_batch)
                    
                    # Validate results
                    if len(gemini_parsed) == len(current_batch):
                        break  # Break if successfully received results for all files
                    else:
                        gemini_retry_count += 1
                        if gemini_retry_count < max_model_retries:
                            print(f"Gemini results are incomplete. Retrying Gemini only... (Attempt {gemini_retry_count}/{max_model_retries})")
                            time.sleep(2)
                        else:
                            print(f"Gemini maximum retry count ({max_model_retries}) exceeded. Proceeding with current results.")
                
                except Exception as e:
                    gemini_retry_count += 1
                    print(f"Error occurred during Gemini processing: {e}")
                    if gemini_retry_count < max_model_retries:
                        print(f"Retrying Gemini... (Attempt {gemini_retry_count}/{max_model_retries})")
                        time.sleep(2)
                    else:
                        print(f"Gemini maximum retry count ({max_model_retries}) exceeded.")
                        gemini_parsed = {}  # Set to empty result
            
            # Add results to log file
            append_to_log(
                combined_log_file, batch_number, gpt_result, claude_result, gemini_result,
                gpt_parsed, claude_parsed, gemini_parsed, current_filenames
            )
            
            # Add results to CSV
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                for img_path in current_batch:
                    filename = os.path.basename(img_path)
                    gpt_judgment = gpt_parsed.get(filename, "")
                    claude_judgment = claude_parsed.get(filename, "")
                    gemini_judgment = gemini_parsed.get(filename, "")
                    
                    csv_writer.writerow([filename, gpt_judgment, claude_judgment, gemini_judgment])
                    print(f"Added to CSV: {filename}, GPT: {gpt_judgment}, Claude: {claude_judgment}, Gemini: {gemini_judgment}")
            
            print(f"Batch {batch_number} results have been saved to CSV file.")
                
        except Exception as e:
            print(f"Unexpected error occurred during processing: {e}")
            # Record error log
            append_error_to_log(combined_log_file, batch_number, e, current_filenames)
            
            # Continue to next batch even if unexpected error occurs (without user input)
            print(f"An error occurred but proceeding to next batch.")
        
        # Move to next batch (automatically proceed without user confirmation)
        batch_index = end_index
    
    print("\nAll processing completed.")
    print(f"Results saved to '{output_csv}' file.")
    print(f"Detailed logs saved to '{combined_log_file}' file.")

if __name__ == "__main__":
    main()
                    if len(gpt_parsed) == len(current_batch):
                        break  # Break if successfully received results for all files
                    else:
                        gpt_retry_count += 1
                        if gpt_retry_count < max_model_retries:
                            print(f"GPT results are incomplete. Retrying GPT only... (Attempt {gpt_retry_count}/{max_model_retries})")
                            time.sleep(2)
                        else:
                            print(f"GPT maximum retry count ({max_model_retries}) exceeded. Proceeding with current results.")
                
                except Exception as e:
                    gpt_retry_count += 1
                    print(f"Error occurred during GPT processing: {e}")
                    if gpt_retry_count < max_model_retries:
                        print(f"Retrying GPT... (Attempt {gpt_retry_count}/{max_model_retries})")
                        time.sleep(2)
                    else:
                        print(f"GPT maximum retry count ({max_model_retries}) exceeded.")
                        gpt_parsed = {}  # Set to empty result
            
            # Claude model processing
            print("\n===== Processing with Claude model... =====")
            claude_result = ""
            claude_retry_count = 0
            
            while claude_retry_count < max_model_retries:
                try:
                    claude_result = process_images_with_claude(current_batch)
                    
                    # Parse results
                    print("\n===== Parsing Claude results... =====")
                    claude_parsed = parse_results(claude_result, current_batch)
                    
                    # Validate results
                    if len(claude_parsed) == len(current_batch):
                        break  # Break if successfully received results for all files
                    else:
                        claude_retry_count += 1
                        if claude_retry_count < max_model_retries:
                            print(f"Claude results are incomplete. Retrying Claude only... (Attempt {claude_retry_count}/{max_model_retries})")
                            time.sleep(2)
                        else:
                            print(f"Claude maximum retry count ({max_model_retries}) exceeded. Proceeding with current results.")
                
                except Exception as e:
                    claude_retry_count += 1
                    print(f"Error occurred during Claude processing: {e}")
                    if claude_retry_count < max_model_retries:
                        print(f"Retrying Claude... (Attempt {claude_retry_count}/{max_model_retries})")
                        time.sleep(2)
                    else:
                        print(f"Claude maximum retry count ({max_model_retries}) exceeded.")
                        claude_parsed = {}  # Set to empty result
            
            # Gemini model processing
            print("\n===== Processing with Gemini model... =====")
            gemini_result = ""
            gemini_retry_count = 0
            
            while gemini_retry_count < max_model_retries:
                try:
                    gemini_result = process_images_with_gemini(current_batch)
                    
                    # Parse results
                    print("\n===== Parsing Gemini results... =====")
                    gemini_parsed = parse_results(gemini_result, current_batch)
                    
                    # Validate results