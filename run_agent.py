import asyncio
import os
import sys
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

def process_workflow(agent_type: str = "ocr_agent", mode: str = "ocr", target_file: str = None):
    """
    Processes specified files or all files in the input folder.
    Agents: 'ocr_agent' or 'ocr_customed_agent'
    Modes: 'ocr' or 'description'
    target_file: Optional single filename to process.
    """
    
    # 1. Dynamic Import of the Agent
    if agent_type == "ocr_customed_agent":
        from ocr_customed_agent.agent import ocr_customed_agent as agent
        from ocr_customed_agent.agent import list_input_images, read_image_file, save_markdown_report
        app_name = "ocr_customed_demo"
    else:
        from ocr_agent.agent import ocr_agent as agent
        from ocr_agent.agent import list_input_images, read_image_file, save_markdown_report
        app_name = "ocr_demo"
    
    # 2. Check for API key
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "YOUR_API_KEY_HERE":
        print("\n⚠️  GOOGLE_API_KEY missing. Update your .env first.")
        return

    # 3. Determine which files to process
    if target_file:
        files = [target_file]
    else:
        files = list_input_images()
    
    if not files:
        print(f"No files found.")
        return

    # 4. Set up the Runner and Session
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    mode_label = "OCR Extraction" if mode == "ocr" else "Image-to-Text Description"
    target_label = f"file: {target_file}" if target_file else f"{len(files)} file(s)"
    print(f"🚀 Starting {mode_label} using [{agent_type}] for {target_label}...")

    for filename in files:
        print(f"\n--- 🔍 Processing: {filename} ---")
        
        # 5. Read file bytes
        try:
            file_bytes = read_image_file(filename)
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            continue
        
        # Determine MIME type
        ext = filename.lower().split('.')[-1]
        mime_type = "application/pdf" if ext == "pdf" else f"image/{ext}"
        if ext == "jpg": mime_type = "image/jpeg"

        # Determine prompt and content
        is_large_pdf = False
        if ext == "pdf":
            try:
                # Use metadata tool (imported dynamically)
                meta = get_pdf_metadata(filename)
                if meta.get("num_pages", 0) > 10:
                    is_large_pdf = True
                    print(f"📄 Large PDF detected ({meta['num_pages']} pages). Using iterative chunking...")
            except: pass

        if is_large_pdf:
            # For large PDFs, we DON'T send the full file content in the first turn
            prompt = (
                f"I need you to process the large PDF file: '{filename}'. "
                "It has over 10 pages, so you MUST use your 'split_pdf_pages' tool "
                "to process it in chunks (e.g., 5-10 pages at a time) to avoid token limits. "
                "Consolidate and return the full text when finished."
            )
            content = types.Content(role="user", parts=[types.Part(text=prompt)])
        else:
            # For smaller files, we send the multi-modal content immediately
            if agent_type == "ocr_customed_agent":
                prompt = (
                    f"Please process the file '{filename}'. Use your advanced tools "
                    "(preprocessing/tiling/zooming) if necessary to ensure 100% OCR accuracy. "
                    "Save the final result using save_markdown_report."
                )
            else:
                prompt = (
                    f"Please process the file '{filename}' using OCR mode. "
                    "Extract all text exactly and save it using the save_markdown_report tool."
                    if mode == "ocr" else
                    f"Please process the file '{filename}' using Description mode. "
                    "Provide a concise summary and save it using the save_markdown_report tool."
                )

            content = types.Content(
                role="user",
                parts=[
                    types.Part(text=prompt),
                    types.Part(inline_data=types.Blob(mime_type=mime_type, data=file_bytes))
                ]
            )

        # 6. Run the agent
        session_id = f"{mode}_{filename}"
        asyncio.run(session_service.create_session(app_name=app_name, user_id="demo_user", session_id=session_id))
        
        events = runner.run(user_id="demo_user", session_id=session_id, new_message=content)
        
        final_text = ""
        for event in events:
            if event.is_final_response() and event.content:
                # Capture all text parts
                for part in event.content.parts:
                    if part.text:
                        final_text += part.text
        
        if final_text:
            # 7. Save report using the runner (for reliability)
            mode_suffix = "ocr_report" if mode == "ocr" else "imgtotxt_report"
            result_msg = save_markdown_report(filename, mode_suffix, final_text)
            print(f"✅ {result_msg}")
        else:
            print(f"❌ Error: Agent returned no result for {filename}")

if __name__ == "__main__":
    # Parsing Command Line Arguments
    args = sys.argv[1:]
    
    # Check for --agent flag
    target_agent = "ocr_agent"
    if "--agent" in args:
        idx = args.index("--agent")
        if idx + 1 < len(args):
            target_agent = args[idx + 1]
    
    # Check for --file flag
    target_file = None
    if "--file" in args:
        idx = args.index("--file")
        if idx + 1 < len(args):
            target_file = args[idx + 1]

    # Check for mode
    target_mode = "description" if "description" in args else "ocr"
    
    process_workflow(agent_type=target_agent, mode=target_mode, target_file=target_file)
    print(f"\n🎉 {target_mode.upper()} Processing (Agent: {target_agent}) Finished!")


