import os

# Updated to use the internal Linux project path
PROJECT_SOURCE_DIR = "/home/faridadatascientist/ntu-work/project_vera_main/source_documents" 

def load_all_industries():
    required_domains = ["medical", "aerospace", "energy", "finance"]
    found = []
    
    # Debug print to see what the script sees
    print(f"[DEBUG] Scanning: {PROJECT_SOURCE_DIR}")
    
    for domain in required_domains:
        path = os.path.join(PROJECT_SOURCE_DIR, domain)
        if os.path.isdir(path): # Use isdir to ensure it's a folder
            found.append(domain)
            
    print(f"✅ VERA Data Check: Found {len(found)} industries: {found}")
    return found
    
    # Check each domain folder in your source_documents directory
    for domain in required_domains:
        path = os.path.join(PROJECT_SOURCE_DIR, domain)
        if os.path.exists(path):
            found.append(domain)
            
    print(f"✅ VERA Data Check: Found {len(found)} industries in source_documents: {found}")
    return found

if __name__ == "__main__":
    load_all_industries()