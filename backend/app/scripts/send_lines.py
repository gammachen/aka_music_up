#!/usr/bin/env python3
import requests
import json
import time
import argparse
import sys

def send_lines_to_api(file_path, api_url, delay=0, content_type="application/json"):
    """
    Read a file line by line and send each line as a POST request to the specified API endpoint.
    
    Args:
        file_path (str): Path to the file to read
        api_url (str): URL of the API endpoint
        delay (float): Delay between requests in seconds
        content_type (str): Content type header value
    """
    headers = {"Content-Type": content_type}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                # Skip empty lines
                line = line.strip()
                if not line:
                    continue
                
                # Prepare the payload
                payload = {"text": line}
                
                try:
                    # Send the POST request
                    print(f"Sending line {line_number}: {line[:50]}{'...' if len(line) > 50 else ''}")
                    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
                    
                    # Print the response status
                    print(f"Response status: {response.status_code}")
                    if response.status_code != 200:
                        print(f"Response content: {response.text[:100]}")
                    
                    # Add delay if specified
                    if delay > 0:
                        time.sleep(delay)
                        
                except requests.exceptions.RequestException as e:
                    print(f"Error sending request for line {line_number}: {str(e)}")
                    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Send file lines to API endpoint")
    parser.add_argument("file_path", help="Path to the file to read")
    parser.add_argument("--api-url", default="http://127.0.0.1:8020/insert", 
                        help="API endpoint URL (default: http://127.0.0.1:8020/insert)")
    parser.add_argument("--delay", type=float, default=0.0, 
                        help="Delay between requests in seconds (default: 0)")
    parser.add_argument("--content-type", default="application/json", 
                        help="Content-Type header (default: application/json)")
    
    args = parser.parse_args()
    
    print(f"Reading from file: {args.file_path}")
    print(f"Sending to API: {args.api_url}")
    print(f"Delay between requests: {args.delay} seconds")
    
    send_lines_to_api(args.file_path, args.api_url, args.delay, args.content_type)
    
    print("Done!")

if __name__ == "__main__":
    main() 