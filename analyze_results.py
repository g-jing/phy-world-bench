import os
import json
import glob
import argparse

def parse_response(response):
    """Parse the response string into a dictionary of Yes/No answers."""
    try:
        # Remove code block markers if present
        if response.startswith('```'):
            # Remove the first line (```json) and last line (```)
            lines = response.strip().split('\n')
            if len(lines) >= 3:
                response = '\n'.join(lines[1:-1])
        
        # Try to parse as JSON
        return json.loads(response)
    except:
        # If not valid JSON, try to parse the text format
        result = {}
        lines = response.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().strip('"')
                value = value.strip().strip('"').strip(',')
                if value in ['Yes', 'No']:
                    result[key] = value
        return result

def analyze_results(gpt_model="gpt-4o", total_frames=8):
    # Find all result JSON files
    result_files = glob.glob(f'automatic_results/{gpt_model}/frame-{total_frames}/is_two_step_False/*.json')
    
    total_files = len(result_files)
    if total_files == 0:
        print(f"No result files found for {gpt_model} with {total_frames} frames!")
        return
    
    # Initialize counters
    objects_events_yes = 0
    all_standards_yes = 0
    everything_yes = 0
    
    # Process each file
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if 'response' not in data:
            continue
            
        response_dict = parse_response(data['response'])
        if not response_dict:
            continue
            
        # Check if Objects and Event are both Yes
        objects_yes = response_dict.get('Objects', 'No') == 'Yes'
        event_yes = response_dict.get('Event', 'No') == 'Yes'
        if objects_yes and event_yes:
            objects_events_yes += 1
            
        # Check if all standards are Yes
        standards_yes = True
        for key, value in response_dict.items():
            if key.startswith('Standard_') and value != 'Yes':
                standards_yes = False
                break
        if standards_yes:
            all_standards_yes += 1
            
        # Check if everything is Yes
        if objects_yes and event_yes and standards_yes:
            everything_yes += 1
    
    # Calculate percentages
    objects_events_percentage = (objects_events_yes / total_files) * 100
    all_standards_percentage = (all_standards_yes / total_files) * 100
    everything_percentage = (everything_yes / total_files) * 100
    
    # Print results
    print(f"\nAnalysis Results for {gpt_model} with {total_frames} frames (Total files: {total_files}):")
    print(f"Percentage of Objects and Events being Yes: {objects_events_percentage:.2f}%")
    print(f"Percentage of All Standards being Yes: {all_standards_percentage:.2f}%")
    print(f"Percentage of Everything being Yes: {everything_percentage:.2f}%")
    
    # Print raw numbers
    print(f"\nRaw Numbers:")
    print(f"Objects and Events Yes: {objects_events_yes}")
    print(f"All Standards Yes: {all_standards_yes}")
    print(f"Everything Yes: {everything_yes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument('--gpt_model', type=str, default='gpt-4o',
                      help='Model to analyze results for (default: gpt-4o)')
    parser.add_argument('--total_frames', type=int, default=8,
                      help='Number of frames to analyze (default: 8)')
    
    args = parser.parse_args()
    analyze_results(args.gpt_model, args.total_frames) 