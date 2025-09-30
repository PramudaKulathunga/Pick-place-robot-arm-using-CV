import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import main

if __name__ == "__main__":
    print("🤖 Robot Arm Color Sorting System")
    print("=" * 40)
    main()