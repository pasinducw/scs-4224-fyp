## Requirements
- Dataset must contain mp3 files named in the following format:
  - 1.mp3, 2.mp3, 3.mp3, ...
- Install dependencies
  - `pip install -r requirements.txt`

## Generated features
- The application applies 4 transforms to the original songs, altogether creating 5 feature files from original mp3
- Feature files are named in the following format
  - [song id]-[transformation-id].mp3
  - Example: 12-3.mp3

## Transformation IDs
- 0: Original
- 1: Noised
- 2-5: Pitch shift
- 6-10: Speed shift
