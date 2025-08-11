# Command

Command to run a specific task.

## 1. Initialize collection

Copy all files in the `./fps/skel` directory to the `~/fps` directory.

```bash
python -m fps.tools.init
```

## 2. Import videos

Step 1: Copy all videos to the collection folder.
Step 2: Create resized videos using ffmpeg.
Step 3: Detect scenes in videos using scenedetect.
Step 4: Post processing detected scenes.
Step 5: Extract frames from videos using scenedetect.
Step 6: Create thumbnails for each video using ffmpeg.

```bash
python -m fps.tools.import VIDEO_PATH --id VIDEO_ID
```

## 3. Analyze videos

Step 1: Extract features.
Step 2: Detect objects.
Step 3: Cluster frames.

```bash
python -m fps.tools.analyze ANALYZERS --id VIDEO_IDs
```

For example, to detect `colors` from video `001`:
```bash
python -m fps.tools.analyze colors --id 001
```

For example, to extract features using `clip-openai` and detect `colors` from videos `001` and `002`:
```bash
python -m fps.tools.analyze clip-openai colors --id 001 002
```

## 4. Create index

Step 1: Encode detected objects as STR.
Step 2: Encode extracted features as STR.
Step 3: Prepare Lucene documents.
Step 4: Add to Lucene index.
Step 5: Add to FAISS index.
Step 6: Compute objects frequency.

```bash
```
