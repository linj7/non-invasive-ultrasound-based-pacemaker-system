const express = require('express');
const timeout = require('connect-timeout');
const fileUpload = require('express-fileupload');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

const app = express();
const port = 8080;

app.use(fileUpload());
app.use(timeout('600s')); 

app.use(express.json());

// check timeout
function haltOnTimedout(req, res, next) {
    if (!req.timedout) next();
}

app.post('/receive-video', haltOnTimedout, (req, res) => {
    if (!req.files || Object.keys(req.files).length === 0) {
        console.error("No files were uploaded.");
        return res.status(400).json({ error: "No files were uploaded." });
    }

    let videoFile = req.files.video;
    console.log("Received video file:", videoFile.name);

    // Use timestamp as the file name
    const now = new Date();
    const timestamp = `${now.getMonth() + 1}_${now.getDate()}_${now.getHours()}_${now.getMinutes()}_${now.getSeconds()}`;
    const newFileName = `${timestamp}${path.extname(videoFile.name)}`;
    console.log("New file name is :", newFileName)
    const parsedPath = path.parse(newFileName);
    const newFileNameWithAvi = `${parsedPath.name}.avi`;

    const savePath = path.join(__dirname, 'a4c-video-dir', 'Videos', newFileName);
    const FileListPath = path.join(__dirname, 'a4c-video-dir', 'FileList.csv');
    const VolumeTracingsPath = path.join(__dirname, 'a4c-video-dir', 'VolumeTracings.csv');
    const modelOutputPath = path.join(__dirname, 'output', 'segmentation', 'deeplabv3_resnet50_random', 'videos', newFileNameWithAvi);

    videoFile.mv(savePath, function(err) {
        if (err) {
            console.error("Error saving file:", err);
            return res.status(500).json({ error: "Failed to save file", details: err });
        }

        console.log("File saved successfully at:", savePath);

        const commandPython = `python3 video_preprocess.py ${savePath} ${FileListPath} ${VolumeTracingsPath}`;
        exec(commandPython, (error, stdout, stderr) => {
            if (error) {
                console.error("Error executing video_preprocess.py:", stderr);
                return res.status(500).json({ error: "Failed to execute Python script", details: stderr });
            }
            console.log("Video preprocess successfully:", stdout);

            const weightsPath = path.join(__dirname, 'output', 'segmentation', 'deeplabv3_resnet50_random', 'best.pt');
            const echonetCommand = `echonet segmentation --save_video --run_test --weights ${weightsPath}`;
            exec(echonetCommand, (echonetError, echonetStdout, echonetStderr) => {
                if (echonetError) {
                    console.error("Error executing echonet segmentation command:", echonetStderr);
                    return res.status(500).json({ error: "Failed to execute echonet command", details: echonetStderr });
                }
                console.log("Echonet segmentation command executed successfully:", echonetStdout);

                const getPositionsCommand = `python3 get_three_positions.py ${modelOutputPath}`;
                exec(getPositionsCommand, (positionsError, positionsStdout, positionsStderr) => {
                    if (positionsError) {
                        console.error("Error executing get_three_positions.py:", positionsStderr);
                        return res.status(500).json({ error: "Failed to execute Python script", details: positionsStderr });
                    }
                    console.log("Annotated video successfully:", positionsStdout);

                    try {
                        // Parse the json output of Python script, extract pythonCoordinates
                        const pythonCoordinates = JSON.parse(positionsStdout);
                        console.log("Parsed Python coordinates:", pythonCoordinates);

                        const videoWeightsPath = path.join(__dirname, 'output', 'video', 'r2plus1d_18_32_2_pretrained', 'best.pt');
                        const echonetCommand = `echonet video --batch_size 1 --run_test --weights ${videoWeightsPath}`;
                        exec(echonetCommand, (echonetError, echonetStdout, echonetStderr) => {
                            if (echonetError) {
                                console.error("Error executing echonet video command:", echonetStderr);
                                return res.status(500).json({ error: "Failed to execute echonet video command", details: echonetStderr });
                            }
                            console.log("Echonet video command executed successfully:", echonetStdout);

                            // Read the test_predictions.csv file and find the row where the file name matches.
                            const csvFilePath = path.join(__dirname, 'output', 'video', 'r2plus1d_18_32_2_pretrained', 'test_predictions.csv');

                            fs.readFile(csvFilePath, 'utf8', (err, data) => {
                                if (err) {
                                    console.error("Error reading CSV file:", err);
                                    return res.status(500).json({ error: "Failed to read CSV file", details: err.message });
                                }

                                // segment CSV data
                                const rows = data.split('\n').map(row => row.split(','));
                                
                                // Find the row with the file name newFileNameWithAvi and retrieve the value in the third column as ejectionFraction.
                                const targetRow = rows.find(row => row[0] === newFileNameWithAvi);
                                if (!targetRow || targetRow.length < 3) {
                                    console.error("File name not found or data is insufficient in CSV.");
                                    return res.status(500).json({ error: "File name not found or data is insufficient in CSV" });
                                }

                                const ejectionFraction = parseFloat(targetRow[2].trim()); 
                                console.log("Extracted ejectionFraction:", ejectionFraction);

                                // return ejectionFraction and coordinates to the client
                                const coordinates = [
                                    { x: pythonCoordinates['2'][0].toString(), y: pythonCoordinates['2'][1].toString() },
                                    { x: pythonCoordinates['3'][0].toString(), y: pythonCoordinates['3'][1].toString() },
                                    { x: pythonCoordinates['4'][0].toString(), y: pythonCoordinates['4'][1].toString() }
                                ];

                                const pacingRequired = ejectionFraction < 40;

                                res.json({
                                    ejectionFraction: ejectionFraction,
                                    pacingRequired: pacingRequired,
                                    coordinates: coordinates
                                });
                            });
                        });
                    } catch (parseError) {
                        console.error("Error parsing JSON from Python script:", parseError);
                        return res.status(500).json({ error: "Failed to parse JSON from Python script", details: parseError.message });
                    }
                });
            });
        });
    });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}/`);
});
