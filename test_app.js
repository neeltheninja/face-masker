const fs = require('fs');

// We don't need a real browser, we just need to see if detect() throws!
const faceDetectionStr = fs.readFileSync('face-detection.js', 'utf8');

// Just parse the file to check for SYNTAX ERRORS!
try {
  new Function(faceDetectionStr);
  console.log("face-detection.js parsed successfully");
} catch (e) {
  console.log("Syntax error in face-detection.js: ", e);
}

const appStr = fs.readFileSync('app.js', 'utf8');
try {
  new Function(appStr);
  console.log("app.js parsed successfully");
} catch (e) {
  console.log("Syntax error in app.js: ", e);
}
