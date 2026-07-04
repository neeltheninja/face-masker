const fs = require('fs');
const jsdom = require("jsdom");
const { JSDOM } = jsdom;
const dom = new JSDOM(`<!DOCTYPE html><html><body><div class="app-container"></div></body></html>`);
global.window = dom.window;
global.document = dom.window.document;
global.Image = dom.window.Image;
global.FileReader = dom.window.FileReader;

// Mock context
window.HTMLCanvasElement.prototype.getContext = function () {
    return {
        drawImage: () => {},
        getImageData: () => ({ data: new Uint8ClampedArray(400) }),
    };
};

const appCode = fs.readFileSync('app.js', 'utf8');
eval(appCode);
console.log("App parsed");
