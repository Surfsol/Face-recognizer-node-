const { canvas } = require("./canvas.js");
const { faceDetectionOptions } = require("./faceDetection.js");
const { faceDetectionNet } = require("./faceDetection.js");
const express = require("express");
const tf = require('@tensorflow/tfjs-node')
const faceapi = require('@vladmandic/face-api');

const router = express.Router();

//"https://api.tinify.com/output/fqrda3484ym3b7ghwc2neujnzmt6a8gy","https://api.tinify.com/output/f2j6y3cgk009dewfp55713xgta4cbnuv","https://api.tinify.com/output/ekdntqh6zfwt4uy39rjvmy4564jtnd8w"
// const REFERENCE_IMAGE =
//   "https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/Thor/2.jpg";

// let arrayOfObj = [
//   {
//     img:
//       "https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/Hawkeye/1.jpg",
//     verified: false,
//     adult: false,
//   },
//   {
//     img:
//       "https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/Thor/2.jpg",
//     verified: false,
//     adult: false,
//   },
//   {
//     img:
//       "https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/Hawkeye/1.jpg",
//     verified: false,
//     adult: false,
//   },
// ];


async function run(avatar, REFERENCE_IMAGE) {
  console.log('in function run')
  await faceDetectionNet.loadFromDisk("./models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
  await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");

  // for (let i = 0; i < arrayOfObj.length; i++) {
    let QUERY_IMAGE = avatar;
    const referenceImage = await canvas.loadImage(REFERENCE_IMAGE);
    const queryImage = await canvas.loadImage(QUERY_IMAGE);

    // detect faces
    const resultsRef = await faceapi
      .detectAllFaces(referenceImage, faceDetectionOptions)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resultsQuery = await faceapi
      .detectAllFaces(queryImage, faceDetectionOptions)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const faceMatcher = new faceapi.FaceMatcher(resultsRef, 0.55);
    console.log({faceMatcher})

    const labels = faceMatcher.labeledDescriptors.map((ld) => ld.label);

    resultsQuery.map((res) => {
      const bestMatch = faceMatcher.findBestMatch(res.descriptor);
      if (bestMatch._label === "person 1") {
        console.log('its a match')
        //arrayOfObj[i]["verified"] = true;
      } else {
        console.log('no match')
      }
    });
  // }
  return ;
}
run("https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/Thor/2.jpg", "https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/Hawkeye/1.jpg")
router.post("/face", async (req, res) => {
  const images = req.body;
  const REFERENCE_IMAGE = images[0]["img"];
  try {
    const newArray = await run(images, REFERENCE_IMAGE);
    if (newArray) {
      res.status(200).json(newArray);
    }
  } catch (err) {
    res.status(500).json(err);
  }
});

router.get("/", async (req, res) => {
  res.status(200).json('Welcome to face recognition.')
});
module.exports = router;
