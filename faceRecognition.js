const { canvas } = require("./canvas.js");
const { faceDetectionOptions } = require("./faceDetection.js");
const { faceDetectionNet } = require("./faceDetection.js");
const express = require("express");
const tf = require('@tensorflow/tfjs-node')
const faceapi = require('@vladmandic/face-api');

const router = express.Router();

async function run(avatar, REFERENCE_IMAGE, profileId) {
  console.log('inside run function', avatar)
  await faceDetectionNet.loadFromDisk("./models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
  await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");
  console.log("modals loaded")
  let match

    let QUERY_IMAGE = avatar;
    const referenceImage = await canvas.loadImage(REFERENCE_IMAGE); 
    const queryImage = await canvas.loadImage(QUERY_IMAGE);
console.log('loaded on canvas')
    // detect selfie face
    const detectFaceRef = await faceapi
      .detectAllFaces(referenceImage, faceDetectionOptions)
      .withFaceLandmarks()
      .withFaceDescriptors();
    console.log({detectFaceRef})

    if (detectFaceRef.length == 0) {
        // no face
        return match = false
      }
    console.log('detected selfie face')
    // detect avatar face
    const detectFaceAvatar = await faceapi
      .detectAllFaces(queryImage, faceDetectionOptions)
      .withFaceLandmarks()
      .withFaceDescriptors();

      if (detectFaceAvatar.length == 0) {
        // no face
       return match = false
      }
        // indicates error, the lower more strict
        const faceMatcher = new faceapi.FaceMatcher(detectFaceRef, 0.55);
        console.log('detected avatar face')
        detectFaceAvatar.map((res) => {
            // compare selfie and avatar
          const bestMatch = faceMatcher.findBestMatch(res.descriptor);
          if (bestMatch._label === "person 1") {
            return match = true
          } else {
            return match = false
          }
        });
        console.log('compared avatar and selfie')
  return match;
}

router.post("/face", async (req, res) => {
  const obj = req.body;
  const REFERENCE_IMAGE = obj.objImages.selfieUrl;
  const avatar = obj.objImages.avatar
  const profileId = obj.profileId
  console.log("body info")
  try {
    const newArray = await run(avatar, REFERENCE_IMAGE, profileId);
    console.log('response after run function', newArray)
    if (res.status(200)) {
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
