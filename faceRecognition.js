const { canvas } = require('./canvas.js');
const { faceDetectionOptions } = require('./faceDetection.js');
const { faceDetectionNet } = require('./faceDetection.js');
const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const faceapi = require('@vladmandic/face-api');

const router = express.Router();

async function run(images, REFERENCE_IMAGE) {
  console.log('inside run function', images);
  await faceDetectionNet.loadFromDisk('./models');
  await faceapi.nets.faceLandmark68Net.loadFromDisk('./models');
  await faceapi.nets.faceRecognitionNet.loadFromDisk('./models');
  console.log('modals loaded');
  let count = 0;
  let success = []
  for (let i = 0; i < images.length; i++) {
    if (images[i].includes('image')) {
      let QUERY_IMAGE = images[i];
      const referenceImage = await canvas.loadImage(REFERENCE_IMAGE);
      const queryImage = await canvas.loadImage(QUERY_IMAGE);
      console.log('loaded on canvas');
      // detect selfie face
      const detectFaceRef = await faceapi
        .detectAllFaces(referenceImage, faceDetectionOptions)
        .withFaceLandmarks()
        .withFaceDescriptors();
      console.log({ detectFaceRef });

      // face detected on reference image
      if (detectFaceRef.length !== 0) {
        console.log('detected selfie face');

        // detect avatar face
        const detectFaceProfileImage = await faceapi
          .detectAllFaces(queryImage, faceDetectionOptions)
          .withFaceLandmarks()
          .withFaceDescriptors();
        // must recognize a fave
        if (detectFaceProfileImage.length !== 0) {
          // indicates error, the lower more strict
          const faceMatcher = new faceapi.FaceMatcher(detectFaceRef, 0.55);
          
          detectFaceProfileImage.map((res) => {
            // compare selfie and avatar
            const bestMatch = faceMatcher.findBestMatch(res.descriptor);
            console.log('detected avatar face', bestMatch)
            if (bestMatch._label === 'person 1') {
              count += 1;
              success.push(images[i])
            }
          });
        }
      }
    }
  }
  console.log('count',count, success)
  let result = {count, success}
  return result;
}

router.post('/face', async (req, res) => {
  const obj = req.body;
  const REFERENCE_IMAGE = obj.objImages.selfieUrl;
  const images = obj.objImages.images;
  const profileId = obj.profileId;
  console.log('body info');
  try {
    const resultObj = await run(images, REFERENCE_IMAGE, profileId);
    if (res.status(200)) {
      res.status(200).json(resultObj);
    }
  } catch (err) {
    res.status(500).json(err);
  }
});
//dev.api.siingly.com/face-recognition/face
router.get('/', async (req, res) => {
  res.status(200).json('Welcome to face recognition.');
});
module.exports = router;
