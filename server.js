const express = require('express');

const face = require('./faceRecognition.js');

const server = express();

server.use(express.json());
server.use('/', face);

module.exports = server;