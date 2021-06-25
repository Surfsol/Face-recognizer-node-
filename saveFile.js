const fs = require('fs');
const path = require( 'path')
const request = require('request')

const baseDir = path.resolve(`${path.resolve()}/out`)

function saveFile(fileName, buf) {
  if (!fs.existsSync(baseDir)) {
    fs.mkdirSync(baseDir)
  }

  fs.writeFileSync(path.resolve(baseDir, fileName), buf)
}




const download = (url) => {
  const selfiePath = path.resolve(`${path.resolve()}/selfie`)
  if (!fs.existsSync(selfiePath)) {
    fs.mkdirSync(selfiePath)
  }
  request.head(url, (err, res, body) => {
    request(url)
      .pipe(fs.createWriteStream(selfiePath))
  })
}

// download(url, selfiePath, () => {
//   console.log('✅ Done!')
// })
module.exports={saveFile, download}
