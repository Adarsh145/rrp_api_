const express = require('express')
const {spawn} = require('child_process')

const bodyParser = require('body-parser')
const app = express()

app.use(express.json())

app.post('/predict', (req, res) => {
  const input_data = req.body
  console.log(input_data)
  const pythonProcess = spawn('python', ['app.py', JSON.stringify(input_data)])
  let result = ''

  pythonProcess.stdout.on('data', data => {
    result += data.toString()
  })
  pythonProcess.on('close', code => {
    console.log(`Python process exited with code ${code}`)
    const [prediction, response] = JSON.parse(result)

    console.log({
      prediction: prediction,
      response: response,
    })

    res.json({
      prediction: prediction,
      response: response,
    })
  })
})

app.use(bodyParser.json())
app.post('/predict2', (req, res) => {
  const input_data = req.body
  console.log(input_data)
  const pythonProcess = spawn('python', ['app2.py', JSON.stringify(input_data)])
  let result = ''

  pythonProcess.stdout.on('data', data => {
    result += data.toString()
  })

  pythonProcess.stderr.on('data', data => {
    console.error(`stderr: ${data}`)
  })

  pythonProcess.on('close', code => {
    console.log(`Python process exited with code ${code}`)
    try {
      if (result) {
        const parsedResult = JSON.parse(result)
        console.log(parsedResult)
        res.json(parsedResult)
      } else {
        res.status(500).json({error: 'Empty response from Python script'})
      }
    } catch (error) {
      console.error('Error parsing JSON from Python script:', error)
      res.status(500).json({error: 'Error parsing JSON from Python script'})
    }
  })
})

const port = 3000
app.listen(port, () => {
  console.log(`Server is running on port ${port}`)
})
