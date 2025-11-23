const express = require('express'); // 1.pacakage.json 2. npm install express
const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const app = express();
const port = process.env.PORT || 3000;
const pythonApiUrl = process.env.PYTHON_API_URL || 'http://localhost:5000/predict/'; // Python API 주소 수정


// QR 코드 업로드를 위한 임시 저장소
const mobileUploads = {};

// 내부 IP 주소 가져오기
function getLocalIp() {
  const nets = os.networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      // IPv4이면서 내부 주소(127.0.0.1)가 아닌 경우
      if (net.family === 'IPv4' && !net.internal) {
        return net.address;
      }
    }
  }
  return 'localhost'; // 찾지 못한 경우 localhost로 대체
}

// Multer 설정
const memoryStorage = multer.memoryStorage(); // 분석용 (메모리)
const diskStorage = multer.diskStorage({ // 저장용 (디스크)
  destination: function (req, file, cb) {
    cb(null, path.join(__dirname, 'images', 'plants'));
  },
  filename: function (req, file, cb) {
    // 파일 이름 중복을 피하기 위해 타임스탬프와 원본 파일명을 조합
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ storage: memoryStorage });
// 정적 파일 제공
app.use(express.static(path.join(__dirname)));

// plants API 엔드포인트
app.get('/api/plants', async (req, res) => {
  try {
    const plantsDir = path.join(__dirname, 'images', 'plants');
    const files = await fs.readdir(plantsDir);
    const plants = files
      .filter(file => /\.(jpg|jpeg|png|gif)$/i.test(file))
      .map(name => ({ name }));
    res.json(plants);
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: '식물 목록을 불러오는데 실패했습니다.' });
  }
});

// QR 코드 세션 ID 생성 API
app.get('/api/qr-session', (req, res) => {
  const serverIp = getLocalIp();
  const sessionId = Date.now() + '-' + Math.random().toString(36).substring(2, 10);
  mobileUploads[sessionId] = {
    status: 'pending',
    file: null,
    createdAt: Date.now()
  };
  // 세션 ID와 함께 서버 IP, 포트 정보 전송
  res.json({ sessionId, serverIp, port });
});

// 업로드된 이미지를 임시 저장하는 API
app.post('/api/upload-from-mobile', upload.single('image'), (req, res) => {
  const { sessionId } = req.body;
  if (!req.file || !sessionId || !mobileUploads[sessionId]) {
    return res.status(400).json({ error: '잘못된 요청입니다.' });
  }

  // 파일을 메모리에 buffer 형태로 저장
  mobileUploads[sessionId] = {
    status: 'completed',
    file: {
      buffer: req.file.buffer.toString('base64'), // 버퍼를 Base64 문자열로 변환
      originalname: req.file.originalname,
      mimetype: req.file.mimetype,
    }
  };
  res.json({ message: '사진이 성공적으로 업로드되었습니다.' });
});

// 데스크톱에서 업로드된 이미지를 확인하는 API (폴링)
app.get('/api/check-upload/:sessionId', (req, res) => {
  const { sessionId } = req.params;
  const SESSION_TIMEOUT = 15 * 60 * 1000; // 15분
  const uploadData = mobileUploads[sessionId];

  if (!uploadData) {
    return res.json({ status: 'expired' });
  }
  if (uploadData.status === 'pending' && Date.now() - uploadData.createdAt > SESSION_TIMEOUT) {
    delete mobileUploads[sessionId]; // 만료된 세션 정리
    return res.json({ status: 'expired' });
  }
  res.json(uploadData);
});


// 이미지 분석 API 엔드포인트
app.post('/api/analyze', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: '이미지 파일이 필요합니다.' });
  }

  try {
    const formData = new FormData();
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const response = await axios.post(pythonApiUrl, formData, { // URL 수정됨
      headers: formData.getHeaders(),
    });

    res.json(response.data);
  } catch (error) {
    // console.error('Python API 호출 중 오류 발생:', error.message);
    // res.status(500).json({ error: '분석 서버와 통신하는 데 실패.' });
    // Python 서버(axios)에서 받은 오류를 더 상세하게 로깅하고 클라이언트에 전달
    if (error.response) {
      // Python 서버가 오류 응답을 반환한 경우 (4xx, 5xx)
      console.error('Python API 오류 응답:', error.response.data);
      res.status(error.response.status).json({ error: error.response.data.detail || '분석 서버에서 오류가 발생했습니다.' });
    } else {
      // 네트워크 오류 또는 Python 서버가 실행되지 않은 경우
      console.error('Python API 호출 중 오류 발생:', error.message);
      res.status(500).json({ error: '분석 서버와 통신할 수 없습니다. 서버가 실행 중인지 확인해주세요.' });
    }
  }
});

// 식물 이미지 저장 API 엔드포인트
const uploadToDisk = multer({ storage: diskStorage });
app.post('/api/save-plant', uploadToDisk.single('image'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: '이미지 파일이 필요합니다.' });
  }
  try {
    res.status(201).json({ 
      message: '식물이 목록에 성공적으로 추가되었습니다!',
      filePath: `/images/plants/${req.file.filename}`
    });
  } catch (error) {
    console.error('Error saving plant image:', error);
    res.status(500).json({ error: '이미지를 저장하는 데 실패했습니다.' });
  }
});

app.listen(port, () => {
  console.log(`서버가 http://localhost:${port} 에서 실행 중입니다`);
});
