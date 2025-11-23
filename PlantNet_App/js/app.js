// DOM elements
const plantGrid = document.querySelector('.plant-grid');
const uploadPrompt = document.getElementById('uploadPrompt');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const selectBtn = document.getElementById('selectBtn'); // '사진 선택' 버튼
const fileInput = document.getElementById('fileInput');
const closeBtn = document.getElementById('closeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const changeBtn = document.getElementById('changeBtn');
const resultsEmpty = document.getElementById('resultsEmpty');
const resultsContent = document.getElementById('resultsContent');
const resetBtn = document.getElementById('resetBtn');
const modalOverlay = document.getElementById('modalOverlay');
const modalClose = document.getElementById('modalClose');
const modalNo = document.getElementById('modalNo');
const modalYes = document.getElementById('modalYes');
const qrModalOverlay = document.getElementById('qrModalOverlay');
const qrModalClose = document.getElementById('qrModalClose');
const uploadFromComputerBtn = document.getElementById('uploadFromComputerBtn');
const qrcodeContainer = document.getElementById('qrcode');

let selectedFile = null;
// 'user', 'mobile', or 'default'
// 'user': 사용자가 직접 업로드한 파일
// 'mobile': QR코드로 업로드한 파일
// 'default': 기본 식물 목록에서 선택한 파일
let imageSourceType = null; 

// 사진 가져와다 plant-grid에 사진 보여줌
async function loadPlants() {
  try {
    const response = await fetch('/api/plants');
    const plants = await response.json();
    
    // Clear existing grid
    plantGrid.innerHTML = '';
    
    // Create plant items
    plants.forEach(plant => {
      const plantId = plant.name.toLowerCase().replace(/\.[^/.]+$/, ""); // Remove file extension
      
      const plantItem = document.createElement('div');
      plantItem.className = 'plant-item';
      plantItem.dataset.plant = plantId;
      
      const img = document.createElement('img');
      img.src = `/images/plants/${plant.name}`;
      img.alt = plantId;
      
      plantItem.appendChild(img);
      plantGrid.appendChild(plantItem);
      
      // Add click event listener
      plantItem.addEventListener('click', () => {
        handlePlantSelection(plantId, img.src);
      });
    });
  } catch (error) {
    console.error('식물 목록을 불러오는데 실패했습니다:', error);
  }
}

async function handlePlantSelection(plantId, imageSrc) {
  try {
    // 이미지 URL로부터 Blob 객체 생성
    const response = await fetch(imageSrc);
    const blob = await response.blob();
    // Blob으로부터 File 객체 생성
    selectedFile = new File([blob], `${plantId}.jpg`, { type: blob.type });
    imageSourceType = 'default'; // 기본 식물 목록에서 선택
    displayImagePreview(imageSrc);
  } catch (error) {
    console.error('이미지를 파일 객체로 변환하는데 실패했습니다:', error);
    alert('이미지 선택에 실패했습니다. 다시 시도해주세요.');
  }
}

function displayImagePreview(imageSrc) {
  // Update UI
  document.querySelectorAll('.plant-item').forEach(p => p.classList.remove('selected'));
  
  if (selectedFile && selectedFile.name) {
    const plantId = selectedFile.name.replace(/\.[^/.]+$/, "");
    const selectedItem = document.querySelector(`[data-plant="${plantId}"]`);
    if (selectedItem) {
      selectedItem.classList.add('selected');
    }
  }
  
  // Show image preview
  uploadPrompt.style.display = 'none';
  imagePreview.classList.add('active');
  previewImage.src = imageSrc;
  
  // Hide results
  resultsEmpty.style.display = 'flex';
  resultsContent.classList.remove('active');
}

// '사진 선택' 버튼 클릭 -> QR 모달 열기
selectBtn.addEventListener('click', () => {
  openQrModal();
});

fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    selectedFile = file;
    imageSourceType = 'user'; // 사용자가 직접 업로드
    const reader = new FileReader();
    reader.onload = (e) => {
      displayImagePreview(e.target.result);
    };
    reader.readAsDataURL(file);
  }
});

// Close button
closeBtn.addEventListener('click', () => {
  resetToInitial();
});

// Change button
changeBtn.addEventListener('click', () => {
  // 사용자가 올린 사진을 분석한 후에만 저장 여부 물어보기
  if ((imageSourceType === 'user' || imageSourceType === 'mobile') && resultsContent.classList.contains('active')) {
    modalOverlay.classList.add('active');
  } else {
    openQrModal();
  }
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    alert('분석할 사진을 선택해주세요.');
    return;
  }

  setLoadingState(true);

  try {
    // 1. AI 모델로 이미지 분석 (Python 서버에서 위키백과 정보까지 모두 가져옴)
    const analysisResult = await analyzeImage(selectedFile);

    // 2. 결과를 화면에 바로 표시
    displayResults(analysisResult);

  } catch (error) {
    console.error('분석 과정에서 오류 발생:', error);
    alert(error.message || '분석에 실패했습니다. 잠시 후 다시 시도해주세요.');
    // 에러 발생 시 결과창을 초기 상태로 되돌릴 수 있습니다.
    resultsEmpty.style.display = 'flex';
    resultsContent.classList.remove('active');
  } finally {
    setLoadingState(false);
  }
});

async function analyzeImage(file) {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch('/api/analyze', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(`이미지 분석 실패: ${errorData.error || response.statusText}`);
  }
  return response.json();
}

// Reset button - show modal
resetBtn.addEventListener('click', () => {  
  // 사용자가 업로드한 이미지이고, 분석이 완료된 경우에만 저장 모달을 띄웁니다.
  if ((imageSourceType === 'user' || imageSourceType === 'mobile') && resultsContent.classList.contains('active')) {
    modalOverlay.classList.add('active');
  } else {
    // 그 외의 경우(기본 식물 선택 등)에는 바로 초기화합니다.
    resetToInitial();
  }
});


// Modal close
modalClose.addEventListener('click', () => {
  modalOverlay.classList.remove('active');
  // '다른 사진 선택' 시 모달을 닫으면 QR 모달이 열리도록
  if ((imageSourceType === 'user' || imageSourceType === 'mobile') && !resultsContent.classList.contains('active')) {
    openQrModal();
  }
});

modalNo.addEventListener('click', () => {
  modalOverlay.classList.remove('active');
  // '다른 사진 선택' 시 모달 '아니오'를 누르면 QR 모달이 열리도록
  if (imageSourceType === 'user' && !resultsContent.classList.contains('active')) {
    resetToInitial();
  } else {
    openQrModal();
  }
});

modalYes.addEventListener('click', async () => {
  modalOverlay.classList.remove('active');
  if (selectedFile) {
    await saveImage(selectedFile);
  }
  // '다른 사진 선택'을 눌렀을 때 '예'를 누르면 QR 모달이 열리도록
  if ((imageSourceType === 'user' || imageSourceType === 'mobile') && !resultsContent.classList.contains('active')) {
    resetToInitial();
  } else {
    openQrModal();
  }
});

// Close modal on overlay click
modalOverlay.addEventListener('click', (e) => {
  if (e.target === modalOverlay) {
    modalOverlay.classList.remove('active');
  }
});

async function saveImage(file) {
  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch('/api/save-plant', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || '사진 저장에 실패했습니다.');
    }

    const result = await response.json();
    alert(result.message || '식물이 목록에 추가되었습니다!');
    
    // 목록을 다시 로드하여 추가된 식물을 보여줍니다.
    await loadPlants();

  } catch (error) {
    console.error('Error saving plant:', error);
    alert(error.message);
  }
}


function setLoadingState(isLoading) {
  analyzeBtn.disabled = isLoading;
  if (isLoading) {
    analyzeBtn.innerHTML = `
      <svg class="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24" style="width: 18px; height: 18px; animation: spin 1s linear infinite;">
        <path d="M12 4V2A10 10 0 002 12h2a8 8 0 018-8z" fill="currentColor"></path>
      </svg>
      분석 중...
    `;
  } else {
    analyzeBtn.innerHTML = `
      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
      식물 분석하기
    `;
  }
}

function displayResults(result) {
  document.getElementById('scientificName').textContent = result.scientificName;
  document.getElementById('commonName').textContent = result.commonName;
  document.getElementById('description').textContent = result.description;

  // Show results
  resultsEmpty.style.display = 'none';
  resultsContent.classList.add('active');
}

function resetToInitial() {
  document.querySelectorAll('.plant-item').forEach(p => p.classList.remove('selected'));
  uploadPrompt.style.display = 'block';
  imagePreview.classList.remove('active');
  selectedFile = null;
  imageSourceType = null;
  fileInput.value = ''; // 파일 입력 초기화
  resultsEmpty.style.display = 'flex';
  resultsContent.classList.remove('active');
}

async function openQrModal() {
  // 이전 QR 코드가 있다면 삭제
  qrcodeContainer.innerHTML = '';
  qrModalOverlay.classList.add('active');

  try {
    // 1. 서버에 세션 ID 요청
    const response = await fetch('/api/qr-session');
    const { sessionId, serverIp, port } = await response.json();

    // 2. 세션 ID를 포함한 URL로 QR 코드 생성
    const uploadUrl = `http://${serverIp}:${port}/mobile-upload.html?session=${sessionId}`;
    new QRCode(qrcodeContainer, {
      text: uploadUrl,
      width: 200,
      height: 200,
    });

    // 3. 폴링 시작: 주기적으로 서버에 업로드 완료 여부 확인
    const pollInterval = setInterval(async () => {
      // 모달이 닫혔으면 폴링 중지
      if (!qrModalOverlay.classList.contains('active')) {
        clearInterval(pollInterval);
        return;
      }

      const checkResponse = await fetch(`/api/check-upload/${sessionId}`);
      const result = await checkResponse.json();

      if (result.status === 'completed') {
        clearInterval(pollInterval);
        qrModalOverlay.classList.remove('active');

        // 서버에서 받은 파일 정보로 File 객체 생성
        // Base64 문자열을 Blob으로 변환
        const byteCharacters = atob(result.file.buffer);
        const byteNumbers = Array.from(byteCharacters, char => char.charCodeAt(0));
        const blob = new Blob([new Uint8Array(byteNumbers)], { type: result.file.mimetype });

        selectedFile = new File([blob], result.file.originalname, { type: result.file.mimetype });
        imageSourceType = 'mobile'; // QR 코드로 업로드
        displayImagePreview(URL.createObjectURL(blob));
      }
    }, 2000); // 2초마다 확인
  } catch (error) {
    console.error('QR 세션 생성 실패:', error);
    qrcodeContainer.innerHTML = '<p>QR 코드 생성에 실패했습니다. 다시 시도해주세요.</p>';
  }
}

qrModalClose.addEventListener('click', () => {
  qrModalOverlay.classList.remove('active');
});

qrModalOverlay.addEventListener('click', (e) => {
  if (e.target === qrModalOverlay) {
    qrModalOverlay.classList.remove('active');
  }
});

uploadFromComputerBtn.addEventListener('click', () => {
  qrModalOverlay.classList.remove('active');
  fileInput.click();
});


// Initialize
loadPlants();