const dropArea = document.querySelector('.drag-area');
const dragText = document.querySelector('.header');
const extractBtn = document.querySelector('.btn-extract');
const reuploadBtn = document.querySelector('.btn-reupload');

let button = dropArea.querySelector('.button');
let input = dropArea.querySelector('input');

let file;

button.onclick = () => {
	input.click();
};

function loading_on() {
	document.querySelector('.overlay').style.display = 'block';
}

function loading_off() {
	document.querySelector('.overlay').style.display = 'none';
}

// when browse
input.addEventListener('change', function () {
	file = this.files[0];
	dropArea.classList.add('active');
	displayFile();
});

// when file is inside drag area
dropArea.addEventListener('dragover', (event) => {
	event.preventDefault();
	dropArea.classList.add('active');
	dragText.textContent = 'Release to Upload';
	// console.log('File is inside the drag area');
});

// when file leave the drag area
dropArea.addEventListener('dragleave', () => {
	dropArea.classList.remove('active');
	// console.log('File left the drag area');
	dragText.textContent = 'Drag & Drop';
});

// when file is dropped
dropArea.addEventListener('drop', (event) => {
	event.preventDefault();
	// console.log('File is dropped in drag area');

	file = event.dataTransfer.files[0]; // grab single file even of user selects multiple files
	// console.log(file);
	displayFile();
	// window.alert(file && file['type'].split('/')[0] === 'image');
});

reuploadBtn.addEventListener('click', (e) => {
	input.click();
});

function displayFile() {
	let fileType = file.type;
	// console.log(fileType);

	let validExtensions = ['image/jpeg', 'image/jpg', 'image/png'];

	if (validExtensions.includes(fileType)) {
		// console.log('This is an image file');
		let fileReader = new FileReader();

		fileReader.onload = () => {
			let fileURL = fileReader.result;
			// console.log(fileURL);
			let imgTag = `<img src="${fileURL}" alt="">`;
			dropArea.innerHTML = imgTag;
		};
		fileReader.readAsDataURL(file);
		extractBtn.value = 'Extract';
		return true;
	} else {
		alert('This file is not supported!');
		dropArea.classList.remove('active');
		return false;
	}
}

document.querySelector('form').addEventListener('submit', function (e) {
	e.preventDefault();

	let method = document.querySelector('#method').value;

	if (file == null) {
		input.click();
	} else {
		loading_on();
		const formData = new FormData();
		formData.append('method', method);
		formData.append('file', file);
		const xhr = new XMLHttpRequest();
		//console.log("hello")
		xhr.onload = function () {
			setTimeout(() => {
				loading_off();
			}, 1000);

			reuploadBtn.classList.remove('d-none');
			console.log(xhr.responseText)
			const data = JSON.parse(xhr.responseText);

			//console.log(data);
			if (data.status == "OK") {
	
				console.log(data.dataset)
				document.querySelector(
					".info__dataset"
				).innerHTML = `<b>Dataset:</b> ${data.dataset}`;
				document.querySelector(
					".info__caption_clipcap"
				).innerHTML = `<b>Caption ClipCap:</b> ${data.caption_clipcap}`;
				document.querySelector(
					".info__caption_smallcap"
				).innerHTML = `<b>Caption SmallCap:</b> ${data.caption_smallcap}`;
				document.querySelector(
					".info__ret1_smallcap"
				).innerHTML = `<b>Retrieval 1:</b> ${data.ret1}`;
				document.querySelector(
					".info__ret2_smallcap"
				).innerHTML = `<b>Retrieval 2:</b> ${data.ret2}`;
				document.querySelector(
					".info__ret3_smallcap"
				).innerHTML = `<b>Retrieval 3:</b> ${data.ret3}`;
				document.querySelector(
					".info__ret4_smallcap"
				).innerHTML = `<b>Retrieval 4:</b> ${data.ret4}`;
			
				document.querySelector(
					".info__elapsed-time"
				).innerHTML = `<b>Elapsed time:</b> ${Math.round(data.elapsed_time * 100) / 100}s`;
	
				document.querySelector(
					".extracted__img"
				).innerHTML = `<img src="/static/src/uploads/${data.file}" />`; // To update avoid using image from cache
	
				document.querySelector(".btn-extract").value = 'Re-upload'
			} 
			else {
				window.alert(data.status);
			}
		};
		//console.log(data)
		

		let URL = '/extract';
		xhr.open('POST', URL);
		xhr.send(formData);
	}
});