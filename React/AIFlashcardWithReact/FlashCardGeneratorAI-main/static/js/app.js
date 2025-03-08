document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById("uploadForm");
    const resultDiv = document.getElementById("result");
    const fileInput = document.getElementById("file");
    const fileText = document.querySelector(".custum-file-upload .text span");
    const fileUpload = document.querySelector(".custum-file-upload");
  
    // Add file input change handler
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        fileText.textContent = file.name;
        fileUpload.classList.add('file-selected');
      } else {
        fileText.textContent = 'Upload PDF or PPTX';
        fileUpload.classList.remove('file-selected');
      }
    });

    function showResults(results) {
      resultDiv.innerHTML = results;
      setTimeout(() => {
        resultDiv.classList.add('show');
      }, 100);
    }
  
    function showLoader() {
      resultDiv.innerHTML = `
        <p class="loading-text">Generating questions... Please wait.</p>
        <div class="loader"></div>
      `;
      resultDiv.classList.add('show');
    }
  
    uploadForm.addEventListener("submit", async function (e) {
      e.preventDefault();
  
      const formData = new FormData(this);
      showLoader();
  
      try {
        const response = await fetch("/generate", {
          method: "POST",
          body: formData,
        });
  
        const data = await response.json();
  
        if (data.error) {
          showResults(`<p class="error">${data.error}</p>`);
        } 
        else if (data.questions && data.questions.length > 0) {
          const questionsHtml = data.questions
            .map(
              (q, index) => `
              <div class="flashcard">
                <div class="flashcard-content">
                  <p><strong>Q${index + 1}:</strong> ${q.question}</p>
                  <p class="answer hidden"><strong>Answer:</strong> ${q.answer}</p>
                  <button class="show-answer-btn">Show Answer</button>
                </div>
              </div>
            `
            )
            .join("");
      
          showResults(questionsHtml);
      
          // Add event listeners to each "Show Answer" button
          const buttons = document.querySelectorAll(".show-answer-btn");
          buttons.forEach((button) => {
            button.addEventListener("click", () => {
              const flashcard = button.closest('.flashcard');
              const answer = flashcard.querySelector('.answer');
              if (answer.classList.contains("hidden")) {
                answer.classList.remove("hidden");
                button.textContent = "Hide Answer";
              } else {
                answer.classList.add("hidden");
                button.textContent = "Show Answer";
              }
            });
          });

          // Reset file input
          fileText.textContent = 'Upload PDF or PPTX';
          fileUpload.classList.remove('file-selected');
          uploadForm.reset();
        } else {
          showResults("<p>No questions generated.</p>");
        }
      } catch (error) {
        console.error('Error:', error);
        showResults("<p class='error'>An error occurred while generating questions. Please try again.</p>");
      }
    });
  });

