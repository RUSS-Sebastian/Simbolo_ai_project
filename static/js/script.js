// const navToggle = document.querySelector('.nav-toggle');
// const navList = document.querySelector('.nav-list');

// const toggleMenu = function(){
//     navList.classList.toggle('active');
//     this.classList.toggle('active');
// }

// navToggle.addEventListener('click',toggleMenu);

// //header slideIn animation
// const headerEle = document.querySelector('.header');
// window.addEventListener('scroll',function(){
//     if(this.scrollY > 50){
//         headerEle.classList.add('active');
//     }else{
//         headerEle.classList.remove('active');
//     }
// })

// //copyright date
// const date = document.querySelector('.date');
// date.innerText = new Date().getFullYear();


let cards = document.querySelectorAll(".card");

      let stackArea = document.querySelector(".stack-area");

      function rotateCards() {
        let angle = 0;
        cards.forEach((card, index) => {
          if (card.classList.contains("away")) {
            card.style.transform = `translateY(-120vh) rotate(-48deg)`;
          } else {
            card.style.transform = ` rotate(${angle}deg)`;
            angle = angle - 10;
            card.style.zIndex = cards.length - index;
          }
        });
      }

      rotateCards();

      window.addEventListener("scroll", () => {
        let distance = window.innerHeight * 0.5;

        let topVal = stackArea.getBoundingClientRect().top;

        let index = -1 * (topVal / distance + 1);

        index = Math.floor(index);

        for (i = 0; i < cards.length; i++) {
          if (i <= index) {
            cards[i].classList.add("away");
          } else {
            cards[i].classList.remove("away");
          }
        }
        rotateCards();
      });

