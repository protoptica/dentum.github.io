const contactForm = document.querySelector("#contact-form");

if (contactForm) {
  contactForm.addEventListener("submit", (event) => {
    event.preventDefault();

    const formData = new FormData(contactForm);
    const payload = Object.fromEntries(formData.entries());

    console.log("Lead form submission:", payload);
    window.alert("Спасибо! Форма сейчас работает как демо-заглушка. Данные выведены в console.log.");
    contactForm.reset();
  });
}
