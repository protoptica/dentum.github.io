const contactForm = document.querySelector("#contact-form");
const phoneInput = document.querySelector('input[name="phone"]');

if (phoneInput) {
  const formatPhone = (rawValue) => {
    const digits = rawValue.replace(/\D/g, "");
    const normalized = digits.startsWith("7") ? digits.slice(1) : digits.startsWith("8") ? digits.slice(1) : digits;
    const trimmed = normalized.slice(0, 10);

    let formatted = "+7";

    if (trimmed.length > 0) {
      formatted += " (" + trimmed.slice(0, 3);
    }

    if (trimmed.length >= 4) {
      formatted += ") " + trimmed.slice(3, 6);
    }

    if (trimmed.length >= 7) {
      formatted += "-" + trimmed.slice(6, 8);
    }

    if (trimmed.length >= 9) {
      formatted += "-" + trimmed.slice(8, 10);
    }

    return formatted;
  };

  phoneInput.addEventListener("focus", () => {
    if (!phoneInput.value) {
      phoneInput.value = "+7";
    }
  });

  phoneInput.addEventListener("input", () => {
    phoneInput.value = formatPhone(phoneInput.value);
  });

  phoneInput.addEventListener("blur", () => {
    if (phoneInput.value === "+7") {
      phoneInput.value = "";
    }
  });
}

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
