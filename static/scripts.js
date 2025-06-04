document.addEventListener('DOMContentLoaded', () => {
    const stockForm = document.getElementById('stock-form');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const profileForm = document.getElementById('profile-form');
    const loading = document.getElementById('loading');

    if (stockForm) {
        stockForm.addEventListener('submit', (e) => {
            const tickerInput = document.getElementById('ticker');
            const selectInput = document.getElementById('selected_stock');
            const selectedStock = selectInput.value;
            const ticker = tickerInput.value.trim().toUpperCase();

            // Dropdown’dan seçim yapıldıysa, text input’u güncelle
            if (selectedStock) {
                tickerInput.value = selectedStock;
            }


            if (!selectedStock && !ticker) {
                e.preventDefault();
                alert('Lütfen bir hisse kodu seçin veya girin.');
                return;
            }


            const validTickers = {{ bist100_list | tojson }};
            if (ticker && !validTickers.includes(ticker)) {
                e.preventDefault();
                alert('Geçersiz hisse kodu. Lütfen BIST100 listesinden bir hisse seçin.');
                return;
            }

            if (loading) {
                loading.classList.remove('hidden');
            }
        });


        document.getElementById('selected_stock').addEventListener('change', () => {
            stockForm.submit();
        });
    }

    if (loginForm) {
        loginForm.addEventListener('submit', (e) => {
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            if (!email || !password) {
                e.preventDefault();
                alert('Lütfen tüm alanları doldurun.');
            }
        });
    }

    if (registerForm) {
        registerForm.addEventListener('submit', (e) => {
            const username = document.getElementById('username').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            if (!username || !email || !password) {
                e.preventDefault();
                alert('Lütfen tüm alanları doldurun.');
            }
        });
    }

    if (profileForm) {
        profileForm.addEventListener('submit', (e) => {
            const username = document.getElementById('username').value;
            const email = document.getElementById('email').value;
            if (!username || !email) {
                e.preventDefault();
                alert('Kullanıcı adı ve e-posta zorunludur.');
            }
        });
    }
});