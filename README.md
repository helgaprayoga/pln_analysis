## Petunjuk Instalasi

Pastikan Python sudah terpasang. Buka PowerShell pada folder proyek, kemudian buat virtual environment:

```powershell
python -m venv .venv
```

Aktifkan virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install semua dependensi:

```powershell
python -m pip install -r requirements.txt
```

## Menjalankan Aplikasi

Pastikan file berikut berada dalam folder proyek:

- `app.py`
- `requirements.txt`

Setelah virtual environment aktif, jalankan:

```powershell
python -m streamlit run app.py
```

Buka alamat yang muncul di terminal, biasanya:

```text
http://localhost:8501
```