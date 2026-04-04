import os

paths = [
    'src/cfm/models/cylindrical_unet.py',
    'tests/test_data/test_dataset.py',
    'tests/test_models/test_unet.py',
    'src/cfm/data/dataset.py',
    'src/cfm/data/transforms.py'
]

translations = {
    'Tłumaczy skalar czasu t na wysokowymiarowy wektor cech (embedding).': 'Translates a time scalar t into a high-dimensional feature vector (embedding).',
    'Dzięki temu sieć "rozumie" upływ czasu podczas generowania.': 'This allows the network to "understand" the progression of time during generation.',
    'Blok resztkowy (Residual Block), który przyjmuje mapę przestrzenną obrazu': 'Residual Block that takes an image spatial map',
    'oraz wstrzykuje w nią informacje o obecnym kroku czasowym.': 'and injects information about the current time step.',
    'Połączenie resztkowe wyrównujące kanały, jeśli zaszła zmiana': 'Residual connection aligning channels if there was a change',
    'Główna gałąź splotowa': 'Main convolutional branch',
    'Wstrzyknięcie czasu (broadcasting na wymiary H i W)': 'Time injection (broadcasting over H and W dimensions)',
    'U-Net dopasowany do topologii walca (R^+ x S^1).': 'U-Net tailored for the cylindrical topology (R^+ x S^1).',
    'Wejście: [B, 3, H, W] (Amplituda, p_x, p_y)': 'Input: [B, 3, H, W] (Amplitude, p_x, p_y)',
    'Wyjście: [B, 2, H, W] (Prędkość Amplitudy, Prędkość Kątowa Fazy)': 'Output: [B, 2, H, W] (Amplitude Velocity, Phase Angular Velocity)',
    'Inicjalna konwolucja (z 3 kanałów manifoldowych)': 'Initial convolution (from 3 manifold channels)',
    'ENKODER (Schodzenie w dół, wyciąganie globalnego kontekstu)': 'ENCODER (Downsampling, extracting global context)',
    'BOTTLENECK (Najgłębsza reprezentacja)': 'BOTTLENECK (Deepest representation)',
    'DEKODER (Wchodzenie w górę + Skip Connections)': 'DECODER (Upsampling + Skip Connections)',
    'Kanały: 4*base (z upsample) + 4*base (ze skip connection) = 8*base': 'Channels: 4*base (from upsample) + 4*base (from skip connection) = 8*base',
    'Kanały: 2*base (z upsample) + 2*base (ze skip connection) = 4*base': 'Channels: 2*base (from upsample) + 2*base (from skip connection) = 4*base',
    'Konwolucja finalna zwracająca nasze 2 wektory prędkości': 'Final convolution outputting our 2 velocity vectors',
    'Tensor z rzutowania topologicznego': 'Tensor from topological projection',
    'Skalary czasu dla każdej próbki w batchu': 'Time scalars for each sample in the batch',
    'Enkoder': 'Encoder',
    'Dekoder ze Skip Connections': 'Decoder with Skip Connections',
    'Tworzy fałszywą strukturę katalogów i plik h5 do testów.': 'Creates a fake directory structure and h5 file for testing.',
    'Testuje inicjalizację i ładowanie bez transformacji.': 'Tests initialization and loading without transformations.',
    'Testuje ładowanie z pełnym przetworzeniem na walec.': 'Tests loading with full processing to the cylinder topology.',
    'Sprawdza, czy U-Net zwraca poprawny kształt wyjścia [B, 2, H, W].': 'Checks if U-Net returns the correct output shape [B, 2, H, W].',
    'Upewnia się, że sieć faktycznie zmienia zachowanie pod wpływem czasu.': 'Ensures the network actually changes behavior under the influence of time.',
    'Sprawdza tolerancję na różne rozdzielczości wejściowe (np. bez hardkodowania wymiarów).': 'Checks tolerance for different input resolutions (e.g., without hardcoding dimensions).'
}

for path in paths:
    if not os.path.exists(path): continue
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for pl, en in translations.items():
        content = content.replace(pl, en)
        
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
