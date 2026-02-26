import matplotlib.pyplot as plt
import seaborn as sns

def plot_data_distribution(df, n_cols=10):
    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
    valid_sensors = []
    
    n_rows = (len(sensor_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(3*n_cols, 3*n_rows))

    for i, sensor in enumerate(sensor_cols):
        if df[sensor].nunique() > 1:
            valid_sensors.append(sensor)
        plt.subplot(n_rows, n_cols, i+1)
        sns.histplot(df[sensor], kde=True)
        plt.title(sensor, fontsize=9)

    plt.tight_layout()
    return valid_sensors

def plot_engine_sensors(df, unit_number, n_cols=3):
    motor_df = df[df['unit_number'] == unit_number]
    
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    
    n_sensors = len(sensor_cols)
    n_rows = -(-n_sensors // n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(sensor_cols):
        axes[i].plot(motor_df['time_in_cycles'], motor_df[col], linewidth=1)
        axes[i].set_title(col, fontsize=8)
        axes[i].set_xlabel('Ciclos', fontsize=7)
        axes[i].set_ylabel('Valor', fontsize=7)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f"Sensores do motor {unit_number}", fontsize=14)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

def plot_sensors_vs_rul(df, engines, sensors):
    fig, axes = plt.subplots(len(engines), len(sensors), figsize=(15, 8))

    for i, engine in enumerate(engines):
        engine_df = df[df['unit_number'] == engine]

        for j, sensor in enumerate(sensors):
            axes[i, j].scatter(
                engine_df['RUL'],
                engine_df[sensor],
                alpha=0.5,
                s=10
            )
            axes[i, j].invert_xaxis()
            axes[i, j].set_title(f"Engine {engine} – {sensor}", fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_sensor_profiles_across_engines(df, engines, sensors):
    fig, axes = plt.subplots(len(sensors), len(engines), figsize=(16, 12))
    
    for i, sensor in enumerate(sensors):
        for j, engine in enumerate(engines):
            engine_data = df[df['unit_number'] == engine]
            axes[i,j].plot(engine_data['time_in_cycles'], engine_data[sensor], linewidth=1.5)
            axes[i,j].set_title(f'Motor {engine} - {sensor}', fontsize=10)
            axes[i,j].set_xlabel('Ciclos', fontsize=8)
            axes[i,j].set_ylabel(sensor, fontsize=8)
            axes[i,j].grid(True, alpha=0.3)
    
    plt.suptitle("Mesmos sensores em motores com diferentes tempos de vida", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    