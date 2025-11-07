import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, SymLogNorm
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import scipy.io
import mat73
from scipy.interpolate import griddata
import cartopy.crs as ccrs

#We load the data and save the different variables
data=np.load("datos_finales_indicios_7param.npz")



params_proposed_history=data["params_proposed_history"]
params_accepted_history=data["params_accepted_history"]
acceptance_history=data["acceptance_history"]
rss_accepted_history=data["rss_accepted_history"]
AR_parameter=data["AR_parameter"]
new_parameter=data["new_parameter"]
sigma_new=data["sigma_new"]

#We define the reference values and the variables names

mu=np.array([161,19,0.035,0.002,1.75])
variables = ['Kmax', 'Kmin', 'λmax', 'λmin', 'Q10']

data=mat73.loadmat('landShelfOceanMask_ContMargMaskKocsisScotese.mat')
landShelfOcean_Lat=data['landShelfOcean_Lat']
landShelfOcean_Lon=data['landShelfOcean_Lon']
landShelfOceanMask=data['landShelfOceanMask']
landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)

LSOmask=np.transpose(landShelfOceanMask[:,:,landShelfOceanMask.shape[2]-1])

datos_proof=np.load("datos_proof_2.npz")
proof=datos_proof["proof"]



#Calculate the mean of the accepted parameters after burn-in
A=[]
for j in range(5):
    A.append(np.mean(params_accepted_history[200:,j]))

# Create figure and axes with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(14, 8))

fig.subplots_adjust(hspace=0.6, wspace=0.3)
# Aplanamos el arreglo de ejes para indexarlo fácilmente
axs = axs.flatten()

ax_blank = fig.add_axes([0.05, 0.9, 0.5, 0.1])  # [left, bottom, width, height]
ax_blank.axis('off')

# Graficamos solo en los primeros 5 subplots
for j in range(5):
    ax = axs[j]
    for i in range(6):

        ax.plot(np.arange(2003), params_accepted_history[:, j, i], alpha=0.5, label=f'chain {i}')

    ax.set_title(variables[j], fontsize=18)
    ax.axhline(y=mu[j], color='red', linestyle='--', linewidth=1, label='Reference value')
    ax.axhline(y=A[j], color='blue', linestyle='--', linewidth=1, label='Mean value')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Parameter value', fontsize=12)
    #ax.axvspan(0, 200, color='black', alpha=0.2, label='burn-in')
    ax.set_xlim(0,2020)


handles, labels = axs[0].get_legend_handles_labels()

fig.legend(handles, labels, loc='lower right', fontsize=15, bbox_to_anchor=(0.9, 0))  # Ajusta la posición según sea necesario
#plt.legend(ncol=2)



# Ocultar el subplot vacío (el sexto)
fig.delaxes(axs[5])


plt.savefig("Chains.pdf", bbox_inches='tight')
plt.show()


#We create the second figure

#We define the colormap

cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue",  "red"])

# Datos de muestra

global_min = np.inf
global_max = -np.inf


fig, axs = plt.subplots(5, 5, figsize=(18, 14), sharex=False, sharey=False)

for i in range(5):

    for j in range(5):

        if i==j:

            ax = axs[i,j]
            line = ax.axvline(x=mu[j], color='red', linestyle='--', linewidth=2, label='Fixed value')
            line2 = ax.axvline(x=np.mean(params_accepted_history[200:,j]), color='darkblue', linestyle='--', linewidth=2, label='Mean value')
            data = params_accepted_history[200:, j].ravel().astype(float)
            # Calcular el histograma
            counts, bins = np.histogram(data, bins=30)

            # Normalizar al máximo = 1
            counts = counts / counts.max()

            # Graficar manualmente
            ax.bar((bins[:-1] + bins[1:]) / 2,   # posiciones de las barras (centros)
                   counts,                       # alturas normalizadas
                   width=np.diff(bins),          # anchura de cada bin
                   align="center",
                   edgecolor='black',
                   color='lightgray')

            ax.set_ylabel(f'{variables[i]} \n Probability', fontsize=15)

            
            #ax.set_title(variables[j])
            if i==0:
                ax.set_title(variables[j], fontsize=15)

        elif j>i:
            
            ax = axs[i, j]
            x = params_accepted_history[200:, j, :].ravel()
            y = params_accepted_history[200:, i, :].ravel()

            weights = np.ones_like(x) / len(x)
            hist = ax.hist2d(x, y, weights=weights, cmap=cmap)



            fixed_value_x=mu[j]
            fixed_value_y=mu[i]

            H, x_edges, y_edges = np.histogram2d(x, y, bins=20)



            global_min = min(global_min, H.min())
            global_max = max(global_max, H.max())

            H=hist[0]
            im=hist[3]

            H_rel = H / H.sum()  
            #im.set_array(H_rel.ravel())  

            x_edges= hist[1]
            y_edges= hist[2]

            x_bin = np.digitize(fixed_value_x, x_edges) - 1
            y_bin = np.digitize(fixed_value_y, y_edges) - 1

            x_start = x_edges[x_bin]
            y_start = y_edges[y_bin]
            bin_width = x_edges[1] - x_edges[0]
            bin_height = y_edges[1] - y_edges[0]

            # Dibujar un rectángulo rojo
            rect = Rectangle((x_start, y_start), bin_width, bin_height,
                             edgecolor='yellow', facecolor='none', linewidth=3, label='Reference value')
            ax.add_patch(rect)

           # ax.set_xlabel(variables[j])
           # ax.set_ylabel(variables[i])
            if i==0:
                ax.set_title(variables[j], fontsize=15)
         
           # ax.set_title(f'{variables[j]} vs {variables[i]}')

        else:
            fig.delaxes(axs[i,j])
            

# Título general


cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.02)
cbar.set_label("Frequency of each combination", fontsize=12)

handles1, labels1 = axs[0,0].get_legend_handles_labels()
handles2, labels2 = axs[0,1].get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2

fig.legend(handles, labels, loc='lower left', title="Reference", fontsize=15, bbox_to_anchor=(0.35, 0.2))

#fig.suptitle('Histograms 2D with 2 sigma in the error', fontsize=18)

#plt.savefig("histogram2D.pdf")
plt.show()

#Figure 3: Diversity maps

data=np.load("datos_mean.npz")
D_mean=data["D_mean"]

colors_bYr = np.loadtxt("mycolormap_bYr.dat")
bYr = ListedColormap(colors_bYr, name="mi_cmap_bYr")
data_color_br = scipy.io.loadmat("mycolormap_br.mat")
colors_br=data_color_br["mycolormap_br"]
br = ListedColormap(colors_br, name="mi_cmap_br")

lats = np.linspace(-90, 90, 360)   # 360 filas -> paso de 0.5°
lons = np.linspace(-180, 180, 720) # 720 columnas -> paso de 0.5°

lon_grid, lat_grid = np.meshgrid(lons, lats)

[X,Y]=np.meshgrid(landShelfOcean_Lon,landShelfOcean_Lat)#

x = X.flatten()
y = Y.flatten()
z_mean = D_mean.flatten()
z_proof = proof.flatten()

# Eliminar NaN (para no meterlos en la interpolación)
mask_proof = ~np.isnan(z_proof) & ~np.isinf(z_proof)
mask_mean = ~np.isnan(z_mean) & ~np.isinf(z_mean)
x_valid = x[mask_proof]
y_valid = y[mask_proof]
z_mean_v = z_mean[mask_mean]
z_proof_v = z_proof[mask_proof]

D_mean_interp = griddata(
    (x_valid, y_valid), 
    z_mean_v, 
    (X, Y), 
    method='linear'   # puedes probar 'nearest' o 'cubic'
)
D_interp = griddata(
    (x_valid, y_valid), 
    z_proof_v, 
    (X, Y), 
    method='linear'   # puedes probar 'nearest' o 'cubic'
)


D_mean_interp[LSOmask != 1] = np.nan
D_interp[LSOmask != 1] = np.nan

K=D_interp-D_mean_interp
print(K.shape)

Q95=np.percentile(K[~np.isnan(K)],95)
Q05=np.percentile(K[~np.isnan(K)],5)
print(Q95, Q05)
K[K>Q95]=Q95
K[K<Q05]=Q05






x = np.linspace(100, 10, 100)
y = x  # gráfico simple y = x

fig = plt.figure(figsize=(18, 16))

ax_blank = fig.add_axes([0.3, 0.9, 0.5, 0.05])  # [left, bottom, width, height]
ax_blank.axis('off')

gs = GridSpec(4, 13, figure=fig, height_ratios=[2,2.5,2,1.5], width_ratios=[1]*13, hspace=0, wspace=0.8)  # 4 filas, 4 columnas de referencia

#fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9,
#                    wspace=1, hspace=0.4)
#fig.subplots_adjust(top=2, bottom=1)
# Fila 1: 2 figuras

vmin = min(np.nanmin(D_interp), np.nanmin(D_mean_interp))
vmax = max(np.nanmax(D_interp), np.nanmax(D_mean_interp))

print(vmin)


ax1 = fig.add_subplot(gs[0, 1:6], projection=ccrs.Mollweide())
scatter1=ax1.pcolormesh(lon_grid, lat_grid,
                     D_interp, cmap=bYr, 
                     norm=LogNorm(vmin=vmin, vmax=vmax),
                       transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.set_global()
#cbar1=plt.colorbar(scatter1, orientation='vertical')
#cbar1.set_label( label='Diversity', fontsize=12)

gl1=ax1.gridlines(draw_labels=False, linestyle='--', color='black', alpha=0.5,  ylocs=[-45, 0, 45])


ax1.set_title('Observed data', fontsize=18)



ax2 = fig.add_subplot(gs[0, 6:11], projection=ccrs.Mollweide())
scatter2=ax2.pcolormesh(lon_grid, lat_grid,
                      D_mean_interp, cmap=bYr, 
                      norm=SymLogNorm(linthresh=0.001, base=2), 
                       transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.set_global()
#cbar2=plt.colorbar(scatter2, orientation='vertical', label='Diversity')
#cbar2.set_label('Diversity', fontsize=12)

gl2=ax2.gridlines(draw_labels=False, linestyle='--', color='black', alpha=0.5,  ylocs=[-45, 0, 45])

ax2.set_title('Estimated data from the ensemble solution', fontsize=18)

cbar_ax = fig.add_axes([0.8, 0.7, 0.01, 0.14])
cbar = fig.colorbar(scatter2, ax=[ax1, ax2], cax=cbar_ax)

cbar.set_label('Diversity (number of genera)')





# Fila 2: 1 figura
ax3 = fig.add_subplot(gs[1, 3:10], projection=ccrs.Mollweide())
scatter=ax3.pcolormesh(lon_grid, lat_grid,
                     K, cmap=br,
                       transform=ccrs.PlateCarree(), vmin=-2, vmax=2.5)
ax3.coastlines()
ax3.set_global()
cbar3=plt.colorbar(scatter, orientation='vertical',  fraction=0.08)

cbar3.set_label(label='Diversity difference (number of genera)', fontsize=12)



# --- Rectángulo definido por las coordenadas exactas ---
lon_min_car, lon_max_car = -68, -55
lat_min_car, lat_max_car = 8.2, 16.2


ax3.legend(loc='lower left', bbox_to_anchor=(-0.24, 0))

gl3=ax3.gridlines(draw_labels=False, linestyle='--', color='black', alpha=0.5,  ylocs=[-45, 0, 45])




ax3.set_title('Data fit: Observed minus \n estimated data', fontsize=18)





panel_labels = ['(a)', '(b)', '(c)']
axes = [ax1, ax2, ax3]
for ax, label in zip(axes, panel_labels):
    # Coordenadas relativas al axes: x=0 para izquierda, y=1 para arriba

    if label!='(c)':
        ax.text(-0.05, 1.3, label, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='top', ha='left')
    else:
        ax.text(-0.05, 1.2, label, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='top', ha='left')
    


# Ajustar layout y guardar solo la figura
#plt.subplots_adjust(top=1.2)
plt.savefig("figura_combinada.pdf", dpi=300)  # PDF de alta resolución
#plt.savefig("Diversity_maps.svg")
plt.show()




    

