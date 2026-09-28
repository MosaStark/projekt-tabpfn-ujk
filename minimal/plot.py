from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sn

def show_heatmap( matrix,
                  title,
                  out_path=None):
    sn.heatmap( matrix,
                cmap="YlGnBu",
                annot=False)#,
    plt.title(title)
    if(out_path):
        out_i=f"{out_path}/{title}"
        plt.tight_layout()
        plt.savefig(out_i,dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot(x,
         y,
         x_label,
         y_label,
         title):
    r, p = pearsonr(x, y)
    plt.scatter(x, y, color="steelblue", edgecolor="black", alpha=0.7)
    text=f"\nPearson correlation: r = {r:.4f}, p = {p:.3e}"
    plt.xlabel(x_label+text)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return r,p

def gen_plot( fun,
              iter,
              text):
    x,y=[],[]
    for id_i,data_i in iter:
        x_i,y_i=fun(id_i,data_i)  
        x.append(x_i)
        y.append(y_i)
    title,x_label,y_label=text
    plot(x=x,
         y=y,
         x_label=x_label,
         y_label=y_label,
         title=title)

def multi_plot( fun,
                iter,
                text):
    output=[]
    x_label,y_label=text
    for id_i,pair_i in iter:
        x_i,y_i=fun(id_i,pair_i)
        plot( x=x_i,
              y=y_i,
              x_label=x_label,
              y_label=y_label,
              title=id_i)
        output.append((id_i,x_i,y_i))
    return output

def set_font(size=12):
    plt.rcParams.update({'font.size': size})

def plot_ts(fun, iter, text, title=None):
    x_label, y_label = text
    output = []
    fig, ax = plt.subplots()
    for id_i, pair_i in iter:
        x_i, y_i = fun(id_i, pair_i)
        ax.plot(x_i, y_i, label=id_i)
        output.append((id_i, x_i, y_i))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.legend()
    plt.show()
    return output