from scipy.stats import pearsonr
import matplotlib.ticker as ticker
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
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
    plt.grid(True,linewidth=1.5)
    plt.tight_layout()
    plt.show()
    return r,p

def plot_txt( x,
              y,
              x_label,
              y_label,
              title):
    keys=list(x.keys())
    x_arr=[x[key_i] for key_i in x]
    y_arr=[y[key_i] for key_i in y]
    r, p = pearsonr(x_arr, y_arr)
    plt.scatter(x_arr, y_arr, color="steelblue", edgecolor="black", alpha=0.7)
    text=f"\nPearson correlation: r = {r:.4f}, p = {p:.3e}"
    plt.xlabel(x_label+text)
    plt.ylabel(y_label)
    plt.title(title)
    for key_i in keys:
        x_i=x[key_i]
        y_i=y[key_i]
        plt.annotate( key_i,
                      (x_i, y_i),
                      textcoords="offset points",
                      xytext=(5, 5),  
                      fontsize=8)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))         
    ax.yaxis.set_major_locator(ticker.MaxNLocator(10))         
    plt.grid(True,linewidth=1.5)
    plt.tight_layout()
    plt.show()

#def gen_plot( fun,
#              iter,
#              text):
#    x,y=[],[]
#    for id_i,data_i in iter:
#        x_i,y_i=fun(id_i,data_i)  
#        x.append(x_i)
#        y.append(y_i)
#    title,x_label,y_label=text
#    plot(x=x,
#         y=y,
#         x_label=x_label,
#         y_label=y_label,
#         title=title)

def gen_plot( fun,
              iter,
              text,
              labels=False):
    x,y={},{}
    for id_i,data_i in iter:
        x_i,y_i=fun(id_i,data_i)  
        x[id_i]=x_i
        y[id_i]=y_i
    title,x_label,y_label=text
    args={ "x":x,
           "y":y,
           "x_label":x_label,
           "y_label":y_label,
           "title":title}
    if(labels):
        plot_txt(**args)
    else:
        plot(**args)

def multi_plot( fun,
                iter,
                text):
    print(list(iter))
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
    for id_i, pair_i in iter:
        x_i, y_i = fun(id_i, pair_i)
        plt.plot(x_i, y_i, label=id_i)
        output.append((id_i, x_i, y_i))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return output