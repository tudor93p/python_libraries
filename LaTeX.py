import os
import numpy as np
from sympy import latex as latex0

def latex(A):
  return latex0(A,mat_delim='(')

#begin_subeq = '\n\\begin{subequations}\n' + '\\begin{gather}\n'
begin_subeq = '\n\\begin{subequations}\n' + '\\begin{align}\n'
#end_subeq = '\n\\end{gather}\n' + '\\end{subequations}\n'
end_subeq = '\n\\end{align}\n' + '\\end{subequations}\n'

begin_eq = '\n\\begin{equation}\n'
end_eq = '\n\\end{equation}\n'


#docclass_aps = '\\documentclass[reprint,superscriptaddress,nofootinbib,amsmath,amssymb,aps,pra]{revtex4-1}'
docclass_aps = '\\documentclass[superscriptaddress,nofootinbib,amsmath,amssymb,aps,pra]{revtex4-2}'

docclass_notes = lambda ncol: '\\documentclass[11p,a4paper'+ncol+']{article}'


my_comm_pack = '\\usepackage{my_commands}\n\\usepackage{my_packages}'


margins_font = '\\usepackage[top=1.8cm,right=1.8cm,left=1.8cm,bottom=1.8cm]{geometry}\n\\usepackage{palatino}\n\\linespread{1.3}'


begindoc = '\\begin{document}'

enddoc = '\\end{document}'



class LaTeX_Document:

  def __init__(self,filelatex,style='notes',twocolumn=False,packs=None):

    self.filename = filelatex

    os.system('rm '+self.filename)
  
    self.Content = []

 
 
    if style == 'aps':
      self.Content += [docclass_aps]
  
    elif style == 'notes':
      if twocolumn == True:
        docclass = docclass_notes(',twocolumn')
      else:
        docclass = docclass_notes('')

      self.Content += [docclass,my_comm_pack,margins_font]
 
    elif style == 'ETH':

      self.Content.append(docclass_notes(''))
      self.Content.append('\\usepackage{ethuebung}')
      self.Content.append(my_comm_pack)

 
    else:
      print('Choose style: aps or notes or ETH.')
      exit()
 
    if type(packs) != type(None): 
      self.Content.append('\\usepackage{'+packs+'}')
    
    self.Content.append(begindoc)
    
    
  def Add_Text(self,ABC):

    if type(ABC) == str:
      self.Content.append(ABC)
    else:
      print('Please input string if you want to write text.')


  def list_to_eq(self,ABC):
    if type(ABC) == list:
      return '\n'.join([latex(abc).replace('$','') for abc in ABC])
    else:
      return latex(ABC).replace('$','')

  def Add_Eq(self,ABC):
 
    eq = self.list_to_eq(ABC) 
 
    self.Add_Text(' '.join([begin_eq,eq,end_eq]))

  def Add_SubEqs(self,ABC):

    eq = '\\\\\n'.join([self.list_to_eq(Eq) for Eq in ABC])

    self.Add_Text(' '.join([begin_subeq,eq,end_subeq]))

  def Add_Figure(self,filename,wid=0.5,caption='',label=None,align='H'):

    p1= '\\begin{figure}['+align+']\n\centering'

    p2 ='\includegraphics[width='+str(wid)+'\linewidth]{'+filename+'}'

    p3 ='\caption{'+caption+'}'

    if label != None: p3 = p3 +'\label{'+label+'}'

    p4 = '\end{figure}'

    self.Add_Text('\n'.join([p1,p2,p3,p4]))
    
  def HRule(self,space=2):

    if np.size(space)==1: space = np.repeat(space,2)

    text = '\hrule'.join(['\\vspace{'+str(s)+ 'em}' for s in space])

    self.Add_Text(text)

  def NewPage(self):

    self.Add_Text('\\newpage')

  def ClPage(self):

    self.Add_Text('\clearpage')

  def Add_Section(self,text,level=0):


    text_ = '\\'+['','sub','subsub'][level]+'section{'+text+'}'

  
    self.Add_Text(text_)



  def Write(self): 
  
    self.Content.append(enddoc)
  
    Doc = '\n\n'.join(self.Content)
  
    with open(self.filename,'w') as f:
      f.write('%s' % Doc)

    print(f"Wrote content to file \'{self.filename}\'. Compiling PDF...")

    os.system('pdflatex '+self.filename+' > junk.txt && rm junk.txt')

    print("Compilation successful!")



"""
    Parameters
    ==========
    fold_frac_powers : boolean, optional
        Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional powers.
    fold_func_brackets : boolean, optional
        Fold function brackets where applicable.
    fold_short_frac : boolean, optional
        Emit ``p / q`` instead of ``\frac{p}{q}`` when the denominator is
        simple enough (at most two terms and no powers). The default value is
        ``True`` for inline mode, ``False`` otherwise.
    inv_trig_style : string, optional
        How inverse trig functions should be displayed. Can be one of
        ``abbreviated``, ``full``, or ``power``. Defaults to ``abbreviated``.
    itex : boolean, optional
        Specifies if itex-specific syntax is used, including emitting
        ``$$...$$``.
    ln_notation : boolean, optional
        If set to ``True``, ``\ln`` is used instead of default ``\log``.
    long_frac_ratio : float or None, optional
        The allowed ratio of the width of the numerator to the width of the
        denominator before the printer breaks off long fractions. If ``None``
        (the default value), long fractions are not broken up.
    mat_delim : string, optional
        The delimiter to wrap around matrices. Can be one of ``[``, ``(``, or
        the empty string. Defaults to ``[``.
    mat_str : string, optional
        Which matrix environment string to emit. ``smallmatrix``, ``matrix``,
        ``array``, etc. Defaults to ``smallmatrix`` for inline mode, ``matrix``
        for matrices of no more than 10 columns, and ``array`` otherwise.
    mode: string, optional
        Specifies how the generated code will be delimited. ``mode`` can be one
        of ``plain``, ``inline``, ``equation`` or ``equation*``.  If ``mode``
        is set to ``plain``, then the resulting code will not be delimited at
        all (this is the default). If ``mode`` is set to ``inline`` then inline
        LaTeX ``$...$`` will be used. If ``mode`` is set to ``equation`` or
        ``equation*``, the resulting code will be enclosed in the ``equation``
        or ``equation*`` environment (remember to import ``amsmath`` for
        ``equation*``), unless the ``itex`` option is set. In the latter case,
        the ``$$...$$`` syntax is used.
    mul_symbol : string or None, optional
        The symbol to use for multiplication. Can be one of ``None``, ``ldot``,
        ``dot``, or ``times``.
    order: string, optional
        Any of the supported monomial orderings (currently ``lex``, ``grlex``,
        or ``grevlex``), ``old``, and ``none``. This parameter does nothing for
        Mul objects. Setting order to ``old`` uses the compatibility ordering
        for Add defined in Printer. For very large expressions, set the
        ``order`` keyword to ``none`` if speed is a concern.
    symbol_names : dictionary of strings mapped to symbols, optional
        Dictionary of symbols and the custom strings they should be emitted as.
    root_notation : boolean, optional
        If set to ``False``, exponents of the form 1/n are printed in fractonal
        form. Default is ``True``, to print exponent in root form.
    mat_symbol_style : string, optional
        Can be either ``plain`` (default) or ``bold``. If set to ``bold``,
        a MatrixSymbol A will be printed as ``\mathbf{A}``, otherwise as ``A``.
    imaginary_unit : string, optional
        String to use for the imaginary unit. Defined options are "i" (default)
        and "j". Adding "r" or "t" in front gives ``\mathrm`` or ``\text``, so
        "ri" leads to ``\mathrm{i}`` which gives `\mathrm{i}`.
    gothic_re_im : boolean, optional
        If set to ``True``, `\Re` and `\Im` is used for ``re`` and ``im``, respectively.
        The default is ``False`` leading to `\operatorname{re}` and `\operatorname{im}`.
"""
