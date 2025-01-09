import{i as x,f}from"./index-SRHFOrZt.js";function _(n){const t=n-1;return t*t*t+1}function S(n,{delay:t=0,duration:o=400,easing:s=x}={}){const c=+getComputedStyle(n).opacity;return{delay:t,duration:o,easing:s,css:a=>`opacity: ${a*c}`}}function U(n,{delay:t=0,duration:o=400,easing:s=_,x:c=0,y:a=0,opacity:u=0}={}){const r=getComputedStyle(n),e=+r.opacity,y=r.transform==="none"?"":r.transform,p=e*(1-u),[l,m]=f(c),[$,d]=f(a);return{delay:t,duration:o,easing:s,css:(i,g)=>`
			transform: ${y} translate(${(1-i)*l}${m}, ${(1-i)*$}${d});
			opacity: ${e-p*g}`}}export{U as a,_ as c,S as f};
//# sourceMappingURL=index-BYNgTMF2.js.map
