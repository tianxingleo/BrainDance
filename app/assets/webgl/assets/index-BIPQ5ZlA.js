(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))n(s);new MutationObserver(s=>{for(const r of s)if(r.type==="childList")for(const o of r.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&n(o)}).observe(document,{childList:!0,subtree:!0});function t(s){const r={};return s.integrity&&(r.integrity=s.integrity),s.referrerPolicy&&(r.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?r.credentials="include":s.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function n(s){if(s.ep)return;s.ep=!0;const r=t(s);fetch(s.href,r)}})();function Ef(i){const e=Object.create(null);for(const t of i.split(","))e[t]=1;return t=>t in e}const At={},Xr=[],Di=()=>{},mm=()=>!1,kl=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&(i.charCodeAt(2)>122||i.charCodeAt(2)<97),wf=i=>i.startsWith("onUpdate:"),ln=Object.assign,Rf=(i,e)=>{const t=i.indexOf(e);t>-1&&i.splice(t,1)},hx=Object.prototype.hasOwnProperty,ct=(i,e)=>hx.call(i,e),Ke=Array.isArray,qr=i=>_a(i)==="[object Map]",gm=i=>_a(i)==="[object Set]",qd=i=>_a(i)==="[object Date]",$e=i=>typeof i=="function",Xt=i=>typeof i=="string",Fi=i=>typeof i=="symbol",mt=i=>i!==null&&typeof i=="object",xm=i=>(mt(i)||$e(i))&&$e(i.then)&&$e(i.catch),_m=Object.prototype.toString,_a=i=>_m.call(i),px=i=>_a(i).slice(8,-1),Am=i=>_a(i)==="[object Object]",If=i=>Xt(i)&&i!=="NaN"&&i[0]!=="-"&&""+parseInt(i,10)===i,Oo=Ef(",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"),Hl=i=>{const e=Object.create(null);return(t=>e[t]||(e[t]=i(t)))},mx=/-\w/g,Ls=Hl(i=>i.replace(mx,e=>e.slice(1).toUpperCase())),gx=/\B([A-Z])/g,ks=Hl(i=>i.replace(gx,"-$1").toLowerCase()),Sm=Hl(i=>i.charAt(0).toUpperCase()+i.slice(1)),oc=Hl(i=>i?`on${Sm(i)}`:""),Rs=(i,e)=>!Object.is(i,e),al=(i,...e)=>{for(let t=0;t<i.length;t++)i[t](...e)},vm=(i,e,t,n=!1)=>{Object.defineProperty(i,e,{configurable:!0,enumerable:!1,writable:n,value:t})},Df=i=>{const e=parseFloat(i);return isNaN(e)?i:e};let Qd;const Vl=()=>Qd||(Qd=typeof globalThis<"u"?globalThis:typeof self<"u"?self:typeof window<"u"?window:typeof global<"u"?global:{});function Pf(i){if(Ke(i)){const e={};for(let t=0;t<i.length;t++){const n=i[t],s=Xt(n)?Sx(n):Pf(n);if(s)for(const r in s)e[r]=s[r]}return e}else if(Xt(i)||mt(i))return i}const xx=/;(?![^(]*\))/g,_x=/:([^]+)/,Ax=/\/\*[^]*?\*\//g;function Sx(i){const e={};return i.replace(Ax,"").split(xx).forEach(t=>{if(t){const n=t.split(_x);n.length>1&&(e[n[0].trim()]=n[1].trim())}}),e}function Qr(i){let e="";if(Xt(i))e=i;else if(Ke(i))for(let t=0;t<i.length;t++){const n=Qr(i[t]);n&&(e+=n+" ")}else if(mt(i))for(const t in i)i[t]&&(e+=t+" ");return e.trim()}const vx="itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly",yx=Ef(vx);function ym(i){return!!i||i===""}function bx(i,e){if(i.length!==e.length)return!1;let t=!0;for(let n=0;t&&n<i.length;n++)t=Ff(i[n],e[n]);return t}function Ff(i,e){if(i===e)return!0;let t=qd(i),n=qd(e);if(t||n)return t&&n?i.getTime()===e.getTime():!1;if(t=Fi(i),n=Fi(e),t||n)return i===e;if(t=Ke(i),n=Ke(e),t||n)return t&&n?bx(i,e):!1;if(t=mt(i),n=mt(e),t||n){if(!t||!n)return!1;const s=Object.keys(i).length,r=Object.keys(e).length;if(s!==r)return!1;for(const o in i){const a=i.hasOwnProperty(o),l=e.hasOwnProperty(o);if(a&&!l||!a&&l||!Ff(i[o],e[o]))return!1}}return String(i)===String(e)}const bm=i=>!!(i&&i.__v_isRef===!0),hi=i=>Xt(i)?i:i==null?"":Ke(i)||mt(i)&&(i.toString===_m||!$e(i.toString))?bm(i)?hi(i.value):JSON.stringify(i,Mm,2):String(i),Mm=(i,e)=>bm(e)?Mm(i,e.value):qr(e)?{[`Map(${e.size})`]:[...e.entries()].reduce((t,[n,s],r)=>(t[ac(n,r)+" =>"]=s,t),{})}:gm(e)?{[`Set(${e.size})`]:[...e.values()].map(t=>ac(t))}:Fi(e)?ac(e):mt(e)&&!Ke(e)&&!Am(e)?String(e):e,ac=(i,e="")=>{var t;return Fi(i)?`Symbol(${(t=i.description)!=null?t:e})`:i};let Cn;class Mx{constructor(e=!1){this.detached=e,this._active=!0,this._on=0,this.effects=[],this.cleanups=[],this._isPaused=!1,this.__v_skip=!0,this.parent=Cn,!e&&Cn&&(this.index=(Cn.scopes||(Cn.scopes=[])).push(this)-1)}get active(){return this._active}pause(){if(this._active){this._isPaused=!0;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].pause();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].pause()}}resume(){if(this._active&&this._isPaused){this._isPaused=!1;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].resume();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].resume()}}run(e){if(this._active){const t=Cn;try{return Cn=this,e()}finally{Cn=t}}}on(){++this._on===1&&(this.prevScope=Cn,Cn=this)}off(){this._on>0&&--this._on===0&&(Cn=this.prevScope,this.prevScope=void 0)}stop(e){if(this._active){this._active=!1;let t,n;for(t=0,n=this.effects.length;t<n;t++)this.effects[t].stop();for(this.effects.length=0,t=0,n=this.cleanups.length;t<n;t++)this.cleanups[t]();if(this.cleanups.length=0,this.scopes){for(t=0,n=this.scopes.length;t<n;t++)this.scopes[t].stop(!0);this.scopes.length=0}if(!this.detached&&this.parent&&!e){const s=this.parent.scopes.pop();s&&s!==this&&(this.parent.scopes[this.index]=s,s.index=this.index)}this.parent=void 0}}}function Cx(){return Cn}let St;const lc=new WeakSet;class Cm{constructor(e){this.fn=e,this.deps=void 0,this.depsTail=void 0,this.flags=5,this.next=void 0,this.cleanup=void 0,this.scheduler=void 0,Cn&&Cn.active&&Cn.effects.push(this)}pause(){this.flags|=64}resume(){this.flags&64&&(this.flags&=-65,lc.has(this)&&(lc.delete(this),this.trigger()))}notify(){this.flags&2&&!(this.flags&32)||this.flags&8||Em(this)}run(){if(!(this.flags&1))return this.fn();this.flags|=2,Yd(this),wm(this);const e=St,t=gi;St=this,gi=!0;try{return this.fn()}finally{Rm(this),St=e,gi=t,this.flags&=-3}}stop(){if(this.flags&1){for(let e=this.deps;e;e=e.nextDep)Uf(e);this.deps=this.depsTail=void 0,Yd(this),this.onStop&&this.onStop(),this.flags&=-2}}trigger(){this.flags&64?lc.add(this):this.scheduler?this.scheduler():this.runIfDirty()}runIfDirty(){fu(this)&&this.run()}get dirty(){return fu(this)}}let Tm=0,No,zo;function Em(i,e=!1){if(i.flags|=8,e){i.next=zo,zo=i;return}i.next=No,No=i}function Lf(){Tm++}function Bf(){if(--Tm>0)return;if(zo){let e=zo;for(zo=void 0;e;){const t=e.next;e.next=void 0,e.flags&=-9,e=t}}let i;for(;No;){let e=No;for(No=void 0;e;){const t=e.next;if(e.next=void 0,e.flags&=-9,e.flags&1)try{e.trigger()}catch(n){i||(i=n)}e=t}}if(i)throw i}function wm(i){for(let e=i.deps;e;e=e.nextDep)e.version=-1,e.prevActiveLink=e.dep.activeLink,e.dep.activeLink=e}function Rm(i){let e,t=i.depsTail,n=t;for(;n;){const s=n.prevDep;n.version===-1?(n===t&&(t=s),Uf(n),Tx(n)):e=n,n.dep.activeLink=n.prevActiveLink,n.prevActiveLink=void 0,n=s}i.deps=e,i.depsTail=t}function fu(i){for(let e=i.deps;e;e=e.nextDep)if(e.dep.version!==e.version||e.dep.computed&&(Im(e.dep.computed)||e.dep.version!==e.version))return!0;return!!i._dirty}function Im(i){if(i.flags&4&&!(i.flags&16)||(i.flags&=-17,i.globalVersion===Zo)||(i.globalVersion=Zo,!i.isSSR&&i.flags&128&&(!i.deps&&!i._dirty||!fu(i))))return;i.flags|=2;const e=i.dep,t=St,n=gi;St=i,gi=!0;try{wm(i);const s=i.fn(i._value);(e.version===0||Rs(s,i._value))&&(i.flags|=128,i._value=s,e.version++)}catch(s){throw e.version++,s}finally{St=t,gi=n,Rm(i),i.flags&=-3}}function Uf(i,e=!1){const{dep:t,prevSub:n,nextSub:s}=i;if(n&&(n.nextSub=s,i.prevSub=void 0),s&&(s.prevSub=n,i.nextSub=void 0),t.subs===i&&(t.subs=n,!n&&t.computed)){t.computed.flags&=-5;for(let r=t.computed.deps;r;r=r.nextDep)Uf(r,!0)}!e&&!--t.sc&&t.map&&t.map.delete(t.key)}function Tx(i){const{prevDep:e,nextDep:t}=i;e&&(e.nextDep=t,i.prevDep=void 0),t&&(t.prevDep=e,i.nextDep=void 0)}let gi=!0;const Dm=[];function rs(){Dm.push(gi),gi=!1}function os(){const i=Dm.pop();gi=i===void 0?!0:i}function Yd(i){const{cleanup:e}=i;if(i.cleanup=void 0,e){const t=St;St=void 0;try{e()}finally{St=t}}}let Zo=0;class Ex{constructor(e,t){this.sub=e,this.dep=t,this.version=t.version,this.nextDep=this.prevDep=this.nextSub=this.prevSub=this.prevActiveLink=void 0}}class Of{constructor(e){this.computed=e,this.version=0,this.activeLink=void 0,this.subs=void 0,this.map=void 0,this.key=void 0,this.sc=0,this.__v_skip=!0}track(e){if(!St||!gi||St===this.computed)return;let t=this.activeLink;if(t===void 0||t.sub!==St)t=this.activeLink=new Ex(St,this),St.deps?(t.prevDep=St.depsTail,St.depsTail.nextDep=t,St.depsTail=t):St.deps=St.depsTail=t,Pm(t);else if(t.version===-1&&(t.version=this.version,t.nextDep)){const n=t.nextDep;n.prevDep=t.prevDep,t.prevDep&&(t.prevDep.nextDep=n),t.prevDep=St.depsTail,t.nextDep=void 0,St.depsTail.nextDep=t,St.depsTail=t,St.deps===t&&(St.deps=n)}return t}trigger(e){this.version++,Zo++,this.notify(e)}notify(e){Lf();try{for(let t=this.subs;t;t=t.prevSub)t.sub.notify()&&t.sub.dep.notify()}finally{Bf()}}}function Pm(i){if(i.dep.sc++,i.sub.flags&4){const e=i.dep.computed;if(e&&!i.dep.subs){e.flags|=20;for(let n=e.deps;n;n=n.nextDep)Pm(n)}const t=i.dep.subs;t!==i&&(i.prevSub=t,t&&(t.nextSub=i)),i.dep.subs=i}}const du=new WeakMap,ar=Symbol(""),hu=Symbol(""),Jo=Symbol("");function nn(i,e,t){if(gi&&St){let n=du.get(i);n||du.set(i,n=new Map);let s=n.get(t);s||(n.set(t,s=new Of),s.map=n,s.key=t),s.track()}}function Ji(i,e,t,n,s,r){const o=du.get(i);if(!o){Zo++;return}const a=l=>{l&&l.trigger()};if(Lf(),e==="clear")o.forEach(a);else{const l=Ke(i),c=l&&If(t);if(l&&t==="length"){const u=Number(n);o.forEach((f,d)=>{(d==="length"||d===Jo||!Fi(d)&&d>=u)&&a(f)})}else switch((t!==void 0||o.has(void 0))&&a(o.get(t)),c&&a(o.get(Jo)),e){case"add":l?c&&a(o.get("length")):(a(o.get(ar)),qr(i)&&a(o.get(hu)));break;case"delete":l||(a(o.get(ar)),qr(i)&&a(o.get(hu)));break;case"set":qr(i)&&a(o.get(ar));break}}Bf()}function _r(i){const e=lt(i);return e===i?e:(nn(e,"iterate",Jo),ai(i)?e:e.map(xi))}function Gl(i){return nn(i=lt(i),"iterate",Jo),i}function As(i,e){return as(i)?so(lr(i)?xi(e):e):xi(e)}const wx={__proto__:null,[Symbol.iterator](){return cc(this,Symbol.iterator,i=>As(this,i))},concat(...i){return _r(this).concat(...i.map(e=>Ke(e)?_r(e):e))},entries(){return cc(this,"entries",i=>(i[1]=As(this,i[1]),i))},every(i,e){return Ni(this,"every",i,e,void 0,arguments)},filter(i,e){return Ni(this,"filter",i,e,t=>t.map(n=>As(this,n)),arguments)},find(i,e){return Ni(this,"find",i,e,t=>As(this,t),arguments)},findIndex(i,e){return Ni(this,"findIndex",i,e,void 0,arguments)},findLast(i,e){return Ni(this,"findLast",i,e,t=>As(this,t),arguments)},findLastIndex(i,e){return Ni(this,"findLastIndex",i,e,void 0,arguments)},forEach(i,e){return Ni(this,"forEach",i,e,void 0,arguments)},includes(...i){return uc(this,"includes",i)},indexOf(...i){return uc(this,"indexOf",i)},join(i){return _r(this).join(i)},lastIndexOf(...i){return uc(this,"lastIndexOf",i)},map(i,e){return Ni(this,"map",i,e,void 0,arguments)},pop(){return Mo(this,"pop")},push(...i){return Mo(this,"push",i)},reduce(i,...e){return Kd(this,"reduce",i,e)},reduceRight(i,...e){return Kd(this,"reduceRight",i,e)},shift(){return Mo(this,"shift")},some(i,e){return Ni(this,"some",i,e,void 0,arguments)},splice(...i){return Mo(this,"splice",i)},toReversed(){return _r(this).toReversed()},toSorted(i){return _r(this).toSorted(i)},toSpliced(...i){return _r(this).toSpliced(...i)},unshift(...i){return Mo(this,"unshift",i)},values(){return cc(this,"values",i=>As(this,i))}};function cc(i,e,t){const n=Gl(i),s=n[e]();return n!==i&&!ai(i)&&(s._next=s.next,s.next=()=>{const r=s._next();return r.done||(r.value=t(r.value)),r}),s}const Rx=Array.prototype;function Ni(i,e,t,n,s,r){const o=Gl(i),a=o!==i&&!ai(i),l=o[e];if(l!==Rx[e]){const f=l.apply(i,r);return a?xi(f):f}let c=t;o!==i&&(a?c=function(f,d){return t.call(this,As(i,f),d,i)}:t.length>2&&(c=function(f,d){return t.call(this,f,d,i)}));const u=l.call(o,c,n);return a&&s?s(u):u}function Kd(i,e,t,n){const s=Gl(i);let r=t;return s!==i&&(ai(i)?t.length>3&&(r=function(o,a,l){return t.call(this,o,a,l,i)}):r=function(o,a,l){return t.call(this,o,As(i,a),l,i)}),s[e](r,...n)}function uc(i,e,t){const n=lt(i);nn(n,"iterate",Jo);const s=n[e](...t);return(s===-1||s===!1)&&Hf(t[0])?(t[0]=lt(t[0]),n[e](...t)):s}function Mo(i,e,t=[]){rs(),Lf();const n=lt(i)[e].apply(i,t);return Bf(),os(),n}const Ix=Ef("__proto__,__v_isRef,__isVue"),Fm=new Set(Object.getOwnPropertyNames(Symbol).filter(i=>i!=="arguments"&&i!=="caller").map(i=>Symbol[i]).filter(Fi));function Dx(i){Fi(i)||(i=String(i));const e=lt(this);return nn(e,"has",i),e.hasOwnProperty(i)}class Lm{constructor(e=!1,t=!1){this._isReadonly=e,this._isShallow=t}get(e,t,n){if(t==="__v_skip")return e.__v_skip;const s=this._isReadonly,r=this._isShallow;if(t==="__v_isReactive")return!s;if(t==="__v_isReadonly")return s;if(t==="__v_isShallow")return r;if(t==="__v_raw")return n===(s?r?Hx:Nm:r?Om:Um).get(e)||Object.getPrototypeOf(e)===Object.getPrototypeOf(n)?e:void 0;const o=Ke(e);if(!s){let l;if(o&&(l=wx[t]))return l;if(t==="hasOwnProperty")return Dx}const a=Reflect.get(e,t,rn(e)?e:n);if((Fi(t)?Fm.has(t):Ix(t))||(s||nn(e,"get",t),r))return a;if(rn(a)){const l=o&&If(t)?a:a.value;return s&&mt(l)?mu(l):l}return mt(a)?s?mu(a):zf(a):a}}class Bm extends Lm{constructor(e=!1){super(!1,e)}set(e,t,n,s){let r=e[t];const o=Ke(e)&&If(t);if(!this._isShallow){const c=as(r);if(!ai(n)&&!as(n)&&(r=lt(r),n=lt(n)),!o&&rn(r)&&!rn(n))return c||(r.value=n),!0}const a=o?Number(t)<e.length:ct(e,t),l=Reflect.set(e,t,n,rn(e)?e:s);return e===lt(s)&&(a?Rs(n,r)&&Ji(e,"set",t,n):Ji(e,"add",t,n)),l}deleteProperty(e,t){const n=ct(e,t);e[t];const s=Reflect.deleteProperty(e,t);return s&&n&&Ji(e,"delete",t,void 0),s}has(e,t){const n=Reflect.has(e,t);return(!Fi(t)||!Fm.has(t))&&nn(e,"has",t),n}ownKeys(e){return nn(e,"iterate",Ke(e)?"length":ar),Reflect.ownKeys(e)}}class Px extends Lm{constructor(e=!1){super(!0,e)}set(e,t){return!0}deleteProperty(e,t){return!0}}const Fx=new Bm,Lx=new Px,Bx=new Bm(!0);const pu=i=>i,Ea=i=>Reflect.getPrototypeOf(i);function Ux(i,e,t){return function(...n){const s=this.__v_raw,r=lt(s),o=qr(r),a=i==="entries"||i===Symbol.iterator&&o,l=i==="keys"&&o,c=s[i](...n),u=t?pu:e?so:xi;return!e&&nn(r,"iterate",l?hu:ar),ln(Object.create(c),{next(){const{value:f,done:d}=c.next();return d?{value:f,done:d}:{value:a?[u(f[0]),u(f[1])]:u(f),done:d}}})}}function wa(i){return function(...e){return i==="delete"?!1:i==="clear"?void 0:this}}function Ox(i,e){const t={get(s){const r=this.__v_raw,o=lt(r),a=lt(s);i||(Rs(s,a)&&nn(o,"get",s),nn(o,"get",a));const{has:l}=Ea(o),c=e?pu:i?so:xi;if(l.call(o,s))return c(r.get(s));if(l.call(o,a))return c(r.get(a));r!==o&&r.get(s)},get size(){const s=this.__v_raw;return!i&&nn(lt(s),"iterate",ar),s.size},has(s){const r=this.__v_raw,o=lt(r),a=lt(s);return i||(Rs(s,a)&&nn(o,"has",s),nn(o,"has",a)),s===a?r.has(s):r.has(s)||r.has(a)},forEach(s,r){const o=this,a=o.__v_raw,l=lt(a),c=e?pu:i?so:xi;return!i&&nn(l,"iterate",ar),a.forEach((u,f)=>s.call(r,c(u),c(f),o))}};return ln(t,i?{add:wa("add"),set:wa("set"),delete:wa("delete"),clear:wa("clear")}:{add(s){!e&&!ai(s)&&!as(s)&&(s=lt(s));const r=lt(this);return Ea(r).has.call(r,s)||(r.add(s),Ji(r,"add",s,s)),this},set(s,r){!e&&!ai(r)&&!as(r)&&(r=lt(r));const o=lt(this),{has:a,get:l}=Ea(o);let c=a.call(o,s);c||(s=lt(s),c=a.call(o,s));const u=l.call(o,s);return o.set(s,r),c?Rs(r,u)&&Ji(o,"set",s,r):Ji(o,"add",s,r),this},delete(s){const r=lt(this),{has:o,get:a}=Ea(r);let l=o.call(r,s);l||(s=lt(s),l=o.call(r,s)),a&&a.call(r,s);const c=r.delete(s);return l&&Ji(r,"delete",s,void 0),c},clear(){const s=lt(this),r=s.size!==0,o=s.clear();return r&&Ji(s,"clear",void 0,void 0),o}}),["keys","values","entries",Symbol.iterator].forEach(s=>{t[s]=Ux(s,i,e)}),t}function Nf(i,e){const t=Ox(i,e);return(n,s,r)=>s==="__v_isReactive"?!i:s==="__v_isReadonly"?i:s==="__v_raw"?n:Reflect.get(ct(t,s)&&s in n?t:n,s,r)}const Nx={get:Nf(!1,!1)},zx={get:Nf(!1,!0)},kx={get:Nf(!0,!1)};const Um=new WeakMap,Om=new WeakMap,Nm=new WeakMap,Hx=new WeakMap;function Vx(i){switch(i){case"Object":case"Array":return 1;case"Map":case"Set":case"WeakMap":case"WeakSet":return 2;default:return 0}}function Gx(i){return i.__v_skip||!Object.isExtensible(i)?0:Vx(px(i))}function zf(i){return as(i)?i:kf(i,!1,Fx,Nx,Um)}function Wx(i){return kf(i,!1,Bx,zx,Om)}function mu(i){return kf(i,!0,Lx,kx,Nm)}function kf(i,e,t,n,s){if(!mt(i)||i.__v_raw&&!(e&&i.__v_isReactive))return i;const r=Gx(i);if(r===0)return i;const o=s.get(i);if(o)return o;const a=new Proxy(i,r===2?n:t);return s.set(i,a),a}function lr(i){return as(i)?lr(i.__v_raw):!!(i&&i.__v_isReactive)}function as(i){return!!(i&&i.__v_isReadonly)}function ai(i){return!!(i&&i.__v_isShallow)}function Hf(i){return i?!!i.__v_raw:!1}function lt(i){const e=i&&i.__v_raw;return e?lt(e):i}function Xx(i){return!ct(i,"__v_skip")&&Object.isExtensible(i)&&vm(i,"__v_skip",!0),i}const xi=i=>mt(i)?zf(i):i,so=i=>mt(i)?mu(i):i;function rn(i){return i?i.__v_isRef===!0:!1}function Bn(i){return qx(i,!1)}function qx(i,e){return rn(i)?i:new Qx(i,e)}class Qx{constructor(e,t){this.dep=new Of,this.__v_isRef=!0,this.__v_isShallow=!1,this._rawValue=t?e:lt(e),this._value=t?e:xi(e),this.__v_isShallow=t}get value(){return this.dep.track(),this._value}set value(e){const t=this._rawValue,n=this.__v_isShallow||ai(e)||as(e);e=n?e:lt(e),Rs(e,t)&&(this._rawValue=e,this._value=n?e:xi(e),this.dep.trigger())}}function Yx(i){return rn(i)?i.value:i}const Kx={get:(i,e,t)=>e==="__v_raw"?i:Yx(Reflect.get(i,e,t)),set:(i,e,t,n)=>{const s=i[e];return rn(s)&&!rn(t)?(s.value=t,!0):Reflect.set(i,e,t,n)}};function zm(i){return lr(i)?i:new Proxy(i,Kx)}class jx{constructor(e,t,n){this.fn=e,this.setter=t,this._value=void 0,this.dep=new Of(this),this.__v_isRef=!0,this.deps=void 0,this.depsTail=void 0,this.flags=16,this.globalVersion=Zo-1,this.next=void 0,this.effect=this,this.__v_isReadonly=!t,this.isSSR=n}notify(){if(this.flags|=16,!(this.flags&8)&&St!==this)return Em(this,!0),!0}get value(){const e=this.dep.track();return Im(this),e&&(e.version=this.dep.version),this._value}set value(e){this.setter&&this.setter(e)}}function $x(i,e,t=!1){let n,s;return $e(i)?n=i:(n=i.get,s=i.set),new jx(n,s,t)}const Ra={},Sl=new WeakMap;let er;function Zx(i,e=!1,t=er){if(t){let n=Sl.get(t);n||Sl.set(t,n=[]),n.push(i)}}function Jx(i,e,t=At){const{immediate:n,deep:s,once:r,scheduler:o,augmentJob:a,call:l}=t,c=S=>s?S:ai(S)||s===!1||s===0?es(S,1):es(S);let u,f,d,h,g=!1,p=!1;if(rn(i)?(f=()=>i.value,g=ai(i)):lr(i)?(f=()=>c(i),g=!0):Ke(i)?(p=!0,g=i.some(S=>lr(S)||ai(S)),f=()=>i.map(S=>{if(rn(S))return S.value;if(lr(S))return c(S);if($e(S))return l?l(S,2):S()})):$e(i)?e?f=l?()=>l(i,2):i:f=()=>{if(d){rs();try{d()}finally{os()}}const S=er;er=u;try{return l?l(i,3,[h]):i(h)}finally{er=S}}:f=Di,e&&s){const S=f,v=s===!0?1/0:s;f=()=>es(S(),v)}const x=Cx(),m=()=>{u.stop(),x&&x.active&&Rf(x.effects,u)};if(r&&e){const S=e;e=(...v)=>{S(...v),m()}}let _=p?new Array(i.length).fill(Ra):Ra;const A=S=>{if(!(!(u.flags&1)||!u.dirty&&!S))if(e){const v=u.run();if(s||g||(p?v.some((b,M)=>Rs(b,_[M])):Rs(v,_))){d&&d();const b=er;er=u;try{const M=[v,_===Ra?void 0:p&&_[0]===Ra?[]:_,h];_=v,l?l(e,3,M):e(...M)}finally{er=b}}}else u.run()};return a&&a(A),u=new Cm(f),u.scheduler=o?()=>o(A,!1):A,h=S=>Zx(S,!1,u),d=u.onStop=()=>{const S=Sl.get(u);if(S){if(l)l(S,4);else for(const v of S)v();Sl.delete(u)}},e?n?A(!0):_=u.run():o?o(A.bind(null,!0),!0):u.run(),m.pause=u.pause.bind(u),m.resume=u.resume.bind(u),m.stop=m,m}function es(i,e=1/0,t){if(e<=0||!mt(i)||i.__v_skip||(t=t||new Map,(t.get(i)||0)>=e))return i;if(t.set(i,e),e--,rn(i))es(i.value,e,t);else if(Ke(i))for(let n=0;n<i.length;n++)es(i[n],e,t);else if(gm(i)||qr(i))i.forEach(n=>{es(n,e,t)});else if(Am(i)){for(const n in i)es(i[n],e,t);for(const n of Object.getOwnPropertySymbols(i))Object.prototype.propertyIsEnumerable.call(i,n)&&es(i[n],e,t)}return i}function Aa(i,e,t,n){try{return n?i(...n):i()}catch(s){Wl(s,e,t)}}function Li(i,e,t,n){if($e(i)){const s=Aa(i,e,t,n);return s&&xm(s)&&s.catch(r=>{Wl(r,e,t)}),s}if(Ke(i)){const s=[];for(let r=0;r<i.length;r++)s.push(Li(i[r],e,t,n));return s}}function Wl(i,e,t,n=!0){const s=e?e.vnode:null,{errorHandler:r,throwUnhandledErrorInProduction:o}=e&&e.appContext.config||At;if(e){let a=e.parent;const l=e.proxy,c=`https://vuejs.org/error-reference/#runtime-${t}`;for(;a;){const u=a.ec;if(u){for(let f=0;f<u.length;f++)if(u[f](i,l,c)===!1)return}a=a.parent}if(r){rs(),Aa(r,null,10,[i,l,c]),os();return}}e_(i,t,s,n,o)}function e_(i,e,t,n=!0,s=!1){if(s)throw i;console.error(i)}const dn=[];let vi=-1;const Yr=[];let Ss=null,Nr=0;const km=Promise.resolve();let vl=null;function t_(i){const e=vl||km;return i?e.then(this?i.bind(this):i):e}function n_(i){let e=vi+1,t=dn.length;for(;e<t;){const n=e+t>>>1,s=dn[n],r=ea(s);r<i||r===i&&s.flags&2?e=n+1:t=n}return e}function Vf(i){if(!(i.flags&1)){const e=ea(i),t=dn[dn.length-1];!t||!(i.flags&2)&&e>=ea(t)?dn.push(i):dn.splice(n_(e),0,i),i.flags|=1,Hm()}}function Hm(){vl||(vl=km.then(Gm))}function i_(i){Ke(i)?Yr.push(...i):Ss&&i.id===-1?Ss.splice(Nr+1,0,i):i.flags&1||(Yr.push(i),i.flags|=1),Hm()}function jd(i,e,t=vi+1){for(;t<dn.length;t++){const n=dn[t];if(n&&n.flags&2){if(i&&n.id!==i.uid)continue;dn.splice(t,1),t--,n.flags&4&&(n.flags&=-2),n(),n.flags&4||(n.flags&=-2)}}}function Vm(i){if(Yr.length){const e=[...new Set(Yr)].sort((t,n)=>ea(t)-ea(n));if(Yr.length=0,Ss){Ss.push(...e);return}for(Ss=e,Nr=0;Nr<Ss.length;Nr++){const t=Ss[Nr];t.flags&4&&(t.flags&=-2),t.flags&8||t(),t.flags&=-2}Ss=null,Nr=0}}const ea=i=>i.id==null?i.flags&2?-1:1/0:i.id;function Gm(i){try{for(vi=0;vi<dn.length;vi++){const e=dn[vi];e&&!(e.flags&8)&&(e.flags&4&&(e.flags&=-2),Aa(e,e.i,e.i?15:14),e.flags&4||(e.flags&=-2))}}finally{for(;vi<dn.length;vi++){const e=dn[vi];e&&(e.flags&=-2)}vi=-1,dn.length=0,Vm(),vl=null,(dn.length||Yr.length)&&Gm()}}let ni=null,Wm=null;function yl(i){const e=ni;return ni=i,Wm=i&&i.type.__scopeId||null,e}function s_(i,e=ni,t){if(!e||i._n)return i;const n=(...s)=>{n._d&&ah(-1);const r=yl(e);let o;try{o=i(...s)}finally{yl(r),n._d&&ah(1)}return o};return n._n=!0,n._c=!0,n._d=!0,n}function r_(i,e){if(ni===null)return i;const t=Yl(ni),n=i.dirs||(i.dirs=[]);for(let s=0;s<e.length;s++){let[r,o,a,l=At]=e[s];r&&($e(r)&&(r={mounted:r,updated:r}),r.deep&&es(o),n.push({dir:r,instance:t,value:o,oldValue:void 0,arg:a,modifiers:l}))}return i}function Ws(i,e,t,n){const s=i.dirs,r=e&&e.dirs;for(let o=0;o<s.length;o++){const a=s[o];r&&(a.oldValue=r[o].value);let l=a.dir[n];l&&(rs(),Li(l,t,8,[i.el,a,i,e]),os())}}function o_(i,e){if(pn){let t=pn.provides;const n=pn.parent&&pn.parent.provides;n===t&&(t=pn.provides=Object.create(n)),t[i]=e}}function ll(i,e,t=!1){const n=oA();if(n||Kr){let s=Kr?Kr._context.provides:n?n.parent==null||n.ce?n.vnode.appContext&&n.vnode.appContext.provides:n.parent.provides:void 0;if(s&&i in s)return s[i];if(arguments.length>1)return t&&$e(e)?e.call(n&&n.proxy):e}}const a_=Symbol.for("v-scx"),l_=()=>ll(a_);function fc(i,e,t){return Xm(i,e,t)}function Xm(i,e,t=At){const{immediate:n,deep:s,flush:r,once:o}=t,a=ln({},t),l=e&&n||!e&&r!=="post";let c;if(na){if(r==="sync"){const h=l_();c=h.__watcherHandles||(h.__watcherHandles=[])}else if(!l){const h=()=>{};return h.stop=Di,h.resume=Di,h.pause=Di,h}}const u=pn;a.call=(h,g,p)=>Li(h,u,g,p);let f=!1;r==="post"?a.scheduler=h=>{bn(h,u&&u.suspense)}:r!=="sync"&&(f=!0,a.scheduler=(h,g)=>{g?h():Vf(h)}),a.augmentJob=h=>{e&&(h.flags|=4),f&&(h.flags|=2,u&&(h.id=u.uid,h.i=u))};const d=Jx(i,e,a);return na&&(c?c.push(d):l&&d()),d}function c_(i,e,t){const n=this.proxy,s=Xt(i)?i.includes(".")?qm(n,i):()=>n[i]:i.bind(n,n);let r;$e(e)?r=e:(r=e.handler,t=e);const o=Sa(this),a=Xm(s,r.bind(n),t);return o(),a}function qm(i,e){const t=e.split(".");return()=>{let n=i;for(let s=0;s<t.length&&n;s++)n=n[t[s]];return n}}const u_=Symbol("_vte"),f_=i=>i.__isTeleport,d_=Symbol("_leaveCb");function Gf(i,e){i.shapeFlag&6&&i.component?(i.transition=e,Gf(i.component.subTree,e)):i.shapeFlag&128?(i.ssContent.transition=e.clone(i.ssContent),i.ssFallback.transition=e.clone(i.ssFallback)):i.transition=e}function Qm(i){i.ids=[i.ids[0]+i.ids[2]+++"-",0,0]}function $d(i,e){let t;return!!((t=Object.getOwnPropertyDescriptor(i,e))&&!t.configurable)}const bl=new WeakMap;function ko(i,e,t,n,s=!1){if(Ke(i)){i.forEach((p,x)=>ko(p,e&&(Ke(e)?e[x]:e),t,n,s));return}if(Ho(n)&&!s){n.shapeFlag&512&&n.type.__asyncResolved&&n.component.subTree.component&&ko(i,e,t,n.component.subTree);return}const r=n.shapeFlag&4?Yl(n.component):n.el,o=s?null:r,{i:a,r:l}=i,c=e&&e.r,u=a.refs===At?a.refs={}:a.refs,f=a.setupState,d=lt(f),h=f===At?mm:p=>$d(u,p)?!1:ct(d,p),g=(p,x)=>!(x&&$d(u,x));if(c!=null&&c!==l){if(Zd(e),Xt(c))u[c]=null,h(c)&&(f[c]=null);else if(rn(c)){const p=e;g(c,p.k)&&(c.value=null),p.k&&(u[p.k]=null)}}if($e(l))Aa(l,a,12,[o,u]);else{const p=Xt(l),x=rn(l);if(p||x){const m=()=>{if(i.f){const _=p?h(l)?f[l]:u[l]:g()||!i.k?l.value:u[i.k];if(s)Ke(_)&&Rf(_,r);else if(Ke(_))_.includes(r)||_.push(r);else if(p)u[l]=[r],h(l)&&(f[l]=u[l]);else{const A=[r];g(l,i.k)&&(l.value=A),i.k&&(u[i.k]=A)}}else p?(u[l]=o,h(l)&&(f[l]=o)):x&&(g(l,i.k)&&(l.value=o),i.k&&(u[i.k]=o))};if(o){const _=()=>{m(),bl.delete(i)};_.id=-1,bl.set(i,_),bn(_,t)}else Zd(i),m()}}}function Zd(i){const e=bl.get(i);e&&(e.flags|=8,bl.delete(i))}Vl().requestIdleCallback;Vl().cancelIdleCallback;const Ho=i=>!!i.type.__asyncLoader,Ym=i=>i.type.__isKeepAlive;function h_(i,e){Km(i,"a",e)}function p_(i,e){Km(i,"da",e)}function Km(i,e,t=pn){const n=i.__wdc||(i.__wdc=()=>{let s=t;for(;s;){if(s.isDeactivated)return;s=s.parent}return i()});if(Xl(e,n,t),t){let s=t.parent;for(;s&&s.parent;)Ym(s.parent.vnode)&&m_(n,e,t,s),s=s.parent}}function m_(i,e,t,n){const s=Xl(e,i,n,!0);Zm(()=>{Rf(n[e],s)},t)}function Xl(i,e,t=pn,n=!1){if(t){const s=t[i]||(t[i]=[]),r=e.__weh||(e.__weh=(...o)=>{rs();const a=Sa(t),l=Li(e,t,i,o);return a(),os(),l});return n?s.unshift(r):s.push(r),r}}const us=i=>(e,t=pn)=>{(!na||i==="sp")&&Xl(i,(...n)=>e(...n),t)},g_=us("bm"),jm=us("m"),x_=us("bu"),__=us("u"),$m=us("bum"),Zm=us("um"),A_=us("sp"),S_=us("rtg"),v_=us("rtc");function y_(i,e=pn){Xl("ec",i,e)}const b_=Symbol.for("v-ndc");function M_(i,e,t,n){let s;const r=t,o=Ke(i);if(o||Xt(i)){const a=o&&lr(i);let l=!1,c=!1;a&&(l=!ai(i),c=as(i),i=Gl(i)),s=new Array(i.length);for(let u=0,f=i.length;u<f;u++)s[u]=e(l?c?so(xi(i[u])):xi(i[u]):i[u],u,void 0,r)}else if(typeof i=="number"){s=new Array(i);for(let a=0;a<i;a++)s[a]=e(a+1,a,void 0,r)}else if(mt(i))if(i[Symbol.iterator])s=Array.from(i,(a,l)=>e(a,l,void 0,r));else{const a=Object.keys(i);s=new Array(a.length);for(let l=0,c=a.length;l<c;l++){const u=a[l];s[l]=e(i[u],u,l,r)}}else s=[];return s}const gu=i=>i?A0(i)?Yl(i):gu(i.parent):null,Vo=ln(Object.create(null),{$:i=>i,$el:i=>i.vnode.el,$data:i=>i.data,$props:i=>i.props,$attrs:i=>i.attrs,$slots:i=>i.slots,$refs:i=>i.refs,$parent:i=>gu(i.parent),$root:i=>gu(i.root),$host:i=>i.ce,$emit:i=>i.emit,$options:i=>e0(i),$forceUpdate:i=>i.f||(i.f=()=>{Vf(i.update)}),$nextTick:i=>i.n||(i.n=t_.bind(i.proxy)),$watch:i=>c_.bind(i)}),dc=(i,e)=>i!==At&&!i.__isScriptSetup&&ct(i,e),C_={get({_:i},e){if(e==="__v_skip")return!0;const{ctx:t,setupState:n,data:s,props:r,accessCache:o,type:a,appContext:l}=i;if(e[0]!=="$"){const d=o[e];if(d!==void 0)switch(d){case 1:return n[e];case 2:return s[e];case 4:return t[e];case 3:return r[e]}else{if(dc(n,e))return o[e]=1,n[e];if(s!==At&&ct(s,e))return o[e]=2,s[e];if(ct(r,e))return o[e]=3,r[e];if(t!==At&&ct(t,e))return o[e]=4,t[e];xu&&(o[e]=0)}}const c=Vo[e];let u,f;if(c)return e==="$attrs"&&nn(i.attrs,"get",""),c(i);if((u=a.__cssModules)&&(u=u[e]))return u;if(t!==At&&ct(t,e))return o[e]=4,t[e];if(f=l.config.globalProperties,ct(f,e))return f[e]},set({_:i},e,t){const{data:n,setupState:s,ctx:r}=i;return dc(s,e)?(s[e]=t,!0):n!==At&&ct(n,e)?(n[e]=t,!0):ct(i.props,e)||e[0]==="$"&&e.slice(1)in i?!1:(r[e]=t,!0)},has({_:{data:i,setupState:e,accessCache:t,ctx:n,appContext:s,props:r,type:o}},a){let l;return!!(t[a]||i!==At&&a[0]!=="$"&&ct(i,a)||dc(e,a)||ct(r,a)||ct(n,a)||ct(Vo,a)||ct(s.config.globalProperties,a)||(l=o.__cssModules)&&l[a])},defineProperty(i,e,t){return t.get!=null?i._.accessCache[e]=0:ct(t,"value")&&this.set(i,e,t.value,null),Reflect.defineProperty(i,e,t)}};function Jd(i){return Ke(i)?i.reduce((e,t)=>(e[t]=null,e),{}):i}let xu=!0;function T_(i){const e=e0(i),t=i.proxy,n=i.ctx;xu=!1,e.beforeCreate&&eh(e.beforeCreate,i,"bc");const{data:s,computed:r,methods:o,watch:a,provide:l,inject:c,created:u,beforeMount:f,mounted:d,beforeUpdate:h,updated:g,activated:p,deactivated:x,beforeDestroy:m,beforeUnmount:_,destroyed:A,unmounted:S,render:v,renderTracked:b,renderTriggered:M,errorCaptured:E,serverPrefetch:y,expose:C,inheritAttrs:I,components:P,directives:U,filters:O}=e;if(c&&E_(c,n,null),o)for(const W in o){const H=o[W];$e(H)&&(n[W]=H.bind(t))}if(s){const W=s.call(t,t);mt(W)&&(i.data=zf(W))}if(xu=!0,r)for(const W in r){const H=r[W],$=$e(H)?H.bind(t,t):$e(H.get)?H.get.bind(t,t):Di,X=!$e(H)&&$e(H.set)?H.set.bind(t):Di,Z=v0({get:$,set:X});Object.defineProperty(n,W,{enumerable:!0,configurable:!0,get:()=>Z.value,set:pe=>Z.value=pe})}if(a)for(const W in a)Jm(a[W],n,t,W);if(l){const W=$e(l)?l.call(t):l;Reflect.ownKeys(W).forEach(H=>{o_(H,W[H])})}u&&eh(u,i,"c");function z(W,H){Ke(H)?H.forEach($=>W($.bind(t))):H&&W(H.bind(t))}if(z(g_,f),z(jm,d),z(x_,h),z(__,g),z(h_,p),z(p_,x),z(y_,E),z(v_,b),z(S_,M),z($m,_),z(Zm,S),z(A_,y),Ke(C))if(C.length){const W=i.exposed||(i.exposed={});C.forEach(H=>{Object.defineProperty(W,H,{get:()=>t[H],set:$=>t[H]=$,enumerable:!0})})}else i.exposed||(i.exposed={});v&&i.render===Di&&(i.render=v),I!=null&&(i.inheritAttrs=I),P&&(i.components=P),U&&(i.directives=U),y&&Qm(i)}function E_(i,e,t=Di){Ke(i)&&(i=_u(i));for(const n in i){const s=i[n];let r;mt(s)?"default"in s?r=ll(s.from||n,s.default,!0):r=ll(s.from||n):r=ll(s),rn(r)?Object.defineProperty(e,n,{enumerable:!0,configurable:!0,get:()=>r.value,set:o=>r.value=o}):e[n]=r}}function eh(i,e,t){Li(Ke(i)?i.map(n=>n.bind(e.proxy)):i.bind(e.proxy),e,t)}function Jm(i,e,t,n){let s=n.includes(".")?qm(t,n):()=>t[n];if(Xt(i)){const r=e[i];$e(r)&&fc(s,r)}else if($e(i))fc(s,i.bind(t));else if(mt(i))if(Ke(i))i.forEach(r=>Jm(r,e,t,n));else{const r=$e(i.handler)?i.handler.bind(t):e[i.handler];$e(r)&&fc(s,r,i)}}function e0(i){const e=i.type,{mixins:t,extends:n}=e,{mixins:s,optionsCache:r,config:{optionMergeStrategies:o}}=i.appContext,a=r.get(e);let l;return a?l=a:!s.length&&!t&&!n?l=e:(l={},s.length&&s.forEach(c=>Ml(l,c,o,!0)),Ml(l,e,o)),mt(e)&&r.set(e,l),l}function Ml(i,e,t,n=!1){const{mixins:s,extends:r}=e;r&&Ml(i,r,t,!0),s&&s.forEach(o=>Ml(i,o,t,!0));for(const o in e)if(!(n&&o==="expose")){const a=w_[o]||t&&t[o];i[o]=a?a(i[o],e[o]):e[o]}return i}const w_={data:th,props:nh,emits:nh,methods:Fo,computed:Fo,beforeCreate:un,created:un,beforeMount:un,mounted:un,beforeUpdate:un,updated:un,beforeDestroy:un,beforeUnmount:un,destroyed:un,unmounted:un,activated:un,deactivated:un,errorCaptured:un,serverPrefetch:un,components:Fo,directives:Fo,watch:I_,provide:th,inject:R_};function th(i,e){return e?i?function(){return ln($e(i)?i.call(this,this):i,$e(e)?e.call(this,this):e)}:e:i}function R_(i,e){return Fo(_u(i),_u(e))}function _u(i){if(Ke(i)){const e={};for(let t=0;t<i.length;t++)e[i[t]]=i[t];return e}return i}function un(i,e){return i?[...new Set([].concat(i,e))]:e}function Fo(i,e){return i?ln(Object.create(null),i,e):e}function nh(i,e){return i?Ke(i)&&Ke(e)?[...new Set([...i,...e])]:ln(Object.create(null),Jd(i),Jd(e??{})):e}function I_(i,e){if(!i)return e;if(!e)return i;const t=ln(Object.create(null),i);for(const n in e)t[n]=un(i[n],e[n]);return t}function t0(){return{app:null,config:{isNativeTag:mm,performance:!1,globalProperties:{},optionMergeStrategies:{},errorHandler:void 0,warnHandler:void 0,compilerOptions:{}},mixins:[],components:{},directives:{},provides:Object.create(null),optionsCache:new WeakMap,propsCache:new WeakMap,emitsCache:new WeakMap}}let D_=0;function P_(i,e){return function(n,s=null){$e(n)||(n=ln({},n)),s!=null&&!mt(s)&&(s=null);const r=t0(),o=new WeakSet,a=[];let l=!1;const c=r.app={_uid:D_++,_component:n,_props:s,_container:null,_context:r,_instance:null,version:dA,get config(){return r.config},set config(u){},use(u,...f){return o.has(u)||(u&&$e(u.install)?(o.add(u),u.install(c,...f)):$e(u)&&(o.add(u),u(c,...f))),c},mixin(u){return r.mixins.includes(u)||r.mixins.push(u),c},component(u,f){return f?(r.components[u]=f,c):r.components[u]},directive(u,f){return f?(r.directives[u]=f,c):r.directives[u]},mount(u,f,d){if(!l){const h=c._ceVNode||Pi(n,s);return h.appContext=r,d===!0?d="svg":d===!1&&(d=void 0),i(h,u,d),l=!0,c._container=u,u.__vue_app__=c,Yl(h.component)}},onUnmount(u){a.push(u)},unmount(){l&&(Li(a,c._instance,16),i(null,c._container),delete c._container.__vue_app__)},provide(u,f){return r.provides[u]=f,c},runWithContext(u){const f=Kr;Kr=c;try{return u()}finally{Kr=f}}};return c}}let Kr=null;const F_=(i,e)=>e==="modelValue"||e==="model-value"?i.modelModifiers:i[`${e}Modifiers`]||i[`${Ls(e)}Modifiers`]||i[`${ks(e)}Modifiers`];function L_(i,e,...t){if(i.isUnmounted)return;const n=i.vnode.props||At;let s=t;const r=e.startsWith("update:"),o=r&&F_(n,e.slice(7));o&&(o.trim&&(s=t.map(u=>Xt(u)?u.trim():u)),o.number&&(s=t.map(Df)));let a,l=n[a=oc(e)]||n[a=oc(Ls(e))];!l&&r&&(l=n[a=oc(ks(e))]),l&&Li(l,i,6,s);const c=n[a+"Once"];if(c){if(!i.emitted)i.emitted={};else if(i.emitted[a])return;i.emitted[a]=!0,Li(c,i,6,s)}}const B_=new WeakMap;function n0(i,e,t=!1){const n=t?B_:e.emitsCache,s=n.get(i);if(s!==void 0)return s;const r=i.emits;let o={},a=!1;if(!$e(i)){const l=c=>{const u=n0(c,e,!0);u&&(a=!0,ln(o,u))};!t&&e.mixins.length&&e.mixins.forEach(l),i.extends&&l(i.extends),i.mixins&&i.mixins.forEach(l)}return!r&&!a?(mt(i)&&n.set(i,null),null):(Ke(r)?r.forEach(l=>o[l]=null):ln(o,r),mt(i)&&n.set(i,o),o)}function ql(i,e){return!i||!kl(e)?!1:(e=e.slice(2).replace(/Once$/,""),ct(i,e[0].toLowerCase()+e.slice(1))||ct(i,ks(e))||ct(i,e))}function ih(i){const{type:e,vnode:t,proxy:n,withProxy:s,propsOptions:[r],slots:o,attrs:a,emit:l,render:c,renderCache:u,props:f,data:d,setupState:h,ctx:g,inheritAttrs:p}=i,x=yl(i);let m,_;try{if(t.shapeFlag&4){const S=s||n,v=S;m=Mi(c.call(v,S,u,f,h,d,g)),_=a}else{const S=e;m=Mi(S.length>1?S(f,{attrs:a,slots:o,emit:l}):S(f,null)),_=e.props?a:U_(a)}}catch(S){Go.length=0,Wl(S,i,1),m=Pi(Bs)}let A=m;if(_&&p!==!1){const S=Object.keys(_),{shapeFlag:v}=A;S.length&&v&7&&(r&&S.some(wf)&&(_=O_(_,r)),A=ro(A,_,!1,!0))}return t.dirs&&(A=ro(A,null,!1,!0),A.dirs=A.dirs?A.dirs.concat(t.dirs):t.dirs),t.transition&&Gf(A,t.transition),m=A,yl(x),m}const U_=i=>{let e;for(const t in i)(t==="class"||t==="style"||kl(t))&&((e||(e={}))[t]=i[t]);return e},O_=(i,e)=>{const t={};for(const n in i)(!wf(n)||!(n.slice(9)in e))&&(t[n]=i[n]);return t};function N_(i,e,t){const{props:n,children:s,component:r}=i,{props:o,children:a,patchFlag:l}=e,c=r.emitsOptions;if(e.dirs||e.transition)return!0;if(t&&l>=0){if(l&1024)return!0;if(l&16)return n?sh(n,o,c):!!o;if(l&8){const u=e.dynamicProps;for(let f=0;f<u.length;f++){const d=u[f];if(i0(o,n,d)&&!ql(c,d))return!0}}}else return(s||a)&&(!a||!a.$stable)?!0:n===o?!1:n?o?sh(n,o,c):!0:!!o;return!1}function sh(i,e,t){const n=Object.keys(e);if(n.length!==Object.keys(i).length)return!0;for(let s=0;s<n.length;s++){const r=n[s];if(i0(e,i,r)&&!ql(t,r))return!0}return!1}function i0(i,e,t){const n=i[t],s=e[t];return t==="style"&&mt(n)&&mt(s)?!Ff(n,s):n!==s}function z_({vnode:i,parent:e},t){for(;e;){const n=e.subTree;if(n.suspense&&n.suspense.activeBranch===i&&(n.el=i.el),n===i)(i=e.vnode).el=t,e=e.parent;else break}}const s0={},r0=()=>Object.create(s0),o0=i=>Object.getPrototypeOf(i)===s0;function k_(i,e,t,n=!1){const s={},r=r0();i.propsDefaults=Object.create(null),a0(i,e,s,r);for(const o in i.propsOptions[0])o in s||(s[o]=void 0);t?i.props=n?s:Wx(s):i.type.props?i.props=s:i.props=r,i.attrs=r}function H_(i,e,t,n){const{props:s,attrs:r,vnode:{patchFlag:o}}=i,a=lt(s),[l]=i.propsOptions;let c=!1;if((n||o>0)&&!(o&16)){if(o&8){const u=i.vnode.dynamicProps;for(let f=0;f<u.length;f++){let d=u[f];if(ql(i.emitsOptions,d))continue;const h=e[d];if(l)if(ct(r,d))h!==r[d]&&(r[d]=h,c=!0);else{const g=Ls(d);s[g]=Au(l,a,g,h,i,!1)}else h!==r[d]&&(r[d]=h,c=!0)}}}else{a0(i,e,s,r)&&(c=!0);let u;for(const f in a)(!e||!ct(e,f)&&((u=ks(f))===f||!ct(e,u)))&&(l?t&&(t[f]!==void 0||t[u]!==void 0)&&(s[f]=Au(l,a,f,void 0,i,!0)):delete s[f]);if(r!==a)for(const f in r)(!e||!ct(e,f))&&(delete r[f],c=!0)}c&&Ji(i.attrs,"set","")}function a0(i,e,t,n){const[s,r]=i.propsOptions;let o=!1,a;if(e)for(let l in e){if(Oo(l))continue;const c=e[l];let u;s&&ct(s,u=Ls(l))?!r||!r.includes(u)?t[u]=c:(a||(a={}))[u]=c:ql(i.emitsOptions,l)||(!(l in n)||c!==n[l])&&(n[l]=c,o=!0)}if(r){const l=lt(t),c=a||At;for(let u=0;u<r.length;u++){const f=r[u];t[f]=Au(s,l,f,c[f],i,!ct(c,f))}}return o}function Au(i,e,t,n,s,r){const o=i[t];if(o!=null){const a=ct(o,"default");if(a&&n===void 0){const l=o.default;if(o.type!==Function&&!o.skipFactory&&$e(l)){const{propsDefaults:c}=s;if(t in c)n=c[t];else{const u=Sa(s);n=c[t]=l.call(null,e),u()}}else n=l;s.ce&&s.ce._setProp(t,n)}o[0]&&(r&&!a?n=!1:o[1]&&(n===""||n===ks(t))&&(n=!0))}return n}const V_=new WeakMap;function l0(i,e,t=!1){const n=t?V_:e.propsCache,s=n.get(i);if(s)return s;const r=i.props,o={},a=[];let l=!1;if(!$e(i)){const u=f=>{l=!0;const[d,h]=l0(f,e,!0);ln(o,d),h&&a.push(...h)};!t&&e.mixins.length&&e.mixins.forEach(u),i.extends&&u(i.extends),i.mixins&&i.mixins.forEach(u)}if(!r&&!l)return mt(i)&&n.set(i,Xr),Xr;if(Ke(r))for(let u=0;u<r.length;u++){const f=Ls(r[u]);rh(f)&&(o[f]=At)}else if(r)for(const u in r){const f=Ls(u);if(rh(f)){const d=r[u],h=o[f]=Ke(d)||$e(d)?{type:d}:ln({},d),g=h.type;let p=!1,x=!0;if(Ke(g))for(let m=0;m<g.length;++m){const _=g[m],A=$e(_)&&_.name;if(A==="Boolean"){p=!0;break}else A==="String"&&(x=!1)}else p=$e(g)&&g.name==="Boolean";h[0]=p,h[1]=x,(p||ct(h,"default"))&&a.push(f)}}const c=[o,a];return mt(i)&&n.set(i,c),c}function rh(i){return i[0]!=="$"&&!Oo(i)}const Wf=i=>i==="_"||i==="_ctx"||i==="$stable",Xf=i=>Ke(i)?i.map(Mi):[Mi(i)],G_=(i,e,t)=>{if(e._n)return e;const n=s_((...s)=>Xf(e(...s)),t);return n._c=!1,n},c0=(i,e,t)=>{const n=i._ctx;for(const s in i){if(Wf(s))continue;const r=i[s];if($e(r))e[s]=G_(s,r,n);else if(r!=null){const o=Xf(r);e[s]=()=>o}}},u0=(i,e)=>{const t=Xf(e);i.slots.default=()=>t},f0=(i,e,t)=>{for(const n in e)(t||!Wf(n))&&(i[n]=e[n])},W_=(i,e,t)=>{const n=i.slots=r0();if(i.vnode.shapeFlag&32){const s=e._;s?(f0(n,e,t),t&&vm(n,"_",s,!0)):c0(e,n)}else e&&u0(i,e)},X_=(i,e,t)=>{const{vnode:n,slots:s}=i;let r=!0,o=At;if(n.shapeFlag&32){const a=e._;a?t&&a===1?r=!1:f0(s,e,t):(r=!e.$stable,c0(e,s)),o=e}else e&&(u0(i,e),o={default:1});if(r)for(const a in s)!Wf(a)&&o[a]==null&&delete s[a]},bn=j_;function q_(i){return Q_(i)}function Q_(i,e){const t=Vl();t.__VUE__=!0;const{insert:n,remove:s,patchProp:r,createElement:o,createText:a,createComment:l,setText:c,setElementText:u,parentNode:f,nextSibling:d,setScopeId:h=Di,insertStaticContent:g}=i,p=(F,B,q,w=null,ne=null,re=null,ue=void 0,ie=null,fe=!!B.dynamicChildren)=>{if(F===B)return;F&&!Co(F,B)&&(w=oe(F),pe(F,ne,re,!0),F=null),B.patchFlag===-2&&(fe=!1,B.dynamicChildren=null);const{type:te,ref:xe,shapeFlag:R}=B;switch(te){case Ql:x(F,B,q,w);break;case Bs:m(F,B,q,w);break;case pc:F==null&&_(B,q,w,ue);break;case bi:P(F,B,q,w,ne,re,ue,ie,fe);break;default:R&1?v(F,B,q,w,ne,re,ue,ie,fe):R&6?U(F,B,q,w,ne,re,ue,ie,fe):(R&64||R&128)&&te.process(F,B,q,w,ne,re,ue,ie,fe,ye)}xe!=null&&ne?ko(xe,F&&F.ref,re,B||F,!B):xe==null&&F&&F.ref!=null&&ko(F.ref,null,re,F,!0)},x=(F,B,q,w)=>{if(F==null)n(B.el=a(B.children),q,w);else{const ne=B.el=F.el;B.children!==F.children&&c(ne,B.children)}},m=(F,B,q,w)=>{F==null?n(B.el=l(B.children||""),q,w):B.el=F.el},_=(F,B,q,w)=>{[F.el,F.anchor]=g(F.children,B,q,w,F.el,F.anchor)},A=({el:F,anchor:B},q,w)=>{let ne;for(;F&&F!==B;)ne=d(F),n(F,q,w),F=ne;n(B,q,w)},S=({el:F,anchor:B})=>{let q;for(;F&&F!==B;)q=d(F),s(F),F=q;s(B)},v=(F,B,q,w,ne,re,ue,ie,fe)=>{if(B.type==="svg"?ue="svg":B.type==="math"&&(ue="mathml"),F==null)b(B,q,w,ne,re,ue,ie,fe);else{const te=F.el&&F.el._isVueCE?F.el:null;try{te&&te._beginPatch(),y(F,B,ne,re,ue,ie,fe)}finally{te&&te._endPatch()}}},b=(F,B,q,w,ne,re,ue,ie)=>{let fe,te;const{props:xe,shapeFlag:R,transition:T,dirs:V}=F;if(fe=F.el=o(F.type,re,xe&&xe.is,xe),R&8?u(fe,F.children):R&16&&E(F.children,fe,null,w,ne,hc(F,re),ue,ie),V&&Ws(F,null,w,"created"),M(fe,F,F.scopeId,ue,w),xe){for(const ce in xe)ce!=="value"&&!Oo(ce)&&r(fe,ce,null,xe[ce],re,w);"value"in xe&&r(fe,"value",null,xe.value,re),(te=xe.onVnodeBeforeMount)&&Si(te,w,F)}V&&Ws(F,null,w,"beforeMount");const se=Y_(ne,T);se&&T.beforeEnter(fe),n(fe,B,q),((te=xe&&xe.onVnodeMounted)||se||V)&&bn(()=>{te&&Si(te,w,F),se&&T.enter(fe),V&&Ws(F,null,w,"mounted")},ne)},M=(F,B,q,w,ne)=>{if(q&&h(F,q),w)for(let re=0;re<w.length;re++)h(F,w[re]);if(ne){let re=ne.subTree;if(B===re||m0(re.type)&&(re.ssContent===B||re.ssFallback===B)){const ue=ne.vnode;M(F,ue,ue.scopeId,ue.slotScopeIds,ne.parent)}}},E=(F,B,q,w,ne,re,ue,ie,fe=0)=>{for(let te=fe;te<F.length;te++){const xe=F[te]=ie?ji(F[te]):Mi(F[te]);p(null,xe,B,q,w,ne,re,ue,ie)}},y=(F,B,q,w,ne,re,ue)=>{const ie=B.el=F.el;let{patchFlag:fe,dynamicChildren:te,dirs:xe}=B;fe|=F.patchFlag&16;const R=F.props||At,T=B.props||At;let V;if(q&&Xs(q,!1),(V=T.onVnodeBeforeUpdate)&&Si(V,q,B,F),xe&&Ws(B,F,q,"beforeUpdate"),q&&Xs(q,!0),(R.innerHTML&&T.innerHTML==null||R.textContent&&T.textContent==null)&&u(ie,""),te?C(F.dynamicChildren,te,ie,q,w,hc(B,ne),re):ue||H(F,B,ie,null,q,w,hc(B,ne),re,!1),fe>0){if(fe&16)I(ie,R,T,q,ne);else if(fe&2&&R.class!==T.class&&r(ie,"class",null,T.class,ne),fe&4&&r(ie,"style",R.style,T.style,ne),fe&8){const se=B.dynamicProps;for(let ce=0;ce<se.length;ce++){const J=se[ce],Pe=R[J],Ae=T[J];(Ae!==Pe||J==="value")&&r(ie,J,Pe,Ae,ne,q)}}fe&1&&F.children!==B.children&&u(ie,B.children)}else!ue&&te==null&&I(ie,R,T,q,ne);((V=T.onVnodeUpdated)||xe)&&bn(()=>{V&&Si(V,q,B,F),xe&&Ws(B,F,q,"updated")},w)},C=(F,B,q,w,ne,re,ue)=>{for(let ie=0;ie<B.length;ie++){const fe=F[ie],te=B[ie],xe=fe.el&&(fe.type===bi||!Co(fe,te)||fe.shapeFlag&198)?f(fe.el):q;p(fe,te,xe,null,w,ne,re,ue,!0)}},I=(F,B,q,w,ne)=>{if(B!==q){if(B!==At)for(const re in B)!Oo(re)&&!(re in q)&&r(F,re,B[re],null,ne,w);for(const re in q){if(Oo(re))continue;const ue=q[re],ie=B[re];ue!==ie&&re!=="value"&&r(F,re,ie,ue,ne,w)}"value"in q&&r(F,"value",B.value,q.value,ne)}},P=(F,B,q,w,ne,re,ue,ie,fe)=>{const te=B.el=F?F.el:a(""),xe=B.anchor=F?F.anchor:a("");let{patchFlag:R,dynamicChildren:T,slotScopeIds:V}=B;V&&(ie=ie?ie.concat(V):V),F==null?(n(te,q,w),n(xe,q,w),E(B.children||[],q,xe,ne,re,ue,ie,fe)):R>0&&R&64&&T&&F.dynamicChildren&&F.dynamicChildren.length===T.length?(C(F.dynamicChildren,T,q,ne,re,ue,ie),(B.key!=null||ne&&B===ne.subTree)&&d0(F,B,!0)):H(F,B,q,xe,ne,re,ue,ie,fe)},U=(F,B,q,w,ne,re,ue,ie,fe)=>{B.slotScopeIds=ie,F==null?B.shapeFlag&512?ne.ctx.activate(B,q,w,ue,fe):O(B,q,w,ne,re,ue,fe):k(F,B,fe)},O=(F,B,q,w,ne,re,ue)=>{const ie=F.component=rA(F,w,ne);if(Ym(F)&&(ie.ctx.renderer=ye),aA(ie,!1,ue),ie.asyncDep){if(ne&&ne.registerDep(ie,z,ue),!F.el){const fe=ie.subTree=Pi(Bs);m(null,fe,B,q),F.placeholder=fe.el}}else z(ie,F,B,q,ne,re,ue)},k=(F,B,q)=>{const w=B.component=F.component;if(N_(F,B,q))if(w.asyncDep&&!w.asyncResolved){W(w,B,q);return}else w.next=B,w.update();else B.el=F.el,w.vnode=B},z=(F,B,q,w,ne,re,ue)=>{const ie=()=>{if(F.isMounted){let{next:R,bu:T,u:V,parent:se,vnode:ce}=F;{const N=h0(F);if(N){R&&(R.el=ce.el,W(F,R,ue)),N.asyncDep.then(()=>{bn(()=>{F.isUnmounted||te()},ne)});return}}let J=R,Pe;Xs(F,!1),R?(R.el=ce.el,W(F,R,ue)):R=ce,T&&al(T),(Pe=R.props&&R.props.onVnodeBeforeUpdate)&&Si(Pe,se,R,ce),Xs(F,!0);const Ae=ih(F),Ue=F.subTree;F.subTree=Ae,p(Ue,Ae,f(Ue.el),oe(Ue),F,ne,re),R.el=Ae.el,J===null&&z_(F,Ae.el),V&&bn(V,ne),(Pe=R.props&&R.props.onVnodeUpdated)&&bn(()=>Si(Pe,se,R,ce),ne)}else{let R;const{el:T,props:V}=B,{bm:se,m:ce,parent:J,root:Pe,type:Ae}=F,Ue=Ho(B);Xs(F,!1),se&&al(se),!Ue&&(R=V&&V.onVnodeBeforeMount)&&Si(R,J,B),Xs(F,!0);{Pe.ce&&Pe.ce._hasShadowRoot()&&Pe.ce._injectChildStyle(Ae);const N=F.subTree=ih(F);p(null,N,q,w,F,ne,re),B.el=N.el}if(ce&&bn(ce,ne),!Ue&&(R=V&&V.onVnodeMounted)){const N=B;bn(()=>Si(R,J,N),ne)}(B.shapeFlag&256||J&&Ho(J.vnode)&&J.vnode.shapeFlag&256)&&F.a&&bn(F.a,ne),F.isMounted=!0,B=q=w=null}};F.scope.on();const fe=F.effect=new Cm(ie);F.scope.off();const te=F.update=fe.run.bind(fe),xe=F.job=fe.runIfDirty.bind(fe);xe.i=F,xe.id=F.uid,fe.scheduler=()=>Vf(xe),Xs(F,!0),te()},W=(F,B,q)=>{B.component=F;const w=F.vnode.props;F.vnode=B,F.next=null,H_(F,B.props,w,q),X_(F,B.children,q),rs(),jd(F),os()},H=(F,B,q,w,ne,re,ue,ie,fe=!1)=>{const te=F&&F.children,xe=F?F.shapeFlag:0,R=B.children,{patchFlag:T,shapeFlag:V}=B;if(T>0){if(T&128){X(te,R,q,w,ne,re,ue,ie,fe);return}else if(T&256){$(te,R,q,w,ne,re,ue,ie,fe);return}}V&8?(xe&16&&Y(te,ne,re),R!==te&&u(q,R)):xe&16?V&16?X(te,R,q,w,ne,re,ue,ie,fe):Y(te,ne,re,!0):(xe&8&&u(q,""),V&16&&E(R,q,w,ne,re,ue,ie,fe))},$=(F,B,q,w,ne,re,ue,ie,fe)=>{F=F||Xr,B=B||Xr;const te=F.length,xe=B.length,R=Math.min(te,xe);let T;for(T=0;T<R;T++){const V=B[T]=fe?ji(B[T]):Mi(B[T]);p(F[T],V,q,null,ne,re,ue,ie,fe)}te>xe?Y(F,ne,re,!0,!1,R):E(B,q,w,ne,re,ue,ie,fe,R)},X=(F,B,q,w,ne,re,ue,ie,fe)=>{let te=0;const xe=B.length;let R=F.length-1,T=xe-1;for(;te<=R&&te<=T;){const V=F[te],se=B[te]=fe?ji(B[te]):Mi(B[te]);if(Co(V,se))p(V,se,q,null,ne,re,ue,ie,fe);else break;te++}for(;te<=R&&te<=T;){const V=F[R],se=B[T]=fe?ji(B[T]):Mi(B[T]);if(Co(V,se))p(V,se,q,null,ne,re,ue,ie,fe);else break;R--,T--}if(te>R){if(te<=T){const V=T+1,se=V<xe?B[V].el:w;for(;te<=T;)p(null,B[te]=fe?ji(B[te]):Mi(B[te]),q,se,ne,re,ue,ie,fe),te++}}else if(te>T)for(;te<=R;)pe(F[te],ne,re,!0),te++;else{const V=te,se=te,ce=new Map;for(te=se;te<=T;te++){const Te=B[te]=fe?ji(B[te]):Mi(B[te]);Te.key!=null&&ce.set(Te.key,te)}let J,Pe=0;const Ae=T-se+1;let Ue=!1,N=0;const ee=new Array(Ae);for(te=0;te<Ae;te++)ee[te]=0;for(te=V;te<=R;te++){const Te=F[te];if(Pe>=Ae){pe(Te,ne,re,!0);continue}let Ie;if(Te.key!=null)Ie=ce.get(Te.key);else for(J=se;J<=T;J++)if(ee[J-se]===0&&Co(Te,B[J])){Ie=J;break}Ie===void 0?pe(Te,ne,re,!0):(ee[Ie-se]=te+1,Ie>=N?N=Ie:Ue=!0,p(Te,B[Ie],q,null,ne,re,ue,ie,fe),Pe++)}const me=Ue?K_(ee):Xr;for(J=me.length-1,te=Ae-1;te>=0;te--){const Te=se+te,Ie=B[Te],we=B[Te+1],He=Te+1<xe?we.el||p0(we):w;ee[te]===0?p(null,Ie,q,He,ne,re,ue,ie,fe):Ue&&(J<0||te!==me[J]?Z(Ie,q,He,2):J--)}}},Z=(F,B,q,w,ne=null)=>{const{el:re,type:ue,transition:ie,children:fe,shapeFlag:te}=F;if(te&6){Z(F.component.subTree,B,q,w);return}if(te&128){F.suspense.move(B,q,w);return}if(te&64){ue.move(F,B,q,ye);return}if(ue===bi){n(re,B,q);for(let R=0;R<fe.length;R++)Z(fe[R],B,q,w);n(F.anchor,B,q);return}if(ue===pc){A(F,B,q);return}if(w!==2&&te&1&&ie)if(w===0)ie.beforeEnter(re),n(re,B,q),bn(()=>ie.enter(re),ne);else{const{leave:R,delayLeave:T,afterLeave:V}=ie,se=()=>{F.ctx.isUnmounted?s(re):n(re,B,q)},ce=()=>{re._isLeaving&&re[d_](!0),R(re,()=>{se(),V&&V()})};T?T(re,se,ce):ce()}else n(re,B,q)},pe=(F,B,q,w=!1,ne=!1)=>{const{type:re,props:ue,ref:ie,children:fe,dynamicChildren:te,shapeFlag:xe,patchFlag:R,dirs:T,cacheIndex:V}=F;if(R===-2&&(ne=!1),ie!=null&&(rs(),ko(ie,null,q,F,!0),os()),V!=null&&(B.renderCache[V]=void 0),xe&256){B.ctx.deactivate(F);return}const se=xe&1&&T,ce=!Ho(F);let J;if(ce&&(J=ue&&ue.onVnodeBeforeUnmount)&&Si(J,B,F),xe&6)Me(F.component,q,w);else{if(xe&128){F.suspense.unmount(q,w);return}se&&Ws(F,null,B,"beforeUnmount"),xe&64?F.type.remove(F,B,q,ye,w):te&&!te.hasOnce&&(re!==bi||R>0&&R&64)?Y(te,B,q,!1,!0):(re===bi&&R&384||!ne&&xe&16)&&Y(fe,B,q),w&&ge(F)}(ce&&(J=ue&&ue.onVnodeUnmounted)||se)&&bn(()=>{J&&Si(J,B,F),se&&Ws(F,null,B,"unmounted")},q)},ge=F=>{const{type:B,el:q,anchor:w,transition:ne}=F;if(B===bi){be(q,w);return}if(B===pc){S(F);return}const re=()=>{s(q),ne&&!ne.persisted&&ne.afterLeave&&ne.afterLeave()};if(F.shapeFlag&1&&ne&&!ne.persisted){const{leave:ue,delayLeave:ie}=ne,fe=()=>ue(q,re);ie?ie(F.el,re,fe):fe()}else re()},be=(F,B)=>{let q;for(;F!==B;)q=d(F),s(F),F=q;s(B)},Me=(F,B,q)=>{const{bum:w,scope:ne,job:re,subTree:ue,um:ie,m:fe,a:te}=F;oh(fe),oh(te),w&&al(w),ne.stop(),re&&(re.flags|=8,pe(ue,F,B,q)),ie&&bn(ie,B),bn(()=>{F.isUnmounted=!0},B)},Y=(F,B,q,w=!1,ne=!1,re=0)=>{for(let ue=re;ue<F.length;ue++)pe(F[ue],B,q,w,ne)},oe=F=>{if(F.shapeFlag&6)return oe(F.component.subTree);if(F.shapeFlag&128)return F.suspense.next();const B=d(F.anchor||F.el),q=B&&B[u_];return q?d(q):B};let he=!1;const Fe=(F,B,q)=>{let w;F==null?B._vnode&&(pe(B._vnode,null,null,!0),w=B._vnode.component):p(B._vnode||null,F,B,null,null,null,q),B._vnode=F,he||(he=!0,jd(w),Vm(),he=!1)},ye={p,um:pe,m:Z,r:ge,mt:O,mc:E,pc:H,pbc:C,n:oe,o:i};return{render:Fe,hydrate:void 0,createApp:P_(Fe)}}function hc({type:i,props:e},t){return t==="svg"&&i==="foreignObject"||t==="mathml"&&i==="annotation-xml"&&e&&e.encoding&&e.encoding.includes("html")?void 0:t}function Xs({effect:i,job:e},t){t?(i.flags|=32,e.flags|=4):(i.flags&=-33,e.flags&=-5)}function Y_(i,e){return(!i||i&&!i.pendingBranch)&&e&&!e.persisted}function d0(i,e,t=!1){const n=i.children,s=e.children;if(Ke(n)&&Ke(s))for(let r=0;r<n.length;r++){const o=n[r];let a=s[r];a.shapeFlag&1&&!a.dynamicChildren&&((a.patchFlag<=0||a.patchFlag===32)&&(a=s[r]=ji(s[r]),a.el=o.el),!t&&a.patchFlag!==-2&&d0(o,a)),a.type===Ql&&(a.patchFlag===-1&&(a=s[r]=ji(a)),a.el=o.el),a.type===Bs&&!a.el&&(a.el=o.el)}}function K_(i){const e=i.slice(),t=[0];let n,s,r,o,a;const l=i.length;for(n=0;n<l;n++){const c=i[n];if(c!==0){if(s=t[t.length-1],i[s]<c){e[n]=s,t.push(n);continue}for(r=0,o=t.length-1;r<o;)a=r+o>>1,i[t[a]]<c?r=a+1:o=a;c<i[t[r]]&&(r>0&&(e[n]=t[r-1]),t[r]=n)}}for(r=t.length,o=t[r-1];r-- >0;)t[r]=o,o=e[o];return t}function h0(i){const e=i.subTree.component;if(e)return e.asyncDep&&!e.asyncResolved?e:h0(e)}function oh(i){if(i)for(let e=0;e<i.length;e++)i[e].flags|=8}function p0(i){if(i.placeholder)return i.placeholder;const e=i.component;return e?p0(e.subTree):null}const m0=i=>i.__isSuspense;function j_(i,e){e&&e.pendingBranch?Ke(i)?e.effects.push(...i):e.effects.push(i):i_(i)}const bi=Symbol.for("v-fgt"),Ql=Symbol.for("v-txt"),Bs=Symbol.for("v-cmt"),pc=Symbol.for("v-stc"),Go=[];let Vn=null;function Mn(i=!1){Go.push(Vn=i?null:[])}function $_(){Go.pop(),Vn=Go[Go.length-1]||null}let ta=1;function ah(i,e=!1){ta+=i,i<0&&Vn&&e&&(Vn.hasOnce=!0)}function g0(i){return i.dynamicChildren=ta>0?Vn||Xr:null,$_(),ta>0&&Vn&&Vn.push(i),i}function Nn(i,e,t,n,s,r){return g0(Ht(i,e,t,n,s,r,!0))}function Z_(i,e,t,n,s){return g0(Pi(i,e,t,n,s,!0))}function x0(i){return i?i.__v_isVNode===!0:!1}function Co(i,e){return i.type===e.type&&i.key===e.key}const _0=({key:i})=>i??null,cl=({ref:i,ref_key:e,ref_for:t})=>(typeof i=="number"&&(i=""+i),i!=null?Xt(i)||rn(i)||$e(i)?{i:ni,r:i,k:e,f:!!t}:i:null);function Ht(i,e=null,t=null,n=0,s=null,r=i===bi?0:1,o=!1,a=!1){const l={__v_isVNode:!0,__v_skip:!0,type:i,props:e,key:e&&_0(e),ref:e&&cl(e),scopeId:Wm,slotScopeIds:null,children:t,component:null,suspense:null,ssContent:null,ssFallback:null,dirs:null,transition:null,el:null,anchor:null,target:null,targetStart:null,targetAnchor:null,staticCount:0,shapeFlag:r,patchFlag:n,dynamicProps:s,dynamicChildren:null,appContext:null,ctx:ni};return a?(qf(l,t),r&128&&i.normalize(l)):t&&(l.shapeFlag|=Xt(t)?8:16),ta>0&&!o&&Vn&&(l.patchFlag>0||r&6)&&l.patchFlag!==32&&Vn.push(l),l}const Pi=J_;function J_(i,e=null,t=null,n=0,s=null,r=!1){if((!i||i===b_)&&(i=Bs),x0(i)){const a=ro(i,e,!0);return t&&qf(a,t),ta>0&&!r&&Vn&&(a.shapeFlag&6?Vn[Vn.indexOf(i)]=a:Vn.push(a)),a.patchFlag=-2,a}if(fA(i)&&(i=i.__vccOpts),e){e=eA(e);let{class:a,style:l}=e;a&&!Xt(a)&&(e.class=Qr(a)),mt(l)&&(Hf(l)&&!Ke(l)&&(l=ln({},l)),e.style=Pf(l))}const o=Xt(i)?1:m0(i)?128:f_(i)?64:mt(i)?4:$e(i)?2:0;return Ht(i,e,t,n,s,o,r,!0)}function eA(i){return i?Hf(i)||o0(i)?ln({},i):i:null}function ro(i,e,t=!1,n=!1){const{props:s,ref:r,patchFlag:o,children:a,transition:l}=i,c=e?nA(s||{},e):s,u={__v_isVNode:!0,__v_skip:!0,type:i.type,props:c,key:c&&_0(c),ref:e&&e.ref?t&&r?Ke(r)?r.concat(cl(e)):[r,cl(e)]:cl(e):r,scopeId:i.scopeId,slotScopeIds:i.slotScopeIds,children:a,target:i.target,targetStart:i.targetStart,targetAnchor:i.targetAnchor,staticCount:i.staticCount,shapeFlag:i.shapeFlag,patchFlag:e&&i.type!==bi?o===-1?16:o|16:o,dynamicProps:i.dynamicProps,dynamicChildren:i.dynamicChildren,appContext:i.appContext,dirs:i.dirs,transition:l,component:i.component,suspense:i.suspense,ssContent:i.ssContent&&ro(i.ssContent),ssFallback:i.ssFallback&&ro(i.ssFallback),placeholder:i.placeholder,el:i.el,anchor:i.anchor,ctx:i.ctx,ce:i.ce};return l&&n&&Gf(u,l.clone(u)),u}function tA(i=" ",e=0){return Pi(Ql,null,i,e)}function fs(i="",e=!1){return e?(Mn(),Z_(Bs,null,i)):Pi(Bs,null,i)}function Mi(i){return i==null||typeof i=="boolean"?Pi(Bs):Ke(i)?Pi(bi,null,i.slice()):x0(i)?ji(i):Pi(Ql,null,String(i))}function ji(i){return i.el===null&&i.patchFlag!==-1||i.memo?i:ro(i)}function qf(i,e){let t=0;const{shapeFlag:n}=i;if(e==null)e=null;else if(Ke(e))t=16;else if(typeof e=="object")if(n&65){const s=e.default;s&&(s._c&&(s._d=!1),qf(i,s()),s._c&&(s._d=!0));return}else{t=32;const s=e._;!s&&!o0(e)?e._ctx=ni:s===3&&ni&&(ni.slots._===1?e._=1:(e._=2,i.patchFlag|=1024))}else $e(e)?(e={default:e,_ctx:ni},t=32):(e=String(e),n&64?(t=16,e=[tA(e)]):t=8);i.children=e,i.shapeFlag|=t}function nA(...i){const e={};for(let t=0;t<i.length;t++){const n=i[t];for(const s in n)if(s==="class")e.class!==n.class&&(e.class=Qr([e.class,n.class]));else if(s==="style")e.style=Pf([e.style,n.style]);else if(kl(s)){const r=e[s],o=n[s];o&&r!==o&&!(Ke(r)&&r.includes(o))&&(e[s]=r?[].concat(r,o):o)}else s!==""&&(e[s]=n[s])}return e}function Si(i,e,t,n=null){Li(i,e,7,[t,n])}const iA=t0();let sA=0;function rA(i,e,t){const n=i.type,s=(e?e.appContext:i.appContext)||iA,r={uid:sA++,vnode:i,type:n,parent:e,appContext:s,root:null,next:null,subTree:null,effect:null,update:null,job:null,scope:new Mx(!0),render:null,proxy:null,exposed:null,exposeProxy:null,withProxy:null,provides:e?e.provides:Object.create(s.provides),ids:e?e.ids:["",0,0],accessCache:null,renderCache:[],components:null,directives:null,propsOptions:l0(n,s),emitsOptions:n0(n,s),emit:null,emitted:null,propsDefaults:At,inheritAttrs:n.inheritAttrs,ctx:At,data:At,props:At,attrs:At,slots:At,refs:At,setupState:At,setupContext:null,suspense:t,suspenseId:t?t.pendingId:0,asyncDep:null,asyncResolved:!1,isMounted:!1,isUnmounted:!1,isDeactivated:!1,bc:null,c:null,bm:null,m:null,bu:null,u:null,um:null,bum:null,da:null,a:null,rtg:null,rtc:null,ec:null,sp:null};return r.ctx={_:r},r.root=e?e.root:r,r.emit=L_.bind(null,r),i.ce&&i.ce(r),r}let pn=null;const oA=()=>pn||ni;let Cl,Su;{const i=Vl(),e=(t,n)=>{let s;return(s=i[t])||(s=i[t]=[]),s.push(n),r=>{s.length>1?s.forEach(o=>o(r)):s[0](r)}};Cl=e("__VUE_INSTANCE_SETTERS__",t=>pn=t),Su=e("__VUE_SSR_SETTERS__",t=>na=t)}const Sa=i=>{const e=pn;return Cl(i),i.scope.on(),()=>{i.scope.off(),Cl(e)}},lh=()=>{pn&&pn.scope.off(),Cl(null)};function A0(i){return i.vnode.shapeFlag&4}let na=!1;function aA(i,e=!1,t=!1){e&&Su(e);const{props:n,children:s}=i.vnode,r=A0(i);k_(i,n,r,e),W_(i,s,t||e);const o=r?lA(i,e):void 0;return e&&Su(!1),o}function lA(i,e){const t=i.type;i.accessCache=Object.create(null),i.proxy=new Proxy(i.ctx,C_);const{setup:n}=t;if(n){rs();const s=i.setupContext=n.length>1?uA(i):null,r=Sa(i),o=Aa(n,i,0,[i.props,s]),a=xm(o);if(os(),r(),(a||i.sp)&&!Ho(i)&&Qm(i),a){if(o.then(lh,lh),e)return o.then(l=>{ch(i,l)}).catch(l=>{Wl(l,i,0)});i.asyncDep=o}else ch(i,o)}else S0(i)}function ch(i,e,t){$e(e)?i.type.__ssrInlineRender?i.ssrRender=e:i.render=e:mt(e)&&(i.setupState=zm(e)),S0(i)}function S0(i,e,t){const n=i.type;i.render||(i.render=n.render||Di);{const s=Sa(i);rs();try{T_(i)}finally{os(),s()}}}const cA={get(i,e){return nn(i,"get",""),i[e]}};function uA(i){const e=t=>{i.exposed=t||{}};return{attrs:new Proxy(i.attrs,cA),slots:i.slots,emit:i.emit,expose:e}}function Yl(i){return i.exposed?i.exposeProxy||(i.exposeProxy=new Proxy(zm(Xx(i.exposed)),{get(e,t){if(t in e)return e[t];if(t in Vo)return Vo[t](i)},has(e,t){return t in e||t in Vo}})):i.proxy}function fA(i){return $e(i)&&"__vccOpts"in i}const v0=(i,e)=>$x(i,e,na),dA="3.5.28";let vu;const uh=typeof window<"u"&&window.trustedTypes;if(uh)try{vu=uh.createPolicy("vue",{createHTML:i=>i})}catch{}const y0=vu?i=>vu.createHTML(i):i=>i,hA="http://www.w3.org/2000/svg",pA="http://www.w3.org/1998/Math/MathML",Yi=typeof document<"u"?document:null,fh=Yi&&Yi.createElement("template"),mA={insert:(i,e,t)=>{e.insertBefore(i,t||null)},remove:i=>{const e=i.parentNode;e&&e.removeChild(i)},createElement:(i,e,t,n)=>{const s=e==="svg"?Yi.createElementNS(hA,i):e==="mathml"?Yi.createElementNS(pA,i):t?Yi.createElement(i,{is:t}):Yi.createElement(i);return i==="select"&&n&&n.multiple!=null&&s.setAttribute("multiple",n.multiple),s},createText:i=>Yi.createTextNode(i),createComment:i=>Yi.createComment(i),setText:(i,e)=>{i.nodeValue=e},setElementText:(i,e)=>{i.textContent=e},parentNode:i=>i.parentNode,nextSibling:i=>i.nextSibling,querySelector:i=>Yi.querySelector(i),setScopeId(i,e){i.setAttribute(e,"")},insertStaticContent(i,e,t,n,s,r){const o=t?t.previousSibling:e.lastChild;if(s&&(s===r||s.nextSibling))for(;e.insertBefore(s.cloneNode(!0),t),!(s===r||!(s=s.nextSibling)););else{fh.innerHTML=y0(n==="svg"?`<svg>${i}</svg>`:n==="mathml"?`<math>${i}</math>`:i);const a=fh.content;if(n==="svg"||n==="mathml"){const l=a.firstChild;for(;l.firstChild;)a.appendChild(l.firstChild);a.removeChild(l)}e.insertBefore(a,t)}return[o?o.nextSibling:e.firstChild,t?t.previousSibling:e.lastChild]}},gA=Symbol("_vtc");function xA(i,e,t){const n=i[gA];n&&(e=(e?[e,...n]:[...n]).join(" ")),e==null?i.removeAttribute("class"):t?i.setAttribute("class",e):i.className=e}const dh=Symbol("_vod"),_A=Symbol("_vsh"),AA=Symbol(""),SA=/(?:^|;)\s*display\s*:/;function vA(i,e,t){const n=i.style,s=Xt(t);let r=!1;if(t&&!s){if(e)if(Xt(e))for(const o of e.split(";")){const a=o.slice(0,o.indexOf(":")).trim();t[a]==null&&ul(n,a,"")}else for(const o in e)t[o]==null&&ul(n,o,"");for(const o in t)o==="display"&&(r=!0),ul(n,o,t[o])}else if(s){if(e!==t){const o=n[AA];o&&(t+=";"+o),n.cssText=t,r=SA.test(t)}}else e&&i.removeAttribute("style");dh in i&&(i[dh]=r?n.display:"",i[_A]&&(n.display="none"))}const hh=/\s*!important$/;function ul(i,e,t){if(Ke(t))t.forEach(n=>ul(i,e,n));else if(t==null&&(t=""),e.startsWith("--"))i.setProperty(e,t);else{const n=yA(i,e);hh.test(t)?i.setProperty(ks(n),t.replace(hh,""),"important"):i[n]=t}}const ph=["Webkit","Moz","ms"],mc={};function yA(i,e){const t=mc[e];if(t)return t;let n=Ls(e);if(n!=="filter"&&n in i)return mc[e]=n;n=Sm(n);for(let s=0;s<ph.length;s++){const r=ph[s]+n;if(r in i)return mc[e]=r}return e}const mh="http://www.w3.org/1999/xlink";function gh(i,e,t,n,s,r=yx(e)){n&&e.startsWith("xlink:")?t==null?i.removeAttributeNS(mh,e.slice(6,e.length)):i.setAttributeNS(mh,e,t):t==null||r&&!ym(t)?i.removeAttribute(e):i.setAttribute(e,r?"":Fi(t)?String(t):t)}function xh(i,e,t,n,s){if(e==="innerHTML"||e==="textContent"){t!=null&&(i[e]=e==="innerHTML"?y0(t):t);return}const r=i.tagName;if(e==="value"&&r!=="PROGRESS"&&!r.includes("-")){const a=r==="OPTION"?i.getAttribute("value")||"":i.value,l=t==null?i.type==="checkbox"?"on":"":String(t);(a!==l||!("_value"in i))&&(i.value=l),t==null&&i.removeAttribute(e),i._value=t;return}let o=!1;if(t===""||t==null){const a=typeof i[e];a==="boolean"?t=ym(t):t==null&&a==="string"?(t="",o=!0):a==="number"&&(t=0,o=!0)}try{i[e]=t}catch{}o&&i.removeAttribute(s||e)}function zr(i,e,t,n){i.addEventListener(e,t,n)}function bA(i,e,t,n){i.removeEventListener(e,t,n)}const _h=Symbol("_vei");function MA(i,e,t,n,s=null){const r=i[_h]||(i[_h]={}),o=r[e];if(n&&o)o.value=n;else{const[a,l]=CA(e);if(n){const c=r[e]=wA(n,s);zr(i,a,c,l)}else o&&(bA(i,a,o,l),r[e]=void 0)}}const Ah=/(?:Once|Passive|Capture)$/;function CA(i){let e;if(Ah.test(i)){e={};let n;for(;n=i.match(Ah);)i=i.slice(0,i.length-n[0].length),e[n[0].toLowerCase()]=!0}return[i[2]===":"?i.slice(3):ks(i.slice(2)),e]}let gc=0;const TA=Promise.resolve(),EA=()=>gc||(TA.then(()=>gc=0),gc=Date.now());function wA(i,e){const t=n=>{if(!n._vts)n._vts=Date.now();else if(n._vts<=t.attached)return;Li(RA(n,t.value),e,5,[n])};return t.value=i,t.attached=EA(),t}function RA(i,e){if(Ke(e)){const t=i.stopImmediatePropagation;return i.stopImmediatePropagation=()=>{t.call(i),i._stopped=!0},e.map(n=>s=>!s._stopped&&n&&n(s))}else return e}const Sh=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&i.charCodeAt(2)>96&&i.charCodeAt(2)<123,IA=(i,e,t,n,s,r)=>{const o=s==="svg";e==="class"?xA(i,n,o):e==="style"?vA(i,t,n):kl(e)?wf(e)||MA(i,e,t,n,r):(e[0]==="."?(e=e.slice(1),!0):e[0]==="^"?(e=e.slice(1),!1):DA(i,e,n,o))?(xh(i,e,n),!i.tagName.includes("-")&&(e==="value"||e==="checked"||e==="selected")&&gh(i,e,n,o,r,e!=="value")):i._isVueCE&&(/[A-Z]/.test(e)||!Xt(n))?xh(i,Ls(e),n,r,e):(e==="true-value"?i._trueValue=n:e==="false-value"&&(i._falseValue=n),gh(i,e,n,o))};function DA(i,e,t,n){if(n)return!!(e==="innerHTML"||e==="textContent"||e in i&&Sh(e)&&$e(t));if(e==="spellcheck"||e==="draggable"||e==="translate"||e==="autocorrect"||e==="sandbox"&&i.tagName==="IFRAME"||e==="form"||e==="list"&&i.tagName==="INPUT"||e==="type"&&i.tagName==="TEXTAREA")return!1;if(e==="width"||e==="height"){const s=i.tagName;if(s==="IMG"||s==="VIDEO"||s==="CANVAS"||s==="SOURCE")return!1}return Sh(e)&&Xt(t)?!1:e in i}const vh=i=>{const e=i.props["onUpdate:modelValue"]||!1;return Ke(e)?t=>al(e,t):e};function PA(i){i.target.composing=!0}function yh(i){const e=i.target;e.composing&&(e.composing=!1,e.dispatchEvent(new Event("input")))}const xc=Symbol("_assign");function bh(i,e,t){return e&&(i=i.trim()),t&&(i=Df(i)),i}const FA={created(i,{modifiers:{lazy:e,trim:t,number:n}},s){i[xc]=vh(s);const r=n||s.props&&s.props.type==="number";zr(i,e?"change":"input",o=>{o.target.composing||i[xc](bh(i.value,t,r))}),(t||r)&&zr(i,"change",()=>{i.value=bh(i.value,t,r)}),e||(zr(i,"compositionstart",PA),zr(i,"compositionend",yh),zr(i,"change",yh))},mounted(i,{value:e}){i.value=e??""},beforeUpdate(i,{value:e,oldValue:t,modifiers:{lazy:n,trim:s,number:r}},o){if(i[xc]=vh(o),i.composing)return;const a=(r||i.type==="number")&&!/^0\d/.test(i.value)?Df(i.value):i.value,l=e??"";a!==l&&(document.activeElement===i&&i.type!=="range"&&(n&&e===t||s&&i.value.trim()===l)||(i.value=l))}},LA={esc:"escape",space:" ",up:"arrow-up",left:"arrow-left",right:"arrow-right",down:"arrow-down",delete:"backspace"},BA=(i,e)=>{const t=i._withKeys||(i._withKeys={}),n=e.join(".");return t[n]||(t[n]=(s=>{if(!("key"in s))return;const r=ks(s.key);if(e.some(o=>o===r||LA[o]===r))return i(s)}))},UA=ln({patchProp:IA},mA);let Mh;function OA(){return Mh||(Mh=q_(UA))}const NA=((...i)=>{const e=OA().createApp(...i),{mount:t}=e;return e.mount=n=>{const s=kA(n);if(!s)return;const r=e._component;!$e(r)&&!r.render&&!r.template&&(r.template=s.innerHTML),s.nodeType===1&&(s.textContent="");const o=t(s,!1,zA(s));return s instanceof Element&&(s.removeAttribute("v-cloak"),s.setAttribute("data-v-app","")),o},e});function zA(i){if(i instanceof SVGElement)return"svg";if(typeof MathMLElement=="function"&&i instanceof MathMLElement)return"mathml"}function kA(i){return Xt(i)?document.querySelector(i):i}const Qf="181",Ar={ROTATE:0,DOLLY:1,PAN:2},Sr={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},HA=0,Ch=1,VA=2,b0=1,GA=2,qi=3,Bi=0,En=1,ti=2,is=0,Is=1,Th=2,Eh=3,wh=4,M0=5,nr=100,WA=101,XA=102,qA=103,QA=104,YA=200,KA=201,jA=202,$A=203,ia=204,sa=205,ZA=206,JA=207,eS=208,tS=209,nS=210,iS=211,sS=212,rS=213,oS=214,yu=0,bu=1,Mu=2,oo=3,Cu=4,Tu=5,Eu=6,wu=7,C0=0,aS=1,lS=2,Ds=0,cS=1,uS=2,fS=3,dS=4,hS=5,pS=6,mS=7,T0=300,ao=301,lo=302,Ru=303,Iu=304,Kl=306,Du=1e3,ns=1001,Pu=1002,qn=1003,gS=1004,Ia=1005,ii=1006,_c=1007,sr=1008,Ui=1009,E0=1010,w0=1011,ra=1012,Yf=1013,si=1014,mi=1015,pr=1016,Kf=1017,jf=1018,oa=1020,R0=35902,I0=35899,D0=1021,P0=1022,gn=1023,co=1026,aa=1027,F0=1028,jl=1029,$f=1030,Zf=1031,jr=1033,fl=33776,dl=33777,hl=33778,pl=33779,Fu=35840,Lu=35841,Bu=35842,Uu=35843,Ou=36196,Nu=37492,zu=37496,ku=37808,Hu=37809,Vu=37810,Gu=37811,Wu=37812,Xu=37813,qu=37814,Qu=37815,Yu=37816,Ku=37817,ju=37818,$u=37819,Zu=37820,Ju=37821,ef=36492,tf=36494,nf=36495,sf=36283,rf=36284,of=36285,af=36286,xS=3200,_S=3201,AS=0,SS=1,bs="",Jn="srgb",uo="srgb-linear",Tl="linear",ht="srgb",vr=7680,Rh=519,vS=512,yS=513,bS=514,L0=515,MS=516,CS=517,TS=518,ES=519,Ih=35044,wS=35048,Dh="300 es",Ei=2e3,El=2001;function B0(i){for(let e=i.length-1;e>=0;--e)if(i[e]>=65535)return!0;return!1}function wl(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function RS(){const i=wl("canvas");return i.style.display="block",i}const Ph={};function Fh(...i){const e="THREE."+i.shift();console.log(e,...i)}function je(...i){const e="THREE."+i.shift();console.warn(e,...i)}function zt(...i){const e="THREE."+i.shift();console.error(e,...i)}function la(...i){const e=i.join(" ");e in Ph||(Ph[e]=!0,je(...i))}function IS(i,e,t){return new Promise(function(n,s){function r(){switch(i.clientWaitSync(e,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:s();break;case i.TIMEOUT_EXPIRED:setTimeout(r,t);break;default:n()}}setTimeout(r,t)})}class mr{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[e]===void 0&&(n[e]=[]),n[e].indexOf(t)===-1&&n[e].push(t)}hasEventListener(e,t){const n=this._listeners;return n===void 0?!1:n[e]!==void 0&&n[e].indexOf(t)!==-1}removeEventListener(e,t){const n=this._listeners;if(n===void 0)return;const s=n[e];if(s!==void 0){const r=s.indexOf(t);r!==-1&&s.splice(r,1)}}dispatchEvent(e){const t=this._listeners;if(t===void 0)return;const n=t[e.type];if(n!==void 0){e.target=this;const s=n.slice(0);for(let r=0,o=s.length;r<o;r++)s[r].call(this,e);e.target=null}}}const en=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],ml=Math.PI/180,lf=180/Math.PI;function va(){const i=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(en[i&255]+en[i>>8&255]+en[i>>16&255]+en[i>>24&255]+"-"+en[e&255]+en[e>>8&255]+"-"+en[e>>16&15|64]+en[e>>24&255]+"-"+en[t&63|128]+en[t>>8&255]+"-"+en[t>>16&255]+en[t>>24&255]+en[n&255]+en[n>>8&255]+en[n>>16&255]+en[n>>24&255]).toLowerCase()}function Je(i,e,t){return Math.max(e,Math.min(t,i))}function DS(i,e){return(i%e+e)%e}function Ac(i,e,t){return(1-t)*i+t*e}function To(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return i/4294967295;case Uint16Array:return i/65535;case Uint8Array:return i/255;case Int32Array:return Math.max(i/2147483647,-1);case Int16Array:return Math.max(i/32767,-1);case Int8Array:return Math.max(i/127,-1);default:throw new Error("Invalid component type.")}}function yn(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return Math.round(i*4294967295);case Uint16Array:return Math.round(i*65535);case Uint8Array:return Math.round(i*255);case Int32Array:return Math.round(i*2147483647);case Int16Array:return Math.round(i*32767);case Int8Array:return Math.round(i*127);default:throw new Error("Invalid component type.")}}const U0={DEG2RAD:ml};class Ne{constructor(e=0,t=0){Ne.prototype.isVector2=!0,this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const t=this.x,n=this.y,s=e.elements;return this.x=s[0]*t+s[3]*n+s[6],this.y=s[1]*t+s[4]*n+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=Je(this.x,e.x,t.x),this.y=Je(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=Je(this.x,e,t),this.y=Je(this.y,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Je(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(Je(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y;return t*t+n*n}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){const n=Math.cos(t),s=Math.sin(t),r=this.x-e.x,o=this.y-e.y;return this.x=r*n-o*s+e.x,this.y=r*s+o*n+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class bt{constructor(e=0,t=0,n=0,s=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=n,this._w=s}static slerpFlat(e,t,n,s,r,o,a){let l=n[s+0],c=n[s+1],u=n[s+2],f=n[s+3],d=r[o+0],h=r[o+1],g=r[o+2],p=r[o+3];if(a<=0){e[t+0]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f;return}if(a>=1){e[t+0]=d,e[t+1]=h,e[t+2]=g,e[t+3]=p;return}if(f!==p||l!==d||c!==h||u!==g){let x=l*d+c*h+u*g+f*p;x<0&&(d=-d,h=-h,g=-g,p=-p,x=-x);let m=1-a;if(x<.9995){const _=Math.acos(x),A=Math.sin(_);m=Math.sin(m*_)/A,a=Math.sin(a*_)/A,l=l*m+d*a,c=c*m+h*a,u=u*m+g*a,f=f*m+p*a}else{l=l*m+d*a,c=c*m+h*a,u=u*m+g*a,f=f*m+p*a;const _=1/Math.sqrt(l*l+c*c+u*u+f*f);l*=_,c*=_,u*=_,f*=_}}e[t]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f}static multiplyQuaternionsFlat(e,t,n,s,r,o){const a=n[s],l=n[s+1],c=n[s+2],u=n[s+3],f=r[o],d=r[o+1],h=r[o+2],g=r[o+3];return e[t]=a*g+u*f+l*h-c*d,e[t+1]=l*g+u*d+c*f-a*h,e[t+2]=c*g+u*h+a*d-l*f,e[t+3]=u*g-a*f-l*d-c*h,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,n,s){return this._x=e,this._y=t,this._z=n,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){const n=e._x,s=e._y,r=e._z,o=e._order,a=Math.cos,l=Math.sin,c=a(n/2),u=a(s/2),f=a(r/2),d=l(n/2),h=l(s/2),g=l(r/2);switch(o){case"XYZ":this._x=d*u*f+c*h*g,this._y=c*h*f-d*u*g,this._z=c*u*g+d*h*f,this._w=c*u*f-d*h*g;break;case"YXZ":this._x=d*u*f+c*h*g,this._y=c*h*f-d*u*g,this._z=c*u*g-d*h*f,this._w=c*u*f+d*h*g;break;case"ZXY":this._x=d*u*f-c*h*g,this._y=c*h*f+d*u*g,this._z=c*u*g+d*h*f,this._w=c*u*f-d*h*g;break;case"ZYX":this._x=d*u*f-c*h*g,this._y=c*h*f+d*u*g,this._z=c*u*g-d*h*f,this._w=c*u*f+d*h*g;break;case"YZX":this._x=d*u*f+c*h*g,this._y=c*h*f+d*u*g,this._z=c*u*g-d*h*f,this._w=c*u*f-d*h*g;break;case"XZY":this._x=d*u*f-c*h*g,this._y=c*h*f-d*u*g,this._z=c*u*g+d*h*f,this._w=c*u*f+d*h*g;break;default:je("Quaternion: .setFromEuler() encountered an unknown order: "+o)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){const n=t/2,s=Math.sin(n);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(e){const t=e.elements,n=t[0],s=t[4],r=t[8],o=t[1],a=t[5],l=t[9],c=t[2],u=t[6],f=t[10],d=n+a+f;if(d>0){const h=.5/Math.sqrt(d+1);this._w=.25/h,this._x=(u-l)*h,this._y=(r-c)*h,this._z=(o-s)*h}else if(n>a&&n>f){const h=2*Math.sqrt(1+n-a-f);this._w=(u-l)/h,this._x=.25*h,this._y=(s+o)/h,this._z=(r+c)/h}else if(a>f){const h=2*Math.sqrt(1+a-n-f);this._w=(r-c)/h,this._x=(s+o)/h,this._y=.25*h,this._z=(l+u)/h}else{const h=2*Math.sqrt(1+f-n-a);this._w=(o-s)/h,this._x=(r+c)/h,this._y=(l+u)/h,this._z=.25*h}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let n=e.dot(t)+1;return n<1e-8?(n=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=n):(this._x=0,this._y=-e.z,this._z=e.y,this._w=n)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=n),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(Je(this.dot(e),-1,1)))}rotateTowards(e,t){const n=this.angleTo(e);if(n===0)return this;const s=Math.min(1,t/n);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){const n=e._x,s=e._y,r=e._z,o=e._w,a=t._x,l=t._y,c=t._z,u=t._w;return this._x=n*u+o*a+s*c-r*l,this._y=s*u+o*l+r*a-n*c,this._z=r*u+o*c+n*l-s*a,this._w=o*u-n*a-s*l-r*c,this._onChangeCallback(),this}slerp(e,t){if(t<=0)return this;if(t>=1)return this.copy(e);let n=e._x,s=e._y,r=e._z,o=e._w,a=this.dot(e);a<0&&(n=-n,s=-s,r=-r,o=-o,a=-a);let l=1-t;if(a<.9995){const c=Math.acos(a),u=Math.sin(c);l=Math.sin(l*c)/u,t=Math.sin(t*c)/u,this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this._onChangeCallback()}else this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this.normalize();return this}slerpQuaternions(e,t,n){return this.copy(e).slerp(t,n)}random(){const e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),n=Math.random(),s=Math.sqrt(1-n),r=Math.sqrt(n);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(t),r*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class L{constructor(e=0,t=0,n=0){L.prototype.isVector3=!0,this.x=e,this.y=t,this.z=n}set(e,t,n){return n===void 0&&(n=this.z),this.x=e,this.y=t,this.z=n,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(Lh.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(Lh.setFromAxisAngle(e,t))}applyMatrix3(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[3]*n+r[6]*s,this.y=r[1]*t+r[4]*n+r[7]*s,this.z=r[2]*t+r[5]*n+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=e.elements,o=1/(r[3]*t+r[7]*n+r[11]*s+r[15]);return this.x=(r[0]*t+r[4]*n+r[8]*s+r[12])*o,this.y=(r[1]*t+r[5]*n+r[9]*s+r[13])*o,this.z=(r[2]*t+r[6]*n+r[10]*s+r[14])*o,this}applyQuaternion(e){const t=this.x,n=this.y,s=this.z,r=e.x,o=e.y,a=e.z,l=e.w,c=2*(o*s-a*n),u=2*(a*t-r*s),f=2*(r*n-o*t);return this.x=t+l*c+o*f-a*u,this.y=n+l*u+a*c-r*f,this.z=s+l*f+r*u-o*c,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[4]*n+r[8]*s,this.y=r[1]*t+r[5]*n+r[9]*s,this.z=r[2]*t+r[6]*n+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=Je(this.x,e.x,t.x),this.y=Je(this.y,e.y,t.y),this.z=Je(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=Je(this.x,e,t),this.y=Je(this.y,e,t),this.z=Je(this.z,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Je(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){const n=e.x,s=e.y,r=e.z,o=t.x,a=t.y,l=t.z;return this.x=s*l-r*a,this.y=r*o-n*l,this.z=n*a-s*o,this}projectOnVector(e){const t=e.lengthSq();if(t===0)return this.set(0,0,0);const n=e.dot(this)/t;return this.copy(e).multiplyScalar(n)}projectOnPlane(e){return Sc.copy(this).projectOnVector(e),this.sub(Sc)}reflect(e){return this.sub(Sc.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(Je(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y,s=this.z-e.z;return t*t+n*n+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,n){const s=Math.sin(t)*e;return this.x=s*Math.sin(n),this.y=Math.cos(t)*e,this.z=s*Math.cos(n),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,n){return this.x=e*Math.sin(t),this.y=n,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){const t=this.setFromMatrixColumn(e,0).length(),n=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=n,this.z=s,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,t=Math.random()*2-1,n=Math.sqrt(1-t*t);return this.x=n*Math.cos(e),this.y=t,this.z=n*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const Sc=new L,Lh=new bt;class Qe{constructor(e,t,n,s,r,o,a,l,c){Qe.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c)}set(e,t,n,s,r,o,a,l,c){const u=this.elements;return u[0]=e,u[1]=s,u[2]=a,u[3]=t,u[4]=r,u[5]=l,u[6]=n,u[7]=o,u[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],this}extractBasis(e,t,n){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[3],l=n[6],c=n[1],u=n[4],f=n[7],d=n[2],h=n[5],g=n[8],p=s[0],x=s[3],m=s[6],_=s[1],A=s[4],S=s[7],v=s[2],b=s[5],M=s[8];return r[0]=o*p+a*_+l*v,r[3]=o*x+a*A+l*b,r[6]=o*m+a*S+l*M,r[1]=c*p+u*_+f*v,r[4]=c*x+u*A+f*b,r[7]=c*m+u*S+f*M,r[2]=d*p+h*_+g*v,r[5]=d*x+h*A+g*b,r[8]=d*m+h*S+g*M,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8];return t*o*u-t*a*c-n*r*u+n*a*l+s*r*c-s*o*l}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=u*o-a*c,d=a*l-u*r,h=c*r-o*l,g=t*f+n*d+s*h;if(g===0)return this.set(0,0,0,0,0,0,0,0,0);const p=1/g;return e[0]=f*p,e[1]=(s*c-u*n)*p,e[2]=(a*n-s*o)*p,e[3]=d*p,e[4]=(u*t-s*l)*p,e[5]=(s*r-a*t)*p,e[6]=h*p,e[7]=(n*l-c*t)*p,e[8]=(o*t-n*r)*p,this}transpose(){let e;const t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,n,s,r,o,a){const l=Math.cos(r),c=Math.sin(r);return this.set(n*l,n*c,-n*(l*o+c*a)+o+e,-s*c,s*l,-s*(-c*o+l*a)+a+t,0,0,1),this}scale(e,t){return this.premultiply(vc.makeScale(e,t)),this}rotate(e){return this.premultiply(vc.makeRotation(-e)),this}translate(e,t){return this.premultiply(vc.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,n,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<9;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<9;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e}clone(){return new this.constructor().fromArray(this.elements)}}const vc=new Qe,Bh=new Qe().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Uh=new Qe().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function PS(){const i={enabled:!0,workingColorSpace:uo,spaces:{},convert:function(s,r,o){return this.enabled===!1||r===o||!r||!o||(this.spaces[r].transfer===ht&&(s.r=ss(s.r),s.g=ss(s.g),s.b=ss(s.b)),this.spaces[r].primaries!==this.spaces[o].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[o].fromXYZ)),this.spaces[o].transfer===ht&&(s.r=$r(s.r),s.g=$r(s.g),s.b=$r(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===bs?Tl:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,o){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[o].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return la("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),i.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return la("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),i.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],n=[.3127,.329];return i.define({[uo]:{primaries:e,whitePoint:n,transfer:Tl,toXYZ:Bh,fromXYZ:Uh,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:Jn},outputColorSpaceConfig:{drawingBufferColorSpace:Jn}},[Jn]:{primaries:e,whitePoint:n,transfer:ht,toXYZ:Bh,fromXYZ:Uh,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:Jn}}}),i}const rt=PS();function ss(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function $r(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}let yr;class FS{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let n;if(e instanceof HTMLCanvasElement)n=e;else{yr===void 0&&(yr=wl("canvas")),yr.width=e.width,yr.height=e.height;const s=yr.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),n=yr}return n.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const t=wl("canvas");t.width=e.width,t.height=e.height;const n=t.getContext("2d");n.drawImage(e,0,0,e.width,e.height);const s=n.getImageData(0,0,e.width,e.height),r=s.data;for(let o=0;o<r.length;o++)r[o]=ss(r[o]/255)*255;return n.putImageData(s,0,0),t}else if(e.data){const t=e.data.slice(0);for(let n=0;n<t.length;n++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[n]=Math.floor(ss(t[n]/255)*255):t[n]=ss(t[n]);return{data:t,width:e.width,height:e.height}}else return je("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let LS=0;class Jf{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:LS++}),this.uuid=va(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):t instanceof VideoFrame?e.set(t.displayHeight,t.displayWidth,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const n={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let o=0,a=s.length;o<a;o++)s[o].isDataTexture?r.push(yc(s[o].image)):r.push(yc(s[o]))}else r=yc(s);n.url=r}return t||(e.images[this.uuid]=n),n}}function yc(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?FS.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(je("Texture: Unable to serialize Texture."),{})}let BS=0;const bc=new L;class xn extends mr{constructor(e=xn.DEFAULT_IMAGE,t=xn.DEFAULT_MAPPING,n=ns,s=ns,r=ii,o=sr,a=gn,l=Ui,c=xn.DEFAULT_ANISOTROPY,u=bs){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:BS++}),this.uuid=va(),this.name="",this.source=new Jf(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=n,this.wrapT=s,this.magFilter=r,this.minFilter=o,this.anisotropy=c,this.format=a,this.internalFormat=null,this.type=l,this.offset=new Ne(0,0),this.repeat=new Ne(1,1),this.center=new Ne(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Qe,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=u,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0}get width(){return this.source.getSize(bc).x}get height(){return this.source.getSize(bc).y}get depth(){return this.source.getSize(bc).z}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const t in e){const n=e[t];if(n===void 0){je(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){je(`Texture.setValues(): property '${t}' does not exist.`);continue}s&&n&&s.isVector2&&n.isVector2||s&&n&&s.isVector3&&n.isVector3||s&&n&&s.isMatrix3&&n.isMatrix3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const n={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),t||(e.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==T0)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case Du:e.x=e.x-Math.floor(e.x);break;case ns:e.x=e.x<0?0:1;break;case Pu:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case Du:e.y=e.y-Math.floor(e.y);break;case ns:e.y=e.y<0?0:1;break;case Pu:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}xn.DEFAULT_IMAGE=null;xn.DEFAULT_MAPPING=T0;xn.DEFAULT_ANISOTROPY=1;class Et{constructor(e=0,t=0,n=0,s=1){Et.prototype.isVector4=!0,this.x=e,this.y=t,this.z=n,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,n,s){return this.x=e,this.y=t,this.z=n,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=this.w,o=e.elements;return this.x=o[0]*t+o[4]*n+o[8]*s+o[12]*r,this.y=o[1]*t+o[5]*n+o[9]*s+o[13]*r,this.z=o[2]*t+o[6]*n+o[10]*s+o[14]*r,this.w=o[3]*t+o[7]*n+o[11]*s+o[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,n,s,r;const l=e.elements,c=l[0],u=l[4],f=l[8],d=l[1],h=l[5],g=l[9],p=l[2],x=l[6],m=l[10];if(Math.abs(u-d)<.01&&Math.abs(f-p)<.01&&Math.abs(g-x)<.01){if(Math.abs(u+d)<.1&&Math.abs(f+p)<.1&&Math.abs(g+x)<.1&&Math.abs(c+h+m-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;const A=(c+1)/2,S=(h+1)/2,v=(m+1)/2,b=(u+d)/4,M=(f+p)/4,E=(g+x)/4;return A>S&&A>v?A<.01?(n=0,s=.707106781,r=.707106781):(n=Math.sqrt(A),s=b/n,r=M/n):S>v?S<.01?(n=.707106781,s=0,r=.707106781):(s=Math.sqrt(S),n=b/s,r=E/s):v<.01?(n=.707106781,s=.707106781,r=0):(r=Math.sqrt(v),n=M/r,s=E/r),this.set(n,s,r,t),this}let _=Math.sqrt((x-g)*(x-g)+(f-p)*(f-p)+(d-u)*(d-u));return Math.abs(_)<.001&&(_=1),this.x=(x-g)/_,this.y=(f-p)/_,this.z=(d-u)/_,this.w=Math.acos((c+h+m-1)/2),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=Je(this.x,e.x,t.x),this.y=Je(this.y,e.y,t.y),this.z=Je(this.z,e.z,t.z),this.w=Je(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=Je(this.x,e,t),this.y=Je(this.y,e,t),this.z=Je(this.z,e,t),this.w=Je(this.w,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Je(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this.w=e.w+(t.w-e.w)*n,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class US extends mr{constructor(e=1,t=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:ii,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},n),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=n.depth,this.scissor=new Et(0,0,e,t),this.scissorTest=!1,this.viewport=new Et(0,0,e,t);const s={width:e,height:t,depth:n.depth},r=new xn(s);this.textures=[];const o=n.count;for(let a=0;a<o;a++)this.textures[a]=r.clone(),this.textures[a].isRenderTargetTexture=!0,this.textures[a].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview}_setTextureOptions(e={}){const t={minFilter:ii,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let n=0;n<this.textures.length;n++)this.textures[n].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,n=1){if(this.width!==e||this.height!==t||this.depth!==n){this.width=e,this.height=t,this.depth=n;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=t,this.textures[s].image.depth=n,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,n=e.textures.length;t<n;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;const s=Object.assign({},e.textures[t].image);this.textures[t].source=new Jf(s)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class Us extends US{constructor(e=1,t=1,n={}){super(e,t,n),this.isWebGLRenderTarget=!0}}class O0 extends xn{constructor(e=null,t=1,n=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=qn,this.minFilter=qn,this.wrapR=ns,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class OS extends xn{constructor(e=null,t=1,n=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=qn,this.minFilter=qn,this.wrapR=ns,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class wi{constructor(e=new L(1/0,1/0,1/0),t=new L(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t+=3)this.expandByPoint(ui.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,n=e.count;t<n;t++)this.expandByPoint(ui.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){const n=ui.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(n),this.max.copy(e).add(n),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);const n=e.geometry;if(n!==void 0){const r=n.getAttribute("position");if(t===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let o=0,a=r.count;o<a;o++)e.isMesh===!0?e.getVertexPosition(o,ui):ui.fromBufferAttribute(r,o),ui.applyMatrix4(e.matrixWorld),this.expandByPoint(ui);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),Da.copy(e.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),Da.copy(n.boundingBox)),Da.applyMatrix4(e.matrixWorld),this.union(Da)}const s=e.children;for(let r=0,o=s.length;r<o;r++)this.expandByObject(s[r],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,ui),ui.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,n;return e.normal.x>0?(t=e.normal.x*this.min.x,n=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,n=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,n+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,n+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,n+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,n+=e.normal.z*this.min.z),t<=-e.constant&&n>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(Eo),Pa.subVectors(this.max,Eo),br.subVectors(e.a,Eo),Mr.subVectors(e.b,Eo),Cr.subVectors(e.c,Eo),ds.subVectors(Mr,br),hs.subVectors(Cr,Mr),qs.subVectors(br,Cr);let t=[0,-ds.z,ds.y,0,-hs.z,hs.y,0,-qs.z,qs.y,ds.z,0,-ds.x,hs.z,0,-hs.x,qs.z,0,-qs.x,-ds.y,ds.x,0,-hs.y,hs.x,0,-qs.y,qs.x,0];return!Mc(t,br,Mr,Cr,Pa)||(t=[1,0,0,0,1,0,0,0,1],!Mc(t,br,Mr,Cr,Pa))?!1:(Fa.crossVectors(ds,hs),t=[Fa.x,Fa.y,Fa.z],Mc(t,br,Mr,Cr,Pa))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,ui).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(ui).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(zi[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),zi[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),zi[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),zi[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),zi[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),zi[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),zi[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),zi[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(zi),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const zi=[new L,new L,new L,new L,new L,new L,new L,new L],ui=new L,Da=new wi,br=new L,Mr=new L,Cr=new L,ds=new L,hs=new L,qs=new L,Eo=new L,Pa=new L,Fa=new L,Qs=new L;function Mc(i,e,t,n,s){for(let r=0,o=i.length-3;r<=o;r+=3){Qs.fromArray(i,r);const a=s.x*Math.abs(Qs.x)+s.y*Math.abs(Qs.y)+s.z*Math.abs(Qs.z),l=e.dot(Qs),c=t.dot(Qs),u=n.dot(Qs);if(Math.max(-Math.max(l,c,u),Math.min(l,c,u))>a)return!1}return!0}const NS=new wi,wo=new L,Cc=new L;class $l{constructor(e=new L,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){const n=this.center;t!==void 0?n.copy(t):NS.setFromPoints(e).getCenter(n);let s=0;for(let r=0,o=e.length;r<o;r++)s=Math.max(s,n.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){const n=this.center.distanceToSquared(e);return t.copy(e),n>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;wo.subVectors(e,this.center);const t=wo.lengthSq();if(t>this.radius*this.radius){const n=Math.sqrt(t),s=(n-this.radius)*.5;this.center.addScaledVector(wo,s/n),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Cc.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(wo.copy(e.center).add(Cc)),this.expandByPoint(wo.copy(e.center).sub(Cc))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}const ki=new L,Tc=new L,La=new L,ps=new L,Ec=new L,Ba=new L,wc=new L;let ed=class{constructor(e=new L,t=new L(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,ki)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);const n=t.dot(this.direction);return n<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const t=ki.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(ki.copy(this.origin).addScaledVector(this.direction,t),ki.distanceToSquared(e))}distanceSqToSegment(e,t,n,s){Tc.copy(e).add(t).multiplyScalar(.5),La.copy(t).sub(e).normalize(),ps.copy(this.origin).sub(Tc);const r=e.distanceTo(t)*.5,o=-this.direction.dot(La),a=ps.dot(this.direction),l=-ps.dot(La),c=ps.lengthSq(),u=Math.abs(1-o*o);let f,d,h,g;if(u>0)if(f=o*l-a,d=o*a-l,g=r*u,f>=0)if(d>=-g)if(d<=g){const p=1/u;f*=p,d*=p,h=f*(f+o*d+2*a)+d*(o*f+d+2*l)+c}else d=r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;else d=-r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;else d<=-g?(f=Math.max(0,-(-o*r+a)),d=f>0?-r:Math.min(Math.max(-r,-l),r),h=-f*f+d*(d+2*l)+c):d<=g?(f=0,d=Math.min(Math.max(-r,-l),r),h=d*(d+2*l)+c):(f=Math.max(0,-(o*r+a)),d=f>0?r:Math.min(Math.max(-r,-l),r),h=-f*f+d*(d+2*l)+c);else d=o>0?-r:r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;return n&&n.copy(this.origin).addScaledVector(this.direction,f),s&&s.copy(Tc).addScaledVector(La,d),h}intersectSphere(e,t){ki.subVectors(e.center,this.origin);const n=ki.dot(this.direction),s=ki.dot(ki)-n*n,r=e.radius*e.radius;if(s>r)return null;const o=Math.sqrt(r-s),a=n-o,l=n+o;return l<0?null:a<0?this.at(l,t):this.at(a,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(e.normal)+e.constant)/t;return n>=0?n:null}intersectPlane(e,t){const n=this.distanceToPlane(e);return n===null?null:this.at(n,t)}intersectsPlane(e){const t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let n,s,r,o,a,l;const c=1/this.direction.x,u=1/this.direction.y,f=1/this.direction.z,d=this.origin;return c>=0?(n=(e.min.x-d.x)*c,s=(e.max.x-d.x)*c):(n=(e.max.x-d.x)*c,s=(e.min.x-d.x)*c),u>=0?(r=(e.min.y-d.y)*u,o=(e.max.y-d.y)*u):(r=(e.max.y-d.y)*u,o=(e.min.y-d.y)*u),n>o||r>s||((r>n||isNaN(n))&&(n=r),(o<s||isNaN(s))&&(s=o),f>=0?(a=(e.min.z-d.z)*f,l=(e.max.z-d.z)*f):(a=(e.max.z-d.z)*f,l=(e.min.z-d.z)*f),n>l||a>s)||((a>n||n!==n)&&(n=a),(l<s||s!==s)&&(s=l),s<0)?null:this.at(n>=0?n:s,t)}intersectsBox(e){return this.intersectBox(e,ki)!==null}intersectTriangle(e,t,n,s,r){Ec.subVectors(t,e),Ba.subVectors(n,e),wc.crossVectors(Ec,Ba);let o=this.direction.dot(wc),a;if(o>0){if(s)return null;a=1}else if(o<0)a=-1,o=-o;else return null;ps.subVectors(this.origin,e);const l=a*this.direction.dot(Ba.crossVectors(ps,Ba));if(l<0)return null;const c=a*this.direction.dot(Ec.cross(ps));if(c<0||l+c>o)return null;const u=-a*ps.dot(wc);return u<0?null:this.at(u/o,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}};class qe{constructor(e,t,n,s,r,o,a,l,c,u,f,d,h,g,p,x){qe.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c,u,f,d,h,g,p,x)}set(e,t,n,s,r,o,a,l,c,u,f,d,h,g,p,x){const m=this.elements;return m[0]=e,m[4]=t,m[8]=n,m[12]=s,m[1]=r,m[5]=o,m[9]=a,m[13]=l,m[2]=c,m[6]=u,m[10]=f,m[14]=d,m[3]=h,m[7]=g,m[11]=p,m[15]=x,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new qe().fromArray(this.elements)}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],this}copyPosition(e){const t=this.elements,n=e.elements;return t[12]=n[12],t[13]=n[13],t[14]=n[14],this}setFromMatrix3(e){const t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,n){return e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this}makeBasis(e,t,n){return this.set(e.x,t.x,n.x,0,e.y,t.y,n.y,0,e.z,t.z,n.z,0,0,0,0,1),this}extractRotation(e){const t=this.elements,n=e.elements,s=1/Tr.setFromMatrixColumn(e,0).length(),r=1/Tr.setFromMatrixColumn(e,1).length(),o=1/Tr.setFromMatrixColumn(e,2).length();return t[0]=n[0]*s,t[1]=n[1]*s,t[2]=n[2]*s,t[3]=0,t[4]=n[4]*r,t[5]=n[5]*r,t[6]=n[6]*r,t[7]=0,t[8]=n[8]*o,t[9]=n[9]*o,t[10]=n[10]*o,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){const t=this.elements,n=e.x,s=e.y,r=e.z,o=Math.cos(n),a=Math.sin(n),l=Math.cos(s),c=Math.sin(s),u=Math.cos(r),f=Math.sin(r);if(e.order==="XYZ"){const d=o*u,h=o*f,g=a*u,p=a*f;t[0]=l*u,t[4]=-l*f,t[8]=c,t[1]=h+g*c,t[5]=d-p*c,t[9]=-a*l,t[2]=p-d*c,t[6]=g+h*c,t[10]=o*l}else if(e.order==="YXZ"){const d=l*u,h=l*f,g=c*u,p=c*f;t[0]=d+p*a,t[4]=g*a-h,t[8]=o*c,t[1]=o*f,t[5]=o*u,t[9]=-a,t[2]=h*a-g,t[6]=p+d*a,t[10]=o*l}else if(e.order==="ZXY"){const d=l*u,h=l*f,g=c*u,p=c*f;t[0]=d-p*a,t[4]=-o*f,t[8]=g+h*a,t[1]=h+g*a,t[5]=o*u,t[9]=p-d*a,t[2]=-o*c,t[6]=a,t[10]=o*l}else if(e.order==="ZYX"){const d=o*u,h=o*f,g=a*u,p=a*f;t[0]=l*u,t[4]=g*c-h,t[8]=d*c+p,t[1]=l*f,t[5]=p*c+d,t[9]=h*c-g,t[2]=-c,t[6]=a*l,t[10]=o*l}else if(e.order==="YZX"){const d=o*l,h=o*c,g=a*l,p=a*c;t[0]=l*u,t[4]=p-d*f,t[8]=g*f+h,t[1]=f,t[5]=o*u,t[9]=-a*u,t[2]=-c*u,t[6]=h*f+g,t[10]=d-p*f}else if(e.order==="XZY"){const d=o*l,h=o*c,g=a*l,p=a*c;t[0]=l*u,t[4]=-f,t[8]=c*u,t[1]=d*f+p,t[5]=o*u,t[9]=h*f-g,t[2]=g*f-h,t[6]=a*u,t[10]=p*f+d}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(zS,e,kS)}lookAt(e,t,n){const s=this.elements;return Un.subVectors(e,t),Un.lengthSq()===0&&(Un.z=1),Un.normalize(),ms.crossVectors(n,Un),ms.lengthSq()===0&&(Math.abs(n.z)===1?Un.x+=1e-4:Un.z+=1e-4,Un.normalize(),ms.crossVectors(n,Un)),ms.normalize(),Ua.crossVectors(Un,ms),s[0]=ms.x,s[4]=Ua.x,s[8]=Un.x,s[1]=ms.y,s[5]=Ua.y,s[9]=Un.y,s[2]=ms.z,s[6]=Ua.z,s[10]=Un.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[4],l=n[8],c=n[12],u=n[1],f=n[5],d=n[9],h=n[13],g=n[2],p=n[6],x=n[10],m=n[14],_=n[3],A=n[7],S=n[11],v=n[15],b=s[0],M=s[4],E=s[8],y=s[12],C=s[1],I=s[5],P=s[9],U=s[13],O=s[2],k=s[6],z=s[10],W=s[14],H=s[3],$=s[7],X=s[11],Z=s[15];return r[0]=o*b+a*C+l*O+c*H,r[4]=o*M+a*I+l*k+c*$,r[8]=o*E+a*P+l*z+c*X,r[12]=o*y+a*U+l*W+c*Z,r[1]=u*b+f*C+d*O+h*H,r[5]=u*M+f*I+d*k+h*$,r[9]=u*E+f*P+d*z+h*X,r[13]=u*y+f*U+d*W+h*Z,r[2]=g*b+p*C+x*O+m*H,r[6]=g*M+p*I+x*k+m*$,r[10]=g*E+p*P+x*z+m*X,r[14]=g*y+p*U+x*W+m*Z,r[3]=_*b+A*C+S*O+v*H,r[7]=_*M+A*I+S*k+v*$,r[11]=_*E+A*P+S*z+v*X,r[15]=_*y+A*U+S*W+v*Z,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[4],s=e[8],r=e[12],o=e[1],a=e[5],l=e[9],c=e[13],u=e[2],f=e[6],d=e[10],h=e[14],g=e[3],p=e[7],x=e[11],m=e[15];return g*(+r*l*f-s*c*f-r*a*d+n*c*d+s*a*h-n*l*h)+p*(+t*l*h-t*c*d+r*o*d-s*o*h+s*c*u-r*l*u)+x*(+t*c*f-t*a*h-r*o*f+n*o*h+r*a*u-n*c*u)+m*(-s*a*u-t*l*f+t*a*d+s*o*f-n*o*d+n*l*u)}transpose(){const e=this.elements;let t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,n){const s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=t,s[14]=n),this}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=e[9],d=e[10],h=e[11],g=e[12],p=e[13],x=e[14],m=e[15],_=f*x*c-p*d*c+p*l*h-a*x*h-f*l*m+a*d*m,A=g*d*c-u*x*c-g*l*h+o*x*h+u*l*m-o*d*m,S=u*p*c-g*f*c+g*a*h-o*p*h-u*a*m+o*f*m,v=g*f*l-u*p*l-g*a*d+o*p*d+u*a*x-o*f*x,b=t*_+n*A+s*S+r*v;if(b===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const M=1/b;return e[0]=_*M,e[1]=(p*d*r-f*x*r-p*s*h+n*x*h+f*s*m-n*d*m)*M,e[2]=(a*x*r-p*l*r+p*s*c-n*x*c-a*s*m+n*l*m)*M,e[3]=(f*l*r-a*d*r-f*s*c+n*d*c+a*s*h-n*l*h)*M,e[4]=A*M,e[5]=(u*x*r-g*d*r+g*s*h-t*x*h-u*s*m+t*d*m)*M,e[6]=(g*l*r-o*x*r-g*s*c+t*x*c+o*s*m-t*l*m)*M,e[7]=(o*d*r-u*l*r+u*s*c-t*d*c-o*s*h+t*l*h)*M,e[8]=S*M,e[9]=(g*f*r-u*p*r-g*n*h+t*p*h+u*n*m-t*f*m)*M,e[10]=(o*p*r-g*a*r+g*n*c-t*p*c-o*n*m+t*a*m)*M,e[11]=(u*a*r-o*f*r-u*n*c+t*f*c+o*n*h-t*a*h)*M,e[12]=v*M,e[13]=(u*p*s-g*f*s+g*n*d-t*p*d-u*n*x+t*f*x)*M,e[14]=(g*a*s-o*p*s-g*n*l+t*p*l+o*n*x-t*a*x)*M,e[15]=(o*f*s-u*a*s+u*n*l-t*f*l-o*n*d+t*a*d)*M,this}scale(e){const t=this.elements,n=e.x,s=e.y,r=e.z;return t[0]*=n,t[4]*=s,t[8]*=r,t[1]*=n,t[5]*=s,t[9]*=r,t[2]*=n,t[6]*=s,t[10]*=r,t[3]*=n,t[7]*=s,t[11]*=r,this}getMaxScaleOnAxis(){const e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],n=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,n,s))}makeTranslation(e,t,n){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,n,0,0,0,1),this}makeRotationX(e){const t=Math.cos(e),n=Math.sin(e);return this.set(1,0,0,0,0,t,-n,0,0,n,t,0,0,0,0,1),this}makeRotationY(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,0,n,0,0,1,0,0,-n,0,t,0,0,0,0,1),this}makeRotationZ(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,0,n,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){const n=Math.cos(t),s=Math.sin(t),r=1-n,o=e.x,a=e.y,l=e.z,c=r*o,u=r*a;return this.set(c*o+n,c*a-s*l,c*l+s*a,0,c*a+s*l,u*a+n,u*l-s*o,0,c*l-s*a,u*l+s*o,r*l*l+n,0,0,0,0,1),this}makeScale(e,t,n){return this.set(e,0,0,0,0,t,0,0,0,0,n,0,0,0,0,1),this}makeShear(e,t,n,s,r,o){return this.set(1,n,r,0,e,1,o,0,t,s,1,0,0,0,0,1),this}compose(e,t,n){const s=this.elements,r=t._x,o=t._y,a=t._z,l=t._w,c=r+r,u=o+o,f=a+a,d=r*c,h=r*u,g=r*f,p=o*u,x=o*f,m=a*f,_=l*c,A=l*u,S=l*f,v=n.x,b=n.y,M=n.z;return s[0]=(1-(p+m))*v,s[1]=(h+S)*v,s[2]=(g-A)*v,s[3]=0,s[4]=(h-S)*b,s[5]=(1-(d+m))*b,s[6]=(x+_)*b,s[7]=0,s[8]=(g+A)*M,s[9]=(x-_)*M,s[10]=(1-(d+p))*M,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,t,n){const s=this.elements;let r=Tr.set(s[0],s[1],s[2]).length();const o=Tr.set(s[4],s[5],s[6]).length(),a=Tr.set(s[8],s[9],s[10]).length();this.determinant()<0&&(r=-r),e.x=s[12],e.y=s[13],e.z=s[14],fi.copy(this);const c=1/r,u=1/o,f=1/a;return fi.elements[0]*=c,fi.elements[1]*=c,fi.elements[2]*=c,fi.elements[4]*=u,fi.elements[5]*=u,fi.elements[6]*=u,fi.elements[8]*=f,fi.elements[9]*=f,fi.elements[10]*=f,t.setFromRotationMatrix(fi),n.x=r,n.y=o,n.z=a,this}makePerspective(e,t,n,s,r,o,a=Ei,l=!1){const c=this.elements,u=2*r/(t-e),f=2*r/(n-s),d=(t+e)/(t-e),h=(n+s)/(n-s);let g,p;if(l)g=r/(o-r),p=o*r/(o-r);else if(a===Ei)g=-(o+r)/(o-r),p=-2*o*r/(o-r);else if(a===El)g=-o/(o-r),p=-o*r/(o-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=d,c[12]=0,c[1]=0,c[5]=f,c[9]=h,c[13]=0,c[2]=0,c[6]=0,c[10]=g,c[14]=p,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(e,t,n,s,r,o,a=Ei,l=!1){const c=this.elements,u=2/(t-e),f=2/(n-s),d=-(t+e)/(t-e),h=-(n+s)/(n-s);let g,p;if(l)g=1/(o-r),p=o/(o-r);else if(a===Ei)g=-2/(o-r),p=-(o+r)/(o-r);else if(a===El)g=-1/(o-r),p=-r/(o-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=0,c[12]=d,c[1]=0,c[5]=f,c[9]=0,c[13]=h,c[2]=0,c[6]=0,c[10]=g,c[14]=p,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<16;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<16;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e[t+9]=n[9],e[t+10]=n[10],e[t+11]=n[11],e[t+12]=n[12],e[t+13]=n[13],e[t+14]=n[14],e[t+15]=n[15],e}}const Tr=new L,fi=new qe,zS=new L(0,0,0),kS=new L(1,1,1),ms=new L,Ua=new L,Un=new L,Oh=new qe,Nh=new bt;class _i{constructor(e=0,t=0,n=0,s=_i.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=n,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,n,s=this._order){return this._x=e,this._y=t,this._z=n,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,n=!0){const s=e.elements,r=s[0],o=s[4],a=s[8],l=s[1],c=s[5],u=s[9],f=s[2],d=s[6],h=s[10];switch(t){case"XYZ":this._y=Math.asin(Je(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-u,h),this._z=Math.atan2(-o,r)):(this._x=Math.atan2(d,c),this._z=0);break;case"YXZ":this._x=Math.asin(-Je(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(a,h),this._z=Math.atan2(l,c)):(this._y=Math.atan2(-f,r),this._z=0);break;case"ZXY":this._x=Math.asin(Je(d,-1,1)),Math.abs(d)<.9999999?(this._y=Math.atan2(-f,h),this._z=Math.atan2(-o,c)):(this._y=0,this._z=Math.atan2(l,r));break;case"ZYX":this._y=Math.asin(-Je(f,-1,1)),Math.abs(f)<.9999999?(this._x=Math.atan2(d,h),this._z=Math.atan2(l,r)):(this._x=0,this._z=Math.atan2(-o,c));break;case"YZX":this._z=Math.asin(Je(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(-u,c),this._y=Math.atan2(-f,r)):(this._x=0,this._y=Math.atan2(a,h));break;case"XZY":this._z=Math.asin(-Je(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(d,c),this._y=Math.atan2(a,r)):(this._x=Math.atan2(-u,h),this._y=0);break;default:je("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,n===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,n){return Oh.makeRotationFromQuaternion(e),this.setFromRotationMatrix(Oh,t,n)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return Nh.setFromEuler(this),this.setFromQuaternion(Nh,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}_i.DEFAULT_ORDER="XYZ";class N0{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let HS=0;const zh=new L,Er=new bt,Hi=new qe,Oa=new L,Ro=new L,VS=new L,GS=new bt,kh=new L(1,0,0),Hh=new L(0,1,0),Vh=new L(0,0,1),Gh={type:"added"},WS={type:"removed"},wr={type:"childadded",child:null},Rc={type:"childremoved",child:null};class Wt extends mr{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:HS++}),this.uuid=va(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=Wt.DEFAULT_UP.clone();const e=new L,t=new _i,n=new bt,s=new L(1,1,1);function r(){n.setFromEuler(t,!1)}function o(){t.setFromQuaternion(n,void 0,!1)}t._onChange(r),n._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new qe},normalMatrix:{value:new Qe}}),this.matrix=new qe,this.matrixWorld=new qe,this.matrixAutoUpdate=Wt.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=Wt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new N0,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return Er.setFromAxisAngle(e,t),this.quaternion.multiply(Er),this}rotateOnWorldAxis(e,t){return Er.setFromAxisAngle(e,t),this.quaternion.premultiply(Er),this}rotateX(e){return this.rotateOnAxis(kh,e)}rotateY(e){return this.rotateOnAxis(Hh,e)}rotateZ(e){return this.rotateOnAxis(Vh,e)}translateOnAxis(e,t){return zh.copy(e).applyQuaternion(this.quaternion),this.position.add(zh.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(kh,e)}translateY(e){return this.translateOnAxis(Hh,e)}translateZ(e){return this.translateOnAxis(Vh,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Hi.copy(this.matrixWorld).invert())}lookAt(e,t,n){e.isVector3?Oa.copy(e):Oa.set(e,t,n);const s=this.parent;this.updateWorldMatrix(!0,!1),Ro.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Hi.lookAt(Ro,Oa,this.up):Hi.lookAt(Oa,Ro,this.up),this.quaternion.setFromRotationMatrix(Hi),s&&(Hi.extractRotation(s.matrixWorld),Er.setFromRotationMatrix(Hi),this.quaternion.premultiply(Er.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(zt("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(Gh),wr.child=e,this.dispatchEvent(wr),wr.child=null):zt("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(WS),Rc.child=e,this.dispatchEvent(Rc),Rc.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Hi.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Hi.multiply(e.parent.matrixWorld)),e.applyMatrix4(Hi),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(Gh),wr.child=e,this.dispatchEvent(wr),wr.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let n=0,s=this.children.length;n<s;n++){const o=this.children[n].getObjectByProperty(e,t);if(o!==void 0)return o}}getObjectsByProperty(e,t,n=[]){this[e]===t&&n.push(this);const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].getObjectsByProperty(e,t,n);return n}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Ro,e,VS),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Ro,GS,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverseVisible(e)}traverseAncestors(e){const t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].updateMatrixWorld(e)}updateWorldMatrix(e,t){const n=this.parent;if(e===!0&&n!==null&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),t===!0){const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].updateWorldMatrix(!1,!0)}}toJSON(e){const t=e===void 0||typeof e=="string",n={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(a=>({...a,boundingBox:a.boundingBox?a.boundingBox.toJSON():void 0,boundingSphere:a.boundingSphere?a.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(a=>({...a})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(a,l){return a[l.uuid]===void 0&&(a[l.uuid]=l.toJSON(e)),l.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);const a=this.geometry.parameters;if(a!==void 0&&a.shapes!==void 0){const l=a.shapes;if(Array.isArray(l))for(let c=0,u=l.length;c<u;c++){const f=l[c];r(e.shapes,f)}else r(e.shapes,l)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const a=[];for(let l=0,c=this.material.length;l<c;l++)a.push(r(e.materials,this.material[l]));s.material=a}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let a=0;a<this.children.length;a++)s.children.push(this.children[a].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let a=0;a<this.animations.length;a++){const l=this.animations[a];s.animations.push(r(e.animations,l))}}if(t){const a=o(e.geometries),l=o(e.materials),c=o(e.textures),u=o(e.images),f=o(e.shapes),d=o(e.skeletons),h=o(e.animations),g=o(e.nodes);a.length>0&&(n.geometries=a),l.length>0&&(n.materials=l),c.length>0&&(n.textures=c),u.length>0&&(n.images=u),f.length>0&&(n.shapes=f),d.length>0&&(n.skeletons=d),h.length>0&&(n.animations=h),g.length>0&&(n.nodes=g)}return n.object=s,n;function o(a){const l=[];for(const c in a){const u=a[c];delete u.metadata,l.push(u)}return l}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let n=0;n<e.children.length;n++){const s=e.children[n];this.add(s.clone())}return this}}Wt.DEFAULT_UP=new L(0,1,0);Wt.DEFAULT_MATRIX_AUTO_UPDATE=!0;Wt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const di=new L,Vi=new L,Ic=new L,Gi=new L,Rr=new L,Ir=new L,Wh=new L,Dc=new L,Pc=new L,Fc=new L,Lc=new Et,Bc=new Et,Uc=new Et;class pi{constructor(e=new L,t=new L,n=new L){this.a=e,this.b=t,this.c=n}static getNormal(e,t,n,s){s.subVectors(n,t),di.subVectors(e,t),s.cross(di);const r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,t,n,s,r){di.subVectors(s,t),Vi.subVectors(n,t),Ic.subVectors(e,t);const o=di.dot(di),a=di.dot(Vi),l=di.dot(Ic),c=Vi.dot(Vi),u=Vi.dot(Ic),f=o*c-a*a;if(f===0)return r.set(0,0,0),null;const d=1/f,h=(c*l-a*u)*d,g=(o*u-a*l)*d;return r.set(1-h-g,g,h)}static containsPoint(e,t,n,s){return this.getBarycoord(e,t,n,s,Gi)===null?!1:Gi.x>=0&&Gi.y>=0&&Gi.x+Gi.y<=1}static getInterpolation(e,t,n,s,r,o,a,l){return this.getBarycoord(e,t,n,s,Gi)===null?(l.x=0,l.y=0,"z"in l&&(l.z=0),"w"in l&&(l.w=0),null):(l.setScalar(0),l.addScaledVector(r,Gi.x),l.addScaledVector(o,Gi.y),l.addScaledVector(a,Gi.z),l)}static getInterpolatedAttribute(e,t,n,s,r,o){return Lc.setScalar(0),Bc.setScalar(0),Uc.setScalar(0),Lc.fromBufferAttribute(e,t),Bc.fromBufferAttribute(e,n),Uc.fromBufferAttribute(e,s),o.setScalar(0),o.addScaledVector(Lc,r.x),o.addScaledVector(Bc,r.y),o.addScaledVector(Uc,r.z),o}static isFrontFacing(e,t,n,s){return di.subVectors(n,t),Vi.subVectors(e,t),di.cross(Vi).dot(s)<0}set(e,t,n){return this.a.copy(e),this.b.copy(t),this.c.copy(n),this}setFromPointsAndIndices(e,t,n,s){return this.a.copy(e[t]),this.b.copy(e[n]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,t,n,s){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,n),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return di.subVectors(this.c,this.b),Vi.subVectors(this.a,this.b),di.cross(Vi).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return pi.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return pi.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,n,s,r){return pi.getInterpolation(e,this.a,this.b,this.c,t,n,s,r)}containsPoint(e){return pi.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return pi.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){const n=this.a,s=this.b,r=this.c;let o,a;Rr.subVectors(s,n),Ir.subVectors(r,n),Dc.subVectors(e,n);const l=Rr.dot(Dc),c=Ir.dot(Dc);if(l<=0&&c<=0)return t.copy(n);Pc.subVectors(e,s);const u=Rr.dot(Pc),f=Ir.dot(Pc);if(u>=0&&f<=u)return t.copy(s);const d=l*f-u*c;if(d<=0&&l>=0&&u<=0)return o=l/(l-u),t.copy(n).addScaledVector(Rr,o);Fc.subVectors(e,r);const h=Rr.dot(Fc),g=Ir.dot(Fc);if(g>=0&&h<=g)return t.copy(r);const p=h*c-l*g;if(p<=0&&c>=0&&g<=0)return a=c/(c-g),t.copy(n).addScaledVector(Ir,a);const x=u*g-h*f;if(x<=0&&f-u>=0&&h-g>=0)return Wh.subVectors(r,s),a=(f-u)/(f-u+(h-g)),t.copy(s).addScaledVector(Wh,a);const m=1/(x+p+d);return o=p*m,a=d*m,t.copy(n).addScaledVector(Rr,o).addScaledVector(Ir,a)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}const z0={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},gs={h:0,s:0,l:0},Na={h:0,s:0,l:0};function Oc(i,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?i+(e-i)*6*t:t<1/2?e:t<2/3?i+(e-i)*6*(2/3-t):i}class nt{constructor(e,t,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,n)}set(e,t,n){if(t===void 0&&n===void 0){const s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,t,n);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=Jn){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,rt.colorSpaceToWorking(this,t),this}setRGB(e,t,n,s=rt.workingColorSpace){return this.r=e,this.g=t,this.b=n,rt.colorSpaceToWorking(this,s),this}setHSL(e,t,n,s=rt.workingColorSpace){if(e=DS(e,1),t=Je(t,0,1),n=Je(n,0,1),t===0)this.r=this.g=this.b=n;else{const r=n<=.5?n*(1+t):n+t-n*t,o=2*n-r;this.r=Oc(o,r,e+1/3),this.g=Oc(o,r,e),this.b=Oc(o,r,e-1/3)}return rt.colorSpaceToWorking(this,s),this}setStyle(e,t=Jn){function n(r){r!==void 0&&parseFloat(r)<1&&je("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r;const o=s[1],a=s[2];switch(o){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,t);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,t);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,t);break;default:je("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){const r=s[1],o=r.length;if(o===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,t);if(o===6)return this.setHex(parseInt(r,16),t);je("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=Jn){const n=z0[e.toLowerCase()];return n!==void 0?this.setHex(n,t):je("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=ss(e.r),this.g=ss(e.g),this.b=ss(e.b),this}copyLinearToSRGB(e){return this.r=$r(e.r),this.g=$r(e.g),this.b=$r(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Jn){return rt.workingToColorSpace(tn.copy(this),e),Math.round(Je(tn.r*255,0,255))*65536+Math.round(Je(tn.g*255,0,255))*256+Math.round(Je(tn.b*255,0,255))}getHexString(e=Jn){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=rt.workingColorSpace){rt.workingToColorSpace(tn.copy(this),t);const n=tn.r,s=tn.g,r=tn.b,o=Math.max(n,s,r),a=Math.min(n,s,r);let l,c;const u=(a+o)/2;if(a===o)l=0,c=0;else{const f=o-a;switch(c=u<=.5?f/(o+a):f/(2-o-a),o){case n:l=(s-r)/f+(s<r?6:0);break;case s:l=(r-n)/f+2;break;case r:l=(n-s)/f+4;break}l/=6}return e.h=l,e.s=c,e.l=u,e}getRGB(e,t=rt.workingColorSpace){return rt.workingToColorSpace(tn.copy(this),t),e.r=tn.r,e.g=tn.g,e.b=tn.b,e}getStyle(e=Jn){rt.workingToColorSpace(tn.copy(this),e);const t=tn.r,n=tn.g,s=tn.b;return e!==Jn?`color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(n*255)},${Math.round(s*255)})`}offsetHSL(e,t,n){return this.getHSL(gs),this.setHSL(gs.h+e,gs.s+t,gs.l+n)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,n){return this.r=e.r+(t.r-e.r)*n,this.g=e.g+(t.g-e.g)*n,this.b=e.b+(t.b-e.b)*n,this}lerpHSL(e,t){this.getHSL(gs),e.getHSL(Na);const n=Ac(gs.h,Na.h,t),s=Ac(gs.s,Na.s,t),r=Ac(gs.l,Na.l,t);return this.setHSL(n,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const t=this.r,n=this.g,s=this.b,r=e.elements;return this.r=r[0]*t+r[3]*n+r[6]*s,this.g=r[1]*t+r[4]*n+r[7]*s,this.b=r[2]*t+r[5]*n+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const tn=new nt;nt.NAMES=z0;let XS=0;class ya extends mr{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:XS++}),this.uuid=va(),this.name="",this.type="Material",this.blending=Is,this.side=Bi,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=ia,this.blendDst=sa,this.blendEquation=nr,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new nt(0,0,0),this.blendAlpha=0,this.depthFunc=oo,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Rh,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=vr,this.stencilZFail=vr,this.stencilZPass=vr,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const t in e){const n=e[t];if(n===void 0){je(`Material: parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){je(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(n):s&&s.isVector3&&n&&n.isVector3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});const n={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(e).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(e).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(e).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(e).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(e).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==Is&&(n.blending=this.blending),this.side!==Bi&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==ia&&(n.blendSrc=this.blendSrc),this.blendDst!==sa&&(n.blendDst=this.blendDst),this.blendEquation!==nr&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==oo&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==Rh&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==vr&&(n.stencilFail=this.stencilFail),this.stencilZFail!==vr&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==vr&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function s(r){const o=[];for(const a in r){const l=r[a];delete l.metadata,o.push(l)}return o}if(t){const r=s(e.textures),o=s(e.images);r.length>0&&(n.textures=r),o.length>0&&(n.images=o)}return n}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const t=e.clippingPlanes;let n=null;if(t!==null){const s=t.length;n=new Array(s);for(let r=0;r!==s;++r)n[r]=t[r].clone()}return this.clippingPlanes=n,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}class hr extends ya{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new nt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new _i,this.combine=C0,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const ts=qS();function qS(){const i=new ArrayBuffer(4),e=new Float32Array(i),t=new Uint32Array(i),n=new Uint32Array(512),s=new Uint32Array(512);for(let l=0;l<256;++l){const c=l-127;c<-27?(n[l]=0,n[l|256]=32768,s[l]=24,s[l|256]=24):c<-14?(n[l]=1024>>-c-14,n[l|256]=1024>>-c-14|32768,s[l]=-c-1,s[l|256]=-c-1):c<=15?(n[l]=c+15<<10,n[l|256]=c+15<<10|32768,s[l]=13,s[l|256]=13):c<128?(n[l]=31744,n[l|256]=64512,s[l]=24,s[l|256]=24):(n[l]=31744,n[l|256]=64512,s[l]=13,s[l|256]=13)}const r=new Uint32Array(2048),o=new Uint32Array(64),a=new Uint32Array(64);for(let l=1;l<1024;++l){let c=l<<13,u=0;for(;(c&8388608)===0;)c<<=1,u-=8388608;c&=-8388609,u+=947912704,r[l]=c|u}for(let l=1024;l<2048;++l)r[l]=939524096+(l-1024<<13);for(let l=1;l<31;++l)o[l]=l<<23;o[31]=1199570944,o[32]=2147483648;for(let l=33;l<63;++l)o[l]=2147483648+(l-32<<23);o[63]=3347054592;for(let l=1;l<64;++l)l!==32&&(a[l]=1024);return{floatView:e,uint32View:t,baseTable:n,shiftTable:s,mantissaTable:r,exponentTable:o,offsetTable:a}}function QS(i){Math.abs(i)>65504&&je("DataUtils.toHalfFloat(): Value out of range."),i=Je(i,-65504,65504),ts.floatView[0]=i;const e=ts.uint32View[0],t=e>>23&511;return ts.baseTable[t]+((e&8388607)>>ts.shiftTable[t])}function YS(i){const e=i>>10;return ts.uint32View[0]=ts.mantissaTable[ts.offsetTable[e]+(i&1023)]+ts.exponentTable[e],ts.floatView[0]}class ca{static toHalfFloat(e){return QS(e)}static fromHalfFloat(e){return YS(e)}}const kt=new L,za=new Ne;let KS=0;class li{constructor(e,t,n=!1){if(Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:KS++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=n,this.usage=Ih,this.updateRanges=[],this.gpuType=mi,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,n){e*=this.itemSize,n*=t.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=t.array[n+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,n=this.count;t<n;t++)za.fromBufferAttribute(this,t),za.applyMatrix3(e),this.setXY(t,za.x,za.y);else if(this.itemSize===3)for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.applyMatrix3(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}applyMatrix4(e){for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.applyMatrix4(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.applyNormalMatrix(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.transformDirection(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let n=this.array[e*this.itemSize+t];return this.normalized&&(n=To(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=yn(n,this.array)),this.array[e*this.itemSize+t]=n,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=To(t,this.array)),t}setX(e,t){return this.normalized&&(t=yn(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=To(t,this.array)),t}setY(e,t){return this.normalized&&(t=yn(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=To(t,this.array)),t}setZ(e,t){return this.normalized&&(t=yn(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=To(t,this.array)),t}setW(e,t){return this.normalized&&(t=yn(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,n){return e*=this.itemSize,this.normalized&&(t=yn(t,this.array),n=yn(n,this.array)),this.array[e+0]=t,this.array[e+1]=n,this}setXYZ(e,t,n,s){return e*=this.itemSize,this.normalized&&(t=yn(t,this.array),n=yn(n,this.array),s=yn(s,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this}setXYZW(e,t,n,s,r){return e*=this.itemSize,this.normalized&&(t=yn(t,this.array),n=yn(n,this.array),s=yn(s,this.array),r=yn(r,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==Ih&&(e.usage=this.usage),e}}class k0 extends li{constructor(e,t,n){super(new Uint16Array(e),t,n)}}class H0 extends li{constructor(e,t,n){super(new Uint32Array(e),t,n)}}class on extends li{constructor(e,t,n){super(new Float32Array(e),t,n)}}let jS=0;const $n=new qe,Nc=new Wt,Dr=new L,On=new wi,Io=new wi,Kt=new L;class An extends mr{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:jS++}),this.uuid=va(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(B0(e)?H0:k0)(e,1):this.index=e,this}setIndirect(e){return this.indirect=e,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,n=0){this.groups.push({start:e,count:t,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){const t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const r=new Qe().getNormalMatrix(e);n.applyNormalMatrix(r),n.needsUpdate=!0}const s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return $n.makeRotationFromQuaternion(e),this.applyMatrix4($n),this}rotateX(e){return $n.makeRotationX(e),this.applyMatrix4($n),this}rotateY(e){return $n.makeRotationY(e),this.applyMatrix4($n),this}rotateZ(e){return $n.makeRotationZ(e),this.applyMatrix4($n),this}translate(e,t,n){return $n.makeTranslation(e,t,n),this.applyMatrix4($n),this}scale(e,t,n){return $n.makeScale(e,t,n),this.applyMatrix4($n),this}lookAt(e){return Nc.lookAt(e),Nc.updateMatrix(),this.applyMatrix4(Nc.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Dr).negate(),this.translate(Dr.x,Dr.y,Dr.z),this}setFromPoints(e){const t=this.getAttribute("position");if(t===void 0){const n=[];for(let s=0,r=e.length;s<r;s++){const o=e[s];n.push(o.x,o.y,o.z||0)}this.setAttribute("position",new on(n,3))}else{const n=Math.min(e.length,t.count);for(let s=0;s<n;s++){const r=e[s];t.setXYZ(s,r.x,r.y,r.z||0)}e.length>t.count&&je("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new wi);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){zt("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new L(-1/0,-1/0,-1/0),new L(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let n=0,s=t.length;n<s;n++){const r=t[n];On.setFromBufferAttribute(r),this.morphTargetsRelative?(Kt.addVectors(this.boundingBox.min,On.min),this.boundingBox.expandByPoint(Kt),Kt.addVectors(this.boundingBox.max,On.max),this.boundingBox.expandByPoint(Kt)):(this.boundingBox.expandByPoint(On.min),this.boundingBox.expandByPoint(On.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&zt('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new $l);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){zt("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new L,1/0);return}if(e){const n=this.boundingSphere.center;if(On.setFromBufferAttribute(e),t)for(let r=0,o=t.length;r<o;r++){const a=t[r];Io.setFromBufferAttribute(a),this.morphTargetsRelative?(Kt.addVectors(On.min,Io.min),On.expandByPoint(Kt),Kt.addVectors(On.max,Io.max),On.expandByPoint(Kt)):(On.expandByPoint(Io.min),On.expandByPoint(Io.max))}On.getCenter(n);let s=0;for(let r=0,o=e.count;r<o;r++)Kt.fromBufferAttribute(e,r),s=Math.max(s,n.distanceToSquared(Kt));if(t)for(let r=0,o=t.length;r<o;r++){const a=t[r],l=this.morphTargetsRelative;for(let c=0,u=a.count;c<u;c++)Kt.fromBufferAttribute(a,c),l&&(Dr.fromBufferAttribute(e,c),Kt.add(Dr)),s=Math.max(s,n.distanceToSquared(Kt))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&zt('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){zt("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=t.position,s=t.normal,r=t.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new li(new Float32Array(4*n.count),4));const o=this.getAttribute("tangent"),a=[],l=[];for(let E=0;E<n.count;E++)a[E]=new L,l[E]=new L;const c=new L,u=new L,f=new L,d=new Ne,h=new Ne,g=new Ne,p=new L,x=new L;function m(E,y,C){c.fromBufferAttribute(n,E),u.fromBufferAttribute(n,y),f.fromBufferAttribute(n,C),d.fromBufferAttribute(r,E),h.fromBufferAttribute(r,y),g.fromBufferAttribute(r,C),u.sub(c),f.sub(c),h.sub(d),g.sub(d);const I=1/(h.x*g.y-g.x*h.y);isFinite(I)&&(p.copy(u).multiplyScalar(g.y).addScaledVector(f,-h.y).multiplyScalar(I),x.copy(f).multiplyScalar(h.x).addScaledVector(u,-g.x).multiplyScalar(I),a[E].add(p),a[y].add(p),a[C].add(p),l[E].add(x),l[y].add(x),l[C].add(x))}let _=this.groups;_.length===0&&(_=[{start:0,count:e.count}]);for(let E=0,y=_.length;E<y;++E){const C=_[E],I=C.start,P=C.count;for(let U=I,O=I+P;U<O;U+=3)m(e.getX(U+0),e.getX(U+1),e.getX(U+2))}const A=new L,S=new L,v=new L,b=new L;function M(E){v.fromBufferAttribute(s,E),b.copy(v);const y=a[E];A.copy(y),A.sub(v.multiplyScalar(v.dot(y))).normalize(),S.crossVectors(b,y);const I=S.dot(l[E])<0?-1:1;o.setXYZW(E,A.x,A.y,A.z,I)}for(let E=0,y=_.length;E<y;++E){const C=_[E],I=C.start,P=C.count;for(let U=I,O=I+P;U<O;U+=3)M(e.getX(U+0)),M(e.getX(U+1)),M(e.getX(U+2))}}computeVertexNormals(){const e=this.index,t=this.getAttribute("position");if(t!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new li(new Float32Array(t.count*3),3),this.setAttribute("normal",n);else for(let d=0,h=n.count;d<h;d++)n.setXYZ(d,0,0,0);const s=new L,r=new L,o=new L,a=new L,l=new L,c=new L,u=new L,f=new L;if(e)for(let d=0,h=e.count;d<h;d+=3){const g=e.getX(d+0),p=e.getX(d+1),x=e.getX(d+2);s.fromBufferAttribute(t,g),r.fromBufferAttribute(t,p),o.fromBufferAttribute(t,x),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),a.fromBufferAttribute(n,g),l.fromBufferAttribute(n,p),c.fromBufferAttribute(n,x),a.add(u),l.add(u),c.add(u),n.setXYZ(g,a.x,a.y,a.z),n.setXYZ(p,l.x,l.y,l.z),n.setXYZ(x,c.x,c.y,c.z)}else for(let d=0,h=t.count;d<h;d+=3)s.fromBufferAttribute(t,d+0),r.fromBufferAttribute(t,d+1),o.fromBufferAttribute(t,d+2),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),n.setXYZ(d+0,u.x,u.y,u.z),n.setXYZ(d+1,u.x,u.y,u.z),n.setXYZ(d+2,u.x,u.y,u.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let t=0,n=e.count;t<n;t++)Kt.fromBufferAttribute(e,t),Kt.normalize(),e.setXYZ(t,Kt.x,Kt.y,Kt.z)}toNonIndexed(){function e(a,l){const c=a.array,u=a.itemSize,f=a.normalized,d=new c.constructor(l.length*u);let h=0,g=0;for(let p=0,x=l.length;p<x;p++){a.isInterleavedBufferAttribute?h=l[p]*a.data.stride+a.offset:h=l[p]*u;for(let m=0;m<u;m++)d[g++]=c[h++]}return new li(d,u,f)}if(this.index===null)return je("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const t=new An,n=this.index.array,s=this.attributes;for(const a in s){const l=s[a],c=e(l,n);t.setAttribute(a,c)}const r=this.morphAttributes;for(const a in r){const l=[],c=r[a];for(let u=0,f=c.length;u<f;u++){const d=c[u],h=e(d,n);l.push(h)}t.morphAttributes[a]=l}t.morphTargetsRelative=this.morphTargetsRelative;const o=this.groups;for(let a=0,l=o.length;a<l;a++){const c=o[a];t.addGroup(c.start,c.count,c.materialIndex)}return t}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const l=this.parameters;for(const c in l)l[c]!==void 0&&(e[c]=l[c]);return e}e.data={attributes:{}};const t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});const n=this.attributes;for(const l in n){const c=n[l];e.data.attributes[l]=c.toJSON(e.data)}const s={};let r=!1;for(const l in this.morphAttributes){const c=this.morphAttributes[l],u=[];for(let f=0,d=c.length;f<d;f++){const h=c[f];u.push(h.toJSON(e.data))}u.length>0&&(s[l]=u,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);const o=this.groups;o.length>0&&(e.data.groups=JSON.parse(JSON.stringify(o)));const a=this.boundingSphere;return a!==null&&(e.data.boundingSphere=a.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const t={};this.name=e.name;const n=e.index;n!==null&&this.setIndex(n.clone());const s=e.attributes;for(const c in s){const u=s[c];this.setAttribute(c,u.clone(t))}const r=e.morphAttributes;for(const c in r){const u=[],f=r[c];for(let d=0,h=f.length;d<h;d++)u.push(f[d].clone(t));this.morphAttributes[c]=u}this.morphTargetsRelative=e.morphTargetsRelative;const o=e.groups;for(let c=0,u=o.length;c<u;c++){const f=o[c];this.addGroup(f.start,f.count,f.materialIndex)}const a=e.boundingBox;a!==null&&(this.boundingBox=a.clone());const l=e.boundingSphere;return l!==null&&(this.boundingSphere=l.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const Xh=new qe,Ys=new ed,ka=new $l,qh=new L,Ha=new L,Va=new L,Ga=new L,zc=new L,Wa=new L,Qh=new L,Xa=new L;class Vt extends Wt{constructor(e=new An,t=new hr){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}getVertexPosition(e,t){const n=this.geometry,s=n.attributes.position,r=n.morphAttributes.position,o=n.morphTargetsRelative;t.fromBufferAttribute(s,e);const a=this.morphTargetInfluences;if(r&&a){Wa.set(0,0,0);for(let l=0,c=r.length;l<c;l++){const u=a[l],f=r[l];u!==0&&(zc.fromBufferAttribute(f,e),o?Wa.addScaledVector(zc,u):Wa.addScaledVector(zc.sub(t),u))}t.add(Wa)}return t}raycast(e,t){const n=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),ka.copy(n.boundingSphere),ka.applyMatrix4(r),Ys.copy(e.ray).recast(e.near),!(ka.containsPoint(Ys.origin)===!1&&(Ys.intersectSphere(ka,qh)===null||Ys.origin.distanceToSquared(qh)>(e.far-e.near)**2))&&(Xh.copy(r).invert(),Ys.copy(e.ray).applyMatrix4(Xh),!(n.boundingBox!==null&&Ys.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(e,t,Ys)))}_computeIntersections(e,t,n){let s;const r=this.geometry,o=this.material,a=r.index,l=r.attributes.position,c=r.attributes.uv,u=r.attributes.uv1,f=r.attributes.normal,d=r.groups,h=r.drawRange;if(a!==null)if(Array.isArray(o))for(let g=0,p=d.length;g<p;g++){const x=d[g],m=o[x.materialIndex],_=Math.max(x.start,h.start),A=Math.min(a.count,Math.min(x.start+x.count,h.start+h.count));for(let S=_,v=A;S<v;S+=3){const b=a.getX(S),M=a.getX(S+1),E=a.getX(S+2);s=qa(this,m,e,n,c,u,f,b,M,E),s&&(s.faceIndex=Math.floor(S/3),s.face.materialIndex=x.materialIndex,t.push(s))}}else{const g=Math.max(0,h.start),p=Math.min(a.count,h.start+h.count);for(let x=g,m=p;x<m;x+=3){const _=a.getX(x),A=a.getX(x+1),S=a.getX(x+2);s=qa(this,o,e,n,c,u,f,_,A,S),s&&(s.faceIndex=Math.floor(x/3),t.push(s))}}else if(l!==void 0)if(Array.isArray(o))for(let g=0,p=d.length;g<p;g++){const x=d[g],m=o[x.materialIndex],_=Math.max(x.start,h.start),A=Math.min(l.count,Math.min(x.start+x.count,h.start+h.count));for(let S=_,v=A;S<v;S+=3){const b=S,M=S+1,E=S+2;s=qa(this,m,e,n,c,u,f,b,M,E),s&&(s.faceIndex=Math.floor(S/3),s.face.materialIndex=x.materialIndex,t.push(s))}}else{const g=Math.max(0,h.start),p=Math.min(l.count,h.start+h.count);for(let x=g,m=p;x<m;x+=3){const _=x,A=x+1,S=x+2;s=qa(this,o,e,n,c,u,f,_,A,S),s&&(s.faceIndex=Math.floor(x/3),t.push(s))}}}}function $S(i,e,t,n,s,r,o,a){let l;if(e.side===En?l=n.intersectTriangle(o,r,s,!0,a):l=n.intersectTriangle(s,r,o,e.side===Bi,a),l===null)return null;Xa.copy(a),Xa.applyMatrix4(i.matrixWorld);const c=t.ray.origin.distanceTo(Xa);return c<t.near||c>t.far?null:{distance:c,point:Xa.clone(),object:i}}function qa(i,e,t,n,s,r,o,a,l,c){i.getVertexPosition(a,Ha),i.getVertexPosition(l,Va),i.getVertexPosition(c,Ga);const u=$S(i,e,t,n,Ha,Va,Ga,Qh);if(u){const f=new L;pi.getBarycoord(Qh,Ha,Va,Ga,f),s&&(u.uv=pi.getInterpolatedAttribute(s,a,l,c,f,new Ne)),r&&(u.uv1=pi.getInterpolatedAttribute(r,a,l,c,f,new Ne)),o&&(u.normal=pi.getInterpolatedAttribute(o,a,l,c,f,new L),u.normal.dot(n.direction)>0&&u.normal.multiplyScalar(-1));const d={a,b:l,c,normal:new L,materialIndex:0};pi.getNormal(Ha,Va,Ga,d.normal),u.face=d,u.barycoord=f}return u}class vo extends An{constructor(e=1,t=1,n=1,s=1,r=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:n,widthSegments:s,heightSegments:r,depthSegments:o};const a=this;s=Math.floor(s),r=Math.floor(r),o=Math.floor(o);const l=[],c=[],u=[],f=[];let d=0,h=0;g("z","y","x",-1,-1,n,t,e,o,r,0),g("z","y","x",1,-1,n,t,-e,o,r,1),g("x","z","y",1,1,e,n,t,s,o,2),g("x","z","y",1,-1,e,n,-t,s,o,3),g("x","y","z",1,-1,e,t,n,s,r,4),g("x","y","z",-1,-1,e,t,-n,s,r,5),this.setIndex(l),this.setAttribute("position",new on(c,3)),this.setAttribute("normal",new on(u,3)),this.setAttribute("uv",new on(f,2));function g(p,x,m,_,A,S,v,b,M,E,y){const C=S/M,I=v/E,P=S/2,U=v/2,O=b/2,k=M+1,z=E+1;let W=0,H=0;const $=new L;for(let X=0;X<z;X++){const Z=X*I-U;for(let pe=0;pe<k;pe++){const ge=pe*C-P;$[p]=ge*_,$[x]=Z*A,$[m]=O,c.push($.x,$.y,$.z),$[p]=0,$[x]=0,$[m]=b>0?1:-1,u.push($.x,$.y,$.z),f.push(pe/M),f.push(1-X/E),W+=1}}for(let X=0;X<E;X++)for(let Z=0;Z<M;Z++){const pe=d+Z+k*X,ge=d+Z+k*(X+1),be=d+(Z+1)+k*(X+1),Me=d+(Z+1)+k*X;l.push(pe,ge,Me),l.push(ge,be,Me),H+=6}a.addGroup(h,H,y),h+=H,d+=W}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new vo(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}function fo(i){const e={};for(const t in i){e[t]={};for(const n in i[t]){const s=i[t][n];s&&(s.isColor||s.isMatrix3||s.isMatrix4||s.isVector2||s.isVector3||s.isVector4||s.isTexture||s.isQuaternion)?s.isRenderTargetTexture?(je("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][n]=null):e[t][n]=s.clone():Array.isArray(s)?e[t][n]=s.slice():e[t][n]=s}}return e}function fn(i){const e={};for(let t=0;t<i.length;t++){const n=fo(i[t]);for(const s in n)e[s]=n[s]}return e}function ZS(i){const e=[];for(let t=0;t<i.length;t++)e.push(i[t].clone());return e}function V0(i){const e=i.getRenderTarget();return e===null?i.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:rt.workingColorSpace}const JS={clone:fo,merge:fn};var ev=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,tv=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class _n extends ya{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=ev,this.fragmentShader=tv,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=fo(e.uniforms),this.uniformsGroups=ZS(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this}toJSON(e){const t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(const s in this.uniforms){const o=this.uniforms[s].value;o&&o.isTexture?t.uniforms[s]={type:"t",value:o.toJSON(e).uuid}:o&&o.isColor?t.uniforms[s]={type:"c",value:o.getHex()}:o&&o.isVector2?t.uniforms[s]={type:"v2",value:o.toArray()}:o&&o.isVector3?t.uniforms[s]={type:"v3",value:o.toArray()}:o&&o.isVector4?t.uniforms[s]={type:"v4",value:o.toArray()}:o&&o.isMatrix3?t.uniforms[s]={type:"m3",value:o.toArray()}:o&&o.isMatrix4?t.uniforms[s]={type:"m4",value:o.toArray()}:t.uniforms[s]={value:o}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;const n={};for(const s in this.extensions)this.extensions[s]===!0&&(n[s]=!0);return Object.keys(n).length>0&&(t.extensions=n),t}}class G0 extends Wt{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new qe,this.projectionMatrix=new qe,this.projectionMatrixInverse=new qe,this.coordinateSystem=Ei,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const xs=new L,Yh=new Ne,Kh=new Ne;class ei extends G0{constructor(e=50,t=1,n=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=n,this.far=s,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const t=.5*this.getFilmHeight()/e;this.fov=lf*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(ml*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return lf*2*Math.atan(Math.tan(ml*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,n){xs.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(xs.x,xs.y).multiplyScalar(-e/xs.z),xs.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(xs.x,xs.y).multiplyScalar(-e/xs.z)}getViewSize(e,t){return this.getViewBounds(e,Yh,Kh),t.subVectors(Kh,Yh)}setViewOffset(e,t,n,s,r,o){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let t=e*Math.tan(ml*.5*this.fov)/this.zoom,n=2*t,s=this.aspect*n,r=-.5*s;const o=this.view;if(this.view!==null&&this.view.enabled){const l=o.fullWidth,c=o.fullHeight;r+=o.offsetX*s/l,t-=o.offsetY*n/c,s*=o.width/l,n*=o.height/c}const a=this.filmOffset;a!==0&&(r+=e*a/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,t,t-n,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}}const Pr=-90,Fr=1;class nv extends Wt{constructor(e,t,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const s=new ei(Pr,Fr,e,t);s.layers=this.layers,this.add(s);const r=new ei(Pr,Fr,e,t);r.layers=this.layers,this.add(r);const o=new ei(Pr,Fr,e,t);o.layers=this.layers,this.add(o);const a=new ei(Pr,Fr,e,t);a.layers=this.layers,this.add(a);const l=new ei(Pr,Fr,e,t);l.layers=this.layers,this.add(l);const c=new ei(Pr,Fr,e,t);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){const e=this.coordinateSystem,t=this.children.concat(),[n,s,r,o,a,l]=t;for(const c of t)this.remove(c);if(e===Ei)n.up.set(0,1,0),n.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),o.up.set(0,0,1),o.lookAt(0,-1,0),a.up.set(0,1,0),a.lookAt(0,0,1),l.up.set(0,1,0),l.lookAt(0,0,-1);else if(e===El)n.up.set(0,-1,0),n.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),o.up.set(0,0,-1),o.lookAt(0,-1,0),a.up.set(0,-1,0),a.lookAt(0,0,1),l.up.set(0,-1,0),l.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const c of t)this.add(c),c.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[r,o,a,l,c,u]=this.children,f=e.getRenderTarget(),d=e.getActiveCubeFace(),h=e.getActiveMipmapLevel(),g=e.xr.enabled;e.xr.enabled=!1;const p=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,e.setRenderTarget(n,0,s),e.render(t,r),e.setRenderTarget(n,1,s),e.render(t,o),e.setRenderTarget(n,2,s),e.render(t,a),e.setRenderTarget(n,3,s),e.render(t,l),e.setRenderTarget(n,4,s),e.render(t,c),n.texture.generateMipmaps=p,e.setRenderTarget(n,5,s),e.render(t,u),e.setRenderTarget(f,d,h),e.xr.enabled=g,n.texture.needsPMREMUpdate=!0}}class W0 extends xn{constructor(e=[],t=ao,n,s,r,o,a,l,c,u){super(e,t,n,s,r,o,a,l,c,u),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class iv extends Us{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;const n={width:e,height:e,depth:1},s=[n,n,n,n,n,n];this.texture=new W0(s),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},s=new vo(5,5,5),r=new _n({name:"CubemapFromEquirect",uniforms:fo(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:En,blending:is});r.uniforms.tEquirect.value=t;const o=new Vt(s,r),a=t.minFilter;return t.minFilter===sr&&(t.minFilter=ii),new nv(1,10,this).update(e,o),t.minFilter=a,o.geometry.dispose(),o.material.dispose(),this}clear(e,t=!0,n=!0,s=!0){const r=e.getRenderTarget();for(let o=0;o<6;o++)e.setRenderTarget(this,o),e.clear(t,n,s);e.setRenderTarget(r)}}class Qa extends Wt{constructor(){super(),this.isGroup=!0,this.type="Group"}}const sv={type:"move"};class kc{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Qa,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Qa,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new L,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new L),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Qa,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new L,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new L),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const t=this._hand;if(t)for(const n of e.hand.values())this._getHandJoint(t,n)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,n){let s=null,r=null,o=null;const a=this._targetRay,l=this._grip,c=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(c&&e.hand){o=!0;for(const p of e.hand.values()){const x=t.getJointPose(p,n),m=this._getHandJoint(c,p);x!==null&&(m.matrix.fromArray(x.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,m.jointRadius=x.radius),m.visible=x!==null}const u=c.joints["index-finger-tip"],f=c.joints["thumb-tip"],d=u.position.distanceTo(f.position),h=.02,g=.005;c.inputState.pinching&&d>h+g?(c.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!c.inputState.pinching&&d<=h-g&&(c.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else l!==null&&e.gripSpace&&(r=t.getPose(e.gripSpace,n),r!==null&&(l.matrix.fromArray(r.transform.matrix),l.matrix.decompose(l.position,l.rotation,l.scale),l.matrixWorldNeedsUpdate=!0,r.linearVelocity?(l.hasLinearVelocity=!0,l.linearVelocity.copy(r.linearVelocity)):l.hasLinearVelocity=!1,r.angularVelocity?(l.hasAngularVelocity=!0,l.angularVelocity.copy(r.angularVelocity)):l.hasAngularVelocity=!1));a!==null&&(s=t.getPose(e.targetRaySpace,n),s===null&&r!==null&&(s=r),s!==null&&(a.matrix.fromArray(s.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),a.matrixWorldNeedsUpdate=!0,s.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(s.linearVelocity)):a.hasLinearVelocity=!1,s.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(s.angularVelocity)):a.hasAngularVelocity=!1,this.dispatchEvent(sv)))}return a!==null&&(a.visible=s!==null),l!==null&&(l.visible=r!==null),c!==null&&(c.visible=o!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){const n=new Qa;n.matrixAutoUpdate=!1,n.visible=!1,e.joints[t.jointName]=n,e.add(n)}return e.joints[t.jointName]}}class rv extends Wt{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new _i,this.environmentIntensity=1,this.environmentRotation=new _i,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}}class Qi extends xn{constructor(e=null,t=1,n=1,s,r,o,a,l,c=qn,u=qn,f,d){super(null,o,a,l,c,u,s,r,f,d),this.isDataTexture=!0,this.image={data:e,width:t,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class ov extends li{constructor(e,t,n,s=1){super(e,t,n),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=s}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}toJSON(){const e=super.toJSON();return e.meshPerAttribute=this.meshPerAttribute,e.isInstancedBufferAttribute=!0,e}}const Hc=new L,av=new L,lv=new Qe;class vs{constructor(e=new L(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,n,s){return this.normal.set(e,t,n),this.constant=s,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,n){const s=Hc.subVectors(n,t).cross(av.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t){const n=e.delta(Hc),s=this.normal.dot(n);if(s===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;const r=-(e.start.dot(this.normal)+this.constant)/s;return r<0||r>1?null:t.copy(e.start).addScaledVector(n,r)}intersectsLine(e){const t=this.distanceToPoint(e.start),n=this.distanceToPoint(e.end);return t<0&&n>0||n<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){const n=t||lv.getNormalMatrix(e),s=this.coplanarPoint(Hc).applyMatrix4(e),r=this.normal.applyMatrix3(n).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const Ks=new $l,cv=new Ne(.5,.5),Ya=new L;class X0{constructor(e=new vs,t=new vs,n=new vs,s=new vs,r=new vs,o=new vs){this.planes=[e,t,n,s,r,o]}set(e,t,n,s,r,o){const a=this.planes;return a[0].copy(e),a[1].copy(t),a[2].copy(n),a[3].copy(s),a[4].copy(r),a[5].copy(o),this}copy(e){const t=this.planes;for(let n=0;n<6;n++)t[n].copy(e.planes[n]);return this}setFromProjectionMatrix(e,t=Ei,n=!1){const s=this.planes,r=e.elements,o=r[0],a=r[1],l=r[2],c=r[3],u=r[4],f=r[5],d=r[6],h=r[7],g=r[8],p=r[9],x=r[10],m=r[11],_=r[12],A=r[13],S=r[14],v=r[15];if(s[0].setComponents(c-o,h-u,m-g,v-_).normalize(),s[1].setComponents(c+o,h+u,m+g,v+_).normalize(),s[2].setComponents(c+a,h+f,m+p,v+A).normalize(),s[3].setComponents(c-a,h-f,m-p,v-A).normalize(),n)s[4].setComponents(l,d,x,S).normalize(),s[5].setComponents(c-l,h-d,m-x,v-S).normalize();else if(s[4].setComponents(c-l,h-d,m-x,v-S).normalize(),t===Ei)s[5].setComponents(c+l,h+d,m+x,v+S).normalize();else if(t===El)s[5].setComponents(l,d,x,S).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),Ks.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),Ks.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(Ks)}intersectsSprite(e){Ks.center.set(0,0,0);const t=cv.distanceTo(e.center);return Ks.radius=.7071067811865476+t,Ks.applyMatrix4(e.matrixWorld),this.intersectsSphere(Ks)}intersectsSphere(e){const t=this.planes,n=e.center,s=-e.radius;for(let r=0;r<6;r++)if(t[r].distanceToPoint(n)<s)return!1;return!0}intersectsBox(e){const t=this.planes;for(let n=0;n<6;n++){const s=t[n];if(Ya.x=s.normal.x>0?e.max.x:e.min.x,Ya.y=s.normal.y>0?e.max.y:e.min.y,Ya.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(Ya)<0)return!1}return!0}containsPoint(e){const t=this.planes;for(let n=0;n<6;n++)if(t[n].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class uv extends ya{constructor(e){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new nt(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.size=e.size,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}}const jh=new qe,cf=new ed,Ka=new $l,ja=new L;class fv extends Wt{constructor(e=new An,t=new uv){super(),this.isPoints=!0,this.type="Points",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}raycast(e,t){const n=this.geometry,s=this.matrixWorld,r=e.params.Points.threshold,o=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),Ka.copy(n.boundingSphere),Ka.applyMatrix4(s),Ka.radius+=r,e.ray.intersectsSphere(Ka)===!1)return;jh.copy(s).invert(),cf.copy(e.ray).applyMatrix4(jh);const a=r/((this.scale.x+this.scale.y+this.scale.z)/3),l=a*a,c=n.index,f=n.attributes.position;if(c!==null){const d=Math.max(0,o.start),h=Math.min(c.count,o.start+o.count);for(let g=d,p=h;g<p;g++){const x=c.getX(g);ja.fromBufferAttribute(f,x),$h(ja,x,l,s,e,t,this)}}else{const d=Math.max(0,o.start),h=Math.min(f.count,o.start+o.count);for(let g=d,p=h;g<p;g++)ja.fromBufferAttribute(f,g),$h(ja,g,l,s,e,t,this)}}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}}function $h(i,e,t,n,s,r,o){const a=cf.distanceSqToPoint(i);if(a<t){const l=new L;cf.closestPointToPoint(i,l),l.applyMatrix4(n);const c=s.ray.origin.distanceTo(l);if(c<s.near||c>s.far)return;r.push({distance:c,distanceToRay:Math.sqrt(a),point:l,index:e,face:null,faceIndex:null,barycoord:null,object:o})}}class td extends xn{constructor(e,t,n=si,s,r,o,a=qn,l=qn,c,u=co,f=1){if(u!==co&&u!==aa)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const d={width:e,height:t,depth:f};super(d,s,r,o,a,l,u,n,c),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new Jf(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}}class q0 extends xn{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class ua extends An{constructor(e=1,t=1,n=1,s=32,r=1,o=!1,a=0,l=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:n,radialSegments:s,heightSegments:r,openEnded:o,thetaStart:a,thetaLength:l};const c=this;s=Math.floor(s),r=Math.floor(r);const u=[],f=[],d=[],h=[];let g=0;const p=[],x=n/2;let m=0;_(),o===!1&&(e>0&&A(!0),t>0&&A(!1)),this.setIndex(u),this.setAttribute("position",new on(f,3)),this.setAttribute("normal",new on(d,3)),this.setAttribute("uv",new on(h,2));function _(){const S=new L,v=new L;let b=0;const M=(t-e)/n;for(let E=0;E<=r;E++){const y=[],C=E/r,I=C*(t-e)+e;for(let P=0;P<=s;P++){const U=P/s,O=U*l+a,k=Math.sin(O),z=Math.cos(O);v.x=I*k,v.y=-C*n+x,v.z=I*z,f.push(v.x,v.y,v.z),S.set(k,M,z).normalize(),d.push(S.x,S.y,S.z),h.push(U,1-C),y.push(g++)}p.push(y)}for(let E=0;E<s;E++)for(let y=0;y<r;y++){const C=p[y][E],I=p[y+1][E],P=p[y+1][E+1],U=p[y][E+1];(e>0||y!==0)&&(u.push(C,I,U),b+=3),(t>0||y!==r-1)&&(u.push(I,P,U),b+=3)}c.addGroup(m,b,0),m+=b}function A(S){const v=g,b=new Ne,M=new L;let E=0;const y=S===!0?e:t,C=S===!0?1:-1;for(let P=1;P<=s;P++)f.push(0,x*C,0),d.push(0,C,0),h.push(.5,.5),g++;const I=g;for(let P=0;P<=s;P++){const O=P/s*l+a,k=Math.cos(O),z=Math.sin(O);M.x=y*z,M.y=x*C,M.z=y*k,f.push(M.x,M.y,M.z),d.push(0,C,0),b.x=k*.5+.5,b.y=z*.5*C+.5,h.push(b.x,b.y),g++}for(let P=0;P<s;P++){const U=v+P,O=I+P;S===!0?u.push(O,O+1,U):u.push(O+1,O,U),E+=3}c.addGroup(m,E,S===!0?1:2),m+=E}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new ua(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class nd extends ua{constructor(e=1,t=1,n=32,s=1,r=!1,o=0,a=Math.PI*2){super(0,e,t,n,s,r,o,a),this.type="ConeGeometry",this.parameters={radius:e,height:t,radialSegments:n,heightSegments:s,openEnded:r,thetaStart:o,thetaLength:a}}static fromJSON(e){return new nd(e.radius,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class ho extends An{constructor(e=1,t=1,n=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:n,heightSegments:s};const r=e/2,o=t/2,a=Math.floor(n),l=Math.floor(s),c=a+1,u=l+1,f=e/a,d=t/l,h=[],g=[],p=[],x=[];for(let m=0;m<u;m++){const _=m*d-o;for(let A=0;A<c;A++){const S=A*f-r;g.push(S,-_,0),p.push(0,0,1),x.push(A/a),x.push(1-m/l)}}for(let m=0;m<l;m++)for(let _=0;_<a;_++){const A=_+c*m,S=_+c*(m+1),v=_+1+c*(m+1),b=_+1+c*m;h.push(A,S,b),h.push(S,v,b)}this.setIndex(h),this.setAttribute("position",new on(g,3)),this.setAttribute("normal",new on(p,3)),this.setAttribute("uv",new on(x,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new ho(e.width,e.height,e.widthSegments,e.heightSegments)}}class Rl extends An{constructor(e=1,t=32,n=16,s=0,r=Math.PI*2,o=0,a=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:e,widthSegments:t,heightSegments:n,phiStart:s,phiLength:r,thetaStart:o,thetaLength:a},t=Math.max(3,Math.floor(t)),n=Math.max(2,Math.floor(n));const l=Math.min(o+a,Math.PI);let c=0;const u=[],f=new L,d=new L,h=[],g=[],p=[],x=[];for(let m=0;m<=n;m++){const _=[],A=m/n;let S=0;m===0&&o===0?S=.5/t:m===n&&l===Math.PI&&(S=-.5/t);for(let v=0;v<=t;v++){const b=v/t;f.x=-e*Math.cos(s+b*r)*Math.sin(o+A*a),f.y=e*Math.cos(o+A*a),f.z=e*Math.sin(s+b*r)*Math.sin(o+A*a),g.push(f.x,f.y,f.z),d.copy(f).normalize(),p.push(d.x,d.y,d.z),x.push(b+S,1-A),_.push(c++)}u.push(_)}for(let m=0;m<n;m++)for(let _=0;_<t;_++){const A=u[m][_+1],S=u[m][_],v=u[m+1][_],b=u[m+1][_+1];(m!==0||o>0)&&h.push(A,S,b),(m!==n-1||l<Math.PI)&&h.push(S,v,b)}this.setIndex(h),this.setAttribute("position",new on(g,3)),this.setAttribute("normal",new on(p,3)),this.setAttribute("uv",new on(x,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Rl(e.radius,e.widthSegments,e.heightSegments,e.phiStart,e.phiLength,e.thetaStart,e.thetaLength)}}class dv extends ya{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=xS,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class hv extends ya{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}class id extends G0{constructor(e=-1,t=1,n=1,s=-1,r=.1,o=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=n,this.bottom=s,this.near=r,this.far=o,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,n,s,r,o){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,s=(this.top+this.bottom)/2;let r=n-e,o=n+e,a=s+t,l=s-t;if(this.view!==null&&this.view.enabled){const c=(this.right-this.left)/this.view.fullWidth/this.zoom,u=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=c*this.view.offsetX,o=r+c*this.view.width,a-=u*this.view.offsetY,l=a-u*this.view.height}this.projectionMatrix.makeOrthographic(r,o,a,l,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}}class pv extends An{constructor(){super(),this.isInstancedBufferGeometry=!0,this.type="InstancedBufferGeometry",this.instanceCount=1/0}copy(e){return super.copy(e),this.instanceCount=e.instanceCount,this}toJSON(){const e=super.toJSON();return e.instanceCount=this.instanceCount,e.isInstancedBufferGeometry=!0,e}}class mv extends ei{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}class Zh{constructor(e=1,t=0,n=0){this.radius=e,this.phi=t,this.theta=n}set(e,t,n){return this.radius=e,this.phi=t,this.theta=n,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=Je(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,n){return this.radius=Math.sqrt(e*e+t*t+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,n),this.phi=Math.acos(Je(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}function Jh(i,e,t,n){const s=gv(n);switch(t){case D0:return i*e;case F0:return i*e/s.components*s.byteLength;case jl:return i*e/s.components*s.byteLength;case $f:return i*e*2/s.components*s.byteLength;case Zf:return i*e*2/s.components*s.byteLength;case P0:return i*e*3/s.components*s.byteLength;case gn:return i*e*4/s.components*s.byteLength;case jr:return i*e*4/s.components*s.byteLength;case fl:case dl:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case hl:case pl:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Lu:case Uu:return Math.max(i,16)*Math.max(e,8)/4;case Fu:case Bu:return Math.max(i,8)*Math.max(e,8)/2;case Ou:case Nu:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case zu:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case ku:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Hu:return Math.floor((i+4)/5)*Math.floor((e+3)/4)*16;case Vu:return Math.floor((i+4)/5)*Math.floor((e+4)/5)*16;case Gu:return Math.floor((i+5)/6)*Math.floor((e+4)/5)*16;case Wu:return Math.floor((i+5)/6)*Math.floor((e+5)/6)*16;case Xu:return Math.floor((i+7)/8)*Math.floor((e+4)/5)*16;case qu:return Math.floor((i+7)/8)*Math.floor((e+5)/6)*16;case Qu:return Math.floor((i+7)/8)*Math.floor((e+7)/8)*16;case Yu:return Math.floor((i+9)/10)*Math.floor((e+4)/5)*16;case Ku:return Math.floor((i+9)/10)*Math.floor((e+5)/6)*16;case ju:return Math.floor((i+9)/10)*Math.floor((e+7)/8)*16;case $u:return Math.floor((i+9)/10)*Math.floor((e+9)/10)*16;case Zu:return Math.floor((i+11)/12)*Math.floor((e+9)/10)*16;case Ju:return Math.floor((i+11)/12)*Math.floor((e+11)/12)*16;case ef:case tf:case nf:return Math.ceil(i/4)*Math.ceil(e/4)*16;case sf:case rf:return Math.ceil(i/4)*Math.ceil(e/4)*8;case of:case af:return Math.ceil(i/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function gv(i){switch(i){case Ui:case E0:return{byteLength:1,components:1};case ra:case w0:case pr:return{byteLength:2,components:1};case Kf:case jf:return{byteLength:2,components:4};case si:case Yf:case mi:return{byteLength:4,components:1};case R0:case I0:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${i}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:Qf}}));typeof window<"u"&&(window.__THREE__?je("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=Qf);function Q0(){let i=null,e=!1,t=null,n=null;function s(r,o){t(r,o),n=i.requestAnimationFrame(s)}return{start:function(){e!==!0&&t!==null&&(n=i.requestAnimationFrame(s),e=!0)},stop:function(){i.cancelAnimationFrame(n),e=!1},setAnimationLoop:function(r){t=r},setContext:function(r){i=r}}}function xv(i){const e=new WeakMap;function t(a,l){const c=a.array,u=a.usage,f=c.byteLength,d=i.createBuffer();i.bindBuffer(l,d),i.bufferData(l,c,u),a.onUploadCallback();let h;if(c instanceof Float32Array)h=i.FLOAT;else if(typeof Float16Array<"u"&&c instanceof Float16Array)h=i.HALF_FLOAT;else if(c instanceof Uint16Array)a.isFloat16BufferAttribute?h=i.HALF_FLOAT:h=i.UNSIGNED_SHORT;else if(c instanceof Int16Array)h=i.SHORT;else if(c instanceof Uint32Array)h=i.UNSIGNED_INT;else if(c instanceof Int32Array)h=i.INT;else if(c instanceof Int8Array)h=i.BYTE;else if(c instanceof Uint8Array)h=i.UNSIGNED_BYTE;else if(c instanceof Uint8ClampedArray)h=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+c);return{buffer:d,type:h,bytesPerElement:c.BYTES_PER_ELEMENT,version:a.version,size:f}}function n(a,l,c){const u=l.array,f=l.updateRanges;if(i.bindBuffer(c,a),f.length===0)i.bufferSubData(c,0,u);else{f.sort((h,g)=>h.start-g.start);let d=0;for(let h=1;h<f.length;h++){const g=f[d],p=f[h];p.start<=g.start+g.count+1?g.count=Math.max(g.count,p.start+p.count-g.start):(++d,f[d]=p)}f.length=d+1;for(let h=0,g=f.length;h<g;h++){const p=f[h];i.bufferSubData(c,p.start*u.BYTES_PER_ELEMENT,u,p.start,p.count)}l.clearUpdateRanges()}l.onUploadCallback()}function s(a){return a.isInterleavedBufferAttribute&&(a=a.data),e.get(a)}function r(a){a.isInterleavedBufferAttribute&&(a=a.data);const l=e.get(a);l&&(i.deleteBuffer(l.buffer),e.delete(a))}function o(a,l){if(a.isInterleavedBufferAttribute&&(a=a.data),a.isGLBufferAttribute){const u=e.get(a);(!u||u.version<a.version)&&e.set(a,{buffer:a.buffer,type:a.type,bytesPerElement:a.elementSize,version:a.version});return}const c=e.get(a);if(c===void 0)e.set(a,t(a,l));else if(c.version<a.version){if(c.size!==a.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(c.buffer,a,l),c.version=a.version}}return{get:s,remove:r,update:o}}var _v=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,Av=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,Sv=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,vv=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,yv=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,bv=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,Mv=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,Cv=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,Tv=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`,Ev=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,wv=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Rv=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,Iv=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,Dv=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,Pv=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,Fv=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,Lv=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,Bv=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,Uv=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,Ov=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,Nv=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,zv=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,kv=`#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`,Hv=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,Vv=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,Gv=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,Wv=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,Xv=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,qv=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,Qv=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,Yv="gl_FragColor = linearToOutputTexel( gl_FragColor );",Kv=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,jv=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`,$v=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,Zv=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,Jv=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,ey=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,ty=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,ny=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,iy=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,sy=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,ry=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,oy=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,ay=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,ly=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,cy=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,uy=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,fy=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,dy=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,hy=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,py=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,my=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,gy=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 uv = vec2( roughness, dotNV );
	return texture2D( dfgLUT, uv ).rg;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
vec3 BRDF_GGX_Multiscatter( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 singleScatter = BRDF_GGX( lightDir, viewDir, normal, material );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 dfgV = DFGApprox( vec3(0.0, 0.0, 1.0), vec3(sqrt(1.0 - dotNV * dotNV), 0.0, dotNV), material.roughness );
	vec2 dfgL = DFGApprox( vec3(0.0, 0.0, 1.0), vec3(sqrt(1.0 - dotNL * dotNL), 0.0, dotNL), material.roughness );
	vec3 FssEss_V = material.specularColor * dfgV.x + material.specularF90 * dfgV.y;
	vec3 FssEss_L = material.specularColor * dfgL.x + material.specularF90 * dfgL.y;
	float Ess_V = dfgV.x + dfgV.y;
	float Ess_L = dfgL.x + dfgL.y;
	float Ems_V = 1.0 - Ess_V;
	float Ems_L = 1.0 - Ess_L;
	vec3 Favg = material.specularColor + ( 1.0 - material.specularColor ) * 0.047619;
	vec3 Fms = FssEss_V * FssEss_L * Favg / ( 1.0 - Ems_V * Ems_L * Favg * Favg + EPSILON );
	float compensationFactor = Ems_V * Ems_L;
	vec3 multiScatter = Fms * compensationFactor;
	return singleScatter + multiScatter;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX_Multiscatter( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,xy=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,_y=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,Ay=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,Sy=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,vy=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,yy=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,by=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,My=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Cy=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,Ty=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,Ey=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,wy=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Ry=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,Iy=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,Dy=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Py=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Fy=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,Ly=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,By=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,Uy=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,Oy=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Ny=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,zy=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,ky=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,Hy=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,Vy=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,Gy=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,Wy=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,Xy=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,qy=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`,Qy=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,Yy=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,Ky=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,jy=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,$y=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,Zy=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,Jy=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		float depth = unpackRGBAToDepth( texture2D( depths, uv ) );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			return step( depth, compare );
		#else
			return step( compare, depth );
		#endif
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow( sampler2D shadow, vec2 uv, float compare ) {
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			float hard_shadow = step( distribution.x, compare );
		#else
			float hard_shadow = step( compare, distribution.x );
		#endif
		if ( hard_shadow != 1.0 ) {
			float distance = compare - distribution.x;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		
		float lightToPositionLength = length( lightToPosition );
		if ( lightToPositionLength - shadowCameraFar <= 0.0 && lightToPositionLength - shadowCameraNear >= 0.0 ) {
			float dp = ( lightToPositionLength - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
			#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
				vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
				shadow = (
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
				) * ( 1.0 / 9.0 );
			#else
				shadow = texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
#endif`,eb=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,tb=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,nb=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,ib=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,sb=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,rb=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,ob=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,ab=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,lb=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,cb=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,ub=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,fb=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,db=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,hb=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,pb=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,mb=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,gb=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const xb=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,_b=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Ab=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Sb=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,vb=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,yb=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,bb=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Mb=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,Cb=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Tb=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`,Eb=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,wb=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Rb=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Ib=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Db=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,Pb=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Fb=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Lb=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Bb=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,Ub=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Ob=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,Nb=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,zb=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,kb=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Hb=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,Vb=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Gb=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Wb=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Xb=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,qb=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Qb=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Yb=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,Kb=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,jb=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,Ze={alphahash_fragment:_v,alphahash_pars_fragment:Av,alphamap_fragment:Sv,alphamap_pars_fragment:vv,alphatest_fragment:yv,alphatest_pars_fragment:bv,aomap_fragment:Mv,aomap_pars_fragment:Cv,batching_pars_vertex:Tv,batching_vertex:Ev,begin_vertex:wv,beginnormal_vertex:Rv,bsdfs:Iv,iridescence_fragment:Dv,bumpmap_pars_fragment:Pv,clipping_planes_fragment:Fv,clipping_planes_pars_fragment:Lv,clipping_planes_pars_vertex:Bv,clipping_planes_vertex:Uv,color_fragment:Ov,color_pars_fragment:Nv,color_pars_vertex:zv,color_vertex:kv,common:Hv,cube_uv_reflection_fragment:Vv,defaultnormal_vertex:Gv,displacementmap_pars_vertex:Wv,displacementmap_vertex:Xv,emissivemap_fragment:qv,emissivemap_pars_fragment:Qv,colorspace_fragment:Yv,colorspace_pars_fragment:Kv,envmap_fragment:jv,envmap_common_pars_fragment:$v,envmap_pars_fragment:Zv,envmap_pars_vertex:Jv,envmap_physical_pars_fragment:uy,envmap_vertex:ey,fog_vertex:ty,fog_pars_vertex:ny,fog_fragment:iy,fog_pars_fragment:sy,gradientmap_pars_fragment:ry,lightmap_pars_fragment:oy,lights_lambert_fragment:ay,lights_lambert_pars_fragment:ly,lights_pars_begin:cy,lights_toon_fragment:fy,lights_toon_pars_fragment:dy,lights_phong_fragment:hy,lights_phong_pars_fragment:py,lights_physical_fragment:my,lights_physical_pars_fragment:gy,lights_fragment_begin:xy,lights_fragment_maps:_y,lights_fragment_end:Ay,logdepthbuf_fragment:Sy,logdepthbuf_pars_fragment:vy,logdepthbuf_pars_vertex:yy,logdepthbuf_vertex:by,map_fragment:My,map_pars_fragment:Cy,map_particle_fragment:Ty,map_particle_pars_fragment:Ey,metalnessmap_fragment:wy,metalnessmap_pars_fragment:Ry,morphinstance_vertex:Iy,morphcolor_vertex:Dy,morphnormal_vertex:Py,morphtarget_pars_vertex:Fy,morphtarget_vertex:Ly,normal_fragment_begin:By,normal_fragment_maps:Uy,normal_pars_fragment:Oy,normal_pars_vertex:Ny,normal_vertex:zy,normalmap_pars_fragment:ky,clearcoat_normal_fragment_begin:Hy,clearcoat_normal_fragment_maps:Vy,clearcoat_pars_fragment:Gy,iridescence_pars_fragment:Wy,opaque_fragment:Xy,packing:qy,premultiplied_alpha_fragment:Qy,project_vertex:Yy,dithering_fragment:Ky,dithering_pars_fragment:jy,roughnessmap_fragment:$y,roughnessmap_pars_fragment:Zy,shadowmap_pars_fragment:Jy,shadowmap_pars_vertex:eb,shadowmap_vertex:tb,shadowmask_pars_fragment:nb,skinbase_vertex:ib,skinning_pars_vertex:sb,skinning_vertex:rb,skinnormal_vertex:ob,specularmap_fragment:ab,specularmap_pars_fragment:lb,tonemapping_fragment:cb,tonemapping_pars_fragment:ub,transmission_fragment:fb,transmission_pars_fragment:db,uv_pars_fragment:hb,uv_pars_vertex:pb,uv_vertex:mb,worldpos_vertex:gb,background_vert:xb,background_frag:_b,backgroundCube_vert:Ab,backgroundCube_frag:Sb,cube_vert:vb,cube_frag:yb,depth_vert:bb,depth_frag:Mb,distanceRGBA_vert:Cb,distanceRGBA_frag:Tb,equirect_vert:Eb,equirect_frag:wb,linedashed_vert:Rb,linedashed_frag:Ib,meshbasic_vert:Db,meshbasic_frag:Pb,meshlambert_vert:Fb,meshlambert_frag:Lb,meshmatcap_vert:Bb,meshmatcap_frag:Ub,meshnormal_vert:Ob,meshnormal_frag:Nb,meshphong_vert:zb,meshphong_frag:kb,meshphysical_vert:Hb,meshphysical_frag:Vb,meshtoon_vert:Gb,meshtoon_frag:Wb,points_vert:Xb,points_frag:qb,shadow_vert:Qb,shadow_frag:Yb,sprite_vert:Kb,sprite_frag:jb},De={common:{diffuse:{value:new nt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Qe},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Qe}},envmap:{envMap:{value:null},envMapRotation:{value:new Qe},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Qe}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Qe}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Qe},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Qe},normalScale:{value:new Ne(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Qe},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Qe}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Qe}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Qe}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new nt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new nt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0},uvTransform:{value:new Qe}},sprite:{diffuse:{value:new nt(16777215)},opacity:{value:1},center:{value:new Ne(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Qe},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0}}},Ci={basic:{uniforms:fn([De.common,De.specularmap,De.envmap,De.aomap,De.lightmap,De.fog]),vertexShader:Ze.meshbasic_vert,fragmentShader:Ze.meshbasic_frag},lambert:{uniforms:fn([De.common,De.specularmap,De.envmap,De.aomap,De.lightmap,De.emissivemap,De.bumpmap,De.normalmap,De.displacementmap,De.fog,De.lights,{emissive:{value:new nt(0)}}]),vertexShader:Ze.meshlambert_vert,fragmentShader:Ze.meshlambert_frag},phong:{uniforms:fn([De.common,De.specularmap,De.envmap,De.aomap,De.lightmap,De.emissivemap,De.bumpmap,De.normalmap,De.displacementmap,De.fog,De.lights,{emissive:{value:new nt(0)},specular:{value:new nt(1118481)},shininess:{value:30}}]),vertexShader:Ze.meshphong_vert,fragmentShader:Ze.meshphong_frag},standard:{uniforms:fn([De.common,De.envmap,De.aomap,De.lightmap,De.emissivemap,De.bumpmap,De.normalmap,De.displacementmap,De.roughnessmap,De.metalnessmap,De.fog,De.lights,{emissive:{value:new nt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:Ze.meshphysical_vert,fragmentShader:Ze.meshphysical_frag},toon:{uniforms:fn([De.common,De.aomap,De.lightmap,De.emissivemap,De.bumpmap,De.normalmap,De.displacementmap,De.gradientmap,De.fog,De.lights,{emissive:{value:new nt(0)}}]),vertexShader:Ze.meshtoon_vert,fragmentShader:Ze.meshtoon_frag},matcap:{uniforms:fn([De.common,De.bumpmap,De.normalmap,De.displacementmap,De.fog,{matcap:{value:null}}]),vertexShader:Ze.meshmatcap_vert,fragmentShader:Ze.meshmatcap_frag},points:{uniforms:fn([De.points,De.fog]),vertexShader:Ze.points_vert,fragmentShader:Ze.points_frag},dashed:{uniforms:fn([De.common,De.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:Ze.linedashed_vert,fragmentShader:Ze.linedashed_frag},depth:{uniforms:fn([De.common,De.displacementmap]),vertexShader:Ze.depth_vert,fragmentShader:Ze.depth_frag},normal:{uniforms:fn([De.common,De.bumpmap,De.normalmap,De.displacementmap,{opacity:{value:1}}]),vertexShader:Ze.meshnormal_vert,fragmentShader:Ze.meshnormal_frag},sprite:{uniforms:fn([De.sprite,De.fog]),vertexShader:Ze.sprite_vert,fragmentShader:Ze.sprite_frag},background:{uniforms:{uvTransform:{value:new Qe},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:Ze.background_vert,fragmentShader:Ze.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Qe}},vertexShader:Ze.backgroundCube_vert,fragmentShader:Ze.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:Ze.cube_vert,fragmentShader:Ze.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:Ze.equirect_vert,fragmentShader:Ze.equirect_frag},distanceRGBA:{uniforms:fn([De.common,De.displacementmap,{referencePosition:{value:new L},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:Ze.distanceRGBA_vert,fragmentShader:Ze.distanceRGBA_frag},shadow:{uniforms:fn([De.lights,De.fog,{color:{value:new nt(0)},opacity:{value:1}}]),vertexShader:Ze.shadow_vert,fragmentShader:Ze.shadow_frag}};Ci.physical={uniforms:fn([Ci.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Qe},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Qe},clearcoatNormalScale:{value:new Ne(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Qe},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Qe},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Qe},sheen:{value:0},sheenColor:{value:new nt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Qe},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Qe},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Qe},transmissionSamplerSize:{value:new Ne},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Qe},attenuationDistance:{value:0},attenuationColor:{value:new nt(0)},specularColor:{value:new nt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Qe},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Qe},anisotropyVector:{value:new Ne},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Qe}}]),vertexShader:Ze.meshphysical_vert,fragmentShader:Ze.meshphysical_frag};const $a={r:0,b:0,g:0},js=new _i,$b=new qe;function Zb(i,e,t,n,s,r,o){const a=new nt(0);let l=r===!0?0:1,c,u,f=null,d=0,h=null;function g(A){let S=A.isScene===!0?A.background:null;return S&&S.isTexture&&(S=(A.backgroundBlurriness>0?t:e).get(S)),S}function p(A){let S=!1;const v=g(A);v===null?m(a,l):v&&v.isColor&&(m(v,1),S=!0);const b=i.xr.getEnvironmentBlendMode();b==="additive"?n.buffers.color.setClear(0,0,0,1,o):b==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,o),(i.autoClear||S)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function x(A,S){const v=g(S);v&&(v.isCubeTexture||v.mapping===Kl)?(u===void 0&&(u=new Vt(new vo(1,1,1),new _n({name:"BackgroundCubeMaterial",uniforms:fo(Ci.backgroundCube.uniforms),vertexShader:Ci.backgroundCube.vertexShader,fragmentShader:Ci.backgroundCube.fragmentShader,side:En,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),u.geometry.deleteAttribute("normal"),u.geometry.deleteAttribute("uv"),u.onBeforeRender=function(b,M,E){this.matrixWorld.copyPosition(E.matrixWorld)},Object.defineProperty(u.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),s.update(u)),js.copy(S.backgroundRotation),js.x*=-1,js.y*=-1,js.z*=-1,v.isCubeTexture&&v.isRenderTargetTexture===!1&&(js.y*=-1,js.z*=-1),u.material.uniforms.envMap.value=v,u.material.uniforms.flipEnvMap.value=v.isCubeTexture&&v.isRenderTargetTexture===!1?-1:1,u.material.uniforms.backgroundBlurriness.value=S.backgroundBlurriness,u.material.uniforms.backgroundIntensity.value=S.backgroundIntensity,u.material.uniforms.backgroundRotation.value.setFromMatrix4($b.makeRotationFromEuler(js)),u.material.toneMapped=rt.getTransfer(v.colorSpace)!==ht,(f!==v||d!==v.version||h!==i.toneMapping)&&(u.material.needsUpdate=!0,f=v,d=v.version,h=i.toneMapping),u.layers.enableAll(),A.unshift(u,u.geometry,u.material,0,0,null)):v&&v.isTexture&&(c===void 0&&(c=new Vt(new ho(2,2),new _n({name:"BackgroundMaterial",uniforms:fo(Ci.background.uniforms),vertexShader:Ci.background.vertexShader,fragmentShader:Ci.background.fragmentShader,side:Bi,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),s.update(c)),c.material.uniforms.t2D.value=v,c.material.uniforms.backgroundIntensity.value=S.backgroundIntensity,c.material.toneMapped=rt.getTransfer(v.colorSpace)!==ht,v.matrixAutoUpdate===!0&&v.updateMatrix(),c.material.uniforms.uvTransform.value.copy(v.matrix),(f!==v||d!==v.version||h!==i.toneMapping)&&(c.material.needsUpdate=!0,f=v,d=v.version,h=i.toneMapping),c.layers.enableAll(),A.unshift(c,c.geometry,c.material,0,0,null))}function m(A,S){A.getRGB($a,V0(i)),n.buffers.color.setClear($a.r,$a.g,$a.b,S,o)}function _(){u!==void 0&&(u.geometry.dispose(),u.material.dispose(),u=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return a},setClearColor:function(A,S=1){a.set(A),l=S,m(a,l)},getClearAlpha:function(){return l},setClearAlpha:function(A){l=A,m(a,l)},render:p,addToRenderList:x,dispose:_}}function Jb(i,e){const t=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},s=d(null);let r=s,o=!1;function a(C,I,P,U,O){let k=!1;const z=f(U,P,I);r!==z&&(r=z,c(r.object)),k=h(C,U,P,O),k&&g(C,U,P,O),O!==null&&e.update(O,i.ELEMENT_ARRAY_BUFFER),(k||o)&&(o=!1,S(C,I,P,U),O!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,e.get(O).buffer))}function l(){return i.createVertexArray()}function c(C){return i.bindVertexArray(C)}function u(C){return i.deleteVertexArray(C)}function f(C,I,P){const U=P.wireframe===!0;let O=n[C.id];O===void 0&&(O={},n[C.id]=O);let k=O[I.id];k===void 0&&(k={},O[I.id]=k);let z=k[U];return z===void 0&&(z=d(l()),k[U]=z),z}function d(C){const I=[],P=[],U=[];for(let O=0;O<t;O++)I[O]=0,P[O]=0,U[O]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:I,enabledAttributes:P,attributeDivisors:U,object:C,attributes:{},index:null}}function h(C,I,P,U){const O=r.attributes,k=I.attributes;let z=0;const W=P.getAttributes();for(const H in W)if(W[H].location>=0){const X=O[H];let Z=k[H];if(Z===void 0&&(H==="instanceMatrix"&&C.instanceMatrix&&(Z=C.instanceMatrix),H==="instanceColor"&&C.instanceColor&&(Z=C.instanceColor)),X===void 0||X.attribute!==Z||Z&&X.data!==Z.data)return!0;z++}return r.attributesNum!==z||r.index!==U}function g(C,I,P,U){const O={},k=I.attributes;let z=0;const W=P.getAttributes();for(const H in W)if(W[H].location>=0){let X=k[H];X===void 0&&(H==="instanceMatrix"&&C.instanceMatrix&&(X=C.instanceMatrix),H==="instanceColor"&&C.instanceColor&&(X=C.instanceColor));const Z={};Z.attribute=X,X&&X.data&&(Z.data=X.data),O[H]=Z,z++}r.attributes=O,r.attributesNum=z,r.index=U}function p(){const C=r.newAttributes;for(let I=0,P=C.length;I<P;I++)C[I]=0}function x(C){m(C,0)}function m(C,I){const P=r.newAttributes,U=r.enabledAttributes,O=r.attributeDivisors;P[C]=1,U[C]===0&&(i.enableVertexAttribArray(C),U[C]=1),O[C]!==I&&(i.vertexAttribDivisor(C,I),O[C]=I)}function _(){const C=r.newAttributes,I=r.enabledAttributes;for(let P=0,U=I.length;P<U;P++)I[P]!==C[P]&&(i.disableVertexAttribArray(P),I[P]=0)}function A(C,I,P,U,O,k,z){z===!0?i.vertexAttribIPointer(C,I,P,O,k):i.vertexAttribPointer(C,I,P,U,O,k)}function S(C,I,P,U){p();const O=U.attributes,k=P.getAttributes(),z=I.defaultAttributeValues;for(const W in k){const H=k[W];if(H.location>=0){let $=O[W];if($===void 0&&(W==="instanceMatrix"&&C.instanceMatrix&&($=C.instanceMatrix),W==="instanceColor"&&C.instanceColor&&($=C.instanceColor)),$!==void 0){const X=$.normalized,Z=$.itemSize,pe=e.get($);if(pe===void 0)continue;const ge=pe.buffer,be=pe.type,Me=pe.bytesPerElement,Y=be===i.INT||be===i.UNSIGNED_INT||$.gpuType===Yf;if($.isInterleavedBufferAttribute){const oe=$.data,he=oe.stride,Fe=$.offset;if(oe.isInstancedInterleavedBuffer){for(let ye=0;ye<H.locationSize;ye++)m(H.location+ye,oe.meshPerAttribute);C.isInstancedMesh!==!0&&U._maxInstanceCount===void 0&&(U._maxInstanceCount=oe.meshPerAttribute*oe.count)}else for(let ye=0;ye<H.locationSize;ye++)x(H.location+ye);i.bindBuffer(i.ARRAY_BUFFER,ge);for(let ye=0;ye<H.locationSize;ye++)A(H.location+ye,Z/H.locationSize,be,X,he*Me,(Fe+Z/H.locationSize*ye)*Me,Y)}else{if($.isInstancedBufferAttribute){for(let oe=0;oe<H.locationSize;oe++)m(H.location+oe,$.meshPerAttribute);C.isInstancedMesh!==!0&&U._maxInstanceCount===void 0&&(U._maxInstanceCount=$.meshPerAttribute*$.count)}else for(let oe=0;oe<H.locationSize;oe++)x(H.location+oe);i.bindBuffer(i.ARRAY_BUFFER,ge);for(let oe=0;oe<H.locationSize;oe++)A(H.location+oe,Z/H.locationSize,be,X,Z*Me,Z/H.locationSize*oe*Me,Y)}}else if(z!==void 0){const X=z[W];if(X!==void 0)switch(X.length){case 2:i.vertexAttrib2fv(H.location,X);break;case 3:i.vertexAttrib3fv(H.location,X);break;case 4:i.vertexAttrib4fv(H.location,X);break;default:i.vertexAttrib1fv(H.location,X)}}}}_()}function v(){E();for(const C in n){const I=n[C];for(const P in I){const U=I[P];for(const O in U)u(U[O].object),delete U[O];delete I[P]}delete n[C]}}function b(C){if(n[C.id]===void 0)return;const I=n[C.id];for(const P in I){const U=I[P];for(const O in U)u(U[O].object),delete U[O];delete I[P]}delete n[C.id]}function M(C){for(const I in n){const P=n[I];if(P[C.id]===void 0)continue;const U=P[C.id];for(const O in U)u(U[O].object),delete U[O];delete P[C.id]}}function E(){y(),o=!0,r!==s&&(r=s,c(r.object))}function y(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:a,reset:E,resetDefaultState:y,dispose:v,releaseStatesOfGeometry:b,releaseStatesOfProgram:M,initAttributes:p,enableAttribute:x,disableUnusedAttributes:_}}function eM(i,e,t){let n;function s(c){n=c}function r(c,u){i.drawArrays(n,c,u),t.update(u,n,1)}function o(c,u,f){f!==0&&(i.drawArraysInstanced(n,c,u,f),t.update(u,n,f))}function a(c,u,f){if(f===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,c,0,u,0,f);let h=0;for(let g=0;g<f;g++)h+=u[g];t.update(h,n,1)}function l(c,u,f,d){if(f===0)return;const h=e.get("WEBGL_multi_draw");if(h===null)for(let g=0;g<c.length;g++)o(c[g],u[g],d[g]);else{h.multiDrawArraysInstancedWEBGL(n,c,0,u,0,d,0,f);let g=0;for(let p=0;p<f;p++)g+=u[p]*d[p];t.update(g,n,1)}}this.setMode=s,this.render=r,this.renderInstances=o,this.renderMultiDraw=a,this.renderMultiDrawInstances=l}function tM(i,e,t,n){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){const M=e.get("EXT_texture_filter_anisotropic");s=i.getParameter(M.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function o(M){return!(M!==gn&&n.convert(M)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function a(M){const E=M===pr&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(M!==Ui&&n.convert(M)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE)&&M!==mi&&!E)}function l(M){if(M==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";M="mediump"}return M==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let c=t.precision!==void 0?t.precision:"highp";const u=l(c);u!==c&&(je("WebGLRenderer:",c,"not supported, using",u,"instead."),c=u);const f=t.logarithmicDepthBuffer===!0,d=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),h=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),g=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),p=i.getParameter(i.MAX_TEXTURE_SIZE),x=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),m=i.getParameter(i.MAX_VERTEX_ATTRIBS),_=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),A=i.getParameter(i.MAX_VARYING_VECTORS),S=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),v=g>0,b=i.getParameter(i.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:l,textureFormatReadable:o,textureTypeReadable:a,precision:c,logarithmicDepthBuffer:f,reversedDepthBuffer:d,maxTextures:h,maxVertexTextures:g,maxTextureSize:p,maxCubemapSize:x,maxAttributes:m,maxVertexUniforms:_,maxVaryings:A,maxFragmentUniforms:S,vertexTextures:v,maxSamples:b}}function nM(i){const e=this;let t=null,n=0,s=!1,r=!1;const o=new vs,a=new Qe,l={value:null,needsUpdate:!1};this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(f,d){const h=f.length!==0||d||n!==0||s;return s=d,n=f.length,h},this.beginShadows=function(){r=!0,u(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(f,d){t=u(f,d,0)},this.setState=function(f,d,h){const g=f.clippingPlanes,p=f.clipIntersection,x=f.clipShadows,m=i.get(f);if(!s||g===null||g.length===0||r&&!x)r?u(null):c();else{const _=r?0:n,A=_*4;let S=m.clippingState||null;l.value=S,S=u(g,d,A,h);for(let v=0;v!==A;++v)S[v]=t[v];m.clippingState=S,this.numIntersection=p?this.numPlanes:0,this.numPlanes+=_}};function c(){l.value!==t&&(l.value=t,l.needsUpdate=n>0),e.numPlanes=n,e.numIntersection=0}function u(f,d,h,g){const p=f!==null?f.length:0;let x=null;if(p!==0){if(x=l.value,g!==!0||x===null){const m=h+p*4,_=d.matrixWorldInverse;a.getNormalMatrix(_),(x===null||x.length<m)&&(x=new Float32Array(m));for(let A=0,S=h;A!==p;++A,S+=4)o.copy(f[A]).applyMatrix4(_,a),o.normal.toArray(x,S),x[S+3]=o.constant}l.value=x,l.needsUpdate=!0}return e.numPlanes=p,e.numIntersection=0,x}}function iM(i){let e=new WeakMap;function t(o,a){return a===Ru?o.mapping=ao:a===Iu&&(o.mapping=lo),o}function n(o){if(o&&o.isTexture){const a=o.mapping;if(a===Ru||a===Iu)if(e.has(o)){const l=e.get(o).texture;return t(l,o.mapping)}else{const l=o.image;if(l&&l.height>0){const c=new iv(l.height);return c.fromEquirectangularTexture(i,o),e.set(o,c),o.addEventListener("dispose",s),t(c.texture,o.mapping)}else return null}}return o}function s(o){const a=o.target;a.removeEventListener("dispose",s);const l=e.get(a);l!==void 0&&(e.delete(a),l.dispose())}function r(){e=new WeakMap}return{get:n,dispose:r}}const Ms=4,ep=[.125,.215,.35,.446,.526,.582],ir=20,sM=256,Do=new id,tp=new nt;let Vc=null,Gc=0,Wc=0,Xc=!1;const rM=new L;class np{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,n=.1,s=100,r={}){const{size:o=256,position:a=rM}=r;Vc=this._renderer.getRenderTarget(),Gc=this._renderer.getActiveCubeFace(),Wc=this._renderer.getActiveMipmapLevel(),Xc=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(o);const l=this._allocateTargets();return l.depthBuffer=!0,this._sceneToCubeUV(e,n,s,l,a),t>0&&this._blur(l,0,0,t),this._applyPMREM(l),this._cleanup(l),l}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=rp(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=sp(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(Vc,Gc,Wc),this._renderer.xr.enabled=Xc,e.scissorTest=!1,Lr(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===ao||e.mapping===lo?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),Vc=this._renderer.getRenderTarget(),Gc=this._renderer.getActiveCubeFace(),Wc=this._renderer.getActiveMipmapLevel(),Xc=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const n=t||this._allocateTargets();return this._textureToCubeUV(e,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,n={magFilter:ii,minFilter:ii,generateMipmaps:!1,type:pr,format:gn,colorSpace:uo,depthBuffer:!1},s=ip(e,t,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=ip(e,t,n);const{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=oM(r)),this._blurMaterial=lM(r,e,t),this._ggxMaterial=aM(r,e,t)}return s}_compileMaterial(e){const t=new Vt(new An,e);this._renderer.compile(t,Do)}_sceneToCubeUV(e,t,n,s,r){const l=new ei(90,1,t,n),c=[1,-1,1,1,1,1],u=[1,1,1,-1,-1,-1],f=this._renderer,d=f.autoClear,h=f.toneMapping;f.getClearColor(tp),f.toneMapping=Ds,f.autoClear=!1,f.state.buffers.depth.getReversed()&&(f.setRenderTarget(s),f.clearDepth(),f.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new Vt(new vo,new hr({name:"PMREM.Background",side:En,depthWrite:!1,depthTest:!1})));const p=this._backgroundBox,x=p.material;let m=!1;const _=e.background;_?_.isColor&&(x.color.copy(_),e.background=null,m=!0):(x.color.copy(tp),m=!0);for(let A=0;A<6;A++){const S=A%3;S===0?(l.up.set(0,c[A],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x+u[A],r.y,r.z)):S===1?(l.up.set(0,0,c[A]),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y+u[A],r.z)):(l.up.set(0,c[A],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y,r.z+u[A]));const v=this._cubeSize;Lr(s,S*v,A>2?v:0,v,v),f.setRenderTarget(s),m&&f.render(p,l),f.render(e,l)}f.toneMapping=h,f.autoClear=d,e.background=_}_textureToCubeUV(e,t){const n=this._renderer,s=e.mapping===ao||e.mapping===lo;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=rp()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=sp());const r=s?this._cubemapMaterial:this._equirectMaterial,o=this._lodMeshes[0];o.material=r;const a=r.uniforms;a.envMap.value=e;const l=this._cubeSize;Lr(t,0,0,3*l,2*l),n.setRenderTarget(t),n.render(o,Do)}_applyPMREM(e){const t=this._renderer,n=t.autoClear;t.autoClear=!1;const s=this._lodMeshes.length;for(let r=1;r<s;r++)this._applyGGXFilter(e,r-1,r);t.autoClear=n}_applyGGXFilter(e,t,n){const s=this._renderer,r=this._pingPongRenderTarget,o=this._ggxMaterial,a=this._lodMeshes[n];a.material=o;const l=o.uniforms,c=n/(this._lodMeshes.length-1),u=t/(this._lodMeshes.length-1),f=Math.sqrt(c*c-u*u),d=.05+c*.95,h=f*d,{_lodMax:g}=this,p=this._sizeLods[n],x=3*p*(n>g-Ms?n-g+Ms:0),m=4*(this._cubeSize-p);l.envMap.value=e.texture,l.roughness.value=h,l.mipInt.value=g-t,Lr(r,x,m,3*p,2*p),s.setRenderTarget(r),s.render(a,Do),l.envMap.value=r.texture,l.roughness.value=0,l.mipInt.value=g-n,Lr(e,x,m,3*p,2*p),s.setRenderTarget(e),s.render(a,Do)}_blur(e,t,n,s,r){const o=this._pingPongRenderTarget;this._halfBlur(e,o,t,n,s,"latitudinal",r),this._halfBlur(o,e,n,n,s,"longitudinal",r)}_halfBlur(e,t,n,s,r,o,a){const l=this._renderer,c=this._blurMaterial;o!=="latitudinal"&&o!=="longitudinal"&&zt("blur direction must be either latitudinal or longitudinal!");const u=3,f=this._lodMeshes[s];f.material=c;const d=c.uniforms,h=this._sizeLods[n]-1,g=isFinite(r)?Math.PI/(2*h):2*Math.PI/(2*ir-1),p=r/g,x=isFinite(r)?1+Math.floor(u*p):ir;x>ir&&je(`sigmaRadians, ${r}, is too large and will clip, as it requested ${x} samples when the maximum is set to ${ir}`);const m=[];let _=0;for(let M=0;M<ir;++M){const E=M/p,y=Math.exp(-E*E/2);m.push(y),M===0?_+=y:M<x&&(_+=2*y)}for(let M=0;M<m.length;M++)m[M]=m[M]/_;d.envMap.value=e.texture,d.samples.value=x,d.weights.value=m,d.latitudinal.value=o==="latitudinal",a&&(d.poleAxis.value=a);const{_lodMax:A}=this;d.dTheta.value=g,d.mipInt.value=A-n;const S=this._sizeLods[s],v=3*S*(s>A-Ms?s-A+Ms:0),b=4*(this._cubeSize-S);Lr(t,v,b,3*S,2*S),l.setRenderTarget(t),l.render(f,Do)}}function oM(i){const e=[],t=[],n=[];let s=i;const r=i-Ms+1+ep.length;for(let o=0;o<r;o++){const a=Math.pow(2,s);e.push(a);let l=1/a;o>i-Ms?l=ep[o-i+Ms-1]:o===0&&(l=0),t.push(l);const c=1/(a-2),u=-c,f=1+c,d=[u,u,f,u,f,f,u,u,f,f,u,f],h=6,g=6,p=3,x=2,m=1,_=new Float32Array(p*g*h),A=new Float32Array(x*g*h),S=new Float32Array(m*g*h);for(let b=0;b<h;b++){const M=b%3*2/3-1,E=b>2?0:-1,y=[M,E,0,M+2/3,E,0,M+2/3,E+1,0,M,E,0,M+2/3,E+1,0,M,E+1,0];_.set(y,p*g*b),A.set(d,x*g*b);const C=[b,b,b,b,b,b];S.set(C,m*g*b)}const v=new An;v.setAttribute("position",new li(_,p)),v.setAttribute("uv",new li(A,x)),v.setAttribute("faceIndex",new li(S,m)),n.push(new Vt(v,null)),s>Ms&&s--}return{lodMeshes:n,sizeLods:e,sigmas:t}}function ip(i,e,t){const n=new Us(i,e,t);return n.texture.mapping=Kl,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function Lr(i,e,t,n,s){i.viewport.set(e,t,n,s),i.scissor.set(e,t,n,s)}function aM(i,e,t){return new _n({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:sM,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:Zl(),fragmentShader:`

			precision highp float;
			precision highp int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float roughness;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359

			// Van der Corput radical inverse
			float radicalInverse_VdC(uint bits) {
				bits = (bits << 16u) | (bits >> 16u);
				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
				return float(bits) * 2.3283064365386963e-10; // / 0x100000000
			}

			// Hammersley sequence
			vec2 hammersley(uint i, uint N) {
				return vec2(float(i) / float(N), radicalInverse_VdC(i));
			}

			// GGX VNDF importance sampling (Eric Heitz 2018)
			// "Sampling the GGX Distribution of Visible Normals"
			// https://jcgt.org/published/0007/04/01/
			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {
				float alpha = roughness * roughness;

				// Section 3.2: Transform view direction to hemisphere configuration
				vec3 Vh = normalize(vec3(alpha * V.x, alpha * V.y, V.z));

				// Section 4.1: Orthonormal basis
				float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
				vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
				vec3 T2 = cross(Vh, T1);

				// Section 4.2: Parameterization of projected area
				float r = sqrt(Xi.x);
				float phi = 2.0 * PI * Xi.y;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + Vh.z);
				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

				// Section 4.3: Reprojection onto hemisphere
				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

				// Section 3.4: Transform back to ellipsoid configuration
				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
			}

			void main() {
				vec3 N = normalize(vOutputDirection);
				vec3 V = N; // Assume view direction equals normal for pre-filtering

				vec3 prefilteredColor = vec3(0.0);
				float totalWeight = 0.0;

				// For very low roughness, just sample the environment directly
				if (roughness < 0.001) {
					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);
					return;
				}

				// Tangent space basis for VNDF sampling
				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
				vec3 tangent = normalize(cross(up, N));
				vec3 bitangent = cross(N, tangent);

				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {
					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));

					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)
					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);

					// Transform H back to world space
					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
					vec3 L = normalize(2.0 * dot(V, H) * H - V);

					float NdotL = max(dot(N, L), 0.0);

					if(NdotL > 0.0) {
						// Sample environment at fixed mip level
						// VNDF importance sampling handles the distribution filtering
						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);

						// Weight by NdotL for the split-sum approximation
						// VNDF PDF naturally accounts for the visible microfacet distribution
						prefilteredColor += sampleColor * NdotL;
						totalWeight += NdotL;
					}
				}

				if (totalWeight > 0.0) {
					prefilteredColor = prefilteredColor / totalWeight;
				}

				gl_FragColor = vec4(prefilteredColor, 1.0);
			}
		`,blending:is,depthTest:!1,depthWrite:!1})}function lM(i,e,t){const n=new Float32Array(ir),s=new L(0,1,0);return new _n({name:"SphericalGaussianBlur",defines:{n:ir,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:Zl(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:is,depthTest:!1,depthWrite:!1})}function sp(){return new _n({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Zl(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:is,depthTest:!1,depthWrite:!1})}function rp(){return new _n({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Zl(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:is,depthTest:!1,depthWrite:!1})}function Zl(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}function cM(i){let e=new WeakMap,t=null;function n(a){if(a&&a.isTexture){const l=a.mapping,c=l===Ru||l===Iu,u=l===ao||l===lo;if(c||u){let f=e.get(a);const d=f!==void 0?f.texture.pmremVersion:0;if(a.isRenderTargetTexture&&a.pmremVersion!==d)return t===null&&(t=new np(i)),f=c?t.fromEquirectangular(a,f):t.fromCubemap(a,f),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),f.texture;if(f!==void 0)return f.texture;{const h=a.image;return c&&h&&h.height>0||u&&h&&s(h)?(t===null&&(t=new np(i)),f=c?t.fromEquirectangular(a):t.fromCubemap(a),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),a.addEventListener("dispose",r),f.texture):null}}}return a}function s(a){let l=0;const c=6;for(let u=0;u<c;u++)a[u]!==void 0&&l++;return l===c}function r(a){const l=a.target;l.removeEventListener("dispose",r);const c=e.get(l);c!==void 0&&(e.delete(l),c.dispose())}function o(){e=new WeakMap,t!==null&&(t.dispose(),t=null)}return{get:n,dispose:o}}function uM(i){const e={};function t(n){if(e[n]!==void 0)return e[n];const s=i.getExtension(n);return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(n){const s=t(n);return s===null&&la("WebGLRenderer: "+n+" extension not supported."),s}}}function fM(i,e,t,n){const s={},r=new WeakMap;function o(f){const d=f.target;d.index!==null&&e.remove(d.index);for(const g in d.attributes)e.remove(d.attributes[g]);d.removeEventListener("dispose",o),delete s[d.id];const h=r.get(d);h&&(e.remove(h),r.delete(d)),n.releaseStatesOfGeometry(d),d.isInstancedBufferGeometry===!0&&delete d._maxInstanceCount,t.memory.geometries--}function a(f,d){return s[d.id]===!0||(d.addEventListener("dispose",o),s[d.id]=!0,t.memory.geometries++),d}function l(f){const d=f.attributes;for(const h in d)e.update(d[h],i.ARRAY_BUFFER)}function c(f){const d=[],h=f.index,g=f.attributes.position;let p=0;if(h!==null){const _=h.array;p=h.version;for(let A=0,S=_.length;A<S;A+=3){const v=_[A+0],b=_[A+1],M=_[A+2];d.push(v,b,b,M,M,v)}}else if(g!==void 0){const _=g.array;p=g.version;for(let A=0,S=_.length/3-1;A<S;A+=3){const v=A+0,b=A+1,M=A+2;d.push(v,b,b,M,M,v)}}else return;const x=new(B0(d)?H0:k0)(d,1);x.version=p;const m=r.get(f);m&&e.remove(m),r.set(f,x)}function u(f){const d=r.get(f);if(d){const h=f.index;h!==null&&d.version<h.version&&c(f)}else c(f);return r.get(f)}return{get:a,update:l,getWireframeAttribute:u}}function dM(i,e,t){let n;function s(d){n=d}let r,o;function a(d){r=d.type,o=d.bytesPerElement}function l(d,h){i.drawElements(n,h,r,d*o),t.update(h,n,1)}function c(d,h,g){g!==0&&(i.drawElementsInstanced(n,h,r,d*o,g),t.update(h,n,g))}function u(d,h,g){if(g===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,h,0,r,d,0,g);let x=0;for(let m=0;m<g;m++)x+=h[m];t.update(x,n,1)}function f(d,h,g,p){if(g===0)return;const x=e.get("WEBGL_multi_draw");if(x===null)for(let m=0;m<d.length;m++)c(d[m]/o,h[m],p[m]);else{x.multiDrawElementsInstancedWEBGL(n,h,0,r,d,0,p,0,g);let m=0;for(let _=0;_<g;_++)m+=h[_]*p[_];t.update(m,n,1)}}this.setMode=s,this.setIndex=a,this.render=l,this.renderInstances=c,this.renderMultiDraw=u,this.renderMultiDrawInstances=f}function hM(i){const e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function n(r,o,a){switch(t.calls++,o){case i.TRIANGLES:t.triangles+=a*(r/3);break;case i.LINES:t.lines+=a*(r/2);break;case i.LINE_STRIP:t.lines+=a*(r-1);break;case i.LINE_LOOP:t.lines+=a*r;break;case i.POINTS:t.points+=a*r;break;default:zt("WebGLInfo: Unknown draw mode:",o);break}}function s(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:s,update:n}}function pM(i,e,t){const n=new WeakMap,s=new Et;function r(o,a,l){const c=o.morphTargetInfluences,u=a.morphAttributes.position||a.morphAttributes.normal||a.morphAttributes.color,f=u!==void 0?u.length:0;let d=n.get(a);if(d===void 0||d.count!==f){let C=function(){E.dispose(),n.delete(a),a.removeEventListener("dispose",C)};var h=C;d!==void 0&&d.texture.dispose();const g=a.morphAttributes.position!==void 0,p=a.morphAttributes.normal!==void 0,x=a.morphAttributes.color!==void 0,m=a.morphAttributes.position||[],_=a.morphAttributes.normal||[],A=a.morphAttributes.color||[];let S=0;g===!0&&(S=1),p===!0&&(S=2),x===!0&&(S=3);let v=a.attributes.position.count*S,b=1;v>e.maxTextureSize&&(b=Math.ceil(v/e.maxTextureSize),v=e.maxTextureSize);const M=new Float32Array(v*b*4*f),E=new O0(M,v,b,f);E.type=mi,E.needsUpdate=!0;const y=S*4;for(let I=0;I<f;I++){const P=m[I],U=_[I],O=A[I],k=v*b*4*I;for(let z=0;z<P.count;z++){const W=z*y;g===!0&&(s.fromBufferAttribute(P,z),M[k+W+0]=s.x,M[k+W+1]=s.y,M[k+W+2]=s.z,M[k+W+3]=0),p===!0&&(s.fromBufferAttribute(U,z),M[k+W+4]=s.x,M[k+W+5]=s.y,M[k+W+6]=s.z,M[k+W+7]=0),x===!0&&(s.fromBufferAttribute(O,z),M[k+W+8]=s.x,M[k+W+9]=s.y,M[k+W+10]=s.z,M[k+W+11]=O.itemSize===4?s.w:1)}}d={count:f,texture:E,size:new Ne(v,b)},n.set(a,d),a.addEventListener("dispose",C)}if(o.isInstancedMesh===!0&&o.morphTexture!==null)l.getUniforms().setValue(i,"morphTexture",o.morphTexture,t);else{let g=0;for(let x=0;x<c.length;x++)g+=c[x];const p=a.morphTargetsRelative?1:1-g;l.getUniforms().setValue(i,"morphTargetBaseInfluence",p),l.getUniforms().setValue(i,"morphTargetInfluences",c)}l.getUniforms().setValue(i,"morphTargetsTexture",d.texture,t),l.getUniforms().setValue(i,"morphTargetsTextureSize",d.size)}return{update:r}}function mM(i,e,t,n){let s=new WeakMap;function r(l){const c=n.render.frame,u=l.geometry,f=e.get(l,u);if(s.get(f)!==c&&(e.update(f),s.set(f,c)),l.isInstancedMesh&&(l.hasEventListener("dispose",a)===!1&&l.addEventListener("dispose",a),s.get(l)!==c&&(t.update(l.instanceMatrix,i.ARRAY_BUFFER),l.instanceColor!==null&&t.update(l.instanceColor,i.ARRAY_BUFFER),s.set(l,c))),l.isSkinnedMesh){const d=l.skeleton;s.get(d)!==c&&(d.update(),s.set(d,c))}return f}function o(){s=new WeakMap}function a(l){const c=l.target;c.removeEventListener("dispose",a),t.remove(c.instanceMatrix),c.instanceColor!==null&&t.remove(c.instanceColor)}return{update:r,dispose:o}}const Y0=new xn,op=new td(1,1),K0=new O0,j0=new OS,$0=new W0,ap=[],lp=[],cp=new Float32Array(16),up=new Float32Array(9),fp=new Float32Array(4);function yo(i,e,t){const n=i[0];if(n<=0||n>0)return i;const s=e*t;let r=ap[s];if(r===void 0&&(r=new Float32Array(s),ap[s]=r),e!==0){n.toArray(r,0);for(let o=1,a=0;o!==e;++o)a+=t,i[o].toArray(r,a)}return r}function Qt(i,e){if(i.length!==e.length)return!1;for(let t=0,n=i.length;t<n;t++)if(i[t]!==e[t])return!1;return!0}function Yt(i,e){for(let t=0,n=e.length;t<n;t++)i[t]=e[t]}function Jl(i,e){let t=lp[e];t===void 0&&(t=new Int32Array(e),lp[e]=t);for(let n=0;n!==e;++n)t[n]=i.allocateTextureUnit();return t}function gM(i,e){const t=this.cache;t[0]!==e&&(i.uniform1f(this.addr,e),t[0]=e)}function xM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Qt(t,e))return;i.uniform2fv(this.addr,e),Yt(t,e)}}function _M(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(i.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(Qt(t,e))return;i.uniform3fv(this.addr,e),Yt(t,e)}}function AM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Qt(t,e))return;i.uniform4fv(this.addr,e),Yt(t,e)}}function SM(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Qt(t,e))return;i.uniformMatrix2fv(this.addr,!1,e),Yt(t,e)}else{if(Qt(t,n))return;fp.set(n),i.uniformMatrix2fv(this.addr,!1,fp),Yt(t,n)}}function vM(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Qt(t,e))return;i.uniformMatrix3fv(this.addr,!1,e),Yt(t,e)}else{if(Qt(t,n))return;up.set(n),i.uniformMatrix3fv(this.addr,!1,up),Yt(t,n)}}function yM(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Qt(t,e))return;i.uniformMatrix4fv(this.addr,!1,e),Yt(t,e)}else{if(Qt(t,n))return;cp.set(n),i.uniformMatrix4fv(this.addr,!1,cp),Yt(t,n)}}function bM(i,e){const t=this.cache;t[0]!==e&&(i.uniform1i(this.addr,e),t[0]=e)}function MM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Qt(t,e))return;i.uniform2iv(this.addr,e),Yt(t,e)}}function CM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Qt(t,e))return;i.uniform3iv(this.addr,e),Yt(t,e)}}function TM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Qt(t,e))return;i.uniform4iv(this.addr,e),Yt(t,e)}}function EM(i,e){const t=this.cache;t[0]!==e&&(i.uniform1ui(this.addr,e),t[0]=e)}function wM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Qt(t,e))return;i.uniform2uiv(this.addr,e),Yt(t,e)}}function RM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Qt(t,e))return;i.uniform3uiv(this.addr,e),Yt(t,e)}}function IM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Qt(t,e))return;i.uniform4uiv(this.addr,e),Yt(t,e)}}function DM(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s);let r;this.type===i.SAMPLER_2D_SHADOW?(op.compareFunction=L0,r=op):r=Y0,t.setTexture2D(e||r,s)}function PM(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture3D(e||j0,s)}function FM(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTextureCube(e||$0,s)}function LM(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture2DArray(e||K0,s)}function BM(i){switch(i){case 5126:return gM;case 35664:return xM;case 35665:return _M;case 35666:return AM;case 35674:return SM;case 35675:return vM;case 35676:return yM;case 5124:case 35670:return bM;case 35667:case 35671:return MM;case 35668:case 35672:return CM;case 35669:case 35673:return TM;case 5125:return EM;case 36294:return wM;case 36295:return RM;case 36296:return IM;case 35678:case 36198:case 36298:case 36306:case 35682:return DM;case 35679:case 36299:case 36307:return PM;case 35680:case 36300:case 36308:case 36293:return FM;case 36289:case 36303:case 36311:case 36292:return LM}}function UM(i,e){i.uniform1fv(this.addr,e)}function OM(i,e){const t=yo(e,this.size,2);i.uniform2fv(this.addr,t)}function NM(i,e){const t=yo(e,this.size,3);i.uniform3fv(this.addr,t)}function zM(i,e){const t=yo(e,this.size,4);i.uniform4fv(this.addr,t)}function kM(i,e){const t=yo(e,this.size,4);i.uniformMatrix2fv(this.addr,!1,t)}function HM(i,e){const t=yo(e,this.size,9);i.uniformMatrix3fv(this.addr,!1,t)}function VM(i,e){const t=yo(e,this.size,16);i.uniformMatrix4fv(this.addr,!1,t)}function GM(i,e){i.uniform1iv(this.addr,e)}function WM(i,e){i.uniform2iv(this.addr,e)}function XM(i,e){i.uniform3iv(this.addr,e)}function qM(i,e){i.uniform4iv(this.addr,e)}function QM(i,e){i.uniform1uiv(this.addr,e)}function YM(i,e){i.uniform2uiv(this.addr,e)}function KM(i,e){i.uniform3uiv(this.addr,e)}function jM(i,e){i.uniform4uiv(this.addr,e)}function $M(i,e,t){const n=this.cache,s=e.length,r=Jl(t,s);Qt(n,r)||(i.uniform1iv(this.addr,r),Yt(n,r));for(let o=0;o!==s;++o)t.setTexture2D(e[o]||Y0,r[o])}function ZM(i,e,t){const n=this.cache,s=e.length,r=Jl(t,s);Qt(n,r)||(i.uniform1iv(this.addr,r),Yt(n,r));for(let o=0;o!==s;++o)t.setTexture3D(e[o]||j0,r[o])}function JM(i,e,t){const n=this.cache,s=e.length,r=Jl(t,s);Qt(n,r)||(i.uniform1iv(this.addr,r),Yt(n,r));for(let o=0;o!==s;++o)t.setTextureCube(e[o]||$0,r[o])}function eC(i,e,t){const n=this.cache,s=e.length,r=Jl(t,s);Qt(n,r)||(i.uniform1iv(this.addr,r),Yt(n,r));for(let o=0;o!==s;++o)t.setTexture2DArray(e[o]||K0,r[o])}function tC(i){switch(i){case 5126:return UM;case 35664:return OM;case 35665:return NM;case 35666:return zM;case 35674:return kM;case 35675:return HM;case 35676:return VM;case 5124:case 35670:return GM;case 35667:case 35671:return WM;case 35668:case 35672:return XM;case 35669:case 35673:return qM;case 5125:return QM;case 36294:return YM;case 36295:return KM;case 36296:return jM;case 35678:case 36198:case 36298:case 36306:case 35682:return $M;case 35679:case 36299:case 36307:return ZM;case 35680:case 36300:case 36308:case 36293:return JM;case 36289:case 36303:case 36311:case 36292:return eC}}class nC{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.setValue=BM(t.type)}}class iC{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=tC(t.type)}}class sC{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,n){const s=this.seq;for(let r=0,o=s.length;r!==o;++r){const a=s[r];a.setValue(e,t[a.id],n)}}}const qc=/(\w+)(\])?(\[|\.)?/g;function dp(i,e){i.seq.push(e),i.map[e.id]=e}function rC(i,e,t){const n=i.name,s=n.length;for(qc.lastIndex=0;;){const r=qc.exec(n),o=qc.lastIndex;let a=r[1];const l=r[2]==="]",c=r[3];if(l&&(a=a|0),c===void 0||c==="["&&o+2===s){dp(t,c===void 0?new nC(a,i,e):new iC(a,i,e));break}else{let f=t.map[a];f===void 0&&(f=new sC(a),dp(t,f)),t=f}}}class gl{constructor(e,t){this.seq=[],this.map={};const n=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let s=0;s<n;++s){const r=e.getActiveUniform(t,s),o=e.getUniformLocation(t,r.name);rC(r,o,this)}}setValue(e,t,n,s){const r=this.map[t];r!==void 0&&r.setValue(e,n,s)}setOptional(e,t,n){const s=t[n];s!==void 0&&this.setValue(e,n,s)}static upload(e,t,n,s){for(let r=0,o=t.length;r!==o;++r){const a=t[r],l=n[a.id];l.needsUpdate!==!1&&a.setValue(e,l.value,s)}}static seqWithValue(e,t){const n=[];for(let s=0,r=e.length;s!==r;++s){const o=e[s];o.id in t&&n.push(o)}return n}}function hp(i,e,t){const n=i.createShader(e);return i.shaderSource(n,t),i.compileShader(n),n}const oC=37297;let aC=0;function lC(i,e){const t=i.split(`
`),n=[],s=Math.max(e-6,0),r=Math.min(e+6,t.length);for(let o=s;o<r;o++){const a=o+1;n.push(`${a===e?">":" "} ${a}: ${t[o]}`)}return n.join(`
`)}const pp=new Qe;function cC(i){rt._getMatrix(pp,rt.workingColorSpace,i);const e=`mat3( ${pp.elements.map(t=>t.toFixed(4))} )`;switch(rt.getTransfer(i)){case Tl:return[e,"LinearTransferOETF"];case ht:return[e,"sRGBTransferOETF"];default:return je("WebGLProgram: Unsupported color space: ",i),[e,"LinearTransferOETF"]}}function mp(i,e,t){const n=i.getShaderParameter(e,i.COMPILE_STATUS),r=(i.getShaderInfoLog(e)||"").trim();if(n&&r==="")return"";const o=/ERROR: 0:(\d+)/.exec(r);if(o){const a=parseInt(o[1]);return t.toUpperCase()+`

`+r+`

`+lC(i.getShaderSource(e),a)}else return r}function uC(i,e){const t=cC(e);return[`vec4 ${i}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}function fC(i,e){let t;switch(e){case cS:t="Linear";break;case uS:t="Reinhard";break;case fS:t="Cineon";break;case dS:t="ACESFilmic";break;case pS:t="AgX";break;case mS:t="Neutral";break;case hS:t="Custom";break;default:je("WebGLProgram: Unsupported toneMapping:",e),t="Linear"}return"vec3 "+i+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}const Za=new L;function dC(){rt.getLuminanceCoefficients(Za);const i=Za.x.toFixed(4),e=Za.y.toFixed(4),t=Za.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function hC(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(Lo).join(`
`)}function pC(i){const e=[];for(const t in i){const n=i[t];n!==!1&&e.push("#define "+t+" "+n)}return e.join(`
`)}function mC(i,e){const t={},n=i.getProgramParameter(e,i.ACTIVE_ATTRIBUTES);for(let s=0;s<n;s++){const r=i.getActiveAttrib(e,s),o=r.name;let a=1;r.type===i.FLOAT_MAT2&&(a=2),r.type===i.FLOAT_MAT3&&(a=3),r.type===i.FLOAT_MAT4&&(a=4),t[o]={type:r.type,location:i.getAttribLocation(e,o),locationSize:a}}return t}function Lo(i){return i!==""}function gp(i,e){const t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return i.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function xp(i,e){return i.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const gC=/^[ \t]*#include +<([\w\d./]+)>/gm;function uf(i){return i.replace(gC,_C)}const xC=new Map;function _C(i,e){let t=Ze[e];if(t===void 0){const n=xC.get(e);if(n!==void 0)t=Ze[n],je('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,n);else throw new Error("Can not resolve #include <"+e+">")}return uf(t)}const AC=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function _p(i){return i.replace(AC,SC)}function SC(i,e,t,n){let s="";for(let r=parseInt(e);r<parseInt(t);r++)s+=n.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function Ap(i){let e=`precision ${i.precision} float;
	precision ${i.precision} int;
	precision ${i.precision} sampler2D;
	precision ${i.precision} samplerCube;
	precision ${i.precision} sampler3D;
	precision ${i.precision} sampler2DArray;
	precision ${i.precision} sampler2DShadow;
	precision ${i.precision} samplerCubeShadow;
	precision ${i.precision} sampler2DArrayShadow;
	precision ${i.precision} isampler2D;
	precision ${i.precision} isampler3D;
	precision ${i.precision} isamplerCube;
	precision ${i.precision} isampler2DArray;
	precision ${i.precision} usampler2D;
	precision ${i.precision} usampler3D;
	precision ${i.precision} usamplerCube;
	precision ${i.precision} usampler2DArray;
	`;return i.precision==="highp"?e+=`
#define HIGH_PRECISION`:i.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:i.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}function vC(i){let e="SHADOWMAP_TYPE_BASIC";return i.shadowMapType===b0?e="SHADOWMAP_TYPE_PCF":i.shadowMapType===GA?e="SHADOWMAP_TYPE_PCF_SOFT":i.shadowMapType===qi&&(e="SHADOWMAP_TYPE_VSM"),e}function yC(i){let e="ENVMAP_TYPE_CUBE";if(i.envMap)switch(i.envMapMode){case ao:case lo:e="ENVMAP_TYPE_CUBE";break;case Kl:e="ENVMAP_TYPE_CUBE_UV";break}return e}function bC(i){let e="ENVMAP_MODE_REFLECTION";return i.envMap&&i.envMapMode===lo&&(e="ENVMAP_MODE_REFRACTION"),e}function MC(i){let e="ENVMAP_BLENDING_NONE";if(i.envMap)switch(i.combine){case C0:e="ENVMAP_BLENDING_MULTIPLY";break;case aS:e="ENVMAP_BLENDING_MIX";break;case lS:e="ENVMAP_BLENDING_ADD";break}return e}function CC(i){const e=i.envMapCubeUVHeight;if(e===null)return null;const t=Math.log2(e)-2,n=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:n,maxMip:t}}function TC(i,e,t,n){const s=i.getContext(),r=t.defines;let o=t.vertexShader,a=t.fragmentShader;const l=vC(t),c=yC(t),u=bC(t),f=MC(t),d=CC(t),h=hC(t),g=pC(r),p=s.createProgram();let x,m,_=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(x=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g].filter(Lo).join(`
`),x.length>0&&(x+=`
`),m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g].filter(Lo).join(`
`),m.length>0&&(m+=`
`)):(x=[Ap(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+u:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(Lo).join(`
`),m=[Ap(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+c:"",t.envMap?"#define "+u:"",t.envMap?"#define "+f:"",d?"#define CUBEUV_TEXEL_WIDTH "+d.texelWidth:"",d?"#define CUBEUV_TEXEL_HEIGHT "+d.texelHeight:"",d?"#define CUBEUV_MAX_MIP "+d.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor||t.batchingColor?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==Ds?"#define TONE_MAPPING":"",t.toneMapping!==Ds?Ze.tonemapping_pars_fragment:"",t.toneMapping!==Ds?fC("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",Ze.colorspace_pars_fragment,uC("linearToOutputTexel",t.outputColorSpace),dC(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(Lo).join(`
`)),o=uf(o),o=gp(o,t),o=xp(o,t),a=uf(a),a=gp(a,t),a=xp(a,t),o=_p(o),a=_p(a),t.isRawShaderMaterial!==!0&&(_=`#version 300 es
`,x=[h,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+x,m=["#define varying in",t.glslVersion===Dh?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===Dh?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+m);const A=_+x+o,S=_+m+a,v=hp(s,s.VERTEX_SHADER,A),b=hp(s,s.FRAGMENT_SHADER,S);s.attachShader(p,v),s.attachShader(p,b),t.index0AttributeName!==void 0?s.bindAttribLocation(p,0,t.index0AttributeName):t.morphTargets===!0&&s.bindAttribLocation(p,0,"position"),s.linkProgram(p);function M(I){if(i.debug.checkShaderErrors){const P=s.getProgramInfoLog(p)||"",U=s.getShaderInfoLog(v)||"",O=s.getShaderInfoLog(b)||"",k=P.trim(),z=U.trim(),W=O.trim();let H=!0,$=!0;if(s.getProgramParameter(p,s.LINK_STATUS)===!1)if(H=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(s,p,v,b);else{const X=mp(s,v,"vertex"),Z=mp(s,b,"fragment");zt("THREE.WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(p,s.VALIDATE_STATUS)+`

Material Name: `+I.name+`
Material Type: `+I.type+`

Program Info Log: `+k+`
`+X+`
`+Z)}else k!==""?je("WebGLProgram: Program Info Log:",k):(z===""||W==="")&&($=!1);$&&(I.diagnostics={runnable:H,programLog:k,vertexShader:{log:z,prefix:x},fragmentShader:{log:W,prefix:m}})}s.deleteShader(v),s.deleteShader(b),E=new gl(s,p),y=mC(s,p)}let E;this.getUniforms=function(){return E===void 0&&M(this),E};let y;this.getAttributes=function(){return y===void 0&&M(this),y};let C=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return C===!1&&(C=s.getProgramParameter(p,oC)),C},this.destroy=function(){n.releaseStatesOfProgram(this),s.deleteProgram(p),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=aC++,this.cacheKey=e,this.usedTimes=1,this.program=p,this.vertexShader=v,this.fragmentShader=b,this}let EC=0;class wC{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const t=e.vertexShader,n=e.fragmentShader,s=this._getShaderStage(t),r=this._getShaderStage(n),o=this._getShaderCacheForMaterial(e);return o.has(s)===!1&&(o.add(s),s.usedTimes++),o.has(r)===!1&&(o.add(r),r.usedTimes++),this}remove(e){const t=this.materialCache.get(e);for(const n of t)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const t=this.materialCache;let n=t.get(e);return n===void 0&&(n=new Set,t.set(e,n)),n}_getShaderStage(e){const t=this.shaderCache;let n=t.get(e);return n===void 0&&(n=new RC(e),t.set(e,n)),n}}class RC{constructor(e){this.id=EC++,this.code=e,this.usedTimes=0}}function IC(i,e,t,n,s,r,o){const a=new N0,l=new wC,c=new Set,u=[],f=s.logarithmicDepthBuffer,d=s.vertexTextures;let h=s.precision;const g={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function p(y){return c.add(y),y===0?"uv":`uv${y}`}function x(y,C,I,P,U){const O=P.fog,k=U.geometry,z=y.isMeshStandardMaterial?P.environment:null,W=(y.isMeshStandardMaterial?t:e).get(y.envMap||z),H=W&&W.mapping===Kl?W.image.height:null,$=g[y.type];y.precision!==null&&(h=s.getMaxPrecision(y.precision),h!==y.precision&&je("WebGLProgram.getParameters:",y.precision,"not supported, using",h,"instead."));const X=k.morphAttributes.position||k.morphAttributes.normal||k.morphAttributes.color,Z=X!==void 0?X.length:0;let pe=0;k.morphAttributes.position!==void 0&&(pe=1),k.morphAttributes.normal!==void 0&&(pe=2),k.morphAttributes.color!==void 0&&(pe=3);let ge,be,Me,Y;if($){const ut=Ci[$];ge=ut.vertexShader,be=ut.fragmentShader}else ge=y.vertexShader,be=y.fragmentShader,l.update(y),Me=l.getVertexShaderID(y),Y=l.getFragmentShaderID(y);const oe=i.getRenderTarget(),he=i.state.buffers.depth.getReversed(),Fe=U.isInstancedMesh===!0,ye=U.isBatchedMesh===!0,ke=!!y.map,F=!!y.matcap,B=!!W,q=!!y.aoMap,w=!!y.lightMap,ne=!!y.bumpMap,re=!!y.normalMap,ue=!!y.displacementMap,ie=!!y.emissiveMap,fe=!!y.metalnessMap,te=!!y.roughnessMap,xe=y.anisotropy>0,R=y.clearcoat>0,T=y.dispersion>0,V=y.iridescence>0,se=y.sheen>0,ce=y.transmission>0,J=xe&&!!y.anisotropyMap,Pe=R&&!!y.clearcoatMap,Ae=R&&!!y.clearcoatNormalMap,Ue=R&&!!y.clearcoatRoughnessMap,N=V&&!!y.iridescenceMap,ee=V&&!!y.iridescenceThicknessMap,me=se&&!!y.sheenColorMap,Te=se&&!!y.sheenRoughnessMap,Ie=!!y.specularMap,we=!!y.specularColorMap,He=!!y.specularIntensityMap,G=ce&&!!y.transmissionMap,Re=ce&&!!y.thicknessMap,Se=!!y.gradientMap,ve=!!y.alphaMap,_e=y.alphaTest>0,de=!!y.alphaHash,Be=!!y.extensions;let We=Ds;y.toneMapped&&(oe===null||oe.isXRRenderTarget===!0)&&(We=i.toneMapping);const vt={shaderID:$,shaderType:y.type,shaderName:y.name,vertexShader:ge,fragmentShader:be,defines:y.defines,customVertexShaderID:Me,customFragmentShaderID:Y,isRawShaderMaterial:y.isRawShaderMaterial===!0,glslVersion:y.glslVersion,precision:h,batching:ye,batchingColor:ye&&U._colorsTexture!==null,instancing:Fe,instancingColor:Fe&&U.instanceColor!==null,instancingMorph:Fe&&U.morphTexture!==null,supportsVertexTextures:d,outputColorSpace:oe===null?i.outputColorSpace:oe.isXRRenderTarget===!0?oe.texture.colorSpace:uo,alphaToCoverage:!!y.alphaToCoverage,map:ke,matcap:F,envMap:B,envMapMode:B&&W.mapping,envMapCubeUVHeight:H,aoMap:q,lightMap:w,bumpMap:ne,normalMap:re,displacementMap:d&&ue,emissiveMap:ie,normalMapObjectSpace:re&&y.normalMapType===SS,normalMapTangentSpace:re&&y.normalMapType===AS,metalnessMap:fe,roughnessMap:te,anisotropy:xe,anisotropyMap:J,clearcoat:R,clearcoatMap:Pe,clearcoatNormalMap:Ae,clearcoatRoughnessMap:Ue,dispersion:T,iridescence:V,iridescenceMap:N,iridescenceThicknessMap:ee,sheen:se,sheenColorMap:me,sheenRoughnessMap:Te,specularMap:Ie,specularColorMap:we,specularIntensityMap:He,transmission:ce,transmissionMap:G,thicknessMap:Re,gradientMap:Se,opaque:y.transparent===!1&&y.blending===Is&&y.alphaToCoverage===!1,alphaMap:ve,alphaTest:_e,alphaHash:de,combine:y.combine,mapUv:ke&&p(y.map.channel),aoMapUv:q&&p(y.aoMap.channel),lightMapUv:w&&p(y.lightMap.channel),bumpMapUv:ne&&p(y.bumpMap.channel),normalMapUv:re&&p(y.normalMap.channel),displacementMapUv:ue&&p(y.displacementMap.channel),emissiveMapUv:ie&&p(y.emissiveMap.channel),metalnessMapUv:fe&&p(y.metalnessMap.channel),roughnessMapUv:te&&p(y.roughnessMap.channel),anisotropyMapUv:J&&p(y.anisotropyMap.channel),clearcoatMapUv:Pe&&p(y.clearcoatMap.channel),clearcoatNormalMapUv:Ae&&p(y.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Ue&&p(y.clearcoatRoughnessMap.channel),iridescenceMapUv:N&&p(y.iridescenceMap.channel),iridescenceThicknessMapUv:ee&&p(y.iridescenceThicknessMap.channel),sheenColorMapUv:me&&p(y.sheenColorMap.channel),sheenRoughnessMapUv:Te&&p(y.sheenRoughnessMap.channel),specularMapUv:Ie&&p(y.specularMap.channel),specularColorMapUv:we&&p(y.specularColorMap.channel),specularIntensityMapUv:He&&p(y.specularIntensityMap.channel),transmissionMapUv:G&&p(y.transmissionMap.channel),thicknessMapUv:Re&&p(y.thicknessMap.channel),alphaMapUv:ve&&p(y.alphaMap.channel),vertexTangents:!!k.attributes.tangent&&(re||xe),vertexColors:y.vertexColors,vertexAlphas:y.vertexColors===!0&&!!k.attributes.color&&k.attributes.color.itemSize===4,pointsUvs:U.isPoints===!0&&!!k.attributes.uv&&(ke||ve),fog:!!O,useFog:y.fog===!0,fogExp2:!!O&&O.isFogExp2,flatShading:y.flatShading===!0&&y.wireframe===!1,sizeAttenuation:y.sizeAttenuation===!0,logarithmicDepthBuffer:f,reversedDepthBuffer:he,skinning:U.isSkinnedMesh===!0,morphTargets:k.morphAttributes.position!==void 0,morphNormals:k.morphAttributes.normal!==void 0,morphColors:k.morphAttributes.color!==void 0,morphTargetsCount:Z,morphTextureStride:pe,numDirLights:C.directional.length,numPointLights:C.point.length,numSpotLights:C.spot.length,numSpotLightMaps:C.spotLightMap.length,numRectAreaLights:C.rectArea.length,numHemiLights:C.hemi.length,numDirLightShadows:C.directionalShadowMap.length,numPointLightShadows:C.pointShadowMap.length,numSpotLightShadows:C.spotShadowMap.length,numSpotLightShadowsWithMaps:C.numSpotLightShadowsWithMaps,numLightProbes:C.numLightProbes,numClippingPlanes:o.numPlanes,numClipIntersection:o.numIntersection,dithering:y.dithering,shadowMapEnabled:i.shadowMap.enabled&&I.length>0,shadowMapType:i.shadowMap.type,toneMapping:We,decodeVideoTexture:ke&&y.map.isVideoTexture===!0&&rt.getTransfer(y.map.colorSpace)===ht,decodeVideoTextureEmissive:ie&&y.emissiveMap.isVideoTexture===!0&&rt.getTransfer(y.emissiveMap.colorSpace)===ht,premultipliedAlpha:y.premultipliedAlpha,doubleSided:y.side===ti,flipSided:y.side===En,useDepthPacking:y.depthPacking>=0,depthPacking:y.depthPacking||0,index0AttributeName:y.index0AttributeName,extensionClipCullDistance:Be&&y.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Be&&y.extensions.multiDraw===!0||ye)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:y.customProgramCacheKey()};return vt.vertexUv1s=c.has(1),vt.vertexUv2s=c.has(2),vt.vertexUv3s=c.has(3),c.clear(),vt}function m(y){const C=[];if(y.shaderID?C.push(y.shaderID):(C.push(y.customVertexShaderID),C.push(y.customFragmentShaderID)),y.defines!==void 0)for(const I in y.defines)C.push(I),C.push(y.defines[I]);return y.isRawShaderMaterial===!1&&(_(C,y),A(C,y),C.push(i.outputColorSpace)),C.push(y.customProgramCacheKey),C.join()}function _(y,C){y.push(C.precision),y.push(C.outputColorSpace),y.push(C.envMapMode),y.push(C.envMapCubeUVHeight),y.push(C.mapUv),y.push(C.alphaMapUv),y.push(C.lightMapUv),y.push(C.aoMapUv),y.push(C.bumpMapUv),y.push(C.normalMapUv),y.push(C.displacementMapUv),y.push(C.emissiveMapUv),y.push(C.metalnessMapUv),y.push(C.roughnessMapUv),y.push(C.anisotropyMapUv),y.push(C.clearcoatMapUv),y.push(C.clearcoatNormalMapUv),y.push(C.clearcoatRoughnessMapUv),y.push(C.iridescenceMapUv),y.push(C.iridescenceThicknessMapUv),y.push(C.sheenColorMapUv),y.push(C.sheenRoughnessMapUv),y.push(C.specularMapUv),y.push(C.specularColorMapUv),y.push(C.specularIntensityMapUv),y.push(C.transmissionMapUv),y.push(C.thicknessMapUv),y.push(C.combine),y.push(C.fogExp2),y.push(C.sizeAttenuation),y.push(C.morphTargetsCount),y.push(C.morphAttributeCount),y.push(C.numDirLights),y.push(C.numPointLights),y.push(C.numSpotLights),y.push(C.numSpotLightMaps),y.push(C.numHemiLights),y.push(C.numRectAreaLights),y.push(C.numDirLightShadows),y.push(C.numPointLightShadows),y.push(C.numSpotLightShadows),y.push(C.numSpotLightShadowsWithMaps),y.push(C.numLightProbes),y.push(C.shadowMapType),y.push(C.toneMapping),y.push(C.numClippingPlanes),y.push(C.numClipIntersection),y.push(C.depthPacking)}function A(y,C){a.disableAll(),C.supportsVertexTextures&&a.enable(0),C.instancing&&a.enable(1),C.instancingColor&&a.enable(2),C.instancingMorph&&a.enable(3),C.matcap&&a.enable(4),C.envMap&&a.enable(5),C.normalMapObjectSpace&&a.enable(6),C.normalMapTangentSpace&&a.enable(7),C.clearcoat&&a.enable(8),C.iridescence&&a.enable(9),C.alphaTest&&a.enable(10),C.vertexColors&&a.enable(11),C.vertexAlphas&&a.enable(12),C.vertexUv1s&&a.enable(13),C.vertexUv2s&&a.enable(14),C.vertexUv3s&&a.enable(15),C.vertexTangents&&a.enable(16),C.anisotropy&&a.enable(17),C.alphaHash&&a.enable(18),C.batching&&a.enable(19),C.dispersion&&a.enable(20),C.batchingColor&&a.enable(21),C.gradientMap&&a.enable(22),y.push(a.mask),a.disableAll(),C.fog&&a.enable(0),C.useFog&&a.enable(1),C.flatShading&&a.enable(2),C.logarithmicDepthBuffer&&a.enable(3),C.reversedDepthBuffer&&a.enable(4),C.skinning&&a.enable(5),C.morphTargets&&a.enable(6),C.morphNormals&&a.enable(7),C.morphColors&&a.enable(8),C.premultipliedAlpha&&a.enable(9),C.shadowMapEnabled&&a.enable(10),C.doubleSided&&a.enable(11),C.flipSided&&a.enable(12),C.useDepthPacking&&a.enable(13),C.dithering&&a.enable(14),C.transmission&&a.enable(15),C.sheen&&a.enable(16),C.opaque&&a.enable(17),C.pointsUvs&&a.enable(18),C.decodeVideoTexture&&a.enable(19),C.decodeVideoTextureEmissive&&a.enable(20),C.alphaToCoverage&&a.enable(21),y.push(a.mask)}function S(y){const C=g[y.type];let I;if(C){const P=Ci[C];I=JS.clone(P.uniforms)}else I=y.uniforms;return I}function v(y,C){let I;for(let P=0,U=u.length;P<U;P++){const O=u[P];if(O.cacheKey===C){I=O,++I.usedTimes;break}}return I===void 0&&(I=new TC(i,C,y,r),u.push(I)),I}function b(y){if(--y.usedTimes===0){const C=u.indexOf(y);u[C]=u[u.length-1],u.pop(),y.destroy()}}function M(y){l.remove(y)}function E(){l.dispose()}return{getParameters:x,getProgramCacheKey:m,getUniforms:S,acquireProgram:v,releaseProgram:b,releaseShaderCache:M,programs:u,dispose:E}}function DC(){let i=new WeakMap;function e(o){return i.has(o)}function t(o){let a=i.get(o);return a===void 0&&(a={},i.set(o,a)),a}function n(o){i.delete(o)}function s(o,a,l){i.get(o)[a]=l}function r(){i=new WeakMap}return{has:e,get:t,remove:n,update:s,dispose:r}}function PC(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.material.id!==e.material.id?i.material.id-e.material.id:i.z!==e.z?i.z-e.z:i.id-e.id}function Sp(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.z!==e.z?e.z-i.z:i.id-e.id}function vp(){const i=[];let e=0;const t=[],n=[],s=[];function r(){e=0,t.length=0,n.length=0,s.length=0}function o(f,d,h,g,p,x){let m=i[e];return m===void 0?(m={id:f.id,object:f,geometry:d,material:h,groupOrder:g,renderOrder:f.renderOrder,z:p,group:x},i[e]=m):(m.id=f.id,m.object=f,m.geometry=d,m.material=h,m.groupOrder=g,m.renderOrder=f.renderOrder,m.z=p,m.group=x),e++,m}function a(f,d,h,g,p,x){const m=o(f,d,h,g,p,x);h.transmission>0?n.push(m):h.transparent===!0?s.push(m):t.push(m)}function l(f,d,h,g,p,x){const m=o(f,d,h,g,p,x);h.transmission>0?n.unshift(m):h.transparent===!0?s.unshift(m):t.unshift(m)}function c(f,d){t.length>1&&t.sort(f||PC),n.length>1&&n.sort(d||Sp),s.length>1&&s.sort(d||Sp)}function u(){for(let f=e,d=i.length;f<d;f++){const h=i[f];if(h.id===null)break;h.id=null,h.object=null,h.geometry=null,h.material=null,h.group=null}}return{opaque:t,transmissive:n,transparent:s,init:r,push:a,unshift:l,finish:u,sort:c}}function FC(){let i=new WeakMap;function e(n,s){const r=i.get(n);let o;return r===void 0?(o=new vp,i.set(n,[o])):s>=r.length?(o=new vp,r.push(o)):o=r[s],o}function t(){i=new WeakMap}return{get:e,dispose:t}}function LC(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new L,color:new nt};break;case"SpotLight":t={position:new L,direction:new L,color:new nt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new L,color:new nt,distance:0,decay:0};break;case"HemisphereLight":t={direction:new L,skyColor:new nt,groundColor:new nt};break;case"RectAreaLight":t={color:new nt,position:new L,halfWidth:new L,halfHeight:new L};break}return i[e.id]=t,t}}}function BC(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ne};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ne};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ne,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[e.id]=t,t}}}let UC=0;function OC(i,e){return(e.castShadow?2:0)-(i.castShadow?2:0)+(e.map?1:0)-(i.map?1:0)}function NC(i){const e=new LC,t=BC(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)n.probe.push(new L);const s=new L,r=new qe,o=new qe;function a(c){let u=0,f=0,d=0;for(let y=0;y<9;y++)n.probe[y].set(0,0,0);let h=0,g=0,p=0,x=0,m=0,_=0,A=0,S=0,v=0,b=0,M=0;c.sort(OC);for(let y=0,C=c.length;y<C;y++){const I=c[y],P=I.color,U=I.intensity,O=I.distance,k=I.shadow&&I.shadow.map?I.shadow.map.texture:null;if(I.isAmbientLight)u+=P.r*U,f+=P.g*U,d+=P.b*U;else if(I.isLightProbe){for(let z=0;z<9;z++)n.probe[z].addScaledVector(I.sh.coefficients[z],U);M++}else if(I.isDirectionalLight){const z=e.get(I);if(z.color.copy(I.color).multiplyScalar(I.intensity),I.castShadow){const W=I.shadow,H=t.get(I);H.shadowIntensity=W.intensity,H.shadowBias=W.bias,H.shadowNormalBias=W.normalBias,H.shadowRadius=W.radius,H.shadowMapSize=W.mapSize,n.directionalShadow[h]=H,n.directionalShadowMap[h]=k,n.directionalShadowMatrix[h]=I.shadow.matrix,_++}n.directional[h]=z,h++}else if(I.isSpotLight){const z=e.get(I);z.position.setFromMatrixPosition(I.matrixWorld),z.color.copy(P).multiplyScalar(U),z.distance=O,z.coneCos=Math.cos(I.angle),z.penumbraCos=Math.cos(I.angle*(1-I.penumbra)),z.decay=I.decay,n.spot[p]=z;const W=I.shadow;if(I.map&&(n.spotLightMap[v]=I.map,v++,W.updateMatrices(I),I.castShadow&&b++),n.spotLightMatrix[p]=W.matrix,I.castShadow){const H=t.get(I);H.shadowIntensity=W.intensity,H.shadowBias=W.bias,H.shadowNormalBias=W.normalBias,H.shadowRadius=W.radius,H.shadowMapSize=W.mapSize,n.spotShadow[p]=H,n.spotShadowMap[p]=k,S++}p++}else if(I.isRectAreaLight){const z=e.get(I);z.color.copy(P).multiplyScalar(U),z.halfWidth.set(I.width*.5,0,0),z.halfHeight.set(0,I.height*.5,0),n.rectArea[x]=z,x++}else if(I.isPointLight){const z=e.get(I);if(z.color.copy(I.color).multiplyScalar(I.intensity),z.distance=I.distance,z.decay=I.decay,I.castShadow){const W=I.shadow,H=t.get(I);H.shadowIntensity=W.intensity,H.shadowBias=W.bias,H.shadowNormalBias=W.normalBias,H.shadowRadius=W.radius,H.shadowMapSize=W.mapSize,H.shadowCameraNear=W.camera.near,H.shadowCameraFar=W.camera.far,n.pointShadow[g]=H,n.pointShadowMap[g]=k,n.pointShadowMatrix[g]=I.shadow.matrix,A++}n.point[g]=z,g++}else if(I.isHemisphereLight){const z=e.get(I);z.skyColor.copy(I.color).multiplyScalar(U),z.groundColor.copy(I.groundColor).multiplyScalar(U),n.hemi[m]=z,m++}}x>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=De.LTC_FLOAT_1,n.rectAreaLTC2=De.LTC_FLOAT_2):(n.rectAreaLTC1=De.LTC_HALF_1,n.rectAreaLTC2=De.LTC_HALF_2)),n.ambient[0]=u,n.ambient[1]=f,n.ambient[2]=d;const E=n.hash;(E.directionalLength!==h||E.pointLength!==g||E.spotLength!==p||E.rectAreaLength!==x||E.hemiLength!==m||E.numDirectionalShadows!==_||E.numPointShadows!==A||E.numSpotShadows!==S||E.numSpotMaps!==v||E.numLightProbes!==M)&&(n.directional.length=h,n.spot.length=p,n.rectArea.length=x,n.point.length=g,n.hemi.length=m,n.directionalShadow.length=_,n.directionalShadowMap.length=_,n.pointShadow.length=A,n.pointShadowMap.length=A,n.spotShadow.length=S,n.spotShadowMap.length=S,n.directionalShadowMatrix.length=_,n.pointShadowMatrix.length=A,n.spotLightMatrix.length=S+v-b,n.spotLightMap.length=v,n.numSpotLightShadowsWithMaps=b,n.numLightProbes=M,E.directionalLength=h,E.pointLength=g,E.spotLength=p,E.rectAreaLength=x,E.hemiLength=m,E.numDirectionalShadows=_,E.numPointShadows=A,E.numSpotShadows=S,E.numSpotMaps=v,E.numLightProbes=M,n.version=UC++)}function l(c,u){let f=0,d=0,h=0,g=0,p=0;const x=u.matrixWorldInverse;for(let m=0,_=c.length;m<_;m++){const A=c[m];if(A.isDirectionalLight){const S=n.directional[f];S.direction.setFromMatrixPosition(A.matrixWorld),s.setFromMatrixPosition(A.target.matrixWorld),S.direction.sub(s),S.direction.transformDirection(x),f++}else if(A.isSpotLight){const S=n.spot[h];S.position.setFromMatrixPosition(A.matrixWorld),S.position.applyMatrix4(x),S.direction.setFromMatrixPosition(A.matrixWorld),s.setFromMatrixPosition(A.target.matrixWorld),S.direction.sub(s),S.direction.transformDirection(x),h++}else if(A.isRectAreaLight){const S=n.rectArea[g];S.position.setFromMatrixPosition(A.matrixWorld),S.position.applyMatrix4(x),o.identity(),r.copy(A.matrixWorld),r.premultiply(x),o.extractRotation(r),S.halfWidth.set(A.width*.5,0,0),S.halfHeight.set(0,A.height*.5,0),S.halfWidth.applyMatrix4(o),S.halfHeight.applyMatrix4(o),g++}else if(A.isPointLight){const S=n.point[d];S.position.setFromMatrixPosition(A.matrixWorld),S.position.applyMatrix4(x),d++}else if(A.isHemisphereLight){const S=n.hemi[p];S.direction.setFromMatrixPosition(A.matrixWorld),S.direction.transformDirection(x),p++}}}return{setup:a,setupView:l,state:n}}function yp(i){const e=new NC(i),t=[],n=[];function s(u){c.camera=u,t.length=0,n.length=0}function r(u){t.push(u)}function o(u){n.push(u)}function a(){e.setup(t)}function l(u){e.setupView(t,u)}const c={lightsArray:t,shadowsArray:n,camera:null,lights:e,transmissionRenderTarget:{}};return{init:s,state:c,setupLights:a,setupLightsView:l,pushLight:r,pushShadow:o}}function zC(i){let e=new WeakMap;function t(s,r=0){const o=e.get(s);let a;return o===void 0?(a=new yp(i),e.set(s,[a])):r>=o.length?(a=new yp(i),o.push(a)):a=o[r],a}function n(){e=new WeakMap}return{get:t,dispose:n}}const kC=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,HC=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;function VC(i,e,t){let n=new X0;const s=new Ne,r=new Ne,o=new Et,a=new dv({depthPacking:_S}),l=new hv,c={},u=t.maxTextureSize,f={[Bi]:En,[En]:Bi,[ti]:ti},d=new _n({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Ne},radius:{value:4}},vertexShader:kC,fragmentShader:HC}),h=d.clone();h.defines.HORIZONTAL_PASS=1;const g=new An;g.setAttribute("position",new li(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const p=new Vt(g,d),x=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=b0;let m=this.type;this.render=function(b,M,E){if(x.enabled===!1||x.autoUpdate===!1&&x.needsUpdate===!1||b.length===0)return;const y=i.getRenderTarget(),C=i.getActiveCubeFace(),I=i.getActiveMipmapLevel(),P=i.state;P.setBlending(is),P.buffers.depth.getReversed()===!0?P.buffers.color.setClear(0,0,0,0):P.buffers.color.setClear(1,1,1,1),P.buffers.depth.setTest(!0),P.setScissorTest(!1);const U=m!==qi&&this.type===qi,O=m===qi&&this.type!==qi;for(let k=0,z=b.length;k<z;k++){const W=b[k],H=W.shadow;if(H===void 0){je("WebGLShadowMap:",W,"has no shadow.");continue}if(H.autoUpdate===!1&&H.needsUpdate===!1)continue;s.copy(H.mapSize);const $=H.getFrameExtents();if(s.multiply($),r.copy(H.mapSize),(s.x>u||s.y>u)&&(s.x>u&&(r.x=Math.floor(u/$.x),s.x=r.x*$.x,H.mapSize.x=r.x),s.y>u&&(r.y=Math.floor(u/$.y),s.y=r.y*$.y,H.mapSize.y=r.y)),H.map===null||U===!0||O===!0){const Z=this.type!==qi?{minFilter:qn,magFilter:qn}:{};H.map!==null&&H.map.dispose(),H.map=new Us(s.x,s.y,Z),H.map.texture.name=W.name+".shadowMap",H.camera.updateProjectionMatrix()}i.setRenderTarget(H.map),i.clear();const X=H.getViewportCount();for(let Z=0;Z<X;Z++){const pe=H.getViewport(Z);o.set(r.x*pe.x,r.y*pe.y,r.x*pe.z,r.y*pe.w),P.viewport(o),H.updateMatrices(W,Z),n=H.getFrustum(),S(M,E,H.camera,W,this.type)}H.isPointLightShadow!==!0&&this.type===qi&&_(H,E),H.needsUpdate=!1}m=this.type,x.needsUpdate=!1,i.setRenderTarget(y,C,I)};function _(b,M){const E=e.update(p);d.defines.VSM_SAMPLES!==b.blurSamples&&(d.defines.VSM_SAMPLES=b.blurSamples,h.defines.VSM_SAMPLES=b.blurSamples,d.needsUpdate=!0,h.needsUpdate=!0),b.mapPass===null&&(b.mapPass=new Us(s.x,s.y)),d.uniforms.shadow_pass.value=b.map.texture,d.uniforms.resolution.value=b.mapSize,d.uniforms.radius.value=b.radius,i.setRenderTarget(b.mapPass),i.clear(),i.renderBufferDirect(M,null,E,d,p,null),h.uniforms.shadow_pass.value=b.mapPass.texture,h.uniforms.resolution.value=b.mapSize,h.uniforms.radius.value=b.radius,i.setRenderTarget(b.map),i.clear(),i.renderBufferDirect(M,null,E,h,p,null)}function A(b,M,E,y){let C=null;const I=E.isPointLight===!0?b.customDistanceMaterial:b.customDepthMaterial;if(I!==void 0)C=I;else if(C=E.isPointLight===!0?l:a,i.localClippingEnabled&&M.clipShadows===!0&&Array.isArray(M.clippingPlanes)&&M.clippingPlanes.length!==0||M.displacementMap&&M.displacementScale!==0||M.alphaMap&&M.alphaTest>0||M.map&&M.alphaTest>0||M.alphaToCoverage===!0){const P=C.uuid,U=M.uuid;let O=c[P];O===void 0&&(O={},c[P]=O);let k=O[U];k===void 0&&(k=C.clone(),O[U]=k,M.addEventListener("dispose",v)),C=k}if(C.visible=M.visible,C.wireframe=M.wireframe,y===qi?C.side=M.shadowSide!==null?M.shadowSide:M.side:C.side=M.shadowSide!==null?M.shadowSide:f[M.side],C.alphaMap=M.alphaMap,C.alphaTest=M.alphaToCoverage===!0?.5:M.alphaTest,C.map=M.map,C.clipShadows=M.clipShadows,C.clippingPlanes=M.clippingPlanes,C.clipIntersection=M.clipIntersection,C.displacementMap=M.displacementMap,C.displacementScale=M.displacementScale,C.displacementBias=M.displacementBias,C.wireframeLinewidth=M.wireframeLinewidth,C.linewidth=M.linewidth,E.isPointLight===!0&&C.isMeshDistanceMaterial===!0){const P=i.properties.get(C);P.light=E}return C}function S(b,M,E,y,C){if(b.visible===!1)return;if(b.layers.test(M.layers)&&(b.isMesh||b.isLine||b.isPoints)&&(b.castShadow||b.receiveShadow&&C===qi)&&(!b.frustumCulled||n.intersectsObject(b))){b.modelViewMatrix.multiplyMatrices(E.matrixWorldInverse,b.matrixWorld);const U=e.update(b),O=b.material;if(Array.isArray(O)){const k=U.groups;for(let z=0,W=k.length;z<W;z++){const H=k[z],$=O[H.materialIndex];if($&&$.visible){const X=A(b,$,y,C);b.onBeforeShadow(i,b,M,E,U,X,H),i.renderBufferDirect(E,null,U,X,b,H),b.onAfterShadow(i,b,M,E,U,X,H)}}}else if(O.visible){const k=A(b,O,y,C);b.onBeforeShadow(i,b,M,E,U,k,null),i.renderBufferDirect(E,null,U,k,b,null),b.onAfterShadow(i,b,M,E,U,k,null)}}const P=b.children;for(let U=0,O=P.length;U<O;U++)S(P[U],M,E,y,C)}function v(b){b.target.removeEventListener("dispose",v);for(const E in c){const y=c[E],C=b.target.uuid;C in y&&(y[C].dispose(),delete y[C])}}}const GC={[yu]:bu,[Mu]:Eu,[Cu]:wu,[oo]:Tu,[bu]:yu,[Eu]:Mu,[wu]:Cu,[Tu]:oo};function WC(i,e){function t(){let G=!1;const Re=new Et;let Se=null;const ve=new Et(0,0,0,0);return{setMask:function(_e){Se!==_e&&!G&&(i.colorMask(_e,_e,_e,_e),Se=_e)},setLocked:function(_e){G=_e},setClear:function(_e,de,Be,We,vt){vt===!0&&(_e*=We,de*=We,Be*=We),Re.set(_e,de,Be,We),ve.equals(Re)===!1&&(i.clearColor(_e,de,Be,We),ve.copy(Re))},reset:function(){G=!1,Se=null,ve.set(-1,0,0,0)}}}function n(){let G=!1,Re=!1,Se=null,ve=null,_e=null;return{setReversed:function(de){if(Re!==de){const Be=e.get("EXT_clip_control");de?Be.clipControlEXT(Be.LOWER_LEFT_EXT,Be.ZERO_TO_ONE_EXT):Be.clipControlEXT(Be.LOWER_LEFT_EXT,Be.NEGATIVE_ONE_TO_ONE_EXT),Re=de;const We=_e;_e=null,this.setClear(We)}},getReversed:function(){return Re},setTest:function(de){de?oe(i.DEPTH_TEST):he(i.DEPTH_TEST)},setMask:function(de){Se!==de&&!G&&(i.depthMask(de),Se=de)},setFunc:function(de){if(Re&&(de=GC[de]),ve!==de){switch(de){case yu:i.depthFunc(i.NEVER);break;case bu:i.depthFunc(i.ALWAYS);break;case Mu:i.depthFunc(i.LESS);break;case oo:i.depthFunc(i.LEQUAL);break;case Cu:i.depthFunc(i.EQUAL);break;case Tu:i.depthFunc(i.GEQUAL);break;case Eu:i.depthFunc(i.GREATER);break;case wu:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}ve=de}},setLocked:function(de){G=de},setClear:function(de){_e!==de&&(Re&&(de=1-de),i.clearDepth(de),_e=de)},reset:function(){G=!1,Se=null,ve=null,_e=null,Re=!1}}}function s(){let G=!1,Re=null,Se=null,ve=null,_e=null,de=null,Be=null,We=null,vt=null;return{setTest:function(ut){G||(ut?oe(i.STENCIL_TEST):he(i.STENCIL_TEST))},setMask:function(ut){Re!==ut&&!G&&(i.stencilMask(ut),Re=ut)},setFunc:function(ut,Ai,ci){(Se!==ut||ve!==Ai||_e!==ci)&&(i.stencilFunc(ut,Ai,ci),Se=ut,ve=Ai,_e=ci)},setOp:function(ut,Ai,ci){(de!==ut||Be!==Ai||We!==ci)&&(i.stencilOp(ut,Ai,ci),de=ut,Be=Ai,We=ci)},setLocked:function(ut){G=ut},setClear:function(ut){vt!==ut&&(i.clearStencil(ut),vt=ut)},reset:function(){G=!1,Re=null,Se=null,ve=null,_e=null,de=null,Be=null,We=null,vt=null}}}const r=new t,o=new n,a=new s,l=new WeakMap,c=new WeakMap;let u={},f={},d=new WeakMap,h=[],g=null,p=!1,x=null,m=null,_=null,A=null,S=null,v=null,b=null,M=new nt(0,0,0),E=0,y=!1,C=null,I=null,P=null,U=null,O=null;const k=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let z=!1,W=0;const H=i.getParameter(i.VERSION);H.indexOf("WebGL")!==-1?(W=parseFloat(/^WebGL (\d)/.exec(H)[1]),z=W>=1):H.indexOf("OpenGL ES")!==-1&&(W=parseFloat(/^OpenGL ES (\d)/.exec(H)[1]),z=W>=2);let $=null,X={};const Z=i.getParameter(i.SCISSOR_BOX),pe=i.getParameter(i.VIEWPORT),ge=new Et().fromArray(Z),be=new Et().fromArray(pe);function Me(G,Re,Se,ve){const _e=new Uint8Array(4),de=i.createTexture();i.bindTexture(G,de),i.texParameteri(G,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(G,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let Be=0;Be<Se;Be++)G===i.TEXTURE_3D||G===i.TEXTURE_2D_ARRAY?i.texImage3D(Re,0,i.RGBA,1,1,ve,0,i.RGBA,i.UNSIGNED_BYTE,_e):i.texImage2D(Re+Be,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,_e);return de}const Y={};Y[i.TEXTURE_2D]=Me(i.TEXTURE_2D,i.TEXTURE_2D,1),Y[i.TEXTURE_CUBE_MAP]=Me(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),Y[i.TEXTURE_2D_ARRAY]=Me(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),Y[i.TEXTURE_3D]=Me(i.TEXTURE_3D,i.TEXTURE_3D,1,1),r.setClear(0,0,0,1),o.setClear(1),a.setClear(0),oe(i.DEPTH_TEST),o.setFunc(oo),ne(!1),re(Ch),oe(i.CULL_FACE),q(is);function oe(G){u[G]!==!0&&(i.enable(G),u[G]=!0)}function he(G){u[G]!==!1&&(i.disable(G),u[G]=!1)}function Fe(G,Re){return f[G]!==Re?(i.bindFramebuffer(G,Re),f[G]=Re,G===i.DRAW_FRAMEBUFFER&&(f[i.FRAMEBUFFER]=Re),G===i.FRAMEBUFFER&&(f[i.DRAW_FRAMEBUFFER]=Re),!0):!1}function ye(G,Re){let Se=h,ve=!1;if(G){Se=d.get(Re),Se===void 0&&(Se=[],d.set(Re,Se));const _e=G.textures;if(Se.length!==_e.length||Se[0]!==i.COLOR_ATTACHMENT0){for(let de=0,Be=_e.length;de<Be;de++)Se[de]=i.COLOR_ATTACHMENT0+de;Se.length=_e.length,ve=!0}}else Se[0]!==i.BACK&&(Se[0]=i.BACK,ve=!0);ve&&i.drawBuffers(Se)}function ke(G){return g!==G?(i.useProgram(G),g=G,!0):!1}const F={[nr]:i.FUNC_ADD,[WA]:i.FUNC_SUBTRACT,[XA]:i.FUNC_REVERSE_SUBTRACT};F[qA]=i.MIN,F[QA]=i.MAX;const B={[YA]:i.ZERO,[KA]:i.ONE,[jA]:i.SRC_COLOR,[ia]:i.SRC_ALPHA,[nS]:i.SRC_ALPHA_SATURATE,[eS]:i.DST_COLOR,[ZA]:i.DST_ALPHA,[$A]:i.ONE_MINUS_SRC_COLOR,[sa]:i.ONE_MINUS_SRC_ALPHA,[tS]:i.ONE_MINUS_DST_COLOR,[JA]:i.ONE_MINUS_DST_ALPHA,[iS]:i.CONSTANT_COLOR,[sS]:i.ONE_MINUS_CONSTANT_COLOR,[rS]:i.CONSTANT_ALPHA,[oS]:i.ONE_MINUS_CONSTANT_ALPHA};function q(G,Re,Se,ve,_e,de,Be,We,vt,ut){if(G===is){p===!0&&(he(i.BLEND),p=!1);return}if(p===!1&&(oe(i.BLEND),p=!0),G!==M0){if(G!==x||ut!==y){if((m!==nr||S!==nr)&&(i.blendEquation(i.FUNC_ADD),m=nr,S=nr),ut)switch(G){case Is:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Th:i.blendFunc(i.ONE,i.ONE);break;case Eh:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case wh:i.blendFuncSeparate(i.DST_COLOR,i.ONE_MINUS_SRC_ALPHA,i.ZERO,i.ONE);break;default:zt("WebGLState: Invalid blending: ",G);break}else switch(G){case Is:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Th:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE,i.ONE,i.ONE);break;case Eh:zt("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case wh:zt("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:zt("WebGLState: Invalid blending: ",G);break}_=null,A=null,v=null,b=null,M.set(0,0,0),E=0,x=G,y=ut}return}_e=_e||Re,de=de||Se,Be=Be||ve,(Re!==m||_e!==S)&&(i.blendEquationSeparate(F[Re],F[_e]),m=Re,S=_e),(Se!==_||ve!==A||de!==v||Be!==b)&&(i.blendFuncSeparate(B[Se],B[ve],B[de],B[Be]),_=Se,A=ve,v=de,b=Be),(We.equals(M)===!1||vt!==E)&&(i.blendColor(We.r,We.g,We.b,vt),M.copy(We),E=vt),x=G,y=!1}function w(G,Re){G.side===ti?he(i.CULL_FACE):oe(i.CULL_FACE);let Se=G.side===En;Re&&(Se=!Se),ne(Se),G.blending===Is&&G.transparent===!1?q(is):q(G.blending,G.blendEquation,G.blendSrc,G.blendDst,G.blendEquationAlpha,G.blendSrcAlpha,G.blendDstAlpha,G.blendColor,G.blendAlpha,G.premultipliedAlpha),o.setFunc(G.depthFunc),o.setTest(G.depthTest),o.setMask(G.depthWrite),r.setMask(G.colorWrite);const ve=G.stencilWrite;a.setTest(ve),ve&&(a.setMask(G.stencilWriteMask),a.setFunc(G.stencilFunc,G.stencilRef,G.stencilFuncMask),a.setOp(G.stencilFail,G.stencilZFail,G.stencilZPass)),ie(G.polygonOffset,G.polygonOffsetFactor,G.polygonOffsetUnits),G.alphaToCoverage===!0?oe(i.SAMPLE_ALPHA_TO_COVERAGE):he(i.SAMPLE_ALPHA_TO_COVERAGE)}function ne(G){C!==G&&(G?i.frontFace(i.CW):i.frontFace(i.CCW),C=G)}function re(G){G!==HA?(oe(i.CULL_FACE),G!==I&&(G===Ch?i.cullFace(i.BACK):G===VA?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):he(i.CULL_FACE),I=G}function ue(G){G!==P&&(z&&i.lineWidth(G),P=G)}function ie(G,Re,Se){G?(oe(i.POLYGON_OFFSET_FILL),(U!==Re||O!==Se)&&(i.polygonOffset(Re,Se),U=Re,O=Se)):he(i.POLYGON_OFFSET_FILL)}function fe(G){G?oe(i.SCISSOR_TEST):he(i.SCISSOR_TEST)}function te(G){G===void 0&&(G=i.TEXTURE0+k-1),$!==G&&(i.activeTexture(G),$=G)}function xe(G,Re,Se){Se===void 0&&($===null?Se=i.TEXTURE0+k-1:Se=$);let ve=X[Se];ve===void 0&&(ve={type:void 0,texture:void 0},X[Se]=ve),(ve.type!==G||ve.texture!==Re)&&($!==Se&&(i.activeTexture(Se),$=Se),i.bindTexture(G,Re||Y[G]),ve.type=G,ve.texture=Re)}function R(){const G=X[$];G!==void 0&&G.type!==void 0&&(i.bindTexture(G.type,null),G.type=void 0,G.texture=void 0)}function T(){try{i.compressedTexImage2D(...arguments)}catch(G){G("WebGLState:",G)}}function V(){try{i.compressedTexImage3D(...arguments)}catch(G){G("WebGLState:",G)}}function se(){try{i.texSubImage2D(...arguments)}catch(G){G("WebGLState:",G)}}function ce(){try{i.texSubImage3D(...arguments)}catch(G){G("WebGLState:",G)}}function J(){try{i.compressedTexSubImage2D(...arguments)}catch(G){G("WebGLState:",G)}}function Pe(){try{i.compressedTexSubImage3D(...arguments)}catch(G){G("WebGLState:",G)}}function Ae(){try{i.texStorage2D(...arguments)}catch(G){G("WebGLState:",G)}}function Ue(){try{i.texStorage3D(...arguments)}catch(G){G("WebGLState:",G)}}function N(){try{i.texImage2D(...arguments)}catch(G){G("WebGLState:",G)}}function ee(){try{i.texImage3D(...arguments)}catch(G){G("WebGLState:",G)}}function me(G){ge.equals(G)===!1&&(i.scissor(G.x,G.y,G.z,G.w),ge.copy(G))}function Te(G){be.equals(G)===!1&&(i.viewport(G.x,G.y,G.z,G.w),be.copy(G))}function Ie(G,Re){let Se=c.get(Re);Se===void 0&&(Se=new WeakMap,c.set(Re,Se));let ve=Se.get(G);ve===void 0&&(ve=i.getUniformBlockIndex(Re,G.name),Se.set(G,ve))}function we(G,Re){const ve=c.get(Re).get(G);l.get(Re)!==ve&&(i.uniformBlockBinding(Re,ve,G.__bindingPointIndex),l.set(Re,ve))}function He(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),o.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),u={},$=null,X={},f={},d=new WeakMap,h=[],g=null,p=!1,x=null,m=null,_=null,A=null,S=null,v=null,b=null,M=new nt(0,0,0),E=0,y=!1,C=null,I=null,P=null,U=null,O=null,ge.set(0,0,i.canvas.width,i.canvas.height),be.set(0,0,i.canvas.width,i.canvas.height),r.reset(),o.reset(),a.reset()}return{buffers:{color:r,depth:o,stencil:a},enable:oe,disable:he,bindFramebuffer:Fe,drawBuffers:ye,useProgram:ke,setBlending:q,setMaterial:w,setFlipSided:ne,setCullFace:re,setLineWidth:ue,setPolygonOffset:ie,setScissorTest:fe,activeTexture:te,bindTexture:xe,unbindTexture:R,compressedTexImage2D:T,compressedTexImage3D:V,texImage2D:N,texImage3D:ee,updateUBOMapping:Ie,uniformBlockBinding:we,texStorage2D:Ae,texStorage3D:Ue,texSubImage2D:se,texSubImage3D:ce,compressedTexSubImage2D:J,compressedTexSubImage3D:Pe,scissor:me,viewport:Te,reset:He}}function XC(i,e,t,n,s,r,o){const a=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new Ne,u=new WeakMap;let f;const d=new WeakMap;let h=!1;try{h=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function g(R,T){return h?new OffscreenCanvas(R,T):wl("canvas")}function p(R,T,V){let se=1;const ce=xe(R);if((ce.width>V||ce.height>V)&&(se=V/Math.max(ce.width,ce.height)),se<1)if(typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&R instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&R instanceof ImageBitmap||typeof VideoFrame<"u"&&R instanceof VideoFrame){const J=Math.floor(se*ce.width),Pe=Math.floor(se*ce.height);f===void 0&&(f=g(J,Pe));const Ae=T?g(J,Pe):f;return Ae.width=J,Ae.height=Pe,Ae.getContext("2d").drawImage(R,0,0,J,Pe),je("WebGLRenderer: Texture has been resized from ("+ce.width+"x"+ce.height+") to ("+J+"x"+Pe+")."),Ae}else return"data"in R&&je("WebGLRenderer: Image in DataTexture is too big ("+ce.width+"x"+ce.height+")."),R;return R}function x(R){return R.generateMipmaps}function m(R){i.generateMipmap(R)}function _(R){return R.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:R.isWebGL3DRenderTarget?i.TEXTURE_3D:R.isWebGLArrayRenderTarget||R.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function A(R,T,V,se,ce=!1){if(R!==null){if(i[R]!==void 0)return i[R];je("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+R+"'")}let J=T;if(T===i.RED&&(V===i.FLOAT&&(J=i.R32F),V===i.HALF_FLOAT&&(J=i.R16F),V===i.UNSIGNED_BYTE&&(J=i.R8)),T===i.RED_INTEGER&&(V===i.UNSIGNED_BYTE&&(J=i.R8UI),V===i.UNSIGNED_SHORT&&(J=i.R16UI),V===i.UNSIGNED_INT&&(J=i.R32UI),V===i.BYTE&&(J=i.R8I),V===i.SHORT&&(J=i.R16I),V===i.INT&&(J=i.R32I)),T===i.RG&&(V===i.FLOAT&&(J=i.RG32F),V===i.HALF_FLOAT&&(J=i.RG16F),V===i.UNSIGNED_BYTE&&(J=i.RG8)),T===i.RG_INTEGER&&(V===i.UNSIGNED_BYTE&&(J=i.RG8UI),V===i.UNSIGNED_SHORT&&(J=i.RG16UI),V===i.UNSIGNED_INT&&(J=i.RG32UI),V===i.BYTE&&(J=i.RG8I),V===i.SHORT&&(J=i.RG16I),V===i.INT&&(J=i.RG32I)),T===i.RGB_INTEGER&&(V===i.UNSIGNED_BYTE&&(J=i.RGB8UI),V===i.UNSIGNED_SHORT&&(J=i.RGB16UI),V===i.UNSIGNED_INT&&(J=i.RGB32UI),V===i.BYTE&&(J=i.RGB8I),V===i.SHORT&&(J=i.RGB16I),V===i.INT&&(J=i.RGB32I)),T===i.RGBA_INTEGER&&(V===i.UNSIGNED_BYTE&&(J=i.RGBA8UI),V===i.UNSIGNED_SHORT&&(J=i.RGBA16UI),V===i.UNSIGNED_INT&&(J=i.RGBA32UI),V===i.BYTE&&(J=i.RGBA8I),V===i.SHORT&&(J=i.RGBA16I),V===i.INT&&(J=i.RGBA32I)),T===i.RGB&&(V===i.UNSIGNED_INT_5_9_9_9_REV&&(J=i.RGB9_E5),V===i.UNSIGNED_INT_10F_11F_11F_REV&&(J=i.R11F_G11F_B10F)),T===i.RGBA){const Pe=ce?Tl:rt.getTransfer(se);V===i.FLOAT&&(J=i.RGBA32F),V===i.HALF_FLOAT&&(J=i.RGBA16F),V===i.UNSIGNED_BYTE&&(J=Pe===ht?i.SRGB8_ALPHA8:i.RGBA8),V===i.UNSIGNED_SHORT_4_4_4_4&&(J=i.RGBA4),V===i.UNSIGNED_SHORT_5_5_5_1&&(J=i.RGB5_A1)}return(J===i.R16F||J===i.R32F||J===i.RG16F||J===i.RG32F||J===i.RGBA16F||J===i.RGBA32F)&&e.get("EXT_color_buffer_float"),J}function S(R,T){let V;return R?T===null||T===si||T===oa?V=i.DEPTH24_STENCIL8:T===mi?V=i.DEPTH32F_STENCIL8:T===ra&&(V=i.DEPTH24_STENCIL8,je("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):T===null||T===si||T===oa?V=i.DEPTH_COMPONENT24:T===mi?V=i.DEPTH_COMPONENT32F:T===ra&&(V=i.DEPTH_COMPONENT16),V}function v(R,T){return x(R)===!0||R.isFramebufferTexture&&R.minFilter!==qn&&R.minFilter!==ii?Math.log2(Math.max(T.width,T.height))+1:R.mipmaps!==void 0&&R.mipmaps.length>0?R.mipmaps.length:R.isCompressedTexture&&Array.isArray(R.image)?T.mipmaps.length:1}function b(R){const T=R.target;T.removeEventListener("dispose",b),E(T),T.isVideoTexture&&u.delete(T)}function M(R){const T=R.target;T.removeEventListener("dispose",M),C(T)}function E(R){const T=n.get(R);if(T.__webglInit===void 0)return;const V=R.source,se=d.get(V);if(se){const ce=se[T.__cacheKey];ce.usedTimes--,ce.usedTimes===0&&y(R),Object.keys(se).length===0&&d.delete(V)}n.remove(R)}function y(R){const T=n.get(R);i.deleteTexture(T.__webglTexture);const V=R.source,se=d.get(V);delete se[T.__cacheKey],o.memory.textures--}function C(R){const T=n.get(R);if(R.depthTexture&&(R.depthTexture.dispose(),n.remove(R.depthTexture)),R.isWebGLCubeRenderTarget)for(let se=0;se<6;se++){if(Array.isArray(T.__webglFramebuffer[se]))for(let ce=0;ce<T.__webglFramebuffer[se].length;ce++)i.deleteFramebuffer(T.__webglFramebuffer[se][ce]);else i.deleteFramebuffer(T.__webglFramebuffer[se]);T.__webglDepthbuffer&&i.deleteRenderbuffer(T.__webglDepthbuffer[se])}else{if(Array.isArray(T.__webglFramebuffer))for(let se=0;se<T.__webglFramebuffer.length;se++)i.deleteFramebuffer(T.__webglFramebuffer[se]);else i.deleteFramebuffer(T.__webglFramebuffer);if(T.__webglDepthbuffer&&i.deleteRenderbuffer(T.__webglDepthbuffer),T.__webglMultisampledFramebuffer&&i.deleteFramebuffer(T.__webglMultisampledFramebuffer),T.__webglColorRenderbuffer)for(let se=0;se<T.__webglColorRenderbuffer.length;se++)T.__webglColorRenderbuffer[se]&&i.deleteRenderbuffer(T.__webglColorRenderbuffer[se]);T.__webglDepthRenderbuffer&&i.deleteRenderbuffer(T.__webglDepthRenderbuffer)}const V=R.textures;for(let se=0,ce=V.length;se<ce;se++){const J=n.get(V[se]);J.__webglTexture&&(i.deleteTexture(J.__webglTexture),o.memory.textures--),n.remove(V[se])}n.remove(R)}let I=0;function P(){I=0}function U(){const R=I;return R>=s.maxTextures&&je("WebGLTextures: Trying to use "+R+" texture units while this GPU supports only "+s.maxTextures),I+=1,R}function O(R){const T=[];return T.push(R.wrapS),T.push(R.wrapT),T.push(R.wrapR||0),T.push(R.magFilter),T.push(R.minFilter),T.push(R.anisotropy),T.push(R.internalFormat),T.push(R.format),T.push(R.type),T.push(R.generateMipmaps),T.push(R.premultiplyAlpha),T.push(R.flipY),T.push(R.unpackAlignment),T.push(R.colorSpace),T.join()}function k(R,T){const V=n.get(R);if(R.isVideoTexture&&fe(R),R.isRenderTargetTexture===!1&&R.isExternalTexture!==!0&&R.version>0&&V.__version!==R.version){const se=R.image;if(se===null)je("WebGLRenderer: Texture marked for update but no image data found.");else if(se.complete===!1)je("WebGLRenderer: Texture marked for update but image is incomplete");else{Y(V,R,T);return}}else R.isExternalTexture&&(V.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(i.TEXTURE_2D,V.__webglTexture,i.TEXTURE0+T)}function z(R,T){const V=n.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&V.__version!==R.version){Y(V,R,T);return}else R.isExternalTexture&&(V.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(i.TEXTURE_2D_ARRAY,V.__webglTexture,i.TEXTURE0+T)}function W(R,T){const V=n.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&V.__version!==R.version){Y(V,R,T);return}t.bindTexture(i.TEXTURE_3D,V.__webglTexture,i.TEXTURE0+T)}function H(R,T){const V=n.get(R);if(R.version>0&&V.__version!==R.version){oe(V,R,T);return}t.bindTexture(i.TEXTURE_CUBE_MAP,V.__webglTexture,i.TEXTURE0+T)}const $={[Du]:i.REPEAT,[ns]:i.CLAMP_TO_EDGE,[Pu]:i.MIRRORED_REPEAT},X={[qn]:i.NEAREST,[gS]:i.NEAREST_MIPMAP_NEAREST,[Ia]:i.NEAREST_MIPMAP_LINEAR,[ii]:i.LINEAR,[_c]:i.LINEAR_MIPMAP_NEAREST,[sr]:i.LINEAR_MIPMAP_LINEAR},Z={[vS]:i.NEVER,[ES]:i.ALWAYS,[yS]:i.LESS,[L0]:i.LEQUAL,[bS]:i.EQUAL,[TS]:i.GEQUAL,[MS]:i.GREATER,[CS]:i.NOTEQUAL};function pe(R,T){if(T.type===mi&&e.has("OES_texture_float_linear")===!1&&(T.magFilter===ii||T.magFilter===_c||T.magFilter===Ia||T.magFilter===sr||T.minFilter===ii||T.minFilter===_c||T.minFilter===Ia||T.minFilter===sr)&&je("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(R,i.TEXTURE_WRAP_S,$[T.wrapS]),i.texParameteri(R,i.TEXTURE_WRAP_T,$[T.wrapT]),(R===i.TEXTURE_3D||R===i.TEXTURE_2D_ARRAY)&&i.texParameteri(R,i.TEXTURE_WRAP_R,$[T.wrapR]),i.texParameteri(R,i.TEXTURE_MAG_FILTER,X[T.magFilter]),i.texParameteri(R,i.TEXTURE_MIN_FILTER,X[T.minFilter]),T.compareFunction&&(i.texParameteri(R,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(R,i.TEXTURE_COMPARE_FUNC,Z[T.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(T.magFilter===qn||T.minFilter!==Ia&&T.minFilter!==sr||T.type===mi&&e.has("OES_texture_float_linear")===!1)return;if(T.anisotropy>1||n.get(T).__currentAnisotropy){const V=e.get("EXT_texture_filter_anisotropic");i.texParameterf(R,V.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(T.anisotropy,s.getMaxAnisotropy())),n.get(T).__currentAnisotropy=T.anisotropy}}}function ge(R,T){let V=!1;R.__webglInit===void 0&&(R.__webglInit=!0,T.addEventListener("dispose",b));const se=T.source;let ce=d.get(se);ce===void 0&&(ce={},d.set(se,ce));const J=O(T);if(J!==R.__cacheKey){ce[J]===void 0&&(ce[J]={texture:i.createTexture(),usedTimes:0},o.memory.textures++,V=!0),ce[J].usedTimes++;const Pe=ce[R.__cacheKey];Pe!==void 0&&(ce[R.__cacheKey].usedTimes--,Pe.usedTimes===0&&y(T)),R.__cacheKey=J,R.__webglTexture=ce[J].texture}return V}function be(R,T,V){return Math.floor(Math.floor(R/V)/T)}function Me(R,T,V,se){const J=R.updateRanges;if(J.length===0)t.texSubImage2D(i.TEXTURE_2D,0,0,0,T.width,T.height,V,se,T.data);else{J.sort((ee,me)=>ee.start-me.start);let Pe=0;for(let ee=1;ee<J.length;ee++){const me=J[Pe],Te=J[ee],Ie=me.start+me.count,we=be(Te.start,T.width,4),He=be(me.start,T.width,4);Te.start<=Ie+1&&we===He&&be(Te.start+Te.count-1,T.width,4)===we?me.count=Math.max(me.count,Te.start+Te.count-me.start):(++Pe,J[Pe]=Te)}J.length=Pe+1;const Ae=i.getParameter(i.UNPACK_ROW_LENGTH),Ue=i.getParameter(i.UNPACK_SKIP_PIXELS),N=i.getParameter(i.UNPACK_SKIP_ROWS);i.pixelStorei(i.UNPACK_ROW_LENGTH,T.width);for(let ee=0,me=J.length;ee<me;ee++){const Te=J[ee],Ie=Math.floor(Te.start/4),we=Math.ceil(Te.count/4),He=Ie%T.width,G=Math.floor(Ie/T.width),Re=we,Se=1;i.pixelStorei(i.UNPACK_SKIP_PIXELS,He),i.pixelStorei(i.UNPACK_SKIP_ROWS,G),t.texSubImage2D(i.TEXTURE_2D,0,He,G,Re,Se,V,se,T.data)}R.clearUpdateRanges(),i.pixelStorei(i.UNPACK_ROW_LENGTH,Ae),i.pixelStorei(i.UNPACK_SKIP_PIXELS,Ue),i.pixelStorei(i.UNPACK_SKIP_ROWS,N)}}function Y(R,T,V){let se=i.TEXTURE_2D;(T.isDataArrayTexture||T.isCompressedArrayTexture)&&(se=i.TEXTURE_2D_ARRAY),T.isData3DTexture&&(se=i.TEXTURE_3D);const ce=ge(R,T),J=T.source;t.bindTexture(se,R.__webglTexture,i.TEXTURE0+V);const Pe=n.get(J);if(J.version!==Pe.__version||ce===!0){t.activeTexture(i.TEXTURE0+V);const Ae=rt.getPrimaries(rt.workingColorSpace),Ue=T.colorSpace===bs?null:rt.getPrimaries(T.colorSpace),N=T.colorSpace===bs||Ae===Ue?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,T.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,T.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,N);let ee=p(T.image,!1,s.maxTextureSize);ee=te(T,ee);const me=r.convert(T.format,T.colorSpace),Te=r.convert(T.type);let Ie=A(T.internalFormat,me,Te,T.colorSpace,T.isVideoTexture);pe(se,T);let we;const He=T.mipmaps,G=T.isVideoTexture!==!0,Re=Pe.__version===void 0||ce===!0,Se=J.dataReady,ve=v(T,ee);if(T.isDepthTexture)Ie=S(T.format===aa,T.type),Re&&(G?t.texStorage2D(i.TEXTURE_2D,1,Ie,ee.width,ee.height):t.texImage2D(i.TEXTURE_2D,0,Ie,ee.width,ee.height,0,me,Te,null));else if(T.isDataTexture)if(He.length>0){G&&Re&&t.texStorage2D(i.TEXTURE_2D,ve,Ie,He[0].width,He[0].height);for(let _e=0,de=He.length;_e<de;_e++)we=He[_e],G?Se&&t.texSubImage2D(i.TEXTURE_2D,_e,0,0,we.width,we.height,me,Te,we.data):t.texImage2D(i.TEXTURE_2D,_e,Ie,we.width,we.height,0,me,Te,we.data);T.generateMipmaps=!1}else G?(Re&&t.texStorage2D(i.TEXTURE_2D,ve,Ie,ee.width,ee.height),Se&&Me(T,ee,me,Te)):t.texImage2D(i.TEXTURE_2D,0,Ie,ee.width,ee.height,0,me,Te,ee.data);else if(T.isCompressedTexture)if(T.isCompressedArrayTexture){G&&Re&&t.texStorage3D(i.TEXTURE_2D_ARRAY,ve,Ie,He[0].width,He[0].height,ee.depth);for(let _e=0,de=He.length;_e<de;_e++)if(we=He[_e],T.format!==gn)if(me!==null)if(G){if(Se)if(T.layerUpdates.size>0){const Be=Jh(we.width,we.height,T.format,T.type);for(const We of T.layerUpdates){const vt=we.data.subarray(We*Be/we.data.BYTES_PER_ELEMENT,(We+1)*Be/we.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,_e,0,0,We,we.width,we.height,1,me,vt)}T.clearLayerUpdates()}else t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,_e,0,0,0,we.width,we.height,ee.depth,me,we.data)}else t.compressedTexImage3D(i.TEXTURE_2D_ARRAY,_e,Ie,we.width,we.height,ee.depth,0,we.data,0,0);else je("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else G?Se&&t.texSubImage3D(i.TEXTURE_2D_ARRAY,_e,0,0,0,we.width,we.height,ee.depth,me,Te,we.data):t.texImage3D(i.TEXTURE_2D_ARRAY,_e,Ie,we.width,we.height,ee.depth,0,me,Te,we.data)}else{G&&Re&&t.texStorage2D(i.TEXTURE_2D,ve,Ie,He[0].width,He[0].height);for(let _e=0,de=He.length;_e<de;_e++)we=He[_e],T.format!==gn?me!==null?G?Se&&t.compressedTexSubImage2D(i.TEXTURE_2D,_e,0,0,we.width,we.height,me,we.data):t.compressedTexImage2D(i.TEXTURE_2D,_e,Ie,we.width,we.height,0,we.data):je("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):G?Se&&t.texSubImage2D(i.TEXTURE_2D,_e,0,0,we.width,we.height,me,Te,we.data):t.texImage2D(i.TEXTURE_2D,_e,Ie,we.width,we.height,0,me,Te,we.data)}else if(T.isDataArrayTexture)if(G){if(Re&&t.texStorage3D(i.TEXTURE_2D_ARRAY,ve,Ie,ee.width,ee.height,ee.depth),Se)if(T.layerUpdates.size>0){const _e=Jh(ee.width,ee.height,T.format,T.type);for(const de of T.layerUpdates){const Be=ee.data.subarray(de*_e/ee.data.BYTES_PER_ELEMENT,(de+1)*_e/ee.data.BYTES_PER_ELEMENT);t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,de,ee.width,ee.height,1,me,Te,Be)}T.clearLayerUpdates()}else t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,ee.width,ee.height,ee.depth,me,Te,ee.data)}else t.texImage3D(i.TEXTURE_2D_ARRAY,0,Ie,ee.width,ee.height,ee.depth,0,me,Te,ee.data);else if(T.isData3DTexture)G?(Re&&t.texStorage3D(i.TEXTURE_3D,ve,Ie,ee.width,ee.height,ee.depth),Se&&t.texSubImage3D(i.TEXTURE_3D,0,0,0,0,ee.width,ee.height,ee.depth,me,Te,ee.data)):t.texImage3D(i.TEXTURE_3D,0,Ie,ee.width,ee.height,ee.depth,0,me,Te,ee.data);else if(T.isFramebufferTexture){if(Re)if(G)t.texStorage2D(i.TEXTURE_2D,ve,Ie,ee.width,ee.height);else{let _e=ee.width,de=ee.height;for(let Be=0;Be<ve;Be++)t.texImage2D(i.TEXTURE_2D,Be,Ie,_e,de,0,me,Te,null),_e>>=1,de>>=1}}else if(He.length>0){if(G&&Re){const _e=xe(He[0]);t.texStorage2D(i.TEXTURE_2D,ve,Ie,_e.width,_e.height)}for(let _e=0,de=He.length;_e<de;_e++)we=He[_e],G?Se&&t.texSubImage2D(i.TEXTURE_2D,_e,0,0,me,Te,we):t.texImage2D(i.TEXTURE_2D,_e,Ie,me,Te,we);T.generateMipmaps=!1}else if(G){if(Re){const _e=xe(ee);t.texStorage2D(i.TEXTURE_2D,ve,Ie,_e.width,_e.height)}Se&&t.texSubImage2D(i.TEXTURE_2D,0,0,0,me,Te,ee)}else t.texImage2D(i.TEXTURE_2D,0,Ie,me,Te,ee);x(T)&&m(se),Pe.__version=J.version,T.onUpdate&&T.onUpdate(T)}R.__version=T.version}function oe(R,T,V){if(T.image.length!==6)return;const se=ge(R,T),ce=T.source;t.bindTexture(i.TEXTURE_CUBE_MAP,R.__webglTexture,i.TEXTURE0+V);const J=n.get(ce);if(ce.version!==J.__version||se===!0){t.activeTexture(i.TEXTURE0+V);const Pe=rt.getPrimaries(rt.workingColorSpace),Ae=T.colorSpace===bs?null:rt.getPrimaries(T.colorSpace),Ue=T.colorSpace===bs||Pe===Ae?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,T.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,T.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,Ue);const N=T.isCompressedTexture||T.image[0].isCompressedTexture,ee=T.image[0]&&T.image[0].isDataTexture,me=[];for(let de=0;de<6;de++)!N&&!ee?me[de]=p(T.image[de],!0,s.maxCubemapSize):me[de]=ee?T.image[de].image:T.image[de],me[de]=te(T,me[de]);const Te=me[0],Ie=r.convert(T.format,T.colorSpace),we=r.convert(T.type),He=A(T.internalFormat,Ie,we,T.colorSpace),G=T.isVideoTexture!==!0,Re=J.__version===void 0||se===!0,Se=ce.dataReady;let ve=v(T,Te);pe(i.TEXTURE_CUBE_MAP,T);let _e;if(N){G&&Re&&t.texStorage2D(i.TEXTURE_CUBE_MAP,ve,He,Te.width,Te.height);for(let de=0;de<6;de++){_e=me[de].mipmaps;for(let Be=0;Be<_e.length;Be++){const We=_e[Be];T.format!==gn?Ie!==null?G?Se&&t.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,Be,0,0,We.width,We.height,Ie,We.data):t.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,Be,He,We.width,We.height,0,We.data):je("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):G?Se&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,Be,0,0,We.width,We.height,Ie,we,We.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,Be,He,We.width,We.height,0,Ie,we,We.data)}}}else{if(_e=T.mipmaps,G&&Re){_e.length>0&&ve++;const de=xe(me[0]);t.texStorage2D(i.TEXTURE_CUBE_MAP,ve,He,de.width,de.height)}for(let de=0;de<6;de++)if(ee){G?Se&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,0,0,me[de].width,me[de].height,Ie,we,me[de].data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,He,me[de].width,me[de].height,0,Ie,we,me[de].data);for(let Be=0;Be<_e.length;Be++){const vt=_e[Be].image[de].image;G?Se&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,Be+1,0,0,vt.width,vt.height,Ie,we,vt.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,Be+1,He,vt.width,vt.height,0,Ie,we,vt.data)}}else{G?Se&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,0,0,Ie,we,me[de]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,He,Ie,we,me[de]);for(let Be=0;Be<_e.length;Be++){const We=_e[Be];G?Se&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,Be+1,0,0,Ie,we,We.image[de]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,Be+1,He,Ie,we,We.image[de])}}}x(T)&&m(i.TEXTURE_CUBE_MAP),J.__version=ce.version,T.onUpdate&&T.onUpdate(T)}R.__version=T.version}function he(R,T,V,se,ce,J){const Pe=r.convert(V.format,V.colorSpace),Ae=r.convert(V.type),Ue=A(V.internalFormat,Pe,Ae,V.colorSpace),N=n.get(T),ee=n.get(V);if(ee.__renderTarget=T,!N.__hasExternalTextures){const me=Math.max(1,T.width>>J),Te=Math.max(1,T.height>>J);ce===i.TEXTURE_3D||ce===i.TEXTURE_2D_ARRAY?t.texImage3D(ce,J,Ue,me,Te,T.depth,0,Pe,Ae,null):t.texImage2D(ce,J,Ue,me,Te,0,Pe,Ae,null)}t.bindFramebuffer(i.FRAMEBUFFER,R),ie(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,se,ce,ee.__webglTexture,0,ue(T)):(ce===i.TEXTURE_2D||ce>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&ce<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,se,ce,ee.__webglTexture,J),t.bindFramebuffer(i.FRAMEBUFFER,null)}function Fe(R,T,V){if(i.bindRenderbuffer(i.RENDERBUFFER,R),T.depthBuffer){const se=T.depthTexture,ce=se&&se.isDepthTexture?se.type:null,J=S(T.stencilBuffer,ce),Pe=T.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Ae=ue(T);ie(T)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,Ae,J,T.width,T.height):V?i.renderbufferStorageMultisample(i.RENDERBUFFER,Ae,J,T.width,T.height):i.renderbufferStorage(i.RENDERBUFFER,J,T.width,T.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,Pe,i.RENDERBUFFER,R)}else{const se=T.textures;for(let ce=0;ce<se.length;ce++){const J=se[ce],Pe=r.convert(J.format,J.colorSpace),Ae=r.convert(J.type),Ue=A(J.internalFormat,Pe,Ae,J.colorSpace),N=ue(T);V&&ie(T)===!1?i.renderbufferStorageMultisample(i.RENDERBUFFER,N,Ue,T.width,T.height):ie(T)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,N,Ue,T.width,T.height):i.renderbufferStorage(i.RENDERBUFFER,Ue,T.width,T.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function ye(R,T){if(T&&T.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(t.bindFramebuffer(i.FRAMEBUFFER,R),!(T.depthTexture&&T.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const se=n.get(T.depthTexture);se.__renderTarget=T,(!se.__webglTexture||T.depthTexture.image.width!==T.width||T.depthTexture.image.height!==T.height)&&(T.depthTexture.image.width=T.width,T.depthTexture.image.height=T.height,T.depthTexture.needsUpdate=!0),k(T.depthTexture,0);const ce=se.__webglTexture,J=ue(T);if(T.depthTexture.format===co)ie(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,ce,0,J):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,ce,0);else if(T.depthTexture.format===aa)ie(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,ce,0,J):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,ce,0);else throw new Error("Unknown depthTexture format")}function ke(R){const T=n.get(R),V=R.isWebGLCubeRenderTarget===!0;if(T.__boundDepthTexture!==R.depthTexture){const se=R.depthTexture;if(T.__depthDisposeCallback&&T.__depthDisposeCallback(),se){const ce=()=>{delete T.__boundDepthTexture,delete T.__depthDisposeCallback,se.removeEventListener("dispose",ce)};se.addEventListener("dispose",ce),T.__depthDisposeCallback=ce}T.__boundDepthTexture=se}if(R.depthTexture&&!T.__autoAllocateDepthBuffer){if(V)throw new Error("target.depthTexture not supported in Cube render targets");const se=R.texture.mipmaps;se&&se.length>0?ye(T.__webglFramebuffer[0],R):ye(T.__webglFramebuffer,R)}else if(V){T.__webglDepthbuffer=[];for(let se=0;se<6;se++)if(t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer[se]),T.__webglDepthbuffer[se]===void 0)T.__webglDepthbuffer[se]=i.createRenderbuffer(),Fe(T.__webglDepthbuffer[se],R,!1);else{const ce=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,J=T.__webglDepthbuffer[se];i.bindRenderbuffer(i.RENDERBUFFER,J),i.framebufferRenderbuffer(i.FRAMEBUFFER,ce,i.RENDERBUFFER,J)}}else{const se=R.texture.mipmaps;if(se&&se.length>0?t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer[0]):t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer),T.__webglDepthbuffer===void 0)T.__webglDepthbuffer=i.createRenderbuffer(),Fe(T.__webglDepthbuffer,R,!1);else{const ce=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,J=T.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,J),i.framebufferRenderbuffer(i.FRAMEBUFFER,ce,i.RENDERBUFFER,J)}}t.bindFramebuffer(i.FRAMEBUFFER,null)}function F(R,T,V){const se=n.get(R);T!==void 0&&he(se.__webglFramebuffer,R,R.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),V!==void 0&&ke(R)}function B(R){const T=R.texture,V=n.get(R),se=n.get(T);R.addEventListener("dispose",M);const ce=R.textures,J=R.isWebGLCubeRenderTarget===!0,Pe=ce.length>1;if(Pe||(se.__webglTexture===void 0&&(se.__webglTexture=i.createTexture()),se.__version=T.version,o.memory.textures++),J){V.__webglFramebuffer=[];for(let Ae=0;Ae<6;Ae++)if(T.mipmaps&&T.mipmaps.length>0){V.__webglFramebuffer[Ae]=[];for(let Ue=0;Ue<T.mipmaps.length;Ue++)V.__webglFramebuffer[Ae][Ue]=i.createFramebuffer()}else V.__webglFramebuffer[Ae]=i.createFramebuffer()}else{if(T.mipmaps&&T.mipmaps.length>0){V.__webglFramebuffer=[];for(let Ae=0;Ae<T.mipmaps.length;Ae++)V.__webglFramebuffer[Ae]=i.createFramebuffer()}else V.__webglFramebuffer=i.createFramebuffer();if(Pe)for(let Ae=0,Ue=ce.length;Ae<Ue;Ae++){const N=n.get(ce[Ae]);N.__webglTexture===void 0&&(N.__webglTexture=i.createTexture(),o.memory.textures++)}if(R.samples>0&&ie(R)===!1){V.__webglMultisampledFramebuffer=i.createFramebuffer(),V.__webglColorRenderbuffer=[],t.bindFramebuffer(i.FRAMEBUFFER,V.__webglMultisampledFramebuffer);for(let Ae=0;Ae<ce.length;Ae++){const Ue=ce[Ae];V.__webglColorRenderbuffer[Ae]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,V.__webglColorRenderbuffer[Ae]);const N=r.convert(Ue.format,Ue.colorSpace),ee=r.convert(Ue.type),me=A(Ue.internalFormat,N,ee,Ue.colorSpace,R.isXRRenderTarget===!0),Te=ue(R);i.renderbufferStorageMultisample(i.RENDERBUFFER,Te,me,R.width,R.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+Ae,i.RENDERBUFFER,V.__webglColorRenderbuffer[Ae])}i.bindRenderbuffer(i.RENDERBUFFER,null),R.depthBuffer&&(V.__webglDepthRenderbuffer=i.createRenderbuffer(),Fe(V.__webglDepthRenderbuffer,R,!0)),t.bindFramebuffer(i.FRAMEBUFFER,null)}}if(J){t.bindTexture(i.TEXTURE_CUBE_MAP,se.__webglTexture),pe(i.TEXTURE_CUBE_MAP,T);for(let Ae=0;Ae<6;Ae++)if(T.mipmaps&&T.mipmaps.length>0)for(let Ue=0;Ue<T.mipmaps.length;Ue++)he(V.__webglFramebuffer[Ae][Ue],R,T,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+Ae,Ue);else he(V.__webglFramebuffer[Ae],R,T,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+Ae,0);x(T)&&m(i.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(Pe){for(let Ae=0,Ue=ce.length;Ae<Ue;Ae++){const N=ce[Ae],ee=n.get(N);let me=i.TEXTURE_2D;(R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(me=R.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(me,ee.__webglTexture),pe(me,N),he(V.__webglFramebuffer,R,N,i.COLOR_ATTACHMENT0+Ae,me,0),x(N)&&m(me)}t.unbindTexture()}else{let Ae=i.TEXTURE_2D;if((R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(Ae=R.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(Ae,se.__webglTexture),pe(Ae,T),T.mipmaps&&T.mipmaps.length>0)for(let Ue=0;Ue<T.mipmaps.length;Ue++)he(V.__webglFramebuffer[Ue],R,T,i.COLOR_ATTACHMENT0,Ae,Ue);else he(V.__webglFramebuffer,R,T,i.COLOR_ATTACHMENT0,Ae,0);x(T)&&m(Ae),t.unbindTexture()}R.depthBuffer&&ke(R)}function q(R){const T=R.textures;for(let V=0,se=T.length;V<se;V++){const ce=T[V];if(x(ce)){const J=_(R),Pe=n.get(ce).__webglTexture;t.bindTexture(J,Pe),m(J),t.unbindTexture()}}}const w=[],ne=[];function re(R){if(R.samples>0){if(ie(R)===!1){const T=R.textures,V=R.width,se=R.height;let ce=i.COLOR_BUFFER_BIT;const J=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Pe=n.get(R),Ae=T.length>1;if(Ae)for(let N=0;N<T.length;N++)t.bindFramebuffer(i.FRAMEBUFFER,Pe.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+N,i.RENDERBUFFER,null),t.bindFramebuffer(i.FRAMEBUFFER,Pe.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+N,i.TEXTURE_2D,null,0);t.bindFramebuffer(i.READ_FRAMEBUFFER,Pe.__webglMultisampledFramebuffer);const Ue=R.texture.mipmaps;Ue&&Ue.length>0?t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Pe.__webglFramebuffer[0]):t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Pe.__webglFramebuffer);for(let N=0;N<T.length;N++){if(R.resolveDepthBuffer&&(R.depthBuffer&&(ce|=i.DEPTH_BUFFER_BIT),R.stencilBuffer&&R.resolveStencilBuffer&&(ce|=i.STENCIL_BUFFER_BIT)),Ae){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,Pe.__webglColorRenderbuffer[N]);const ee=n.get(T[N]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,ee,0)}i.blitFramebuffer(0,0,V,se,0,0,V,se,ce,i.NEAREST),l===!0&&(w.length=0,ne.length=0,w.push(i.COLOR_ATTACHMENT0+N),R.depthBuffer&&R.resolveDepthBuffer===!1&&(w.push(J),ne.push(J),i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,ne)),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,w))}if(t.bindFramebuffer(i.READ_FRAMEBUFFER,null),t.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),Ae)for(let N=0;N<T.length;N++){t.bindFramebuffer(i.FRAMEBUFFER,Pe.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+N,i.RENDERBUFFER,Pe.__webglColorRenderbuffer[N]);const ee=n.get(T[N]).__webglTexture;t.bindFramebuffer(i.FRAMEBUFFER,Pe.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+N,i.TEXTURE_2D,ee,0)}t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Pe.__webglMultisampledFramebuffer)}else if(R.depthBuffer&&R.resolveDepthBuffer===!1&&l){const T=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[T])}}}function ue(R){return Math.min(s.maxSamples,R.samples)}function ie(R){const T=n.get(R);return R.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&T.__useRenderToTexture!==!1}function fe(R){const T=o.render.frame;u.get(R)!==T&&(u.set(R,T),R.update())}function te(R,T){const V=R.colorSpace,se=R.format,ce=R.type;return R.isCompressedTexture===!0||R.isVideoTexture===!0||V!==uo&&V!==bs&&(rt.getTransfer(V)===ht?(se!==gn||ce!==Ui)&&je("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):zt("WebGLTextures: Unsupported texture color space:",V)),T}function xe(R){return typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement?(c.width=R.naturalWidth||R.width,c.height=R.naturalHeight||R.height):typeof VideoFrame<"u"&&R instanceof VideoFrame?(c.width=R.displayWidth,c.height=R.displayHeight):(c.width=R.width,c.height=R.height),c}this.allocateTextureUnit=U,this.resetTextureUnits=P,this.setTexture2D=k,this.setTexture2DArray=z,this.setTexture3D=W,this.setTextureCube=H,this.rebindTextures=F,this.setupRenderTarget=B,this.updateRenderTargetMipmap=q,this.updateMultisampleRenderTarget=re,this.setupDepthRenderbuffer=ke,this.setupFrameBufferTexture=he,this.useMultisampledRTT=ie}function Z0(i,e){function t(n,s=bs){let r;const o=rt.getTransfer(s);if(n===Ui)return i.UNSIGNED_BYTE;if(n===Kf)return i.UNSIGNED_SHORT_4_4_4_4;if(n===jf)return i.UNSIGNED_SHORT_5_5_5_1;if(n===R0)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===I0)return i.UNSIGNED_INT_10F_11F_11F_REV;if(n===E0)return i.BYTE;if(n===w0)return i.SHORT;if(n===ra)return i.UNSIGNED_SHORT;if(n===Yf)return i.INT;if(n===si)return i.UNSIGNED_INT;if(n===mi)return i.FLOAT;if(n===pr)return i.HALF_FLOAT;if(n===D0)return i.ALPHA;if(n===P0)return i.RGB;if(n===gn)return i.RGBA;if(n===co)return i.DEPTH_COMPONENT;if(n===aa)return i.DEPTH_STENCIL;if(n===F0)return i.RED;if(n===jl)return i.RED_INTEGER;if(n===$f)return i.RG;if(n===Zf)return i.RG_INTEGER;if(n===jr)return i.RGBA_INTEGER;if(n===fl||n===dl||n===hl||n===pl)if(o===ht)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(n===fl)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===dl)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===hl)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===pl)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(n===fl)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===dl)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===hl)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===pl)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===Fu||n===Lu||n===Bu||n===Uu)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(n===Fu)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===Lu)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===Bu)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===Uu)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===Ou||n===Nu||n===zu)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(n===Ou||n===Nu)return o===ht?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(n===zu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(n===ku||n===Hu||n===Vu||n===Gu||n===Wu||n===Xu||n===qu||n===Qu||n===Yu||n===Ku||n===ju||n===$u||n===Zu||n===Ju)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(n===ku)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===Hu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===Vu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===Gu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===Wu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===Xu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===qu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===Qu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===Yu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===Ku)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===ju)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===$u)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===Zu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===Ju)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===ef||n===tf||n===nf)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(n===ef)return o===ht?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===tf)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===nf)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===sf||n===rf||n===of||n===af)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(n===sf)return r.COMPRESSED_RED_RGTC1_EXT;if(n===rf)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===of)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===af)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===oa?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:t}}const qC=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,QC=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;class YC{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){const n=new q0(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=n}}getMesh(e){if(this.texture!==null&&this.mesh===null){const t=e.cameras[0].viewport,n=new _n({vertexShader:qC,fragmentShader:QC,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new Vt(new ho(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class KC extends mr{constructor(e,t){super();const n=this;let s=null,r=1,o=null,a="local-floor",l=1,c=null,u=null,f=null,d=null,h=null,g=null;const p=typeof XRWebGLBinding<"u",x=new YC,m={},_=t.getContextAttributes();let A=null,S=null;const v=[],b=[],M=new Ne;let E=null;const y=new ei;y.viewport=new Et;const C=new ei;C.viewport=new Et;const I=[y,C],P=new mv;let U=null,O=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(Y){let oe=v[Y];return oe===void 0&&(oe=new kc,v[Y]=oe),oe.getTargetRaySpace()},this.getControllerGrip=function(Y){let oe=v[Y];return oe===void 0&&(oe=new kc,v[Y]=oe),oe.getGripSpace()},this.getHand=function(Y){let oe=v[Y];return oe===void 0&&(oe=new kc,v[Y]=oe),oe.getHandSpace()};function k(Y){const oe=b.indexOf(Y.inputSource);if(oe===-1)return;const he=v[oe];he!==void 0&&(he.update(Y.inputSource,Y.frame,c||o),he.dispatchEvent({type:Y.type,data:Y.inputSource}))}function z(){s.removeEventListener("select",k),s.removeEventListener("selectstart",k),s.removeEventListener("selectend",k),s.removeEventListener("squeeze",k),s.removeEventListener("squeezestart",k),s.removeEventListener("squeezeend",k),s.removeEventListener("end",z),s.removeEventListener("inputsourceschange",W);for(let Y=0;Y<v.length;Y++){const oe=b[Y];oe!==null&&(b[Y]=null,v[Y].disconnect(oe))}U=null,O=null,x.reset();for(const Y in m)delete m[Y];e.setRenderTarget(A),h=null,d=null,f=null,s=null,S=null,Me.stop(),n.isPresenting=!1,e.setPixelRatio(E),e.setSize(M.width,M.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(Y){r=Y,n.isPresenting===!0&&je("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(Y){a=Y,n.isPresenting===!0&&je("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return c||o},this.setReferenceSpace=function(Y){c=Y},this.getBaseLayer=function(){return d!==null?d:h},this.getBinding=function(){return f===null&&p&&(f=new XRWebGLBinding(s,t)),f},this.getFrame=function(){return g},this.getSession=function(){return s},this.setSession=async function(Y){if(s=Y,s!==null){if(A=e.getRenderTarget(),s.addEventListener("select",k),s.addEventListener("selectstart",k),s.addEventListener("selectend",k),s.addEventListener("squeeze",k),s.addEventListener("squeezestart",k),s.addEventListener("squeezeend",k),s.addEventListener("end",z),s.addEventListener("inputsourceschange",W),_.xrCompatible!==!0&&await t.makeXRCompatible(),E=e.getPixelRatio(),e.getSize(M),p&&"createProjectionLayer"in XRWebGLBinding.prototype){let he=null,Fe=null,ye=null;_.depth&&(ye=_.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,he=_.stencil?aa:co,Fe=_.stencil?oa:si);const ke={colorFormat:t.RGBA8,depthFormat:ye,scaleFactor:r};f=this.getBinding(),d=f.createProjectionLayer(ke),s.updateRenderState({layers:[d]}),e.setPixelRatio(1),e.setSize(d.textureWidth,d.textureHeight,!1),S=new Us(d.textureWidth,d.textureHeight,{format:gn,type:Ui,depthTexture:new td(d.textureWidth,d.textureHeight,Fe,void 0,void 0,void 0,void 0,void 0,void 0,he),stencilBuffer:_.stencil,colorSpace:e.outputColorSpace,samples:_.antialias?4:0,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1})}else{const he={antialias:_.antialias,alpha:!0,depth:_.depth,stencil:_.stencil,framebufferScaleFactor:r};h=new XRWebGLLayer(s,t,he),s.updateRenderState({baseLayer:h}),e.setPixelRatio(1),e.setSize(h.framebufferWidth,h.framebufferHeight,!1),S=new Us(h.framebufferWidth,h.framebufferHeight,{format:gn,type:Ui,colorSpace:e.outputColorSpace,stencilBuffer:_.stencil,resolveDepthBuffer:h.ignoreDepthValues===!1,resolveStencilBuffer:h.ignoreDepthValues===!1})}S.isXRRenderTarget=!0,this.setFoveation(l),c=null,o=await s.requestReferenceSpace(a),Me.setContext(s),Me.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return x.getDepthTexture()};function W(Y){for(let oe=0;oe<Y.removed.length;oe++){const he=Y.removed[oe],Fe=b.indexOf(he);Fe>=0&&(b[Fe]=null,v[Fe].disconnect(he))}for(let oe=0;oe<Y.added.length;oe++){const he=Y.added[oe];let Fe=b.indexOf(he);if(Fe===-1){for(let ke=0;ke<v.length;ke++)if(ke>=b.length){b.push(he),Fe=ke;break}else if(b[ke]===null){b[ke]=he,Fe=ke;break}if(Fe===-1)break}const ye=v[Fe];ye&&ye.connect(he)}}const H=new L,$=new L;function X(Y,oe,he){H.setFromMatrixPosition(oe.matrixWorld),$.setFromMatrixPosition(he.matrixWorld);const Fe=H.distanceTo($),ye=oe.projectionMatrix.elements,ke=he.projectionMatrix.elements,F=ye[14]/(ye[10]-1),B=ye[14]/(ye[10]+1),q=(ye[9]+1)/ye[5],w=(ye[9]-1)/ye[5],ne=(ye[8]-1)/ye[0],re=(ke[8]+1)/ke[0],ue=F*ne,ie=F*re,fe=Fe/(-ne+re),te=fe*-ne;if(oe.matrixWorld.decompose(Y.position,Y.quaternion,Y.scale),Y.translateX(te),Y.translateZ(fe),Y.matrixWorld.compose(Y.position,Y.quaternion,Y.scale),Y.matrixWorldInverse.copy(Y.matrixWorld).invert(),ye[10]===-1)Y.projectionMatrix.copy(oe.projectionMatrix),Y.projectionMatrixInverse.copy(oe.projectionMatrixInverse);else{const xe=F+fe,R=B+fe,T=ue-te,V=ie+(Fe-te),se=q*B/R*xe,ce=w*B/R*xe;Y.projectionMatrix.makePerspective(T,V,se,ce,xe,R),Y.projectionMatrixInverse.copy(Y.projectionMatrix).invert()}}function Z(Y,oe){oe===null?Y.matrixWorld.copy(Y.matrix):Y.matrixWorld.multiplyMatrices(oe.matrixWorld,Y.matrix),Y.matrixWorldInverse.copy(Y.matrixWorld).invert()}this.updateCamera=function(Y){if(s===null)return;let oe=Y.near,he=Y.far;x.texture!==null&&(x.depthNear>0&&(oe=x.depthNear),x.depthFar>0&&(he=x.depthFar)),P.near=C.near=y.near=oe,P.far=C.far=y.far=he,(U!==P.near||O!==P.far)&&(s.updateRenderState({depthNear:P.near,depthFar:P.far}),U=P.near,O=P.far),P.layers.mask=Y.layers.mask|6,y.layers.mask=P.layers.mask&3,C.layers.mask=P.layers.mask&5;const Fe=Y.parent,ye=P.cameras;Z(P,Fe);for(let ke=0;ke<ye.length;ke++)Z(ye[ke],Fe);ye.length===2?X(P,y,C):P.projectionMatrix.copy(y.projectionMatrix),pe(Y,P,Fe)};function pe(Y,oe,he){he===null?Y.matrix.copy(oe.matrixWorld):(Y.matrix.copy(he.matrixWorld),Y.matrix.invert(),Y.matrix.multiply(oe.matrixWorld)),Y.matrix.decompose(Y.position,Y.quaternion,Y.scale),Y.updateMatrixWorld(!0),Y.projectionMatrix.copy(oe.projectionMatrix),Y.projectionMatrixInverse.copy(oe.projectionMatrixInverse),Y.isPerspectiveCamera&&(Y.fov=lf*2*Math.atan(1/Y.projectionMatrix.elements[5]),Y.zoom=1)}this.getCamera=function(){return P},this.getFoveation=function(){if(!(d===null&&h===null))return l},this.setFoveation=function(Y){l=Y,d!==null&&(d.fixedFoveation=Y),h!==null&&h.fixedFoveation!==void 0&&(h.fixedFoveation=Y)},this.hasDepthSensing=function(){return x.texture!==null},this.getDepthSensingMesh=function(){return x.getMesh(P)},this.getCameraTexture=function(Y){return m[Y]};let ge=null;function be(Y,oe){if(u=oe.getViewerPose(c||o),g=oe,u!==null){const he=u.views;h!==null&&(e.setRenderTargetFramebuffer(S,h.framebuffer),e.setRenderTarget(S));let Fe=!1;he.length!==P.cameras.length&&(P.cameras.length=0,Fe=!0);for(let B=0;B<he.length;B++){const q=he[B];let w=null;if(h!==null)w=h.getViewport(q);else{const re=f.getViewSubImage(d,q);w=re.viewport,B===0&&(e.setRenderTargetTextures(S,re.colorTexture,re.depthStencilTexture),e.setRenderTarget(S))}let ne=I[B];ne===void 0&&(ne=new ei,ne.layers.enable(B),ne.viewport=new Et,I[B]=ne),ne.matrix.fromArray(q.transform.matrix),ne.matrix.decompose(ne.position,ne.quaternion,ne.scale),ne.projectionMatrix.fromArray(q.projectionMatrix),ne.projectionMatrixInverse.copy(ne.projectionMatrix).invert(),ne.viewport.set(w.x,w.y,w.width,w.height),B===0&&(P.matrix.copy(ne.matrix),P.matrix.decompose(P.position,P.quaternion,P.scale)),Fe===!0&&P.cameras.push(ne)}const ye=s.enabledFeatures;if(ye&&ye.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&p){f=n.getBinding();const B=f.getDepthInformation(he[0]);B&&B.isValid&&B.texture&&x.init(B,s.renderState)}if(ye&&ye.includes("camera-access")&&p){e.state.unbindTexture(),f=n.getBinding();for(let B=0;B<he.length;B++){const q=he[B].camera;if(q){let w=m[q];w||(w=new q0,m[q]=w);const ne=f.getCameraImage(q);w.sourceTexture=ne}}}}for(let he=0;he<v.length;he++){const Fe=b[he],ye=v[he];Fe!==null&&ye!==void 0&&ye.update(Fe,oe,c||o)}ge&&ge(Y,oe),oe.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:oe}),g=null}const Me=new Q0;Me.setAnimationLoop(be),this.setAnimationLoop=function(Y){ge=Y},this.dispose=function(){}}}const $s=new _i,jC=new qe;function $C(i,e){function t(x,m){x.matrixAutoUpdate===!0&&x.updateMatrix(),m.value.copy(x.matrix)}function n(x,m){m.color.getRGB(x.fogColor.value,V0(i)),m.isFog?(x.fogNear.value=m.near,x.fogFar.value=m.far):m.isFogExp2&&(x.fogDensity.value=m.density)}function s(x,m,_,A,S){m.isMeshBasicMaterial||m.isMeshLambertMaterial?r(x,m):m.isMeshToonMaterial?(r(x,m),f(x,m)):m.isMeshPhongMaterial?(r(x,m),u(x,m)):m.isMeshStandardMaterial?(r(x,m),d(x,m),m.isMeshPhysicalMaterial&&h(x,m,S)):m.isMeshMatcapMaterial?(r(x,m),g(x,m)):m.isMeshDepthMaterial?r(x,m):m.isMeshDistanceMaterial?(r(x,m),p(x,m)):m.isMeshNormalMaterial?r(x,m):m.isLineBasicMaterial?(o(x,m),m.isLineDashedMaterial&&a(x,m)):m.isPointsMaterial?l(x,m,_,A):m.isSpriteMaterial?c(x,m):m.isShadowMaterial?(x.color.value.copy(m.color),x.opacity.value=m.opacity):m.isShaderMaterial&&(m.uniformsNeedUpdate=!1)}function r(x,m){x.opacity.value=m.opacity,m.color&&x.diffuse.value.copy(m.color),m.emissive&&x.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity),m.map&&(x.map.value=m.map,t(m.map,x.mapTransform)),m.alphaMap&&(x.alphaMap.value=m.alphaMap,t(m.alphaMap,x.alphaMapTransform)),m.bumpMap&&(x.bumpMap.value=m.bumpMap,t(m.bumpMap,x.bumpMapTransform),x.bumpScale.value=m.bumpScale,m.side===En&&(x.bumpScale.value*=-1)),m.normalMap&&(x.normalMap.value=m.normalMap,t(m.normalMap,x.normalMapTransform),x.normalScale.value.copy(m.normalScale),m.side===En&&x.normalScale.value.negate()),m.displacementMap&&(x.displacementMap.value=m.displacementMap,t(m.displacementMap,x.displacementMapTransform),x.displacementScale.value=m.displacementScale,x.displacementBias.value=m.displacementBias),m.emissiveMap&&(x.emissiveMap.value=m.emissiveMap,t(m.emissiveMap,x.emissiveMapTransform)),m.specularMap&&(x.specularMap.value=m.specularMap,t(m.specularMap,x.specularMapTransform)),m.alphaTest>0&&(x.alphaTest.value=m.alphaTest);const _=e.get(m),A=_.envMap,S=_.envMapRotation;A&&(x.envMap.value=A,$s.copy(S),$s.x*=-1,$s.y*=-1,$s.z*=-1,A.isCubeTexture&&A.isRenderTargetTexture===!1&&($s.y*=-1,$s.z*=-1),x.envMapRotation.value.setFromMatrix4(jC.makeRotationFromEuler($s)),x.flipEnvMap.value=A.isCubeTexture&&A.isRenderTargetTexture===!1?-1:1,x.reflectivity.value=m.reflectivity,x.ior.value=m.ior,x.refractionRatio.value=m.refractionRatio),m.lightMap&&(x.lightMap.value=m.lightMap,x.lightMapIntensity.value=m.lightMapIntensity,t(m.lightMap,x.lightMapTransform)),m.aoMap&&(x.aoMap.value=m.aoMap,x.aoMapIntensity.value=m.aoMapIntensity,t(m.aoMap,x.aoMapTransform))}function o(x,m){x.diffuse.value.copy(m.color),x.opacity.value=m.opacity,m.map&&(x.map.value=m.map,t(m.map,x.mapTransform))}function a(x,m){x.dashSize.value=m.dashSize,x.totalSize.value=m.dashSize+m.gapSize,x.scale.value=m.scale}function l(x,m,_,A){x.diffuse.value.copy(m.color),x.opacity.value=m.opacity,x.size.value=m.size*_,x.scale.value=A*.5,m.map&&(x.map.value=m.map,t(m.map,x.uvTransform)),m.alphaMap&&(x.alphaMap.value=m.alphaMap,t(m.alphaMap,x.alphaMapTransform)),m.alphaTest>0&&(x.alphaTest.value=m.alphaTest)}function c(x,m){x.diffuse.value.copy(m.color),x.opacity.value=m.opacity,x.rotation.value=m.rotation,m.map&&(x.map.value=m.map,t(m.map,x.mapTransform)),m.alphaMap&&(x.alphaMap.value=m.alphaMap,t(m.alphaMap,x.alphaMapTransform)),m.alphaTest>0&&(x.alphaTest.value=m.alphaTest)}function u(x,m){x.specular.value.copy(m.specular),x.shininess.value=Math.max(m.shininess,1e-4)}function f(x,m){m.gradientMap&&(x.gradientMap.value=m.gradientMap)}function d(x,m){x.metalness.value=m.metalness,m.metalnessMap&&(x.metalnessMap.value=m.metalnessMap,t(m.metalnessMap,x.metalnessMapTransform)),x.roughness.value=m.roughness,m.roughnessMap&&(x.roughnessMap.value=m.roughnessMap,t(m.roughnessMap,x.roughnessMapTransform)),m.envMap&&(x.envMapIntensity.value=m.envMapIntensity)}function h(x,m,_){x.ior.value=m.ior,m.sheen>0&&(x.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen),x.sheenRoughness.value=m.sheenRoughness,m.sheenColorMap&&(x.sheenColorMap.value=m.sheenColorMap,t(m.sheenColorMap,x.sheenColorMapTransform)),m.sheenRoughnessMap&&(x.sheenRoughnessMap.value=m.sheenRoughnessMap,t(m.sheenRoughnessMap,x.sheenRoughnessMapTransform))),m.clearcoat>0&&(x.clearcoat.value=m.clearcoat,x.clearcoatRoughness.value=m.clearcoatRoughness,m.clearcoatMap&&(x.clearcoatMap.value=m.clearcoatMap,t(m.clearcoatMap,x.clearcoatMapTransform)),m.clearcoatRoughnessMap&&(x.clearcoatRoughnessMap.value=m.clearcoatRoughnessMap,t(m.clearcoatRoughnessMap,x.clearcoatRoughnessMapTransform)),m.clearcoatNormalMap&&(x.clearcoatNormalMap.value=m.clearcoatNormalMap,t(m.clearcoatNormalMap,x.clearcoatNormalMapTransform),x.clearcoatNormalScale.value.copy(m.clearcoatNormalScale),m.side===En&&x.clearcoatNormalScale.value.negate())),m.dispersion>0&&(x.dispersion.value=m.dispersion),m.iridescence>0&&(x.iridescence.value=m.iridescence,x.iridescenceIOR.value=m.iridescenceIOR,x.iridescenceThicknessMinimum.value=m.iridescenceThicknessRange[0],x.iridescenceThicknessMaximum.value=m.iridescenceThicknessRange[1],m.iridescenceMap&&(x.iridescenceMap.value=m.iridescenceMap,t(m.iridescenceMap,x.iridescenceMapTransform)),m.iridescenceThicknessMap&&(x.iridescenceThicknessMap.value=m.iridescenceThicknessMap,t(m.iridescenceThicknessMap,x.iridescenceThicknessMapTransform))),m.transmission>0&&(x.transmission.value=m.transmission,x.transmissionSamplerMap.value=_.texture,x.transmissionSamplerSize.value.set(_.width,_.height),m.transmissionMap&&(x.transmissionMap.value=m.transmissionMap,t(m.transmissionMap,x.transmissionMapTransform)),x.thickness.value=m.thickness,m.thicknessMap&&(x.thicknessMap.value=m.thicknessMap,t(m.thicknessMap,x.thicknessMapTransform)),x.attenuationDistance.value=m.attenuationDistance,x.attenuationColor.value.copy(m.attenuationColor)),m.anisotropy>0&&(x.anisotropyVector.value.set(m.anisotropy*Math.cos(m.anisotropyRotation),m.anisotropy*Math.sin(m.anisotropyRotation)),m.anisotropyMap&&(x.anisotropyMap.value=m.anisotropyMap,t(m.anisotropyMap,x.anisotropyMapTransform))),x.specularIntensity.value=m.specularIntensity,x.specularColor.value.copy(m.specularColor),m.specularColorMap&&(x.specularColorMap.value=m.specularColorMap,t(m.specularColorMap,x.specularColorMapTransform)),m.specularIntensityMap&&(x.specularIntensityMap.value=m.specularIntensityMap,t(m.specularIntensityMap,x.specularIntensityMapTransform))}function g(x,m){m.matcap&&(x.matcap.value=m.matcap)}function p(x,m){const _=e.get(m).light;x.referencePosition.value.setFromMatrixPosition(_.matrixWorld),x.nearDistance.value=_.shadow.camera.near,x.farDistance.value=_.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:s}}function ZC(i,e,t,n){let s={},r={},o=[];const a=i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);function l(_,A){const S=A.program;n.uniformBlockBinding(_,S)}function c(_,A){let S=s[_.id];S===void 0&&(g(_),S=u(_),s[_.id]=S,_.addEventListener("dispose",x));const v=A.program;n.updateUBOMapping(_,v);const b=e.render.frame;r[_.id]!==b&&(d(_),r[_.id]=b)}function u(_){const A=f();_.__bindingPointIndex=A;const S=i.createBuffer(),v=_.__size,b=_.usage;return i.bindBuffer(i.UNIFORM_BUFFER,S),i.bufferData(i.UNIFORM_BUFFER,v,b),i.bindBuffer(i.UNIFORM_BUFFER,null),i.bindBufferBase(i.UNIFORM_BUFFER,A,S),S}function f(){for(let _=0;_<a;_++)if(o.indexOf(_)===-1)return o.push(_),_;return zt("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function d(_){const A=s[_.id],S=_.uniforms,v=_.__cache;i.bindBuffer(i.UNIFORM_BUFFER,A);for(let b=0,M=S.length;b<M;b++){const E=Array.isArray(S[b])?S[b]:[S[b]];for(let y=0,C=E.length;y<C;y++){const I=E[y];if(h(I,b,y,v)===!0){const P=I.__offset,U=Array.isArray(I.value)?I.value:[I.value];let O=0;for(let k=0;k<U.length;k++){const z=U[k],W=p(z);typeof z=="number"||typeof z=="boolean"?(I.__data[0]=z,i.bufferSubData(i.UNIFORM_BUFFER,P+O,I.__data)):z.isMatrix3?(I.__data[0]=z.elements[0],I.__data[1]=z.elements[1],I.__data[2]=z.elements[2],I.__data[3]=0,I.__data[4]=z.elements[3],I.__data[5]=z.elements[4],I.__data[6]=z.elements[5],I.__data[7]=0,I.__data[8]=z.elements[6],I.__data[9]=z.elements[7],I.__data[10]=z.elements[8],I.__data[11]=0):(z.toArray(I.__data,O),O+=W.storage/Float32Array.BYTES_PER_ELEMENT)}i.bufferSubData(i.UNIFORM_BUFFER,P,I.__data)}}}i.bindBuffer(i.UNIFORM_BUFFER,null)}function h(_,A,S,v){const b=_.value,M=A+"_"+S;if(v[M]===void 0)return typeof b=="number"||typeof b=="boolean"?v[M]=b:v[M]=b.clone(),!0;{const E=v[M];if(typeof b=="number"||typeof b=="boolean"){if(E!==b)return v[M]=b,!0}else if(E.equals(b)===!1)return E.copy(b),!0}return!1}function g(_){const A=_.uniforms;let S=0;const v=16;for(let M=0,E=A.length;M<E;M++){const y=Array.isArray(A[M])?A[M]:[A[M]];for(let C=0,I=y.length;C<I;C++){const P=y[C],U=Array.isArray(P.value)?P.value:[P.value];for(let O=0,k=U.length;O<k;O++){const z=U[O],W=p(z),H=S%v,$=H%W.boundary,X=H+$;S+=$,X!==0&&v-X<W.storage&&(S+=v-X),P.__data=new Float32Array(W.storage/Float32Array.BYTES_PER_ELEMENT),P.__offset=S,S+=W.storage}}}const b=S%v;return b>0&&(S+=v-b),_.__size=S,_.__cache={},this}function p(_){const A={boundary:0,storage:0};return typeof _=="number"||typeof _=="boolean"?(A.boundary=4,A.storage=4):_.isVector2?(A.boundary=8,A.storage=8):_.isVector3||_.isColor?(A.boundary=16,A.storage=12):_.isVector4?(A.boundary=16,A.storage=16):_.isMatrix3?(A.boundary=48,A.storage=48):_.isMatrix4?(A.boundary=64,A.storage=64):_.isTexture?je("WebGLRenderer: Texture samplers can not be part of an uniforms group."):je("WebGLRenderer: Unsupported uniform value type.",_),A}function x(_){const A=_.target;A.removeEventListener("dispose",x);const S=o.indexOf(A.__bindingPointIndex);o.splice(S,1),i.deleteBuffer(s[A.id]),delete s[A.id],delete r[A.id]}function m(){for(const _ in s)i.deleteBuffer(s[_]);o=[],s={},r={}}return{bind:l,update:c,dispose:m}}const JC=new Uint16Array([11481,15204,11534,15171,11808,15015,12385,14843,12894,14716,13396,14600,13693,14483,13976,14366,14237,14171,14405,13961,14511,13770,14605,13598,14687,13444,14760,13305,14822,13066,14876,12857,14923,12675,14963,12517,14997,12379,15025,12230,15049,12023,15070,11843,15086,11687,15100,11551,15111,11433,15120,11330,15127,11217,15132,11060,15135,10922,15138,10801,15139,10695,15139,10600,13012,14923,13020,14917,13064,14886,13176,14800,13349,14666,13513,14526,13724,14398,13960,14230,14200,14020,14383,13827,14488,13651,14583,13491,14667,13348,14740,13132,14803,12908,14856,12713,14901,12542,14938,12394,14968,12241,14992,12017,15010,11822,15024,11654,15034,11507,15041,11380,15044,11269,15044,11081,15042,10913,15037,10764,15031,10635,15023,10520,15014,10419,15003,10330,13657,14676,13658,14673,13670,14660,13698,14622,13750,14547,13834,14442,13956,14317,14112,14093,14291,13889,14407,13704,14499,13538,14586,13389,14664,13201,14733,12966,14792,12758,14842,12577,14882,12418,14915,12272,14940,12033,14959,11826,14972,11646,14980,11490,14983,11355,14983,11212,14979,11008,14971,10830,14961,10675,14950,10540,14936,10420,14923,10315,14909,10204,14894,10041,14089,14460,14090,14459,14096,14452,14112,14431,14141,14388,14186,14305,14252,14130,14341,13941,14399,13756,14467,13585,14539,13430,14610,13272,14677,13026,14737,12808,14790,12617,14833,12449,14869,12303,14896,12065,14916,11845,14929,11655,14937,11490,14939,11347,14936,11184,14930,10970,14921,10783,14912,10621,14900,10480,14885,10356,14867,10247,14848,10062,14827,9894,14805,9745,14400,14208,14400,14206,14402,14198,14406,14174,14415,14122,14427,14035,14444,13913,14469,13767,14504,13613,14548,13463,14598,13324,14651,13082,14704,12858,14752,12658,14795,12483,14831,12330,14860,12106,14881,11875,14895,11675,14903,11501,14905,11351,14903,11178,14900,10953,14892,10757,14880,10589,14865,10442,14847,10313,14827,10162,14805,9965,14782,9792,14757,9642,14731,9507,14562,13883,14562,13883,14563,13877,14566,13862,14570,13830,14576,13773,14584,13689,14595,13582,14613,13461,14637,13336,14668,13120,14704,12897,14741,12695,14776,12516,14808,12358,14835,12150,14856,11910,14870,11701,14878,11519,14882,11361,14884,11187,14880,10951,14871,10748,14858,10572,14842,10418,14823,10286,14801,10099,14777,9897,14751,9722,14725,9567,14696,9430,14666,9309,14702,13604,14702,13604,14702,13600,14703,13591,14705,13570,14707,13533,14709,13477,14712,13400,14718,13305,14727,13106,14743,12907,14762,12716,14784,12539,14807,12380,14827,12190,14844,11943,14855,11727,14863,11539,14870,11376,14871,11204,14868,10960,14858,10748,14845,10565,14829,10406,14809,10269,14786,10058,14761,9852,14734,9671,14705,9512,14674,9374,14641,9253,14608,9076,14821,13366,14821,13365,14821,13364,14821,13358,14821,13344,14821,13320,14819,13252,14817,13145,14815,13011,14814,12858,14817,12698,14823,12539,14832,12389,14841,12214,14850,11968,14856,11750,14861,11558,14866,11390,14867,11226,14862,10972,14853,10754,14840,10565,14823,10401,14803,10259,14780,10032,14754,9820,14725,9635,14694,9473,14661,9333,14627,9203,14593,8988,14557,8798,14923,13014,14922,13014,14922,13012,14922,13004,14920,12987,14919,12957,14915,12907,14909,12834,14902,12738,14894,12623,14888,12498,14883,12370,14880,12203,14878,11970,14875,11759,14873,11569,14874,11401,14872,11243,14865,10986,14855,10762,14842,10568,14825,10401,14804,10255,14781,10017,14754,9799,14725,9611,14692,9445,14658,9301,14623,9139,14587,8920,14548,8729,14509,8562,15008,12672,15008,12672,15008,12671,15007,12667,15005,12656,15001,12637,14997,12605,14989,12556,14978,12490,14966,12407,14953,12313,14940,12136,14927,11934,14914,11742,14903,11563,14896,11401,14889,11247,14879,10992,14866,10767,14851,10570,14833,10400,14812,10252,14789,10007,14761,9784,14731,9592,14698,9424,14663,9279,14627,9088,14588,8868,14548,8676,14508,8508,14467,8360,15080,12386,15080,12386,15079,12385,15078,12383,15076,12378,15072,12367,15066,12347,15057,12315,15045,12253,15030,12138,15012,11998,14993,11845,14972,11685,14951,11530,14935,11383,14920,11228,14904,10981,14887,10762,14870,10567,14850,10397,14827,10248,14803,9997,14774,9771,14743,9578,14710,9407,14674,9259,14637,9048,14596,8826,14555,8632,14514,8464,14471,8317,14427,8182,15139,12008,15139,12008,15138,12008,15137,12007,15135,12003,15130,11990,15124,11969,15115,11929,15102,11872,15086,11794,15064,11693,15041,11581,15013,11459,14987,11336,14966,11170,14944,10944,14921,10738,14898,10552,14875,10387,14850,10239,14824,9983,14794,9758,14762,9563,14728,9392,14692,9244,14653,9014,14611,8791,14569,8597,14526,8427,14481,8281,14436,8110,14391,7885,15188,11617,15188,11617,15187,11617,15186,11618,15183,11617,15179,11612,15173,11601,15163,11581,15150,11546,15133,11495,15110,11427,15083,11346,15051,11246,15024,11057,14996,10868,14967,10687,14938,10517,14911,10362,14882,10206,14853,9956,14821,9737,14787,9543,14752,9375,14715,9228,14675,8980,14632,8760,14589,8565,14544,8395,14498,8248,14451,8049,14404,7824,14357,7630,15228,11298,15228,11298,15227,11299,15226,11301,15223,11303,15219,11302,15213,11299,15204,11290,15191,11271,15174,11217,15150,11129,15119,11015,15087,10886,15057,10744,15024,10599,14990,10455,14957,10318,14924,10143,14891,9911,14856,9701,14820,9516,14782,9352,14744,9200,14703,8946,14659,8725,14615,8533,14568,8366,14521,8220,14472,7992,14423,7770,14374,7578,14315,7408,15260,10819,15260,10819,15259,10822,15258,10826,15256,10832,15251,10836,15246,10841,15237,10838,15225,10821,15207,10788,15183,10734,15151,10660,15120,10571,15087,10469,15049,10359,15012,10249,14974,10041,14937,9837,14900,9647,14860,9475,14820,9320,14779,9147,14736,8902,14691,8688,14646,8499,14598,8335,14549,8189,14499,7940,14448,7720,14397,7529,14347,7363,14256,7218,15285,10410,15285,10411,15285,10413,15284,10418,15282,10425,15278,10434,15272,10442,15264,10449,15252,10445,15235,10433,15210,10403,15179,10358,15149,10301,15113,10218,15073,10059,15033,9894,14991,9726,14951,9565,14909,9413,14865,9273,14822,9073,14777,8845,14730,8641,14682,8459,14633,8300,14583,8129,14531,7883,14479,7670,14426,7482,14373,7321,14305,7176,14201,6939,15305,9939,15305,9940,15305,9945,15304,9955,15302,9967,15298,9989,15293,10010,15286,10033,15274,10044,15258,10045,15233,10022,15205,9975,15174,9903,15136,9808,15095,9697,15053,9578,15009,9451,14965,9327,14918,9198,14871,8973,14825,8766,14775,8579,14725,8408,14675,8259,14622,8058,14569,7821,14515,7615,14460,7435,14405,7276,14350,7108,14256,6866,14149,6653,15321,9444,15321,9445,15321,9448,15320,9458,15317,9470,15314,9490,15310,9515,15302,9540,15292,9562,15276,9579,15251,9577,15226,9559,15195,9519,15156,9463,15116,9389,15071,9304,15025,9208,14978,9023,14927,8838,14878,8661,14827,8496,14774,8344,14722,8206,14667,7973,14612,7749,14556,7555,14499,7382,14443,7229,14385,7025,14322,6791,14210,6588,14100,6409,15333,8920,15333,8921,15332,8927,15332,8943,15329,8965,15326,9002,15322,9048,15316,9106,15307,9162,15291,9204,15267,9221,15244,9221,15212,9196,15175,9134,15133,9043,15088,8930,15040,8801,14990,8665,14938,8526,14886,8391,14830,8261,14775,8087,14719,7866,14661,7664,14603,7482,14544,7322,14485,7178,14426,6936,14367,6713,14281,6517,14166,6348,14054,6198,15341,8360,15341,8361,15341,8366,15341,8379,15339,8399,15336,8431,15332,8473,15326,8527,15318,8585,15302,8632,15281,8670,15258,8690,15227,8690,15191,8664,15149,8612,15104,8543,15055,8456,15001,8360,14948,8259,14892,8122,14834,7923,14776,7734,14716,7558,14656,7397,14595,7250,14534,7070,14472,6835,14410,6628,14350,6443,14243,6283,14125,6135,14010,5889,15348,7715,15348,7717,15348,7725,15347,7745,15345,7780,15343,7836,15339,7905,15334,8e3,15326,8103,15310,8193,15293,8239,15270,8270,15240,8287,15204,8283,15163,8260,15118,8223,15067,8143,15014,8014,14958,7873,14899,7723,14839,7573,14778,7430,14715,7293,14652,7164,14588,6931,14524,6720,14460,6531,14396,6362,14330,6210,14207,6015,14086,5781,13969,5576,15352,7114,15352,7116,15352,7128,15352,7159,15350,7195,15348,7237,15345,7299,15340,7374,15332,7457,15317,7544,15301,7633,15280,7703,15251,7754,15216,7775,15176,7767,15131,7733,15079,7670,15026,7588,14967,7492,14906,7387,14844,7278,14779,7171,14714,6965,14648,6770,14581,6587,14515,6420,14448,6269,14382,6123,14299,5881,14172,5665,14049,5477,13929,5310,15355,6329,15355,6330,15355,6339,15355,6362,15353,6410,15351,6472,15349,6572,15344,6688,15337,6835,15323,6985,15309,7142,15287,7220,15260,7277,15226,7310,15188,7326,15142,7318,15090,7285,15036,7239,14976,7177,14914,7045,14849,6892,14782,6736,14714,6581,14645,6433,14576,6293,14506,6164,14438,5946,14369,5733,14270,5540,14140,5369,14014,5216,13892,5043,15357,5483,15357,5484,15357,5496,15357,5528,15356,5597,15354,5692,15351,5835,15347,6011,15339,6195,15328,6317,15314,6446,15293,6566,15268,6668,15235,6746,15197,6796,15152,6811,15101,6790,15046,6748,14985,6673,14921,6583,14854,6479,14785,6371,14714,6259,14643,6149,14571,5946,14499,5750,14428,5567,14358,5401,14242,5250,14109,5111,13980,4870,13856,4657,15359,4555,15359,4557,15358,4573,15358,4633,15357,4715,15355,4841,15353,5061,15349,5216,15342,5391,15331,5577,15318,5770,15299,5967,15274,6150,15243,6223,15206,6280,15161,6310,15111,6317,15055,6300,14994,6262,14928,6208,14860,6141,14788,5994,14715,5838,14641,5684,14566,5529,14492,5384,14418,5247,14346,5121,14216,4892,14079,4682,13948,4496,13822,4330,15359,3498,15359,3501,15359,3520,15359,3598,15358,3719,15356,3860,15355,4137,15351,4305,15344,4563,15334,4809,15321,5116,15303,5273,15280,5418,15250,5547,15214,5653,15170,5722,15120,5761,15064,5763,15002,5733,14935,5673,14865,5597,14792,5504,14716,5400,14640,5294,14563,5185,14486,5041,14410,4841,14335,4655,14191,4482,14051,4325,13918,4183,13790,4012,15360,2282,15360,2285,15360,2306,15360,2401,15359,2547,15357,2748,15355,3103,15352,3349,15345,3675,15336,4020,15324,4272,15307,4496,15285,4716,15255,4908,15220,5086,15178,5170,15128,5214,15072,5234,15010,5231,14943,5206,14871,5166,14796,5102,14718,4971,14639,4833,14559,4687,14480,4541,14402,4401,14315,4268,14167,4142,14025,3958,13888,3747,13759,3556,15360,923,15360,925,15360,946,15360,1052,15359,1214,15357,1494,15356,1892,15352,2274,15346,2663,15338,3099,15326,3393,15309,3679,15288,3980,15260,4183,15226,4325,15185,4437,15136,4517,15080,4570,15018,4591,14950,4581,14877,4545,14800,4485,14720,4411,14638,4325,14556,4231,14475,4136,14395,3988,14297,3803,14145,3628,13999,3465,13861,3314,13729,3177,15360,263,15360,264,15360,272,15360,325,15359,407,15358,548,15356,780,15352,1144,15347,1580,15339,2099,15328,2425,15312,2795,15292,3133,15264,3329,15232,3517,15191,3689,15143,3819,15088,3923,15025,3978,14956,3999,14882,3979,14804,3931,14722,3855,14639,3756,14554,3645,14470,3529,14388,3409,14279,3289,14124,3173,13975,3055,13834,2848,13701,2658,15360,49,15360,49,15360,52,15360,75,15359,111,15358,201,15356,283,15353,519,15348,726,15340,1045,15329,1415,15314,1795,15295,2173,15269,2410,15237,2649,15197,2866,15150,3054,15095,3140,15032,3196,14963,3228,14888,3236,14808,3224,14725,3191,14639,3146,14553,3088,14466,2976,14382,2836,14262,2692,14103,2549,13952,2409,13808,2278,13674,2154,15360,4,15360,4,15360,4,15360,13,15359,33,15358,59,15357,112,15353,199,15348,302,15341,456,15331,628,15316,827,15297,1082,15272,1332,15241,1601,15202,1851,15156,2069,15101,2172,15039,2256,14970,2314,14894,2348,14813,2358,14728,2344,14640,2311,14551,2263,14463,2203,14376,2133,14247,2059,14084,1915,13930,1761,13784,1609,13648,1464,15360,0,15360,0,15360,0,15360,3,15359,18,15358,26,15357,53,15354,80,15348,97,15341,165,15332,238,15318,326,15299,427,15275,529,15245,654,15207,771,15161,885,15108,994,15046,1089,14976,1170,14900,1229,14817,1266,14731,1284,14641,1282,14550,1260,14460,1223,14370,1174,14232,1116,14066,1050,13909,981,13761,910,13623,839]);let Wi=null;function eT(){return Wi===null&&(Wi=new Qi(JC,32,32,$f,pr),Wi.minFilter=ii,Wi.magFilter=ii,Wi.wrapS=ns,Wi.wrapT=ns,Wi.generateMipmaps=!1,Wi.needsUpdate=!0),Wi}class tT{constructor(e={}){const{canvas:t=RS(),context:n=null,depth:s=!0,stencil:r=!1,alpha:o=!1,antialias:a=!1,premultipliedAlpha:l=!0,preserveDrawingBuffer:c=!1,powerPreference:u="default",failIfMajorPerformanceCaveat:f=!1,reversedDepthBuffer:d=!1}=e;this.isWebGLRenderer=!0;let h;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");h=n.getContextAttributes().alpha}else h=o;const g=new Set([jr,Zf,jl]),p=new Set([Ui,si,ra,oa,Kf,jf]),x=new Uint32Array(4),m=new Int32Array(4);let _=null,A=null;const S=[],v=[];this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=Ds,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const b=this;let M=!1;this._outputColorSpace=Jn;let E=0,y=0,C=null,I=-1,P=null;const U=new Et,O=new Et;let k=null;const z=new nt(0);let W=0,H=t.width,$=t.height,X=1,Z=null,pe=null;const ge=new Et(0,0,H,$),be=new Et(0,0,H,$);let Me=!1;const Y=new X0;let oe=!1,he=!1;const Fe=new qe,ye=new L,ke=new Et,F={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let B=!1;function q(){return C===null?X:1}let w=n;function ne(D,Q){return t.getContext(D,Q)}try{const D={alpha:!0,depth:s,stencil:r,antialias:a,premultipliedAlpha:l,preserveDrawingBuffer:c,powerPreference:u,failIfMajorPerformanceCaveat:f};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${Qf}`),t.addEventListener("webglcontextlost",_e,!1),t.addEventListener("webglcontextrestored",de,!1),t.addEventListener("webglcontextcreationerror",Be,!1),w===null){const Q="webgl2";if(w=ne(Q,D),w===null)throw ne(Q)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(D){throw D("WebGLRenderer: "+D.message),D}let re,ue,ie,fe,te,xe,R,T,V,se,ce,J,Pe,Ae,Ue,N,ee,me,Te,Ie,we,He,G,Re;function Se(){re=new uM(w),re.init(),He=new Z0(w,re),ue=new tM(w,re,e,He),ie=new WC(w,re),ue.reversedDepthBuffer&&d&&ie.buffers.depth.setReversed(!0),fe=new hM(w),te=new DC,xe=new XC(w,re,ie,te,ue,He,fe),R=new iM(b),T=new cM(b),V=new xv(w),G=new Jb(w,V),se=new fM(w,V,fe,G),ce=new mM(w,se,V,fe),Te=new pM(w,ue,xe),N=new nM(te),J=new IC(b,R,T,re,ue,G,N),Pe=new $C(b,te),Ae=new FC,Ue=new zC(re),me=new Zb(b,R,T,ie,ce,h,l),ee=new VC(b,ce,ue),Re=new ZC(w,fe,ue,ie),Ie=new eM(w,re,fe),we=new dM(w,re,fe),fe.programs=J.programs,b.capabilities=ue,b.extensions=re,b.properties=te,b.renderLists=Ae,b.shadowMap=ee,b.state=ie,b.info=fe}Se();const ve=new KC(b,w);this.xr=ve,this.getContext=function(){return w},this.getContextAttributes=function(){return w.getContextAttributes()},this.forceContextLoss=function(){const D=re.get("WEBGL_lose_context");D&&D.loseContext()},this.forceContextRestore=function(){const D=re.get("WEBGL_lose_context");D&&D.restoreContext()},this.getPixelRatio=function(){return X},this.setPixelRatio=function(D){D!==void 0&&(X=D,this.setSize(H,$,!1))},this.getSize=function(D){return D.set(H,$)},this.setSize=function(D,Q,ae=!0){if(ve.isPresenting){je("WebGLRenderer: Can't change size while VR device is presenting.");return}H=D,$=Q,t.width=Math.floor(D*X),t.height=Math.floor(Q*X),ae===!0&&(t.style.width=D+"px",t.style.height=Q+"px"),this.setViewport(0,0,D,Q)},this.getDrawingBufferSize=function(D){return D.set(H*X,$*X).floor()},this.setDrawingBufferSize=function(D,Q,ae){H=D,$=Q,X=ae,t.width=Math.floor(D*ae),t.height=Math.floor(Q*ae),this.setViewport(0,0,D,Q)},this.getCurrentViewport=function(D){return D.copy(U)},this.getViewport=function(D){return D.copy(ge)},this.setViewport=function(D,Q,ae,le){D.isVector4?ge.set(D.x,D.y,D.z,D.w):ge.set(D,Q,ae,le),ie.viewport(U.copy(ge).multiplyScalar(X).round())},this.getScissor=function(D){return D.copy(be)},this.setScissor=function(D,Q,ae,le){D.isVector4?be.set(D.x,D.y,D.z,D.w):be.set(D,Q,ae,le),ie.scissor(O.copy(be).multiplyScalar(X).round())},this.getScissorTest=function(){return Me},this.setScissorTest=function(D){ie.setScissorTest(Me=D)},this.setOpaqueSort=function(D){Z=D},this.setTransparentSort=function(D){pe=D},this.getClearColor=function(D){return D.copy(me.getClearColor())},this.setClearColor=function(){me.setClearColor(...arguments)},this.getClearAlpha=function(){return me.getClearAlpha()},this.setClearAlpha=function(){me.setClearAlpha(...arguments)},this.clear=function(D=!0,Q=!0,ae=!0){let le=0;if(D){let K=!1;if(C!==null){const Ce=C.texture.format;K=g.has(Ce)}if(K){const Ce=C.texture.type,Le=p.has(Ce),ze=me.getClearColor(),Oe=me.getClearAlpha(),Xe=ze.r,Ye=ze.g,Ve=ze.b;Le?(x[0]=Xe,x[1]=Ye,x[2]=Ve,x[3]=Oe,w.clearBufferuiv(w.COLOR,0,x)):(m[0]=Xe,m[1]=Ye,m[2]=Ve,m[3]=Oe,w.clearBufferiv(w.COLOR,0,m))}else le|=w.COLOR_BUFFER_BIT}Q&&(le|=w.DEPTH_BUFFER_BIT),ae&&(le|=w.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),w.clear(le)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){t.removeEventListener("webglcontextlost",_e,!1),t.removeEventListener("webglcontextrestored",de,!1),t.removeEventListener("webglcontextcreationerror",Be,!1),me.dispose(),Ae.dispose(),Ue.dispose(),te.dispose(),R.dispose(),T.dispose(),ce.dispose(),G.dispose(),Re.dispose(),J.dispose(),ve.dispose(),ve.removeEventListener("sessionstart",zd),ve.removeEventListener("sessionend",kd),Vs.stop()};function _e(D){D.preventDefault(),Fh("WebGLRenderer: Context Lost."),M=!0}function de(){Fh("WebGLRenderer: Context Restored."),M=!1;const D=fe.autoReset,Q=ee.enabled,ae=ee.autoUpdate,le=ee.needsUpdate,K=ee.type;Se(),fe.autoReset=D,ee.enabled=Q,ee.autoUpdate=ae,ee.needsUpdate=le,ee.type=K}function Be(D){zt("WebGLRenderer: A WebGL context could not be created. Reason: ",D.statusMessage)}function We(D){const Q=D.target;Q.removeEventListener("dispose",We),vt(Q)}function vt(D){ut(D),te.remove(D)}function ut(D){const Q=te.get(D).programs;Q!==void 0&&(Q.forEach(function(ae){J.releaseProgram(ae)}),D.isShaderMaterial&&J.releaseShaderCache(D))}this.renderBufferDirect=function(D,Q,ae,le,K,Ce){Q===null&&(Q=F);const Le=K.isMesh&&K.matrixWorld.determinant()<0,ze=ax(D,Q,ae,le,K);ie.setMaterial(le,Le);let Oe=ae.index,Xe=1;if(le.wireframe===!0){if(Oe=se.getWireframeAttribute(ae),Oe===void 0)return;Xe=2}const Ye=ae.drawRange,Ve=ae.attributes.position;let et=Ye.start*Xe,ft=(Ye.start+Ye.count)*Xe;Ce!==null&&(et=Math.max(et,Ce.start*Xe),ft=Math.min(ft,(Ce.start+Ce.count)*Xe)),Oe!==null?(et=Math.max(et,0),ft=Math.min(ft,Oe.count)):Ve!=null&&(et=Math.max(et,0),ft=Math.min(ft,Ve.count));const Lt=ft-et;if(Lt<0||Lt===1/0)return;G.setup(K,le,ze,ae,Oe);let Bt,gt=Ie;if(Oe!==null&&(Bt=V.get(Oe),gt=we,gt.setIndex(Bt)),K.isMesh)le.wireframe===!0?(ie.setLineWidth(le.wireframeLinewidth*q()),gt.setMode(w.LINES)):gt.setMode(w.TRIANGLES);else if(K.isLine){let Ge=le.linewidth;Ge===void 0&&(Ge=1),ie.setLineWidth(Ge*q()),K.isLineSegments?gt.setMode(w.LINES):K.isLineLoop?gt.setMode(w.LINE_LOOP):gt.setMode(w.LINE_STRIP)}else K.isPoints?gt.setMode(w.POINTS):K.isSprite&&gt.setMode(w.TRIANGLES);if(K.isBatchedMesh)if(K._multiDrawInstances!==null)la("WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."),gt.renderMultiDrawInstances(K._multiDrawStarts,K._multiDrawCounts,K._multiDrawCount,K._multiDrawInstances);else if(re.get("WEBGL_multi_draw"))gt.renderMultiDraw(K._multiDrawStarts,K._multiDrawCounts,K._multiDrawCount);else{const Ge=K._multiDrawStarts,wt=K._multiDrawCounts,it=K._multiDrawCount,Fn=Oe?V.get(Oe).bytesPerElement:1,xr=te.get(le).currentProgram.getUniforms();for(let Ln=0;Ln<it;Ln++)xr.setValue(w,"_gl_DrawID",Ln),gt.render(Ge[Ln]/Fn,wt[Ln])}else if(K.isInstancedMesh)gt.renderInstances(et,Lt,K.count);else if(ae.isInstancedBufferGeometry){const Ge=ae._maxInstanceCount!==void 0?ae._maxInstanceCount:1/0,wt=Math.min(ae.instanceCount,Ge);gt.renderInstances(et,Lt,wt)}else gt.render(et,Lt)};function Ai(D,Q,ae){D.transparent===!0&&D.side===ti&&D.forceSinglePass===!1?(D.side=En,D.needsUpdate=!0,Ta(D,Q,ae),D.side=Bi,D.needsUpdate=!0,Ta(D,Q,ae),D.side=ti):Ta(D,Q,ae)}this.compile=function(D,Q,ae=null){ae===null&&(ae=D),A=Ue.get(ae),A.init(Q),v.push(A),ae.traverseVisible(function(K){K.isLight&&K.layers.test(Q.layers)&&(A.pushLight(K),K.castShadow&&A.pushShadow(K))}),D!==ae&&D.traverseVisible(function(K){K.isLight&&K.layers.test(Q.layers)&&(A.pushLight(K),K.castShadow&&A.pushShadow(K))}),A.setupLights();const le=new Set;return D.traverse(function(K){if(!(K.isMesh||K.isPoints||K.isLine||K.isSprite))return;const Ce=K.material;if(Ce)if(Array.isArray(Ce))for(let Le=0;Le<Ce.length;Le++){const ze=Ce[Le];Ai(ze,ae,K),le.add(ze)}else Ai(Ce,ae,K),le.add(Ce)}),A=v.pop(),le},this.compileAsync=function(D,Q,ae=null){const le=this.compile(D,Q,ae);return new Promise(K=>{function Ce(){if(le.forEach(function(Le){te.get(Le).currentProgram.isReady()&&le.delete(Le)}),le.size===0){K(D);return}setTimeout(Ce,10)}re.get("KHR_parallel_shader_compile")!==null?Ce():setTimeout(Ce,10)})};let ci=null;function ox(D){ci&&ci(D)}function zd(){Vs.stop()}function kd(){Vs.start()}const Vs=new Q0;Vs.setAnimationLoop(ox),typeof self<"u"&&Vs.setContext(self),this.setAnimationLoop=function(D){ci=D,ve.setAnimationLoop(D),D===null?Vs.stop():Vs.start()},ve.addEventListener("sessionstart",zd),ve.addEventListener("sessionend",kd),this.render=function(D,Q){if(Q!==void 0&&Q.isCamera!==!0){zt("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(M===!0)return;if(D.matrixWorldAutoUpdate===!0&&D.updateMatrixWorld(),Q.parent===null&&Q.matrixWorldAutoUpdate===!0&&Q.updateMatrixWorld(),ve.enabled===!0&&ve.isPresenting===!0&&(ve.cameraAutoUpdate===!0&&ve.updateCamera(Q),Q=ve.getCamera()),D.isScene===!0&&D.onBeforeRender(b,D,Q,C),A=Ue.get(D,v.length),A.init(Q),v.push(A),Fe.multiplyMatrices(Q.projectionMatrix,Q.matrixWorldInverse),Y.setFromProjectionMatrix(Fe,Ei,Q.reversedDepth),he=this.localClippingEnabled,oe=N.init(this.clippingPlanes,he),_=Ae.get(D,S.length),_.init(),S.push(_),ve.enabled===!0&&ve.isPresenting===!0){const Ce=b.xr.getDepthSensingMesh();Ce!==null&&sc(Ce,Q,-1/0,b.sortObjects)}sc(D,Q,0,b.sortObjects),_.finish(),b.sortObjects===!0&&_.sort(Z,pe),B=ve.enabled===!1||ve.isPresenting===!1||ve.hasDepthSensing()===!1,B&&me.addToRenderList(_,D),this.info.render.frame++,oe===!0&&N.beginShadows();const ae=A.state.shadowsArray;ee.render(ae,D,Q),oe===!0&&N.endShadows(),this.info.autoReset===!0&&this.info.reset();const le=_.opaque,K=_.transmissive;if(A.setupLights(),Q.isArrayCamera){const Ce=Q.cameras;if(K.length>0)for(let Le=0,ze=Ce.length;Le<ze;Le++){const Oe=Ce[Le];Vd(le,K,D,Oe)}B&&me.render(D);for(let Le=0,ze=Ce.length;Le<ze;Le++){const Oe=Ce[Le];Hd(_,D,Oe,Oe.viewport)}}else K.length>0&&Vd(le,K,D,Q),B&&me.render(D),Hd(_,D,Q);C!==null&&y===0&&(xe.updateMultisampleRenderTarget(C),xe.updateRenderTargetMipmap(C)),D.isScene===!0&&D.onAfterRender(b,D,Q),G.resetDefaultState(),I=-1,P=null,v.pop(),v.length>0?(A=v[v.length-1],oe===!0&&N.setGlobalState(b.clippingPlanes,A.state.camera)):A=null,S.pop(),S.length>0?_=S[S.length-1]:_=null};function sc(D,Q,ae,le){if(D.visible===!1)return;if(D.layers.test(Q.layers)){if(D.isGroup)ae=D.renderOrder;else if(D.isLOD)D.autoUpdate===!0&&D.update(Q);else if(D.isLight)A.pushLight(D),D.castShadow&&A.pushShadow(D);else if(D.isSprite){if(!D.frustumCulled||Y.intersectsSprite(D)){le&&ke.setFromMatrixPosition(D.matrixWorld).applyMatrix4(Fe);const Le=ce.update(D),ze=D.material;ze.visible&&_.push(D,Le,ze,ae,ke.z,null)}}else if((D.isMesh||D.isLine||D.isPoints)&&(!D.frustumCulled||Y.intersectsObject(D))){const Le=ce.update(D),ze=D.material;if(le&&(D.boundingSphere!==void 0?(D.boundingSphere===null&&D.computeBoundingSphere(),ke.copy(D.boundingSphere.center)):(Le.boundingSphere===null&&Le.computeBoundingSphere(),ke.copy(Le.boundingSphere.center)),ke.applyMatrix4(D.matrixWorld).applyMatrix4(Fe)),Array.isArray(ze)){const Oe=Le.groups;for(let Xe=0,Ye=Oe.length;Xe<Ye;Xe++){const Ve=Oe[Xe],et=ze[Ve.materialIndex];et&&et.visible&&_.push(D,Le,et,ae,ke.z,Ve)}}else ze.visible&&_.push(D,Le,ze,ae,ke.z,null)}}const Ce=D.children;for(let Le=0,ze=Ce.length;Le<ze;Le++)sc(Ce[Le],Q,ae,le)}function Hd(D,Q,ae,le){const{opaque:K,transmissive:Ce,transparent:Le}=D;A.setupLightsView(ae),oe===!0&&N.setGlobalState(b.clippingPlanes,ae),le&&ie.viewport(U.copy(le)),K.length>0&&Ca(K,Q,ae),Ce.length>0&&Ca(Ce,Q,ae),Le.length>0&&Ca(Le,Q,ae),ie.buffers.depth.setTest(!0),ie.buffers.depth.setMask(!0),ie.buffers.color.setMask(!0),ie.setPolygonOffset(!1)}function Vd(D,Q,ae,le){if((ae.isScene===!0?ae.overrideMaterial:null)!==null)return;A.state.transmissionRenderTarget[le.id]===void 0&&(A.state.transmissionRenderTarget[le.id]=new Us(1,1,{generateMipmaps:!0,type:re.has("EXT_color_buffer_half_float")||re.has("EXT_color_buffer_float")?pr:Ui,minFilter:sr,samples:4,stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:rt.workingColorSpace}));const Ce=A.state.transmissionRenderTarget[le.id],Le=le.viewport||U;Ce.setSize(Le.z*b.transmissionResolutionScale,Le.w*b.transmissionResolutionScale);const ze=b.getRenderTarget(),Oe=b.getActiveCubeFace(),Xe=b.getActiveMipmapLevel();b.setRenderTarget(Ce),b.getClearColor(z),W=b.getClearAlpha(),W<1&&b.setClearColor(16777215,.5),b.clear(),B&&me.render(ae);const Ye=b.toneMapping;b.toneMapping=Ds;const Ve=le.viewport;if(le.viewport!==void 0&&(le.viewport=void 0),A.setupLightsView(le),oe===!0&&N.setGlobalState(b.clippingPlanes,le),Ca(D,ae,le),xe.updateMultisampleRenderTarget(Ce),xe.updateRenderTargetMipmap(Ce),re.has("WEBGL_multisampled_render_to_texture")===!1){let et=!1;for(let ft=0,Lt=Q.length;ft<Lt;ft++){const Bt=Q[ft],{object:gt,geometry:Ge,material:wt,group:it}=Bt;if(wt.side===ti&&gt.layers.test(le.layers)){const Fn=wt.side;wt.side=En,wt.needsUpdate=!0,Gd(gt,ae,le,Ge,wt,it),wt.side=Fn,wt.needsUpdate=!0,et=!0}}et===!0&&(xe.updateMultisampleRenderTarget(Ce),xe.updateRenderTargetMipmap(Ce))}b.setRenderTarget(ze,Oe,Xe),b.setClearColor(z,W),Ve!==void 0&&(le.viewport=Ve),b.toneMapping=Ye}function Ca(D,Q,ae){const le=Q.isScene===!0?Q.overrideMaterial:null;for(let K=0,Ce=D.length;K<Ce;K++){const Le=D[K],{object:ze,geometry:Oe,group:Xe}=Le;let Ye=Le.material;Ye.allowOverride===!0&&le!==null&&(Ye=le),ze.layers.test(ae.layers)&&Gd(ze,Q,ae,Oe,Ye,Xe)}}function Gd(D,Q,ae,le,K,Ce){D.onBeforeRender(b,Q,ae,le,K,Ce),D.modelViewMatrix.multiplyMatrices(ae.matrixWorldInverse,D.matrixWorld),D.normalMatrix.getNormalMatrix(D.modelViewMatrix),K.onBeforeRender(b,Q,ae,le,D,Ce),K.transparent===!0&&K.side===ti&&K.forceSinglePass===!1?(K.side=En,K.needsUpdate=!0,b.renderBufferDirect(ae,Q,le,K,D,Ce),K.side=Bi,K.needsUpdate=!0,b.renderBufferDirect(ae,Q,le,K,D,Ce),K.side=ti):b.renderBufferDirect(ae,Q,le,K,D,Ce),D.onAfterRender(b,Q,ae,le,K,Ce)}function Ta(D,Q,ae){Q.isScene!==!0&&(Q=F);const le=te.get(D),K=A.state.lights,Ce=A.state.shadowsArray,Le=K.state.version,ze=J.getParameters(D,K.state,Ce,Q,ae),Oe=J.getProgramCacheKey(ze);let Xe=le.programs;le.environment=D.isMeshStandardMaterial?Q.environment:null,le.fog=Q.fog,le.envMap=(D.isMeshStandardMaterial?T:R).get(D.envMap||le.environment),le.envMapRotation=le.environment!==null&&D.envMap===null?Q.environmentRotation:D.envMapRotation,Xe===void 0&&(D.addEventListener("dispose",We),Xe=new Map,le.programs=Xe);let Ye=Xe.get(Oe);if(Ye!==void 0){if(le.currentProgram===Ye&&le.lightsStateVersion===Le)return Xd(D,ze),Ye}else ze.uniforms=J.getUniforms(D),D.onBeforeCompile(ze,b),Ye=J.acquireProgram(ze,Oe),Xe.set(Oe,Ye),le.uniforms=ze.uniforms;const Ve=le.uniforms;return(!D.isShaderMaterial&&!D.isRawShaderMaterial||D.clipping===!0)&&(Ve.clippingPlanes=N.uniform),Xd(D,ze),le.needsLights=cx(D),le.lightsStateVersion=Le,le.needsLights&&(Ve.ambientLightColor.value=K.state.ambient,Ve.lightProbe.value=K.state.probe,Ve.directionalLights.value=K.state.directional,Ve.directionalLightShadows.value=K.state.directionalShadow,Ve.spotLights.value=K.state.spot,Ve.spotLightShadows.value=K.state.spotShadow,Ve.rectAreaLights.value=K.state.rectArea,Ve.ltc_1.value=K.state.rectAreaLTC1,Ve.ltc_2.value=K.state.rectAreaLTC2,Ve.pointLights.value=K.state.point,Ve.pointLightShadows.value=K.state.pointShadow,Ve.hemisphereLights.value=K.state.hemi,Ve.directionalShadowMap.value=K.state.directionalShadowMap,Ve.directionalShadowMatrix.value=K.state.directionalShadowMatrix,Ve.spotShadowMap.value=K.state.spotShadowMap,Ve.spotLightMatrix.value=K.state.spotLightMatrix,Ve.spotLightMap.value=K.state.spotLightMap,Ve.pointShadowMap.value=K.state.pointShadowMap,Ve.pointShadowMatrix.value=K.state.pointShadowMatrix),le.currentProgram=Ye,le.uniformsList=null,Ye}function Wd(D){if(D.uniformsList===null){const Q=D.currentProgram.getUniforms();D.uniformsList=gl.seqWithValue(Q.seq,D.uniforms)}return D.uniformsList}function Xd(D,Q){const ae=te.get(D);ae.outputColorSpace=Q.outputColorSpace,ae.batching=Q.batching,ae.batchingColor=Q.batchingColor,ae.instancing=Q.instancing,ae.instancingColor=Q.instancingColor,ae.instancingMorph=Q.instancingMorph,ae.skinning=Q.skinning,ae.morphTargets=Q.morphTargets,ae.morphNormals=Q.morphNormals,ae.morphColors=Q.morphColors,ae.morphTargetsCount=Q.morphTargetsCount,ae.numClippingPlanes=Q.numClippingPlanes,ae.numIntersection=Q.numClipIntersection,ae.vertexAlphas=Q.vertexAlphas,ae.vertexTangents=Q.vertexTangents,ae.toneMapping=Q.toneMapping}function ax(D,Q,ae,le,K){Q.isScene!==!0&&(Q=F),xe.resetTextureUnits();const Ce=Q.fog,Le=le.isMeshStandardMaterial?Q.environment:null,ze=C===null?b.outputColorSpace:C.isXRRenderTarget===!0?C.texture.colorSpace:uo,Oe=(le.isMeshStandardMaterial?T:R).get(le.envMap||Le),Xe=le.vertexColors===!0&&!!ae.attributes.color&&ae.attributes.color.itemSize===4,Ye=!!ae.attributes.tangent&&(!!le.normalMap||le.anisotropy>0),Ve=!!ae.morphAttributes.position,et=!!ae.morphAttributes.normal,ft=!!ae.morphAttributes.color;let Lt=Ds;le.toneMapped&&(C===null||C.isXRRenderTarget===!0)&&(Lt=b.toneMapping);const Bt=ae.morphAttributes.position||ae.morphAttributes.normal||ae.morphAttributes.color,gt=Bt!==void 0?Bt.length:0,Ge=te.get(le),wt=A.state.lights;if(oe===!0&&(he===!0||D!==P)){const cn=D===P&&le.id===I;N.setState(le,D,cn)}let it=!1;le.version===Ge.__version?(Ge.needsLights&&Ge.lightsStateVersion!==wt.state.version||Ge.outputColorSpace!==ze||K.isBatchedMesh&&Ge.batching===!1||!K.isBatchedMesh&&Ge.batching===!0||K.isBatchedMesh&&Ge.batchingColor===!0&&K.colorTexture===null||K.isBatchedMesh&&Ge.batchingColor===!1&&K.colorTexture!==null||K.isInstancedMesh&&Ge.instancing===!1||!K.isInstancedMesh&&Ge.instancing===!0||K.isSkinnedMesh&&Ge.skinning===!1||!K.isSkinnedMesh&&Ge.skinning===!0||K.isInstancedMesh&&Ge.instancingColor===!0&&K.instanceColor===null||K.isInstancedMesh&&Ge.instancingColor===!1&&K.instanceColor!==null||K.isInstancedMesh&&Ge.instancingMorph===!0&&K.morphTexture===null||K.isInstancedMesh&&Ge.instancingMorph===!1&&K.morphTexture!==null||Ge.envMap!==Oe||le.fog===!0&&Ge.fog!==Ce||Ge.numClippingPlanes!==void 0&&(Ge.numClippingPlanes!==N.numPlanes||Ge.numIntersection!==N.numIntersection)||Ge.vertexAlphas!==Xe||Ge.vertexTangents!==Ye||Ge.morphTargets!==Ve||Ge.morphNormals!==et||Ge.morphColors!==ft||Ge.toneMapping!==Lt||Ge.morphTargetsCount!==gt)&&(it=!0):(it=!0,Ge.__version=le.version);let Fn=Ge.currentProgram;it===!0&&(Fn=Ta(le,Q,K));let xr=!1,Ln=!1,bo=!1;const Rt=Fn.getUniforms(),Sn=Ge.uniforms;if(ie.useProgram(Fn.program)&&(xr=!0,Ln=!0,bo=!0),le.id!==I&&(I=le.id,Ln=!0),xr||P!==D){ie.buffers.depth.getReversed()&&D.reversedDepth!==!0&&(D._reversedDepth=!0,D.updateProjectionMatrix()),Rt.setValue(w,"projectionMatrix",D.projectionMatrix),Rt.setValue(w,"viewMatrix",D.matrixWorldInverse);const vn=Rt.map.cameraPosition;vn!==void 0&&vn.setValue(w,ye.setFromMatrixPosition(D.matrixWorld)),ue.logarithmicDepthBuffer&&Rt.setValue(w,"logDepthBufFC",2/(Math.log(D.far+1)/Math.LN2)),(le.isMeshPhongMaterial||le.isMeshToonMaterial||le.isMeshLambertMaterial||le.isMeshBasicMaterial||le.isMeshStandardMaterial||le.isShaderMaterial)&&Rt.setValue(w,"isOrthographic",D.isOrthographicCamera===!0),P!==D&&(P=D,Ln=!0,bo=!0)}if(K.isSkinnedMesh){Rt.setOptional(w,K,"bindMatrix"),Rt.setOptional(w,K,"bindMatrixInverse");const cn=K.skeleton;cn&&(cn.boneTexture===null&&cn.computeBoneTexture(),Rt.setValue(w,"boneTexture",cn.boneTexture,xe))}K.isBatchedMesh&&(Rt.setOptional(w,K,"batchingTexture"),Rt.setValue(w,"batchingTexture",K._matricesTexture,xe),Rt.setOptional(w,K,"batchingIdTexture"),Rt.setValue(w,"batchingIdTexture",K._indirectTexture,xe),Rt.setOptional(w,K,"batchingColorTexture"),K._colorsTexture!==null&&Rt.setValue(w,"batchingColorTexture",K._colorsTexture,xe));const jn=ae.morphAttributes;if((jn.position!==void 0||jn.normal!==void 0||jn.color!==void 0)&&Te.update(K,ae,Fn),(Ln||Ge.receiveShadow!==K.receiveShadow)&&(Ge.receiveShadow=K.receiveShadow,Rt.setValue(w,"receiveShadow",K.receiveShadow)),le.isMeshGouraudMaterial&&le.envMap!==null&&(Sn.envMap.value=Oe,Sn.flipEnvMap.value=Oe.isCubeTexture&&Oe.isRenderTargetTexture===!1?-1:1),le.isMeshStandardMaterial&&le.envMap===null&&Q.environment!==null&&(Sn.envMapIntensity.value=Q.environmentIntensity),Sn.dfgLUT!==void 0&&(Sn.dfgLUT.value=eT()),Ln&&(Rt.setValue(w,"toneMappingExposure",b.toneMappingExposure),Ge.needsLights&&lx(Sn,bo),Ce&&le.fog===!0&&Pe.refreshFogUniforms(Sn,Ce),Pe.refreshMaterialUniforms(Sn,le,X,$,A.state.transmissionRenderTarget[D.id]),gl.upload(w,Wd(Ge),Sn,xe)),le.isShaderMaterial&&le.uniformsNeedUpdate===!0&&(gl.upload(w,Wd(Ge),Sn,xe),le.uniformsNeedUpdate=!1),le.isSpriteMaterial&&Rt.setValue(w,"center",K.center),Rt.setValue(w,"modelViewMatrix",K.modelViewMatrix),Rt.setValue(w,"normalMatrix",K.normalMatrix),Rt.setValue(w,"modelMatrix",K.matrixWorld),le.isShaderMaterial||le.isRawShaderMaterial){const cn=le.uniformsGroups;for(let vn=0,rc=cn.length;vn<rc;vn++){const Gs=cn[vn];Re.update(Gs,Fn),Re.bind(Gs,Fn)}}return Fn}function lx(D,Q){D.ambientLightColor.needsUpdate=Q,D.lightProbe.needsUpdate=Q,D.directionalLights.needsUpdate=Q,D.directionalLightShadows.needsUpdate=Q,D.pointLights.needsUpdate=Q,D.pointLightShadows.needsUpdate=Q,D.spotLights.needsUpdate=Q,D.spotLightShadows.needsUpdate=Q,D.rectAreaLights.needsUpdate=Q,D.hemisphereLights.needsUpdate=Q}function cx(D){return D.isMeshLambertMaterial||D.isMeshToonMaterial||D.isMeshPhongMaterial||D.isMeshStandardMaterial||D.isShadowMaterial||D.isShaderMaterial&&D.lights===!0}this.getActiveCubeFace=function(){return E},this.getActiveMipmapLevel=function(){return y},this.getRenderTarget=function(){return C},this.setRenderTargetTextures=function(D,Q,ae){const le=te.get(D);le.__autoAllocateDepthBuffer=D.resolveDepthBuffer===!1,le.__autoAllocateDepthBuffer===!1&&(le.__useRenderToTexture=!1),te.get(D.texture).__webglTexture=Q,te.get(D.depthTexture).__webglTexture=le.__autoAllocateDepthBuffer?void 0:ae,le.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(D,Q){const ae=te.get(D);ae.__webglFramebuffer=Q,ae.__useDefaultFramebuffer=Q===void 0};const ux=w.createFramebuffer();this.setRenderTarget=function(D,Q=0,ae=0){C=D,E=Q,y=ae;let le=!0,K=null,Ce=!1,Le=!1;if(D){const Oe=te.get(D);if(Oe.__useDefaultFramebuffer!==void 0)ie.bindFramebuffer(w.FRAMEBUFFER,null),le=!1;else if(Oe.__webglFramebuffer===void 0)xe.setupRenderTarget(D);else if(Oe.__hasExternalTextures)xe.rebindTextures(D,te.get(D.texture).__webglTexture,te.get(D.depthTexture).__webglTexture);else if(D.depthBuffer){const Ve=D.depthTexture;if(Oe.__boundDepthTexture!==Ve){if(Ve!==null&&te.has(Ve)&&(D.width!==Ve.image.width||D.height!==Ve.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");xe.setupDepthRenderbuffer(D)}}const Xe=D.texture;(Xe.isData3DTexture||Xe.isDataArrayTexture||Xe.isCompressedArrayTexture)&&(Le=!0);const Ye=te.get(D).__webglFramebuffer;D.isWebGLCubeRenderTarget?(Array.isArray(Ye[Q])?K=Ye[Q][ae]:K=Ye[Q],Ce=!0):D.samples>0&&xe.useMultisampledRTT(D)===!1?K=te.get(D).__webglMultisampledFramebuffer:Array.isArray(Ye)?K=Ye[ae]:K=Ye,U.copy(D.viewport),O.copy(D.scissor),k=D.scissorTest}else U.copy(ge).multiplyScalar(X).floor(),O.copy(be).multiplyScalar(X).floor(),k=Me;if(ae!==0&&(K=ux),ie.bindFramebuffer(w.FRAMEBUFFER,K)&&le&&ie.drawBuffers(D,K),ie.viewport(U),ie.scissor(O),ie.setScissorTest(k),Ce){const Oe=te.get(D.texture);w.framebufferTexture2D(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_CUBE_MAP_POSITIVE_X+Q,Oe.__webglTexture,ae)}else if(Le){const Oe=Q;for(let Xe=0;Xe<D.textures.length;Xe++){const Ye=te.get(D.textures[Xe]);w.framebufferTextureLayer(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0+Xe,Ye.__webglTexture,ae,Oe)}}else if(D!==null&&ae!==0){const Oe=te.get(D.texture);w.framebufferTexture2D(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,Oe.__webglTexture,ae)}I=-1},this.readRenderTargetPixels=function(D,Q,ae,le,K,Ce,Le,ze=0){if(!(D&&D.isWebGLRenderTarget)){zt("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let Oe=te.get(D).__webglFramebuffer;if(D.isWebGLCubeRenderTarget&&Le!==void 0&&(Oe=Oe[Le]),Oe){ie.bindFramebuffer(w.FRAMEBUFFER,Oe);try{const Xe=D.textures[ze],Ye=Xe.format,Ve=Xe.type;if(!ue.textureFormatReadable(Ye)){zt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!ue.textureTypeReadable(Ve)){zt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}Q>=0&&Q<=D.width-le&&ae>=0&&ae<=D.height-K&&(D.textures.length>1&&w.readBuffer(w.COLOR_ATTACHMENT0+ze),w.readPixels(Q,ae,le,K,He.convert(Ye),He.convert(Ve),Ce))}finally{const Xe=C!==null?te.get(C).__webglFramebuffer:null;ie.bindFramebuffer(w.FRAMEBUFFER,Xe)}}},this.readRenderTargetPixelsAsync=async function(D,Q,ae,le,K,Ce,Le,ze=0){if(!(D&&D.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let Oe=te.get(D).__webglFramebuffer;if(D.isWebGLCubeRenderTarget&&Le!==void 0&&(Oe=Oe[Le]),Oe)if(Q>=0&&Q<=D.width-le&&ae>=0&&ae<=D.height-K){ie.bindFramebuffer(w.FRAMEBUFFER,Oe);const Xe=D.textures[ze],Ye=Xe.format,Ve=Xe.type;if(!ue.textureFormatReadable(Ye))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!ue.textureTypeReadable(Ve))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const et=w.createBuffer();w.bindBuffer(w.PIXEL_PACK_BUFFER,et),w.bufferData(w.PIXEL_PACK_BUFFER,Ce.byteLength,w.STREAM_READ),D.textures.length>1&&w.readBuffer(w.COLOR_ATTACHMENT0+ze),w.readPixels(Q,ae,le,K,He.convert(Ye),He.convert(Ve),0);const ft=C!==null?te.get(C).__webglFramebuffer:null;ie.bindFramebuffer(w.FRAMEBUFFER,ft);const Lt=w.fenceSync(w.SYNC_GPU_COMMANDS_COMPLETE,0);return w.flush(),await IS(w,Lt,4),w.bindBuffer(w.PIXEL_PACK_BUFFER,et),w.getBufferSubData(w.PIXEL_PACK_BUFFER,0,Ce),w.deleteBuffer(et),w.deleteSync(Lt),Ce}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(D,Q=null,ae=0){const le=Math.pow(2,-ae),K=Math.floor(D.image.width*le),Ce=Math.floor(D.image.height*le),Le=Q!==null?Q.x:0,ze=Q!==null?Q.y:0;xe.setTexture2D(D,0),w.copyTexSubImage2D(w.TEXTURE_2D,ae,0,0,Le,ze,K,Ce),ie.unbindTexture()};const fx=w.createFramebuffer(),dx=w.createFramebuffer();this.copyTextureToTexture=function(D,Q,ae=null,le=null,K=0,Ce=null){Ce===null&&(K!==0?(la("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."),Ce=K,K=0):Ce=0);let Le,ze,Oe,Xe,Ye,Ve,et,ft,Lt;const Bt=D.isCompressedTexture?D.mipmaps[Ce]:D.image;if(ae!==null)Le=ae.max.x-ae.min.x,ze=ae.max.y-ae.min.y,Oe=ae.isBox3?ae.max.z-ae.min.z:1,Xe=ae.min.x,Ye=ae.min.y,Ve=ae.isBox3?ae.min.z:0;else{const jn=Math.pow(2,-K);Le=Math.floor(Bt.width*jn),ze=Math.floor(Bt.height*jn),D.isDataArrayTexture?Oe=Bt.depth:D.isData3DTexture?Oe=Math.floor(Bt.depth*jn):Oe=1,Xe=0,Ye=0,Ve=0}le!==null?(et=le.x,ft=le.y,Lt=le.z):(et=0,ft=0,Lt=0);const gt=He.convert(Q.format),Ge=He.convert(Q.type);let wt;Q.isData3DTexture?(xe.setTexture3D(Q,0),wt=w.TEXTURE_3D):Q.isDataArrayTexture||Q.isCompressedArrayTexture?(xe.setTexture2DArray(Q,0),wt=w.TEXTURE_2D_ARRAY):(xe.setTexture2D(Q,0),wt=w.TEXTURE_2D),w.pixelStorei(w.UNPACK_FLIP_Y_WEBGL,Q.flipY),w.pixelStorei(w.UNPACK_PREMULTIPLY_ALPHA_WEBGL,Q.premultiplyAlpha),w.pixelStorei(w.UNPACK_ALIGNMENT,Q.unpackAlignment);const it=w.getParameter(w.UNPACK_ROW_LENGTH),Fn=w.getParameter(w.UNPACK_IMAGE_HEIGHT),xr=w.getParameter(w.UNPACK_SKIP_PIXELS),Ln=w.getParameter(w.UNPACK_SKIP_ROWS),bo=w.getParameter(w.UNPACK_SKIP_IMAGES);w.pixelStorei(w.UNPACK_ROW_LENGTH,Bt.width),w.pixelStorei(w.UNPACK_IMAGE_HEIGHT,Bt.height),w.pixelStorei(w.UNPACK_SKIP_PIXELS,Xe),w.pixelStorei(w.UNPACK_SKIP_ROWS,Ye),w.pixelStorei(w.UNPACK_SKIP_IMAGES,Ve);const Rt=D.isDataArrayTexture||D.isData3DTexture,Sn=Q.isDataArrayTexture||Q.isData3DTexture;if(D.isDepthTexture){const jn=te.get(D),cn=te.get(Q),vn=te.get(jn.__renderTarget),rc=te.get(cn.__renderTarget);ie.bindFramebuffer(w.READ_FRAMEBUFFER,vn.__webglFramebuffer),ie.bindFramebuffer(w.DRAW_FRAMEBUFFER,rc.__webglFramebuffer);for(let Gs=0;Gs<Oe;Gs++)Rt&&(w.framebufferTextureLayer(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,te.get(D).__webglTexture,K,Ve+Gs),w.framebufferTextureLayer(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,te.get(Q).__webglTexture,Ce,Lt+Gs)),w.blitFramebuffer(Xe,Ye,Le,ze,et,ft,Le,ze,w.DEPTH_BUFFER_BIT,w.NEAREST);ie.bindFramebuffer(w.READ_FRAMEBUFFER,null),ie.bindFramebuffer(w.DRAW_FRAMEBUFFER,null)}else if(K!==0||D.isRenderTargetTexture||te.has(D)){const jn=te.get(D),cn=te.get(Q);ie.bindFramebuffer(w.READ_FRAMEBUFFER,fx),ie.bindFramebuffer(w.DRAW_FRAMEBUFFER,dx);for(let vn=0;vn<Oe;vn++)Rt?w.framebufferTextureLayer(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,jn.__webglTexture,K,Ve+vn):w.framebufferTexture2D(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,jn.__webglTexture,K),Sn?w.framebufferTextureLayer(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,cn.__webglTexture,Ce,Lt+vn):w.framebufferTexture2D(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,cn.__webglTexture,Ce),K!==0?w.blitFramebuffer(Xe,Ye,Le,ze,et,ft,Le,ze,w.COLOR_BUFFER_BIT,w.NEAREST):Sn?w.copyTexSubImage3D(wt,Ce,et,ft,Lt+vn,Xe,Ye,Le,ze):w.copyTexSubImage2D(wt,Ce,et,ft,Xe,Ye,Le,ze);ie.bindFramebuffer(w.READ_FRAMEBUFFER,null),ie.bindFramebuffer(w.DRAW_FRAMEBUFFER,null)}else Sn?D.isDataTexture||D.isData3DTexture?w.texSubImage3D(wt,Ce,et,ft,Lt,Le,ze,Oe,gt,Ge,Bt.data):Q.isCompressedArrayTexture?w.compressedTexSubImage3D(wt,Ce,et,ft,Lt,Le,ze,Oe,gt,Bt.data):w.texSubImage3D(wt,Ce,et,ft,Lt,Le,ze,Oe,gt,Ge,Bt):D.isDataTexture?w.texSubImage2D(w.TEXTURE_2D,Ce,et,ft,Le,ze,gt,Ge,Bt.data):D.isCompressedTexture?w.compressedTexSubImage2D(w.TEXTURE_2D,Ce,et,ft,Bt.width,Bt.height,gt,Bt.data):w.texSubImage2D(w.TEXTURE_2D,Ce,et,ft,Le,ze,gt,Ge,Bt);w.pixelStorei(w.UNPACK_ROW_LENGTH,it),w.pixelStorei(w.UNPACK_IMAGE_HEIGHT,Fn),w.pixelStorei(w.UNPACK_SKIP_PIXELS,xr),w.pixelStorei(w.UNPACK_SKIP_ROWS,Ln),w.pixelStorei(w.UNPACK_SKIP_IMAGES,bo),Ce===0&&Q.generateMipmaps&&w.generateMipmap(wt),ie.unbindTexture()},this.initRenderTarget=function(D){te.get(D).__webglFramebuffer===void 0&&xe.setupRenderTarget(D)},this.initTexture=function(D){D.isCubeTexture?xe.setTextureCube(D,0):D.isData3DTexture?xe.setTexture3D(D,0):D.isDataArrayTexture||D.isCompressedArrayTexture?xe.setTexture2DArray(D,0):xe.setTexture2D(D,0),ie.unbindTexture()},this.resetState=function(){E=0,y=0,C=null,ie.reset(),G.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Ei}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const t=this.getContext();t.drawingBufferColorSpace=rt._getDrawingBufferColorSpace(e),t.unpackColorSpace=rt._getUnpackColorSpace()}}class Cs{static idGen=0;constructor(e,t){let n,s;this.promise=new Promise((c,u)=>{n=c,s=u});const r=n.bind(this),o=s.bind(this),a=(...c)=>{r(...c)},l=c=>{o(c)};e(a.bind(this),l.bind(this)),this.abortHandler=t,this.id=Cs.idGen++}then(e){return new Cs((t,n)=>{this.promise=this.promise.then((...s)=>{const r=e(...s);r instanceof Promise||r instanceof Cs?r.then((...o)=>{t(...o)}):t(r)}).catch(s=>{n(s)})},this.abortHandler)}catch(e){return new Cs(t=>{this.promise=this.promise.then((...n)=>{t(...n)}).catch(e)},this.abortHandler)}abort(e){this.abortHandler&&this.abortHandler(e)}}class J0 extends Error{constructor(e){super(e)}}(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){i[0]=t;const n=e[0];let s=n>>16&32768,r=n>>12&2047;const o=n>>23&255;return o<103?s:o>142?(s|=31744,s|=(o==255?0:1)&&n&8388607,s):o<113?(r|=2048,s|=(r>>114-o)+(r>>113-o&1),s):(s|=o-112<<10|r>>1,s+=r&1,s)}})();const Qc=(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){return i[0]=t,e[0]}})(),nT=function(i,e){return i[e]+(i[e+1]<<8)+(i[e+2]<<16)+(i[e+3]<<24)},ec=function(i,e,t=!0,n){const s=new AbortController,r=s.signal;let o=!1;const a=u=>{s.abort(u),o=!0};let l=!1;const c=(u,f,d,h)=>{e&&!l&&(e(u,f,d,h),u===100&&(l=!0))};return new Cs((u,f)=>{const d={signal:r};n&&(d.headers=n),fetch(i,d).then(async h=>{if(!h.ok){const A=await h.text();f(new Error(`Fetch failed: ${h.status} ${h.statusText} ${A}`));return}const g=h.body.getReader();let p=0,x=h.headers.get("Content-Length"),m=x?parseInt(x):void 0;const _=[];for(;!o;)try{const{value:A,done:S}=await g.read();if(S){if(c(100,"100%",A,m),t){const M=new Blob(_).arrayBuffer();u(M)}else u();break}p+=A.length;let v,b;m!==void 0&&(v=p/m*100,b=`${v.toFixed(2)}%`),t&&_.push(A),c(v,b,A,m)}catch(A){f(A);return}}).catch(h=>{f(new J0(h))})},a)},Ct=function(i,e,t){return Math.max(Math.min(i,t),e)},Br=function(){return performance.now()/1e3},kr=i=>{if(i.geometry&&(i.geometry.dispose(),i.geometry=null),i.material&&(i.material.dispose(),i.material=null),i.children)for(let e of i.children)kr(e)},Gn=(i,e)=>new Promise(t=>{window.setTimeout(()=>{t(i?i():void 0)},e?1:50)}),Zr=(i=0)=>{let e=0;if(i===1)e=9;else if(i===2)e=24;else if(i===3)e=45;else if(i>3)throw new Error("getSphericalHarmonicsComponentCountForDegree() -> Invalid spherical harmonics degree");return e},sd=()=>{let i,e;return{promise:new Promise((n,s)=>{i=n,e=s}),resolve:i,reject:e}},Yc=i=>{let e,t;return i||(i=()=>{}),{promise:new Cs((s,r)=>{e=s,t=r},i),resolve:e,reject:t}};class iT{constructor(e,t,n){this.major=e,this.minor=t,this.patch=n}toString(){return`${this.major}_${this.minor}_${this.patch}`}}function rd(){const i=navigator.userAgent;return i.indexOf("iPhone")>0||i.indexOf("iPad")>0}function eg(){if(rd()){const i=navigator.userAgent.match(/OS (\d+)_(\d+)_?(\d+)?/);return new iT(parseInt(i[1]||0,10),parseInt(i[2]||0,10),parseInt(i[3]||0,10))}else return null}const sT=14;class Ee{static OFFSET={X:0,Y:1,Z:2,SCALE0:3,SCALE1:4,SCALE2:5,ROTATION0:6,ROTATION1:7,ROTATION2:8,ROTATION3:9,FDC0:10,FDC1:11,FDC2:12,OPACITY:13,FRC0:14,FRC1:15,FRC2:16,FRC3:17,FRC4:18,FRC5:19,FRC6:20,FRC7:21,FRC8:22,FRC9:23,FRC10:24,FRC11:25,FRC12:26,FRC13:27,FRC14:28,FRC15:29,FRC16:30,FRC17:31,FRC18:32,FRC19:33,FRC20:34,FRC21:35,FRC22:36,FRC23:37};constructor(e=0){this.sphericalHarmonicsDegree=e,this.sphericalHarmonicsCount=Zr(this.sphericalHarmonicsDegree),this.componentCount=this.sphericalHarmonicsCount+sT,this.defaultSphericalHarmonics=new Array(this.sphericalHarmonicsCount).fill(0),this.splats=[],this.splatCount=0}static createSplat(e=0){const t=[0,0,0,1,1,1,1,0,0,0,0,0,0,0];let n=Zr(e);for(let s=0;s<n;s++)t.push(0);return t}addSplat(e){this.splats.push(e),this.splatCount++}getSplat(e){return this.splats[e]}addDefaultSplat(){const e=Ee.createSplat(this.sphericalHarmonicsDegree);return this.addSplat(e),e}addSplatFromComonents(e,t,n,s,r,o,a,l,c,u,f,d,h,g,...p){const x=[e,t,n,s,r,o,a,l,c,u,f,d,h,g,...this.defaultSphericalHarmonics];for(let m=0;m<p.length&&m<this.sphericalHarmonicsCount;m++)x[m]=p[m];return this.addSplat(x),x}addSplatFromArray(e,t){const n=e.splats[t],s=Ee.createSplat(this.sphericalHarmonicsDegree);for(let r=0;r<this.componentCount&&r<n.length;r++)s[r]=n[r];this.addSplat(s)}}class pt{static DefaultSplatSortDistanceMapPrecision=16;static MemoryPageSize=65536;static BytesPerFloat=4;static BytesPerInt=4;static MaxScenes=32;static ProgressiveLoadSectionSize=262144;static ProgressiveLoadSectionDelayDuration=15;static SphericalHarmonics8BitCompressionRange=3}const rT=pt.SphericalHarmonics8BitCompressionRange,_s=rT/2,qt=ca.toHalfFloat.bind(ca),od=ca.fromHalfFloat.bind(ca),Mt=(i,e,t=!1,n,s)=>{if(e===0)return i;if(e===1||e===2&&!t)return ca.fromHalfFloat(i);if(e===2)return ad(i,n,s)},Wo=(i,e,t)=>{i=Ct(i,e,t);const n=t-e;return Ct(Math.floor((i-e)/n*255),0,255)},ad=(i,e,t)=>{const n=t-e;return i/255*n+e},tg=(i,e,t)=>Wo(od(i,e,t)),oT=(i,e,t)=>qt(ad(i,e,t)),at=(i,e,t,n=!1)=>t===0?i.getFloat32(e*4,!0):t===1||t===2&&!n?i.getUint16(e*2,!0):i.getUint8(e,!0),aT=(function(){const i=e=>e;return function(e,t,n,s=!1){if(t===n)return e;let r=i;return t===2&&s?n===1?r=oT:n==0&&(r=ad):t===2||t===1?n===0?r=od:n==2&&(s?r=tg:r=i):t===0&&(n===1?r=qt:n==2&&(s?r=Wo:r=qt)),r(e)}})(),Ur=(i,e,t,n,s=0)=>{const r=new Uint8Array(i,e),o=new Uint8Array(t,n);for(let a=0;a<s;a++)o[a]=r[a]};class j{static CurrentMajorVersion=0;static CurrentMinorVersion=1;static CenterComponentCount=3;static ScaleComponentCount=3;static RotationComponentCount=4;static ColorComponentCount=4;static CovarianceComponentCount=6;static SplatScaleOffsetFloat=3;static SplatRotationOffsetFloat=6;static CompressionLevels={0:{BytesPerCenter:12,BytesPerScale:12,BytesPerRotation:16,BytesPerColor:4,ScaleOffsetBytes:12,RotationffsetBytes:24,ColorOffsetBytes:40,SphericalHarmonicsOffsetBytes:44,ScaleRange:1,BytesPerSphericalHarmonicsComponent:4,SphericalHarmonicsOffsetFloat:11,SphericalHarmonicsDegrees:{0:{BytesPerSplat:44},1:{BytesPerSplat:80},2:{BytesPerSplat:140}}},1:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:2,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:42},2:{BytesPerSplat:72}}},2:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:1,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:33},2:{BytesPerSplat:48}}}};static CovarianceSizeFloats=6;static HeaderSizeBytes=4096;static SectionHeaderSizeBytes=1024;static BucketStorageSizeBytes=12;static BucketStorageSizeFloats=3;static BucketBlockSize=5;static BucketSize=256;constructor(e,t=!0){this.constructFromBuffer(e,t)}getSplatCount(){return this.splatCount}getMaxSplatCount(){return this.maxSplatCount}getMinSphericalHarmonicsDegree(){let e=0;for(let t=0;t<this.sections.length;t++){const n=this.sections[t];(t===0||n.sphericalHarmonicsDegree<e)&&(e=n.sphericalHarmonicsDegree)}return e}getBucketIndex(e,t){let n;const s=e.fullBucketCount*e.bucketSize;if(t<s)n=Math.floor(t/e.bucketSize);else{let r=s;n=e.fullBucketCount;let o=0;for(;r<e.splatCount;){let a=e.partiallyFilledBucketLengths[o];if(t>=r&&t<r+a)break;r+=a,n++,o++}}return n}getSplatCenter(e,t,n){const s=this.globalSplatIndexToSectionMap[e],r=this.sections[s],o=e-r.splatCountOffset,a=r.bytesPerSplat*o,l=new DataView(this.bufferData,r.dataBase+a),c=at(l,0,this.compressionLevel),u=at(l,1,this.compressionLevel),f=at(l,2,this.compressionLevel);if(this.compressionLevel>=1){const h=this.getBucketIndex(r,o)*j.BucketStorageSizeFloats,g=r.compressionScaleFactor,p=r.compressionScaleRange;t.x=(c-p)*g+r.bucketArray[h],t.y=(u-p)*g+r.bucketArray[h+1],t.z=(f-p)*g+r.bucketArray[h+2]}else t.x=c,t.y=u,t.z=f;n&&t.applyMatrix4(n)}getSplatScaleAndRotation=(function(){const e=new qe,t=new qe,n=new qe,s=new L,r=new L,o=new bt;return function(a,l,c,u,f){const d=this.globalSplatIndexToSectionMap[a],h=this.sections[d],g=a-h.splatCountOffset,p=h.bytesPerSplat*g+j.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,x=new DataView(this.bufferData,h.dataBase+p);r.set(Mt(at(x,0,this.compressionLevel),this.compressionLevel),Mt(at(x,1,this.compressionLevel),this.compressionLevel),Mt(at(x,2,this.compressionLevel),this.compressionLevel)),f&&(f.x!==void 0&&(r.x=f.x),f.y!==void 0&&(r.y=f.y),f.z!==void 0&&(r.z=f.z)),o.set(Mt(at(x,4,this.compressionLevel),this.compressionLevel),Mt(at(x,5,this.compressionLevel),this.compressionLevel),Mt(at(x,6,this.compressionLevel),this.compressionLevel),Mt(at(x,3,this.compressionLevel),this.compressionLevel)),u?(e.makeScale(r.x,r.y,r.z),t.makeRotationFromQuaternion(o),n.copy(e).multiply(t).multiply(u),n.decompose(s,c,l)):(l.copy(r),c.copy(o))}})();getSplatColor(e,t){const n=this.globalSplatIndexToSectionMap[e],s=this.sections[n],r=e-s.splatCountOffset,o=s.bytesPerSplat*r+j.CompressionLevels[this.compressionLevel].ColorOffsetBytes,a=new Uint8Array(this.bufferData,s.dataBase+o,4);t.set(a[0],a[1],a[2],a[3])}fillSplatCenterArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);const a=new L;for(let l=n;l<=s;l++){const c=this.globalSplatIndexToSectionMap[l],u=this.sections[c],f=l-u.splatCountOffset,d=(l-n+r)*j.CenterComponentCount,h=u.bytesPerSplat*f,g=new DataView(this.bufferData,u.dataBase+h),p=at(g,0,this.compressionLevel),x=at(g,1,this.compressionLevel),m=at(g,2,this.compressionLevel);if(this.compressionLevel>=1){const A=this.getBucketIndex(u,f)*j.BucketStorageSizeFloats,S=u.compressionScaleFactor,v=u.compressionScaleRange;a.x=(p-v)*S+u.bucketArray[A],a.y=(x-v)*S+u.bucketArray[A+1],a.z=(m-v)*S+u.bucketArray[A+2]}else a.x=p,a.y=x,a.z=m;t&&a.applyMatrix4(t),e[d]=a.x,e[d+1]=a.y,e[d+2]=a.z}}fillSplatScaleRotationArray=(function(){const e=new qe,t=new qe,n=new qe,s=new L,r=new bt,o=new L,a=l=>{const c=l.w<0?-1:1;l.x*=c,l.y*=c,l.z*=c,l.w*=c};return function(l,c,u,f,d,h,g,p){const x=this.splatCount;f=f||0,d=d||x-1,h===void 0&&(h=f);const m=(_,A)=>aT(_,A,g);for(let _=f;_<=d;_++){const A=this.globalSplatIndexToSectionMap[_],S=this.sections[A],v=_-S.splatCountOffset,b=S.bytesPerSplat*v+j.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,M=(_-f+h)*j.ScaleComponentCount,E=(_-f+h)*j.RotationComponentCount,y=new DataView(this.bufferData,S.dataBase+b),C=p&&p.x!==void 0?p.x:at(y,0,this.compressionLevel),I=p&&p.y!==void 0?p.y:at(y,1,this.compressionLevel),P=p&&p.z!==void 0?p.z:at(y,2,this.compressionLevel),U=at(y,3,this.compressionLevel),O=at(y,4,this.compressionLevel),k=at(y,5,this.compressionLevel),z=at(y,6,this.compressionLevel);s.set(Mt(C,this.compressionLevel),Mt(I,this.compressionLevel),Mt(P,this.compressionLevel)),r.set(Mt(O,this.compressionLevel),Mt(k,this.compressionLevel),Mt(z,this.compressionLevel),Mt(U,this.compressionLevel)).normalize(),u&&(o.set(0,0,0),e.makeScale(s.x,s.y,s.z),t.makeRotationFromQuaternion(r),n.identity().premultiply(e).premultiply(t),n.premultiply(u),n.decompose(o,r,s),r.normalize()),a(r),l&&(l[M]=m(s.x,0),l[M+1]=m(s.y,0),l[M+2]=m(s.z,0)),c&&(c[E]=m(r.x,0),c[E+1]=m(r.y,0),c[E+2]=m(r.z,0),c[E+3]=m(r.w,0))}}})();static computeCovariance=(function(){const e=new qe,t=new Qe,n=new Qe,s=new Qe,r=new Qe,o=new Qe,a=new Qe;return function(l,c,u,f,d=0,h){e.makeScale(l.x,l.y,l.z),t.setFromMatrix4(e),e.makeRotationFromQuaternion(c),n.setFromMatrix4(e),s.copy(n).multiply(t),r.copy(s).transpose().premultiply(s),u&&(o.setFromMatrix4(u),a.copy(o).transpose(),r.multiply(a),r.premultiply(o)),h>=1?(f[d]=qt(r.elements[0]),f[d+1]=qt(r.elements[3]),f[d+2]=qt(r.elements[6]),f[d+3]=qt(r.elements[4]),f[d+4]=qt(r.elements[7]),f[d+5]=qt(r.elements[8])):(f[d]=r.elements[0],f[d+1]=r.elements[3],f[d+2]=r.elements[6],f[d+3]=r.elements[4],f[d+4]=r.elements[7],f[d+5]=r.elements[8])}})();fillSplatCovarianceArray(e,t,n,s,r,o){const a=this.splatCount,l=new L,c=new bt;n=n||0,s=s||a-1,r===void 0&&(r=n);for(let u=n;u<=s;u++){const f=this.globalSplatIndexToSectionMap[u],d=this.sections[f],h=u-d.splatCountOffset,g=(u-n+r)*j.CovarianceComponentCount,p=d.bytesPerSplat*h+j.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,x=new DataView(this.bufferData,d.dataBase+p);l.set(Mt(at(x,0,this.compressionLevel),this.compressionLevel),Mt(at(x,1,this.compressionLevel),this.compressionLevel),Mt(at(x,2,this.compressionLevel),this.compressionLevel)),c.set(Mt(at(x,4,this.compressionLevel),this.compressionLevel),Mt(at(x,5,this.compressionLevel),this.compressionLevel),Mt(at(x,6,this.compressionLevel),this.compressionLevel),Mt(at(x,3,this.compressionLevel),this.compressionLevel)),j.computeCovariance(l,c,t,e,g,o)}}fillSplatColorArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);for(let a=n;a<=s;a++){const l=this.globalSplatIndexToSectionMap[a],c=this.sections[l],u=a-c.splatCountOffset,f=(a-n+r)*j.ColorComponentCount,d=c.bytesPerSplat*u+j.CompressionLevels[this.compressionLevel].ColorOffsetBytes,h=new Uint8Array(this.bufferData,c.dataBase+d);let g=h[3];g=g>=t?g:0,e[f]=h[0],e[f+1]=h[1],e[f+2]=h[2],e[f+3]=g}}fillSphericalHarmonicsArray=(function(){for(let O=0;O<15;O++)new L;const e=new Qe,t=new qe,n=new L,s=new L,r=new bt,o=[],a=[],l=[],c=[],u=[],f=[],d=[],h=[],g=[],p=[],x=[],m=[],_=[],A=[],S=[],v=[],b=[],M=[],E=O=>O,y=(O,k,z,W)=>{O[0]=k,O[1]=z,O[2]=W},C=(O,k,z,W,H)=>{O[0]=at(k,W,H,!0),O[1]=at(k,W+z,H,!0),O[2]=at(k,W+z+z,H,!0)},I=(O,k)=>{k[0]=O[0],k[1]=O[1],k[2]=O[2]},P=(O,k,z,W)=>{k[z]=W(O[0]),k[z+1]=W(O[1]),k[z+2]=W(O[2])},U=(O,k,z,W,H)=>(k[0]=Mt(O[0],z,!0,W,H),k[1]=Mt(O[1],z,!0,W,H),k[2]=Mt(O[2],z,!0,W,H),k);return function(O,k,z,W,H,$,X){const Z=this.splatCount;W=W||0,H=H||Z-1,$===void 0&&($=W),z&&k>=1&&(t.copy(z),t.decompose(n,r,s),r.normalize(),t.makeRotationFromQuaternion(r),e.setFromMatrix4(t),y(o,e.elements[4],-e.elements[7],e.elements[1]),y(a,-e.elements[5],e.elements[8],-e.elements[2]),y(l,e.elements[3],-e.elements[6],e.elements[0]));const pe=be=>tg(be,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff),ge=be=>Wo(be,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff);for(let be=W;be<=H;be++){const Me=this.globalSplatIndexToSectionMap[be],Y=this.sections[Me];k=Math.min(k,Y.sphericalHarmonicsDegree);const oe=Zr(k),he=be-Y.splatCountOffset,Fe=Y.bytesPerSplat*he+j.CompressionLevels[this.compressionLevel].SphericalHarmonicsOffsetBytes,ye=new DataView(this.bufferData,Y.dataBase+Fe),ke=(be-W+$)*oe;let F=z?0:this.compressionLevel,B=E;F!==X&&(F===1?X===0?B=od:X==2&&(B=pe):F===0&&(X===1?B=qt:X==2&&(B=ge)));const q=this.minSphericalHarmonicsCoeff,w=this.maxSphericalHarmonicsCoeff;k>=1&&(C(g,ye,3,0,this.compressionLevel),C(p,ye,3,1,this.compressionLevel),C(x,ye,3,2,this.compressionLevel),z?(U(g,g,this.compressionLevel,q,w),U(p,p,this.compressionLevel,q,w),U(x,x,this.compressionLevel,q,w),j.rotateSphericalHarmonics3(g,p,x,o,a,l,A,S,v)):(I(g,A),I(p,S),I(x,v)),P(A,O,ke,B),P(S,O,ke+3,B),P(v,O,ke+6,B),k>=2&&(C(g,ye,5,9,this.compressionLevel),C(p,ye,5,10,this.compressionLevel),C(x,ye,5,11,this.compressionLevel),C(m,ye,5,12,this.compressionLevel),C(_,ye,5,13,this.compressionLevel),z?(U(g,g,this.compressionLevel,q,w),U(p,p,this.compressionLevel,q,w),U(x,x,this.compressionLevel,q,w),U(m,m,this.compressionLevel,q,w),U(_,_,this.compressionLevel,q,w),j.rotateSphericalHarmonics5(g,p,x,m,_,o,a,l,c,u,f,d,h,A,S,v,b,M)):(I(g,A),I(p,S),I(x,v),I(m,b),I(_,M)),P(A,O,ke+9,B),P(S,O,ke+12,B),P(v,O,ke+15,B),P(b,O,ke+18,B),P(M,O,ke+21,B)))}}})();static dot3=(e,t,n,s,r)=>{r[0]=r[1]=r[2]=0;const o=s[0],a=s[1],l=s[2];j.addInto3(e[0]*o,e[1]*o,e[2]*o,r),j.addInto3(t[0]*a,t[1]*a,t[2]*a,r),j.addInto3(n[0]*l,n[1]*l,n[2]*l,r)};static addInto3=(e,t,n,s)=>{s[0]=s[0]+e,s[1]=s[1]+t,s[2]=s[2]+n};static dot5=(e,t,n,s,r,o,a)=>{a[0]=a[1]=a[2]=0;const l=o[0],c=o[1],u=o[2],f=o[3],d=o[4];j.addInto3(e[0]*l,e[1]*l,e[2]*l,a),j.addInto3(t[0]*c,t[1]*c,t[2]*c,a),j.addInto3(n[0]*u,n[1]*u,n[2]*u,a),j.addInto3(s[0]*f,s[1]*f,s[2]*f,a),j.addInto3(r[0]*d,r[1]*d,r[2]*d,a)};static rotateSphericalHarmonics3=(e,t,n,s,r,o,a,l,c)=>{j.dot3(e,t,n,s,a),j.dot3(e,t,n,r,l),j.dot3(e,t,n,o,c)};static rotateSphericalHarmonics5=(e,t,n,s,r,o,a,l,c,u,f,d,h,g,p,x,m,_)=>{const A=Math.sqrt(.25),S=Math.sqrt(3/4),v=Math.sqrt(1/3),b=Math.sqrt(4/3),M=Math.sqrt(1/12);c[0]=A*(l[2]*o[0]+l[0]*o[2]+(o[2]*l[0]+o[0]*l[2])),c[1]=l[1]*o[0]+o[1]*l[0],c[2]=S*(l[1]*o[1]+o[1]*l[1]),c[3]=l[1]*o[2]+o[1]*l[2],c[4]=A*(l[2]*o[2]-l[0]*o[0]+(o[2]*l[2]-o[0]*l[0])),j.dot5(e,t,n,s,r,c,g),u[0]=A*(a[2]*o[0]+a[0]*o[2]+(o[2]*a[0]+o[0]*a[2])),u[1]=a[1]*o[0]+o[1]*a[0],u[2]=S*(a[1]*o[1]+o[1]*a[1]),u[3]=a[1]*o[2]+o[1]*a[2],u[4]=A*(a[2]*o[2]-a[0]*o[0]+(o[2]*a[2]-o[0]*a[0])),j.dot5(e,t,n,s,r,u,p),f[0]=v*(a[2]*a[0]+a[0]*a[2])+-M*(l[2]*l[0]+l[0]*l[2]+(o[2]*o[0]+o[0]*o[2])),f[1]=b*a[1]*a[0]+-v*(l[1]*l[0]+o[1]*o[0]),f[2]=a[1]*a[1]+-A*(l[1]*l[1]+o[1]*o[1]),f[3]=b*a[1]*a[2]+-v*(l[1]*l[2]+o[1]*o[2]),f[4]=v*(a[2]*a[2]-a[0]*a[0])+-M*(l[2]*l[2]-l[0]*l[0]+(o[2]*o[2]-o[0]*o[0])),j.dot5(e,t,n,s,r,f,x),d[0]=A*(a[2]*l[0]+a[0]*l[2]+(l[2]*a[0]+l[0]*a[2])),d[1]=a[1]*l[0]+l[1]*a[0],d[2]=S*(a[1]*l[1]+l[1]*a[1]),d[3]=a[1]*l[2]+l[1]*a[2],d[4]=A*(a[2]*l[2]-a[0]*l[0]+(l[2]*a[2]-l[0]*a[0])),j.dot5(e,t,n,s,r,d,m),h[0]=A*(l[2]*l[0]+l[0]*l[2]-(o[2]*o[0]+o[0]*o[2])),h[1]=l[1]*l[0]-o[1]*o[0],h[2]=S*(l[1]*l[1]-o[1]*o[1]),h[3]=l[1]*l[2]-o[1]*o[2],h[4]=A*(l[2]*l[2]-l[0]*l[0]-(o[2]*o[2]-o[0]*o[0])),j.dot5(e,t,n,s,r,h,_)};static parseHeader(e){const t=new Uint8Array(e,0,j.HeaderSizeBytes),n=new Uint16Array(e,0,j.HeaderSizeBytes/2),s=new Uint32Array(e,0,j.HeaderSizeBytes/4),r=new Float32Array(e,0,j.HeaderSizeBytes/4),o=t[0],a=t[1],l=s[1],c=s[2],u=s[3],f=s[4],d=n[10],h=new L(r[6],r[7],r[8]),g=r[9]||-_s,p=r[10]||_s;return{versionMajor:o,versionMinor:a,maxSectionCount:l,sectionCount:c,maxSplatCount:u,splatCount:f,compressionLevel:d,sceneCenter:h,minSphericalHarmonicsCoeff:g,maxSphericalHarmonicsCoeff:p}}static writeHeaderCountsToBuffer(e,t,n){const s=new Uint32Array(n,0,j.HeaderSizeBytes/4);s[2]=e,s[4]=t}static writeHeaderToBuffer(e,t){const n=new Uint8Array(t,0,j.HeaderSizeBytes),s=new Uint16Array(t,0,j.HeaderSizeBytes/2),r=new Uint32Array(t,0,j.HeaderSizeBytes/4),o=new Float32Array(t,0,j.HeaderSizeBytes/4);n[0]=e.versionMajor,n[1]=e.versionMinor,n[2]=0,n[3]=0,r[1]=e.maxSectionCount,r[2]=e.sectionCount,r[3]=e.maxSplatCount,r[4]=e.splatCount,s[10]=e.compressionLevel,o[6]=e.sceneCenter.x,o[7]=e.sceneCenter.y,o[8]=e.sceneCenter.z,o[9]=e.minSphericalHarmonicsCoeff||-_s,o[10]=e.maxSphericalHarmonicsCoeff||_s}static parseSectionHeaders(e,t,n=0,s){const r=e.compressionLevel,o=e.maxSectionCount,a=new Uint16Array(t,n,o*j.SectionHeaderSizeBytes/2),l=new Uint32Array(t,n,o*j.SectionHeaderSizeBytes/4),c=new Float32Array(t,n,o*j.SectionHeaderSizeBytes/4),u=[];let f=0,d=f/2,h=f/4,g=j.HeaderSizeBytes+e.maxSectionCount*j.SectionHeaderSizeBytes,p=0;for(let x=0;x<o;x++){const m=l[h+1],_=l[h+2],A=l[h+3],S=c[h+4],v=S/2,b=a[d+10],M=l[h+6]||j.CompressionLevels[r].ScaleRange,E=l[h+8],y=l[h+9],C=y*4,I=b*A+C,P=a[d+20],{bytesPerSplat:U}=j.calculateComponentStorage(r,P),O=U*m,k=O+I,z={bytesPerSplat:U,splatCountOffset:p,splatCount:s?m:0,maxSplatCount:m,bucketSize:_,bucketCount:A,bucketBlockSize:S,halfBucketBlockSize:v,bucketStorageSizeBytes:b,bucketsStorageSizeBytes:I,splatDataStorageSizeBytes:O,storageSizeBytes:k,compressionScaleRange:M,compressionScaleFactor:v/M,base:g,bucketsBase:g+C,dataBase:g+I,fullBucketCount:E,partiallyFilledBucketCount:y,sphericalHarmonicsDegree:P};u[x]=z,g+=k,f+=j.SectionHeaderSizeBytes,d=f/2,h=f/4,p+=m}return u}static writeSectionHeaderToBuffer(e,t,n,s=0){const r=new Uint16Array(n,s,j.SectionHeaderSizeBytes/2),o=new Uint32Array(n,s,j.SectionHeaderSizeBytes/4),a=new Float32Array(n,s,j.SectionHeaderSizeBytes/4);o[0]=e.splatCount,o[1]=e.maxSplatCount,o[2]=t>=1?e.bucketSize:0,o[3]=t>=1?e.bucketCount:0,a[4]=t>=1?e.bucketBlockSize:0,r[10]=t>=1?j.BucketStorageSizeBytes:0,o[6]=t>=1?e.compressionScaleRange:0,o[7]=e.storageSizeBytes,o[8]=t>=1?e.fullBucketCount:0,o[9]=t>=1?e.partiallyFilledBucketCount:0,r[20]=e.sphericalHarmonicsDegree}static writeSectionHeaderSplatCountToBuffer(e,t,n=0){const s=new Uint32Array(t,n,j.SectionHeaderSizeBytes/4);s[0]=e}constructFromBuffer(e,t){this.bufferData=e,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSectionMap=[];const n=j.parseHeader(this.bufferData);this.versionMajor=n.versionMajor,this.versionMinor=n.versionMinor,this.maxSectionCount=n.maxSectionCount,this.sectionCount=t?n.maxSectionCount:0,this.maxSplatCount=n.maxSplatCount,this.splatCount=t?n.maxSplatCount:0,this.compressionLevel=n.compressionLevel,this.sceneCenter=new L().copy(n.sceneCenter),this.minSphericalHarmonicsCoeff=n.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff=n.maxSphericalHarmonicsCoeff,this.sections=j.parseSectionHeaders(n,this.bufferData,j.HeaderSizeBytes,t),this.linkBufferArrays(),this.buildMaps()}static calculateComponentStorage(e,t){const n=j.CompressionLevels[e].BytesPerCenter,s=j.CompressionLevels[e].BytesPerScale,r=j.CompressionLevels[e].BytesPerRotation,o=j.CompressionLevels[e].BytesPerColor,a=Zr(t),l=j.CompressionLevels[e].BytesPerSphericalHarmonicsComponent*a,c=n+s+r+o+l;return{bytesPerCenter:n,bytesPerScale:s,bytesPerRotation:r,bytesPerColor:o,sphericalHarmonicsComponentsPerSplat:a,sphericalHarmonicsBytesPerSplat:l,bytesPerSplat:c}}linkBufferArrays(){for(let e=0;e<this.maxSectionCount;e++){const t=this.sections[e];t.bucketArray=new Float32Array(this.bufferData,t.bucketsBase,t.bucketCount*j.BucketStorageSizeFloats),t.partiallyFilledBucketCount>0&&(t.partiallyFilledBucketLengths=new Uint32Array(this.bufferData,t.base,t.partiallyFilledBucketCount))}}buildMaps(){let e=0;for(let t=0;t<this.maxSectionCount;t++){const n=this.sections[t];for(let s=0;s<n.maxSplatCount;s++){const r=e+s;this.globalSplatIndexToLocalSplatIndexMap[r]=s,this.globalSplatIndexToSectionMap[r]=t}e+=n.maxSplatCount}}updateLoadedCounts(e,t){j.writeHeaderCountsToBuffer(e,t,this.bufferData),this.sectionCount=e,this.splatCount=t}updateSectionLoadedCounts(e,t){const n=j.HeaderSizeBytes+j.SectionHeaderSizeBytes*e;j.writeSectionHeaderSplatCountToBuffer(t,this.bufferData,n),this.sections[e].splatCount=t}static writeSplatDataToSectionBuffer=(function(){const e=new ArrayBuffer(12),t=new ArrayBuffer(12),n=new ArrayBuffer(16),s=new ArrayBuffer(4),r=new ArrayBuffer(256),o=new bt,a=new L,l=new L,{X:c,Y:u,Z:f,SCALE0:d,SCALE1:h,SCALE2:g,ROTATION0:p,ROTATION1:x,ROTATION2:m,ROTATION3:_,FDC0:A,FDC1:S,FDC2:v,OPACITY:b,FRC0:M,FRC9:E}=Ee.OFFSET,y=(C,I,P)=>{const U=P*2+1;return C=Math.round(C*I)+P,Ct(C,0,U)};return function(C,I,P,U,O,k,z,W,H=-_s,$=_s){const X=Zr(O),Z=j.CompressionLevels[U].BytesPerCenter,pe=j.CompressionLevels[U].BytesPerScale,ge=j.CompressionLevels[U].BytesPerRotation,be=j.CompressionLevels[U].BytesPerColor,Me=P,Y=Me+Z,oe=Y+pe,he=oe+ge,Fe=he+be;if(C[p]!==void 0?(o.set(C[p],C[x],C[m],C[_]),o.normalize()):o.set(1,0,0,0),C[d]!==void 0?a.set(C[d]||0,C[h]||0,C[g]||0):a.set(0,0,0),U===0){const ke=new Float32Array(I,Me,j.CenterComponentCount),F=new Float32Array(I,oe,j.RotationComponentCount),B=new Float32Array(I,Y,j.ScaleComponentCount);if(F.set([o.x,o.y,o.z,o.w]),B.set([a.x,a.y,a.z]),ke.set([C[c],C[u],C[f]]),O>0){const q=new Float32Array(I,Fe,X);if(O>=1){for(let w=0;w<9;w++)q[w]=C[M+w]||0;if(O>=2)for(let w=0;w<15;w++)q[w+9]=C[E+w]||0}}}else{const ke=new Uint16Array(e,0,j.CenterComponentCount),F=new Uint16Array(n,0,j.RotationComponentCount),B=new Uint16Array(t,0,j.ScaleComponentCount);if(F.set([qt(o.x),qt(o.y),qt(o.z),qt(o.w)]),B.set([qt(a.x),qt(a.y),qt(a.z)]),l.set(C[c],C[u],C[f]).sub(k),l.x=y(l.x,z,W),l.y=y(l.y,z,W),l.z=y(l.z,z,W),ke.set([l.x,l.y,l.z]),O>0){const q=U===1?Uint16Array:Uint8Array,w=U===1?2:1,ne=new q(r,0,X);if(O>=1){for(let ue=0;ue<9;ue++){const ie=C[M+ue]||0;ne[ue]=U===1?qt(ie):Wo(ie,H,$)}const re=9*w;if(Ur(ne.buffer,0,I,Fe,re),O>=2){for(let ue=0;ue<15;ue++){const ie=C[E+ue]||0;ne[ue+9]=U===1?qt(ie):Wo(ie,H,$)}Ur(ne.buffer,re,I,Fe+re,15*w)}}}Ur(ke.buffer,0,I,Me,6),Ur(B.buffer,0,I,Y,6),Ur(F.buffer,0,I,oe,8)}const ye=new Uint8ClampedArray(s,0,4);ye.set([C[A]||0,C[S]||0,C[v]||0]),ye[3]=C[b]||0,Ur(ye.buffer,0,I,he,4)}})();static generateFromUncompressedSplatArrays(e,t,n,s,r,o,a=[]){let l=0;for(let v=0;v<e.length;v++){const b=e[v];l=Math.max(b.sphericalHarmonicsDegree,l)}let c,u;for(let v=0;v<e.length;v++){const b=e[v];for(let M=0;M<b.splats.length;M++){const E=b.splats[M];for(let y=Ee.OFFSET.FRC0;y<Ee.OFFSET.FRC23&&y<E.length;y++)(!c||E[y]<c)&&(c=E[y]),(!u||E[y]>u)&&(u=E[y])}}c=c||-_s,u=u||_s;const{bytesPerSplat:f}=j.calculateComponentStorage(n,l),d=j.CompressionLevels[n].ScaleRange,h=[],g=[];let p=0;for(let v=0;v<e.length;v++){const b=e[v],M=new Ee(l);for(let Me=0;Me<b.splatCount;Me++){const Y=b.splats[Me];(Y[Ee.OFFSET.OPACITY]||0)>=t&&M.addSplat(Y)}const E=a[v]||{},y=(E.blockSizeFactor||1)*(r||j.BucketBlockSize),C=Math.ceil((E.bucketSizeFactor||1)*(o||j.BucketSize)),I=j.computeBucketsForUncompressedSplatArray(M,y,C),P=I.fullBuckets.length,U=I.partiallyFullBuckets.map(Me=>Me.splats.length),O=U.length,k=[...I.fullBuckets,...I.partiallyFullBuckets],z=M.splats.length*f,W=O*4,H=n>=1?k.length*j.BucketStorageSizeBytes+W:0,$=z+H,X=new ArrayBuffer($),Z=d/(y*.5),pe=new L;let ge=0;for(let Me=0;Me<k.length;Me++){const Y=k[Me];pe.fromArray(Y.center);for(let oe=0;oe<Y.splats.length;oe++){let he=Y.splats[oe];const Fe=M.splats[he],ye=H+ge*f;j.writeSplatDataToSectionBuffer(Fe,X,ye,n,l,pe,Z,d,c,u),ge++}}if(p+=ge,n>=1){const Me=new Uint32Array(X,0,U.length*4);for(let oe=0;oe<U.length;oe++)Me[oe]=U[oe];const Y=new Float32Array(X,W,k.length*j.BucketStorageSizeFloats);for(let oe=0;oe<k.length;oe++){const he=k[oe],Fe=oe*3;Y[Fe]=he.center[0],Y[Fe+1]=he.center[1],Y[Fe+2]=he.center[2]}}h.push(X);const be=new ArrayBuffer(j.SectionHeaderSizeBytes);j.writeSectionHeaderToBuffer({maxSplatCount:ge,splatCount:ge,bucketSize:C,bucketCount:k.length,bucketBlockSize:y,compressionScaleRange:d,storageSizeBytes:$,fullBucketCount:P,partiallyFilledBucketCount:O,sphericalHarmonicsDegree:l},n,be,0),g.push(be)}let x=0;for(let v of h)x+=v.byteLength;const m=j.HeaderSizeBytes+j.SectionHeaderSizeBytes*h.length+x,_=new ArrayBuffer(m);j.writeHeaderToBuffer({versionMajor:0,versionMinor:1,maxSectionCount:h.length,sectionCount:h.length,maxSplatCount:p,splatCount:p,compressionLevel:n,sceneCenter:s,minSphericalHarmonicsCoeff:c,maxSphericalHarmonicsCoeff:u},_);let A=j.HeaderSizeBytes;for(let v of g)new Uint8Array(_,A,j.SectionHeaderSizeBytes).set(new Uint8Array(v)),A+=j.SectionHeaderSizeBytes;for(let v of h)new Uint8Array(_,A,v.byteLength).set(new Uint8Array(v)),A+=v.byteLength;return new j(_)}static computeBucketsForUncompressedSplatArray(e,t,n){let s=e.splatCount;const r=t/2,o=new L,a=new L;for(let p=0;p<s;p++){const x=e.splats[p],m=[x[Ee.OFFSET.X],x[Ee.OFFSET.Y],x[Ee.OFFSET.Z]];(p===0||m[0]<o.x)&&(o.x=m[0]),(p===0||m[0]>a.x)&&(a.x=m[0]),(p===0||m[1]<o.y)&&(o.y=m[1]),(p===0||m[1]>a.y)&&(a.y=m[1]),(p===0||m[2]<o.z)&&(o.z=m[2]),(p===0||m[2]>a.z)&&(a.z=m[2])}const l=new L().copy(a).sub(o),c=Math.ceil(l.y/t),u=Math.ceil(l.z/t),f=new L,d=[],h={};for(let p=0;p<s;p++){const x=e.splats[p],m=[x[Ee.OFFSET.X],x[Ee.OFFSET.Y],x[Ee.OFFSET.Z]],_=Math.floor((m[0]-o.x)/t),A=Math.floor((m[1]-o.y)/t),S=Math.floor((m[2]-o.z)/t);f.x=_*t+o.x+r,f.y=A*t+o.y+r,f.z=S*t+o.z+r;const v=_*(c*u)+A*u+S;let b=h[v];b||(h[v]=b={splats:[],center:f.toArray()}),b.splats.push(p),b.splats.length>=n&&(d.push(b),h[v]=null)}const g=[];for(let p in h)if(h.hasOwnProperty(p)){const x=h[p];x&&g.push(x)}return{fullBuckets:d,partiallyFullBuckets:g}}static preallocateUncompressed(e,t){const n=j.CompressionLevels[0].SphericalHarmonicsDegrees[t],s=j.HeaderSizeBytes+j.SectionHeaderSizeBytes,r=s+n.BytesPerSplat*e,o=new ArrayBuffer(r);return j.writeHeaderToBuffer({versionMajor:j.CurrentMajorVersion,versionMinor:j.CurrentMinorVersion,maxSectionCount:1,sectionCount:1,maxSplatCount:e,splatCount:e,compressionLevel:0,sceneCenter:new L},o),j.writeSectionHeaderToBuffer({maxSplatCount:e,splatCount:e,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:t},0,o,j.HeaderSizeBytes),{splatBuffer:new j(o,!0),splatBufferDataOffsetBytes:s}}}const bp=new Uint8Array([112,108,121,10]),Mp=new Uint8Array([10,101,110,100,95,104,101,97,100,101,114,10]),Kc="end_header",jc=new Map([["char",Int8Array],["uchar",Uint8Array],["short",Int16Array],["ushort",Uint16Array],["int",Int32Array],["uint",Uint32Array],["float",Float32Array],["double",Float64Array]]),Ri=(i,e)=>{const t=(1<<e)-1;return(i&t)/t},Cp=(i,e)=>{i.x=Ri(e>>>21,11),i.y=Ri(e>>>11,10),i.z=Ri(e,11)},lT=(i,e)=>{i.x=Ri(e>>>24,8),i.y=Ri(e>>>16,8),i.z=Ri(e>>>8,8),i.w=Ri(e,8)},cT=(i,e)=>{const t=1/(Math.sqrt(2)*.5),n=(Ri(e>>>20,10)-.5)*t,s=(Ri(e>>>10,10)-.5)*t,r=(Ri(e,10)-.5)*t,o=Math.sqrt(1-(n*n+s*s+r*r));switch(e>>>30){case 0:i.set(o,n,s,r);break;case 1:i.set(n,o,s,r);break;case 2:i.set(n,s,o,r);break;case 3:i.set(n,s,r,o);break}},Xi=(i,e,t)=>i*(1-t)+e*t,It=(i,e)=>i.properties.find(t=>t.name===e&&t.storage)?.storage;class st{static decodeHeaderText(e){let t,n,s,r;const o=e.split(`
`).filter(f=>!f.startsWith("comment "));let a=0,l=!1;for(let f=1;f<o.length;++f){const d=o[f].split(" ");switch(d[0]){case"format":if(d[1]!=="binary_little_endian")throw new Error("Unsupported ply format");break;case"element":t={name:d[1],count:parseInt(d[2],10),properties:[],storageSizeBytes:0},t.name==="chunk"?n=t:t.name==="vertex"?s=t:t.name==="sh"&&(r=t);break;case"property":{if(!jc.has(d[1]))throw new Error(`Unrecognized property data type '${d[1]}' in ply header`);const h=jc.get(d[1]),g=h.BYTES_PER_ELEMENT*t.count;t.name==="vertex"&&(a+=h.BYTES_PER_ELEMENT),t.properties.push({type:d[1],name:d[2],storage:null,byteSize:h.BYTES_PER_ELEMENT,storageSizeByes:g}),t.storageSizeBytes+=g;break}case Kc:l=!0;break;default:throw new Error(`Unrecognized header value '${d[0]}' in ply header`)}if(l)break}let c=0,u=0;return r&&(u=r.properties.length,r.properties.length>=45?c=3:r.properties.length>=24?c=2:r.properties.length>=9&&(c=1)),{chunkElement:n,vertexElement:s,shElement:r,bytesPerSplat:a,headerSizeBytes:e.indexOf(Kc)+Kc.length+1,sphericalHarmonicsDegree:c,sphericalHarmonicsPerSplat:u}}static decodeHeader(e){const t=(h,g)=>{const p=h.length-g.length;let x,m;for(x=0;x<=p;++x){for(m=0;m<g.length&&h[x+m]===g[m];++m);if(m===g.length)return x}return-1},n=(h,g)=>{if(h.length<g.length)return!1;for(let p=0;p<g.length;++p)if(h[p]!==g[p])return!1;return!0};let s=new Uint8Array(e),r;if(s.length>=bp.length&&!n(s,bp))throw new Error("Invalid PLY header");if(r=t(s,Mp),r===-1)throw new Error("End of PLY header not found");const o=new TextDecoder("ascii").decode(s.slice(0,r)),{chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f,bytesPerSplat:d}=st.decodeHeaderText(o);return{headerSizeBytes:r+Mp.length,bytesPerSplat:d,chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f}}static readElementData(e,t,n,s,r,o=null){let a=t instanceof DataView?t:new DataView(t);s=s||0,r=r||e.count-1;for(let l=s;l<=r;++l)for(let c=0;c<e.properties.length;++c){const u=e.properties[c],f=jc.get(u.type),d=f.BYTES_PER_ELEMENT*e.count;if((!u.storage||u.storage.byteLength<d)&&(!o||o(u.name))&&(u.storage=new f(e.count)),u.storage)switch(u.type){case"char":u.storage[l]=a.getInt8(n);break;case"uchar":u.storage[l]=a.getUint8(n);break;case"short":u.storage[l]=a.getInt16(n,!0);break;case"ushort":u.storage[l]=a.getUint16(n,!0);break;case"int":u.storage[l]=a.getInt32(n,!0);break;case"uint":u.storage[l]=a.getUint32(n,!0);break;case"float":u.storage[l]=a.getFloat32(n,!0);break;case"double":u.storage[l]=a.getFloat64(n,!0);break}n+=u.byteSize}return n}static readPly(e,t=null){const n=st.decodeHeader(e);let s=st.readElementData(n.chunkElement,e,n.headerSizeBytes,null,null,t);return s=st.readElementData(n.vertexElement,e,s,null,null,t),st.readElementData(n.shElement,e,s,null,null,t),{chunkElement:n.chunkElement,vertexElement:n.vertexElement,shElement:n.shElement,sphericalHarmonicsDegree:n.sphericalHarmonicsDegree,sphericalHarmonicsPerSplat:n.sphericalHarmonicsPerSplat}}static getElementStorageArrays(e,t,n){const s={};if(t){const r=It(e,"min_r"),o=It(e,"min_g"),a=It(e,"min_b"),l=It(e,"max_r"),c=It(e,"max_g"),u=It(e,"max_b"),f=It(e,"min_x"),d=It(e,"min_y"),h=It(e,"min_z"),g=It(e,"max_x"),p=It(e,"max_y"),x=It(e,"max_z"),m=It(e,"min_scale_x"),_=It(e,"min_scale_y"),A=It(e,"min_scale_z"),S=It(e,"max_scale_x"),v=It(e,"max_scale_y"),b=It(e,"max_scale_z"),M=It(t,"packed_position"),E=It(t,"packed_rotation"),y=It(t,"packed_scale"),C=It(t,"packed_color");s.colorExtremes={minR:r,maxR:l,minG:o,maxG:c,minB:a,maxB:u},s.positionExtremes={minX:f,maxX:g,minY:d,maxY:p,minZ:h,maxZ:x},s.scaleExtremes={minScaleX:m,maxScaleX:S,minScaleY:_,maxScaleY:v,minScaleZ:A,maxScaleZ:b},s.position=M,s.rotation=E,s.scale=y,s.color=C}if(n){const r={};for(let o=0;o<45;o++){const a=`f_rest_${o}`,l=It(n,a);if(l)r[a]=l;else break}s.sh=r}return s}static decompressBaseSplat=(function(){const e=new L,t=new bt,n=new L,s=new Et,r=Ee.OFFSET;return function(o,a,l,c,u,f,d,h,g,p){p=p||Ee.createSplat();const x=Math.floor((a+o)/256);return Cp(e,l[o]),cT(t,d[o]),Cp(n,u[o]),lT(s,g[o]),p[r.X]=Xi(c.minX[x],c.maxX[x],e.x),p[r.Y]=Xi(c.minY[x],c.maxY[x],e.y),p[r.Z]=Xi(c.minZ[x],c.maxZ[x],e.z),p[r.ROTATION0]=t.x,p[r.ROTATION1]=t.y,p[r.ROTATION2]=t.z,p[r.ROTATION3]=t.w,p[r.SCALE0]=Math.exp(Xi(f.minScaleX[x],f.maxScaleX[x],n.x)),p[r.SCALE1]=Math.exp(Xi(f.minScaleY[x],f.maxScaleY[x],n.y)),p[r.SCALE2]=Math.exp(Xi(f.minScaleZ[x],f.maxScaleZ[x],n.z)),h.minR&&h.maxR?p[r.FDC0]=Ct(Math.round(Xi(h.minR[x],h.maxR[x],s.x)*255),0,255):p[r.FDC0]=Ct(Math.floor(s.x*255),0,255),h.minG&&h.maxG?p[r.FDC1]=Ct(Math.round(Xi(h.minG[x],h.maxG[x],s.y)*255),0,255):p[r.FDC1]=Ct(Math.floor(s.y*255),0,255),h.minB&&h.maxB?p[r.FDC2]=Ct(Math.round(Xi(h.minB[x],h.maxB[x],s.z)*255),0,255):p[r.FDC2]=Ct(Math.floor(s.z*255),0,255),p[r.OPACITY]=Ct(Math.floor(s.w*255),0,255),p}})();static decompressSphericalHarmonics=(function(){const e=[0,3,8,15],t=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(n,s,r,o,a){a=a||Ee.createSplat();let l=e[r],c=e[o];for(let u=0;u<3;++u)for(let f=0;f<15;++f){const d=t[u*15+f];f<l&&f<c&&(a[Ee.OFFSET.FRC0+d]=s[u*c+f][n]*(8/255)-4)}return a}})();static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l,c=null){st.readElementData(t,o,0,n,s,c);const u=j.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,{positionExtremes:f,scaleExtremes:d,colorExtremes:h,position:g,rotation:p,scale:x,color:m}=st.getElementStorageArrays(e,t),_=Ee.createSplat();for(let A=n;A<=s;++A){st.decompressBaseSplat(A,r,g,f,x,d,p,h,m,_);const S=A*u+l;j.writeSplatDataToSectionBuffer(_,a,S,0,0)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a,l=null){st.readElementData(t,o,0,n,s,l);const{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:d,rotation:h,scale:g,color:p}=st.getElementStorageArrays(e,t);for(let x=n;x<=s;++x){const m=Ee.createSplat();st.decompressBaseSplat(x,r,d,c,g,u,h,f,p,m),a.addSplat(m)}}static parseSphericalHarmonicsToUncompressedSplatArraySection(e,t,n,s,r,o,a,l,c,u=null){st.readElementData(t,r,o,n,s,u);const{sh:f}=st.getElementStorageArrays(e,void 0,t),d=Object.values(f);for(let h=n;h<=s;++h)st.decompressSphericalHarmonics(h,d,a,l,c.splats[h])}static parseToUncompressedSplatArray(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=st.readPly(e);t=Math.min(t,o);const a=new Ee(t),{positionExtremes:l,scaleExtremes:c,colorExtremes:u,position:f,rotation:d,scale:h,color:g}=st.getElementStorageArrays(n,s);let p;if(t>0){const{sh:x}=st.getElementStorageArrays(n,void 0,r);p=Object.values(x)}for(let x=0;x<s.count;++x){a.addDefaultSplat();const m=a.getSplat(a.splatCount-1);st.decompressBaseSplat(x,0,f,l,h,c,d,u,g,m),t>0&&st.decompressSphericalHarmonics(x,p,t,o,m)}return a}static parseToUncompressedSplatBuffer(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=st.readPly(e);t=Math.min(t,o);const{splatBuffer:a,splatBufferDataOffsetBytes:l}=j.preallocateUncompressed(s.count,t),{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:d,rotation:h,scale:g,color:p}=st.getElementStorageArrays(n,s);let x;if(t>0){const{sh:A}=st.getElementStorageArrays(n,void 0,r);x=Object.values(A)}const m=j.CompressionLevels[0].SphericalHarmonicsDegrees[t].BytesPerSplat,_=Ee.createSplat(t);for(let A=0;A<s.count;++A){st.decompressBaseSplat(A,0,d,c,g,u,h,f,p,_),t>0&&st.decompressSphericalHarmonics(A,x,t,o,_);const S=A*m+l;j.writeSplatDataToSectionBuffer(_,a.bufferData,S,0,t)}return a}}const hn={INRIAV1:0,INRIAV2:1,PlayCanvasCompressed:2},[ng,ld,cd,ud,fd,dd,hd]=[0,1,2,3,4,5,6],Tp={double:ng,int:ld,uint:cd,float:ud,short:fd,ushort:dd,uchar:hd},uT={[ng]:8,[ld]:4,[cd]:4,[ud]:4,[fd]:2,[dd]:2,[hd]:1};class ot{static HeaderEndToken="end_header";static decodeSectionHeader(e,t,n=0){const s=[];let r=!1,o=-1,a=0,l=!1,c=null;const u=[],f=[],d=[],h={};for(let m=n;m<e.length;m++){const _=e[m].trim();if(_.startsWith("element"))if(r){o--;break}else{r=!0,n=m,o=m;const A=_.split(" ");let S=0;for(let v of A){const b=v.trim();b.length>0&&(S++,S===2?c=b:S===3&&(a=parseInt(b)))}}else if(_.startsWith("property")){const A=_.match(/(\w+)\s+(\w+)\s+(\w+)/);if(A){const S=A[2],v=A[3];d.push(v);const b=t[v];h[v]=S;const M=Tp[S];b!==void 0&&(u.push(b),f[b]=M)}}if(_===ot.HeaderEndToken){l=!0;break}r&&(s.push(_),o++)}const g=[];let p=0;for(let m of d){const _=h[m];if(h.hasOwnProperty(m)){const A=t[m];A!==void 0&&(g[A]=p)}p+=uT[Tp[_]]}const x=ot.decodeSphericalHarmonicsFromSectionHeader(d,t);return{headerLines:s,headerStartLine:n,headerEndLine:o,fieldTypes:f,fieldIds:u,fieldOffsets:g,bytesPerVertex:p,vertexCount:a,dataSizeBytes:p*a,endOfHeader:l,sectionName:c,sphericalHarmonicsDegree:x.degree,sphericalHarmonicsCoefficientsPerChannel:x.coefficientsPerChannel,sphericalHarmonicsDegree1Fields:x.degree1Fields,sphericalHarmonicsDegree2Fields:x.degree2Fields}}static decodeSphericalHarmonicsFromSectionHeader(e,t){let n=0,s=0;for(let l of e)l.startsWith("f_rest")&&n++;s=n/3;let r=0;s>=3&&(r=1),s>=8&&(r=2);let o=[],a=[];for(let l=0;l<3;l++){if(r>=1)for(let c=0;c<3;c++)o.push(t["f_rest_"+(c+s*l)]);if(r>=2)for(let c=0;c<5;c++)a.push(t["f_rest_"+(c+s*l+3)])}return{degree:r,coefficientsPerChannel:s,degree1Fields:o,degree2Fields:a}}static getHeaderSectionNames(e){const t=[];for(let n of e)if(n.startsWith("element")){const s=n.split(" ");let r=0;for(let o of s){const a=o.trim();a.length>0&&(r++,r===2&&t.push(a))}}return t}static checkTextForEndHeader(e){return!!e.includes(ot.HeaderEndToken)}static checkBufferForEndHeader(e,t,n,s){const r=new Uint8Array(e,Math.max(0,t-n),n),o=s.decode(r);return ot.checkTextForEndHeader(o)}static extractHeaderFromBufferToText(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,ot.checkBufferForEndHeader(e,n,r*2,t))break}return s}static readHeaderFromBuffer(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,ot.checkBufferForEndHeader(e,n,r*2,t))break}return s}static convertHeaderTextToLines(e){const t=e.split(`
`),n=[];for(let s=0;s<t.length;s++){const r=t[s].trim();if(n.push(r),r===ot.HeaderEndToken)break}return n}static determineHeaderFormatFromHeaderText(e){const t=ot.convertHeaderTextToLines(e);let n=hn.INRIAV1;for(let s=0;s<t.length;s++){const r=t[s].trim();if(r.startsWith("element chunk")||r.match(/[A-Za-z]*packed_[A-Za-z]*/))n=hn.PlayCanvasCompressed;else if(r.startsWith("element codebook_centers"))n=hn.INRIAV2;else if(r===ot.HeaderEndToken)break}return n}static determineHeaderFormatFromPlyBuffer(e){const t=ot.extractHeaderFromBufferToText(e);return ot.determineHeaderFormatFromHeaderText(t)}static readVertex(e,t,n,s,r,o,a=!0){const l=n*t.bytesPerVertex+s,c=t.fieldOffsets,u=t.fieldTypes;for(let f of r){const d=u[f];d===ud?o[f]=e.getFloat32(l+c[f],!0):d===fd?o[f]=e.getInt16(l+c[f],!0):d===dd?o[f]=e.getUint16(l+c[f],!0):d===ld?o[f]=e.getInt32(l+c[f],!0):d===cd?o[f]=e.getUint32(l+c[f],!0):d===hd&&(a?o[f]=e.getUint8(l+c[f])/255:o[f]=e.getUint8(l+c[f]))}}}const ig=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0"],fT=ig.map((i,e)=>e),[Ep,dT,hT,pT,mT,gT,xT,_T,AT,ST,wp,vT,yT,Rp,Ip,bT,MT,CT]=fT;class Zt{static decodeHeaderLines(e){let t=0;e.forEach(u=>{u.includes("f_rest_")&&t++});let n=0;t>=45?n=45:t>=24?n=24:t>=9&&(n=9);let r=Array.from(Array(Math.max(n-1,0))).map((u,f)=>`f_rest_${f+1}`);const o=[...ig,...r],a=o.map((u,f)=>f),l=a.reduce((u,f)=>(u[o[f]]=f,u),{}),c=ot.decodeSectionHeader(e,l,0);return c.splatCount=c.vertexCount,c.bytesPerSplat=c.bytesPerVertex,c.fieldsToReadIndexes=a,c}static decodeHeaderText(e){const t=ot.convertHeaderTextToLines(e),n=Zt.decodeHeaderLines(t);return n.headerText=e,n.headerSizeBytes=e.indexOf(ot.HeaderEndToken)+ot.HeaderEndToken.length+1,n}static decodeHeaderFromBuffer(e){const t=ot.readHeaderFromBuffer(e);return Zt.decodeHeaderText(t)}static findSplatData(e,t){return new DataView(e,t.headerSizeBytes)}static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l=0){l=Math.min(l,e.sphericalHarmonicsDegree);const c=j.CompressionLevels[0].SphericalHarmonicsDegrees[l].BytesPerSplat;for(let u=t;u<=n;u++){const f=Zt.parseToUncompressedSplat(s,u,e,r,l),d=u*c+a;j.writeSplatDataToSectionBuffer(f,o,d,0,l)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a=0){a=Math.min(a,e.sphericalHarmonicsDegree);for(let l=t;l<=n;l++){const c=Zt.parseToUncompressedSplat(s,l,e,r,a);o.addSplat(c)}}static decodeSectionSplatData(e,t,n,s,r=!0){if(s=Math.min(s,n.sphericalHarmonicsDegree),r){const o=new Ee(s);for(let a=0;a<t;a++){const l=Zt.parseToUncompressedSplat(e,a,n,0,s);o.addSplat(l)}return o}else{const{splatBuffer:o,splatBufferDataOffsetBytes:a}=j.preallocateUncompressed(t,s);return Zt.parseToUncompressedSplatBufferSection(n,0,t-1,e,0,o.bufferData,a,s),o}}static parseToUncompressedSplat=(function(){let e=[];const t=new bt,n=Ee.OFFSET.X,s=Ee.OFFSET.Y,r=Ee.OFFSET.Z,o=Ee.OFFSET.SCALE0,a=Ee.OFFSET.SCALE1,l=Ee.OFFSET.SCALE2,c=Ee.OFFSET.ROTATION0,u=Ee.OFFSET.ROTATION1,f=Ee.OFFSET.ROTATION2,d=Ee.OFFSET.ROTATION3,h=Ee.OFFSET.FDC0,g=Ee.OFFSET.FDC1,p=Ee.OFFSET.FDC2,x=Ee.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=Ee.OFFSET.FRC0+_;return function(_,A,S,v=0,b=0){b=Math.min(b,S.sphericalHarmonicsDegree),Zt.readSplat(_,S,A,v,e);const M=Ee.createSplat(b);if(e[Ep]!==void 0?(M[o]=Math.exp(e[Ep]),M[a]=Math.exp(e[dT]),M[l]=Math.exp(e[hT])):(M[o]=.01,M[a]=.01,M[l]=.01),e[wp]!==void 0){const E=.28209479177387814;M[h]=(.5+E*e[wp])*255,M[g]=(.5+E*e[vT])*255,M[p]=(.5+E*e[yT])*255}else e[Ip]!==void 0?(M[h]=e[Ip]*255,M[g]=e[bT]*255,M[p]=e[MT]*255):(M[h]=0,M[g]=0,M[p]=0);if(e[Rp]!==void 0&&(M[x]=1/(1+Math.exp(-e[Rp]))*255),M[h]=Ct(Math.floor(M[h]),0,255),M[g]=Ct(Math.floor(M[g]),0,255),M[p]=Ct(Math.floor(M[p]),0,255),M[x]=Ct(Math.floor(M[x]),0,255),b>=1&&e[CT]!==void 0){for(let E=0;E<9;E++)M[m[E]]=e[S.sphericalHarmonicsDegree1Fields[E]];if(b>=2)for(let E=0;E<15;E++)M[m[9+E]]=e[S.sphericalHarmonicsDegree2Fields[E]]}return t.set(e[pT],e[mT],e[gT],e[xT]),t.normalize(),M[c]=t.x,M[u]=t.y,M[f]=t.z,M[d]=t.w,M[n]=e[_T],M[s]=e[AT],M[r]=e[ST],M}})();static readSplat(e,t,n,s,r){return ot.readVertex(e,t,n,s,t.fieldsToReadIndexes,r,!0)}static parseToUncompressedSplatArray(e,t=0){const{header:n,splatCount:s,splatData:r}=Dp(e);return Zt.decodeSectionSplatData(r,s,n,t,!0)}static parseToUncompressedSplatBuffer(e,t=0){const{header:n,splatCount:s,splatData:r}=Dp(e);return Zt.decodeSectionSplatData(r,s,n,t,!1)}}function Dp(i){const e=Zt.decodeHeaderFromBuffer(i),t=e.splatCount,n=Zt.findSplatData(i,e);return{header:e,splatCount:t,splatData:n}}const sg=["features_dc","features_rest_0","features_rest_1","features_rest_2","features_rest_3","features_rest_4","features_rest_5","features_rest_6","features_rest_7","features_rest_8","features_rest_9","features_rest_10","features_rest_11","features_rest_12","features_rest_13","features_rest_14","opacity","scaling","rotation_re","rotation_im"],Ja=sg.map((i,e)=>e),[el,TT,ET,Pp,tl,wT,$c]=[0,1,4,16,17,18,19],rg=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0","f_rest_1","f_rest_2","f_rest_3","f_rest_4","f_rest_5","f_rest_6","f_rest_7","f_rest_8","f_rest_9","f_rest_10","f_rest_11","f_rest_12","f_rest_13","f_rest_14","f_rest_15","f_rest_16","f_rest_17","f_rest_18","f_rest_19","f_rest_20","f_rest_21","f_rest_22","f_rest_23","f_rest_24","f_rest_25","f_rest_26","f_rest_27","f_rest_28","f_rest_29","f_rest_30","f_rest_31","f_rest_32","f_rest_33","f_rest_34","f_rest_35","f_rest_36","f_rest_37","f_rest_38","f_rest_39","f_rest_40","f_rest_41","f_rest_42","f_rest_43","f_rest_44","f_rest_45"],ff=rg.map((i,e)=>e),[Fp,RT,IT,DT,PT,FT,LT,BT,UT,OT,df,og,ag,Lp]=ff,Bp=df,NT=og,zT=ag,nl=i=>{const e=(31744&i)>>10,t=1023&i;return(i>>15?-1:1)*(e?e===31?t?NaN:1/0:Math.pow(2,e-15)*(1+t/1024):t/1024*6103515625e-14)};class zn{static decodeSectionHeadersFromHeaderLines(e){const t=ff.reduce((u,f)=>(u[rg[f]]=f,u),{}),n=Ja.reduce((u,f)=>(u[sg[f]]=f,u),{}),s=ot.getHeaderSectionNames(e);let r;for(let u=0;u<s.length;u++)s[u]==="codebook_centers"&&(r=u);let o=0,a=!1;const l=[];let c=0;for(;!a;){let u;c===r?u=ot.decodeSectionHeader(e,n,o):u=ot.decodeSectionHeader(e,t,o),a=u.endOfHeader,o=u.headerEndLine+1,a||(u.splatCount=u.vertexCount,u.bytesPerSplat=u.bytesPerVertex),l.push(u),c++}return l}static decodeSectionHeadersFromHeaderText(e){const t=ot.convertHeaderTextToLines(e);return zn.decodeSectionHeadersFromHeaderLines(t)}static getSplatCountFromSectionHeaders(e){let t=0;for(let n of e)n.sectionName!=="codebook_centers"&&(t+=n.vertexCount);return t}static decodeHeaderFromHeaderText(e){const t=e.indexOf(ot.HeaderEndToken)+ot.HeaderEndToken.length+1,n=zn.decodeSectionHeadersFromHeaderText(e),s=zn.getSplatCountFromSectionHeaders(n);return{headerSizeBytes:t,sectionHeaders:n,splatCount:s}}static decodeHeaderFromBuffer(e){const t=ot.readHeaderFromBuffer(e);return zn.decodeHeaderFromHeaderText(t)}static findVertexData(e,t,n){let s=t.headerSizeBytes;for(let r=0;r<n&&r<t.sectionHeaders.length;r++){const o=t.sectionHeaders[r];s+=o.dataSizeBytes}return new DataView(e,s,t.sectionHeaders[n].dataSizeBytes)}static decodeCodeBook(e,t){const n=[],s=[];for(let r=0;r<t.vertexCount;r++){ot.readVertex(e,t,r,0,Ja,n);for(let o of Ja){const a=Ja[o];let l=s[a];l||(s[a]=l=[]),l.push(n[o])}}for(let r=0;r<s.length;r++){const o=s[r],a=.28209479177387814;for(let l=0;l<o.length;l++){const c=nl(o[l]);r===Pp?o[l]=Math.round(1/(1+Math.exp(-c))*255):r===el?o[l]=Math.round((.5+a*c)*255):r===tl?o[l]=Math.exp(c):o[l]=c}}return s}static decodeSectionSplatData(e,t,n,s,r){r=Math.min(r,n.sphericalHarmonicsDegree);const o=new Ee(r);for(let a=0;a<t;a++){const l=zn.parseToUncompressedSplat(e,a,n,s,0,r);o.addSplat(l)}return o}static parseToUncompressedSplat=(function(){let e=[];const t=new bt,n=Ee.OFFSET.X,s=Ee.OFFSET.Y,r=Ee.OFFSET.Z,o=Ee.OFFSET.SCALE0,a=Ee.OFFSET.SCALE1,l=Ee.OFFSET.SCALE2,c=Ee.OFFSET.ROTATION0,u=Ee.OFFSET.ROTATION1,f=Ee.OFFSET.ROTATION2,d=Ee.OFFSET.ROTATION3,h=Ee.OFFSET.FDC0,g=Ee.OFFSET.FDC1,p=Ee.OFFSET.FDC2,x=Ee.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=Ee.OFFSET.FRC0+_;return function(_,A,S,v,b=0,M=0){M=Math.min(M,S.sphericalHarmonicsDegree),zn.readSplat(_,S,A,b,e);const E=Ee.createSplat(M);if(e[Fp]!==void 0?(E[o]=v[tl][e[Fp]],E[a]=v[tl][e[RT]],E[l]=v[tl][e[IT]]):(E[o]=.01,E[a]=.01,E[l]=.01),e[df]!==void 0?(E[h]=v[el][e[df]],E[g]=v[el][e[og]],E[p]=v[el][e[ag]]):e[Bp]!==void 0?(E[h]=e[Bp]*255,E[g]=e[NT]*255,E[p]=e[zT]*255):(E[h]=0,E[g]=0,E[p]=0),e[Lp]!==void 0&&(E[x]=v[Pp][e[Lp]]),E[h]=Ct(Math.floor(E[h]),0,255),E[g]=Ct(Math.floor(E[g]),0,255),E[p]=Ct(Math.floor(E[p]),0,255),E[x]=Ct(Math.floor(E[x]),0,255),M>=1&&S.sphericalHarmonicsDegree>=1){for(let U=0;U<9;U++){const O=v[TT+U%3];E[m[U]]=O[e[S.sphericalHarmonicsDegree1Fields[U]]]}if(M>=2&&S.sphericalHarmonicsDegree>=2)for(let U=0;U<15;U++){const O=v[ET+U%5];E[m[9+U]]=O[e[S.sphericalHarmonicsDegree2Fields[U]]]}}const y=v[wT][e[DT]],C=v[$c][e[PT]],I=v[$c][e[FT]],P=v[$c][e[LT]];return t.set(y,C,I,P),t.normalize(),E[c]=t.x,E[u]=t.y,E[f]=t.z,E[d]=t.w,E[n]=nl(e[BT]),E[s]=nl(e[UT]),E[r]=nl(e[OT]),E}})();static readSplat(e,t,n,s,r){return ot.readVertex(e,t,n,s,ff,r,!1)}static parseToUncompressedSplatArray(e,t=0){const n=[],s=zn.decodeHeaderFromBuffer(e,t);let r;for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName==="codebook_centers"){const c=zn.findVertexData(e,s,a);r=zn.decodeCodeBook(c,l)}}for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName!=="codebook_centers"){const c=l.vertexCount,u=zn.findVertexData(e,s,a),f=zn.decodeSectionSplatData(u,c,l,r,t);n.push(f)}}const o=new Ee(t);for(let a of n)for(let l of a.splats)o.addSplat(l);return o}}class Up{static parseToUncompressedSplatArray(e,t=0){const n=ot.determineHeaderFormatFromPlyBuffer(e);if(n===hn.PlayCanvasCompressed)return st.parseToUncompressedSplatArray(e,t);if(n===hn.INRIAV1)return Zt.parseToUncompressedSplatArray(e,t);if(n===hn.INRIAV2)return zn.parseToUncompressedSplatArray(e,t)}static parseToUncompressedSplatBuffer(e,t=0){const n=ot.determineHeaderFormatFromPlyBuffer(e);if(n===hn.PlayCanvasCompressed)return st.parseToUncompressedSplatBuffer(e,t);if(n===hn.INRIAV1)return Zt.parseToUncompressedSplatBuffer(e,t);if(n===hn.INRIAV2)throw new Error("parseToUncompressedSplatBuffer() is not implemented for INRIA V2 PLY files")}}class pd{constructor(e,t,n,s){this.sectionCount=e,this.sectionFilters=t,this.groupingParameters=n,this.partitionGenerator=s}partitionUncompressedSplatArray(e){let t,n,s;if(this.partitionGenerator){const o=this.partitionGenerator(e);t=o.groupingParameters,n=o.sectionCount,s=o.sectionFilters}else t=this.groupingParameters,n=this.sectionCount,s=this.sectionFilters;const r=[];for(let o=0;o<n;o++){const a=new Ee(e.sphericalHarmonicsDegree),l=s[o];for(let c=0;c<e.splatCount;c++)l(c)&&a.addSplat(e.splats[c]);r.push(a)}return{splatArrays:r,parameters:t}}static getStandardPartitioner(e=0,t=new L,n=j.BucketBlockSize,s=j.BucketSize){const r=o=>{const a=Ee.OFFSET.X,l=Ee.OFFSET.Y,c=Ee.OFFSET.Z;e<=0&&(e=o.splatCount);const u=new L,f=.5,d=m=>{m.x=Math.floor(m.x/f)*f,m.y=Math.floor(m.y/f)*f,m.z=Math.floor(m.z/f)*f};o.splats.forEach(m=>{u.set(m[a],m[l],m[c]).sub(t),d(u),m.centerDist=u.lengthSq()}),o.splats.sort((m,_)=>{let A=m.centerDist,S=_.centerDist;return A>S?1:-1});const h=[],g=[];e=Math.min(o.splatCount,e);const p=Math.ceil(o.splatCount/e);let x=0;for(let m=0;m<p;m++){let _=x;h.push(A=>A>=_&&A<_+e),g.push({blocksSize:n,bucketSize:s}),x+=e}return{sectionCount:h.length,sectionFilters:h,groupingParameters:g}};return new pd(void 0,void 0,void 0,r)}}class ba{constructor(e,t,n,s,r,o,a){this.splatPartitioner=e,this.alphaRemovalThreshold=t,this.compressionLevel=n,this.sectionSize=s,this.sceneCenter=r?new L().copy(r):void 0,this.blockSize=o,this.bucketSize=a}generateFromUncompressedSplatArray(e){const t=this.splatPartitioner.partitionUncompressedSplatArray(e);return j.generateFromUncompressedSplatArrays(t.splatArrays,this.alphaRemovalThreshold,this.compressionLevel,this.sceneCenter,this.blockSize,this.bucketSize,t.parameters)}static getStandardGenerator(e=1,t=1,n=0,s=new L,r=j.BucketBlockSize,o=j.BucketSize){const a=pd.getStandardPartitioner(n,s,r,o);return new ba(a,e,t,n,s,r,o)}}const Nt={Downloading:0,Processing:1,Done:2};class Il extends Error{constructor(e){super(e)}}const yt={ProgressiveToSplatBuffer:0,ProgressiveToSplatArray:1,DownloadBeforeProcessing:2};function Op(i,e){let t=0;for(let s of i)t+=s.sizeBytes;(!e||e.byteLength<t)&&(e=new ArrayBuffer(t));let n=0;for(let s of i)new Uint8Array(e,n,s.sizeBytes).set(s.data),n+=s.sizeBytes;return e}function Np(i,e,t,n,s,r,o,a){return e?ba.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):j.generateFromUncompressedSplatArrays([i],t,0,new L)}class md{static loadFromURL(e,t,n,s,r,o,a=!0,l=0,c,u,f,d,h){let g;!n&&!a?g=yt.DownloadBeforeProcessing:a?g=yt.ProgressiveToSplatArray:g=yt.ProgressiveToSplatBuffer;const p=pt.ProgressiveLoadSectionSize,x=j.HeaderSizeBytes+j.SectionHeaderSizeBytes,m=1;let _,A,S,v,b,M=0,E=0,y=0,C=!1,I=!1,P=!1;const U=sd();let O=0,k=0,z=0,W=0,H="",$=null,X=[],Z;const pe=new TextDecoder,ge=(be,Me,Y)=>{const oe=be>=100;if(Y&&(X.push({data:Y,sizeBytes:Y.byteLength,startBytes:z,endBytes:z+Y.byteLength}),z+=Y.byteLength),g===yt.DownloadBeforeProcessing)oe&&U.resolve(X);else{if(C){if(_===hn.PlayCanvasCompressed&&!I){const he=$.headerSizeBytes+$.chunkElement.storageSizeBytes;b=Op(X,b),b.byteLength>=he&&(st.readElementData($.chunkElement,b,$.headerSizeBytes),O=he,k=he,I=!0)}}else if(H+=pe.decode(Y),ot.checkTextForEndHeader(H)){if(_=ot.determineHeaderFormatFromHeaderText(H),_===hn.INRIAV1)$=Zt.decodeHeaderText(H),l=Math.min(l,$.sphericalHarmonicsDegree),M=$.splatCount,I=!0,W=$.headerSizeBytes+$.bytesPerSplat*M;else if(_===hn.PlayCanvasCompressed){if($=st.decodeHeaderText(H),l=Math.min(l,$.sphericalHarmonicsDegree),g===yt.ProgressiveToSplatBuffer&&l>0)throw new Il("PlyLoader.loadFromURL() -> Selected PLY format has spherical harmonics data that cannot be progressively loaded.");M=$.vertexElement.count,W=$.headerSizeBytes+$.bytesPerSplat*M+$.chunkElement.storageSizeBytes}else{if(g===yt.ProgressiveToSplatBuffer)throw new Il("PlyLoader.loadFromURL() -> Selected PLY format cannot be progressively loaded.");g=yt.DownloadBeforeProcessing;return}if(g===yt.ProgressiveToSplatBuffer){const he=j.CompressionLevels[0].SphericalHarmonicsDegrees[l],Fe=x+he.BytesPerSplat*M;S=new ArrayBuffer(Fe),j.writeHeaderToBuffer({versionMajor:j.CurrentMajorVersion,versionMinor:j.CurrentMinorVersion,maxSectionCount:m,sectionCount:m,maxSplatCount:M,splatCount:0,compressionLevel:0,sceneCenter:new L},S)}else Z=new Ee(l);O=$.headerSizeBytes,k=$.headerSizeBytes,C=!0}if(C&&I&&X.length>0&&(A=Op(X,A),z-O>p||z>=W&&!P||oe)){const Fe=P?$.sphericalHarmonicsPerSplat:$.bytesPerSplat,ke=(P?z:Math.min(W,z))-k,F=Math.floor(ke/Fe),B=F*Fe,q=z-k-B,w=k-X[0].startBytes,ne=new DataView(A,w,B);if(P)_===hn.PlayCanvasCompressed&&g===yt.ProgressiveToSplatArray&&(st.parseSphericalHarmonicsToUncompressedSplatArraySection($.chunkElement,$.shElement,y,y+F-1,ne,0,l,$.sphericalHarmonicsDegree,Z),y+=F);else{if(g===yt.ProgressiveToSplatBuffer){const re=j.CompressionLevels[0].SphericalHarmonicsDegrees[l],ue=E*re.BytesPerSplat+x;_===hn.PlayCanvasCompressed?st.parseToUncompressedSplatBufferSection($.chunkElement,$.vertexElement,0,F-1,E,ne,S,ue):Zt.parseToUncompressedSplatBufferSection($,0,F-1,ne,0,S,ue,l)}else _===hn.PlayCanvasCompressed?st.parseToUncompressedSplatArraySection($.chunkElement,$.vertexElement,0,F-1,E,ne,Z):Zt.parseToUncompressedSplatArraySection($,0,F-1,ne,0,Z,l);E+=F,g===yt.ProgressiveToSplatBuffer&&(v||(j.writeSectionHeaderToBuffer({maxSplatCount:M,splatCount:E,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:l},0,S,j.HeaderSizeBytes),v=new j(S,!1)),v.updateLoadedCounts(1,E)),z>=W&&(P=!0)}if(q===0)X=[];else{let re=[],ue=0;for(let ie=X.length-1;ie>=0;ie--){const fe=X[ie];if(ue+=fe.sizeBytes,re.unshift(fe),ue>=q)break}X=re}O+=p,k+=B}s&&v&&s(v,oe),oe&&(g===yt.ProgressiveToSplatBuffer?U.resolve(v):U.resolve(Z))}t&&t(be,Me,Nt.Downloading)};return t&&t(0,"0%",Nt.Downloading),ec(e,ge,!1,c).then(()=>(t&&t(0,"0%",Nt.Processing),U.promise.then(be=>{if(t&&t(100,"100%",Nt.Done),g===yt.DownloadBeforeProcessing){const Me=X.map(Y=>Y.data);return new Blob(Me).arrayBuffer().then(Y=>md.loadFromFileData(Y,r,o,a,l,u,f,d,h))}else return g===yt.ProgressiveToSplatBuffer?be:Gn(()=>Np(be,a,r,o,u,f,d,h))})))}static loadFromFileData(e,t,n,s,r=0,o,a,l,c){return s?Gn(()=>Up.parseToUncompressedSplatArray(e,r)).then(u=>Np(u,s,t,n,o,a,l,c)):Gn(()=>Up.parseToUncompressedSplatBuffer(e,r))}}const kT=i=>new ReadableStream({async start(e){e.enqueue(i),e.close()}});async function HT(i){try{const e=kT(i);if(!e)throw new Error("Failed to create stream from data");return await VT(e)}catch(e){throw console.error("Error decompressing gzipped data:",e),e}}async function VT(i){const e=i.pipeThrough(new DecompressionStream("gzip")),n=await new Response(e).arrayBuffer();return new Uint8Array(n)}const GT=1347635022,WT=1,XT=.15;function qT(i){const e=i>>15&1,t=i>>10&31,n=i&1023,s=e===1?-1:1;return t===0?s*Math.pow(2,-14)*n/1024:t===31?n!==0?NaN:s*(1/0):s*Math.pow(2,t-15)*(1+n/1024)}function QT(i){return(i-128)/128}function rr(i){switch(i){case 0:return 0;case 1:return 3;case 2:return 8;case 3:return 15;default:return console.error(`[SPZ: ERROR] Unsupported SH degree: ${i}`),0}}const YT=(function(){let i=[];const e=new bt,t=Ee.OFFSET.X,n=Ee.OFFSET.Y,s=Ee.OFFSET.Z,r=Ee.OFFSET.SCALE0,o=Ee.OFFSET.SCALE1,a=Ee.OFFSET.SCALE2,l=Ee.OFFSET.ROTATION0,c=Ee.OFFSET.ROTATION1,u=Ee.OFFSET.ROTATION2,f=Ee.OFFSET.ROTATION3,d=Ee.OFFSET.FDC0,h=Ee.OFFSET.FDC1,g=Ee.OFFSET.FDC2,p=Ee.OFFSET.OPACITY,x=[rr(0),rr(1),rr(2),rr(3)],m=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(_,A,S){S=Math.min(A,S);const v=Ee.createSplat(S);_.scale[0]!==void 0?(v[r]=_.scale[0],v[o]=_.scale[1],v[a]=_.scale[2]):(v[r]=.01,v[o]=.01,v[a]=.01),_.color[0]!==void 0?(v[d]=_.color[0],v[h]=_.color[1],v[g]=_.color[2]):i[RED]!==void 0?(v[d]=i[RED]*255,v[h]=i[GREEN]*255,v[g]=i[BLUE]*255):(v[d]=0,v[h]=0,v[g]=0),_.alpha!==void 0&&(v[p]=_.alpha),v[d]=Ct(Math.floor(v[d]),0,255),v[h]=Ct(Math.floor(v[h]),0,255),v[g]=Ct(Math.floor(v[g]),0,255),v[p]=Ct(Math.floor(v[p]),0,255);let b=x[S],M=x[A];for(let E=0;E<3;++E)for(let y=0;y<15;++y){const C=m[E*15+y];y<b&&y<M&&(v[Ee.OFFSET.FRC0+C]=_.sh[E*M+y])}return e.set(_.rotation[3],_.rotation[0],_.rotation[1],_.rotation[2]),e.normalize(),v[l]=e.x,v[c]=e.y,v[u]=e.z,v[f]=e.w,v[t]=_.position[0],v[n]=_.position[1],v[s]=_.position[2],v}})();function KT(i,e,t,n){return!(i.positions.length!==e*3*(n?2:3)||i.scales.length!==e*3||i.rotations.length!==e*3||i.alphas.length!==e||i.colors.length!==e*3||i.sh.length!==e*t*3)}function zp(i,e,t,n,s){e=Math.min(e,i.shDegree);const r=i.numPoints,o=rr(i.shDegree),a=i.positions.length===r*3*2;if(!KT(i,r,o,a))return null;const l={position:[],scale:[],rotation:[],alpha:void 0,color:[],sh:[]};let c;a&&(c=new Uint16Array(i.positions.buffer,i.positions.byteOffset,r*3));const u=1/(1<<i.fractionalBits),f=rr(i.shDegree),d=.28209479177387814;for(let h=0;h<r;h++){if(a)for(let _=0;_<3;_++)l.position[_]=qT(c[h*3+_]);else for(let _=0;_<3;_++){const A=h*9+_*3;let S=i.positions[A];S|=i.positions[A+1]<<8,S|=i.positions[A+2]<<16,S|=S&8388608?4278190080:0,l.position[_]=S*u}for(let _=0;_<3;_++)l.scale[_]=Math.exp(i.scales[h*3+_]/16-10);const g=i.rotations.subarray(h*3,h*3+3),p=[g[0]/127.5-1,g[1]/127.5-1,g[2]/127.5-1];l.rotation[0]=p[0],l.rotation[1]=p[1],l.rotation[2]=p[2];const x=p[0]*p[0]+p[1]*p[1]+p[2]*p[2];l.rotation[3]=Math.sqrt(Math.max(0,1-x)),l.alpha=Math.floor(i.alphas[h]);for(let _=0;_<3;_++)l.color[_]=Math.floor(((i.colors[h*3+_]/255-.5)/XT*d+.5)*255);for(let _=0;_<3;_++)for(let A=0;A<f;A++)l.sh[_*f+A]=QT(i.sh[f*3*h+A*3+_]);const m=YT(l,i.shDegree,e);if(t){const _=j.CompressionLevels[0].SphericalHarmonicsDegrees[e].BytesPerSplat,A=h*_+s;j.writeSplatDataToSectionBuffer(m,n,A,0,e)}else n.addSplat(m)}}const jT=16,$T=1e7;function ZT(i){const e=new DataView(i);let t=0;const n={magic:e.getUint32(t,!0),version:e.getUint32(t+4,!0),numPoints:e.getUint32(t+8,!0),shDegree:e.getUint8(t+12),fractionalBits:e.getUint8(t+13),flags:e.getUint8(t+14),reserved:e.getUint8(t+15)};if(t+=jT,n.magic!==GT)return console.error("[SPZ ERROR] deserializePackedGaussians: header not found"),null;if(n.version<1||n.version>2)return console.error(`[SPZ ERROR] deserializePackedGaussians: version not supported: ${n.version}`),null;if(n.numPoints>$T)return console.error(`[SPZ ERROR] deserializePackedGaussians: Too many points: ${n.numPoints}`),null;if(n.shDegree>3)return console.error(`[SPZ ERROR] deserializePackedGaussians: Unsupported SH degree: ${n.shDegree}`),null;const s=n.numPoints,r=rr(n.shDegree),o=n.version===1,a={numPoints:s,shDegree:n.shDegree,fractionalBits:n.fractionalBits,antialiased:(n.flags&WT)!==0,positions:new Uint8Array(s*3*(o?2:3)),scales:new Uint8Array(s*3),rotations:new Uint8Array(s*3),alphas:new Uint8Array(s),colors:new Uint8Array(s*3),sh:new Uint8Array(s*r*3)};try{const l=new Uint8Array(i);let c=a.positions.length,u=t;if(a.positions.set(l.slice(u,u+c)),u+=c,a.alphas.set(l.slice(u,u+a.alphas.length)),u+=a.alphas.length,a.colors.set(l.slice(u,u+a.colors.length)),u+=a.colors.length,a.scales.set(l.slice(u,u+a.scales.length)),u+=a.scales.length,a.rotations.set(l.slice(u,u+a.rotations.length)),u+=a.rotations.length,a.sh.set(l.slice(u,u+a.sh.length)),u+a.sh.length!==i.byteLength)return console.error("[SPZ ERROR] deserializePackedGaussians: incorrect buffer size"),null}catch(l){return console.error("[SPZ ERROR] deserializePackedGaussians: read error",l),null}return a}async function JT(i){try{const e=await HT(i);return ZT(e.buffer)}catch(e){return console.error("[SPZ ERROR] loadSpzPacked: decompression error",e),null}}class gd{static loadFromURL(e,t,n,s,r=!0,o=0,a,l,c,u,f){return t&&t(0,"0%",Nt.Downloading),ec(e,t,!0,a).then(d=>(t&&t(0,"0%",Nt.Processing),gd.loadFromFileData(d,n,s,r,o,l,c,u,f)))}static async loadFromFileData(e,t,n,s,r=0,o,a,l,c){await Gn();const u=await JT(e);r=Math.min(u.shDegree,r);const f=new Ee(r);if(s)return zp(u,r,!1,f,0),ba.getStandardGenerator(t,n,o,a,l,c).generateFromUncompressedSplatArray(f);{const{splatBuffer:d,splatBufferDataOffsetBytes:h}=j.preallocateUncompressed(u.numPoints,r);return zp(u,r,!0,d.bufferData,h),d}}}class dt{static RowSizeBytes=32;static CenterSizeBytes=12;static ScaleSizeBytes=12;static RotationSizeBytes=4;static ColorSizeBytes=4;static parseToUncompressedSplatBufferSection(e,t,n,s,r,o){const a=j.CompressionLevels[0].BytesPerCenter,l=j.CompressionLevels[0].BytesPerScale,c=j.CompressionLevels[0].BytesPerRotation,u=j.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat;for(let f=e;f<=t;f++){const d=f*dt.RowSizeBytes+s,h=new Float32Array(n,d,3),g=new Float32Array(n,d+dt.CenterSizeBytes,3),p=new Uint8Array(n,d+dt.CenterSizeBytes+dt.ScaleSizeBytes,4),x=new Uint8Array(n,d+dt.CenterSizeBytes+dt.ScaleSizeBytes+dt.RotationSizeBytes,4),m=new bt((x[1]-128)/128,(x[2]-128)/128,(x[3]-128)/128,(x[0]-128)/128);m.normalize();const _=f*u+o,A=new Float32Array(r,_,3),S=new Float32Array(r,_+a,3),v=new Float32Array(r,_+a+l,4),b=new Uint8Array(r,_+a+l+c,4);A[0]=h[0],A[1]=h[1],A[2]=h[2],S[0]=g[0],S[1]=g[1],S[2]=g[2],v[0]=m.w,v[1]=m.x,v[2]=m.y,v[3]=m.z,b[0]=p[0],b[1]=p[1],b[2]=p[2],b[3]=p[3]}}static parseToUncompressedSplatArraySection(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*dt.RowSizeBytes+s,l=new Float32Array(n,a,3),c=new Float32Array(n,a+dt.CenterSizeBytes,3),u=new Uint8Array(n,a+dt.CenterSizeBytes+dt.ScaleSizeBytes,4),f=new Uint8Array(n,a+dt.CenterSizeBytes+dt.ScaleSizeBytes+dt.RotationSizeBytes,4),d=new bt((f[1]-128)/128,(f[2]-128)/128,(f[3]-128)/128,(f[0]-128)/128);d.normalize(),r.addSplatFromComonents(l[0],l[1],l[2],c[0],c[1],c[2],d.w,d.x,d.y,d.z,u[0],u[1],u[2],u[3])}}static parseStandardSplatToUncompressedSplatArray(e){const t=e.byteLength/dt.RowSizeBytes,n=new Ee;for(let s=0;s<t;s++){const r=s*dt.RowSizeBytes,o=new Float32Array(e,r,3),a=new Float32Array(e,r+dt.CenterSizeBytes,3),l=new Uint8Array(e,r+dt.CenterSizeBytes+dt.ScaleSizeBytes,4),c=new Uint8Array(e,r+dt.CenterSizeBytes+dt.ScaleSizeBytes+dt.ColorSizeBytes,4),u=new bt((c[1]-128)/128,(c[2]-128)/128,(c[3]-128)/128,(c[0]-128)/128);u.normalize(),n.addSplatFromComonents(o[0],o[1],o[2],a[0],a[1],a[2],u.w,u.x,u.y,u.z,l[0],l[1],l[2],l[3])}return n}}function kp(i,e,t,n,s,r,o,a){return e?ba.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):j.generateFromUncompressedSplatArrays([i],t,0,new L)}class xd{static loadFromURL(e,t,n,s,r,o,a=!0,l,c,u,f,d){let h=n?yt.ProgressiveToSplatBuffer:yt.ProgressiveToSplatArray;a&&(h=yt.ProgressiveToSplatArray);const g=j.HeaderSizeBytes+j.SectionHeaderSizeBytes,p=pt.ProgressiveLoadSectionSize,x=1;let m,_,A,S=0,v=0,b;const M=sd();let E=0,y=0,C=[];const I=(P,U,O,k)=>{const z=P>=100;if(O&&C.push(O),h===yt.DownloadBeforeProcessing){z&&M.resolve(C);return}if(!k){if(n)throw new Il("Cannon directly load .splat because no file size info is available.");h=yt.DownloadBeforeProcessing;return}if(!m){S=k/dt.RowSizeBytes,m=new ArrayBuffer(k);const W=j.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,H=g+W*S;h===yt.ProgressiveToSplatBuffer?(_=new ArrayBuffer(H),j.writeHeaderToBuffer({versionMajor:j.CurrentMajorVersion,versionMinor:j.CurrentMinorVersion,maxSectionCount:x,sectionCount:x,maxSplatCount:S,splatCount:v,compressionLevel:0,sceneCenter:new L},_)):b=new Ee(0)}if(O){new Uint8Array(m,y,O.byteLength).set(new Uint8Array(O)),y+=O.byteLength;const W=y-E;if(W>p||z){const $=(z?W:p)/dt.RowSizeBytes,X=v+$;h===yt.ProgressiveToSplatBuffer?dt.parseToUncompressedSplatBufferSection(v,X-1,m,0,_,g):dt.parseToUncompressedSplatArraySection(v,X-1,m,0,b),v=X,h===yt.ProgressiveToSplatBuffer&&(A||(j.writeSectionHeaderToBuffer({maxSplatCount:S,splatCount:v,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0},0,_,j.HeaderSizeBytes),A=new j(_,!1)),A.updateLoadedCounts(1,v),s&&s(A,z)),E+=p}}z&&(h===yt.ProgressiveToSplatBuffer?M.resolve(A):M.resolve(b)),t&&t(P,U,Nt.Downloading)};return t&&t(0,"0%",Nt.Downloading),ec(e,I,!1,l).then(()=>(t&&t(0,"0%",Nt.Processing),M.promise.then(P=>(t&&t(100,"100%",Nt.Done),h===yt.DownloadBeforeProcessing?new Blob(C).arrayBuffer().then(U=>xd.loadFromFileData(U,r,o,a,c,u,f,d)):h===yt.ProgressiveToSplatBuffer?P:Gn(()=>kp(P,a,r,o,c,u,f,d))))))}static loadFromFileData(e,t,n,s,r,o,a,l){return Gn(()=>{const c=dt.parseStandardSplatToUncompressedSplatArray(e);return kp(c,s,t,n,r,o,a,l)})}}class Xo{static checkVersion(e){const t=j.CurrentMajorVersion,n=j.CurrentMinorVersion,s=j.parseHeader(e);if(s.versionMajor===t&&s.versionMinor>=n||s.versionMajor>t)return!0;throw new Error(`KSplat version not supported: v${s.versionMajor}.${s.versionMinor}. Minimum required: v${t}.${n}`)}static loadFromURL(e,t,n,s,r){let o,a,l,c,u=!1,f=!1,d,h=[],g=!1,p=!1,x=0,m=0,_=0,A=!1,S=!1,v=!1,b=[];const M=sd(),E=()=>{!u&&!f&&x>=j.HeaderSizeBytes&&(f=!0,new Blob(b).arrayBuffer().then(k=>{l=new ArrayBuffer(j.HeaderSizeBytes),new Uint8Array(l).set(new Uint8Array(k,0,j.HeaderSizeBytes)),Xo.checkVersion(l),f=!1,u=!0,c=j.parseHeader(l),window.setTimeout(()=>{I()},1)}))};let y=0;const C=()=>{y===0&&(y++,window.setTimeout(()=>{y--,P()},1))},I=()=>{const O=()=>{p=!0,new Blob(b).arrayBuffer().then(z=>{p=!1,g=!0,d=new ArrayBuffer(c.maxSectionCount*j.SectionHeaderSizeBytes),new Uint8Array(d).set(new Uint8Array(z,j.HeaderSizeBytes,c.maxSectionCount*j.SectionHeaderSizeBytes)),h=j.parseSectionHeaders(c,d,0,!1);let W=0;for(let $=0;$<c.maxSectionCount;$++)W+=h[$].storageSizeBytes;const H=j.HeaderSizeBytes+c.maxSectionCount*j.SectionHeaderSizeBytes+W;if(!o){o=new ArrayBuffer(H);let $=0;for(let X=0;X<b.length;X++){const Z=b[X];new Uint8Array(o,$,Z.byteLength).set(new Uint8Array(Z)),$+=Z.byteLength}}_=j.HeaderSizeBytes+j.SectionHeaderSizeBytes*c.maxSectionCount;for(let $=0;$<=h.length&&$<c.maxSectionCount;$++)_+=h[$].storageSizeBytes;C()})};!p&&!g&&u&&x>=j.HeaderSizeBytes+j.SectionHeaderSizeBytes*c.maxSectionCount&&O()},P=()=>{if(v)return;v=!0;const O=()=>{if(v=!1,g){if(S)return;if(A=x>=_,x-m>pt.ProgressiveLoadSectionSize||A){m+=pt.ProgressiveLoadSectionSize,S=m>=_,a||(a=new j(o,!1));const z=j.HeaderSizeBytes+j.SectionHeaderSizeBytes*c.maxSectionCount;let W=0,H=0,$=0;for(let pe=0;pe<c.maxSectionCount;pe++){const ge=h[pe],be=W+ge.partiallyFilledBucketCount*4+ge.bucketStorageSizeBytes*ge.bucketCount,Me=z+be;if(m>=Me){H++;const Y=m-Me,Fe=j.CompressionLevels[c.compressionLevel].SphericalHarmonicsDegrees[ge.sphericalHarmonicsDegree].BytesPerSplat;let ye=Math.floor(Y/Fe);ye=Math.min(ye,ge.maxSplatCount),$+=ye,a.updateLoadedCounts(H,$),a.updateSectionLoadedCounts(pe,ye)}else break;W+=ge.storageSizeBytes}s(a,S);const X=m/_*100,Z=X.toFixed(2)+"%";t&&t(X,Z,Nt.Downloading),S?M.resolve(a):P()}}};window.setTimeout(O,pt.ProgressiveLoadSectionDelayDuration)};return ec(e,(O,k,z)=>{z&&(b.push(z),o&&new Uint8Array(o,x,z.byteLength).set(new Uint8Array(z)),x+=z.byteLength),n?(E(),I(),P()):t&&t(O,k,Nt.Downloading)},!n,r).then(O=>(t&&t(0,"0%",Nt.Processing),(n?M.promise:Xo.loadFromFileData(O)).then(z=>(t&&t(100,"100%",Nt.Done),z))))}static loadFromFileData(e){return Gn(()=>(Xo.checkVersion(e),new j(e)))}static downloadFile=(function(){let e;return function(t,n){const s=new Blob([t.bufferData],{type:"application/octet-stream"});e||(e=document.createElement("a"),document.body.appendChild(e)),e.download=n,e.href=URL.createObjectURL(s),e.click()}})()}const Tn={Splat:0,KSplat:1,Ply:2,Spz:3},Hp=i=>i.endsWith(".ply")?Tn.Ply:i.endsWith(".splat")?Tn.Splat:i.endsWith(".ksplat")?Tn.KSplat:i.endsWith(".spz")?Tn.Spz:null,Vp={type:"change"},Zc={type:"start"},Gp={type:"end"},il=new ed,Wp=new vs,eE=Math.cos(70*U0.DEG2RAD);class sl extends mr{constructor(e,t){super(),this.object=e,this.domElement=t,this.domElement.style.touchAction="none",this.enabled=!0,this.target=new L,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"KeyA",UP:"KeyW",RIGHT:"KeyD",BOTTOM:"KeyS"},this.mouseButtons={LEFT:Ar.ROTATE,MIDDLE:Ar.DOLLY,RIGHT:Ar.PAN},this.touches={ONE:Sr.ROTATE,TWO:Sr.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this.getPolarAngle=function(){return a.phi},this.getAzimuthalAngle=function(){return a.theta},this.getDistance=function(){return this.object.position.distanceTo(this.target)},this.listenToKeyEvents=function(N){N.addEventListener("keydown",T),this._domElementKeyEvents=N},this.stopListenToKeyEvents=function(){this._domElementKeyEvents.removeEventListener("keydown",T),this._domElementKeyEvents=null},this.saveState=function(){n.target0.copy(n.target),n.position0.copy(n.object.position),n.zoom0=n.object.zoom},this.reset=function(){n.target.copy(n.target0),n.object.position.copy(n.position0),n.object.zoom=n.zoom0,this.clearDampedRotation(),this.clearDampedPan(),n.object.updateProjectionMatrix(),n.dispatchEvent(Vp),n.update(),r=s.NONE},this.clearDampedRotation=function(){l.theta=0,l.phi=0},this.clearDampedPan=function(){u.set(0,0,0)},this.update=(function(){const N=new L,ee=new bt().setFromUnitVectors(e.up,new L(0,1,0)),me=ee.clone().invert(),Te=new L,Ie=new bt,we=new L,He=2*Math.PI;return function(){ee.setFromUnitVectors(e.up,new L(0,1,0)),me.copy(ee).invert();const Re=n.object.position;N.copy(Re).sub(n.target),N.applyQuaternion(ee),a.setFromVector3(N),n.autoRotate&&r===s.NONE&&I(y()),n.enableDamping?(a.theta+=l.theta*n.dampingFactor,a.phi+=l.phi*n.dampingFactor):(a.theta+=l.theta,a.phi+=l.phi);let Se=n.minAzimuthAngle,ve=n.maxAzimuthAngle;isFinite(Se)&&isFinite(ve)&&(Se<-Math.PI?Se+=He:Se>Math.PI&&(Se-=He),ve<-Math.PI?ve+=He:ve>Math.PI&&(ve-=He),Se<=ve?a.theta=Math.max(Se,Math.min(ve,a.theta)):a.theta=a.theta>(Se+ve)/2?Math.max(Se,a.theta):Math.min(ve,a.theta)),a.phi=Math.max(n.minPolarAngle,Math.min(n.maxPolarAngle,a.phi)),a.makeSafe(),n.enableDamping===!0?n.target.addScaledVector(u,n.dampingFactor):n.target.add(u),n.zoomToCursor&&b||n.object.isOrthographicCamera?a.radius=$(a.radius):a.radius=$(a.radius*c),N.setFromSpherical(a),N.applyQuaternion(me),Re.copy(n.target).add(N),n.object.lookAt(n.target),n.enableDamping===!0?(l.theta*=1-n.dampingFactor,l.phi*=1-n.dampingFactor,u.multiplyScalar(1-n.dampingFactor)):(l.set(0,0,0),u.set(0,0,0));let _e=!1;if(n.zoomToCursor&&b){let de=null;if(n.object.isPerspectiveCamera){const Be=N.length();de=$(Be*c);const We=Be-de;n.object.position.addScaledVector(S,We),n.object.updateMatrixWorld()}else if(n.object.isOrthographicCamera){const Be=new L(v.x,v.y,0);Be.unproject(n.object),n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),_e=!0;const We=new L(v.x,v.y,0);We.unproject(n.object),n.object.position.sub(We).add(Be),n.object.updateMatrixWorld(),de=N.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),n.zoomToCursor=!1;de!==null&&(this.screenSpacePanning?n.target.set(0,0,-1).transformDirection(n.object.matrix).multiplyScalar(de).add(n.object.position):(il.origin.copy(n.object.position),il.direction.set(0,0,-1).transformDirection(n.object.matrix),Math.abs(n.object.up.dot(il.direction))<eE?e.lookAt(n.target):(Wp.setFromNormalAndCoplanarPoint(n.object.up,n.target),il.intersectPlane(Wp,n.target))))}else n.object.isOrthographicCamera&&(n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),_e=!0);return c=1,b=!1,_e||Te.distanceToSquared(n.object.position)>o||8*(1-Ie.dot(n.object.quaternion))>o||we.distanceToSquared(n.target)>0?(n.dispatchEvent(Vp),Te.copy(n.object.position),Ie.copy(n.object.quaternion),we.copy(n.target),_e=!1,!0):!1}})(),this.dispose=function(){n.domElement.removeEventListener("contextmenu",ce),n.domElement.removeEventListener("pointerdown",ue),n.domElement.removeEventListener("pointercancel",fe),n.domElement.removeEventListener("wheel",R),n.domElement.removeEventListener("pointermove",ie),n.domElement.removeEventListener("pointerup",fe),n._domElementKeyEvents!==null&&(n._domElementKeyEvents.removeEventListener("keydown",T),n._domElementKeyEvents=null)};const n=this,s={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6};let r=s.NONE;const o=1e-6,a=new Zh,l=new Zh;let c=1;const u=new L,f=new Ne,d=new Ne,h=new Ne,g=new Ne,p=new Ne,x=new Ne,m=new Ne,_=new Ne,A=new Ne,S=new L,v=new Ne;let b=!1;const M=[],E={};function y(){return 2*Math.PI/60/60*n.autoRotateSpeed}function C(){return Math.pow(.95,n.zoomSpeed)}function I(N){l.theta-=N}function P(N){l.phi-=N}const U=(function(){const N=new L;return function(me,Te){N.setFromMatrixColumn(Te,0),N.multiplyScalar(-me),u.add(N)}})(),O=(function(){const N=new L;return function(me,Te){n.screenSpacePanning===!0?N.setFromMatrixColumn(Te,1):(N.setFromMatrixColumn(Te,0),N.crossVectors(n.object.up,N)),N.multiplyScalar(me),u.add(N)}})(),k=(function(){const N=new L;return function(me,Te){const Ie=n.domElement;if(n.object.isPerspectiveCamera){const we=n.object.position;N.copy(we).sub(n.target);let He=N.length();He*=Math.tan(n.object.fov/2*Math.PI/180),U(2*me*He/Ie.clientHeight,n.object.matrix),O(2*Te*He/Ie.clientHeight,n.object.matrix)}else n.object.isOrthographicCamera?(U(me*(n.object.right-n.object.left)/n.object.zoom/Ie.clientWidth,n.object.matrix),O(Te*(n.object.top-n.object.bottom)/n.object.zoom/Ie.clientHeight,n.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),n.enablePan=!1)}})();function z(N){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c/=N:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function W(N){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c*=N:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function H(N){if(!n.zoomToCursor)return;b=!0;const ee=n.domElement.getBoundingClientRect(),me=N.clientX-ee.left,Te=N.clientY-ee.top,Ie=ee.width,we=ee.height;v.x=me/Ie*2-1,v.y=-(Te/we)*2+1,S.set(v.x,v.y,1).unproject(e).sub(e.position).normalize()}function $(N){return Math.max(n.minDistance,Math.min(n.maxDistance,N))}function X(N){f.set(N.clientX,N.clientY)}function Z(N){H(N),m.set(N.clientX,N.clientY)}function pe(N){g.set(N.clientX,N.clientY)}function ge(N){d.set(N.clientX,N.clientY),h.subVectors(d,f).multiplyScalar(n.rotateSpeed);const ee=n.domElement;I(2*Math.PI*h.x/ee.clientHeight),P(2*Math.PI*h.y/ee.clientHeight),f.copy(d),n.update()}function be(N){_.set(N.clientX,N.clientY),A.subVectors(_,m),A.y>0?z(C()):A.y<0&&W(C()),m.copy(_),n.update()}function Me(N){p.set(N.clientX,N.clientY),x.subVectors(p,g).multiplyScalar(n.panSpeed),k(x.x,x.y),g.copy(p),n.update()}function Y(N){H(N),N.deltaY<0?W(C()):N.deltaY>0&&z(C()),n.update()}function oe(N){let ee=!1;switch(N.code){case n.keys.UP:N.ctrlKey||N.metaKey||N.shiftKey?P(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):k(0,n.keyPanSpeed),ee=!0;break;case n.keys.BOTTOM:N.ctrlKey||N.metaKey||N.shiftKey?P(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):k(0,-n.keyPanSpeed),ee=!0;break;case n.keys.LEFT:N.ctrlKey||N.metaKey||N.shiftKey?I(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):k(n.keyPanSpeed,0),ee=!0;break;case n.keys.RIGHT:N.ctrlKey||N.metaKey||N.shiftKey?I(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):k(-n.keyPanSpeed,0),ee=!0;break}ee&&(N.preventDefault(),n.update())}function he(){if(M.length===1)f.set(M[0].pageX,M[0].pageY);else{const N=.5*(M[0].pageX+M[1].pageX),ee=.5*(M[0].pageY+M[1].pageY);f.set(N,ee)}}function Fe(){if(M.length===1)g.set(M[0].pageX,M[0].pageY);else{const N=.5*(M[0].pageX+M[1].pageX),ee=.5*(M[0].pageY+M[1].pageY);g.set(N,ee)}}function ye(){const N=M[0].pageX-M[1].pageX,ee=M[0].pageY-M[1].pageY,me=Math.sqrt(N*N+ee*ee);m.set(0,me)}function ke(){n.enableZoom&&ye(),n.enablePan&&Fe()}function F(){n.enableZoom&&ye(),n.enableRotate&&he()}function B(N){if(M.length==1)d.set(N.pageX,N.pageY);else{const me=Ue(N),Te=.5*(N.pageX+me.x),Ie=.5*(N.pageY+me.y);d.set(Te,Ie)}h.subVectors(d,f).multiplyScalar(n.rotateSpeed);const ee=n.domElement;I(2*Math.PI*h.x/ee.clientHeight),P(2*Math.PI*h.y/ee.clientHeight),f.copy(d)}function q(N){if(M.length===1)p.set(N.pageX,N.pageY);else{const ee=Ue(N),me=.5*(N.pageX+ee.x),Te=.5*(N.pageY+ee.y);p.set(me,Te)}x.subVectors(p,g).multiplyScalar(n.panSpeed),k(x.x,x.y),g.copy(p)}function w(N){const ee=Ue(N),me=N.pageX-ee.x,Te=N.pageY-ee.y,Ie=Math.sqrt(me*me+Te*Te);_.set(0,Ie),A.set(0,Math.pow(_.y/m.y,n.zoomSpeed)),z(A.y),m.copy(_)}function ne(N){n.enableZoom&&w(N),n.enablePan&&q(N)}function re(N){n.enableZoom&&w(N),n.enableRotate&&B(N)}function ue(N){n.enabled!==!1&&(M.length===0&&(n.domElement.setPointerCapture(N.pointerId),n.domElement.addEventListener("pointermove",ie),n.domElement.addEventListener("pointerup",fe)),J(N),N.pointerType==="touch"?V(N):te(N))}function ie(N){n.enabled!==!1&&(N.pointerType==="touch"?se(N):xe(N))}function fe(N){Pe(N),M.length===0&&(n.domElement.releasePointerCapture(N.pointerId),n.domElement.removeEventListener("pointermove",ie),n.domElement.removeEventListener("pointerup",fe)),n.dispatchEvent(Gp),r=s.NONE}function te(N){let ee;switch(N.button){case 0:ee=n.mouseButtons.LEFT;break;case 1:ee=n.mouseButtons.MIDDLE;break;case 2:ee=n.mouseButtons.RIGHT;break;default:ee=-1}switch(ee){case Ar.DOLLY:if(n.enableZoom===!1)return;Z(N),r=s.DOLLY;break;case Ar.ROTATE:if(N.ctrlKey||N.metaKey||N.shiftKey){if(n.enablePan===!1)return;pe(N),r=s.PAN}else{if(n.enableRotate===!1)return;X(N),r=s.ROTATE}break;case Ar.PAN:if(N.ctrlKey||N.metaKey||N.shiftKey){if(n.enableRotate===!1)return;X(N),r=s.ROTATE}else{if(n.enablePan===!1)return;pe(N),r=s.PAN}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(Zc)}function xe(N){switch(r){case s.ROTATE:if(n.enableRotate===!1)return;ge(N);break;case s.DOLLY:if(n.enableZoom===!1)return;be(N);break;case s.PAN:if(n.enablePan===!1)return;Me(N);break}}function R(N){n.enabled===!1||n.enableZoom===!1||r!==s.NONE||(N.preventDefault(),n.dispatchEvent(Zc),Y(N),n.dispatchEvent(Gp))}function T(N){n.enabled===!1||n.enablePan===!1||oe(N)}function V(N){switch(Ae(N),M.length){case 1:switch(n.touches.ONE){case Sr.ROTATE:if(n.enableRotate===!1)return;he(),r=s.TOUCH_ROTATE;break;case Sr.PAN:if(n.enablePan===!1)return;Fe(),r=s.TOUCH_PAN;break;default:r=s.NONE}break;case 2:switch(n.touches.TWO){case Sr.DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;ke(),r=s.TOUCH_DOLLY_PAN;break;case Sr.DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;F(),r=s.TOUCH_DOLLY_ROTATE;break;default:r=s.NONE}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(Zc)}function se(N){switch(Ae(N),r){case s.TOUCH_ROTATE:if(n.enableRotate===!1)return;B(N),n.update();break;case s.TOUCH_PAN:if(n.enablePan===!1)return;q(N),n.update();break;case s.TOUCH_DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;ne(N),n.update();break;case s.TOUCH_DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;re(N),n.update();break;default:r=s.NONE}}function ce(N){n.enabled!==!1&&N.preventDefault()}function J(N){M.push(N)}function Pe(N){delete E[N.pointerId];for(let ee=0;ee<M.length;ee++)if(M[ee].pointerId==N.pointerId){M.splice(ee,1);return}}function Ae(N){let ee=E[N.pointerId];ee===void 0&&(ee=new Ne,E[N.pointerId]=ee),ee.set(N.pageX,N.pageY)}function Ue(N){const ee=N.pointerId===M[0].pointerId?M[1]:M[0];return E[ee.pointerId]}n.domElement.addEventListener("contextmenu",ce),n.domElement.addEventListener("pointerdown",ue),n.domElement.addEventListener("pointercancel",fe),n.domElement.addEventListener("wheel",R,{passive:!1}),this.update()}}const tE=(i,e,t,n,s)=>{const r=performance.now();let o=i.style.display==="none"?0:parseFloat(i.style.opacity);isNaN(o)&&(o=1);const a=window.setInterval(()=>{const c=performance.now()-r;let u=Math.min(c/n,1);u>.999&&(u=1);let f;e?(f=(1-u)*o,f<1e-4&&(f=0)):f=(1-o)*u+o,f>0?(i.style.display=t,i.style.opacity=f):i.style.display="none",u>=1&&(s&&s(),window.clearInterval(a))},16);return a},nE=500;class _d{static elementIDGen=0;constructor(e,t){this.taskIDGen=0,this.elementID=_d.elementIDGen++,this.tasks=[],this.message=e||"Loading...",this.container=t||document.body,this.spinnerContainerOuter=document.createElement("div"),this.spinnerContainerOuter.className=`spinnerOuterContainer${this.elementID}`,this.spinnerContainerOuter.style.display="none",this.spinnerContainerPrimary=document.createElement("div"),this.spinnerContainerPrimary.className=`spinnerContainerPrimary${this.elementID}`,this.spinnerPrimary=document.createElement("div"),this.spinnerPrimary.classList.add(`spinner${this.elementID}`,`spinnerPrimary${this.elementID}`),this.messageContainerPrimary=document.createElement("div"),this.messageContainerPrimary.classList.add(`messageContainer${this.elementID}`,`messageContainerPrimary${this.elementID}`),this.messageContainerPrimary.innerHTML=this.message,this.spinnerContainerMin=document.createElement("div"),this.spinnerContainerMin.className=`spinnerContainerMin${this.elementID}`,this.spinnerMin=document.createElement("div"),this.spinnerMin.classList.add(`spinner${this.elementID}`,`spinnerMin${this.elementID}`),this.messageContainerMin=document.createElement("div"),this.messageContainerMin.classList.add(`messageContainer${this.elementID}`,`messageContainerMin${this.elementID}`),this.messageContainerMin.innerHTML=this.message,this.spinnerContainerPrimary.appendChild(this.spinnerPrimary),this.spinnerContainerPrimary.appendChild(this.messageContainerPrimary),this.spinnerContainerOuter.appendChild(this.spinnerContainerPrimary),this.spinnerContainerMin.appendChild(this.spinnerMin),this.spinnerContainerMin.appendChild(this.messageContainerMin),this.spinnerContainerOuter.appendChild(this.spinnerContainerMin);const n=document.createElement("style");n.innerHTML=`

            .spinnerOuterContainer${this.elementID} {
                width: 100%;
                height: 100%;
                margin: 0;
                top: 0;
                left: 0;
                position: absolute;
                pointer-events: none;
            }

            .messageContainer${this.elementID} {
                height: 20px;
                font-family: arial;
                font-size: 12pt;
                color: #ffffff;
                text-align: center;
                vertical-align: middle;
            }

            .spinner${this.elementID} {
                padding: 15px;
                background: #07e8d6;
                z-index:99999;
            
                aspect-ratio: 1;
                border-radius: 50%;
                --_m: 
                    conic-gradient(#0000,#000),
                    linear-gradient(#000 0 0) content-box;
                -webkit-mask: var(--_m);
                    mask: var(--_m);
                -webkit-mask-composite: source-out;
                    mask-composite: subtract;
                box-sizing: border-box;
                animation: load 1s linear infinite;
            }

            .spinnerContainerPrimary${this.elementID} {
                z-index:99999;
                background-color: rgba(128, 128, 128, 0.75);
                border: #666666 1px solid;
                border-radius: 5px;
                padding-top: 20px;
                padding-bottom: 10px;
                margin: 0;
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-80px, -80px);
                width: 180px;
                pointer-events: auto;
            }

            .spinnerPrimary${this.elementID} {
                width: 120px;
                margin-left: 30px;
            }

            .messageContainerPrimary${this.elementID} {
                padding-top: 15px;
            }

            .spinnerContainerMin${this.elementID} {
                z-index:99999;
                background-color: rgba(128, 128, 128, 0.75);
                border: #666666 1px solid;
                border-radius: 5px;
                padding-top: 20px;
                padding-bottom: 15px;
                margin: 0;
                position: absolute;
                bottom: 50px;
                left: 50%;
                transform: translate(-50%, 0);
                display: flex;
                flex-direction: left;
                pointer-events: auto;
                min-width: 250px;
            }

            .messageContainerMin${this.elementID} {
                margin-right: 15px;
            }

            .spinnerMin${this.elementID} {
                width: 50px;
                height: 50px;
                margin-left: 15px;
                margin-right: 25px;
            }

            .messageContainerMin${this.elementID} {
                padding-top: 15px;
            }
            
            @keyframes load {
                to{transform: rotate(1turn)}
            }

        `,this.spinnerContainerOuter.appendChild(n),this.container.appendChild(this.spinnerContainerOuter),this.setMinimized(!1,!0),this.fadeTransitions=[]}addTask(e){const t={message:e,id:this.taskIDGen++};return this.tasks.push(t),this.update(),t.id}removeTask(e){let t=0;for(let n of this.tasks){if(n.id===e){this.tasks.splice(t,1);break}t++}this.update()}removeAllTasks(){this.tasks=[],this.update()}setMessageForTask(e,t){for(let n of this.tasks)if(n.id===e){n.message=t;break}this.update()}update(){this.tasks.length>0?(this.show(),this.setMessage(this.tasks[this.tasks.length-1].message)):this.hide()}show(){this.spinnerContainerOuter.style.display="block",this.visible=!0}hide(){this.spinnerContainerOuter.style.display="none",this.visible=!1}setContainer(e){this.container&&this.spinnerContainerOuter.parentElement===this.container&&this.container.removeChild(this.spinnerContainerOuter),e&&(this.container=e,this.container.appendChild(this.spinnerContainerOuter),this.spinnerContainerOuter.style.zIndex=this.container.style.zIndex+1)}setMinimized(e,t){const n=(s,r,o,a,l)=>{o?s.style.display=r?a:"none":this.fadeTransitions[l]=tE(s,!r,a,nE,()=>{this.fadeTransitions[l]=null})};n(this.spinnerContainerPrimary,!e,t,"block",0),n(this.spinnerContainerMin,e,t,"flex",1),this.minimized=e}setMessage(e){this.messageContainerPrimary.innerHTML=e,this.messageContainerMin.innerHTML=e}}class iE{constructor(e){this.idGen=0,this.tasks=[],this.container=e||document.body,this.progressBarContainerOuter=document.createElement("div"),this.progressBarContainerOuter.className="progressBarOuterContainer",this.progressBarContainerOuter.style.display="none",this.progressBarBox=document.createElement("div"),this.progressBarBox.className="progressBarBox",this.progressBarBackground=document.createElement("div"),this.progressBarBackground.className="progressBarBackground",this.progressBar=document.createElement("div"),this.progressBar.className="progressBar",this.progressBarBackground.appendChild(this.progressBar),this.progressBarBox.appendChild(this.progressBarBackground),this.progressBarContainerOuter.appendChild(this.progressBarBox);const t=document.createElement("style");t.innerHTML=`

            .progressBarOuterContainer {
                width: 100%;
                height: 100%;
                margin: 0;
                top: 0;
                left: 0;
                position: absolute;
                pointer-events: none;
            }

            .progressBarBox {
                z-index:99999;
                padding: 7px 9px 5px 7px;
                background-color: rgba(190, 190, 190, 0.75);
                border: #555555 1px solid;
                border-radius: 15px;
                margin: 0;
                position: absolute;
                bottom: 50px;
                left: 50%;
                transform: translate(-50%, 0);
                width: 180px;
                height: 30px;
                pointer-events: auto;
            }

            .progressBarBackground {
                width: 100%;
                height: 25px;
                border-radius:10px;
                background-color: rgba(128, 128, 128, 0.75);
                border: #444444 1px solid;
                box-shadow: inset 0 0 10px #333333;
            }

            .progressBar {
                height: 25px;
                width: 0px;
                border-radius:10px;
                background-color: rgba(0, 200, 0, 0.75);
                box-shadow: inset 0 0 10px #003300;
            }

        `,this.progressBarContainerOuter.appendChild(t),this.container.appendChild(this.progressBarContainerOuter)}show(){this.progressBarContainerOuter.style.display="block"}hide(){this.progressBarContainerOuter.style.display="none"}setProgress(e){this.progressBar.style.width=e+"%"}setContainer(e){this.container&&this.progressBarContainerOuter.parentElement===this.container&&this.container.removeChild(this.progressBarContainerOuter),e&&(this.container=e,this.container.appendChild(this.progressBarContainerOuter),this.progressBarContainerOuter.style.zIndex=this.container.style.zIndex+1)}}class sE{constructor(e){this.container=e||document.body,this.infoCells={};const t=[["Camera position","cameraPosition"],["Camera look-at","cameraLookAt"],["Camera up","cameraUp"],["Camera mode","orthographicCamera"],["Cursor position","cursorPosition"],["FPS","fps"],["Rendering:","renderSplatCount"],["Sort time","sortTime"],["Render window","renderWindow"],["Focal adjustment","focalAdjustment"],["Splat scale","splatScale"],["Point cloud mode","pointCloudMode"]];this.infoPanelContainer=document.createElement("div");const n=document.createElement("style");n.innerHTML=`

            .infoPanel {
                width: 430px;
                padding: 10px;
                background-color: rgba(50, 50, 50, 0.85);
                border: #555555 2px solid;
                color: #dddddd;
                border-radius: 10px;
                z-index: 9999;
                font-family: arial;
                font-size: 11pt;
                text-align: left;
                margin: 0;
                top: 10px;
                left:10px;
                position: absolute;
                pointer-events: auto;
            }

            .info-panel-cell {
                margin-bottom: 5px;
                padding-bottom: 2px;
            }

            .label-cell {
                font-weight: bold;
                font-size: 12pt;
                width: 140px;
            }

        `,this.infoPanelContainer.append(n),this.infoPanel=document.createElement("div"),this.infoPanel.className="infoPanel";const s=document.createElement("div");s.style.display="table";for(let r of t){const o=document.createElement("div");o.style.display="table-row",o.className="info-panel-row";const a=document.createElement("div");a.style.display="table-cell",a.innerHTML=`${r[0]}: `,a.classList.add("info-panel-cell","label-cell");const l=document.createElement("div");l.style.display="table-cell",l.style.width="10px",l.innerHTML=" ",l.className="info-panel-cell";const c=document.createElement("div");c.style.display="table-cell",c.innerHTML="",c.className="info-panel-cell",this.infoCells[r[1]]=c,o.appendChild(a),o.appendChild(l),o.appendChild(c),s.appendChild(o)}this.infoPanel.appendChild(s),this.infoPanelContainer.append(this.infoPanel),this.infoPanelContainer.style.display="none",this.container.appendChild(this.infoPanelContainer),this.visible=!1}update=function(e,t,n,s,r,o,a,l,c,u,f,d,h,g){const p=`${t.x.toFixed(5)}, ${t.y.toFixed(5)}, ${t.z.toFixed(5)}`;if(this.infoCells.cameraPosition.innerHTML!==p&&(this.infoCells.cameraPosition.innerHTML=p),n){const m=n,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cameraLookAt.innerHTML!==_&&(this.infoCells.cameraLookAt.innerHTML=_)}const x=`${s.x.toFixed(5)}, ${s.y.toFixed(5)}, ${s.z.toFixed(5)}`;if(this.infoCells.cameraUp.innerHTML!==x&&(this.infoCells.cameraUp.innerHTML=x),this.infoCells.orthographicCamera.innerHTML=r?"Orthographic":"Perspective",o){const m=o,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cursorPosition.innerHTML=_}else this.infoCells.cursorPosition.innerHTML="N/A";this.infoCells.fps.innerHTML=a,this.infoCells.renderWindow.innerHTML=`${e.x} x ${e.y}`,this.infoCells.renderSplatCount.innerHTML=`${c} splats out of ${l} (${u.toFixed(2)}%)`,this.infoCells.sortTime.innerHTML=`${f.toFixed(3)} ms`,this.infoCells.focalAdjustment.innerHTML=`${d.toFixed(3)}`,this.infoCells.splatScale.innerHTML=`${h.toFixed(3)}`,this.infoCells.pointCloudMode.innerHTML=`${g}`};setContainer(e){this.container&&this.infoPanelContainer.parentElement===this.container&&this.container.removeChild(this.infoPanelContainer),e&&(this.container=e,this.container.appendChild(this.infoPanelContainer),this.infoPanelContainer.style.zIndex=this.container.style.zIndex+1)}show(){this.infoPanelContainer.style.display="block",this.visible=!0}hide(){this.infoPanelContainer.style.display="none",this.visible=!1}}const Xp=new L;class rE extends Wt{constructor(e=new L(0,0,1),t=new L(0,0,0),n=1,s=.1,r=16776960,o=n*.2,a=o*.2){super(),this.type="ArrowHelper";const l=new ua(s,s,n,32);l.translate(0,n/2,0);const c=new ua(0,a,o,32);c.translate(0,n,0),this.position.copy(t),this.line=new Vt(l,new hr({color:r,toneMapped:!1})),this.line.matrixAutoUpdate=!1,this.add(this.line),this.cone=new Vt(c,new hr({color:r,toneMapped:!1})),this.cone.matrixAutoUpdate=!1,this.add(this.cone),this.setDirection(e)}setDirection(e){if(e.y>.99999)this.quaternion.set(0,0,0,1);else if(e.y<-.99999)this.quaternion.set(1,0,0,0);else{Xp.set(e.z,0,-e.x).normalize();const t=Math.acos(e.y);this.quaternion.setFromAxisAngle(Xp,t)}}setColor(e){this.line.material.color.set(e),this.cone.material.color.set(e)}copy(e){return super.copy(e,!1),this.line.copy(e.line),this.cone.copy(e.cone),this}dispose(){this.line.geometry.dispose(),this.line.material.dispose(),this.cone.geometry.dispose(),this.cone.material.dispose()}}class qo{constructor(e){this.threeScene=e,this.splatRenderTarget=null,this.renderTargetCopyQuad=null,this.renderTargetCopyCamera=null,this.meshCursor=null,this.focusMarker=null,this.controlPlane=null,this.debugRoot=null,this.secondaryDebugRoot=null}updateSplatRenderTargetForRenderDimensions(e,t){this.destroySplatRendertarget(),this.splatRenderTarget=new Us(e,t,{format:gn,stencilBuffer:!1,depthBuffer:!0}),this.splatRenderTarget.depthTexture=new td(e,t),this.splatRenderTarget.depthTexture.format=co,this.splatRenderTarget.depthTexture.type=si}destroySplatRendertarget(){this.splatRenderTarget&&(this.splatRenderTarget=null)}setupRenderTargetCopyObjects(){const e={sourceColorTexture:{type:"t",value:null},sourceDepthTexture:{type:"t",value:null}},t=new _n({vertexShader:`
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = vec4( position.xy, 0.0, 1.0 );    
                }
            `,fragmentShader:`
                #include <common>
                #include <packing>
                varying vec2 vUv;
                uniform sampler2D sourceColorTexture;
                uniform sampler2D sourceDepthTexture;
                void main() {
                    vec4 color = texture2D(sourceColorTexture, vUv);
                    float fragDepth = texture2D(sourceDepthTexture, vUv).x;
                    gl_FragDepth = fragDepth;
                    gl_FragColor = vec4(color.rgb, color.a * 2.0);
              }
            `,uniforms:e,depthWrite:!1,depthTest:!1,transparent:!0,blending:M0,blendSrc:ia,blendSrcAlpha:ia,blendDst:sa,blendDstAlpha:sa});t.extensions.fragDepth=!0,this.renderTargetCopyQuad=new Vt(new ho(2,2),t),this.renderTargetCopyCamera=new id(-1,1,1,-1,0,1)}destroyRenderTargetCopyObjects(){this.renderTargetCopyQuad&&(kr(this.renderTargetCopyQuad),this.renderTargetCopyQuad=null)}setupMeshCursor(){if(!this.meshCursor){const e=new nd(.5,1.5,32),t=new hr({color:16777215}),n=new Vt(e,t);n.rotation.set(0,0,Math.PI),n.position.set(0,1,0);const s=new Vt(e,t);s.position.set(0,-1,0);const r=new Vt(e,t);r.rotation.set(0,0,Math.PI/2),r.position.set(1,0,0);const o=new Vt(e,t);o.rotation.set(0,0,-Math.PI/2),o.position.set(-1,0,0),this.meshCursor=new Wt,this.meshCursor.add(n),this.meshCursor.add(s),this.meshCursor.add(r),this.meshCursor.add(o),this.meshCursor.scale.set(.1,.1,.1),this.threeScene.add(this.meshCursor),this.meshCursor.visible=!1}}destroyMeshCursor(){this.meshCursor&&(kr(this.meshCursor),this.threeScene.remove(this.meshCursor),this.meshCursor=null)}setMeshCursorVisibility(e){this.meshCursor.visible=e}getMeschCursorVisibility(){return this.meshCursor.visible}setMeshCursorPosition(e){this.meshCursor.position.copy(e)}positionAndOrientMeshCursor(e,t){this.meshCursor.position.copy(e),this.meshCursor.up.copy(t.up),this.meshCursor.lookAt(t.position)}setupFocusMarker(){if(!this.focusMarker){const e=new Rl(.5,32,32),t=qo.buildFocusMarkerMaterial();t.depthTest=!1,t.depthWrite=!1,t.transparent=!0,this.focusMarker=new Vt(e,t)}}destroyFocusMarker(){this.focusMarker&&(kr(this.focusMarker),this.focusMarker=null)}updateFocusMarker=(function(){const e=new L,t=new qe,n=new L;return function(s,r,o){t.copy(r.matrixWorld).invert(),e.copy(s).applyMatrix4(t),e.normalize().multiplyScalar(10),e.applyMatrix4(r.matrixWorld),n.copy(r.position).sub(s);const a=n.length();this.focusMarker.position.copy(s),this.focusMarker.scale.set(a,a,a),this.focusMarker.material.uniforms.realFocusPosition.value.copy(s),this.focusMarker.material.uniforms.viewport.value.copy(o),this.focusMarker.material.uniformsNeedUpdate=!0}})();setFocusMarkerVisibility(e){this.focusMarker.visible=e}setFocusMarkerOpacity(e){this.focusMarker.material.uniforms.opacity.value=e,this.focusMarker.material.uniformsNeedUpdate=!0}getFocusMarkerOpacity(){return this.focusMarker.material.uniforms.opacity.value}setupControlPlane(){if(!this.controlPlane){const e=new ho(1,1);e.rotateX(-Math.PI/2);const t=new hr({color:16777215});t.transparent=!0,t.opacity=.6,t.depthTest=!1,t.depthWrite=!1,t.side=ti;const n=new Vt(e,t),s=new L(0,1,0);s.normalize();const r=new L(0,0,0),o=.5,a=.01,l=56576,c=new rE(s,r,o,a,l,.1,.03);this.controlPlane=new Wt,this.controlPlane.add(n),this.controlPlane.add(c)}}destroyControlPlane(){this.controlPlane&&(kr(this.controlPlane),this.controlPlane=null)}setControlPlaneVisibility(e){this.controlPlane.visible=e}positionAndOrientControlPlane=(function(){const e=new bt,t=new L(0,1,0);return function(n,s){e.setFromUnitVectors(t,s),this.controlPlane.position.copy(n),this.controlPlane.quaternion.copy(e)}})();addDebugMeshes(){this.debugRoot=this.createDebugMeshes(),this.secondaryDebugRoot=this.createSecondaryDebugMeshes(),this.threeScene.add(this.debugRoot),this.threeScene.add(this.secondaryDebugRoot)}destroyDebugMeshes(){for(let e of[this.debugRoot,this.secondaryDebugRoot])e&&(kr(e),this.threeScene.remove(e));this.debugRoot=null,this.secondaryDebugRoot=null}createDebugMeshes(e){const t=new Rl(1,32,32),n=new Wt,s=(r,o)=>{let a=new Vt(t,qo.buildDebugMaterial(r));a.renderOrder=e,n.add(a),a.position.fromArray(o)};return s(16711680,[-50,0,0]),s(16711680,[50,0,0]),s(65280,[0,0,-50]),s(65280,[0,0,50]),s(16755200,[5,0,5]),n}createSecondaryDebugMeshes(e){const t=new vo(3,3,3),n=new Wt;let s=12303291;const r=a=>{let l=new Vt(t,qo.buildDebugMaterial(s));l.renderOrder=e,n.add(l),l.position.fromArray(a)};let o=10;return r([-o,0,-o]),r([-o,0,o]),r([o,0,-o]),r([o,0,o]),n}static buildDebugMaterial(e){const t=`
            #include <common>
            varying float ndcDepth;

            void main() {
                gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position.xyz, 1.0);
                ndcDepth = gl_Position.z / gl_Position.w;
                gl_Position.x = gl_Position.x / gl_Position.w;
                gl_Position.y = gl_Position.y / gl_Position.w;
                gl_Position.z = 0.0;
                gl_Position.w = 1.0;
    
            }
        `,n=`
            #include <common>
            uniform vec3 color;
            varying float ndcDepth;
            void main() {
                gl_FragDepth = (ndcDepth + 1.0) / 2.0;
                gl_FragColor = vec4(color.rgb, 0.0);
            }
        `,s={color:{type:"v3",value:new nt(e)}},r=new _n({uniforms:s,vertexShader:t,fragmentShader:n,transparent:!1,depthTest:!0,depthWrite:!0,side:Bi});return r.extensions.fragDepth=!0,r}static buildFocusMarkerMaterial(e){const t=`
            #include <common>

            uniform vec2 viewport;
            uniform vec3 realFocusPosition;

            varying vec4 ndcPosition;
            varying vec4 ndcCenter;
            varying vec4 ndcFocusPosition;

            void main() {
                float radius = 0.01;

                vec4 viewPosition = modelViewMatrix * vec4(position.xyz, 1.0);
                vec4 viewCenter = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);

                vec4 viewFocusPosition = modelViewMatrix * vec4(realFocusPosition, 1.0);

                ndcPosition = projectionMatrix * viewPosition;
                ndcPosition = ndcPosition * vec4(1.0 / ndcPosition.w);
                ndcCenter = projectionMatrix * viewCenter;
                ndcCenter = ndcCenter * vec4(1.0 / ndcCenter.w);

                ndcFocusPosition = projectionMatrix * viewFocusPosition;
                ndcFocusPosition = ndcFocusPosition * vec4(1.0 / ndcFocusPosition.w);

                gl_Position = projectionMatrix * viewPosition;

            }
        `,n=`
            #include <common>
            uniform vec3 color;
            uniform vec2 viewport;
            uniform float opacity;

            varying vec4 ndcPosition;
            varying vec4 ndcCenter;
            varying vec4 ndcFocusPosition;

            void main() {
                vec2 screenPosition = vec2(ndcPosition) * viewport;
                vec2 screenCenter = vec2(ndcCenter) * viewport;

                vec2 screenVec = screenPosition - screenCenter;

                float projectedRadius = length(screenVec);

                float lineWidth = 0.0005 * viewport.y;
                float aaRange = 0.0025 * viewport.y;
                float radius = 0.06 * viewport.y;
                float radDiff = abs(projectedRadius - radius) - lineWidth;
                float alpha = 1.0 - clamp(radDiff / 5.0, 0.0, 1.0); 

                gl_FragColor = vec4(color.rgb, alpha * opacity);
            }
        `,s={color:{type:"v3",value:new nt(e)},realFocusPosition:{type:"v3",value:new L},viewport:{type:"v2",value:new Ne},opacity:{value:0}};return new _n({uniforms:s,vertexShader:t,fragmentShader:n,transparent:!0,depthTest:!1,depthWrite:!1,side:Bi})}dispose(){this.destroyMeshCursor(),this.destroyFocusMarker(),this.destroyDebugMeshes(),this.destroyControlPlane(),this.destroyRenderTargetCopyObjects(),this.destroySplatRendertarget()}}const oE=new L(1,0,0),aE=new L(0,1,0),lE=new L(0,0,1);class Jc{constructor(e=new L,t=new L){this.origin=new L,this.direction=new L,this.setParameters(e,t)}setParameters(e,t){this.origin.copy(e),this.direction.copy(t).normalize()}boxContainsPoint(e,t,n){return!(t.x<e.min.x-n||t.x>e.max.x+n||t.y<e.min.y-n||t.y>e.max.y+n||t.z<e.min.z-n||t.z>e.max.z+n)}intersectBox=(function(){const e=new L,t=[],n=[],s=[];return function(r,o){if(n[0]=this.origin.x,n[1]=this.origin.y,n[2]=this.origin.z,s[0]=this.direction.x,s[1]=this.direction.y,s[2]=this.direction.z,this.boxContainsPoint(r,this.origin,1e-4))return o&&(o.origin.copy(this.origin),o.normal.set(0,0,0),o.distance=-1),!0;for(let a=0;a<3;a++){if(s[a]==0)continue;const l=a==0?oE:a==1?aE:lE,c=s[a]<0?r.max:r.min;let u=-Math.sign(s[a]);t[0]=a==0?c.x:a==1?c.y:c.z;let f=t[0]-n[a];if(f*u<0){const d=(a+1)%3,h=(a+2)%3;if(t[2]=s[d]/s[a]*f+n[d],t[1]=s[h]/s[a]*f+n[h],e.set(t[a],t[h],t[d]),this.boxContainsPoint(r,e,1e-4))return o&&(o.origin.copy(e),o.normal.copy(l).multiplyScalar(u),o.distance=e.sub(this.origin).length()),!0}}return!1}})();intersectSphere=(function(){const e=new L;return function(t,n,s){e.copy(t).sub(this.origin);const r=e.dot(this.direction),o=r*r,l=e.dot(e)-o,c=n*n;if(l>c)return!1;const u=Math.sqrt(c-l),f=r-u,d=r+u;if(d<0)return!1;let h=f<0?d:f;return s&&(s.origin.copy(this.origin).addScaledVector(this.direction,h),s.normal.copy(s.origin).sub(t).normalize(),s.distance=h),!0}})()}class Ad{constructor(){this.origin=new L,this.normal=new L,this.distance=0,this.splatIndex=0}set(e,t,n,s){this.origin.copy(e),this.normal.copy(t),this.distance=n,this.splatIndex=s}clone(){const e=new Ad;return e.origin.copy(this.origin),e.normal.copy(this.normal),e.distance=this.distance,e.splatIndex=this.splatIndex,e}}const $i={ThreeD:0,TwoD:1};class cE{constructor(e,t,n=!1){this.ray=new Jc(e,t),this.raycastAgainstTrueSplatEllipsoid=n}setFromCameraAndScreenPosition=(function(){const e=new Ne;return function(t,n,s){if(e.x=n.x/s.x*2-1,e.y=(s.y-n.y)/s.y*2-1,t.isPerspectiveCamera)this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(e.x,e.y,.5).unproject(t).sub(this.ray.origin).normalize(),this.camera=t;else if(t.isOrthographicCamera)this.ray.origin.set(e.x,e.y,(t.near+t.far)/(t.near-t.far)).unproject(t),this.ray.direction.set(0,0,-1).transformDirection(t.matrixWorld),this.camera=t;else throw new Error("Raycaster::setFromCameraAndScreenPosition() -> Unsupported camera type")}})();intersectSplatMesh=(function(){const e=new qe,t=new qe,n=new qe,s=new Jc,r=new L;return function(o,a=[]){const l=o.getSplatTree();if(l){for(let c=0;c<l.subTrees.length;c++){const u=l.subTrees[c];t.copy(o.matrixWorld),o.dynamicMode&&(o.getSceneTransform(c,n),t.multiply(n)),e.copy(t).invert(),s.origin.copy(this.ray.origin).applyMatrix4(e),s.direction.copy(this.ray.origin).add(this.ray.direction),s.direction.applyMatrix4(e).sub(s.origin).normalize();const f=[];u.rootNode&&this.castRayAtSplatTreeNode(s,l,u.rootNode,f),f.forEach(d=>{d.origin.applyMatrix4(t),d.normal.applyMatrix4(t).normalize(),d.distance=r.copy(d.origin).sub(this.ray.origin).length()}),a.push(...f)}return a.sort((c,u)=>c.distance>u.distance?1:-1),a}}})();castRayAtSplatTreeNode=(function(){const e=new Et,t=new L,n=new L,s=new bt,r=new Ad,o=1e-7,a=new L(0,0,0),l=new qe,c=new qe,u=new qe,f=new qe,d=new qe,h=new Jc;return function(g,p,x,m=[]){if(g.intersectBox(x.boundingBox)){if(x.data&&x.data.indexes&&x.data.indexes.length>0)for(let _=0;_<x.data.indexes.length;_++){const A=x.data.indexes[_],S=p.splatMesh.getSceneIndexForSplat(A);if(p.splatMesh.getScene(S).visible&&(p.splatMesh.getSplatColor(A,e),p.splatMesh.getSplatCenter(A,t),p.splatMesh.getSplatScaleAndRotation(A,n,s),!(n.x<=o||n.y<=o||p.splatMesh.splatRenderMode===$i.ThreeD&&n.z<=o)))if(this.raycastAgainstTrueSplatEllipsoid){c.makeScale(n.x,n.y,n.z),u.makeRotationFromQuaternion(s);const b=Math.log10(e.w)*2;if(l.makeScale(b,b,b),d.copy(l).multiply(u).multiply(c),f.copy(d).invert(),h.origin.copy(g.origin).sub(t).applyMatrix4(f),h.direction.copy(g.origin).add(g.direction).sub(t),h.direction.applyMatrix4(f).sub(h.origin).normalize(),h.intersectSphere(a,1,r)){const M=r.clone();M.splatIndex=A,M.origin.applyMatrix4(d).add(t),m.push(M)}}else{let b=n.x+n.y,M=2;if(p.splatMesh.splatRenderMode===$i.ThreeD&&(b+=n.z,M=3),b=b/M,g.intersectSphere(t,b,r)){const E=r.clone();E.splatIndex=A,m.push(E)}}}if(x.children&&x.children.length>0)for(let _ of x.children)this.castRayAtSplatTreeNode(g,p,_,m);return m}}})()}class Jr{static buildVertexShaderBase(e=!1,t=!1,n=0,s=""){let r=`
        precision highp float;
        #include <common>

        attribute uint splatIndex;
        uniform highp usampler2D centersColorsTexture;
        uniform highp sampler2D sphericalHarmonicsTexture;
        uniform highp sampler2D sphericalHarmonicsTextureR;
        uniform highp sampler2D sphericalHarmonicsTextureG;
        uniform highp sampler2D sphericalHarmonicsTextureB;

        uniform highp usampler2D sceneIndexesTexture;
        uniform vec2 sceneIndexesTextureSize;
        uniform int sceneCount;
    `;return t&&(r+=`
            uniform float sceneOpacity[${pt.MaxScenes}];
            uniform int sceneVisibility[${pt.MaxScenes}];
        `),e&&(r+=`
            uniform highp mat4 transforms[${pt.MaxScenes}];
        `),r+=`
        ${s}
        uniform vec2 focal;
        uniform float orthoZoom;
        uniform int orthographicMode;
        uniform int pointCloudModeEnabled;
        uniform float inverseFocalAdjustment;
        uniform vec2 viewport;
        uniform vec2 basisViewport;
        uniform vec2 centersColorsTextureSize;
        uniform int sphericalHarmonicsDegree;
        uniform vec2 sphericalHarmonicsTextureSize;
        uniform int sphericalHarmonics8BitMode;
        uniform int sphericalHarmonicsMultiTextureMode;
        uniform float visibleRegionRadius;
        uniform float visibleRegionFadeStartRadius;
        uniform float firstRenderTime;
        uniform float currentTime;
        uniform int fadeInComplete;
        uniform vec3 sceneCenter;
        uniform float splatScale;
        uniform float sphericalHarmonics8BitCompressionRangeMin[${pt.MaxScenes}];
        uniform float sphericalHarmonics8BitCompressionRangeMax[${pt.MaxScenes}];

        varying vec4 vColor;
        varying vec2 vUv;
        varying vec2 vPosition;

        mat3 quaternionToRotationMatrix(float x, float y, float z, float w) {
            float s = 1.0 / sqrt(w * w + x * x + y * y + z * z);
        
            return mat3(
                1. - 2. * (y * y + z * z),
                2. * (x * y + w * z),
                2. * (x * z - w * y),
                2. * (x * y - w * z),
                1. - 2. * (x * x + z * z),
                2. * (y * z + w * x),
                2. * (x * z + w * y),
                2. * (y * z - w * x),
                1. - 2. * (x * x + y * y)
            );
        }

        const float sqrt8 = sqrt(8.0);
        const float minAlpha = 1.0 / 255.0;

        const vec4 encodeNorm4 = vec4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);
        const uvec4 mask4 = uvec4(uint(0x000000FF), uint(0x0000FF00), uint(0x00FF0000), uint(0xFF000000));
        const uvec4 shift4 = uvec4(0, 8, 16, 24);
        vec4 uintToRGBAVec (uint u) {
           uvec4 urgba = mask4 & u;
           urgba = urgba >> shift4;
           vec4 rgba = vec4(urgba) * encodeNorm4;
           return rgba;
        }

        vec2 getDataUV(in int stride, in int offset, in vec2 dimensions) {
            vec2 samplerUV = vec2(0.0, 0.0);
            float d = float(splatIndex * uint(stride) + uint(offset)) / dimensions.x;
            samplerUV.y = float(floor(d)) / dimensions.y;
            samplerUV.x = fract(d);
            return samplerUV;
        }

        vec2 getDataUVF(in uint sIndex, in float stride, in uint offset, in vec2 dimensions) {
            vec2 samplerUV = vec2(0.0, 0.0);
            float d = float(uint(float(sIndex) * stride) + offset) / dimensions.x;
            samplerUV.y = float(floor(d)) / dimensions.y;
            samplerUV.x = fract(d);
            return samplerUV;
        }

        const float SH_C1 = 0.4886025119029199f;
        const float[5] SH_C2 = float[](1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742);

        void main () {

            uint oddOffset = splatIndex & uint(0x00000001);
            uint doubleOddOffset = oddOffset * uint(2);
            bool isEven = oddOffset == uint(0);
            uint nearestEvenIndex = splatIndex - oddOffset;
            float fOddOffset = float(oddOffset);

            uvec4 sampledCenterColor = texture(centersColorsTexture, getDataUV(1, 0, centersColorsTextureSize));
            vec3 splatCenter = uintBitsToFloat(uvec3(sampledCenterColor.gba));

            uint sceneIndex = uint(0);
            if (sceneCount > 1) {
                sceneIndex = texture(sceneIndexesTexture, getDataUV(1, 0, sceneIndexesTextureSize)).r;
            }
            `,t&&(r+=`
                float splatOpacityFromScene = sceneOpacity[sceneIndex];
                int sceneVisible = sceneVisibility[sceneIndex];
                if (splatOpacityFromScene <= 0.01 || sceneVisible == 0) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }
            `),e?r+=`
                mat4 transform = transforms[sceneIndex];
                mat4 transformModelViewMatrix = viewMatrix * transform;
            `:r+="mat4 transformModelViewMatrix = modelViewMatrix;",r+=`
            float sh8BitCompressionRangeMinForScene = sphericalHarmonics8BitCompressionRangeMin[sceneIndex];
            float sh8BitCompressionRangeMaxForScene = sphericalHarmonics8BitCompressionRangeMax[sceneIndex];
            float sh8BitCompressionRangeForScene = sh8BitCompressionRangeMaxForScene - sh8BitCompressionRangeMinForScene;
            float sh8BitCompressionHalfRangeForScene = sh8BitCompressionRangeForScene / 2.0;
            vec3 vec8BitSHShift = vec3(sh8BitCompressionRangeMinForScene);

            vec4 viewCenter = transformModelViewMatrix * vec4(splatCenter, 1.0);

            vec4 clipCenter = projectionMatrix * viewCenter;

            float clip = 1.2 * clipCenter.w;
            if (clipCenter.z < -clip || clipCenter.x < -clip || clipCenter.x > clip || clipCenter.y < -clip || clipCenter.y > clip) {
                gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                return;
            }

            vec3 ndcCenter = clipCenter.xyz / clipCenter.w;

            vPosition = position.xy;
            vColor = uintToRGBAVec(sampledCenterColor.r);
        `,n>=1&&(r+=`   
            if (sphericalHarmonicsDegree >= 1) {
            `,e?r+=`
                    vec3 worldViewDir = normalize(splatCenter - vec3(inverse(transform) * vec4(cameraPosition, 1.0)));
                `:r+=`
                    vec3 worldViewDir = normalize(splatCenter - cameraPosition);
                `,r+=`
                vec3 sh1;
                vec3 sh2;
                vec3 sh3;
            `,n>=2&&(r+=`
                    vec3 sh4;
                    vec3 sh5;
                    vec3 sh6;
                    vec3 sh7;
                    vec3 sh8;
                `),n===1?r+=`
                    if (sphericalHarmonicsMultiTextureMode == 0) {
                        vec2 shUV = getDataUVF(nearestEvenIndex, 2.5, doubleOddOffset, sphericalHarmonicsTextureSize);
                        vec4 sampledSH0123 = texture(sphericalHarmonicsTexture, shUV);
                        shUV = getDataUVF(nearestEvenIndex, 2.5, doubleOddOffset + uint(1), sphericalHarmonicsTextureSize);
                        vec4 sampledSH4567 = texture(sphericalHarmonicsTexture, shUV);
                        shUV = getDataUVF(nearestEvenIndex, 2.5, doubleOddOffset + uint(2), sphericalHarmonicsTextureSize);
                        vec4 sampledSH891011 = texture(sphericalHarmonicsTexture, shUV);
                        sh1 = vec3(sampledSH0123.rgb) * (1.0 - fOddOffset) + vec3(sampledSH0123.ba, sampledSH4567.r) * fOddOffset;
                        sh2 = vec3(sampledSH0123.a, sampledSH4567.rg) * (1.0 - fOddOffset) + vec3(sampledSH4567.gba) * fOddOffset;
                        sh3 = vec3(sampledSH4567.ba, sampledSH891011.r) * (1.0 - fOddOffset) + vec3(sampledSH891011.rgb) * fOddOffset;
                    } else {
                        vec2 sampledSH01R = texture(sphericalHarmonicsTextureR, getDataUV(2, 0, sphericalHarmonicsTextureSize)).rg;
                        vec2 sampledSH23R = texture(sphericalHarmonicsTextureR, getDataUV(2, 1, sphericalHarmonicsTextureSize)).rg;
                        vec2 sampledSH01G = texture(sphericalHarmonicsTextureG, getDataUV(2, 0, sphericalHarmonicsTextureSize)).rg;
                        vec2 sampledSH23G = texture(sphericalHarmonicsTextureG, getDataUV(2, 1, sphericalHarmonicsTextureSize)).rg;
                        vec2 sampledSH01B = texture(sphericalHarmonicsTextureB, getDataUV(2, 0, sphericalHarmonicsTextureSize)).rg;
                        vec2 sampledSH23B = texture(sphericalHarmonicsTextureB, getDataUV(2, 1, sphericalHarmonicsTextureSize)).rg;
                        sh1 = vec3(sampledSH01R.rg, sampledSH23R.r);
                        sh2 = vec3(sampledSH01G.rg, sampledSH23G.r);
                        sh3 = vec3(sampledSH01B.rg, sampledSH23B.r);
                    }
                `:n===2&&(r+=`
                    vec4 sampledSH0123;
                    vec4 sampledSH4567;
                    vec4 sampledSH891011;

                    vec4 sampledSH0123R;
                    vec4 sampledSH0123G;
                    vec4 sampledSH0123B;

                    if (sphericalHarmonicsMultiTextureMode == 0) {
                        sampledSH0123 = texture(sphericalHarmonicsTexture, getDataUV(6, 0, sphericalHarmonicsTextureSize));
                        sampledSH4567 = texture(sphericalHarmonicsTexture, getDataUV(6, 1, sphericalHarmonicsTextureSize));
                        sampledSH891011 = texture(sphericalHarmonicsTexture, getDataUV(6, 2, sphericalHarmonicsTextureSize));
                        sh1 = sampledSH0123.rgb;
                        sh2 = vec3(sampledSH0123.a, sampledSH4567.rg);
                        sh3 = vec3(sampledSH4567.ba, sampledSH891011.r);
                    } else {
                        sampledSH0123R = texture(sphericalHarmonicsTextureR, getDataUV(2, 0, sphericalHarmonicsTextureSize));
                        sampledSH0123G = texture(sphericalHarmonicsTextureG, getDataUV(2, 0, sphericalHarmonicsTextureSize));
                        sampledSH0123B = texture(sphericalHarmonicsTextureB, getDataUV(2, 0, sphericalHarmonicsTextureSize));
                        sh1 = vec3(sampledSH0123R.rgb);
                        sh2 = vec3(sampledSH0123G.rgb);
                        sh3 = vec3(sampledSH0123B.rgb);
                    }
                `),r+=`
                    if (sphericalHarmonics8BitMode == 1) {
                        sh1 = sh1 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                        sh2 = sh2 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                        sh3 = sh3 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                    }
                    float x = worldViewDir.x;
                    float y = worldViewDir.y;
                    float z = worldViewDir.z;
                    vColor.rgb += SH_C1 * (-sh1 * y + sh2 * z - sh3 * x);
            `,n>=2&&(r+=`
                    if (sphericalHarmonicsDegree >= 2) {
                        float xx = x * x;
                        float yy = y * y;
                        float zz = z * z;
                        float xy = x * y;
                        float yz = y * z;
                        float xz = x * z;
                `,n===2&&(r+=`
                        if (sphericalHarmonicsMultiTextureMode == 0) {
                            vec4 sampledSH12131415 = texture(sphericalHarmonicsTexture, getDataUV(6, 3, sphericalHarmonicsTextureSize));
                            vec4 sampledSH16171819 = texture(sphericalHarmonicsTexture, getDataUV(6, 4, sphericalHarmonicsTextureSize));
                            vec4 sampledSH20212223 = texture(sphericalHarmonicsTexture, getDataUV(6, 5, sphericalHarmonicsTextureSize));
                            sh4 = sampledSH891011.gba;
                            sh5 = sampledSH12131415.rgb;
                            sh6 = vec3(sampledSH12131415.a, sampledSH16171819.rg);
                            sh7 = vec3(sampledSH16171819.ba, sampledSH20212223.r);
                            sh8 = sampledSH20212223.gba;
                        } else {
                            vec4 sampledSH4567R = texture(sphericalHarmonicsTextureR, getDataUV(2, 1, sphericalHarmonicsTextureSize));
                            vec4 sampledSH4567G = texture(sphericalHarmonicsTextureG, getDataUV(2, 1, sphericalHarmonicsTextureSize));
                            vec4 sampledSH4567B = texture(sphericalHarmonicsTextureB, getDataUV(2, 1, sphericalHarmonicsTextureSize));
                            sh4 = vec3(sampledSH0123R.a, sampledSH4567R.rg);
                            sh5 = vec3(sampledSH4567R.ba, sampledSH0123G.a);
                            sh6 = vec3(sampledSH4567G.rgb);
                            sh7 = vec3(sampledSH4567G.a, sampledSH0123B.a, sampledSH4567B.r);
                            sh8 = vec3(sampledSH4567B.gba);
                        }
                    `),r+=`
                        if (sphericalHarmonics8BitMode == 1) {
                            sh4 = sh4 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                            sh5 = sh5 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                            sh6 = sh6 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                            sh7 = sh7 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                            sh8 = sh8 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                        }

                        vColor.rgb +=
                            (SH_C2[0] * xy) * sh4 +
                            (SH_C2[1] * yz) * sh5 +
                            (SH_C2[2] * (2.0 * zz - xx - yy)) * sh6 +
                            (SH_C2[3] * xz) * sh7 +
                            (SH_C2[4] * (xx - yy)) * sh8;
                    }
                `),r+=`

                vColor.rgb = clamp(vColor.rgb, vec3(0.), vec3(1.));

            }

            `),r}static getVertexShaderFadeIn(){return`
            if (fadeInComplete == 0) {
                float opacityAdjust = 1.0;
                float centerDist = length(splatCenter - sceneCenter);
                float renderTime = max(currentTime - firstRenderTime, 0.0);

                float fadeDistance = 0.75;
                float distanceLoadFadeInFactor = step(visibleRegionFadeStartRadius, centerDist);
                distanceLoadFadeInFactor = (1.0 - distanceLoadFadeInFactor) +
                                        (1.0 - clamp((centerDist - visibleRegionFadeStartRadius) / fadeDistance, 0.0, 1.0)) *
                                        distanceLoadFadeInFactor;
                opacityAdjust *= distanceLoadFadeInFactor;
                vColor.a *= opacityAdjust;
            }
        `}static getUniforms(e=!1,t=!1,n=0,s=1,r=!1){const o={sceneCenter:{type:"v3",value:new L},fadeInComplete:{type:"i",value:0},orthographicMode:{type:"i",value:0},visibleRegionFadeStartRadius:{type:"f",value:0},visibleRegionRadius:{type:"f",value:0},currentTime:{type:"f",value:0},firstRenderTime:{type:"f",value:0},centersColorsTexture:{type:"t",value:null},sphericalHarmonicsTexture:{type:"t",value:null},sphericalHarmonicsTextureR:{type:"t",value:null},sphericalHarmonicsTextureG:{type:"t",value:null},sphericalHarmonicsTextureB:{type:"t",value:null},sphericalHarmonics8BitCompressionRangeMin:{type:"f",value:[]},sphericalHarmonics8BitCompressionRangeMax:{type:"f",value:[]},focal:{type:"v2",value:new Ne},orthoZoom:{type:"f",value:1},inverseFocalAdjustment:{type:"f",value:1},viewport:{type:"v2",value:new Ne},basisViewport:{type:"v2",value:new Ne},debugColor:{type:"v3",value:new nt},centersColorsTextureSize:{type:"v2",value:new Ne(1024,1024)},sphericalHarmonicsDegree:{type:"i",value:n},sphericalHarmonicsTextureSize:{type:"v2",value:new Ne(1024,1024)},sphericalHarmonics8BitMode:{type:"i",value:0},sphericalHarmonicsMultiTextureMode:{type:"i",value:0},splatScale:{type:"f",value:s},pointCloudModeEnabled:{type:"i",value:r?1:0},sceneIndexesTexture:{type:"t",value:null},sceneIndexesTextureSize:{type:"v2",value:new Ne(1024,1024)},sceneCount:{type:"i",value:1}};for(let a=0;a<pt.MaxScenes;a++)o.sphericalHarmonics8BitCompressionRangeMin.value.push(-3/2),o.sphericalHarmonics8BitCompressionRangeMax.value.push(pt.SphericalHarmonics8BitCompressionRange/2);if(t){const a=[];for(let c=0;c<pt.MaxScenes;c++)a.push(1);o.sceneOpacity={type:"f",value:a};const l=[];for(let c=0;c<pt.MaxScenes;c++)l.push(1);o.sceneVisibility={type:"i",value:l}}if(e){const a=[];for(let l=0;l<pt.MaxScenes;l++)a.push(new qe);o.transforms={type:"mat4",value:a}}return o}}class Dl{static build(e=!1,t=!1,n=!1,s=2048,r=1,o=!1,a=0,l=.3){let u=Jr.buildVertexShaderBase(e,t,a,`
            uniform vec2 covariancesTextureSize;
            uniform highp sampler2D covariancesTexture;
            uniform highp usampler2D covariancesTextureHalfFloat;
            uniform int covariancesAreHalfFloat;

            void fromCovarianceHalfFloatV4(uvec4 val, out vec4 first, out vec4 second) {
                vec2 r = unpackHalf2x16(val.r);
                vec2 g = unpackHalf2x16(val.g);
                vec2 b = unpackHalf2x16(val.b);

                first = vec4(r.x, r.y, g.x, g.y);
                second = vec4(b.x, b.y, 0.0, 0.0);
            }
        `);u+=Dl.buildVertexShaderProjection(n,t,s,l);const f=Dl.buildFragmentShader(),d=Jr.getUniforms(e,t,a,r,o);return d.covariancesTextureSize={type:"v2",value:new Ne(1024,1024)},d.covariancesTexture={type:"t",value:null},d.covariancesTextureHalfFloat={type:"t",value:null},d.covariancesAreHalfFloat={type:"i",value:0},new _n({uniforms:d,vertexShader:u,fragmentShader:f,transparent:!0,alphaTest:1,blending:Is,depthTest:!0,depthWrite:!1,side:ti})}static buildVertexShaderProjection(e,t,n,s){let r=`

            vec4 sampledCovarianceA;
            vec4 sampledCovarianceB;
            vec3 cov3D_M11_M12_M13;
            vec3 cov3D_M22_M23_M33;
            if (covariancesAreHalfFloat == 0) {
                sampledCovarianceA = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset,
                                                                            covariancesTextureSize));
                sampledCovarianceB = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset + uint(1),
                                                                            covariancesTextureSize));

                cov3D_M11_M12_M13 = vec3(sampledCovarianceA.rgb) * (1.0 - fOddOffset) +
                                    vec3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;
                cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0 - fOddOffset) +
                                    vec3(sampledCovarianceB.gba) * fOddOffset;
            } else {
                uvec4 sampledCovarianceU = texture(covariancesTextureHalfFloat, getDataUV(1, 0, covariancesTextureSize));
                fromCovarianceHalfFloatV4(sampledCovarianceU, sampledCovarianceA, sampledCovarianceB);
                cov3D_M11_M12_M13 = sampledCovarianceA.rgb;
                cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg);
            }
        
            // Construct the 3D covariance matrix
            mat3 Vrk = mat3(
                cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z,
                cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x, cov3D_M22_M23_M33.y,
                cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z
            );

            mat3 J;
            if (orthographicMode == 1) {
                // Since the projection is linear, we don't need an approximation
                J = transpose(mat3(orthoZoom, 0.0, 0.0,
                                0.0, orthoZoom, 0.0,
                                0.0, 0.0, 0.0));
            } else {
                // Construct the Jacobian of the affine approximation of the projection matrix. It will be used to transform the
                // 3D covariance matrix instead of using the actual projection matrix because that transformation would
                // require a non-linear component (perspective division) which would yield a non-gaussian result.
                float s = 1.0 / (viewCenter.z * viewCenter.z);
                J = mat3(
                    focal.x / viewCenter.z, 0., -(focal.x * viewCenter.x) * s,
                    0., focal.y / viewCenter.z, -(focal.y * viewCenter.y) * s,
                    0., 0., 0.
                );
            }

            // Concatenate the projection approximation with the model-view transformation
            mat3 W = transpose(mat3(transformModelViewMatrix));
            mat3 T = W * J;

            // Transform the 3D covariance matrix (Vrk) to compute the 2D covariance matrix
            mat3 cov2Dm = transpose(T) * Vrk * T;
            `;return e?r+=`
                float detOrig = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
                cov2Dm[0][0] += ${s};
                cov2Dm[1][1] += ${s};
                float detBlur = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
                vColor.a *= sqrt(max(detOrig / detBlur, 0.0));
                if (vColor.a < minAlpha) return;
            `:r+=`
                cov2Dm[0][0] += ${s};
                cov2Dm[1][1] += ${s};
            `,r+=`

            // We are interested in the upper-left 2x2 portion of the projected 3D covariance matrix because
            // we only care about the X and Y values. We want the X-diagonal, cov2Dm[0][0],
            // the Y-diagonal, cov2Dm[1][1], and the correlation between the two cov2Dm[0][1]. We don't
            // need cov2Dm[1][0] because it is a symetric matrix.
            vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

            // We now need to solve for the eigen-values and eigen vectors of the 2D covariance matrix
            // so that we can determine the 2D basis for the splat. This is done using the method described
            // here: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
            // After calculating the eigen-values and eigen-vectors, we calculate the basis for rendering the splat
            // by normalizing the eigen-vectors and then multiplying them by (sqrt(8) * sqrt(eigen-value)), which is
            // equal to scaling them by sqrt(8) standard deviations.
            //
            // This is a different approach than in the original work at INRIA. In that work they compute the
            // max extents of the projected splat in screen space to form a screen-space aligned bounding rectangle
            // which forms the geometry that is actually rasterized. The dimensions of that bounding box are 3.0
            // times the square root of the maximum eigen-value, or 3 standard deviations. They then use the inverse
            // 2D covariance matrix (called 'conic') in the CUDA rendering thread to determine fragment opacity by
            // calculating the full gaussian: exp(-0.5 * (X - mean) * conic * (X - mean)) * splat opacity
            float a = cov2Dv.x;
            float d = cov2Dv.z;
            float b = cov2Dv.y;
            float D = a * d - b * b;
            float trace = a + d;
            float traceOver2 = 0.5 * trace;
            float term2 = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
            float eigenValue1 = traceOver2 + term2;
            float eigenValue2 = traceOver2 - term2;

            if (pointCloudModeEnabled == 1) {
                eigenValue1 = eigenValue2 = 0.2;
            }

            if (eigenValue2 <= 0.0) return;

            vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
            // since the eigen vectors are orthogonal, we derive the second one from the first
            vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

            // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
            vec2 basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), ${parseInt(n)}.0);
            vec2 basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), ${parseInt(n)}.0);
            `,t&&(r+=`
                vColor.a *= splatOpacityFromScene;
            `),r+=`
            vec2 ndcOffset = vec2(vPosition.x * basisVector1 + vPosition.y * basisVector2) *
                             basisViewport * 2.0 * inverseFocalAdjustment;

            vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
            gl_Position = quadPos;

            // Scale the position data we send to the fragment shader
            vPosition *= sqrt8;
        `,r+=Jr.getVertexShaderFadeIn(),r+="}",r}static buildFragmentShader(){let e=`
            precision highp float;
            #include <common>
 
            uniform vec3 debugColor;

            varying vec4 vColor;
            varying vec2 vUv;
            varying vec2 vPosition;
        `;return e+=`
            void main () {
                // Compute the positional squared distance from the center of the splat to the current fragment.
                float A = dot(vPosition, vPosition);
                // Since the positional data in vPosition has been scaled by sqrt(8), the squared result will be
                // scaled by a factor of 8. If the squared result is larger than 8, it means it is outside the ellipse
                // defined by the rectangle formed by vPosition. It also means it's farther
                // away than sqrt(8) standard deviations from the mean.
                if (A > 8.0) discard;
                vec3 color = vColor.rgb;

                // Since the rendered splat is scaled by sqrt(8), the inverse covariance matrix that is part of
                // the gaussian formula becomes the identity matrix. We're then left with (X - mean) * (X - mean),
                // and since 'mean' is zero, we have X * X, which is the same as A:
                float opacity = exp(-0.5 * A) * vColor.a;

                gl_FragColor = vec4(color.rgb, opacity);
            }
        `,e}}class Pl{static build(e=!1,t=!1,n=1,s=!1,r=0){let a=Jr.buildVertexShaderBase(e,t,r,`
            uniform vec2 scaleRotationsTextureSize;
            uniform highp sampler2D scaleRotationsTexture;
            varying mat3 vT;
            varying vec2 vQuadCenter;
            varying vec2 vFragCoord;
        `);a+=Pl.buildVertexShaderProjection();const l=Pl.buildFragmentShader(),c=Jr.getUniforms(e,t,r,n,s);return c.scaleRotationsTexture={type:"t",value:null},c.scaleRotationsTextureSize={type:"v2",value:new Ne(1024,1024)},new _n({uniforms:c,vertexShader:a,fragmentShader:l,transparent:!0,alphaTest:1,blending:Is,depthTest:!0,depthWrite:!1,side:ti})}static buildVertexShaderProjection(){let e=`

            vec4 scaleRotationA = texture(scaleRotationsTexture, getDataUVF(nearestEvenIndex, 1.5,
                                                                            oddOffset, scaleRotationsTextureSize));
            vec4 scaleRotationB = texture(scaleRotationsTexture, getDataUVF(nearestEvenIndex, 1.5,
                                                                            oddOffset + uint(1), scaleRotationsTextureSize));

            vec3 scaleRotation123 = vec3(scaleRotationA.rgb) * (1.0 - fOddOffset) +
                                    vec3(scaleRotationA.ba, scaleRotationB.r) * fOddOffset;
            vec3 scaleRotation456 = vec3(scaleRotationA.a, scaleRotationB.rg) * (1.0 - fOddOffset) +
                                    vec3(scaleRotationB.gba) * fOddOffset;

            float missingW = sqrt(1.0 - scaleRotation456.x * scaleRotation456.x - scaleRotation456.y *
                                    scaleRotation456.y - scaleRotation456.z * scaleRotation456.z);
            mat3 R = quaternionToRotationMatrix(scaleRotation456.r, scaleRotation456.g, scaleRotation456.b, missingW);
            mat3 S = mat3(scaleRotation123.r, 0.0, 0.0,
                            0.0, scaleRotation123.g, 0.0,
                            0.0, 0.0, scaleRotation123.b);
            
            mat3 L = R * S;

            mat3x4 splat2World = mat3x4(vec4(L[0], 0.0),
                                        vec4(L[1], 0.0),
                                        vec4(splatCenter.x, splatCenter.y, splatCenter.z, 1.0));

            mat4 world2ndc = transpose(projectionMatrix * transformModelViewMatrix);

            mat3x4 ndc2pix = mat3x4(vec4(viewport.x / 2.0, 0.0, 0.0, (viewport.x - 1.0) / 2.0),
                                    vec4(0.0, viewport.y / 2.0, 0.0, (viewport.y - 1.0) / 2.0),
                                    vec4(0.0, 0.0, 0.0, 1.0));

            mat3 T = transpose(splat2World) * world2ndc * ndc2pix;
            vec3 normal = vec3(viewMatrix * vec4(L[0][2], L[1][2], L[2][2], 0.0));
        `;return e+=`

                mat4 splat2World4 = mat4(vec4(L[0], 0.0),
                                        vec4(L[1], 0.0),
                                        vec4(L[2], 0.0),
                                        vec4(splatCenter.x, splatCenter.y, splatCenter.z, 1.0));

                mat4 Tt = transpose(transpose(splat2World4) * world2ndc);

                vec4 tempPoint1 = Tt * vec4(1.0, 0.0, 0.0, 1.0);
                tempPoint1 /= tempPoint1.w;

                vec4 tempPoint2 = Tt * vec4(0.0, 1.0, 0.0, 1.0);
                tempPoint2 /= tempPoint2.w;

                vec4 center = Tt * vec4(0.0, 0.0, 0.0, 1.0);
                center /= center.w;

                vec2 basisVector1 = tempPoint1.xy - center.xy;
                vec2 basisVector2 = tempPoint2.xy - center.xy;

                vec2 basisVector1Screen = basisVector1 * 0.5 * viewport;
                vec2 basisVector2Screen = basisVector2 * 0.5 * viewport;

                const float minPix = 1.;
                if (length(basisVector1Screen) < minPix || length(basisVector2Screen) < minPix) {
                    
            vec3 T0 = vec3(T[0][0], T[0][1], T[0][2]);
            vec3 T1 = vec3(T[1][0], T[1][1], T[1][2]);
            vec3 T3 = vec3(T[2][0], T[2][1], T[2][2]);

            vec3 tempPoint = vec3(1.0, 1.0, -1.0);
            float distance = (T3.x * T3.x * tempPoint.x) + (T3.y * T3.y * tempPoint.y) + (T3.z * T3.z * tempPoint.z);
            vec3 f = (1.0 / distance) * tempPoint;
            if (abs(distance) < 0.00001) return;

            float pointImageX = (T0.x * T3.x * f.x) + (T0.y * T3.y * f.y) + (T0.z * T3.z * f.z);
            float pointImageY = (T1.x * T3.x * f.x) + (T1.y * T3.y * f.y) + (T1.z * T3.z * f.z);
            vec2 pointImage = vec2(pointImageX, pointImageY);

            float tempX = (T0.x * T0.x * f.x) + (T0.y * T0.y * f.y) + (T0.z * T0.z * f.z);
            float tempY = (T1.x * T1.x * f.x) + (T1.y * T1.y * f.y) + (T1.z * T1.z * f.z);
            vec2 temp = vec2(tempX, tempY);

            vec2 halfExtend = pointImage * pointImage - temp;
            vec2 extent = sqrt(max(vec2(0.0001), halfExtend));
            float radius = max(extent.x, extent.y);

            vec2 ndcOffset = ((position.xy * radius * 3.0) * basisViewport * 2.0);

            vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
            gl_Position = quadPos;

            vT = T;
            vQuadCenter = pointImage;
            vFragCoord = (quadPos.xy * 0.5 + 0.5) * viewport;
        
                } else {
                    vec2 ndcOffset = vec2(position.x * basisVector1 + position.y * basisVector2) * 3.0 * inverseFocalAdjustment;
                    vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
                    gl_Position = quadPos;

                    vT = T;
                    vQuadCenter = center.xy;
                    vFragCoord = (quadPos.xy * 0.5 + 0.5) * viewport;
                }
            `,e+=Jr.getVertexShaderFadeIn(),e+="}",e}static buildFragmentShader(){return`
            precision highp float;
            #include <common>

            uniform vec3 debugColor;

            varying vec4 vColor;
            varying vec2 vUv;
            varying vec2 vPosition;
            varying mat3 vT;
            varying vec2 vQuadCenter;
            varying vec2 vFragCoord;

            void main () {

                const float FilterInvSquare = 2.0;
                const float near_n = 0.2;
                const float T = 1.0;

                vec2 xy = vQuadCenter;
                vec3 Tu = vT[0];
                vec3 Tv = vT[1];
                vec3 Tw = vT[2];
                vec3 k = vFragCoord.x * Tw - Tu;
                vec3 l = vFragCoord.y * Tw - Tv;
                vec3 p = cross(k, l);
                if (p.z == 0.0) discard;
                vec2 s = vec2(p.x / p.z, p.y / p.z);
                float rho3d = (s.x * s.x + s.y * s.y); 
                vec2 d = vec2(xy.x - vFragCoord.x, xy.y - vFragCoord.y);
                float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

                // compute intersection and depth
                float rho = min(rho3d, rho2d);
                float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
                if (depth < near_n) discard;
                //  vec4 nor_o = collected_normal_opacity[j];
                //  float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
                float opa = vColor.a;

                float power = -0.5f * rho;
                if (power > 0.0f) discard;

                // Eq. (2) from 3D Gaussian splatting paper.
                // Obtain alpha by multiplying with Gaussian opacity
                // and its exponential falloff from mean.
                // Avoid numerical instabilities (see paper appendix). 
                float alpha = min(0.99f, opa * exp(power));
                if (alpha < 1.0f / 255.0f) discard;
                float test_T = T * (1.0 - alpha);
                if (test_T < 0.0001)discard;

                float w = alpha * T;
                gl_FragColor = vec4(vColor.rgb, w);
            }
        `}}class uE{static build(e){const t=new An;t.setIndex([0,1,2,0,2,3]);const n=new Float32Array(12),s=new li(n,3);t.setAttribute("position",s),s.setXYZ(0,-1,-1,0),s.setXYZ(1,-1,1,0),s.setXYZ(2,1,1,0),s.setXYZ(3,1,-1,0),s.needsUpdate=!0;const r=new pv().copy(t),o=new Uint32Array(e),a=new ov(o,1,!1);return a.setUsage(wS),r.setAttribute("splatIndex",a),r.instanceCount=0,r}}class fE extends Wt{constructor(e,t=new L,n=new bt,s=new L(1,1,1),r=1,o=1,a=!0){super(),this.splatBuffer=e,this.position.copy(t),this.quaternion.copy(n),this.scale.copy(s),this.transform=new qe,this.minimumAlpha=r,this.opacity=o,this.visible=a}copyTransformData(e){this.position.copy(e.position),this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.transform.copy(e.transform)}updateTransform(e){e?(this.matrixWorldAutoUpdate&&this.updateWorldMatrix(!0,!1),this.transform.copy(this.matrixWorld)):(this.matrixAutoUpdate&&this.updateMatrix(),this.transform.copy(this.matrix))}}class Sd{static idGen=0;constructor(e,t,n,s){this.min=new L().copy(e),this.max=new L().copy(t),this.boundingBox=new wi(this.min,this.max),this.center=new L().copy(this.max).sub(this.min).multiplyScalar(.5).add(this.min),this.depth=n,this.children=[],this.data=null,this.id=s||Sd.idGen++}}class Qo{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.sceneDimensions=new L,this.sceneMin=new L,this.sceneMax=new L,this.rootNode=null,this.nodesWithIndexes=[],this.splatMesh=null}static convertWorkerSubTreeNode(e){const t=new L().fromArray(e.min),n=new L().fromArray(e.max),s=new Sd(t,n,e.depth,e.id);if(e.data.indexes){s.data={indexes:[]};for(let r of e.data.indexes)s.data.indexes.push(r)}if(e.children)for(let r of e.children)s.children.push(Qo.convertWorkerSubTreeNode(r));return s}static convertWorkerSubTree(e,t){const n=new Qo(e.maxDepth,e.maxCentersPerNode);n.sceneMin=new L().fromArray(e.sceneMin),n.sceneMax=new L().fromArray(e.sceneMax),n.splatMesh=t,n.rootNode=Qo.convertWorkerSubTreeNode(e.rootNode);const s=(r,o)=>{r.children.length===0&&o(r);for(let a of r.children)s(a,o)};return n.nodesWithIndexes=[],s(n.rootNode,r=>{r.data&&r.data.indexes&&r.data.indexes.length>0&&n.nodesWithIndexes.push(r)}),n}}function dE(i){let e=0;class t{constructor(l,c){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]]}containsPoint(l){return l[0]>=this.min[0]&&l[0]<=this.max[0]&&l[1]>=this.min[1]&&l[1]<=this.max[1]&&l[2]>=this.min[2]&&l[2]<=this.max[2]}}class n{constructor(l,c){this.maxDepth=l,this.maxCentersPerNode=c,this.sceneDimensions=[],this.sceneMin=[],this.sceneMax=[],this.rootNode=null,this.addedIndexes={},this.nodesWithIndexes=[],this.splatMesh=null,this.disposed=!1}}class s{constructor(l,c,u,f){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]],this.center=[(c[0]-l[0])*.5+l[0],(c[1]-l[1])*.5+l[1],(c[2]-l[2])*.5+l[2]],this.depth=u,this.children=[],this.data=null,this.id=f||e++}}processSplatTreeNode=function(a,l,c,u){const f=l.data.indexes.length;if(f<a.maxCentersPerNode||l.depth>a.maxDepth){const _=[];for(let A=0;A<l.data.indexes.length;A++)a.addedIndexes[l.data.indexes[A]]||(_.push(l.data.indexes[A]),a.addedIndexes[l.data.indexes[A]]=!0);l.data.indexes=_,l.data.indexes.sort((A,S)=>A>S?1:-1),a.nodesWithIndexes.push(l);return}const d=[l.max[0]-l.min[0],l.max[1]-l.min[1],l.max[2]-l.min[2]],h=[d[0]*.5,d[1]*.5,d[2]*.5],g=[l.min[0]+h[0],l.min[1]+h[1],l.min[2]+h[2]],p=[new t([g[0]-h[0],g[1],g[2]-h[2]],[g[0],g[1]+h[1],g[2]]),new t([g[0],g[1],g[2]-h[2]],[g[0]+h[0],g[1]+h[1],g[2]]),new t([g[0],g[1],g[2]],[g[0]+h[0],g[1]+h[1],g[2]+h[2]]),new t([g[0]-h[0],g[1],g[2]],[g[0],g[1]+h[1],g[2]+h[2]]),new t([g[0]-h[0],g[1]-h[1],g[2]-h[2]],[g[0],g[1],g[2]]),new t([g[0],g[1]-h[1],g[2]-h[2]],[g[0]+h[0],g[1],g[2]]),new t([g[0],g[1]-h[1],g[2]],[g[0]+h[0],g[1],g[2]+h[2]]),new t([g[0]-h[0],g[1]-h[1],g[2]],[g[0],g[1],g[2]+h[2]])],x=[];for(let _=0;_<p.length;_++)x[_]=[];const m=[0,0,0];for(let _=0;_<f;_++){const A=l.data.indexes[_],S=c[A];m[0]=u[S],m[1]=u[S+1],m[2]=u[S+2];for(let v=0;v<p.length;v++)p[v].containsPoint(m)&&x[v].push(A)}for(let _=0;_<p.length;_++){const A=new s(p[_].min,p[_].max,l.depth+1);A.data={indexes:x[_]},l.children.push(A)}l.data={};for(let _ of l.children)processSplatTreeNode(a,_,c,u)};const r=(a,l,c)=>{const u=[0,0,0],f=[0,0,0],d=[],h=Math.floor(a.length/4);for(let p=0;p<h;p++){const x=p*4,m=a[x],_=a[x+1],A=a[x+2],S=Math.round(a[x+3]);(p===0||m<u[0])&&(u[0]=m),(p===0||m>f[0])&&(f[0]=m),(p===0||_<u[1])&&(u[1]=_),(p===0||_>f[1])&&(f[1]=_),(p===0||A<u[2])&&(u[2]=A),(p===0||A>f[2])&&(f[2]=A),d.push(S)}const g=new n(l,c);return g.sceneMin=u,g.sceneMax=f,g.rootNode=new s(g.sceneMin,g.sceneMax,0),g.rootNode.data={indexes:d},g};function o(a,l,c){const u=[];for(let d of a){const h=Math.floor(d.length/4);for(let g=0;g<h;g++){const p=g*4,x=Math.round(d[p+3]);u[x]=p}}const f=[];for(let d of a){const h=r(d,l,c);f.push(h),processSplatTreeNode(h,h.rootNode,u,d)}i.postMessage({subTrees:f})}i.onmessage=a=>{a.data.process&&o(a.data.process.centers,a.data.process.maxDepth,a.data.process.maxCentersPerNode)}}function hE(i,e,t,n,s){i.postMessage({process:{centers:e,maxDepth:n,maxCentersPerNode:s}},t)}function pE(){return new Worker(URL.createObjectURL(new Blob(["(",dE.toString(),")(self)"],{type:"application/javascript"})))}class mE{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.subTrees=[],this.splatMesh=null}dispose(){this.diposeSplatTreeWorker(),this.disposed=!0}diposeSplatTreeWorker(){this.splatTreeWorker&&this.splatTreeWorker.terminate(),this.splatTreeWorker=null}processSplatMesh=function(e,t=()=>!0,n,s){this.splatTreeWorker||(this.splatTreeWorker=pE()),this.splatMesh=e,this.subTrees=[];const r=new L,o=(a,l)=>{const c=new Float32Array(l*4);let u=0;for(let f=0;f<l;f++){const d=f+a;if(t(d)){e.getSplatCenter(d,r);const h=u*4;c[h]=r.x,c[h+1]=r.y,c[h+2]=r.z,c[h+3]=d,u++}}return c};return new Promise(a=>{const l=()=>this.disposed?(this.diposeSplatTreeWorker(),a(),!0):!1;n&&n(!1),Gn(()=>{if(l())return;const c=[];if(e.dynamicMode){let u=0;for(let f=0;f<e.scenes.length;f++){const h=e.getScene(f).splatBuffer.getSplatCount(),g=o(u,h);c.push(g),u+=h}}else{const u=o(0,e.getSplatCount());c.push(u)}this.splatTreeWorker.onmessage=u=>{l()||u.data.subTrees&&(s&&s(!1),Gn(()=>{if(!l()){for(let f of u.data.subTrees){const d=Qo.convertWorkerSubTree(f,e);this.subTrees.push(d)}this.diposeSplatTreeWorker(),s&&s(!0),Gn(()=>{a()})}}))},Gn(()=>{if(l())return;n&&n(!0);const u=c.map(f=>f.buffer);hE(this.splatTreeWorker,c,u,this.maxDepth,this.maxCentersPerNode)})})})};countLeaves(){let e=0;return this.visitLeaves(()=>{e++}),e}visitLeaves(e){const t=(n,s)=>{n.children.length===0&&s(n);for(let r of n.children)t(r,s)};for(let n of this.subTrees)t(n.rootNode,e)}}function gE(i){const e={};function t(n){if(e[n]!==void 0)return e[n];let s;switch(n){case"WEBGL_depth_texture":s=i.getExtension("WEBGL_depth_texture")||i.getExtension("MOZ_WEBGL_depth_texture")||i.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":s=i.getExtension("EXT_texture_filter_anisotropic")||i.getExtension("MOZ_EXT_texture_filter_anisotropic")||i.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":s=i.getExtension("WEBGL_compressed_texture_s3tc")||i.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":s=i.getExtension("WEBGL_compressed_texture_pvrtc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:s=i.getExtension(n)}return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(n){n.isWebGL2?(t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance")):(t("WEBGL_depth_texture"),t("OES_texture_float"),t("OES_texture_half_float"),t("OES_texture_half_float_linear"),t("OES_standard_derivatives"),t("OES_element_index_uint"),t("OES_vertex_array_object"),t("ANGLE_instanced_arrays")),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture")},get:function(n){const s=t(n);return s===null&&console.warn("THREE.WebGLRenderer: "+n+" extension not supported."),s}}}function xE(i,e,t){let n;function s(){if(n!==void 0)return n;if(e.has("EXT_texture_filter_anisotropic")===!0){const M=e.get("EXT_texture_filter_anisotropic");n=i.getParameter(M.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else n=0;return n}function r(M){if(M==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";M="mediump"}return M==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}const o=typeof WebGL2RenderingContext<"u"&&i.constructor.name==="WebGL2RenderingContext";let a=t.precision!==void 0?t.precision:"highp";const l=r(a);l!==a&&(console.warn("THREE.WebGLRenderer:",a,"not supported, using",l,"instead."),a=l);const c=o||e.has("WEBGL_draw_buffers"),u=t.logarithmicDepthBuffer===!0,f=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),d=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),h=i.getParameter(i.MAX_TEXTURE_SIZE),g=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),p=i.getParameter(i.MAX_VERTEX_ATTRIBS),x=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),m=i.getParameter(i.MAX_VARYING_VECTORS),_=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),A=d>0,S=o||e.has("OES_texture_float"),v=A&&S,b=o?i.getParameter(i.MAX_SAMPLES):0;return{isWebGL2:o,drawBuffers:c,getMaxAnisotropy:s,getMaxPrecision:r,precision:a,logarithmicDepthBuffer:u,maxTextures:f,maxVertexTextures:d,maxTextureSize:h,maxCubemapSize:g,maxAttributes:p,maxVertexUniforms:x,maxVaryings:m,maxFragmentUniforms:_,vertexTextures:A,floatFragmentTextures:S,floatVertexTextures:v,maxSamples:b}}const Yo={Default:0,Instant:2},eo={None:0,Info:3},qp=new An,_E=new hr,rl=6,AE=4,SE=4,vE=4,yE=6,bE=8,eu=4,tu=4,Qp=1,ME=.012,CE=.003,Yp=1,Kp=16777216;class $t extends Vt{constructor(e=$i.ThreeD,t=!1,n=!1,s=!1,r=1,o=!0,a=!1,l=!1,c=1024,u=eo.None,f=0,d=1,h=.3){super(qp,_E),this.renderer=void 0,this.splatRenderMode=e,this.dynamicMode=t,this.enableOptionalEffects=n,this.halfPrecisionCovariancesOnGPU=s,this.devicePixelRatio=r,this.enableDistancesComputationOnGPU=o,this.integerBasedDistancesComputation=a,this.antialiased=l,this.kernel2DSize=h,this.maxScreenSpaceSplatSize=c,this.logLevel=u,this.sphericalHarmonicsDegree=f,this.minSphericalHarmonicsDegree=0,this.sceneFadeInRateMultiplier=d,this.scenes=[],this.splatTree=null,this.baseSplatTree=null,this.splatDataTextures={},this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new wi,this.calculatedSceneCenter=new L,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!1,this.lastRenderer=null,this.visible=!1}static buildScenes(e,t,n){const s=[];s.length=t.length;for(let r=0;r<t.length;r++){const o=t[r],a=n[r]||{};let l=a.position||[0,0,0],c=a.rotation||[0,0,0,1],u=a.scale||[1,1,1];const f=new L().fromArray(l),d=new bt().fromArray(c),h=new L().fromArray(u),g=$t.createScene(o,f,d,h,a.splatAlphaRemovalThreshold||1,a.opacity,a.visible);e.add(g),s[r]=g}return s}static createScene(e,t,n,s,r,o=1,a=!0){return new fE(e,t,n,s,r,o,a)}static buildSplatIndexMaps(e){const t=[],n=[];let s=0;for(let r=0;r<e.length;r++){const a=e[r].getMaxSplatCount();for(let l=0;l<a;l++)t[s]=l,n[s]=r,s++}return{localSplatIndexMap:t,sceneIndexMap:n}}buildSplatTree=function(e=[],t,n){return new Promise(s=>{this.disposeSplatTree(),this.baseSplatTree=new mE(8,1e3);const r=performance.now(),o=new Et;this.baseSplatTree.processSplatMesh(this,a=>{this.getSplatColor(a,o);const l=this.getSceneIndexForSplat(a),c=e[l]||1;return o.w>=c},t,n).then(()=>{const a=performance.now()-r;if(this.logLevel>=eo.Info&&console.log("SplatTree build: "+a+" ms"),this.disposed)s();else{this.splatTree=this.baseSplatTree,this.baseSplatTree=null;let l=0,c=0,u=0;this.splatTree.visitLeaves(f=>{const d=f.data.indexes.length;d>0&&(c+=d,u++,l++)}),this.logLevel>=eo.Info&&(console.log(`SplatTree leaves: ${this.splatTree.countLeaves()}`),console.log(`SplatTree leaves with splats:${l}`),c=c/u,console.log(`Avg splat count per node: ${c}`),console.log(`Total splat count: ${this.getSplatCount()}`)),s()}})})};build(e,t,n=!0,s=!1,r,o,a=!0){this.sceneOptions=t,this.finalBuild=s;const l=$t.getTotalMaxSplatCountForSplatBuffers(e),c=$t.buildScenes(this,e,t);if(n)for(let p=0;p<this.scenes.length&&p<c.length;p++){const x=c[p],m=this.getScene(p);x.copyTransformData(m)}this.scenes=c;let u=3;for(let p of e){const x=p.getMinSphericalHarmonicsDegree();x<u&&(u=x)}this.minSphericalHarmonicsDegree=Math.min(u,this.sphericalHarmonicsDegree);let f=!1;if(e.length!==this.lastBuildScenes.length)f=!0;else for(let p=0;p<e.length;p++)if(e[p]!==this.lastBuildScenes[p].splatBuffer){f=!0;break}let d=!0;if((this.scenes.length!==1||this.lastBuildSceneCount!==this.scenes.length||this.lastBuildMaxSplatCount!==l||f)&&(d=!1),!d){this.boundingBox=new wi,a||(this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.firstRenderTime=-1),this.lastBuildScenes=[],this.lastBuildSplatCount=0,this.lastBuildMaxSplatCount=0,this.disposeMeshData(),this.geometry=uE.build(l),this.splatRenderMode===$i.ThreeD?this.material=Dl.build(this.dynamicMode,this.enableOptionalEffects,this.antialiased,this.maxScreenSpaceSplatSize,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree,this.kernel2DSize):this.material=Pl.build(this.dynamicMode,this.enableOptionalEffects,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree);const p=$t.buildSplatIndexMaps(e);this.globalSplatIndexToLocalSplatIndexMap=p.localSplatIndexMap,this.globalSplatIndexToSceneIndexMap=p.sceneIndexMap}const h=this.getSplatCount(!0);this.enableDistancesComputationOnGPU&&this.setupDistancesComputationTransformFeedback();const g=this.refreshGPUDataFromSplatBuffers(d);for(let p=0;p<this.scenes.length;p++)this.lastBuildScenes[p]=this.scenes[p];return this.lastBuildSplatCount=h,this.lastBuildMaxSplatCount=this.getMaxSplatCount(),this.lastBuildSceneCount=this.scenes.length,s&&this.scenes.length>0&&this.buildSplatTree(t.map(p=>p.splatAlphaRemovalThreshold||1),r,o).then(()=>{this.onSplatTreeReadyCallback&&this.onSplatTreeReadyCallback(this.splatTree),this.onSplatTreeReadyCallback=null}),this.visible=this.scenes.length>0,g}freeIntermediateSplatData(){const e=t=>{delete t.source.data,delete t.image,t.onUpdate=null};delete this.splatDataTextures.baseData.covariances,delete this.splatDataTextures.baseData.centers,delete this.splatDataTextures.baseData.colors,delete this.splatDataTextures.baseData.sphericalHarmonics,delete this.splatDataTextures.centerColors.data,delete this.splatDataTextures.covariances.data,this.splatDataTextures.sphericalHarmonics&&delete this.splatDataTextures.sphericalHarmonics.data,this.splatDataTextures.sceneIndexes&&delete this.splatDataTextures.sceneIndexes.data,this.splatDataTextures.centerColors.texture.needsUpdate=!0,this.splatDataTextures.centerColors.texture.onUpdate=()=>{e(this.splatDataTextures.centerColors.texture)},this.splatDataTextures.covariances.texture.needsUpdate=!0,this.splatDataTextures.covariances.texture.onUpdate=()=>{e(this.splatDataTextures.covariances.texture)},this.splatDataTextures.sphericalHarmonics&&(this.splatDataTextures.sphericalHarmonics.texture?(this.splatDataTextures.sphericalHarmonics.texture.needsUpdate=!0,this.splatDataTextures.sphericalHarmonics.texture.onUpdate=()=>{e(this.splatDataTextures.sphericalHarmonics.texture)}):this.splatDataTextures.sphericalHarmonics.textures.forEach(t=>{t.needsUpdate=!0,t.onUpdate=()=>{e(t)}})),this.splatDataTextures.sceneIndexes&&(this.splatDataTextures.sceneIndexes.texture.needsUpdate=!0,this.splatDataTextures.sceneIndexes.texture.onUpdate=()=>{e(this.splatDataTextures.sceneIndexes.texture)})}dispose(){this.disposeMeshData(),this.disposeTextures(),this.disposeSplatTree(),this.enableDistancesComputationOnGPU&&(this.computeDistancesOnGPUSyncTimeout&&(clearTimeout(this.computeDistancesOnGPUSyncTimeout),this.computeDistancesOnGPUSyncTimeout=null),this.disposeDistancesComputationGPUResources()),this.scenes=[],this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.renderer=null,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new wi,this.calculatedSceneCenter=new L,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!0,this.lastRenderer=null,this.visible=!1}disposeMeshData(){this.geometry&&this.geometry!==qp&&(this.geometry.dispose(),this.geometry=null),this.material&&(this.material.dispose(),this.material=null)}disposeTextures(){for(let e in this.splatDataTextures)if(this.splatDataTextures.hasOwnProperty(e)){const t=this.splatDataTextures[e];t.texture&&(t.texture.dispose(),t.texture=null)}this.splatDataTextures=null}disposeSplatTree(){this.splatTree&&(this.splatTree.dispose(),this.splatTree=null),this.baseSplatTree&&(this.baseSplatTree.dispose(),this.baseSplatTree=null)}getSplatTree(){return this.splatTree}onSplatTreeReady(e){this.onSplatTreeReadyCallback=e}getDataForDistancesComputation(e,t){const n=this.integerBasedDistancesComputation?this.getIntegerCenters(e,t,!0):this.getFloatCenters(e,t,!0),s=this.getSceneIndexes(e,t);return{centers:n,sceneIndexes:s}}refreshGPUDataFromSplatBuffers(e){const t=this.getSplatCount(!0);this.refreshDataTexturesFromSplatBuffers(e);const n=e?this.lastBuildSplatCount:0,{centers:s,sceneIndexes:r}=this.getDataForDistancesComputation(n,t-1);return this.enableDistancesComputationOnGPU&&this.refreshGPUBuffersForDistancesComputation(s,r,e),{from:n,to:t-1,count:t-n,centers:s,sceneIndexes:r}}refreshGPUBuffersForDistancesComputation(e,t,n=!1){const s=n?this.lastBuildSplatCount:0;this.updateGPUCentersBufferForDistancesComputation(n,e,s),this.updateGPUTransformIndexesBufferForDistancesComputation(n,t,s)}refreshDataTexturesFromSplatBuffers(e){const t=this.getSplatCount(!0),n=this.lastBuildSplatCount,s=t-1;e?this.updateBaseDataFromSplatBuffers(n,s):(this.setupDataTextures(),this.updateBaseDataFromSplatBuffers()),this.updateDataTexturesFromBaseData(n,s),this.updateVisibleRegion(e)}setupDataTextures(){const e=this.getMaxSplatCount(),t=this.getSplatCount(!0);this.disposeTextures();const n=(M,E)=>{const y=new Ne(4096,1024);for(;y.x*y.y*M<e*E;)y.y*=2;return y},s=M=>M>=1?yE:SE,r=M=>{const E=s(M),y=n(E,6);return{elementsPerTexelStored:E,texSize:y}};let o=this.getTargetCovarianceCompressionLevel();const a=0,l=this.getTargetSphericalHarmonicsCompressionLevel();let c,u,f;if(this.splatRenderMode===$i.ThreeD){const M=r(o);M.texSize.x*M.texSize.y>Kp&&o===0&&(o=1),c=new Float32Array(e*rl)}else u=new Float32Array(e*3),f=new Float32Array(e*4);const d=new Float32Array(e*3),h=new Uint8Array(e*4);let g=Float32Array;l===1?g=Uint16Array:l===2&&(g=Uint8Array);const p=Zr(this.minSphericalHarmonicsDegree),x=this.minSphericalHarmonicsDegree?new g(e*p):void 0,m=n(tu,4),_=new Uint32Array(m.x*m.y*tu);$t.updateCenterColorsPaddedData(0,t-1,d,h,_);const A=new Qi(_,m.x,m.y,jr,si);if(A.internalFormat="RGBA32UI",A.needsUpdate=!0,this.material.uniforms.centersColorsTexture.value=A,this.material.uniforms.centersColorsTextureSize.value.copy(m),this.material.uniformsNeedUpdate=!0,this.splatDataTextures={baseData:{covariances:c,scales:u,rotations:f,centers:d,colors:h,sphericalHarmonics:x},centerColors:{data:_,texture:A,size:m}},this.splatRenderMode===$i.ThreeD){const M=r(o),E=M.elementsPerTexelStored,y=M.texSize;let C=o>=1?Uint32Array:Float32Array;const I=o>=1?bE:vE,P=new C(y.x*y.y*I);o===0?P.set(c):$t.updatePaddedCompressedCovariancesTextureData(c,P,0,0,c.length);let U;if(o>=1)U=new Qi(P,y.x,y.y,jr,si),U.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=U;else{U=new Qi(P,y.x,y.y,gn,mi),this.material.uniforms.covariancesTexture.value=U;const O=new Qi(new Uint32Array(32),2,2,jr,si);O.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=O,O.needsUpdate=!0}U.needsUpdate=!0,this.material.uniforms.covariancesAreHalfFloat.value=o>=1?1:0,this.material.uniforms.covariancesTextureSize.value.copy(y),this.splatDataTextures.covariances={data:P,texture:U,size:y,compressionLevel:o,elementsPerTexelStored:E,elementsPerTexelAllocated:I}}else{const E=n(eu,6);let y=Float32Array,C=mi;const I=new y(E.x*E.y*eu);$t.updateScaleRotationsPaddedData(0,t-1,u,f,I);const P=new Qi(I,E.x,E.y,gn,C);P.needsUpdate=!0,this.material.uniforms.scaleRotationsTexture.value=P,this.material.uniforms.scaleRotationsTextureSize.value.copy(E),this.splatDataTextures.scaleRotations={data:I,texture:P,size:E,compressionLevel:a}}if(x){const M=l===2?Ui:pr;let E=p;E%2!==0&&E++;const y=4,C=gn;let I=n(y,E);if(I.x*I.y<=Kp){const P=I.x*I.y*y,U=new g(P);for(let k=0;k<t;k++){const z=p*k,W=E*k;for(let H=0;H<p;H++)U[W+H]=x[z+H]}const O=new Qi(U,I.x,I.y,C,M);O.needsUpdate=!0,this.material.uniforms.sphericalHarmonicsTexture.value=O,this.splatDataTextures.sphericalHarmonics={componentCount:p,paddedComponentCount:E,data:U,textureCount:1,texture:O,size:I,compressionLevel:l,elementsPerTexel:y}}else{const P=p/3;E=P,E%2!==0&&E++,I=n(y,E);const U=I.x*I.y*y,O=[this.material.uniforms.sphericalHarmonicsTextureR,this.material.uniforms.sphericalHarmonicsTextureG,this.material.uniforms.sphericalHarmonicsTextureB],k=[],z=[];for(let W=0;W<3;W++){const H=new g(U);k.push(H);for(let X=0;X<t;X++){const Z=p*X,pe=E*X;if(P>=3){for(let ge=0;ge<3;ge++)H[pe+ge]=x[Z+W*3+ge];if(P>=8)for(let ge=0;ge<5;ge++)H[pe+3+ge]=x[Z+9+W*5+ge]}}const $=new Qi(H,I.x,I.y,C,M);z.push($),$.needsUpdate=!0,O[W].value=$}this.material.uniforms.sphericalHarmonicsMultiTextureMode.value=1,this.splatDataTextures.sphericalHarmonics={componentCount:p,componentCountPerChannel:P,paddedComponentCount:E,data:k,textureCount:3,textures:z,size:I,compressionLevel:l,elementsPerTexel:y}}this.material.uniforms.sphericalHarmonicsTextureSize.value.copy(I),this.material.uniforms.sphericalHarmonics8BitMode.value=l===2?1:0;for(let P=0;P<this.scenes.length;P++){const U=this.scenes[P].splatBuffer;this.material.uniforms.sphericalHarmonics8BitCompressionRangeMin.value[P]=U.minSphericalHarmonicsCoeff,this.material.uniforms.sphericalHarmonics8BitCompressionRangeMax.value[P]=U.maxSphericalHarmonicsCoeff}this.material.uniformsNeedUpdate=!0}const S=n(Qp,4),v=new Uint32Array(S.x*S.y*Qp);for(let M=0;M<t;M++)v[M]=this.globalSplatIndexToSceneIndexMap[M];const b=new Qi(v,S.x,S.y,jl,si);b.internalFormat="R32UI",b.needsUpdate=!0,this.material.uniforms.sceneIndexesTexture.value=b,this.material.uniforms.sceneIndexesTextureSize.value.copy(S),this.material.uniformsNeedUpdate=!0,this.splatDataTextures.sceneIndexes={data:v,texture:b,size:S},this.material.uniforms.sceneCount.value=this.scenes.length}updateBaseDataFromSplatBuffers(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0;this.fillSplatDataArrays(this.splatDataTextures.baseData.covariances,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,this.splatDataTextures.baseData.sphericalHarmonics,void 0,s,o,l,e,t,e)}updateDataTexturesFromBaseData(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0,c=this.splatDataTextures.centerColors,u=c.data,f=c.texture;$t.updateCenterColorsPaddedData(e,t,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,u);const d=this.renderer?this.renderer.properties.get(f):null;if(!d||!d.__webglTexture?f.needsUpdate=!0:this.updateDataTexture(u,c.texture,c.size,d,tu,AE,4,e,t),n){const _=n.texture,A=e*rl,S=t*rl;if(s===0)for(let b=A;b<=S;b++){const M=this.splatDataTextures.baseData.covariances[b];n.data[b]=M}else $t.updatePaddedCompressedCovariancesTextureData(this.splatDataTextures.baseData.covariances,n.data,e*n.elementsPerTexelAllocated,A,S);const v=this.renderer?this.renderer.properties.get(_):null;!v||!v.__webglTexture?_.needsUpdate=!0:s===0?this.updateDataTexture(n.data,n.texture,n.size,v,n.elementsPerTexelStored,rl,4,e,t):this.updateDataTexture(n.data,n.texture,n.size,v,n.elementsPerTexelAllocated,n.elementsPerTexelAllocated,2,e,t)}if(r){const _=r.data,A=r.texture,S=6,v=o===0?4:2;$t.updateScaleRotationsPaddedData(e,t,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,_);const b=this.renderer?this.renderer.properties.get(A):null;!b||!b.__webglTexture?A.needsUpdate=!0:this.updateDataTexture(_,r.texture,r.size,b,eu,S,v,e,t)}const h=this.splatDataTextures.baseData.sphericalHarmonics;if(h){let _=4;l===1?_=2:l===2&&(_=1);const A=(b,M,E,y,C)=>{const I=this.renderer?this.renderer.properties.get(b):null;!I||!I.__webglTexture?b.needsUpdate=!0:this.updateDataTexture(y,b,M,I,E,C,_,e,t)},S=a.componentCount,v=a.paddedComponentCount;if(a.textureCount===1){const b=a.data;for(let M=e;M<=t;M++){const E=S*M,y=v*M;for(let C=0;C<S;C++)b[y+C]=h[E+C]}A(a.texture,a.size,a.elementsPerTexel,b,v)}else{const b=a.componentCountPerChannel;for(let M=0;M<3;M++){const E=a.data[M];for(let y=e;y<=t;y++){const C=S*y,I=v*y;if(b>=3){for(let P=0;P<3;P++)E[I+P]=h[C+M*3+P];if(b>=8)for(let P=0;P<5;P++)E[I+3+P]=h[C+9+M*5+P]}}A(a.textures[M],a.size,a.elementsPerTexel,E,v)}}}const g=this.splatDataTextures.sceneIndexes,p=g.data;for(let _=this.lastBuildSplatCount;_<=t;_++)p[_]=this.globalSplatIndexToSceneIndexMap[_];const x=g.texture,m=this.renderer?this.renderer.properties.get(x):null;!m||!m.__webglTexture?x.needsUpdate=!0:this.updateDataTexture(p,g.texture,g.size,m,1,1,1,this.lastBuildSplatCount,t)}getTargetCovarianceCompressionLevel(){return this.halfPrecisionCovariancesOnGPU?1:0}getTargetSphericalHarmonicsCompressionLevel(){return Math.max(1,this.getMaximumSplatBufferCompressionLevel())}getMaximumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel>e)&&(e=s.compressionLevel)}return e}getMinimumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel<e)&&(e=s.compressionLevel)}return e}static computeTextureUpdateRegion(e,t,n,s,r){const o=r/s,a=e*o,l=Math.floor(a/n),c=l*n*s,u=t*o,f=Math.floor(u/n),d=f*n*s+n*s;return{dataStart:c,dataEnd:d,startRow:l,endRow:f}}updateDataTexture(e,t,n,s,r,o,a,l,c){const u=this.renderer.getContext(),f=$t.computeTextureUpdateRegion(l,c,n.x,r,o),d=f.dataEnd-f.dataStart,h=new e.constructor(e.buffer,f.dataStart*a,d),g=f.endRow-f.startRow+1,p=this.webGLUtils.convert(t.type),x=this.webGLUtils.convert(t.format,t.colorSpace),m=u.getParameter(u.TEXTURE_BINDING_2D);u.bindTexture(u.TEXTURE_2D,s.__webglTexture),u.texSubImage2D(u.TEXTURE_2D,0,0,f.startRow,n.x,g,x,p,h),u.bindTexture(u.TEXTURE_2D,m)}static updatePaddedCompressedCovariancesTextureData(e,t,n,s,r){let o=new DataView(t.buffer),a=n,l=0;for(let c=s;c<=r;c+=2)o.setUint16(a*2,e[c],!0),o.setUint16(a*2+2,e[c+1],!0),a+=2,l++,l>=3&&(a+=2,l=0)}static updateCenterColorsPaddedData(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*4,l=o*3,c=o*4;r[c]=nT(s,a),r[c+1]=Qc(n[l]),r[c+2]=Qc(n[l+1]),r[c+3]=Qc(n[l+2])}}static updateScaleRotationsPaddedData(e,t,n,s,r){for(let a=e;a<=t;a++){const l=a*3,c=a*4,u=a*6;r[u]=n[l],r[u+1]=n[l+1],r[u+2]=n[l+2],r[u+3]=s[c],r[u+4]=s[c+1],r[u+5]=s[c+2]}}updateVisibleRegion(e){const t=this.getSplatCount(!0),n=new L;if(!e){const r=new L;this.scenes.forEach(o=>{r.add(o.splatBuffer.sceneCenter)}),r.multiplyScalar(1/this.scenes.length),this.calculatedSceneCenter.copy(r),this.material.uniforms.sceneCenter.value.copy(this.calculatedSceneCenter),this.material.uniformsNeedUpdate=!0}const s=e?this.lastBuildSplatCount:0;for(let r=s;r<t;r++){this.getSplatCenter(r,n,!0);const o=n.sub(this.calculatedSceneCenter).length();o>this.maxSplatDistanceFromSceneCenter&&(this.maxSplatDistanceFromSceneCenter=o)}this.maxSplatDistanceFromSceneCenter-this.visibleRegionBufferRadius>Yp&&(this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter,this.visibleRegionRadius=Math.max(this.visibleRegionBufferRadius-Yp,0)),this.finalBuild&&(this.visibleRegionRadius=this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter),this.updateVisibleRegionFadeDistance()}updateVisibleRegionFadeDistance(e=Yo.Default){const t=ME*this.sceneFadeInRateMultiplier,n=CE*this.sceneFadeInRateMultiplier,s=this.finalBuild?t:n,r=e===Yo.Default?s:n;this.visibleRegionFadeStartRadius=(this.visibleRegionRadius-this.visibleRegionFadeStartRadius)*r+this.visibleRegionFadeStartRadius;const a=(this.visibleRegionBufferRadius>0?this.visibleRegionFadeStartRadius/this.visibleRegionBufferRadius:0)>.99,l=a||e===Yo.Instant?1:0;this.material.uniforms.visibleRegionFadeStartRadius.value=this.visibleRegionFadeStartRadius,this.material.uniforms.visibleRegionRadius.value=this.visibleRegionRadius,this.material.uniforms.firstRenderTime.value=this.firstRenderTime,this.material.uniforms.currentTime.value=performance.now(),this.material.uniforms.fadeInComplete.value=l,this.material.uniformsNeedUpdate=!0,this.visibleRegionChanging=!a}updateRenderIndexes(e,t){const n=this.geometry;n.attributes.splatIndex.set(e),n.attributes.splatIndex.needsUpdate=!0,t>0&&this.firstRenderTime===-1&&(this.firstRenderTime=performance.now()),n.instanceCount=t,n.setDrawRange(0,t)}updateTransforms(){for(let e=0;e<this.scenes.length;e++)this.getScene(e).updateTransform(this.dynamicMode)}updateUniforms=(function(){const e=new Ne;return function(t,n,s,r,o,a){if(this.getSplatCount()>0){if(e.set(t.x*this.devicePixelRatio,t.y*this.devicePixelRatio),this.material.uniforms.viewport.value.copy(e),this.material.uniforms.basisViewport.value.set(1/e.x,1/e.y),this.material.uniforms.focal.value.set(n,s),this.material.uniforms.orthographicMode.value=r?1:0,this.material.uniforms.orthoZoom.value=o,this.material.uniforms.inverseFocalAdjustment.value=a,this.dynamicMode)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.transforms.value[c].copy(this.getScene(c).transform);if(this.enableOptionalEffects)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.sceneOpacity.value[c]=Ct(this.getScene(c).opacity,0,1),this.material.uniforms.sceneVisibility.value[c]=this.getScene(c).visible?1:0,this.material.uniformsNeedUpdate=!0;this.material.uniformsNeedUpdate=!0}}})();setSplatScale(e=1){this.splatScale=e,this.material.uniforms.splatScale.value=e,this.material.uniformsNeedUpdate=!0}getSplatScale(){return this.splatScale}setPointCloudModeEnabled(e){this.pointCloudModeEnabled=e,this.material.uniforms.pointCloudModeEnabled.value=e?1:0,this.material.uniformsNeedUpdate=!0}getPointCloudModeEnabled(){return this.pointCloudModeEnabled}getSplatDataTextures(){return this.splatDataTextures}getSplatCount(e=!1){return e?$t.getTotalSplatCountForScenes(this.scenes):this.lastBuildSplatCount}static getTotalSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getSplatCount());return t}static getTotalSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getSplatCount();return t}getMaxSplatCount(){return $t.getTotalMaxSplatCountForScenes(this.scenes)}static getTotalMaxSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getMaxSplatCount());return t}static getTotalMaxSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getMaxSplatCount();return t}disposeDistancesComputationGPUResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.vao&&(e.deleteVertexArray(this.distancesTransformFeedback.vao),this.distancesTransformFeedback.vao=null),this.distancesTransformFeedback.program&&(e.deleteProgram(this.distancesTransformFeedback.program),e.deleteShader(this.distancesTransformFeedback.vertexShader),e.deleteShader(this.distancesTransformFeedback.fragmentShader),this.distancesTransformFeedback.program=null,this.distancesTransformFeedback.vertexShader=null,this.distancesTransformFeedback.fragmentShader=null),this.disposeDistancesComputationGPUBufferResources(),this.distancesTransformFeedback.id&&(e.deleteTransformFeedback(this.distancesTransformFeedback.id),this.distancesTransformFeedback.id=null)}disposeDistancesComputationGPUBufferResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.centersBuffer&&(this.distancesTransformFeedback.centersBuffer=null,e.deleteBuffer(this.distancesTransformFeedback.centersBuffer)),this.distancesTransformFeedback.outDistancesBuffer&&(e.deleteBuffer(this.distancesTransformFeedback.outDistancesBuffer),this.distancesTransformFeedback.outDistancesBuffer=null)}setRenderer(e){if(e!==this.renderer){this.renderer=e;const t=this.renderer.getContext(),n=new gE(t),s=new xE(t,n,{});if(n.init(s),this.webGLUtils=new Z0(t,n),this.enableDistancesComputationOnGPU&&this.getSplatCount()>0){this.setupDistancesComputationTransformFeedback();const{centers:r,sceneIndexes:o}=this.getDataForDistancesComputation(0,this.getSplatCount()-1);this.refreshGPUBuffersForDistancesComputation(r,o)}}}setupDistancesComputationTransformFeedback=(function(){let e;return function(){const t=this.getMaxSplatCount();if(!this.renderer)return;const n=this.lastRenderer!==this.renderer,s=e!==t;if(!n&&!s)return;n?this.disposeDistancesComputationGPUResources():s&&this.disposeDistancesComputationGPUBufferResources();const r=this.renderer.getContext(),o=(d,h,g)=>{const p=d.createShader(h);if(!p)return console.error("Fatal error: gl could not create a shader object."),null;if(d.shaderSource(p,g),d.compileShader(p),!d.getShaderParameter(p,d.COMPILE_STATUS)){let m="unknown";h===d.VERTEX_SHADER?m="vertex shader":h===d.FRAGMENT_SHADER&&(m="fragement shader");const _=d.getShaderInfoLog(p);return console.error("Failed to compile "+m+" with these errors:"+_),d.deleteShader(p),null}return p};let a;this.integerBasedDistancesComputation?(a=`#version 300 es
                in ivec4 center;
                flat out int distance;`,this.dynamicMode?a+=`
                        in uint sceneIndex;
                        uniform ivec4 transforms[${pt.MaxScenes}];
                        void main(void) {
                            ivec4 transform = transforms[sceneIndex];
                            distance = center.x * transform.x + center.y * transform.y + center.z * transform.z + transform.w * center.w;
                        }
                    `:a+=`
                        uniform ivec3 modelViewProj;
                        void main(void) {
                            distance = center.x * modelViewProj.x + center.y * modelViewProj.y + center.z * modelViewProj.z;
                        }
                    `):(a=`#version 300 es
                in vec4 center;
                flat out float distance;`,this.dynamicMode?a+=`
                        in uint sceneIndex;
                        uniform mat4 transforms[${pt.MaxScenes}];
                        void main(void) {
                            vec4 transformedCenter = transforms[sceneIndex] * vec4(center.xyz, 1.0);
                            distance = transformedCenter.z;
                        }
                    `:a+=`
                        uniform vec3 modelViewProj;
                        void main(void) {
                            distance = center.x * modelViewProj.x + center.y * modelViewProj.y + center.z * modelViewProj.z;
                        }
                    `);const l=`#version 300 es
                precision lowp float;
                out vec4 fragColor;
                void main(){}
            `,c=r.getParameter(r.VERTEX_ARRAY_BINDING),u=r.getParameter(r.CURRENT_PROGRAM),f=u?r.getProgramParameter(u,r.DELETE_STATUS):!1;if(n&&(this.distancesTransformFeedback.vao=r.createVertexArray()),r.bindVertexArray(this.distancesTransformFeedback.vao),n){const d=r.createProgram(),h=o(r,r.VERTEX_SHADER,a),g=o(r,r.FRAGMENT_SHADER,l);if(!h||!g)throw new Error("Could not compile shaders for distances computation on GPU.");if(r.attachShader(d,h),r.attachShader(d,g),r.transformFeedbackVaryings(d,["distance"],r.SEPARATE_ATTRIBS),r.linkProgram(d),!r.getProgramParameter(d,r.LINK_STATUS)){const x=r.getProgramInfoLog(d);throw console.error("Fatal error: Failed to link program: "+x),r.deleteProgram(d),r.deleteShader(g),r.deleteShader(h),new Error("Could not link shaders for distances computation on GPU.")}this.distancesTransformFeedback.program=d,this.distancesTransformFeedback.vertexShader=h,this.distancesTransformFeedback.vertexShader=g}if(r.useProgram(this.distancesTransformFeedback.program),this.distancesTransformFeedback.centersLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"center"),this.dynamicMode){this.distancesTransformFeedback.sceneIndexesLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"sceneIndex");for(let d=0;d<this.scenes.length;d++)this.distancesTransformFeedback.transformsLocs[d]=r.getUniformLocation(this.distancesTransformFeedback.program,`transforms[${d}]`)}else this.distancesTransformFeedback.modelViewProjLoc=r.getUniformLocation(this.distancesTransformFeedback.program,"modelViewProj");(n||s)&&(this.distancesTransformFeedback.centersBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?r.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,r.INT,0,0):r.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,r.FLOAT,!1,0,0),this.dynamicMode&&(this.distancesTransformFeedback.sceneIndexesBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),r.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,r.UNSIGNED_INT,0,0))),(n||s)&&(this.distancesTransformFeedback.outDistancesBuffer=r.createBuffer()),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),r.bufferData(r.ARRAY_BUFFER,t*4,r.STATIC_READ),n&&(this.distancesTransformFeedback.id=r.createTransformFeedback()),r.bindTransformFeedback(r.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),r.bindBufferBase(r.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),u&&f!==!0&&r.useProgram(u),c&&r.bindVertexArray(c),this.lastRenderer=this.renderer,e=t}})();updateGPUCentersBufferForDistancesComputation(e,t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=this.integerBasedDistancesComputation?Uint32Array:Float32Array,a=16,l=n*a;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,l,t);else{const c=new o(this.getMaxSplatCount()*a);c.set(t),s.bufferData(s.ARRAY_BUFFER,c,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}updateGPUTransformIndexesBufferForDistancesComputation(e,t,n){if(!this.renderer||!this.dynamicMode)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=n*4;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,o,t);else{const a=new Uint32Array(this.getMaxSplatCount()*4);a.set(t),s.bufferData(s.ARRAY_BUFFER,a,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}getSceneIndexes(e,t){let n;const s=t-e+1;n=new Uint32Array(s);for(let r=e;r<=t;r++)n[r]=this.globalSplatIndexToSceneIndexMap[r];return n}fillTransformsArray=(function(){const e=[];return function(t){e.length!==t.length&&(e.length=t.length);for(let n=0;n<this.scenes.length;n++){const r=this.getScene(n).transform.elements;for(let o=0;o<16;o++)e[n*16+o]=r[o]}t.set(e)}})();computeDistancesOnGPU=(function(){const e=new qe;return function(t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING),o=s.getParameter(s.CURRENT_PROGRAM),a=o?s.getProgramParameter(o,s.DELETE_STATUS):!1;if(s.bindVertexArray(this.distancesTransformFeedback.vao),s.useProgram(this.distancesTransformFeedback.program),s.enable(s.RASTERIZER_DISCARD),this.dynamicMode)for(let u=0;u<this.scenes.length;u++)if(e.copy(this.getScene(u).transform),e.premultiply(t),this.integerBasedDistancesComputation){const f=$t.getIntegerMatrixArray(e),d=[f[2],f[6],f[10],f[14]];s.uniform4i(this.distancesTransformFeedback.transformsLocs[u],d[0],d[1],d[2],d[3])}else s.uniformMatrix4fv(this.distancesTransformFeedback.transformsLocs[u],!1,e.elements);else if(this.integerBasedDistancesComputation){const u=$t.getIntegerMatrixArray(t),f=[u[2],u[6],u[10]];s.uniform3i(this.distancesTransformFeedback.modelViewProjLoc,f[0],f[1],f[2])}else{const u=[t.elements[2],t.elements[6],t.elements[10]];s.uniform3f(this.distancesTransformFeedback.modelViewProjLoc,u[0],u[1],u[2])}s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?s.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,s.INT,0,0):s.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,s.FLOAT,!1,0,0),this.dynamicMode&&(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),s.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,s.UNSIGNED_INT,0,0)),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),s.beginTransformFeedback(s.POINTS),s.drawArrays(s.POINTS,0,this.getSplatCount()),s.endTransformFeedback(),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,null),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,null),s.disable(s.RASTERIZER_DISCARD);const l=s.fenceSync(s.SYNC_GPU_COMMANDS_COMPLETE,0);s.flush();const c=new Promise(u=>{const f=()=>{if(this.disposed)u();else switch(s.clientWaitSync(l,0,0)){case s.TIMEOUT_EXPIRED:return this.computeDistancesOnGPUSyncTimeout=setTimeout(f),this.computeDistancesOnGPUSyncTimeout;case s.WAIT_FAILED:throw new Error("should never get here");default:this.computeDistancesOnGPUSyncTimeout=null,s.deleteSync(l);const p=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao),s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),s.getBufferSubData(s.ARRAY_BUFFER,0,n),s.bindBuffer(s.ARRAY_BUFFER,null),p&&s.bindVertexArray(p),u()}};this.computeDistancesOnGPUSyncTimeout=setTimeout(f)});return o&&a!==!0&&s.useProgram(o),r&&s.bindVertexArray(r),c}})();getLocalSplatParameters(e,t,n){n==null&&(n=!this.dynamicMode),t.splatBuffer=this.getSplatBufferForSplat(e),t.localIndex=this.getSplatLocalIndex(e),t.sceneTransform=n?this.getSceneTransformForSplat(e):null}fillSplatDataArrays(e,t,n,s,r,o,a,l=0,c=0,u=1,f,d,h=0,g){const p=new L;p.x=void 0,p.y=void 0,this.splatRenderMode===$i.ThreeD?p.z=void 0:p.z=1;const x=new qe;let m=0,_=this.scenes.length-1;g!=null&&g>=0&&g<=this.scenes.length&&(m=g,_=g);for(let A=m;A<=_;A++){a==null&&(a=!this.dynamicMode);const S=this.getScene(A),v=S.splatBuffer;let b;if(a&&(this.getSceneTransform(A,x),b=x),e&&v.fillSplatCovarianceArray(e,b,f,d,h,l),t||n){if(!t||!n)throw new Error('SplatMesh::fillSplatDataArrays() -> "scales" and "rotations" must both be valid.');v.fillSplatScaleRotationArray(t,n,b,f,d,h,c,p)}s&&v.fillSplatCenterArray(s,b,f,d,h),r&&v.fillSplatColorArray(r,S.minimumAlpha,f,d,h),o&&v.fillSphericalHarmonicsArray(o,this.minSphericalHarmonicsDegree,b,f,d,h,u),h+=v.getSplatCount()}}getIntegerCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e);let o,a=n?4:3;o=new Int32Array(s*a);for(let l=0;l<s;l++){for(let c=0;c<3;c++)o[l*a+c]=Math.round(r[l*3+c]*1e3);n&&(o[l*a+3]=1e3)}return o}getFloatCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);if(this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e),!n)return r;let o=new Float32Array(s*4);for(let a=0;a<s;a++){for(let l=0;l<3;l++)o[a*4+l]=r[a*3+l];o[a*4+3]=1}return o}getSplatCenter=(function(){const e={};return function(t,n,s){this.getLocalSplatParameters(t,e,s),e.splatBuffer.getSplatCenter(e.localIndex,n,e.sceneTransform)}})();getSplatScaleAndRotation=(function(){const e={},t=new L;return function(n,s,r,o){this.getLocalSplatParameters(n,e,o),t.x=void 0,t.y=void 0,t.z=void 0,this.splatRenderMode===$i.TwoD&&(t.z=0),e.splatBuffer.getSplatScaleAndRotation(e.localIndex,s,r,e.sceneTransform,t)}})();getSplatColor=(function(){const e={};return function(t,n){this.getLocalSplatParameters(t,e),e.splatBuffer.getSplatColor(e.localIndex,n)}})();getSceneTransform(e,t){const n=this.getScene(e);n.updateTransform(this.dynamicMode),t.copy(n.transform)}getScene(e){if(e<0||e>=this.scenes.length)throw new Error("SplatMesh::getScene() -> Invalid scene index.");return this.scenes[e]}getSceneCount(){return this.scenes.length}getSplatBufferForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).splatBuffer}getSceneIndexForSplat(e){return this.globalSplatIndexToSceneIndexMap[e]}getSceneTransformForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).transform}getSplatLocalIndex(e){return this.globalSplatIndexToLocalSplatIndexMap[e]}static getIntegerMatrixArray(e){const t=e.elements,n=[];for(let s=0;s<16;s++)n[s]=Math.round(t[s]*1e3);return n}computeBoundingBox(e=!1,t){let n=this.getSplatCount();if(t!=null){if(t<0||t>=this.scenes.length)throw new Error("SplatMesh::computeBoundingBox() -> Invalid scene index.");n=this.scenes[t].splatBuffer.getSplatCount()}const s=new Float32Array(n*3);this.fillSplatDataArrays(null,null,null,s,null,null,e,void 0,void 0,void 0,void 0,t);const r=new L,o=new L;for(let a=0;a<n;a++){const l=a*3,c=s[l],u=s[l+1],f=s[l+2];(a===0||c<r.x)&&(r.x=c),(a===0||u<r.y)&&(r.y=u),(a===0||f<r.z)&&(r.z=f),(a===0||c>o.x)&&(o.x=c),(a===0||u>o.y)&&(o.y=u),(a===0||f>o.z)&&(o.z=f)}return new wi(r,o)}}var TE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEbA2AAAGAQf39/f39/f39/f39/f39/fwBgAAF/AhIBA2VudgZtZW1vcnkCAwCAgAQDBAMAAQIHVAQRX193YXNtX2NhbGxfY3RvcnMAABhfX3dhc21fYXBwbHlfZGF0YV9yZWxvY3MAAAtzb3J0SW5kZXhlcwABE2Vtc2NyaXB0ZW5fdGxzX2luaXQAAgqWEAMDAAELihAEAXwDewN/A30gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEBA0AgAyABQQJ0IgVqIAIgACAFaigCAEECdGooAgAiBTYCACAFIAogBSAKSBshCiAFIA0gBSANShshDSABQQFqIgEgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiFWooAgAiFkECdGooAgAiFEcEQAJ/IAX9CQI4IAggFEEGdGoiDv0JAgwgDioCHP0gASAOKgIs/SACIA4qAjz9IAP95gEgBf0JAiggDv0JAgggDioCGP0gASAOKgIo/SACIA4qAjj9IAP95gEgBf0JAgggDv0JAgAgDioCEP0gASAOKgIg/SACIA4qAjD9IAP95gEgBf0JAhggDv0JAgQgDioCFP0gASAOKgIk/SACIA4qAjT9IAP95gH95AH95AH95AEiEf1f/QwAAAAAAECPQAAAAAAAQI9AIhL98gEiE/0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBP9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/REgDv0cAQJ/IBEgEf0NCAkKCwwNDg8AAAAAAAAAAP1fIBL98gEiEf0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9HAICfyAR/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAyESIBQhDwsgAyAVaiABIBZBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmogEf0bA2oiDjYCACAOIAogCiAOShshCiAOIA0gDSAOSBshDSACQQFqIgIgC0cNAAsMAwsCfyAFKgIIu/0UIAUqAhi7/SIB/QwAAAAAAECPQAAAAAAAQI9A/fIBIhH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIQ4CfyAR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyECAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQIgAv0RIA79HAEgBf0cAiESIAwhBQNAIAMgBUECdCICaiABIAAgAmooAgBBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmoiAjYCACACIAogAiAKSBshCiACIA0gAiANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEBA0AgAyABQQJ0IgVqAn8gAiAAIAVqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAFBAWoiASALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIRcgBSoCGCEYIAUqAgghGUH4////ByEKQYiAgIB4IQ0gDCEFA0ACfyAXIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCAZIAIqAgCUIBggAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIUaigCAEECdCIVaigCACIORwRAIAX9CQI4IAggDkEGdGoiD/0JAgwgDyoCHP0gASAPKgIs/SACIA8qAjz9IAP95gEgBf0JAiggD/0JAgggDyoCGP0gASAPKgIo/SACIA8qAjj9IAP95gEgBf0JAgggD/0JAgAgDyoCEP0gASAPKgIg/SACIA8qAjD9IAP95gEgBf0JAhggD/0JAgQgDyoCFP0gASAPKgIk/SACIA8qAjT9IAP95gH95AH95AH95AEhESAOIQ8LIAMgFGoCfyAR/R8DIAEgFUECdCIOQQxyaioCAJQgEf0fAiABIA5BCHJqKgIAlCAR/R8AIAEgDmoqAgCUIBH9HwEgASAOQQRyaioCAJSSkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSACQQFqIgIgC0cNAAsMAQtBiICAgHghDUH4////ByEKCyALIAxLBEAgCUEBa7MgDbIgCrKTlSEXIAwhDQNAAn8gFyADIA1BAnRqIgEoAgAgCmuylCIYi0MAAABPXQRAIBioDAELQYCAgIB4CyEOIAEgDjYCACAEIA5BAnRqIgEgASgCAEEBajYCACANQQFqIg0gC0cNAAsLIAlBAk8EQCAEKAIAIQ1BASEKA0AgBCAKQQJ0aiIBIAEoAgAgDWoiDTYCACAKQQFqIgogCUcNAAsLIAxBAEoEQCAMIQoDQCAGIApBAWsiAUECdCICaiAAIAJqKAIANgIAIApBAUshAiABIQogAg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCwsEAEEACw==",jp="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACEgEDZW52Bm1lbW9yeQIDAICABAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=",EE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQrrDwICAAvlDwQBfAN7B30DfyALIAprIQwCQAJAIA4EQCANBEBB+P///wchCkGIgICAeCENIAsgDE0NAyAMIQUDQCADIAVBAnQiAWogAiAAIAFqKAIAQQJ0aigCACIBNgIAIAEgCiABIApIGyEKIAEgDSABIA1KGyENIAVBAWoiBSALRw0ACwwDCyAPBEAgCyAMTQ0CQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIcaigCACIdQQJ0aigCACIbRwRAAn8gBf0JAjggCCAbQQZ0aiIO/QkCDCAOKgIc/SABIA4qAiz9IAIgDioCPP0gA/3mASAF/QkCKCAO/QkCCCAOKgIY/SABIA4qAij9IAIgDioCOP0gA/3mASAF/QkCCCAO/QkCACAOKgIQ/SABIA4qAiD9IAIgDioCMP0gA/3mASAF/QkCGCAO/QkCBCAOKgIU/SABIA4qAiT9IAIgDioCNP0gA/3mAf3kAf3kAf3kASIR/V/9DAAAAAAAQI9AAAAAAABAj0AiEv3yASIT/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOAn8gE/0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9ESAO/RwBAn8gESAR/Q0ICQoLDA0ODwABAgMAAQID/V8gEv3yASIR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAgJ/IBH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/RwDIRIgGyEPCyADIBxqIAEgHUEEdGr9AAAAIBL9tQEiEf0bACAR/RsBaiAR/RsCaiAR/RsDaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAgi7/RQgBSoCGLv9IgH9DAAAAAAAQI9AAAAAAABAj0D98gEiEf0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBH9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQL9ESAO/RwBIAX9HAIhEiAMIQUDQCADIAVBAnQiAmogASAAIAJqKAIAQQR0av0AAAAgEv21ASIR/RsAIBH9GwFqIBH9GwJqIgI2AgAgAiAKIAIgCkgbIQogAiANIAIgDUobIQ0gBUEBaiIFIAtHDQALDAILIA0EQEH4////ByEKQYiAgIB4IQ0gCyAMTQ0CIAwhBQNAIAMgBUECdCIBagJ/IAIgACABaigCAEECdGoqAgC7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgD0UEQCALIAxNDQEgBSoCKCEUIAUqAhghFSAFKgIIIRZB+P///wchCkGIgICAeCENIAwhBQNAAn8gFCABIAAgBUECdCIHaigCAEEEdGoiAioCCJQgFiACKgIAlCAVIAIqAgSUkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDiADIAdqIA42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gBUEBaiIFIAtHDQALDAILIAsgDE0NAEF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiG2ooAgBBAnQiHGooAgAiDkcEQCAFKgI4IhQgCCAOQQZ0aiIPKgI8lCAFKgIoIhUgDyoCOJQgBSoCCCIWIA8qAjCUIAUqAhgiFyAPKgI0lJKSkiEYIBQgDyoCLJQgFSAPKgIolCAWIA8qAiCUIBcgDyoCJJSSkpIhGSAUIA8qAhyUIBUgDyoCGJQgFiAPKgIQlCAXIA8qAhSUkpKSIRogFCAPKgIMlCAVIA8qAgiUIBYgDyoCAJQgFyAPKgIElJKSkiEUIA4hDwsgAyAbagJ/IBggASAcQQJ0aiIOKgIMlCAZIA4qAgiUIBQgDioCAJQgGiAOKgIElJKSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAJBAWoiAiALRw0ACwwBC0GIgICAeCENQfj///8HIQoLIAsgDEsEQCAJQQFrsyANsiAKspOVIRQgDCENA0ACfyAUIAMgDUECdGoiASgCACAKa7KUIhWLQwAAAE9dBEAgFagMAQtBgICAgHgLIQ4gASAONgIAIAQgDkECdGoiASABKAIAQQFqNgIAIA1BAWoiDSALRw0ACwsgCUECTwRAIAQoAgAhDUEBIQoDQCAEIApBAnRqIgEgASgCACANaiINNgIAIApBAWoiCiAJRw0ACwsgDEEASgRAIAwhCgNAIAYgCkEBayIBQQJ0IgJqIAAgAmooAgA2AgAgCkEBSyABIQoNAAsLIAsgDEoEQCALIQoDQCAGIAsgBCADIApBAWsiCkECdCIBaigCAEECdGoiAigCACIFa0ECdGogACABaigCADYCACACIAVBAWs2AgAgCiAMSg0ACwsL",wE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=";function RE(i){let e,t,n,s,r,o,a,l,c,u,f,d,h,g,p,x,m,_,A,S;function v(b,M,E,y,C,I,P){const U=performance.now();if(!n&&(new Uint32Array(t,a,C.byteLength/S.BytesPerInt).set(C),new Float32Array(t,u,P.byteLength/S.BytesPerFloat).set(P),y)){let H;s?H=new Int32Array(t,f,I.byteLength/S.BytesPerInt):H=new Float32Array(t,f,I.byteLength/S.BytesPerFloat),H.set(I)}x||(x=new Uint32Array(_)),new Float32Array(t,p,16).set(E),new Uint32Array(t,h,_).set(x),e.exports.sortIndexes(a,g,f,d,h,p,l,c,u,_,b,M,o,y,s,r);const O={sortDone:!0,splatSortCount:b,splatRenderCount:M,sortTime:0};if(!n){const z=new Uint32Array(t,l,M);(!m||m.length<M)&&(m=new Uint32Array(M)),m.set(z),O.sortedIndexes=m}const k=performance.now();O.sortTime=k-U,i.postMessage(O)}i.onmessage=b=>{if(b.data.centers)centers=b.data.centers,sceneIndexes=b.data.sceneIndexes,s?new Int32Array(t,g+b.data.range.from*S.BytesPerInt*4,b.data.range.count*4).set(new Int32Array(centers)):new Float32Array(t,g+b.data.range.from*S.BytesPerFloat*4,b.data.range.count*4).set(new Float32Array(centers)),r&&new Uint32Array(t,c+b.data.range.from*4,b.data.range.count).set(new Uint32Array(sceneIndexes)),A=b.data.range.from+b.data.range.count;else if(b.data.sort){const M=Math.min(b.data.sort.splatRenderCount||0,A),E=Math.min(b.data.sort.splatSortCount||0,A),y=b.data.sort.usePrecomputedDistances;let C,I,P;n||(C=b.data.sort.indexesToSort,P=b.data.sort.transforms,y&&(I=b.data.sort.precomputedDistances)),v(E,M,b.data.sort.modelViewProj,y,C,I,P)}else if(b.data.init){S=b.data.init.Constants,o=b.data.init.splatCount,n=b.data.init.useSharedMemory,s=b.data.init.integerBasedSort,r=b.data.init.dynamicMode,_=b.data.init.distanceMapRange,A=0;const M=s?S.BytesPerInt*4:S.BytesPerFloat*4,E=new Uint8Array(b.data.init.sorterWasmBytes),y=16*S.BytesPerFloat,C=o*S.BytesPerInt,I=o*M,P=y,U=s?o*S.BytesPerInt:o*S.BytesPerFloat,O=o*S.BytesPerInt,k=o*S.BytesPerInt,z=s?_*S.BytesPerInt*2:_*S.BytesPerFloat*2,W=r?o*S.BytesPerInt:0,H=r?S.MaxScenes*y:0,$=S.MemoryPageSize*32,X=C+I+P+U+O+z+k+W+H+$,Z=Math.floor(X/S.MemoryPageSize)+1,pe={module:{},env:{memory:new WebAssembly.Memory({initial:Z,maximum:Z,shared:!0})}};WebAssembly.compile(E).then(ge=>WebAssembly.instantiate(ge,pe)).then(ge=>{e=ge,a=0,g=a+C,p=g+I,f=p+P,d=f+U,h=d+O,l=h+z,c=l+k,u=c+W,t=pe.env.memory.buffer,n?i.postMessage({sortSetupPhase1Complete:!0,indexesToSortBuffer:t,indexesToSortOffset:a,sortedIndexesBuffer:t,sortedIndexesOffset:l,precomputedDistancesBuffer:t,precomputedDistancesOffset:f,transformsBuffer:t,transformsOffset:u}):i.postMessage({sortSetupPhase1Complete:!0})})}}}function IE(i,e,t,n,s,r=pt.DefaultSplatSortDistanceMapPrecision){const o=new Worker(URL.createObjectURL(new Blob(["(",RE.toString(),")(self)"],{type:"application/javascript"})));let a=TE;const l=rd()?eg():null;!t&&!e?(a=jp,l&&l.major<=16&&l.minor<4&&(a=wE)):t?e||l&&l.major<=16&&l.minor<4&&(a=EE):a=jp;const c=atob(a),u=new Uint8Array(c.length);for(let f=0;f<c.length;f++)u[f]=c.charCodeAt(f);return o.postMessage({init:{sorterWasmBytes:u.buffer,splatCount:i,useSharedMemory:e,integerBasedSort:n,dynamicMode:s,distanceMapRange:1<<r,Constants:{BytesPerFloat:pt.BytesPerFloat,BytesPerInt:pt.BytesPerInt,MemoryPageSize:pt.MemoryPageSize,MaxScenes:pt.MaxScenes}}}),o}const ys={None:0,VR:1,AR:2};class po{static createButton(e,t={}){const n=document.createElement("button");function s(){let c=null;async function u(h){h.addEventListener("end",f),await e.xr.setSession(h),n.textContent="EXIT VR",c=h}function f(){c.removeEventListener("end",f),n.textContent="ENTER VR",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="ENTER VR";const d={...t,optionalFeatures:["local-floor","bounded-floor","layers",...t.optionalFeatures||[]]};n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-vr",d).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",d).then(u).catch(h=>{console.warn(h)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",d).then(u).catch(h=>{console.warn(h)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="VR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="VR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="VRButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-vr").then(function(c){c?s():o(),c&&po.xrSessionIsGranted&&n.click()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}static registerSessionGrantedListener(){if(typeof navigator<"u"&&"xr"in navigator){if(/WebXRViewer\//i.test(navigator.userAgent))return;navigator.xr.addEventListener("sessiongranted",()=>{po.xrSessionIsGranted=!0})}}}po.xrSessionIsGranted=!1;po.registerSessionGrantedListener();class DE{static createButton(e,t={}){const n=document.createElement("button");function s(){if(t.domOverlay===void 0){const d=document.createElement("div");d.style.display="none",document.body.appendChild(d);const h=document.createElementNS("http://www.w3.org/2000/svg","svg");h.setAttribute("width",38),h.setAttribute("height",38),h.style.position="absolute",h.style.right="20px",h.style.top="20px",h.addEventListener("click",function(){c.end()}),d.appendChild(h);const g=document.createElementNS("http://www.w3.org/2000/svg","path");g.setAttribute("d","M 12,12 L 28,28 M 28,12 12,28"),g.setAttribute("stroke","#fff"),g.setAttribute("stroke-width",2),h.appendChild(g),t.optionalFeatures===void 0&&(t.optionalFeatures=[]),t.optionalFeatures.push("dom-overlay"),t.domOverlay={root:d}}let c=null;async function u(d){d.addEventListener("end",f),e.xr.setReferenceSpaceType("local"),await e.xr.setSession(d),n.textContent="STOP AR",t.domOverlay.root.style.display="",c=d}function f(){c.removeEventListener("end",f),n.textContent="START AR",t.domOverlay.root.style.display="none",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="START AR",n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-ar",t).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(d=>{console.warn(d)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(d=>{console.warn(d)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="AR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="AR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="ARButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-ar").then(function(c){c?s():o()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}}const nu={Always:0,Never:2},PE=50,FE=.75,LE=15e5,BE=10,UE=2.5,OE=60;class Vr{constructor(e={}){if(e.cameraUp||(e.cameraUp=[0,1,0]),this.cameraUp=new L().fromArray(e.cameraUp),e.initialCameraPosition||(e.initialCameraPosition=[0,10,15]),this.initialCameraPosition=new L().fromArray(e.initialCameraPosition),e.initialCameraLookAt||(e.initialCameraLookAt=[0,0,0]),this.initialCameraLookAt=new L().fromArray(e.initialCameraLookAt),this.dropInMode=e.dropInMode||!1,(e.selfDrivenMode===void 0||e.selfDrivenMode===null)&&(e.selfDrivenMode=!0),this.selfDrivenMode=e.selfDrivenMode&&!this.dropInMode,this.selfDrivenUpdateFunc=this.selfDrivenUpdate.bind(this),e.useBuiltInControls===void 0&&(e.useBuiltInControls=!0),this.useBuiltInControls=e.useBuiltInControls,this.rootElement=e.rootElement,this.ignoreDevicePixelRatio=e.ignoreDevicePixelRatio||!1,this.devicePixelRatio=this.ignoreDevicePixelRatio?1:window.devicePixelRatio||1,this.halfPrecisionCovariancesOnGPU=e.halfPrecisionCovariancesOnGPU||!1,this.threeScene=e.threeScene,this.renderer=e.renderer,this.camera=e.camera,this.gpuAcceleratedSort=e.gpuAcceleratedSort||!1,(e.integerBasedSort===void 0||e.integerBasedSort===null)&&(e.integerBasedSort=!0),this.integerBasedSort=e.integerBasedSort,(e.sharedMemoryForWorkers===void 0||e.sharedMemoryForWorkers===null)&&(e.sharedMemoryForWorkers=!0),this.sharedMemoryForWorkers=e.sharedMemoryForWorkers,this.dynamicScene=!!e.dynamicScene,this.antialiased=e.antialiased||!1,this.kernel2DSize=e.kernel2DSize===void 0?.3:e.kernel2DSize,this.webXRMode=e.webXRMode||ys.None,this.webXRMode!==ys.None&&(this.gpuAcceleratedSort=!1),this.webXRActive=!1,this.webXRSessionInit=e.webXRSessionInit||{},this.renderMode=e.renderMode||nu.Always,this.sceneRevealMode=e.sceneRevealMode||Yo.Default,this.focalAdjustment=e.focalAdjustment||1,this.maxScreenSpaceSplatSize=e.maxScreenSpaceSplatSize||1024,this.logLevel=e.logLevel||eo.None,this.sphericalHarmonicsDegree=e.sphericalHarmonicsDegree||0,this.enableOptionalEffects=e.enableOptionalEffects||!1,(e.enableSIMDInSort===void 0||e.enableSIMDInSort===null)&&(e.enableSIMDInSort=!0),this.enableSIMDInSort=e.enableSIMDInSort,(e.inMemoryCompressionLevel===void 0||e.inMemoryCompressionLevel===null)&&(e.inMemoryCompressionLevel=0),this.inMemoryCompressionLevel=e.inMemoryCompressionLevel,(e.optimizeSplatData===void 0||e.optimizeSplatData===null)&&(e.optimizeSplatData=!0),this.optimizeSplatData=e.optimizeSplatData,(e.freeIntermediateSplatData===void 0||e.freeIntermediateSplatData===null)&&(e.freeIntermediateSplatData=!1),this.freeIntermediateSplatData=e.freeIntermediateSplatData,rd()){const n=eg();n.major<17&&(this.enableSIMDInSort=!1),n.major<16&&(this.sharedMemoryForWorkers=!1)}(e.splatRenderMode===void 0||e.splatRenderMode===null)&&(e.splatRenderMode=$i.ThreeD),this.splatRenderMode=e.splatRenderMode,this.sceneFadeInRateMultiplier=e.sceneFadeInRateMultiplier||1,this.splatSortDistanceMapPrecision=e.splatSortDistanceMapPrecision||pt.DefaultSplatSortDistanceMapPrecision;const t=this.integerBasedSort?20:24;this.splatSortDistanceMapPrecision=Ct(this.splatSortDistanceMapPrecision,10,t),this.onSplatMeshChangedCallback=null,this.createSplatMesh(),this.controls=null,this.perspectiveControls=null,this.orthographicControls=null,this.orthographicCamera=null,this.perspectiveCamera=null,this.showMeshCursor=!1,this.showControlPlane=!1,this.showInfo=!1,this.sceneHelper=null,this.sortWorker=null,this.sortRunning=!1,this.splatRenderCount=0,this.splatSortCount=0,this.lastSplatSortCount=0,this.sortWorkerIndexesToSort=null,this.sortWorkerSortedIndexes=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.preSortMessages=[],this.runAfterNextSort=[],this.selfDrivenModeRunning=!1,this.splatRenderReady=!1,this.raycaster=new cE,this.infoPanel=null,this.startInOrthographicMode=!1,this.currentFPS=0,this.lastSortTime=0,this.consecutiveRenderFrames=0,this.previousCameraTarget=new L,this.nextCameraTarget=new L,this.mousePosition=new Ne,this.mouseDownPosition=new Ne,this.mouseDownTime=null,this.resizeObserver=null,this.mouseMoveListener=null,this.mouseDownListener=null,this.mouseUpListener=null,this.keyDownListener=null,this.sortPromise=null,this.sortPromiseResolver=null,this.splatSceneDownloadPromises={},this.splatSceneDownloadAndBuildPromise=null,this.splatSceneRemovalPromise=null,this.loadingSpinner=new _d(null,this.rootElement||document.body),this.loadingSpinner.hide(),this.loadingProgressBar=new iE(this.rootElement||document.body),this.loadingProgressBar.hide(),this.infoPanel=new sE(this.rootElement||document.body),this.infoPanel.hide(),this.usingExternalCamera=!!(this.dropInMode||this.camera),this.usingExternalRenderer=!!(this.dropInMode||this.renderer),this.initialized=!1,this.disposing=!1,this.disposed=!1,this.disposePromise=null,this.dropInMode||this.init()}createSplatMesh(){this.splatMesh=new $t(this.splatRenderMode,this.dynamicScene,this.enableOptionalEffects,this.halfPrecisionCovariancesOnGPU,this.devicePixelRatio,this.gpuAcceleratedSort,this.integerBasedSort,this.antialiased,this.maxScreenSpaceSplatSize,this.logLevel,this.sphericalHarmonicsDegree,this.sceneFadeInRateMultiplier,this.kernel2DSize),this.splatMesh.frustumCulled=!1,this.onSplatMeshChangedCallback&&this.onSplatMeshChangedCallback()}init(){this.initialized||(this.rootElement||(this.usingExternalRenderer?this.rootElement=this.renderer.domElement||document.body:(this.rootElement=document.createElement("div"),this.rootElement.style.width="100%",this.rootElement.style.height="100%",this.rootElement.style.position="absolute",document.body.appendChild(this.rootElement))),this.setupCamera(),this.setupRenderer(),this.setupWebXR(this.webXRSessionInit),this.setupControls(),this.setupEventHandlers(),this.threeScene=this.threeScene||new rv,this.sceneHelper=new qo(this.threeScene),this.sceneHelper.setupMeshCursor(),this.sceneHelper.setupFocusMarker(),this.sceneHelper.setupControlPlane(),this.loadingProgressBar.setContainer(this.rootElement),this.loadingSpinner.setContainer(this.rootElement),this.infoPanel.setContainer(this.rootElement),this.initialized=!0)}setupCamera(){if(!this.usingExternalCamera){const e=new Ne;this.getRenderDimensions(e),this.perspectiveCamera=new ei(PE,e.x/e.y,.1,1e3),this.orthographicCamera=new id(e.x/-2,e.x/2,e.y/2,e.y/-2,.1,1e3),this.camera=this.startInOrthographicMode?this.orthographicCamera:this.perspectiveCamera,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt)}}setupRenderer(){if(!this.usingExternalRenderer){const e=new Ne;this.getRenderDimensions(e),this.renderer=new tT({antialias:!1,precision:"highp"}),this.renderer.setPixelRatio(this.devicePixelRatio),this.renderer.autoClear=!0,this.renderer.setClearColor(new nt(0),0),this.renderer.setSize(e.x,e.y),this.resizeObserver=new ResizeObserver(()=>{this.getRenderDimensions(e),this.renderer.setSize(e.x,e.y),this.forceRenderNextFrame()}),this.resizeObserver.observe(this.rootElement),this.rootElement.appendChild(this.renderer.domElement)}}setupWebXR(e){this.webXRMode&&(this.webXRMode===ys.VR?this.rootElement.appendChild(po.createButton(this.renderer,e)):this.webXRMode===ys.AR&&this.rootElement.appendChild(DE.createButton(this.renderer,e)),this.renderer.xr.addEventListener("sessionstart",t=>{this.webXRActive=!0}),this.renderer.xr.addEventListener("sessionend",t=>{this.webXRActive=!1}),this.renderer.xr.enabled=!0,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt))}setupControls(){if(this.useBuiltInControls&&this.webXRMode===ys.None){this.usingExternalCamera?this.camera.isOrthographicCamera?this.orthographicControls=new sl(this.camera,this.renderer.domElement):this.perspectiveControls=new sl(this.camera,this.renderer.domElement):(this.perspectiveControls=new sl(this.perspectiveCamera,this.renderer.domElement),this.orthographicControls=new sl(this.orthographicCamera,this.renderer.domElement));for(let e of[this.orthographicControls,this.perspectiveControls])e&&(e.listenToKeyEvents(window),e.rotateSpeed=.5,e.maxPolarAngle=Math.PI*.75,e.minPolarAngle=.1,e.enableDamping=!0,e.dampingFactor=.05,e.target.copy(this.initialCameraLookAt),e.update());this.controls=this.camera.isOrthographicCamera?this.orthographicControls:this.perspectiveControls,this.controls.update()}}setupEventHandlers(){this.useBuiltInControls&&this.webXRMode===ys.None&&(this.mouseMoveListener=this.onMouseMove.bind(this),this.renderer.domElement.addEventListener("pointermove",this.mouseMoveListener,!1),this.mouseDownListener=this.onMouseDown.bind(this),this.renderer.domElement.addEventListener("pointerdown",this.mouseDownListener,!1),this.mouseUpListener=this.onMouseUp.bind(this),this.renderer.domElement.addEventListener("pointerup",this.mouseUpListener,!1),this.keyDownListener=this.onKeyDown.bind(this),window.addEventListener("keydown",this.keyDownListener,!1))}removeEventHandlers(){this.useBuiltInControls&&(this.renderer.domElement.removeEventListener("pointermove",this.mouseMoveListener),this.mouseMoveListener=null,this.renderer.domElement.removeEventListener("pointerdown",this.mouseDownListener),this.mouseDownListener=null,this.renderer.domElement.removeEventListener("pointerup",this.mouseUpListener),this.mouseUpListener=null,window.removeEventListener("keydown",this.keyDownListener),this.keyDownListener=null)}setRenderMode(e){this.renderMode=e}setActiveSphericalHarmonicsDegrees(e){this.splatMesh.material.uniforms.sphericalHarmonicsDegree.value=e,this.splatMesh.material.uniformsNeedUpdate=!0}onSplatMeshChanged(e){this.onSplatMeshChangedCallback=e}onKeyDown=(function(){const e=new L,t=new qe,n=new qe;return function(s){switch(e.set(0,0,-1),e.transformDirection(this.camera.matrixWorld),t.makeRotationAxis(e,Math.PI/128),n.makeRotationAxis(e,-Math.PI/128),s.code){case"KeyG":this.focalAdjustment+=.02,this.forceRenderNextFrame();break;case"KeyF":this.focalAdjustment-=.02,this.forceRenderNextFrame();break;case"ArrowLeft":this.camera.up.transformDirection(t);break;case"ArrowRight":this.camera.up.transformDirection(n);break;case"KeyC":this.showMeshCursor=!this.showMeshCursor;break;case"KeyU":this.showControlPlane=!this.showControlPlane;break;case"KeyI":this.showInfo=!this.showInfo,this.showInfo?this.infoPanel.show():this.infoPanel.hide();break;case"KeyO":this.usingExternalCamera||this.setOrthographicMode(!this.camera.isOrthographicCamera);break;case"KeyP":this.usingExternalCamera||this.splatMesh.setPointCloudModeEnabled(!this.splatMesh.getPointCloudModeEnabled());break;case"Equal":this.usingExternalCamera||this.splatMesh.setSplatScale(this.splatMesh.getSplatScale()+.05);break;case"Minus":this.usingExternalCamera||this.splatMesh.setSplatScale(Math.max(this.splatMesh.getSplatScale()-.05,0));break}}})();onMouseMove(e){this.mousePosition.set(e.offsetX,e.offsetY)}onMouseDown(){this.mouseDownPosition.copy(this.mousePosition),this.mouseDownTime=Br()}onMouseUp=(function(){const e=new Ne;return function(t){e.copy(this.mousePosition).sub(this.mouseDownPosition),Br()-this.mouseDownTime<.5&&e.length()<2&&this.onMouseClick(t)}})();onMouseClick(e){this.mousePosition.set(e.offsetX,e.offsetY),this.checkForFocalPointChange()}checkForFocalPointChange=(function(){const e=new Ne,t=new L,n=[];return function(){if(!this.transitioningCameraTarget&&(this.getRenderDimensions(e),n.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,e),this.raycaster.intersectSplatMesh(this.splatMesh,n),n.length>0)){const r=n[0].origin;t.copy(r).sub(this.camera.position),t.length()>FE&&(this.previousCameraTarget.copy(this.controls.target),this.nextCameraTarget.copy(r),this.transitioningCameraTarget=!0,this.transitioningCameraTargetStartTime=Br())}}})();getRenderDimensions(e){this.rootElement?(e.x=this.rootElement.offsetWidth,e.y=this.rootElement.offsetHeight):this.renderer.getSize(e)}setOrthographicMode(e){if(e===this.camera.isOrthographicCamera)return;const t=this.camera,n=e?this.orthographicCamera:this.perspectiveCamera;if(n.position.copy(t.position),n.up.copy(t.up),n.rotation.copy(t.rotation),n.quaternion.copy(t.quaternion),n.matrix.copy(t.matrix),this.camera=n,this.controls){const s=a=>{a.saveState(),a.reset()},r=this.controls,o=e?this.orthographicControls:this.perspectiveControls;s(o),s(r),o.target.copy(r.target),e?Vr.setCameraZoomFromPosition(n,t,r):Vr.setCameraPositionFromZoom(n,t,o),this.controls=o,this.camera.lookAt(this.controls.target)}}static setCameraPositionFromZoom=(function(){const e=new L;return function(t,n,s){const r=1/(n.zoom*.001);e.copy(s.target).sub(t.position).normalize().multiplyScalar(r).negate(),t.position.copy(s.target).add(e)}})();static setCameraZoomFromPosition=(function(){const e=new L;return function(t,n,s){const r=e.copy(s.target).sub(n.position).length();t.zoom=1/(r*.001)}})();updateSplatMesh=(function(){const e=new Ne;return function(){if(!this.splatMesh)return;if(this.splatMesh.getSplatCount()>0){this.splatMesh.updateVisibleRegionFadeDistance(this.sceneRevealMode),this.splatMesh.updateTransforms(),this.getRenderDimensions(e);const n=this.camera.projectionMatrix.elements[0]*.5*this.devicePixelRatio*e.x,s=this.camera.projectionMatrix.elements[5]*.5*this.devicePixelRatio*e.y,r=this.camera.isOrthographicCamera?1/this.devicePixelRatio:1,o=this.focalAdjustment*r,a=1/o;this.adjustForWebXRStereo(e),this.splatMesh.updateUniforms(e,n*o,s*o,this.camera.isOrthographicCamera,this.camera.zoom||1,a)}}})();adjustForWebXRStereo(e){if(this.camera&&this.webXRActive){const n=this.renderer.xr.getCamera().projectionMatrix.elements[0],s=this.camera.projectionMatrix.elements[0];e.x*=s/n}}isLoadingOrUnloading(){return Object.keys(this.splatSceneDownloadPromises).length>0||this.splatSceneDownloadAndBuildPromise!==null||this.splatSceneRemovalPromise!==null}isDisposingOrDisposed(){return this.disposing||this.disposed}addSplatSceneDownloadPromise(e){this.splatSceneDownloadPromises[e.id]=e}removeSplatSceneDownloadPromise(e){delete this.splatSceneDownloadPromises[e.id]}setSplatSceneDownloadAndBuildPromise(e){this.splatSceneDownloadAndBuildPromise=e}clearSplatSceneDownloadAndBuildPromise(){this.splatSceneDownloadAndBuildPromise=null}addSplatScene(e,t={}){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");t.progressiveLoad&&this.splatMesh.scenes&&this.splatMesh.scenes.length>0&&(console.log('addSplatScene(): "progressiveLoad" option ignore because there are multiple splat scenes'),t.progressiveLoad=!1);const n=t.format!==void 0&&t.format!==null?t.format:Hp(e),s=Vr.isProgressivelyLoadable(n)&&t.progressiveLoad,r=t.showLoadingUI!==void 0&&t.showLoadingUI!==null?t.showLoadingUI:!0;let o=null;r&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=()=>{this.loadingProgressBar.hide(),this.loadingSpinner.removeAllTasks()},l=(p,x,m)=>{if(r)if(m===Nt.Downloading)if(p==100)this.loadingSpinner.setMessageForTask(o,"Download complete!");else if(s)this.loadingSpinner.setMessageForTask(o,"Downloading splats...");else{const _=x?`: ${x}`:"...";this.loadingSpinner.setMessageForTask(o,`Downloading${_}`)}else m===Nt.Processing&&this.loadingSpinner.setMessageForTask(o,"Processing splats...")};let c=!1,u=0;const f=(p,x)=>{r&&((p&&s||x&&!s)&&(this.loadingSpinner.removeTask(o),!x&&!c&&this.loadingProgressBar.show()),s&&(x?(c=!0,this.loadingProgressBar.hide()):this.loadingProgressBar.setProgress(u)))},d=(p,x,m)=>{u=p,l(p,x,m),t.onProgress&&t.onProgress(p,x,m)},h=(p,x,m)=>{!s&&t.onProgress&&t.onProgress(0,"0%",Nt.Processing);const _={rotation:t.rotation||t.orientation,position:t.position,scale:t.scale,splatAlphaRemovalThreshold:t.splatAlphaRemovalThreshold};return this.addSplatBuffers([p],[_],m,x&&r,r,s,s).then(()=>{!s&&t.onProgress&&t.onProgress(100,"100%",Nt.Processing),f(x,m)})};return(s?this.downloadAndBuildSingleSplatSceneProgressiveLoad.bind(this):this.downloadAndBuildSingleSplatSceneStandardLoad.bind(this))(e,n,t.splatAlphaRemovalThreshold,h.bind(this),d,a.bind(this),t.headers)}downloadAndBuildSingleSplatSceneStandardLoad(e,t,n,s,r,o,a){const l=this.downloadSplatSceneToSplatBuffer(e,n,r,!1,void 0,t,a),c=Yc(l.abortHandler);return l.then(u=>(this.removeSplatSceneDownloadPromise(l),s(u,!0,!0).then(()=>{c.resolve(),this.clearSplatSceneDownloadAndBuildPromise()}))).catch(u=>{o&&o(),this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(l),c.reject(this.updateError(u,`Viewer::addSplatScene -> Could not load file ${e}`))}),this.addSplatSceneDownloadPromise(l),this.setSplatSceneDownloadAndBuildPromise(c.promise),c.promise}downloadAndBuildSingleSplatSceneProgressiveLoad(e,t,n,s,r,o,a){let l=0,c=!1;const u=[],f=()=>{if(u.length>0&&!c&&!this.isDisposingOrDisposed()){c=!0;const x=u.shift();s(x.splatBuffer,x.firstBuild,x.finalBuild).then(()=>{c=!1,x.firstBuild?g.resolve():x.finalBuild&&(p.resolve(),this.clearSplatSceneDownloadAndBuildPromise()),u.length>0&&Gn(()=>f())})}},d=(x,m)=>{this.isDisposingOrDisposed()||(m||u.length===0||x.getSplatCount()>u[0].splatBuffer.getSplatCount())&&(u.push({splatBuffer:x,firstBuild:l===0,finalBuild:m}),l++,f())},h=this.downloadSplatSceneToSplatBuffer(e,n,r,!0,d,t,a),g=Yc(h.abortHandler),p=Yc();return this.addSplatSceneDownloadPromise(h),this.setSplatSceneDownloadAndBuildPromise(p.promise),h.then(()=>{this.removeSplatSceneDownloadPromise(h)}).catch(x=>{this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(h);const m=this.updateError(x,"Viewer::addSplatScene -> Could not load one or more scenes");g.reject(m),o&&o(m)}),g.promise}addSplatScenes(e,t=!0,n=void 0){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");const s=e.length,r=[];let o;t&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=(f,d,h,g)=>{r[f]=d;let p=0;for(let x=0;x<s;x++)p+=r[x]||0;p=p/s,h=`${p.toFixed(2)}%`,t&&g===Nt.Downloading&&this.loadingSpinner.setMessageForTask(o,p==100?"Download complete!":`Downloading: ${h}`),n&&n(p,h,g)},l=[],c=[];for(let f=0;f<e.length;f++){const d=e[f],h=d.format!==void 0&&d.format!==null?d.format:Hp(d.path),g=this.downloadSplatSceneToSplatBuffer(d.path,d.splatAlphaRemovalThreshold,a.bind(this,f),!1,void 0,h,d.headers);l.push(g),c.push(g.promise)}const u=new Cs((f,d)=>{Promise.all(c).then(h=>{t&&this.loadingSpinner.removeTask(o),n&&n(0,"0%",Nt.Processing),this.addSplatBuffers(h,e,!0,t,t,!1,!1).then(()=>{n&&n(100,"100%",Nt.Processing),this.clearSplatSceneDownloadAndBuildPromise(),f()})}).catch(h=>{t&&this.loadingSpinner.removeTask(o),this.clearSplatSceneDownloadAndBuildPromise(),d(this.updateError(h,"Viewer::addSplatScenes -> Could not load one or more splat scenes."))}).finally(()=>{this.removeSplatSceneDownloadPromise(u)})},f=>{for(let d of l)d.abort(f)});return this.addSplatSceneDownloadPromise(u),this.setSplatSceneDownloadAndBuildPromise(u),u}downloadSplatSceneToSplatBuffer(e,t=1,n=void 0,s=!1,r=void 0,o,a){try{if(o===Tn.Splat||o===Tn.KSplat||o===Tn.Ply){const l=s?!1:this.optimizeSplatData;if(o===Tn.Splat)return xd.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,a);if(o===Tn.KSplat)return Xo.loadFromURL(e,n,s,r,a);if(o===Tn.Ply)return md.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,this.sphericalHarmonicsDegree,a)}else if(o===Tn.Spz)return gd.loadFromURL(e,n,t,this.inMemoryCompressionLevel,this.optimizeSplatData,this.sphericalHarmonicsDegree,a)}catch(l){throw this.updateError(l,null)}throw new Error(`Viewer::downloadSplatSceneToSplatBuffer -> File format not supported: ${e}`)}static isProgressivelyLoadable(e){return e===Tn.Splat||e===Tn.KSplat||e===Tn.Ply}addSplatBuffers=(function(){return function(e,t=[],n=!0,s=!0,r=!0,o=!1,a=!1,l=!0){if(this.isDisposingOrDisposed())return Promise.resolve();let c=null;const u=()=>{c!==null&&(this.loadingSpinner.removeTask(c),c=null)};return this.splatRenderReady=!1,new Promise(f=>{s&&(c=this.loadingSpinner.addTask("Processing splats...")),Gn(()=>{if(this.isDisposingOrDisposed())f();else{const d=this.addSplatBuffersToMesh(e,t,n,r,o,l),h=this.splatMesh.getMaxSplatCount();this.sortWorker&&this.sortWorker.maxSplatCount!==h&&this.disposeSortWorker(),this.gpuAcceleratedSort||this.preSortMessages.push({centers:d.centers.buffer,sceneIndexes:d.sceneIndexes.buffer,range:{from:d.from,to:d.to,count:d.count}}),(!this.sortWorker&&h>0?this.setupSortWorker(this.splatMesh):Promise.resolve()).then(()=>{this.isDisposingOrDisposed()||this.runSplatSort(!0,!0).then(p=>{!this.sortWorker||!p?(this.splatRenderReady=!0,u(),f()):(a?this.splatRenderReady=!0:this.runAfterNextSort.push(()=>{this.splatRenderReady=!0}),this.runAfterNextSort.push(()=>{u(),f()}))})})}},!0)})}})();addSplatBuffersToMesh=(function(){let e;return function(t,n,s=!0,r=!1,o=!1,a=!0){if(this.isDisposingOrDisposed())return;let l=[],c=[];o||(l=this.splatMesh.scenes.map(h=>h.splatBuffer)||[],c=this.splatMesh.sceneOptions?this.splatMesh.sceneOptions.map(h=>h):[]),l.push(...t),c.push(...n),this.renderer&&this.splatMesh.setRenderer(this.renderer);const u=h=>{if(this.isDisposingOrDisposed())return;const g=this.splatMesh.getSplatCount();r&&g>=LE&&!h&&!e&&(this.loadingSpinner.setMinimized(!0,!0),e=this.loadingSpinner.addTask("Optimizing data structures..."))},f=h=>{this.isDisposingOrDisposed()||h&&e&&(this.loadingSpinner.removeTask(e),e=null)},d=this.splatMesh.build(l,c,!0,s,u,f,a);return s&&this.freeIntermediateSplatData&&this.splatMesh.freeIntermediateSplatData(),d}})();setupSortWorker(e){if(!this.isDisposingOrDisposed())return new Promise(t=>{const n=this.integerBasedSort?Int32Array:Float32Array,s=e.getSplatCount(),r=e.getMaxSplatCount();this.sortWorker=IE(r,this.sharedMemoryForWorkers,this.enableSIMDInSort,this.integerBasedSort,this.splatMesh.dynamicMode,this.splatSortDistanceMapPrecision),this.sortWorker.onmessage=o=>{if(o.data.sortDone){if(this.sortRunning=!1,this.sharedMemoryForWorkers)this.splatMesh.updateRenderIndexes(this.sortWorkerSortedIndexes,o.data.splatRenderCount);else{const a=new Uint32Array(o.data.sortedIndexes.buffer,0,o.data.splatRenderCount);this.splatMesh.updateRenderIndexes(a,o.data.splatRenderCount)}this.lastSplatSortCount=this.splatSortCount,this.lastSortTime=o.data.sortTime,this.sortPromiseResolver(),this.sortPromiseResolver=null,this.forceRenderNextFrame(),this.runAfterNextSort.length>0&&(this.runAfterNextSort.forEach(a=>{a()}),this.runAfterNextSort.length=0)}else if(o.data.sortCanceled)this.sortRunning=!1;else if(o.data.sortSetupPhase1Complete){this.logLevel>=eo.Info&&console.log("Sorting web worker WASM setup complete."),this.sharedMemoryForWorkers?(this.sortWorkerSortedIndexes=new Uint32Array(o.data.sortedIndexesBuffer,o.data.sortedIndexesOffset,r),this.sortWorkerIndexesToSort=new Uint32Array(o.data.indexesToSortBuffer,o.data.indexesToSortOffset,r),this.sortWorkerPrecomputedDistances=new n(o.data.precomputedDistancesBuffer,o.data.precomputedDistancesOffset,r),this.sortWorkerTransforms=new Float32Array(o.data.transformsBuffer,o.data.transformsOffset,pt.MaxScenes*16)):(this.sortWorkerIndexesToSort=new Uint32Array(r),this.sortWorkerPrecomputedDistances=new n(r),this.sortWorkerTransforms=new Float32Array(pt.MaxScenes*16));for(let a=0;a<s;a++)this.sortWorkerIndexesToSort[a]=a;if(this.sortWorker.maxSplatCount=r,this.logLevel>=eo.Info){console.log("Sorting web worker ready.");const a=this.splatMesh.getSplatDataTextures(),l=a.covariances.size,c=a.centerColors.size;console.log("Covariances texture size: "+l.x+" x "+l.y),console.log("Centers/colors texture size: "+c.x+" x "+c.y)}t()}}})}updateError(e,t){return e instanceof J0?e:e instanceof Il?new Error("File type or server does not support progressive loading."):t?new Error(t):e}disposeSortWorker(){this.sortWorker&&this.sortWorker.terminate(),this.sortWorker=null,this.sortPromise=null,this.sortPromiseResolver&&(this.sortPromiseResolver(),this.sortPromiseResolver=null),this.preSortMessages=[],this.sortRunning=!1}removeSplatScene(e,t=!0){return this.removeSplatScenes([e],t)}removeSplatScenes(e,t=!0){if(this.isLoadingOrUnloading())throw new Error("Cannot remove splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot remove splat scene after dispose() is called.");let n;return this.splatSceneRemovalPromise=new Promise((s,r)=>{let o;t&&(this.loadingSpinner.removeAllTasks(),this.loadingSpinner.show(),o=this.loadingSpinner.addTask("Removing splat scene..."));const a=()=>{t&&(this.loadingSpinner.hide(),this.loadingSpinner.removeTask(o))},l=u=>{a(),this.splatSceneRemovalPromise=null,u?r(u):s()},c=()=>this.isDisposingOrDisposed()?(l(),!0):!1;n=this.sortPromise||Promise.resolve(),n.then(()=>{if(c())return;const u=[],f=[],d=[];for(let h=0;h<this.splatMesh.scenes.length;h++){let g=!1;for(let p of e)if(p===h){g=!0;break}if(!g){const p=this.splatMesh.scenes[h];u.push(p.splatBuffer),f.push(this.splatMesh.sceneOptions[h]),d.push({position:p.position.clone(),quaternion:p.quaternion.clone(),scale:p.scale.clone()})}}this.disposeSortWorker(),this.splatMesh.dispose(),this.sceneRevealMode=Yo.Instant,this.createSplatMesh(),this.addSplatBuffers(u,f,!0,!1,!0).then(()=>{c()||(a(),this.splatMesh.scenes.forEach((h,g)=>{h.position.copy(d[g].position),h.quaternion.copy(d[g].quaternion),h.scale.copy(d[g].scale)}),this.splatMesh.updateTransforms(),this.splatRenderReady=!1,this.runSplatSort(!0).then(()=>{if(c()){this.splatRenderReady=!0;return}n=this.sortPromise||Promise.resolve(),n.then(()=>{this.splatRenderReady=!0,l()})}))}).catch(h=>{l(h)})})}),this.splatSceneRemovalPromise}start(){if(this.selfDrivenMode)this.webXRMode?this.renderer.setAnimationLoop(this.selfDrivenUpdateFunc):this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc),this.selfDrivenModeRunning=!0;else throw new Error("Cannot start viewer unless it is in self driven mode.")}stop(){this.selfDrivenMode&&this.selfDrivenModeRunning&&(this.webXRMode?this.renderer.setAnimationLoop(null):cancelAnimationFrame(this.requestFrameId),this.selfDrivenModeRunning=!1)}async dispose(){if(this.isDisposingOrDisposed())return this.disposePromise;let e=[],t=[];for(let n in this.splatSceneDownloadPromises)if(this.splatSceneDownloadPromises.hasOwnProperty(n)){const s=this.splatSceneDownloadPromises[n];t.push(s),e.push(s.promise)}return this.sortPromise&&e.push(this.sortPromise),this.disposing=!0,this.disposePromise=Promise.all(e).finally(()=>{this.stop(),this.orthographicControls&&(this.orthographicControls.dispose(),this.orthographicControls=null),this.perspectiveControls&&(this.perspectiveControls.dispose(),this.perspectiveControls=null),this.controls=null,this.splatMesh&&(this.splatMesh.dispose(),this.splatMesh=null),this.sceneHelper&&(this.sceneHelper.dispose(),this.sceneHelper=null),this.resizeObserver&&(this.resizeObserver.unobserve(this.rootElement),this.resizeObserver=null),this.disposeSortWorker(),this.removeEventHandlers(),this.loadingSpinner.removeAllTasks(),this.loadingSpinner.setContainer(null),this.loadingProgressBar.hide(),this.loadingProgressBar.setContainer(null),this.infoPanel.setContainer(null),this.camera=null,this.threeScene=null,this.splatRenderReady=!1,this.initialized=!1,this.renderer&&(this.usingExternalRenderer||(this.rootElement.removeChild(this.renderer.domElement),this.renderer.dispose()),this.renderer=null),this.usingExternalRenderer||document.body.removeChild(this.rootElement),this.sortWorkerSortedIndexes=null,this.sortWorkerIndexesToSort=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.disposed=!0,this.disposing=!1,this.disposePromise=null}),t.forEach(n=>{n.abort("Scene disposed")}),this.disposePromise}selfDrivenUpdate(){this.selfDrivenMode&&!this.webXRMode&&(this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc)),this.update(),this.shouldRender()?(this.render(),this.consecutiveRenderFrames++):this.consecutiveRenderFrames=0,this.renderNextFrame=!1}forceRenderNextFrame(){this.renderNextFrame=!0}shouldRender=(function(){let e=0;const t=new L,n=new bt,s=1e-4;return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return!1;let r=!1,o=!1;if(this.camera){const a=this.camera.position,l=this.camera.quaternion;o=Math.abs(a.x-t.x)>s||Math.abs(a.y-t.y)>s||Math.abs(a.z-t.z)>s||Math.abs(l.x-n.x)>s||Math.abs(l.y-n.y)>s||Math.abs(l.z-n.z)>s||Math.abs(l.w-n.w)>s}return r=this.renderMode!==nu.Never&&(e===0||this.splatMesh.visibleRegionChanging||o||this.renderMode===nu.Always||this.dynamicMode===!0||this.renderNextFrame),this.camera&&(t.copy(this.camera.position),n.copy(this.camera.quaternion)),e++,r}})();render=(function(){return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return;const e=n=>{for(let s of n.children)if(s.visible)return!0;return!1},t=this.renderer.autoClear;e(this.threeScene)&&(this.renderer.render(this.threeScene,this.camera),this.renderer.autoClear=!1),this.renderer.render(this.splatMesh,this.camera),this.renderer.autoClear=!1,this.sceneHelper.getFocusMarkerOpacity()>0&&this.renderer.render(this.sceneHelper.focusMarker,this.camera),this.showControlPlane&&this.renderer.render(this.sceneHelper.controlPlane,this.camera),this.renderer.autoClear=t}})();update(e,t){this.dropInMode&&this.updateForDropInMode(e,t),!(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())&&(this.controls&&(this.controls.update(),this.camera.isOrthographicCamera&&!this.usingExternalCamera&&Vr.setCameraPositionFromZoom(this.camera,this.camera,this.controls)),this.runSplatSort(),this.updateForRendererSizeChanges(),this.updateSplatMesh(),this.updateMeshCursor(),this.updateFPS(),this.timingSensitiveUpdates(),this.updateInfoPanel(),this.updateControlPlane())}updateForDropInMode(e,t){this.renderer=e,this.splatMesh&&this.splatMesh.setRenderer(this.renderer),this.camera=t,this.controls&&(this.controls.object=t),this.init()}updateFPS=(function(){let e=Br(),t=0;return function(){if(this.consecutiveRenderFrames>OE){const n=Br();n-e>=1?(this.currentFPS=t,t=0,e=n):t++}else this.currentFPS=null}})();updateForRendererSizeChanges=(function(){const e=new Ne,t=new Ne;let n;return function(){this.usingExternalCamera||(this.renderer.getSize(t),(n===void 0||n!==this.camera.isOrthographicCamera||t.x!==e.x||t.y!==e.y)&&(this.camera.isOrthographicCamera?(this.camera.left=-t.x/2,this.camera.right=t.x/2,this.camera.top=t.y/2,this.camera.bottom=-t.y/2):this.camera.aspect=t.x/t.y,this.camera.updateProjectionMatrix(),e.copy(t),n=this.camera.isOrthographicCamera))}})();timingSensitiveUpdates=(function(){let e;return function(){const t=Br();e||(e=t);const n=t-e;this.updateCameraTransition(t),this.updateFocusMarker(n),e=t}})();updateCameraTransition=(function(){let e=new L,t=new L,n=new L;return function(s){if(this.transitioningCameraTarget){t.copy(this.previousCameraTarget).sub(this.camera.position).normalize(),n.copy(this.nextCameraTarget).sub(this.camera.position).normalize();const r=Math.acos(t.dot(n)),a=(r/(Math.PI/3)*.65+.3)/r*(s-this.transitioningCameraTargetStartTime);e.copy(this.previousCameraTarget).lerp(this.nextCameraTarget,a),this.camera.lookAt(e),this.controls.target.copy(e),a>=1&&(this.transitioningCameraTarget=!1)}}})();updateFocusMarker=(function(){const e=new Ne;let t=!1;return function(n){if(this.getRenderDimensions(e),this.transitioningCameraTarget){this.sceneHelper.setFocusMarkerVisibility(!0);const s=Math.max(this.sceneHelper.getFocusMarkerOpacity(),0);let r=Math.min(s+BE*n,1);this.sceneHelper.setFocusMarkerOpacity(r),this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e),t=!0,this.forceRenderNextFrame()}else{let s;if(t?s=1:s=Math.min(this.sceneHelper.getFocusMarkerOpacity(),1),s>0){this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e);let r=Math.max(s-UE*n,0);this.sceneHelper.setFocusMarkerOpacity(r),r===0&&this.sceneHelper.setFocusMarkerVisibility(!1)}s>0&&this.forceRenderNextFrame(),t=!1}}})();updateMeshCursor=(function(){const e=[],t=new Ne;return function(){this.showMeshCursor?(this.forceRenderNextFrame(),this.getRenderDimensions(t),e.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,t),this.raycaster.intersectSplatMesh(this.splatMesh,e),e.length>0?(this.sceneHelper.setMeshCursorVisibility(!0),this.sceneHelper.positionAndOrientMeshCursor(e[0].origin,this.camera)):this.sceneHelper.setMeshCursorVisibility(!1)):(this.sceneHelper.getMeschCursorVisibility()&&this.forceRenderNextFrame(),this.sceneHelper.setMeshCursorVisibility(!1))}})();updateInfoPanel=(function(){const e=new Ne;return function(){if(!this.showInfo)return;const t=this.splatMesh.getSplatCount();this.getRenderDimensions(e);const n=this.controls?this.controls.target:null,s=this.showMeshCursor?this.sceneHelper.meshCursor.position:null,r=t>0?this.splatRenderCount/t*100:0;this.infoPanel.update(e,this.camera.position,n,this.camera.up,this.camera.isOrthographicCamera,s,this.currentFPS||"N/A",t,this.splatRenderCount,r,this.lastSortTime,this.focalAdjustment,this.splatMesh.getSplatScale(),this.splatMesh.getPointCloudModeEnabled())}})();updateControlPlane(){this.showControlPlane?(this.sceneHelper.setControlPlaneVisibility(!0),this.sceneHelper.positionAndOrientControlPlane(this.controls.target,this.camera.up)):this.sceneHelper.setControlPlaneVisibility(!1)}runSplatSort=(function(){const e=new qe,t=[],n=new L(0,0,-1),s=new L(0,0,-1),r=new L,o=new L,a=[],l=[{angleThreshold:.55,sortFractions:[.125,.33333,.75]},{angleThreshold:.65,sortFractions:[.33333,.66667]},{angleThreshold:.8,sortFractions:[.5]}];return function(c=!1,u=!1){if(!this.initialized)return Promise.resolve(!1);if(this.sortRunning)return Promise.resolve(!0);if(this.splatMesh.getSplatCount()<=0)return this.splatRenderCount=0,Promise.resolve(!1);let f=0,d=0,h=!1,g=!1;if(s.set(0,0,-1).applyQuaternion(this.camera.quaternion),f=s.dot(n),d=o.copy(this.camera.position).sub(r).length(),!c&&!this.splatMesh.dynamicMode&&a.length===0&&(f<=.99&&(h=!0),d>=1&&(g=!0),!h&&!g))return Promise.resolve(!1);this.sortRunning=!0;let{splatRenderCount:p,shouldSortAll:x}=this.gatherSceneNodesForSort();x=x||u,this.splatRenderCount=p,e.copy(this.camera.matrixWorld).invert();const m=this.perspectiveCamera||this.camera;e.premultiply(m.projectionMatrix),this.splatMesh.dynamicMode||e.multiply(this.splatMesh.matrixWorld);let _=Promise.resolve(!0);return this.gpuAcceleratedSort&&(a.length<=1||a.length%2===0)&&(_=this.splatMesh.computeDistancesOnGPU(e,this.sortWorkerPrecomputedDistances)),_.then(()=>{if(a.length===0)if(this.splatMesh.dynamicMode||x)a.push(this.splatRenderCount);else{for(let v of l)if(f<v.angleThreshold){for(let b of v.sortFractions)a.push(Math.floor(this.splatRenderCount*b));break}a.push(this.splatRenderCount)}let A=Math.min(a.shift(),this.splatRenderCount);this.splatSortCount=A,t[0]=this.camera.position.x,t[1]=this.camera.position.y,t[2]=this.camera.position.z;const S={modelViewProj:e.elements,cameraPosition:t,splatRenderCount:this.splatRenderCount,splatSortCount:A,usePrecomputedDistances:this.gpuAcceleratedSort};return this.splatMesh.dynamicMode&&this.splatMesh.fillTransformsArray(this.sortWorkerTransforms),this.sharedMemoryForWorkers||(S.indexesToSort=this.sortWorkerIndexesToSort,S.transforms=this.sortWorkerTransforms,this.gpuAcceleratedSort&&(S.precomputedDistances=this.sortWorkerPrecomputedDistances)),this.sortPromise=new Promise(v=>{this.sortPromiseResolver=v}),this.preSortMessages.length>0&&(this.preSortMessages.forEach(v=>{this.sortWorker.postMessage(v)}),this.preSortMessages=[]),this.sortWorker.postMessage({sort:S}),a.length===0&&(r.copy(this.camera.position),n.copy(s)),!0}),_}})();gatherSceneNodesForSort=(function(){const e=[];let t=null;const n=new L,s=new L,r=new L,o=new qe,a=new qe,l=new qe,c=new L,u=new L(0,0,-1),f=new L,d=h=>f.copy(h.max).sub(h.min).length();return function(h=!1){this.getRenderDimensions(c);const g=c.y/2/Math.tan(this.camera.fov/2*U0.DEG2RAD),p=Math.atan(c.x/2/g),x=Math.atan(c.y/2/g),m=Math.cos(p),_=Math.cos(x),A=this.splatMesh.getSplatTree();if(A){a.copy(this.camera.matrixWorld).invert(),this.splatMesh.dynamicMode||a.multiply(this.splatMesh.matrixWorld);let S=0,v=0;for(let M=0;M<A.subTrees.length;M++){const E=A.subTrees[M];o.copy(a),this.splatMesh.dynamicMode&&(this.splatMesh.getSceneTransform(M,l),o.multiply(l));const y=E.nodesWithIndexes.length;for(let C=0;C<y;C++){const I=E.nodesWithIndexes[C];if(!I.data||!I.data.indexes||I.data.indexes.length===0)continue;r.copy(I.center).applyMatrix4(o);const P=r.length();r.normalize(),n.copy(r).setX(0).normalize(),s.copy(r).setY(0).normalize();const U=u.dot(s),O=u.dot(n),k=d(I),z=O<_-.6,W=U<m-.6;!h&&(W||z)&&P>k||(v+=I.data.indexes.length,e[S]=I,I.data.distanceToNode=P,S++)}}e.length=S,e.sort((M,E)=>M.data.distanceToNode<E.data.distanceToNode?-1:1);let b=v*pt.BytesPerInt;for(let M=0;M<S;M++){const E=e[M],y=E.data.indexes.length,C=y*pt.BytesPerInt;new Uint32Array(this.sortWorkerIndexesToSort.buffer,b-C,y).set(E.data.indexes),b-=C}return{splatRenderCount:v,shouldSortAll:!1}}else{const S=this.splatMesh.getSplatCount();if(!t||t.length!==S){t=new Uint32Array(S);for(let v=0;v<S;v++)t[v]=v}return this.sortWorkerIndexesToSort.set(t),{splatRenderCount:S,shouldSortAll:!0}}}})();getSplatMesh(){return this.splatMesh}getSplatScene(e){return this.splatMesh.getScene(e)}getSceneCount(){return this.splatMesh.getSceneCount()}isMobile(){return navigator.userAgent.includes("Mobi")}}function Ki(i){if(i===void 0)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return i}function lg(i,e){i.prototype=Object.create(e.prototype),i.prototype.constructor=i,i.__proto__=e}var Qn={autoSleep:120,force3D:"auto",nullTargetWarn:1,units:{lineHeight:""}},mo={duration:.5,overwrite:!1,delay:0},vd,Jt,Tt,ri=1e8,_t=1/ri,hf=Math.PI*2,NE=hf/4,zE=0,cg=Math.sqrt,kE=Math.cos,HE=Math.sin,jt=function(e){return typeof e=="string"},Ut=function(e){return typeof e=="function"},ls=function(e){return typeof e=="number"},yd=function(e){return typeof e>"u"},Oi=function(e){return typeof e=="object"},wn=function(e){return e!==!1},bd=function(){return typeof window<"u"},ol=function(e){return Ut(e)||jt(e)},ug=typeof ArrayBuffer=="function"&&ArrayBuffer.isView||function(){},an=Array.isArray,VE=/random\([^)]+\)/g,GE=/,\s*/g,$p=/(?:-?\.?\d|\.)+/gi,fg=/[-+=.]*\d+[.e\-+]*\d*[e\-+]*\d*/g,Gr=/[-+=.]*\d+[.e-]*\d*[a-z%]*/g,iu=/[-+=.]*\d+\.?\d*(?:e-|e\+)?\d*/gi,dg=/[+-]=-?[.\d]+/,WE=/[^,'"\[\]\s]+/gi,XE=/^[+\-=e\s\d]*\d+[.\d]*([a-z]*|%)\s*$/i,Pt,yi,pf,Md,Yn={},Fl={},hg,pg=function(e){return(Fl=go(e,Yn))&&Pn},Cd=function(e,t){return console.warn("Invalid property",e,"set to",t,"Missing plugin? gsap.registerPlugin()")},fa=function(e,t){return!t&&console.warn(e)},mg=function(e,t){return e&&(Yn[e]=t)&&Fl&&(Fl[e]=t)||Yn},da=function(){return 0},qE={suppressEvents:!0,isStart:!0,kill:!1},xl={suppressEvents:!0,kill:!1},QE={suppressEvents:!0},Td={},Ps=[],mf={},gg,kn={},su={},Zp=30,_l=[],Ed="",wd=function(e){var t=e[0],n,s;if(Oi(t)||Ut(t)||(e=[e]),!(n=(t._gsap||{}).harness)){for(s=_l.length;s--&&!_l[s].targetTest(t););n=_l[s]}for(s=e.length;s--;)e[s]&&(e[s]._gsap||(e[s]._gsap=new kg(e[s],n)))||e.splice(s,1);return e},cr=function(e){return e._gsap||wd(oi(e))[0]._gsap},xg=function(e,t,n){return(n=e[t])&&Ut(n)?e[t]():yd(n)&&e.getAttribute&&e.getAttribute(t)||n},Rn=function(e,t){return(e=e.split(",")).forEach(t)||e},Ot=function(e){return Math.round(e*1e5)/1e5||0},Dt=function(e){return Math.round(e*1e7)/1e7||0},to=function(e,t){var n=t.charAt(0),s=parseFloat(t.substr(2));return e=parseFloat(e),n==="+"?e+s:n==="-"?e-s:n==="*"?e*s:e/s},YE=function(e,t){for(var n=t.length,s=0;e.indexOf(t[s])<0&&++s<n;);return s<n},Ll=function(){var e=Ps.length,t=Ps.slice(0),n,s;for(mf={},Ps.length=0,n=0;n<e;n++)s=t[n],s&&s._lazy&&(s.render(s._lazy[0],s._lazy[1],!0)._lazy=0)},Rd=function(e){return!!(e._initted||e._startAt||e.add)},_g=function(e,t,n,s){Ps.length&&!Jt&&Ll(),e.render(t,n,!!(Jt&&t<0&&Rd(e))),Ps.length&&!Jt&&Ll()},Ag=function(e){var t=parseFloat(e);return(t||t===0)&&(e+"").match(WE).length<2?t:jt(e)?e.trim():e},Sg=function(e){return e},Kn=function(e,t){for(var n in t)n in e||(e[n]=t[n]);return e},KE=function(e){return function(t,n){for(var s in n)s in t||s==="duration"&&e||s==="ease"||(t[s]=n[s])}},go=function(e,t){for(var n in t)e[n]=t[n];return e},Jp=function i(e,t){for(var n in t)n!=="__proto__"&&n!=="constructor"&&n!=="prototype"&&(e[n]=Oi(t[n])?i(e[n]||(e[n]={}),t[n]):t[n]);return e},Bl=function(e,t){var n={},s;for(s in e)s in t||(n[s]=e[s]);return n},Ko=function(e){var t=e.parent||Pt,n=e.keyframes?KE(an(e.keyframes)):Kn;if(wn(e.inherit))for(;t;)n(e,t.vars.defaults),t=t.parent||t._dp;return e},jE=function(e,t){for(var n=e.length,s=n===t.length;s&&n--&&e[n]===t[n];);return n<0},vg=function(e,t,n,s,r){var o=e[s],a;if(r)for(a=t[r];o&&o[r]>a;)o=o._prev;return o?(t._next=o._next,o._next=t):(t._next=e[n],e[n]=t),t._next?t._next._prev=t:e[s]=t,t._prev=o,t.parent=t._dp=e,t},tc=function(e,t,n,s){n===void 0&&(n="_first"),s===void 0&&(s="_last");var r=t._prev,o=t._next;r?r._next=o:e[n]===t&&(e[n]=o),o?o._prev=r:e[s]===t&&(e[s]=r),t._next=t._prev=t.parent=null},Os=function(e,t){e.parent&&(!t||e.parent.autoRemoveChildren)&&e.parent.remove&&e.parent.remove(e),e._act=0},ur=function(e,t){if(e&&(!t||t._end>e._dur||t._start<0))for(var n=e;n;)n._dirty=1,n=n.parent;return e},$E=function(e){for(var t=e.parent;t&&t.parent;)t._dirty=1,t.totalDuration(),t=t.parent;return e},gf=function(e,t,n,s){return e._startAt&&(Jt?e._startAt.revert(xl):e.vars.immediateRender&&!e.vars.autoRevert||e._startAt.render(t,!0,s))},ZE=function i(e){return!e||e._ts&&i(e.parent)},em=function(e){return e._repeat?xo(e._tTime,e=e.duration()+e._rDelay)*e:0},xo=function(e,t){var n=Math.floor(e=Dt(e/t));return e&&n===e?n-1:n},Ul=function(e,t){return(e-t._start)*t._ts+(t._ts>=0?0:t._dirty?t.totalDuration():t._tDur)},nc=function(e){return e._end=Dt(e._start+(e._tDur/Math.abs(e._ts||e._rts||_t)||0))},ic=function(e,t){var n=e._dp;return n&&n.smoothChildTiming&&e._ts&&(e._start=Dt(n._time-(e._ts>0?t/e._ts:((e._dirty?e.totalDuration():e._tDur)-t)/-e._ts)),nc(e),n._dirty||ur(n,e)),e},yg=function(e,t){var n;if((t._time||!t._dur&&t._initted||t._start<e._time&&(t._dur||!t.add))&&(n=Ul(e.rawTime(),t),(!t._dur||Ma(0,t.totalDuration(),n)-t._tTime>_t)&&t.render(n,!0)),ur(e,t)._dp&&e._initted&&e._time>=e._dur&&e._ts){if(e._dur<e.duration())for(n=e;n._dp;)n.rawTime()>=0&&n.totalTime(n._tTime),n=n._dp;e._zTime=-_t}},Ti=function(e,t,n,s){return t.parent&&Os(t),t._start=Dt((ls(n)?n:n||e!==Pt?Zn(e,n,t):e._time)+t._delay),t._end=Dt(t._start+(t.totalDuration()/Math.abs(t.timeScale())||0)),vg(e,t,"_first","_last",e._sort?"_start":0),xf(t)||(e._recent=t),s||yg(e,t),e._ts<0&&ic(e,e._tTime),e},bg=function(e,t){return(Yn.ScrollTrigger||Cd("scrollTrigger",t))&&Yn.ScrollTrigger.create(t,e)},Mg=function(e,t,n,s,r){if(Dd(e,t,r),!e._initted)return 1;if(!n&&e._pt&&!Jt&&(e._dur&&e.vars.lazy!==!1||!e._dur&&e.vars.lazy)&&gg!==Hn.frame)return Ps.push(e),e._lazy=[r,s],1},JE=function i(e){var t=e.parent;return t&&t._ts&&t._initted&&!t._lock&&(t.rawTime()<0||i(t))},xf=function(e){var t=e.data;return t==="isFromStart"||t==="isStart"},e1=function(e,t,n,s){var r=e.ratio,o=t<0||!t&&(!e._start&&JE(e)&&!(!e._initted&&xf(e))||(e._ts<0||e._dp._ts<0)&&!xf(e))?0:1,a=e._rDelay,l=0,c,u,f;if(a&&e._repeat&&(l=Ma(0,e._tDur,t),u=xo(l,a),e._yoyo&&u&1&&(o=1-o),u!==xo(e._tTime,a)&&(r=1-o,e.vars.repeatRefresh&&e._initted&&e.invalidate())),o!==r||Jt||s||e._zTime===_t||!t&&e._zTime){if(!e._initted&&Mg(e,t,s,n,l))return;for(f=e._zTime,e._zTime=t||(n?_t:0),n||(n=t&&!f),e.ratio=o,e._from&&(o=1-o),e._time=0,e._tTime=l,c=e._pt;c;)c.r(o,c.d),c=c._next;t<0&&gf(e,t,n,!0),e._onUpdate&&!n&&Wn(e,"onUpdate"),l&&e._repeat&&!n&&e.parent&&Wn(e,"onRepeat"),(t>=e._tDur||t<0)&&e.ratio===o&&(o&&Os(e,1),!n&&!Jt&&(Wn(e,o?"onComplete":"onReverseComplete",!0),e._prom&&e._prom()))}else e._zTime||(e._zTime=t)},t1=function(e,t,n){var s;if(n>t)for(s=e._first;s&&s._start<=n;){if(s.data==="isPause"&&s._start>t)return s;s=s._next}else for(s=e._last;s&&s._start>=n;){if(s.data==="isPause"&&s._start<t)return s;s=s._prev}},_o=function(e,t,n,s){var r=e._repeat,o=Dt(t)||0,a=e._tTime/e._tDur;return a&&!s&&(e._time*=o/e._dur),e._dur=o,e._tDur=r?r<0?1e10:Dt(o*(r+1)+e._rDelay*r):o,a>0&&!s&&ic(e,e._tTime=e._tDur*a),e.parent&&nc(e),n||ur(e.parent,e),e},tm=function(e){return e instanceof mn?ur(e):_o(e,e._dur)},n1={_start:0,endTime:da,totalDuration:da},Zn=function i(e,t,n){var s=e.labels,r=e._recent||n1,o=e.duration()>=ri?r.endTime(!1):e._dur,a,l,c;return jt(t)&&(isNaN(t)||t in s)?(l=t.charAt(0),c=t.substr(-1)==="%",a=t.indexOf("="),l==="<"||l===">"?(a>=0&&(t=t.replace(/=/,"")),(l==="<"?r._start:r.endTime(r._repeat>=0))+(parseFloat(t.substr(1))||0)*(c?(a<0?r:n).totalDuration()/100:1)):a<0?(t in s||(s[t]=o),s[t]):(l=parseFloat(t.charAt(a-1)+t.substr(a+1)),c&&n&&(l=l/100*(an(n)?n[0]:n).totalDuration()),a>1?i(e,t.substr(0,a-1),n)+l:o+l)):t==null?o:+t},jo=function(e,t,n){var s=ls(t[1]),r=(s?2:1)+(e<2?0:1),o=t[r],a,l;if(s&&(o.duration=t[1]),o.parent=n,e){for(a=o,l=n;l&&!("immediateRender"in a);)a=l.vars.defaults||{},l=wn(l.vars.inherit)&&l.parent;o.immediateRender=wn(a.immediateRender),e<2?o.runBackwards=1:o.startAt=t[r-1]}return new Gt(t[0],o,t[r+1])},Hs=function(e,t){return e||e===0?t(e):t},Ma=function(e,t,n){return n<e?e:n>t?t:n},sn=function(e,t){return!jt(e)||!(t=XE.exec(e))?"":t[1]},i1=function(e,t,n){return Hs(n,function(s){return Ma(e,t,s)})},_f=[].slice,Cg=function(e,t){return e&&Oi(e)&&"length"in e&&(!t&&!e.length||e.length-1 in e&&Oi(e[0]))&&!e.nodeType&&e!==yi},s1=function(e,t,n){return n===void 0&&(n=[]),e.forEach(function(s){var r;return jt(s)&&!t||Cg(s,1)?(r=n).push.apply(r,oi(s)):n.push(s)})||n},oi=function(e,t,n){return Tt&&!t&&Tt.selector?Tt.selector(e):jt(e)&&!n&&(pf||!Ao())?_f.call((t||Md).querySelectorAll(e),0):an(e)?s1(e,n):Cg(e)?_f.call(e,0):e?[e]:[]},Af=function(e){return e=oi(e)[0]||fa("Invalid scope")||{},function(t){var n=e.current||e.nativeElement||e;return oi(t,n.querySelectorAll?n:n===e?fa("Invalid scope")||Md.createElement("div"):e)}},Tg=function(e){return e.sort(function(){return .5-Math.random()})},Eg=function(e){if(Ut(e))return e;var t=Oi(e)?e:{each:e},n=fr(t.ease),s=t.from||0,r=parseFloat(t.base)||0,o={},a=s>0&&s<1,l=isNaN(s)||a,c=t.axis,u=s,f=s;return jt(s)?u=f={center:.5,edges:.5,end:1}[s]||0:!a&&l&&(u=s[0],f=s[1]),function(d,h,g){var p=(g||t).length,x=o[p],m,_,A,S,v,b,M,E,y;if(!x){if(y=t.grid==="auto"?0:(t.grid||[1,ri])[1],!y){for(M=-ri;M<(M=g[y++].getBoundingClientRect().left)&&y<p;);y<p&&y--}for(x=o[p]=[],m=l?Math.min(y,p)*u-.5:s%y,_=y===ri?0:l?p*f/y-.5:s/y|0,M=0,E=ri,b=0;b<p;b++)A=b%y-m,S=_-(b/y|0),x[b]=v=c?Math.abs(c==="y"?S:A):cg(A*A+S*S),v>M&&(M=v),v<E&&(E=v);s==="random"&&Tg(x),x.max=M-E,x.min=E,x.v=p=(parseFloat(t.amount)||parseFloat(t.each)*(y>p?p-1:c?c==="y"?p/y:y:Math.max(y,p/y))||0)*(s==="edges"?-1:1),x.b=p<0?r-p:r,x.u=sn(t.amount||t.each)||0,n=n&&p<0?Og(n):n}return p=(x[d]-x.min)/x.max||0,Dt(x.b+(n?n(p):p)*x.v)+x.u}},Sf=function(e){var t=Math.pow(10,((e+"").split(".")[1]||"").length);return function(n){var s=Dt(Math.round(parseFloat(n)/e)*e*t);return(s-s%1)/t+(ls(n)?0:sn(n))}},wg=function(e,t){var n=an(e),s,r;return!n&&Oi(e)&&(s=n=e.radius||ri,e.values?(e=oi(e.values),(r=!ls(e[0]))&&(s*=s)):e=Sf(e.increment)),Hs(t,n?Ut(e)?function(o){return r=e(o),Math.abs(r-o)<=s?r:o}:function(o){for(var a=parseFloat(r?o.x:o),l=parseFloat(r?o.y:0),c=ri,u=0,f=e.length,d,h;f--;)r?(d=e[f].x-a,h=e[f].y-l,d=d*d+h*h):d=Math.abs(e[f]-a),d<c&&(c=d,u=f);return u=!s||c<=s?e[u]:o,r||u===o||ls(o)?u:u+sn(o)}:Sf(e))},Rg=function(e,t,n,s){return Hs(an(e)?!t:n===!0?!!(n=0):!s,function(){return an(e)?e[~~(Math.random()*e.length)]:(n=n||1e-5)&&(s=n<1?Math.pow(10,(n+"").length-2):1)&&Math.floor(Math.round((e-n/2+Math.random()*(t-e+n*.99))/n)*n*s)/s})},r1=function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(s){return t.reduce(function(r,o){return o(r)},s)}},o1=function(e,t){return function(n){return e(parseFloat(n))+(t||sn(n))}},a1=function(e,t,n){return Dg(e,t,0,1,n)},Ig=function(e,t,n){return Hs(n,function(s){return e[~~t(s)]})},l1=function i(e,t,n){var s=t-e;return an(e)?Ig(e,i(0,e.length),t):Hs(n,function(r){return(s+(r-e)%s)%s+e})},c1=function i(e,t,n){var s=t-e,r=s*2;return an(e)?Ig(e,i(0,e.length-1),t):Hs(n,function(o){return o=(r+(o-e)%r)%r||0,e+(o>s?r-o:o)})},ha=function(e){return e.replace(VE,function(t){var n=t.indexOf("[")+1,s=t.substring(n||7,n?t.indexOf("]"):t.length-1).split(GE);return Rg(n?s:+s[0],n?0:+s[1],+s[2]||1e-5)})},Dg=function(e,t,n,s,r){var o=t-e,a=s-n;return Hs(r,function(l){return n+((l-e)/o*a||0)})},u1=function i(e,t,n,s){var r=isNaN(e+t)?0:function(h){return(1-h)*e+h*t};if(!r){var o=jt(e),a={},l,c,u,f,d;if(n===!0&&(s=1)&&(n=null),o)e={p:e},t={p:t};else if(an(e)&&!an(t)){for(u=[],f=e.length,d=f-2,c=1;c<f;c++)u.push(i(e[c-1],e[c]));f--,r=function(g){g*=f;var p=Math.min(d,~~g);return u[p](g-p)},n=t}else s||(e=go(an(e)?[]:{},e));if(!u){for(l in t)Id.call(a,e,l,"get",t[l]);r=function(g){return Ld(g,a)||(o?e.p:e)}}}return Hs(n,r)},nm=function(e,t,n){var s=e.labels,r=ri,o,a,l;for(o in s)a=s[o]-t,a<0==!!n&&a&&r>(a=Math.abs(a))&&(l=o,r=a);return l},Wn=function(e,t,n){var s=e.vars,r=s[t],o=Tt,a=e._ctx,l,c,u;if(r)return l=s[t+"Params"],c=s.callbackScope||e,n&&Ps.length&&Ll(),a&&(Tt=a),u=l?r.apply(c,l):r.call(c),Tt=o,u},Bo=function(e){return Os(e),e.scrollTrigger&&e.scrollTrigger.kill(!!Jt),e.progress()<1&&Wn(e,"onInterrupt"),e},Wr,Pg=[],Fg=function(e){if(e)if(e=!e.name&&e.default||e,bd()||e.headless){var t=e.name,n=Ut(e),s=t&&!n&&e.init?function(){this._props=[]}:e,r={init:da,render:Ld,add:Id,kill:T1,modifier:C1,rawVars:0},o={targetTest:0,get:0,getSetter:Fd,aliases:{},register:0};if(Ao(),e!==s){if(kn[t])return;Kn(s,Kn(Bl(e,r),o)),go(s.prototype,go(r,Bl(e,o))),kn[s.prop=t]=s,e.targetTest&&(_l.push(s),Td[t]=1),t=(t==="css"?"CSS":t.charAt(0).toUpperCase()+t.substr(1))+"Plugin"}mg(t,s),e.register&&e.register(Pn,s,In)}else Pg.push(e)},xt=255,Uo={aqua:[0,xt,xt],lime:[0,xt,0],silver:[192,192,192],black:[0,0,0],maroon:[128,0,0],teal:[0,128,128],blue:[0,0,xt],navy:[0,0,128],white:[xt,xt,xt],olive:[128,128,0],yellow:[xt,xt,0],orange:[xt,165,0],gray:[128,128,128],purple:[128,0,128],green:[0,128,0],red:[xt,0,0],pink:[xt,192,203],cyan:[0,xt,xt],transparent:[xt,xt,xt,0]},ru=function(e,t,n){return e+=e<0?1:e>1?-1:0,(e*6<1?t+(n-t)*e*6:e<.5?n:e*3<2?t+(n-t)*(2/3-e)*6:t)*xt+.5|0},Lg=function(e,t,n){var s=e?ls(e)?[e>>16,e>>8&xt,e&xt]:0:Uo.black,r,o,a,l,c,u,f,d,h,g;if(!s){if(e.substr(-1)===","&&(e=e.substr(0,e.length-1)),Uo[e])s=Uo[e];else if(e.charAt(0)==="#"){if(e.length<6&&(r=e.charAt(1),o=e.charAt(2),a=e.charAt(3),e="#"+r+r+o+o+a+a+(e.length===5?e.charAt(4)+e.charAt(4):"")),e.length===9)return s=parseInt(e.substr(1,6),16),[s>>16,s>>8&xt,s&xt,parseInt(e.substr(7),16)/255];e=parseInt(e.substr(1),16),s=[e>>16,e>>8&xt,e&xt]}else if(e.substr(0,3)==="hsl"){if(s=g=e.match($p),!t)l=+s[0]%360/360,c=+s[1]/100,u=+s[2]/100,o=u<=.5?u*(c+1):u+c-u*c,r=u*2-o,s.length>3&&(s[3]*=1),s[0]=ru(l+1/3,r,o),s[1]=ru(l,r,o),s[2]=ru(l-1/3,r,o);else if(~e.indexOf("="))return s=e.match(fg),n&&s.length<4&&(s[3]=1),s}else s=e.match($p)||Uo.transparent;s=s.map(Number)}return t&&!g&&(r=s[0]/xt,o=s[1]/xt,a=s[2]/xt,f=Math.max(r,o,a),d=Math.min(r,o,a),u=(f+d)/2,f===d?l=c=0:(h=f-d,c=u>.5?h/(2-f-d):h/(f+d),l=f===r?(o-a)/h+(o<a?6:0):f===o?(a-r)/h+2:(r-o)/h+4,l*=60),s[0]=~~(l+.5),s[1]=~~(c*100+.5),s[2]=~~(u*100+.5)),n&&s.length<4&&(s[3]=1),s},Bg=function(e){var t=[],n=[],s=-1;return e.split(Fs).forEach(function(r){var o=r.match(Gr)||[];t.push.apply(t,o),n.push(s+=o.length+1)}),t.c=n,t},im=function(e,t,n){var s="",r=(e+s).match(Fs),o=t?"hsla(":"rgba(",a=0,l,c,u,f;if(!r)return e;if(r=r.map(function(d){return(d=Lg(d,t,1))&&o+(t?d[0]+","+d[1]+"%,"+d[2]+"%,"+d[3]:d.join(","))+")"}),n&&(u=Bg(e),l=n.c,l.join(s)!==u.c.join(s)))for(c=e.replace(Fs,"1").split(Gr),f=c.length-1;a<f;a++)s+=c[a]+(~l.indexOf(a)?r.shift()||o+"0,0,0,0)":(u.length?u:r.length?r:n).shift());if(!c)for(c=e.split(Fs),f=c.length-1;a<f;a++)s+=c[a]+r[a];return s+c[f]},Fs=(function(){var i="(?:\\b(?:(?:rgb|rgba|hsl|hsla)\\(.+?\\))|\\B#(?:[0-9a-f]{3,4}){1,2}\\b",e;for(e in Uo)i+="|"+e+"\\b";return new RegExp(i+")","gi")})(),f1=/hsl[a]?\(/,Ug=function(e){var t=e.join(" "),n;if(Fs.lastIndex=0,Fs.test(t))return n=f1.test(t),e[1]=im(e[1],n),e[0]=im(e[0],n,Bg(e[1])),!0},pa,Hn=(function(){var i=Date.now,e=500,t=33,n=i(),s=n,r=1e3/240,o=r,a=[],l,c,u,f,d,h,g=function p(x){var m=i()-s,_=x===!0,A,S,v,b;if((m>e||m<0)&&(n+=m-t),s+=m,v=s-n,A=v-o,(A>0||_)&&(b=++f.frame,d=v-f.time*1e3,f.time=v=v/1e3,o+=A+(A>=r?4:r-A),S=1),_||(l=c(p)),S)for(h=0;h<a.length;h++)a[h](v,d,b,x)};return f={time:0,frame:0,tick:function(){g(!0)},deltaRatio:function(x){return d/(1e3/(x||60))},wake:function(){hg&&(!pf&&bd()&&(yi=pf=window,Md=yi.document||{},Yn.gsap=Pn,(yi.gsapVersions||(yi.gsapVersions=[])).push(Pn.version),pg(Fl||yi.GreenSockGlobals||!yi.gsap&&yi||{}),Pg.forEach(Fg)),u=typeof requestAnimationFrame<"u"&&requestAnimationFrame,l&&f.sleep(),c=u||function(x){return setTimeout(x,o-f.time*1e3+1|0)},pa=1,g(2))},sleep:function(){(u?cancelAnimationFrame:clearTimeout)(l),pa=0,c=da},lagSmoothing:function(x,m){e=x||1/0,t=Math.min(m||33,e)},fps:function(x){r=1e3/(x||240),o=f.time*1e3+r},add:function(x,m,_){var A=m?function(S,v,b,M){x(S,v,b,M),f.remove(A)}:x;return f.remove(x),a[_?"unshift":"push"](A),Ao(),A},remove:function(x,m){~(m=a.indexOf(x))&&a.splice(m,1)&&h>=m&&h--},_listeners:a},f})(),Ao=function(){return!pa&&Hn.wake()},tt={},d1=/^[\d.\-M][\d.\-,\s]/,h1=/["']/g,p1=function(e){for(var t={},n=e.substr(1,e.length-3).split(":"),s=n[0],r=1,o=n.length,a,l,c;r<o;r++)l=n[r],a=r!==o-1?l.lastIndexOf(","):l.length,c=l.substr(0,a),t[s]=isNaN(c)?c.replace(h1,"").trim():+c,s=l.substr(a+1).trim();return t},m1=function(e){var t=e.indexOf("(")+1,n=e.indexOf(")"),s=e.indexOf("(",t);return e.substring(t,~s&&s<n?e.indexOf(")",n+1):n)},g1=function(e){var t=(e+"").split("("),n=tt[t[0]];return n&&t.length>1&&n.config?n.config.apply(null,~e.indexOf("{")?[p1(t[1])]:m1(e).split(",").map(Ag)):tt._CE&&d1.test(e)?tt._CE("",e):n},Og=function(e){return function(t){return 1-e(1-t)}},Ng=function i(e,t){for(var n=e._first,s;n;)n instanceof mn?i(n,t):n.vars.yoyoEase&&(!n._yoyo||!n._repeat)&&n._yoyo!==t&&(n.timeline?i(n.timeline,t):(s=n._ease,n._ease=n._yEase,n._yEase=s,n._yoyo=t)),n=n._next},fr=function(e,t){return e&&(Ut(e)?e:tt[e]||g1(e))||t},gr=function(e,t,n,s){n===void 0&&(n=function(l){return 1-t(1-l)}),s===void 0&&(s=function(l){return l<.5?t(l*2)/2:1-t((1-l)*2)/2});var r={easeIn:t,easeOut:n,easeInOut:s},o;return Rn(e,function(a){tt[a]=Yn[a]=r,tt[o=a.toLowerCase()]=n;for(var l in r)tt[o+(l==="easeIn"?".in":l==="easeOut"?".out":".inOut")]=tt[a+"."+l]=r[l]}),r},zg=function(e){return function(t){return t<.5?(1-e(1-t*2))/2:.5+e((t-.5)*2)/2}},ou=function i(e,t,n){var s=t>=1?t:1,r=(n||(e?.3:.45))/(t<1?t:1),o=r/hf*(Math.asin(1/s)||0),a=function(u){return u===1?1:s*Math.pow(2,-10*u)*HE((u-o)*r)+1},l=e==="out"?a:e==="in"?function(c){return 1-a(1-c)}:zg(a);return r=hf/r,l.config=function(c,u){return i(e,c,u)},l},au=function i(e,t){t===void 0&&(t=1.70158);var n=function(o){return o?--o*o*((t+1)*o+t)+1:0},s=e==="out"?n:e==="in"?function(r){return 1-n(1-r)}:zg(n);return s.config=function(r){return i(e,r)},s};Rn("Linear,Quad,Cubic,Quart,Quint,Strong",function(i,e){var t=e<5?e+1:e;gr(i+",Power"+(t-1),e?function(n){return Math.pow(n,t)}:function(n){return n},function(n){return 1-Math.pow(1-n,t)},function(n){return n<.5?Math.pow(n*2,t)/2:1-Math.pow((1-n)*2,t)/2})});tt.Linear.easeNone=tt.none=tt.Linear.easeIn;gr("Elastic",ou("in"),ou("out"),ou());(function(i,e){var t=1/e,n=2*t,s=2.5*t,r=function(a){return a<t?i*a*a:a<n?i*Math.pow(a-1.5/e,2)+.75:a<s?i*(a-=2.25/e)*a+.9375:i*Math.pow(a-2.625/e,2)+.984375};gr("Bounce",function(o){return 1-r(1-o)},r)})(7.5625,2.75);gr("Expo",function(i){return Math.pow(2,10*(i-1))*i+i*i*i*i*i*i*(1-i)});gr("Circ",function(i){return-(cg(1-i*i)-1)});gr("Sine",function(i){return i===1?1:-kE(i*NE)+1});gr("Back",au("in"),au("out"),au());tt.SteppedEase=tt.steps=Yn.SteppedEase={config:function(e,t){e===void 0&&(e=1);var n=1/e,s=e+(t?0:1),r=t?1:0,o=1-_t;return function(a){return((s*Ma(0,o,a)|0)+r)*n}}};mo.ease=tt["quad.out"];Rn("onComplete,onUpdate,onStart,onRepeat,onReverseComplete,onInterrupt",function(i){return Ed+=i+","+i+"Params,"});var kg=function(e,t){this.id=zE++,e._gsap=this,this.target=e,this.harness=t,this.get=t?t.get:xg,this.set=t?t.getSetter:Fd},ma=(function(){function i(t){this.vars=t,this._delay=+t.delay||0,(this._repeat=t.repeat===1/0?-2:t.repeat||0)&&(this._rDelay=t.repeatDelay||0,this._yoyo=!!t.yoyo||!!t.yoyoEase),this._ts=1,_o(this,+t.duration,1,1),this.data=t.data,Tt&&(this._ctx=Tt,Tt.data.push(this)),pa||Hn.wake()}var e=i.prototype;return e.delay=function(n){return n||n===0?(this.parent&&this.parent.smoothChildTiming&&this.startTime(this._start+n-this._delay),this._delay=n,this):this._delay},e.duration=function(n){return arguments.length?this.totalDuration(this._repeat>0?n+(n+this._rDelay)*this._repeat:n):this.totalDuration()&&this._dur},e.totalDuration=function(n){return arguments.length?(this._dirty=0,_o(this,this._repeat<0?n:(n-this._repeat*this._rDelay)/(this._repeat+1))):this._tDur},e.totalTime=function(n,s){if(Ao(),!arguments.length)return this._tTime;var r=this._dp;if(r&&r.smoothChildTiming&&this._ts){for(ic(this,n),!r._dp||r.parent||yg(r,this);r&&r.parent;)r.parent._time!==r._start+(r._ts>=0?r._tTime/r._ts:(r.totalDuration()-r._tTime)/-r._ts)&&r.totalTime(r._tTime,!0),r=r.parent;!this.parent&&this._dp.autoRemoveChildren&&(this._ts>0&&n<this._tDur||this._ts<0&&n>0||!this._tDur&&!n)&&Ti(this._dp,this,this._start-this._delay)}return(this._tTime!==n||!this._dur&&!s||this._initted&&Math.abs(this._zTime)===_t||!this._initted&&this._dur&&n||!n&&!this._initted&&(this.add||this._ptLookup))&&(this._ts||(this._pTime=n),_g(this,n,s)),this},e.time=function(n,s){return arguments.length?this.totalTime(Math.min(this.totalDuration(),n+em(this))%(this._dur+this._rDelay)||(n?this._dur:0),s):this._time},e.totalProgress=function(n,s){return arguments.length?this.totalTime(this.totalDuration()*n,s):this.totalDuration()?Math.min(1,this._tTime/this._tDur):this.rawTime()>=0&&this._initted?1:0},e.progress=function(n,s){return arguments.length?this.totalTime(this.duration()*(this._yoyo&&!(this.iteration()&1)?1-n:n)+em(this),s):this.duration()?Math.min(1,this._time/this._dur):this.rawTime()>0?1:0},e.iteration=function(n,s){var r=this.duration()+this._rDelay;return arguments.length?this.totalTime(this._time+(n-1)*r,s):this._repeat?xo(this._tTime,r)+1:1},e.timeScale=function(n,s){if(!arguments.length)return this._rts===-_t?0:this._rts;if(this._rts===n)return this;var r=this.parent&&this._ts?Ul(this.parent._time,this):this._tTime;return this._rts=+n||0,this._ts=this._ps||n===-_t?0:this._rts,this.totalTime(Ma(-Math.abs(this._delay),this.totalDuration(),r),s!==!1),nc(this),$E(this)},e.paused=function(n){return arguments.length?(this._ps!==n&&(this._ps=n,n?(this._pTime=this._tTime||Math.max(-this._delay,this.rawTime()),this._ts=this._act=0):(Ao(),this._ts=this._rts,this.totalTime(this.parent&&!this.parent.smoothChildTiming?this.rawTime():this._tTime||this._pTime,this.progress()===1&&Math.abs(this._zTime)!==_t&&(this._tTime-=_t)))),this):this._ps},e.startTime=function(n){if(arguments.length){this._start=Dt(n);var s=this.parent||this._dp;return s&&(s._sort||!this.parent)&&Ti(s,this,this._start-this._delay),this}return this._start},e.endTime=function(n){return this._start+(wn(n)?this.totalDuration():this.duration())/Math.abs(this._ts||1)},e.rawTime=function(n){var s=this.parent||this._dp;return s?n&&(!this._ts||this._repeat&&this._time&&this.totalProgress()<1)?this._tTime%(this._dur+this._rDelay):this._ts?Ul(s.rawTime(n),this):this._tTime:this._tTime},e.revert=function(n){n===void 0&&(n=QE);var s=Jt;return Jt=n,Rd(this)&&(this.timeline&&this.timeline.revert(n),this.totalTime(-.01,n.suppressEvents)),this.data!=="nested"&&n.kill!==!1&&this.kill(),Jt=s,this},e.globalTime=function(n){for(var s=this,r=arguments.length?n:s.rawTime();s;)r=s._start+r/(Math.abs(s._ts)||1),s=s._dp;return!this.parent&&this._sat?this._sat.globalTime(n):r},e.repeat=function(n){return arguments.length?(this._repeat=n===1/0?-2:n,tm(this)):this._repeat===-2?1/0:this._repeat},e.repeatDelay=function(n){if(arguments.length){var s=this._time;return this._rDelay=n,tm(this),s?this.time(s):this}return this._rDelay},e.yoyo=function(n){return arguments.length?(this._yoyo=n,this):this._yoyo},e.seek=function(n,s){return this.totalTime(Zn(this,n),wn(s))},e.restart=function(n,s){return this.play().totalTime(n?-this._delay:0,wn(s)),this._dur||(this._zTime=-_t),this},e.play=function(n,s){return n!=null&&this.seek(n,s),this.reversed(!1).paused(!1)},e.reverse=function(n,s){return n!=null&&this.seek(n||this.totalDuration(),s),this.reversed(!0).paused(!1)},e.pause=function(n,s){return n!=null&&this.seek(n,s),this.paused(!0)},e.resume=function(){return this.paused(!1)},e.reversed=function(n){return arguments.length?(!!n!==this.reversed()&&this.timeScale(-this._rts||(n?-_t:0)),this):this._rts<0},e.invalidate=function(){return this._initted=this._act=0,this._zTime=-_t,this},e.isActive=function(){var n=this.parent||this._dp,s=this._start,r;return!!(!n||this._ts&&this._initted&&n.isActive()&&(r=n.rawTime(!0))>=s&&r<this.endTime(!0)-_t)},e.eventCallback=function(n,s,r){var o=this.vars;return arguments.length>1?(s?(o[n]=s,r&&(o[n+"Params"]=r),n==="onUpdate"&&(this._onUpdate=s)):delete o[n],this):o[n]},e.then=function(n){var s=this,r=s._prom;return new Promise(function(o){var a=Ut(n)?n:Sg,l=function(){var u=s.then;s.then=null,r&&r(),Ut(a)&&(a=a(s))&&(a.then||a===s)&&(s.then=u),o(a),s.then=u};s._initted&&s.totalProgress()===1&&s._ts>=0||!s._tTime&&s._ts<0?l():s._prom=l})},e.kill=function(){Bo(this)},i})();Kn(ma.prototype,{_time:0,_start:0,_end:0,_tTime:0,_tDur:0,_dirty:0,_repeat:0,_yoyo:!1,parent:null,_initted:!1,_rDelay:0,_ts:1,_dp:0,ratio:0,_zTime:-_t,_prom:0,_ps:!1,_rts:1});var mn=(function(i){lg(e,i);function e(n,s){var r;return n===void 0&&(n={}),r=i.call(this,n)||this,r.labels={},r.smoothChildTiming=!!n.smoothChildTiming,r.autoRemoveChildren=!!n.autoRemoveChildren,r._sort=wn(n.sortChildren),Pt&&Ti(n.parent||Pt,Ki(r),s),n.reversed&&r.reverse(),n.paused&&r.paused(!0),n.scrollTrigger&&bg(Ki(r),n.scrollTrigger),r}var t=e.prototype;return t.to=function(s,r,o){return jo(0,arguments,this),this},t.from=function(s,r,o){return jo(1,arguments,this),this},t.fromTo=function(s,r,o,a){return jo(2,arguments,this),this},t.set=function(s,r,o){return r.duration=0,r.parent=this,Ko(r).repeatDelay||(r.repeat=0),r.immediateRender=!!r.immediateRender,new Gt(s,r,Zn(this,o),1),this},t.call=function(s,r,o){return Ti(this,Gt.delayedCall(0,s,r),o)},t.staggerTo=function(s,r,o,a,l,c,u){return o.duration=r,o.stagger=o.stagger||a,o.onComplete=c,o.onCompleteParams=u,o.parent=this,new Gt(s,o,Zn(this,l)),this},t.staggerFrom=function(s,r,o,a,l,c,u){return o.runBackwards=1,Ko(o).immediateRender=wn(o.immediateRender),this.staggerTo(s,r,o,a,l,c,u)},t.staggerFromTo=function(s,r,o,a,l,c,u,f){return a.startAt=o,Ko(a).immediateRender=wn(a.immediateRender),this.staggerTo(s,r,a,l,c,u,f)},t.render=function(s,r,o){var a=this._time,l=this._dirty?this.totalDuration():this._tDur,c=this._dur,u=s<=0?0:Dt(s),f=this._zTime<0!=s<0&&(this._initted||!c),d,h,g,p,x,m,_,A,S,v,b,M;if(this!==Pt&&u>l&&s>=0&&(u=l),u!==this._tTime||o||f){if(a!==this._time&&c&&(u+=this._time-a,s+=this._time-a),d=u,S=this._start,A=this._ts,m=!A,f&&(c||(a=this._zTime),(s||!r)&&(this._zTime=s)),this._repeat){if(b=this._yoyo,x=c+this._rDelay,this._repeat<-1&&s<0)return this.totalTime(x*100+s,r,o);if(d=Dt(u%x),u===l?(p=this._repeat,d=c):(v=Dt(u/x),p=~~v,p&&p===v&&(d=c,p--),d>c&&(d=c)),v=xo(this._tTime,x),!a&&this._tTime&&v!==p&&this._tTime-v*x-this._dur<=0&&(v=p),b&&p&1&&(d=c-d,M=1),p!==v&&!this._lock){var E=b&&v&1,y=E===(b&&p&1);if(p<v&&(E=!E),a=E?0:u%c?c:u,this._lock=1,this.render(a||(M?0:Dt(p*x)),r,!c)._lock=0,this._tTime=u,!r&&this.parent&&Wn(this,"onRepeat"),this.vars.repeatRefresh&&!M&&(this.invalidate()._lock=1,v=p),a&&a!==this._time||m!==!this._ts||this.vars.onRepeat&&!this.parent&&!this._act)return this;if(c=this._dur,l=this._tDur,y&&(this._lock=2,a=E?c:-1e-4,this.render(a,!0),this.vars.repeatRefresh&&!M&&this.invalidate()),this._lock=0,!this._ts&&!m)return this;Ng(this,M)}}if(this._hasPause&&!this._forcing&&this._lock<2&&(_=t1(this,Dt(a),Dt(d)),_&&(u-=d-(d=_._start))),this._tTime=u,this._time=d,this._act=!A,this._initted||(this._onUpdate=this.vars.onUpdate,this._initted=1,this._zTime=s,a=0),!a&&u&&c&&!r&&!v&&(Wn(this,"onStart"),this._tTime!==u))return this;if(d>=a&&s>=0)for(h=this._first;h;){if(g=h._next,(h._act||d>=h._start)&&h._ts&&_!==h){if(h.parent!==this)return this.render(s,r,o);if(h.render(h._ts>0?(d-h._start)*h._ts:(h._dirty?h.totalDuration():h._tDur)+(d-h._start)*h._ts,r,o),d!==this._time||!this._ts&&!m){_=0,g&&(u+=this._zTime=-_t);break}}h=g}else{h=this._last;for(var C=s<0?s:d;h;){if(g=h._prev,(h._act||C<=h._end)&&h._ts&&_!==h){if(h.parent!==this)return this.render(s,r,o);if(h.render(h._ts>0?(C-h._start)*h._ts:(h._dirty?h.totalDuration():h._tDur)+(C-h._start)*h._ts,r,o||Jt&&Rd(h)),d!==this._time||!this._ts&&!m){_=0,g&&(u+=this._zTime=C?-_t:_t);break}}h=g}}if(_&&!r&&(this.pause(),_.render(d>=a?0:-_t)._zTime=d>=a?1:-1,this._ts))return this._start=S,nc(this),this.render(s,r,o);this._onUpdate&&!r&&Wn(this,"onUpdate",!0),(u===l&&this._tTime>=this.totalDuration()||!u&&a)&&(S===this._start||Math.abs(A)!==Math.abs(this._ts))&&(this._lock||((s||!c)&&(u===l&&this._ts>0||!u&&this._ts<0)&&Os(this,1),!r&&!(s<0&&!a)&&(u||a||!l)&&(Wn(this,u===l&&s>=0?"onComplete":"onReverseComplete",!0),this._prom&&!(u<l&&this.timeScale()>0)&&this._prom())))}return this},t.add=function(s,r){var o=this;if(ls(r)||(r=Zn(this,r,s)),!(s instanceof ma)){if(an(s))return s.forEach(function(a){return o.add(a,r)}),this;if(jt(s))return this.addLabel(s,r);if(Ut(s))s=Gt.delayedCall(0,s);else return this}return this!==s?Ti(this,s,r):this},t.getChildren=function(s,r,o,a){s===void 0&&(s=!0),r===void 0&&(r=!0),o===void 0&&(o=!0),a===void 0&&(a=-ri);for(var l=[],c=this._first;c;)c._start>=a&&(c instanceof Gt?r&&l.push(c):(o&&l.push(c),s&&l.push.apply(l,c.getChildren(!0,r,o)))),c=c._next;return l},t.getById=function(s){for(var r=this.getChildren(1,1,1),o=r.length;o--;)if(r[o].vars.id===s)return r[o]},t.remove=function(s){return jt(s)?this.removeLabel(s):Ut(s)?this.killTweensOf(s):(s.parent===this&&tc(this,s),s===this._recent&&(this._recent=this._last),ur(this))},t.totalTime=function(s,r){return arguments.length?(this._forcing=1,!this._dp&&this._ts&&(this._start=Dt(Hn.time-(this._ts>0?s/this._ts:(this.totalDuration()-s)/-this._ts))),i.prototype.totalTime.call(this,s,r),this._forcing=0,this):this._tTime},t.addLabel=function(s,r){return this.labels[s]=Zn(this,r),this},t.removeLabel=function(s){return delete this.labels[s],this},t.addPause=function(s,r,o){var a=Gt.delayedCall(0,r||da,o);return a.data="isPause",this._hasPause=1,Ti(this,a,Zn(this,s))},t.removePause=function(s){var r=this._first;for(s=Zn(this,s);r;)r._start===s&&r.data==="isPause"&&Os(r),r=r._next},t.killTweensOf=function(s,r,o){for(var a=this.getTweensOf(s,o),l=a.length;l--;)Ts!==a[l]&&a[l].kill(s,r);return this},t.getTweensOf=function(s,r){for(var o=[],a=oi(s),l=this._first,c=ls(r),u;l;)l instanceof Gt?YE(l._targets,a)&&(c?(!Ts||l._initted&&l._ts)&&l.globalTime(0)<=r&&l.globalTime(l.totalDuration())>r:!r||l.isActive())&&o.push(l):(u=l.getTweensOf(a,r)).length&&o.push.apply(o,u),l=l._next;return o},t.tweenTo=function(s,r){r=r||{};var o=this,a=Zn(o,s),l=r,c=l.startAt,u=l.onStart,f=l.onStartParams,d=l.immediateRender,h,g=Gt.to(o,Kn({ease:r.ease||"none",lazy:!1,immediateRender:!1,time:a,overwrite:"auto",duration:r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale())||_t,onStart:function(){if(o.pause(),!h){var x=r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale());g._dur!==x&&_o(g,x,0,1).render(g._time,!0,!0),h=1}u&&u.apply(g,f||[])}},r));return d?g.render(0):g},t.tweenFromTo=function(s,r,o){return this.tweenTo(r,Kn({startAt:{time:Zn(this,s)}},o))},t.recent=function(){return this._recent},t.nextLabel=function(s){return s===void 0&&(s=this._time),nm(this,Zn(this,s))},t.previousLabel=function(s){return s===void 0&&(s=this._time),nm(this,Zn(this,s),1)},t.currentLabel=function(s){return arguments.length?this.seek(s,!0):this.previousLabel(this._time+_t)},t.shiftChildren=function(s,r,o){o===void 0&&(o=0);var a=this._first,l=this.labels,c;for(s=Dt(s);a;)a._start>=o&&(a._start+=s,a._end+=s),a=a._next;if(r)for(c in l)l[c]>=o&&(l[c]+=s);return ur(this)},t.invalidate=function(s){var r=this._first;for(this._lock=0;r;)r.invalidate(s),r=r._next;return i.prototype.invalidate.call(this,s)},t.clear=function(s){s===void 0&&(s=!0);for(var r=this._first,o;r;)o=r._next,this.remove(r),r=o;return this._dp&&(this._time=this._tTime=this._pTime=0),s&&(this.labels={}),ur(this)},t.totalDuration=function(s){var r=0,o=this,a=o._last,l=ri,c,u,f;if(arguments.length)return o.timeScale((o._repeat<0?o.duration():o.totalDuration())/(o.reversed()?-s:s));if(o._dirty){for(f=o.parent;a;)c=a._prev,a._dirty&&a.totalDuration(),u=a._start,u>l&&o._sort&&a._ts&&!o._lock?(o._lock=1,Ti(o,a,u-a._delay,1)._lock=0):l=u,u<0&&a._ts&&(r-=u,(!f&&!o._dp||f&&f.smoothChildTiming)&&(o._start+=Dt(u/o._ts),o._time-=u,o._tTime-=u),o.shiftChildren(-u,!1,-1/0),l=0),a._end>r&&a._ts&&(r=a._end),a=c;_o(o,o===Pt&&o._time>r?o._time:r,1,1),o._dirty=0}return o._tDur},e.updateRoot=function(s){if(Pt._ts&&(_g(Pt,Ul(s,Pt)),gg=Hn.frame),Hn.frame>=Zp){Zp+=Qn.autoSleep||120;var r=Pt._first;if((!r||!r._ts)&&Qn.autoSleep&&Hn._listeners.length<2){for(;r&&!r._ts;)r=r._next;r||Hn.sleep()}}},e})(ma);Kn(mn.prototype,{_lock:0,_hasPause:0,_forcing:0});var x1=function(e,t,n,s,r,o,a){var l=new In(this._pt,e,t,0,1,qg,null,r),c=0,u=0,f,d,h,g,p,x,m,_;for(l.b=n,l.e=s,n+="",s+="",(m=~s.indexOf("random("))&&(s=ha(s)),o&&(_=[n,s],o(_,e,t),n=_[0],s=_[1]),d=n.match(iu)||[];f=iu.exec(s);)g=f[0],p=s.substring(c,f.index),h?h=(h+1)%5:p.substr(-5)==="rgba("&&(h=1),g!==d[u++]&&(x=parseFloat(d[u-1])||0,l._pt={_next:l._pt,p:p||u===1?p:",",s:x,c:g.charAt(1)==="="?to(x,g)-x:parseFloat(g)-x,m:h&&h<4?Math.round:0},c=iu.lastIndex);return l.c=c<s.length?s.substring(c,s.length):"",l.fp=a,(dg.test(s)||m)&&(l.e=0),this._pt=l,l},Id=function(e,t,n,s,r,o,a,l,c,u){Ut(s)&&(s=s(r||0,e,o));var f=e[t],d=n!=="get"?n:Ut(f)?c?e[t.indexOf("set")||!Ut(e["get"+t.substr(3)])?t:"get"+t.substr(3)](c):e[t]():f,h=Ut(f)?c?y1:Wg:Pd,g;if(jt(s)&&(~s.indexOf("random(")&&(s=ha(s)),s.charAt(1)==="="&&(g=to(d,s)+(sn(d)||0),(g||g===0)&&(s=g))),!u||d!==s||vf)return!isNaN(d*s)&&s!==""?(g=new In(this._pt,e,t,+d||0,s-(d||0),typeof f=="boolean"?M1:Xg,0,h),c&&(g.fp=c),a&&g.modifier(a,this,e),this._pt=g):(!f&&!(t in e)&&Cd(t,s),x1.call(this,e,t,d,s,h,l||Qn.stringFilter,c))},_1=function(e,t,n,s,r){if(Ut(e)&&(e=$o(e,r,t,n,s)),!Oi(e)||e.style&&e.nodeType||an(e)||ug(e))return jt(e)?$o(e,r,t,n,s):e;var o={},a;for(a in e)o[a]=$o(e[a],r,t,n,s);return o},Hg=function(e,t,n,s,r,o){var a,l,c,u;if(kn[e]&&(a=new kn[e]).init(r,a.rawVars?t[e]:_1(t[e],s,r,o,n),n,s,o)!==!1&&(n._pt=l=new In(n._pt,r,e,0,1,a.render,a,0,a.priority),n!==Wr))for(c=n._ptLookup[n._targets.indexOf(r)],u=a._props.length;u--;)c[a._props[u]]=l;return a},Ts,vf,Dd=function i(e,t,n){var s=e.vars,r=s.ease,o=s.startAt,a=s.immediateRender,l=s.lazy,c=s.onUpdate,u=s.runBackwards,f=s.yoyoEase,d=s.keyframes,h=s.autoRevert,g=e._dur,p=e._startAt,x=e._targets,m=e.parent,_=m&&m.data==="nested"?m.vars.targets:x,A=e._overwrite==="auto"&&!vd,S=e.timeline,v,b,M,E,y,C,I,P,U,O,k,z,W;if(S&&(!d||!r)&&(r="none"),e._ease=fr(r,mo.ease),e._yEase=f?Og(fr(f===!0?r:f,mo.ease)):0,f&&e._yoyo&&!e._repeat&&(f=e._yEase,e._yEase=e._ease,e._ease=f),e._from=!S&&!!s.runBackwards,!S||d&&!s.stagger){if(P=x[0]?cr(x[0]).harness:0,z=P&&s[P.prop],v=Bl(s,Td),p&&(p._zTime<0&&p.progress(1),t<0&&u&&a&&!h?p.render(-1,!0):p.revert(u&&g?xl:qE),p._lazy=0),o){if(Os(e._startAt=Gt.set(x,Kn({data:"isStart",overwrite:!1,parent:m,immediateRender:!0,lazy:!p&&wn(l),startAt:null,delay:0,onUpdate:c&&function(){return Wn(e,"onUpdate")},stagger:0},o))),e._startAt._dp=0,e._startAt._sat=e,t<0&&(Jt||!a&&!h)&&e._startAt.revert(xl),a&&g&&t<=0&&n<=0){t&&(e._zTime=t);return}}else if(u&&g&&!p){if(t&&(a=!1),M=Kn({overwrite:!1,data:"isFromStart",lazy:a&&!p&&wn(l),immediateRender:a,stagger:0,parent:m},v),z&&(M[P.prop]=z),Os(e._startAt=Gt.set(x,M)),e._startAt._dp=0,e._startAt._sat=e,t<0&&(Jt?e._startAt.revert(xl):e._startAt.render(-1,!0)),e._zTime=t,!a)i(e._startAt,_t,_t);else if(!t)return}for(e._pt=e._ptCache=0,l=g&&wn(l)||l&&!g,b=0;b<x.length;b++){if(y=x[b],I=y._gsap||wd(x)[b]._gsap,e._ptLookup[b]=O={},mf[I.id]&&Ps.length&&Ll(),k=_===x?b:_.indexOf(y),P&&(U=new P).init(y,z||v,e,k,_)!==!1&&(e._pt=E=new In(e._pt,y,U.name,0,1,U.render,U,0,U.priority),U._props.forEach(function(H){O[H]=E}),U.priority&&(C=1)),!P||z)for(M in v)kn[M]&&(U=Hg(M,v,e,k,y,_))?U.priority&&(C=1):O[M]=E=Id.call(e,y,M,"get",v[M],k,_,0,s.stringFilter);e._op&&e._op[b]&&e.kill(y,e._op[b]),A&&e._pt&&(Ts=e,Pt.killTweensOf(y,O,e.globalTime(t)),W=!e.parent,Ts=0),e._pt&&l&&(mf[I.id]=1)}C&&Qg(e),e._onInit&&e._onInit(e)}e._onUpdate=c,e._initted=(!e._op||e._pt)&&!W,d&&t<=0&&S.render(ri,!0,!0)},A1=function(e,t,n,s,r,o,a,l){var c=(e._pt&&e._ptCache||(e._ptCache={}))[t],u,f,d,h;if(!c)for(c=e._ptCache[t]=[],d=e._ptLookup,h=e._targets.length;h--;){if(u=d[h][t],u&&u.d&&u.d._pt)for(u=u.d._pt;u&&u.p!==t&&u.fp!==t;)u=u._next;if(!u)return vf=1,e.vars[t]="+=0",Dd(e,a),vf=0,l?fa(t+" not eligible for reset"):1;c.push(u)}for(h=c.length;h--;)f=c[h],u=f._pt||f,u.s=(s||s===0)&&!r?s:u.s+(s||0)+o*u.c,u.c=n-u.s,f.e&&(f.e=Ot(n)+sn(f.e)),f.b&&(f.b=u.s+sn(f.b))},S1=function(e,t){var n=e[0]?cr(e[0]).harness:0,s=n&&n.aliases,r,o,a,l;if(!s)return t;r=go({},t);for(o in s)if(o in r)for(l=s[o].split(","),a=l.length;a--;)r[l[a]]=r[o];return r},v1=function(e,t,n,s){var r=t.ease||s||"power1.inOut",o,a;if(an(t))a=n[e]||(n[e]=[]),t.forEach(function(l,c){return a.push({t:c/(t.length-1)*100,v:l,e:r})});else for(o in t)a=n[o]||(n[o]=[]),o==="ease"||a.push({t:parseFloat(e),v:t[o],e:r})},$o=function(e,t,n,s,r){return Ut(e)?e.call(t,n,s,r):jt(e)&&~e.indexOf("random(")?ha(e):e},Vg=Ed+"repeat,repeatDelay,yoyo,repeatRefresh,yoyoEase,autoRevert",Gg={};Rn(Vg+",id,stagger,delay,duration,paused,scrollTrigger",function(i){return Gg[i]=1});var Gt=(function(i){lg(e,i);function e(n,s,r,o){var a;typeof s=="number"&&(r.duration=s,s=r,r=null),a=i.call(this,o?s:Ko(s))||this;var l=a.vars,c=l.duration,u=l.delay,f=l.immediateRender,d=l.stagger,h=l.overwrite,g=l.keyframes,p=l.defaults,x=l.scrollTrigger,m=l.yoyoEase,_=s.parent||Pt,A=(an(n)||ug(n)?ls(n[0]):"length"in s)?[n]:oi(n),S,v,b,M,E,y,C,I;if(a._targets=A.length?wd(A):fa("GSAP target "+n+" not found. https://gsap.com",!Qn.nullTargetWarn)||[],a._ptLookup=[],a._overwrite=h,g||d||ol(c)||ol(u)){if(s=a.vars,S=a.timeline=new mn({data:"nested",defaults:p||{},targets:_&&_.data==="nested"?_.vars.targets:A}),S.kill(),S.parent=S._dp=Ki(a),S._start=0,d||ol(c)||ol(u)){if(M=A.length,C=d&&Eg(d),Oi(d))for(E in d)~Vg.indexOf(E)&&(I||(I={}),I[E]=d[E]);for(v=0;v<M;v++)b=Bl(s,Gg),b.stagger=0,m&&(b.yoyoEase=m),I&&go(b,I),y=A[v],b.duration=+$o(c,Ki(a),v,y,A),b.delay=(+$o(u,Ki(a),v,y,A)||0)-a._delay,!d&&M===1&&b.delay&&(a._delay=u=b.delay,a._start+=u,b.delay=0),S.to(y,b,C?C(v,y,A):0),S._ease=tt.none;S.duration()?c=u=0:a.timeline=0}else if(g){Ko(Kn(S.vars.defaults,{ease:"none"})),S._ease=fr(g.ease||s.ease||"none");var P=0,U,O,k;if(an(g))g.forEach(function(z){return S.to(A,z,">")}),S.duration();else{b={};for(E in g)E==="ease"||E==="easeEach"||v1(E,g[E],b,g.easeEach);for(E in b)for(U=b[E].sort(function(z,W){return z.t-W.t}),P=0,v=0;v<U.length;v++)O=U[v],k={ease:O.e,duration:(O.t-(v?U[v-1].t:0))/100*c},k[E]=O.v,S.to(A,k,P),P+=k.duration;S.duration()<c&&S.to({},{duration:c-S.duration()})}}c||a.duration(c=S.duration())}else a.timeline=0;return h===!0&&!vd&&(Ts=Ki(a),Pt.killTweensOf(A),Ts=0),Ti(_,Ki(a),r),s.reversed&&a.reverse(),s.paused&&a.paused(!0),(f||!c&&!g&&a._start===Dt(_._time)&&wn(f)&&ZE(Ki(a))&&_.data!=="nested")&&(a._tTime=-_t,a.render(Math.max(0,-u)||0)),x&&bg(Ki(a),x),a}var t=e.prototype;return t.render=function(s,r,o){var a=this._time,l=this._tDur,c=this._dur,u=s<0,f=s>l-_t&&!u?l:s<_t?0:s,d,h,g,p,x,m,_,A,S;if(!c)e1(this,s,r,o);else if(f!==this._tTime||!s||o||!this._initted&&this._tTime||this._startAt&&this._zTime<0!==u||this._lazy){if(d=f,A=this.timeline,this._repeat){if(p=c+this._rDelay,this._repeat<-1&&u)return this.totalTime(p*100+s,r,o);if(d=Dt(f%p),f===l?(g=this._repeat,d=c):(x=Dt(f/p),g=~~x,g&&g===x?(d=c,g--):d>c&&(d=c)),m=this._yoyo&&g&1,m&&(S=this._yEase,d=c-d),x=xo(this._tTime,p),d===a&&!o&&this._initted&&g===x)return this._tTime=f,this;g!==x&&(A&&this._yEase&&Ng(A,m),this.vars.repeatRefresh&&!m&&!this._lock&&d!==p&&this._initted&&(this._lock=o=1,this.render(Dt(p*g),!0).invalidate()._lock=0))}if(!this._initted){if(Mg(this,u?s:d,o,r,f))return this._tTime=0,this;if(a!==this._time&&!(o&&this.vars.repeatRefresh&&g!==x))return this;if(c!==this._dur)return this.render(s,r,o)}if(this._tTime=f,this._time=d,!this._act&&this._ts&&(this._act=1,this._lazy=0),this.ratio=_=(S||this._ease)(d/c),this._from&&(this.ratio=_=1-_),!a&&f&&!r&&!x&&(Wn(this,"onStart"),this._tTime!==f))return this;for(h=this._pt;h;)h.r(_,h.d),h=h._next;A&&A.render(s<0?s:A._dur*A._ease(d/this._dur),r,o)||this._startAt&&(this._zTime=s),this._onUpdate&&!r&&(u&&gf(this,s,r,o),Wn(this,"onUpdate")),this._repeat&&g!==x&&this.vars.onRepeat&&!r&&this.parent&&Wn(this,"onRepeat"),(f===this._tDur||!f)&&this._tTime===f&&(u&&!this._onUpdate&&gf(this,s,!0,!0),(s||!c)&&(f===this._tDur&&this._ts>0||!f&&this._ts<0)&&Os(this,1),!r&&!(u&&!a)&&(f||a||m)&&(Wn(this,f===l?"onComplete":"onReverseComplete",!0),this._prom&&!(f<l&&this.timeScale()>0)&&this._prom()))}return this},t.targets=function(){return this._targets},t.invalidate=function(s){return(!s||!this.vars.runBackwards)&&(this._startAt=0),this._pt=this._op=this._onUpdate=this._lazy=this.ratio=0,this._ptLookup=[],this.timeline&&this.timeline.invalidate(s),i.prototype.invalidate.call(this,s)},t.resetTo=function(s,r,o,a,l){pa||Hn.wake(),this._ts||this.play();var c=Math.min(this._dur,(this._dp._time-this._start)*this._ts),u;return this._initted||Dd(this,c),u=this._ease(c/this._dur),A1(this,s,r,o,a,u,c,l)?this.resetTo(s,r,o,a,1):(ic(this,0),this.parent||vg(this._dp,this,"_first","_last",this._dp._sort?"_start":0),this.render(0))},t.kill=function(s,r){if(r===void 0&&(r="all"),!s&&(!r||r==="all"))return this._lazy=this._pt=0,this.parent?Bo(this):this.scrollTrigger&&this.scrollTrigger.kill(!!Jt),this;if(this.timeline){var o=this.timeline.totalDuration();return this.timeline.killTweensOf(s,r,Ts&&Ts.vars.overwrite!==!0)._first||Bo(this),this.parent&&o!==this.timeline.totalDuration()&&_o(this,this._dur*this.timeline._tDur/o,0,1),this}var a=this._targets,l=s?oi(s):a,c=this._ptLookup,u=this._pt,f,d,h,g,p,x,m;if((!r||r==="all")&&jE(a,l))return r==="all"&&(this._pt=0),Bo(this);for(f=this._op=this._op||[],r!=="all"&&(jt(r)&&(p={},Rn(r,function(_){return p[_]=1}),r=p),r=S1(a,r)),m=a.length;m--;)if(~l.indexOf(a[m])){d=c[m],r==="all"?(f[m]=r,g=d,h={}):(h=f[m]=f[m]||{},g=r);for(p in g)x=d&&d[p],x&&((!("kill"in x.d)||x.d.kill(p)===!0)&&tc(this,x,"_pt"),delete d[p]),h!=="all"&&(h[p]=1)}return this._initted&&!this._pt&&u&&Bo(this),this},e.to=function(s,r){return new e(s,r,arguments[2])},e.from=function(s,r){return jo(1,arguments)},e.delayedCall=function(s,r,o,a){return new e(r,0,{immediateRender:!1,lazy:!1,overwrite:!1,delay:s,onComplete:r,onReverseComplete:r,onCompleteParams:o,onReverseCompleteParams:o,callbackScope:a})},e.fromTo=function(s,r,o){return jo(2,arguments)},e.set=function(s,r){return r.duration=0,r.repeatDelay||(r.repeat=0),new e(s,r)},e.killTweensOf=function(s,r,o){return Pt.killTweensOf(s,r,o)},e})(ma);Kn(Gt.prototype,{_targets:[],_lazy:0,_startAt:0,_op:0,_onInit:0});Rn("staggerTo,staggerFrom,staggerFromTo",function(i){Gt[i]=function(){var e=new mn,t=_f.call(arguments,0);return t.splice(i==="staggerFromTo"?5:4,0,0),e[i].apply(e,t)}});var Pd=function(e,t,n){return e[t]=n},Wg=function(e,t,n){return e[t](n)},y1=function(e,t,n,s){return e[t](s.fp,n)},b1=function(e,t,n){return e.setAttribute(t,n)},Fd=function(e,t){return Ut(e[t])?Wg:yd(e[t])&&e.setAttribute?b1:Pd},Xg=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e6)/1e6,t)},M1=function(e,t){return t.set(t.t,t.p,!!(t.s+t.c*e),t)},qg=function(e,t){var n=t._pt,s="";if(!e&&t.b)s=t.b;else if(e===1&&t.e)s=t.e;else{for(;n;)s=n.p+(n.m?n.m(n.s+n.c*e):Math.round((n.s+n.c*e)*1e4)/1e4)+s,n=n._next;s+=t.c}t.set(t.t,t.p,s,t)},Ld=function(e,t){for(var n=t._pt;n;)n.r(e,n.d),n=n._next},C1=function(e,t,n,s){for(var r=this._pt,o;r;)o=r._next,r.p===s&&r.modifier(e,t,n),r=o},T1=function(e){for(var t=this._pt,n,s;t;)s=t._next,t.p===e&&!t.op||t.op===e?tc(this,t,"_pt"):t.dep||(n=1),t=s;return!n},E1=function(e,t,n,s){s.mSet(e,t,s.m.call(s.tween,n,s.mt),s)},Qg=function(e){for(var t=e._pt,n,s,r,o;t;){for(n=t._next,s=r;s&&s.pr>t.pr;)s=s._next;(t._prev=s?s._prev:o)?t._prev._next=t:r=t,(t._next=s)?s._prev=t:o=t,t=n}e._pt=r},In=(function(){function i(t,n,s,r,o,a,l,c,u){this.t=n,this.s=r,this.c=o,this.p=s,this.r=a||Xg,this.d=l||this,this.set=c||Pd,this.pr=u||0,this._next=t,t&&(t._prev=this)}var e=i.prototype;return e.modifier=function(n,s,r){this.mSet=this.mSet||this.set,this.set=E1,this.m=n,this.mt=r,this.tween=s},i})();Rn(Ed+"parent,duration,ease,delay,overwrite,runBackwards,startAt,yoyo,immediateRender,repeat,repeatDelay,data,paused,reversed,lazy,callbackScope,stringFilter,id,yoyoEase,stagger,inherit,repeatRefresh,keyframes,autoRevert,scrollTrigger",function(i){return Td[i]=1});Yn.TweenMax=Yn.TweenLite=Gt;Yn.TimelineLite=Yn.TimelineMax=mn;Pt=new mn({sortChildren:!1,defaults:mo,autoRemoveChildren:!0,id:"root",smoothChildTiming:!0});Qn.stringFilter=Ug;var dr=[],Al={},w1=[],sm=0,R1=0,lu=function(e){return(Al[e]||w1).map(function(t){return t()})},yf=function(){var e=Date.now(),t=[];e-sm>2&&(lu("matchMediaInit"),dr.forEach(function(n){var s=n.queries,r=n.conditions,o,a,l,c;for(a in s)o=yi.matchMedia(s[a]).matches,o&&(l=1),o!==r[a]&&(r[a]=o,c=1);c&&(n.revert(),l&&t.push(n))}),lu("matchMediaRevert"),t.forEach(function(n){return n.onMatch(n,function(s){return n.add(null,s)})}),sm=e,lu("matchMedia"))},Yg=(function(){function i(t,n){this.selector=n&&Af(n),this.data=[],this._r=[],this.isReverted=!1,this.id=R1++,t&&this.add(t)}var e=i.prototype;return e.add=function(n,s,r){Ut(n)&&(r=s,s=n,n=Ut);var o=this,a=function(){var c=Tt,u=o.selector,f;return c&&c!==o&&c.data.push(o),r&&(o.selector=Af(r)),Tt=o,f=s.apply(o,arguments),Ut(f)&&o._r.push(f),Tt=c,o.selector=u,o.isReverted=!1,f};return o.last=a,n===Ut?a(o,function(l){return o.add(null,l)}):n?o[n]=a:a},e.ignore=function(n){var s=Tt;Tt=null,n(this),Tt=s},e.getTweens=function(){var n=[];return this.data.forEach(function(s){return s instanceof i?n.push.apply(n,s.getTweens()):s instanceof Gt&&!(s.parent&&s.parent.data==="nested")&&n.push(s)}),n},e.clear=function(){this._r.length=this.data.length=0},e.kill=function(n,s){var r=this;if(n?(function(){for(var a=r.getTweens(),l=r.data.length,c;l--;)c=r.data[l],c.data==="isFlip"&&(c.revert(),c.getChildren(!0,!0,!1).forEach(function(u){return a.splice(a.indexOf(u),1)}));for(a.map(function(u){return{g:u._dur||u._delay||u._sat&&!u._sat.vars.immediateRender?u.globalTime(0):-1/0,t:u}}).sort(function(u,f){return f.g-u.g||-1/0}).forEach(function(u){return u.t.revert(n)}),l=r.data.length;l--;)c=r.data[l],c instanceof mn?c.data!=="nested"&&(c.scrollTrigger&&c.scrollTrigger.revert(),c.kill()):!(c instanceof Gt)&&c.revert&&c.revert(n);r._r.forEach(function(u){return u(n,r)}),r.isReverted=!0})():this.data.forEach(function(a){return a.kill&&a.kill()}),this.clear(),s)for(var o=dr.length;o--;)dr[o].id===this.id&&dr.splice(o,1)},e.revert=function(n){this.kill(n||{})},i})(),I1=(function(){function i(t){this.contexts=[],this.scope=t,Tt&&Tt.data.push(this)}var e=i.prototype;return e.add=function(n,s,r){Oi(n)||(n={matches:n});var o=new Yg(0,r||this.scope),a=o.conditions={},l,c,u;Tt&&!o.selector&&(o.selector=Tt.selector),this.contexts.push(o),s=o.add("onMatch",s),o.queries=n;for(c in n)c==="all"?u=1:(l=yi.matchMedia(n[c]),l&&(dr.indexOf(o)<0&&dr.push(o),(a[c]=l.matches)&&(u=1),l.addListener?l.addListener(yf):l.addEventListener("change",yf)));return u&&s(o,function(f){return o.add(null,f)}),this},e.revert=function(n){this.kill(n||{})},e.kill=function(n){this.contexts.forEach(function(s){return s.kill(n,!0)})},i})(),Ol={registerPlugin:function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];t.forEach(function(s){return Fg(s)})},timeline:function(e){return new mn(e)},getTweensOf:function(e,t){return Pt.getTweensOf(e,t)},getProperty:function(e,t,n,s){jt(e)&&(e=oi(e)[0]);var r=cr(e||{}).get,o=n?Sg:Ag;return n==="native"&&(n=""),e&&(t?o((kn[t]&&kn[t].get||r)(e,t,n,s)):function(a,l,c){return o((kn[a]&&kn[a].get||r)(e,a,l,c))})},quickSetter:function(e,t,n){if(e=oi(e),e.length>1){var s=e.map(function(u){return Pn.quickSetter(u,t,n)}),r=s.length;return function(u){for(var f=r;f--;)s[f](u)}}e=e[0]||{};var o=kn[t],a=cr(e),l=a.harness&&(a.harness.aliases||{})[t]||t,c=o?function(u){var f=new o;Wr._pt=0,f.init(e,n?u+n:u,Wr,0,[e]),f.render(1,f),Wr._pt&&Ld(1,Wr)}:a.set(e,l);return o?c:function(u){return c(e,l,n?u+n:u,a,1)}},quickTo:function(e,t,n){var s,r=Pn.to(e,Kn((s={},s[t]="+=0.1",s.paused=!0,s.stagger=0,s),n||{})),o=function(l,c,u){return r.resetTo(t,l,c,u)};return o.tween=r,o},isTweening:function(e){return Pt.getTweensOf(e,!0).length>0},defaults:function(e){return e&&e.ease&&(e.ease=fr(e.ease,mo.ease)),Jp(mo,e||{})},config:function(e){return Jp(Qn,e||{})},registerEffect:function(e){var t=e.name,n=e.effect,s=e.plugins,r=e.defaults,o=e.extendTimeline;(s||"").split(",").forEach(function(a){return a&&!kn[a]&&!Yn[a]&&fa(t+" effect requires "+a+" plugin.")}),su[t]=function(a,l,c){return n(oi(a),Kn(l||{},r),c)},o&&(mn.prototype[t]=function(a,l,c){return this.add(su[t](a,Oi(l)?l:(c=l)&&{},this),c)})},registerEase:function(e,t){tt[e]=fr(t)},parseEase:function(e,t){return arguments.length?fr(e,t):tt},getById:function(e){return Pt.getById(e)},exportRoot:function(e,t){e===void 0&&(e={});var n=new mn(e),s,r;for(n.smoothChildTiming=wn(e.smoothChildTiming),Pt.remove(n),n._dp=0,n._time=n._tTime=Pt._time,s=Pt._first;s;)r=s._next,(t||!(!s._dur&&s instanceof Gt&&s.vars.onComplete===s._targets[0]))&&Ti(n,s,s._start-s._delay),s=r;return Ti(Pt,n,0),n},context:function(e,t){return e?new Yg(e,t):Tt},matchMedia:function(e){return new I1(e)},matchMediaRefresh:function(){return dr.forEach(function(e){var t=e.conditions,n,s;for(s in t)t[s]&&(t[s]=!1,n=1);n&&e.revert()})||yf()},addEventListener:function(e,t){var n=Al[e]||(Al[e]=[]);~n.indexOf(t)||n.push(t)},removeEventListener:function(e,t){var n=Al[e],s=n&&n.indexOf(t);s>=0&&n.splice(s,1)},utils:{wrap:l1,wrapYoyo:c1,distribute:Eg,random:Rg,snap:wg,normalize:a1,getUnit:sn,clamp:i1,splitColor:Lg,toArray:oi,selector:Af,mapRange:Dg,pipe:r1,unitize:o1,interpolate:u1,shuffle:Tg},install:pg,effects:su,ticker:Hn,updateRoot:mn.updateRoot,plugins:kn,globalTimeline:Pt,core:{PropTween:In,globals:mg,Tween:Gt,Timeline:mn,Animation:ma,getCache:cr,_removeLinkedListItem:tc,reverting:function(){return Jt},context:function(e){return e&&Tt&&(Tt.data.push(e),e._ctx=Tt),Tt},suppressOverwrites:function(e){return vd=e}}};Rn("to,from,fromTo,delayedCall,set,killTweensOf",function(i){return Ol[i]=Gt[i]});Hn.add(mn.updateRoot);Wr=Ol.to({},{duration:0});var D1=function(e,t){for(var n=e._pt;n&&n.p!==t&&n.op!==t&&n.fp!==t;)n=n._next;return n},P1=function(e,t){var n=e._targets,s,r,o;for(s in t)for(r=n.length;r--;)o=e._ptLookup[r][s],o&&(o=o.d)&&(o._pt&&(o=D1(o,s)),o&&o.modifier&&o.modifier(t[s],e,n[r],s))},cu=function(e,t){return{name:e,headless:1,rawVars:1,init:function(s,r,o){o._onInit=function(a){var l,c;if(jt(r)&&(l={},Rn(r,function(u){return l[u]=1}),r=l),t){l={};for(c in r)l[c]=t(r[c]);r=l}P1(a,r)}}}},Pn=Ol.registerPlugin({name:"attr",init:function(e,t,n,s,r){var o,a,l;this.tween=n;for(o in t)l=e.getAttribute(o)||"",a=this.add(e,"setAttribute",(l||0)+"",t[o],s,r,0,0,o),a.op=o,a.b=l,this._props.push(o)},render:function(e,t){for(var n=t._pt;n;)Jt?n.set(n.t,n.p,n.b,n):n.r(e,n.d),n=n._next}},{name:"endArray",headless:1,init:function(e,t){for(var n=t.length;n--;)this.add(e,n,e[n]||0,t[n],0,0,0,0,0,1)}},cu("roundProps",Sf),cu("modifiers"),cu("snap",wg))||Ol;Gt.version=mn.version=Pn.version="3.14.2";hg=1;bd()&&Ao();tt.Power0;tt.Power1;tt.Power2;tt.Power3;tt.Power4;tt.Linear;tt.Quad;tt.Cubic;tt.Quart;tt.Quint;tt.Strong;tt.Elastic;tt.Back;tt.SteppedEase;tt.Bounce;tt.Sine;tt.Expo;tt.Circ;var rm,Es,no,Bd,or,om,Ud,F1=function(){return typeof window<"u"},cs={},tr=180/Math.PI,io=Math.PI/180,Or=Math.atan2,am=1e8,Od=/([A-Z])/g,L1=/(left|right|width|margin|padding|x)/i,B1=/[\s,\(]\S/,Ii={autoAlpha:"opacity,visibility",scale:"scaleX,scaleY",alpha:"opacity"},bf=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},U1=function(e,t){return t.set(t.t,t.p,e===1?t.e:Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},O1=function(e,t){return t.set(t.t,t.p,e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},N1=function(e,t){return t.set(t.t,t.p,e===1?t.e:e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},z1=function(e,t){var n=t.s+t.c*e;t.set(t.t,t.p,~~(n+(n<0?-.5:.5))+t.u,t)},Kg=function(e,t){return t.set(t.t,t.p,e?t.e:t.b,t)},jg=function(e,t){return t.set(t.t,t.p,e!==1?t.b:t.e,t)},k1=function(e,t,n){return e.style[t]=n},H1=function(e,t,n){return e.style.setProperty(t,n)},V1=function(e,t,n){return e._gsap[t]=n},G1=function(e,t,n){return e._gsap.scaleX=e._gsap.scaleY=n},W1=function(e,t,n,s,r){var o=e._gsap;o.scaleX=o.scaleY=n,o.renderTransform(r,o)},X1=function(e,t,n,s,r){var o=e._gsap;o[t]=n,o.renderTransform(r,o)},Ft="transform",Dn=Ft+"Origin",q1=function i(e,t){var n=this,s=this.target,r=s.style,o=s._gsap;if(e in cs&&r){if(this.tfm=this.tfm||{},e!=="transform")e=Ii[e]||e,~e.indexOf(",")?e.split(",").forEach(function(a){return n.tfm[a]=Zi(s,a)}):this.tfm[e]=o.x?o[e]:Zi(s,e),e===Dn&&(this.tfm.zOrigin=o.zOrigin);else return Ii.transform.split(",").forEach(function(a){return i.call(n,a,t)});if(this.props.indexOf(Ft)>=0)return;o.svg&&(this.svgo=s.getAttribute("data-svg-origin"),this.props.push(Dn,t,"")),e=Ft}(r||t)&&this.props.push(e,t,r[e])},$g=function(e){e.translate&&(e.removeProperty("translate"),e.removeProperty("scale"),e.removeProperty("rotate"))},Q1=function(){var e=this.props,t=this.target,n=t.style,s=t._gsap,r,o;for(r=0;r<e.length;r+=3)e[r+1]?e[r+1]===2?t[e[r]](e[r+2]):t[e[r]]=e[r+2]:e[r+2]?n[e[r]]=e[r+2]:n.removeProperty(e[r].substr(0,2)==="--"?e[r]:e[r].replace(Od,"-$1").toLowerCase());if(this.tfm){for(o in this.tfm)s[o]=this.tfm[o];s.svg&&(s.renderTransform(),t.setAttribute("data-svg-origin",this.svgo||"")),r=Ud(),(!r||!r.isStart)&&!n[Ft]&&($g(n),s.zOrigin&&n[Dn]&&(n[Dn]+=" "+s.zOrigin+"px",s.zOrigin=0,s.renderTransform()),s.uncache=1)}},Zg=function(e,t){var n={target:e,props:[],revert:Q1,save:q1};return e._gsap||Pn.core.getCache(e),t&&e.style&&e.nodeType&&t.split(",").forEach(function(s){return n.save(s)}),n},Jg,Mf=function(e,t){var n=Es.createElementNS?Es.createElementNS((t||"http://www.w3.org/1999/xhtml").replace(/^https/,"http"),e):Es.createElement(e);return n&&n.style?n:Es.createElement(e)},Xn=function i(e,t,n){var s=getComputedStyle(e);return s[t]||s.getPropertyValue(t.replace(Od,"-$1").toLowerCase())||s.getPropertyValue(t)||!n&&i(e,So(t)||t,1)||""},lm="O,Moz,ms,Ms,Webkit".split(","),So=function(e,t,n){var s=t||or,r=s.style,o=5;if(e in r&&!n)return e;for(e=e.charAt(0).toUpperCase()+e.substr(1);o--&&!(lm[o]+e in r););return o<0?null:(o===3?"ms":o>=0?lm[o]:"")+e},Cf=function(){F1()&&window.document&&(rm=window,Es=rm.document,no=Es.documentElement,or=Mf("div")||{style:{}},Mf("div"),Ft=So(Ft),Dn=Ft+"Origin",or.style.cssText="border-width:0;line-height:0;position:absolute;padding:0",Jg=!!So("perspective"),Ud=Pn.core.reverting,Bd=1)},cm=function(e){var t=e.ownerSVGElement,n=Mf("svg",t&&t.getAttribute("xmlns")||"http://www.w3.org/2000/svg"),s=e.cloneNode(!0),r;s.style.display="block",n.appendChild(s),no.appendChild(n);try{r=s.getBBox()}catch{}return n.removeChild(s),no.removeChild(n),r},um=function(e,t){for(var n=t.length;n--;)if(e.hasAttribute(t[n]))return e.getAttribute(t[n])},ex=function(e){var t,n;try{t=e.getBBox()}catch{t=cm(e),n=1}return t&&(t.width||t.height)||n||(t=cm(e)),t&&!t.width&&!t.x&&!t.y?{x:+um(e,["x","cx","x1"])||0,y:+um(e,["y","cy","y1"])||0,width:0,height:0}:t},tx=function(e){return!!(e.getCTM&&(!e.parentNode||e.ownerSVGElement)&&ex(e))},Ns=function(e,t){if(t){var n=e.style,s;t in cs&&t!==Dn&&(t=Ft),n.removeProperty?(s=t.substr(0,2),(s==="ms"||t.substr(0,6)==="webkit")&&(t="-"+t),n.removeProperty(s==="--"?t:t.replace(Od,"-$1").toLowerCase())):n.removeAttribute(t)}},ws=function(e,t,n,s,r,o){var a=new In(e._pt,t,n,0,1,o?jg:Kg);return e._pt=a,a.b=s,a.e=r,e._props.push(n),a},fm={deg:1,rad:1,turn:1},Y1={grid:1,flex:1},zs=function i(e,t,n,s){var r=parseFloat(n)||0,o=(n+"").trim().substr((r+"").length)||"px",a=or.style,l=L1.test(t),c=e.tagName.toLowerCase()==="svg",u=(c?"client":"offset")+(l?"Width":"Height"),f=100,d=s==="px",h=s==="%",g,p,x,m;if(s===o||!r||fm[s]||fm[o])return r;if(o!=="px"&&!d&&(r=i(e,t,n,"px")),m=e.getCTM&&tx(e),(h||o==="%")&&(cs[t]||~t.indexOf("adius")))return g=m?e.getBBox()[l?"width":"height"]:e[u],Ot(h?r/g*f:r/100*g);if(a[l?"width":"height"]=f+(d?o:s),p=s!=="rem"&&~t.indexOf("adius")||s==="em"&&e.appendChild&&!c?e:e.parentNode,m&&(p=(e.ownerSVGElement||{}).parentNode),(!p||p===Es||!p.appendChild)&&(p=Es.body),x=p._gsap,x&&h&&x.width&&l&&x.time===Hn.time&&!x.uncache)return Ot(r/x.width*f);if(h&&(t==="height"||t==="width")){var _=e.style[t];e.style[t]=f+s,g=e[u],_?e.style[t]=_:Ns(e,t)}else(h||o==="%")&&!Y1[Xn(p,"display")]&&(a.position=Xn(e,"position")),p===e&&(a.position="static"),p.appendChild(or),g=or[u],p.removeChild(or),a.position="absolute";return l&&h&&(x=cr(p),x.time=Hn.time,x.width=p[u]),Ot(d?g*r/f:g&&r?f/g*r:0)},Zi=function(e,t,n,s){var r;return Bd||Cf(),t in Ii&&t!=="transform"&&(t=Ii[t],~t.indexOf(",")&&(t=t.split(",")[0])),cs[t]&&t!=="transform"?(r=xa(e,s),r=t!=="transformOrigin"?r[t]:r.svg?r.origin:zl(Xn(e,Dn))+" "+r.zOrigin+"px"):(r=e.style[t],(!r||r==="auto"||s||~(r+"").indexOf("calc("))&&(r=Nl[t]&&Nl[t](e,t,n)||Xn(e,t)||xg(e,t)||(t==="opacity"?1:0))),n&&!~(r+"").trim().indexOf(" ")?zs(e,t,r,n)+n:r},K1=function(e,t,n,s){if(!n||n==="none"){var r=So(t,e,1),o=r&&Xn(e,r,1);o&&o!==n?(t=r,n=o):t==="borderColor"&&(n=Xn(e,"borderTopColor"))}var a=new In(this._pt,e.style,t,0,1,qg),l=0,c=0,u,f,d,h,g,p,x,m,_,A,S,v;if(a.b=n,a.e=s,n+="",s+="",s.substring(0,6)==="var(--"&&(s=Xn(e,s.substring(4,s.indexOf(")")))),s==="auto"&&(p=e.style[t],e.style[t]=s,s=Xn(e,t)||s,p?e.style[t]=p:Ns(e,t)),u=[n,s],Ug(u),n=u[0],s=u[1],d=n.match(Gr)||[],v=s.match(Gr)||[],v.length){for(;f=Gr.exec(s);)x=f[0],_=s.substring(l,f.index),g?g=(g+1)%5:(_.substr(-5)==="rgba("||_.substr(-5)==="hsla(")&&(g=1),x!==(p=d[c++]||"")&&(h=parseFloat(p)||0,S=p.substr((h+"").length),x.charAt(1)==="="&&(x=to(h,x)+S),m=parseFloat(x),A=x.substr((m+"").length),l=Gr.lastIndex-A.length,A||(A=A||Qn.units[t]||S,l===s.length&&(s+=A,a.e+=A)),S!==A&&(h=zs(e,t,p,A)||0),a._pt={_next:a._pt,p:_||c===1?_:",",s:h,c:m-h,m:g&&g<4||t==="zIndex"?Math.round:0});a.c=l<s.length?s.substring(l,s.length):""}else a.r=t==="display"&&s==="none"?jg:Kg;return dg.test(s)&&(a.e=0),this._pt=a,a},dm={top:"0%",bottom:"100%",left:"0%",right:"100%",center:"50%"},j1=function(e){var t=e.split(" "),n=t[0],s=t[1]||"50%";return(n==="top"||n==="bottom"||s==="left"||s==="right")&&(e=n,n=s,s=e),t[0]=dm[n]||n,t[1]=dm[s]||s,t.join(" ")},$1=function(e,t){if(t.tween&&t.tween._time===t.tween._dur){var n=t.t,s=n.style,r=t.u,o=n._gsap,a,l,c;if(r==="all"||r===!0)s.cssText="",l=1;else for(r=r.split(","),c=r.length;--c>-1;)a=r[c],cs[a]&&(l=1,a=a==="transformOrigin"?Dn:Ft),Ns(n,a);l&&(Ns(n,Ft),o&&(o.svg&&n.removeAttribute("transform"),s.scale=s.rotate=s.translate="none",xa(n,1),o.uncache=1,$g(s)))}},Nl={clearProps:function(e,t,n,s,r){if(r.data!=="isFromStart"){var o=e._pt=new In(e._pt,t,n,0,0,$1);return o.u=s,o.pr=-10,o.tween=r,e._props.push(n),1}}},ga=[1,0,0,1,0,0],nx={},ix=function(e){return e==="matrix(1, 0, 0, 1, 0, 0)"||e==="none"||!e},hm=function(e){var t=Xn(e,Ft);return ix(t)?ga:t.substr(7).match(fg).map(Ot)},Nd=function(e,t){var n=e._gsap||cr(e),s=e.style,r=hm(e),o,a,l,c;return n.svg&&e.getAttribute("transform")?(l=e.transform.baseVal.consolidate().matrix,r=[l.a,l.b,l.c,l.d,l.e,l.f],r.join(",")==="1,0,0,1,0,0"?ga:r):(r===ga&&!e.offsetParent&&e!==no&&!n.svg&&(l=s.display,s.display="block",o=e.parentNode,(!o||!e.offsetParent&&!e.getBoundingClientRect().width)&&(c=1,a=e.nextElementSibling,no.appendChild(e)),r=hm(e),l?s.display=l:Ns(e,"display"),c&&(a?o.insertBefore(e,a):o?o.appendChild(e):no.removeChild(e))),t&&r.length>6?[r[0],r[1],r[4],r[5],r[12],r[13]]:r)},Tf=function(e,t,n,s,r,o){var a=e._gsap,l=r||Nd(e,!0),c=a.xOrigin||0,u=a.yOrigin||0,f=a.xOffset||0,d=a.yOffset||0,h=l[0],g=l[1],p=l[2],x=l[3],m=l[4],_=l[5],A=t.split(" "),S=parseFloat(A[0])||0,v=parseFloat(A[1])||0,b,M,E,y;n?l!==ga&&(M=h*x-g*p)&&(E=S*(x/M)+v*(-p/M)+(p*_-x*m)/M,y=S*(-g/M)+v*(h/M)-(h*_-g*m)/M,S=E,v=y):(b=ex(e),S=b.x+(~A[0].indexOf("%")?S/100*b.width:S),v=b.y+(~(A[1]||A[0]).indexOf("%")?v/100*b.height:v)),s||s!==!1&&a.smooth?(m=S-c,_=v-u,a.xOffset=f+(m*h+_*p)-m,a.yOffset=d+(m*g+_*x)-_):a.xOffset=a.yOffset=0,a.xOrigin=S,a.yOrigin=v,a.smooth=!!s,a.origin=t,a.originIsAbsolute=!!n,e.style[Dn]="0px 0px",o&&(ws(o,a,"xOrigin",c,S),ws(o,a,"yOrigin",u,v),ws(o,a,"xOffset",f,a.xOffset),ws(o,a,"yOffset",d,a.yOffset)),e.setAttribute("data-svg-origin",S+" "+v)},xa=function(e,t){var n=e._gsap||new kg(e);if("x"in n&&!t&&!n.uncache)return n;var s=e.style,r=n.scaleX<0,o="px",a="deg",l=getComputedStyle(e),c=Xn(e,Dn)||"0",u,f,d,h,g,p,x,m,_,A,S,v,b,M,E,y,C,I,P,U,O,k,z,W,H,$,X,Z,pe,ge,be,Me;return u=f=d=p=x=m=_=A=S=0,h=g=1,n.svg=!!(e.getCTM&&tx(e)),l.translate&&((l.translate!=="none"||l.scale!=="none"||l.rotate!=="none")&&(s[Ft]=(l.translate!=="none"?"translate3d("+(l.translate+" 0 0").split(" ").slice(0,3).join(", ")+") ":"")+(l.rotate!=="none"?"rotate("+l.rotate+") ":"")+(l.scale!=="none"?"scale("+l.scale.split(" ").join(",")+") ":"")+(l[Ft]!=="none"?l[Ft]:"")),s.scale=s.rotate=s.translate="none"),M=Nd(e,n.svg),n.svg&&(n.uncache?(H=e.getBBox(),c=n.xOrigin-H.x+"px "+(n.yOrigin-H.y)+"px",W=""):W=!t&&e.getAttribute("data-svg-origin"),Tf(e,W||c,!!W||n.originIsAbsolute,n.smooth!==!1,M)),v=n.xOrigin||0,b=n.yOrigin||0,M!==ga&&(I=M[0],P=M[1],U=M[2],O=M[3],u=k=M[4],f=z=M[5],M.length===6?(h=Math.sqrt(I*I+P*P),g=Math.sqrt(O*O+U*U),p=I||P?Or(P,I)*tr:0,_=U||O?Or(U,O)*tr+p:0,_&&(g*=Math.abs(Math.cos(_*io))),n.svg&&(u-=v-(v*I+b*U),f-=b-(v*P+b*O))):(Me=M[6],ge=M[7],X=M[8],Z=M[9],pe=M[10],be=M[11],u=M[12],f=M[13],d=M[14],E=Or(Me,pe),x=E*tr,E&&(y=Math.cos(-E),C=Math.sin(-E),W=k*y+X*C,H=z*y+Z*C,$=Me*y+pe*C,X=k*-C+X*y,Z=z*-C+Z*y,pe=Me*-C+pe*y,be=ge*-C+be*y,k=W,z=H,Me=$),E=Or(-U,pe),m=E*tr,E&&(y=Math.cos(-E),C=Math.sin(-E),W=I*y-X*C,H=P*y-Z*C,$=U*y-pe*C,be=O*C+be*y,I=W,P=H,U=$),E=Or(P,I),p=E*tr,E&&(y=Math.cos(E),C=Math.sin(E),W=I*y+P*C,H=k*y+z*C,P=P*y-I*C,z=z*y-k*C,I=W,k=H),x&&Math.abs(x)+Math.abs(p)>359.9&&(x=p=0,m=180-m),h=Ot(Math.sqrt(I*I+P*P+U*U)),g=Ot(Math.sqrt(z*z+Me*Me)),E=Or(k,z),_=Math.abs(E)>2e-4?E*tr:0,S=be?1/(be<0?-be:be):0),n.svg&&(W=e.getAttribute("transform"),n.forceCSS=e.setAttribute("transform","")||!ix(Xn(e,Ft)),W&&e.setAttribute("transform",W))),Math.abs(_)>90&&Math.abs(_)<270&&(r?(h*=-1,_+=p<=0?180:-180,p+=p<=0?180:-180):(g*=-1,_+=_<=0?180:-180)),t=t||n.uncache,n.x=u-((n.xPercent=u&&(!t&&n.xPercent||(Math.round(e.offsetWidth/2)===Math.round(-u)?-50:0)))?e.offsetWidth*n.xPercent/100:0)+o,n.y=f-((n.yPercent=f&&(!t&&n.yPercent||(Math.round(e.offsetHeight/2)===Math.round(-f)?-50:0)))?e.offsetHeight*n.yPercent/100:0)+o,n.z=d+o,n.scaleX=Ot(h),n.scaleY=Ot(g),n.rotation=Ot(p)+a,n.rotationX=Ot(x)+a,n.rotationY=Ot(m)+a,n.skewX=_+a,n.skewY=A+a,n.transformPerspective=S+o,(n.zOrigin=parseFloat(c.split(" ")[2])||!t&&n.zOrigin||0)&&(s[Dn]=zl(c)),n.xOffset=n.yOffset=0,n.force3D=Qn.force3D,n.renderTransform=n.svg?J1:Jg?sx:Z1,n.uncache=0,n},zl=function(e){return(e=e.split(" "))[0]+" "+e[1]},uu=function(e,t,n){var s=sn(t);return Ot(parseFloat(t)+parseFloat(zs(e,"x",n+"px",s)))+s},Z1=function(e,t){t.z="0px",t.rotationY=t.rotationX="0deg",t.force3D=0,sx(e,t)},Zs="0deg",Po="0px",Js=") ",sx=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.z,c=n.rotation,u=n.rotationY,f=n.rotationX,d=n.skewX,h=n.skewY,g=n.scaleX,p=n.scaleY,x=n.transformPerspective,m=n.force3D,_=n.target,A=n.zOrigin,S="",v=m==="auto"&&e&&e!==1||m===!0;if(A&&(f!==Zs||u!==Zs)){var b=parseFloat(u)*io,M=Math.sin(b),E=Math.cos(b),y;b=parseFloat(f)*io,y=Math.cos(b),o=uu(_,o,M*y*-A),a=uu(_,a,-Math.sin(b)*-A),l=uu(_,l,E*y*-A+A)}x!==Po&&(S+="perspective("+x+Js),(s||r)&&(S+="translate("+s+"%, "+r+"%) "),(v||o!==Po||a!==Po||l!==Po)&&(S+=l!==Po||v?"translate3d("+o+", "+a+", "+l+") ":"translate("+o+", "+a+Js),c!==Zs&&(S+="rotate("+c+Js),u!==Zs&&(S+="rotateY("+u+Js),f!==Zs&&(S+="rotateX("+f+Js),(d!==Zs||h!==Zs)&&(S+="skew("+d+", "+h+Js),(g!==1||p!==1)&&(S+="scale("+g+", "+p+Js),_.style[Ft]=S||"translate(0, 0)"},J1=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.rotation,c=n.skewX,u=n.skewY,f=n.scaleX,d=n.scaleY,h=n.target,g=n.xOrigin,p=n.yOrigin,x=n.xOffset,m=n.yOffset,_=n.forceCSS,A=parseFloat(o),S=parseFloat(a),v,b,M,E,y;l=parseFloat(l),c=parseFloat(c),u=parseFloat(u),u&&(u=parseFloat(u),c+=u,l+=u),l||c?(l*=io,c*=io,v=Math.cos(l)*f,b=Math.sin(l)*f,M=Math.sin(l-c)*-d,E=Math.cos(l-c)*d,c&&(u*=io,y=Math.tan(c-u),y=Math.sqrt(1+y*y),M*=y,E*=y,u&&(y=Math.tan(u),y=Math.sqrt(1+y*y),v*=y,b*=y)),v=Ot(v),b=Ot(b),M=Ot(M),E=Ot(E)):(v=f,E=d,b=M=0),(A&&!~(o+"").indexOf("px")||S&&!~(a+"").indexOf("px"))&&(A=zs(h,"x",o,"px"),S=zs(h,"y",a,"px")),(g||p||x||m)&&(A=Ot(A+g-(g*v+p*M)+x),S=Ot(S+p-(g*b+p*E)+m)),(s||r)&&(y=h.getBBox(),A=Ot(A+s/100*y.width),S=Ot(S+r/100*y.height)),y="matrix("+v+","+b+","+M+","+E+","+A+","+S+")",h.setAttribute("transform",y),_&&(h.style[Ft]=y)},ew=function(e,t,n,s,r){var o=360,a=jt(r),l=parseFloat(r)*(a&&~r.indexOf("rad")?tr:1),c=l-s,u=s+c+"deg",f,d;return a&&(f=r.split("_")[1],f==="short"&&(c%=o,c!==c%(o/2)&&(c+=c<0?o:-o)),f==="cw"&&c<0?c=(c+o*am)%o-~~(c/o)*o:f==="ccw"&&c>0&&(c=(c-o*am)%o-~~(c/o)*o)),e._pt=d=new In(e._pt,t,n,s,c,U1),d.e=u,d.u="deg",e._props.push(n),d},pm=function(e,t){for(var n in t)e[n]=t[n];return e},tw=function(e,t,n){var s=pm({},n._gsap),r="perspective,force3D,transformOrigin,svgOrigin",o=n.style,a,l,c,u,f,d,h,g;s.svg?(c=n.getAttribute("transform"),n.setAttribute("transform",""),o[Ft]=t,a=xa(n,1),Ns(n,Ft),n.setAttribute("transform",c)):(c=getComputedStyle(n)[Ft],o[Ft]=t,a=xa(n,1),o[Ft]=c);for(l in cs)c=s[l],u=a[l],c!==u&&r.indexOf(l)<0&&(h=sn(c),g=sn(u),f=h!==g?zs(n,l,c,g):parseFloat(c),d=parseFloat(u),e._pt=new In(e._pt,a,l,f,d-f,bf),e._pt.u=g||0,e._props.push(l));pm(a,s)};Rn("padding,margin,Width,Radius",function(i,e){var t="Top",n="Right",s="Bottom",r="Left",o=(e<3?[t,n,s,r]:[t+r,t+n,s+n,s+r]).map(function(a){return e<2?i+a:"border"+a+i});Nl[e>1?"border"+i:i]=function(a,l,c,u,f){var d,h;if(arguments.length<4)return d=o.map(function(g){return Zi(a,g,c)}),h=d.join(" "),h.split(d[0]).length===5?d[0]:h;d=(u+"").split(" "),h={},o.forEach(function(g,p){return h[g]=d[p]=d[p]||d[(p-1)/2|0]}),a.init(l,h,f)}});var rx={name:"css",register:Cf,targetTest:function(e){return e.style&&e.nodeType},init:function(e,t,n,s,r){var o=this._props,a=e.style,l=n.vars.startAt,c,u,f,d,h,g,p,x,m,_,A,S,v,b,M,E,y;Bd||Cf(),this.styles=this.styles||Zg(e),E=this.styles.props,this.tween=n;for(p in t)if(p!=="autoRound"&&(u=t[p],!(kn[p]&&Hg(p,t,n,s,e,r)))){if(h=typeof u,g=Nl[p],h==="function"&&(u=u.call(n,s,e,r),h=typeof u),h==="string"&&~u.indexOf("random(")&&(u=ha(u)),g)g(this,e,p,u,n)&&(M=1);else if(p.substr(0,2)==="--")c=(getComputedStyle(e).getPropertyValue(p)+"").trim(),u+="",Fs.lastIndex=0,Fs.test(c)||(x=sn(c),m=sn(u),m?x!==m&&(c=zs(e,p,c,m)+m):x&&(u+=x)),this.add(a,"setProperty",c,u,s,r,0,0,p),o.push(p),E.push(p,0,a[p]);else if(h!=="undefined"){if(l&&p in l?(c=typeof l[p]=="function"?l[p].call(n,s,e,r):l[p],jt(c)&&~c.indexOf("random(")&&(c=ha(c)),sn(c+"")||c==="auto"||(c+=Qn.units[p]||sn(Zi(e,p))||""),(c+"").charAt(1)==="="&&(c=Zi(e,p))):c=Zi(e,p),d=parseFloat(c),_=h==="string"&&u.charAt(1)==="="&&u.substr(0,2),_&&(u=u.substr(2)),f=parseFloat(u),p in Ii&&(p==="autoAlpha"&&(d===1&&Zi(e,"visibility")==="hidden"&&f&&(d=0),E.push("visibility",0,a.visibility),ws(this,a,"visibility",d?"inherit":"hidden",f?"inherit":"hidden",!f)),p!=="scale"&&p!=="transform"&&(p=Ii[p],~p.indexOf(",")&&(p=p.split(",")[0]))),A=p in cs,A){if(this.styles.save(p),y=u,h==="string"&&u.substring(0,6)==="var(--"){if(u=Xn(e,u.substring(4,u.indexOf(")"))),u.substring(0,5)==="calc("){var C=e.style.perspective;e.style.perspective=u,u=Xn(e,"perspective"),C?e.style.perspective=C:Ns(e,"perspective")}f=parseFloat(u)}if(S||(v=e._gsap,v.renderTransform&&!t.parseTransform||xa(e,t.parseTransform),b=t.smoothOrigin!==!1&&v.smooth,S=this._pt=new In(this._pt,a,Ft,0,1,v.renderTransform,v,0,-1),S.dep=1),p==="scale")this._pt=new In(this._pt,v,"scaleY",v.scaleY,(_?to(v.scaleY,_+f):f)-v.scaleY||0,bf),this._pt.u=0,o.push("scaleY",p),p+="X";else if(p==="transformOrigin"){E.push(Dn,0,a[Dn]),u=j1(u),v.svg?Tf(e,u,0,b,0,this):(m=parseFloat(u.split(" ")[2])||0,m!==v.zOrigin&&ws(this,v,"zOrigin",v.zOrigin,m),ws(this,a,p,zl(c),zl(u)));continue}else if(p==="svgOrigin"){Tf(e,u,1,b,0,this);continue}else if(p in nx){ew(this,v,p,d,_?to(d,_+u):u);continue}else if(p==="smoothOrigin"){ws(this,v,"smooth",v.smooth,u);continue}else if(p==="force3D"){v[p]=u;continue}else if(p==="transform"){tw(this,u,e);continue}}else p in a||(p=So(p)||p);if(A||(f||f===0)&&(d||d===0)&&!B1.test(u)&&p in a)x=(c+"").substr((d+"").length),f||(f=0),m=sn(u)||(p in Qn.units?Qn.units[p]:x),x!==m&&(d=zs(e,p,c,m)),this._pt=new In(this._pt,A?v:a,p,d,(_?to(d,_+f):f)-d,!A&&(m==="px"||p==="zIndex")&&t.autoRound!==!1?z1:bf),this._pt.u=m||0,A&&y!==u?(this._pt.b=c,this._pt.e=y,this._pt.r=N1):x!==m&&m!=="%"&&(this._pt.b=c,this._pt.r=O1);else if(p in a)K1.call(this,e,p,c,_?_+u:u);else if(p in e)this.add(e,p,c||e[p],_?_+u:u,s,r);else if(p!=="parseTransform"){Cd(p,u);continue}A||(p in a?E.push(p,0,a[p]):typeof e[p]=="function"?E.push(p,2,e[p]()):E.push(p,1,c||e[p])),o.push(p)}}M&&Qg(this)},render:function(e,t){if(t.tween._time||!Ud())for(var n=t._pt;n;)n.r(e,n.d),n=n._next;else t.styles.revert()},get:Zi,aliases:Ii,getSetter:function(e,t,n){var s=Ii[t];return s&&s.indexOf(",")<0&&(t=s),t in cs&&t!==Dn&&(e._gsap.x||Zi(e,"x"))?n&&om===n?t==="scale"?G1:V1:(om=n||{})&&(t==="scale"?W1:X1):e.style&&!yd(e.style[t])?k1:~t.indexOf("-")?H1:Fd(e,t)},core:{_removeProperty:Ns,_getMatrix:Nd}};Pn.utils.checkPrefix=So;Pn.core.getStyleSaver=Zg;(function(i,e,t,n){var s=Rn(i+","+e+","+t,function(r){cs[r]=1});Rn(e,function(r){Qn.units[r]="deg",nx[r]=1}),Ii[s[13]]=i+","+e,Rn(n,function(r){var o=r.split(":");Ii[o[1]]=s[o[0]]})})("x,y,z,scale,scaleX,scaleY,xPercent,yPercent","rotation,rotationX,rotationY,skewX,skewY","transform,transformOrigin,svgOrigin,force3D,smoothOrigin,transformPerspective","0:translateX,1:translateY,2:translateZ,8:rotate,8:rotationZ,8:rotateZ,9:rotateX,10:rotateY");Rn("x,y,z,top,right,bottom,left,width,height,fontSize,padding,margin,perspective",function(i){Qn.units[i]="px"});Pn.registerPlugin(rx);var Hr=Pn.registerPlugin(rx)||Pn;Hr.core.Tween;const nw=(i,e)=>{const t=i.__vccOpts||i;for(const[n,s]of e)t[n]=s;return t},iw={key:0,class:"loading-overlay"},sw={key:1,class:"error-overlay"},rw={class:"error-msg"},ow={class:"controls-ui"},aw={class:"search-panel"},lw={key:2,class:"camera-track"},cw=["onClick"],uw=["src"],fw={key:1,class:"camera-tag-overlay"},dw={class:"camera-title-mini"},hw={class:"camera-tag-text"},pw={key:2},mw=["src"],gw={key:0,class:"ref-info"},xw={class:"info-tag"},_w={class:"info-tag"},Aw={class:"info-tag"},Sw={__name:"GaussianViewer",setup(i){const e=Bn(null),t=Bn(!1),n=Bn(!1),s=Bn(!1),r=Bn(!1),o=Bn([]),a=Bn(""),l=Bn(""),c=Bn({}),u=Bn({x:0,y:0,z:0}),f=Bn({x:0,y:0,z:0}),d=v0(()=>{if(!a.value.trim())return o.value.filter(Z=>Z.tag);const X=a.value.trim().toLowerCase();return o.value.filter(Z=>Z.tag&&Z.tag.toLowerCase().includes(X))}),h=()=>{d.value.length>0?M(d.value[0]):alert("场景中没有找到符合该描述的视角哦~")};let g,p;const x=Bn({x:0,y:0}),m=()=>{if(!g||!g.camera)return;const X=new _i().setFromQuaternion(g.camera.quaternion,"YXZ");u.value={x:(X.x*180/Math.PI).toFixed(1),y:(X.y*180/Math.PI).toFixed(1),z:(X.z*180/Math.PI).toFixed(1)}},_={FLY_IN:0,DIFFUSION:1,COLORING:2,FINISHED:3},A={isLoaded:!1,lastFrameTime:0,phase:_.FLY_IN,flyDuration:1.5,diffusionDuration:1,colorDuration:4},S={uTime:{value:0},uCenter:{value:new L(0,0,0)},uGeoRadius:{value:0},uColorRadius:{value:0},uMaxRadius:{value:50},uParticleProgress:{value:0}},v=X=>{if(!g)return;const Z=X.getSplatCount();X.updateMatrixWorld();let pe=1/0,ge=1/0,be=1/0,Me=-1/0,Y=-1/0,oe=-1/0;const he=new L,Fe=Math.max(1,Math.floor(Z/1e3));for(let R=0;R<Z;R+=Fe)X.getSplatCenter(R,he),he.applyMatrix4(X.matrixWorld),he.x<pe&&(pe=he.x),he.x>Me&&(Me=he.x),he.y<ge&&(ge=he.y),he.y>Y&&(Y=he.y),he.z<be&&(be=he.z),he.z>oe&&(oe=he.z);const ye=(pe+Me)/2,ke=(ge+Y)/2,F=(be+oe)/2,B=Math.max(Me-pe,Y-ge,oe-be);S.uCenter.value.set(ye,ke,F),S.uMaxRadius.value=B*.7;let q=6e4;Z<4e4?q=Z:Z>1e6&&(q=4e5);const w=Math.ceil(Z/q);let ne=B/200*window.devicePixelRatio;ne<.5&&(ne=.5);const re=B*1;console.log(`[Adaptive] MaxDim: ${B.toFixed(2)}, Particles: ~${Math.floor(Z/w)}, Size: ${ne.toFixed(2)}`);const ue=new An,ie=[],fe=[],te=[];for(let R=0;R<Z;R+=w){X.getSplatCenter(R,he),he.applyMatrix4(X.matrixWorld),fe.push(he.x,he.y,he.z);const T=re+Math.random()*(B*.5),V=Math.random()*Math.PI*2,se=Math.acos(2*Math.random()-1),ce=ye+T*Math.sin(se)*Math.cos(V),J=ke+T*Math.sin(se)*Math.sin(V),Pe=F+T*Math.cos(se);ie.push(ce,J,Pe),te.push(Math.random())}ue.setAttribute("position",new on(ie,3)),ue.setAttribute("aTarget",new on(fe,3)),ue.setAttribute("aRandom",new on(te,1));const xe=new _n({uniforms:{uProgress:S.uParticleProgress,uSize:{value:ne},uColor:{value:new nt(.6,.6,.6)}},vertexShader:`
      uniform float uProgress;
      uniform float uSize;
      attribute vec3 aTarget;
      attribute float aRandom;
      
      float easeOutCubic(float x) { return 1.0 - pow(1.0 - x, 3.0); }
      
      void main() {
        float t = (uProgress - aRandom * 0.1) / 0.9;
        t = clamp(t, 0.0, 1.0);
        vec3 pos = mix(position, aTarget, easeOutCubic(t));
        
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        
        // 距离衰减 (20.0 是透视缩放因子，配合世界单位的 uSize 使用)
        gl_PointSize = uSize * (20.0 / -mvPosition.z);
        if(gl_PointSize < 1.0) gl_PointSize = 1.0;
      }
    `,fragmentShader:`
      uniform vec3 uColor;
      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if(length(coord) > 0.5) discard;
        gl_FragColor = vec4(uColor, 1.0);
      }
    `,transparent:!0,opacity:1,depthTest:!0,depthWrite:!1});p=new fv(ue,xe),p.frustumCulled=!1,g.threeScene.add(p)},b=X=>{if(!X||!X.material)return;const Z=X.material;Z.uniforms=Z.uniforms||{},Z.uniforms.uGeoRadius=S.uGeoRadius,Z.uniforms.uColorRadius=S.uColorRadius,Z.uniforms.uMaxRadius=S.uMaxRadius,Z.uniforms.uCenter=S.uCenter,Z.vertexShader=`varying vec3 vWorldPosition;
`+Z.vertexShader;const pe=Z.vertexShader.lastIndexOf("}");if(pe!==-1){const Me=`vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
`;Z.vertexShader=Z.vertexShader.substring(0,pe)+Me+"}"}const ge=`
    uniform float uGeoRadius;
    uniform float uColorRadius;
    uniform float uMaxRadius;
    uniform vec3 uCenter;
    varying vec3 vWorldPosition;
  `;Z.fragmentShader=ge+Z.fragmentShader;const be=Z.fragmentShader.lastIndexOf("}");if(be!==-1){const Me=Z.fragmentShader.substring(0,be),Y=`
      float distFromCenter = distance(vWorldPosition, uCenter);
      
      if (distFromCenter > uGeoRadius) {
          discard;
      }
      if (distFromCenter > uColorRadius) {
          if (gl_FragColor.a < 0.8) discard; 
          gl_FragColor.a = 1.0; 
          gl_FragColor.rgb = vec3(0.6, 0.6, 0.6);
      } 
    `;Z.fragmentShader=Me+Y+"}"}Z.needsUpdate=!0},M=X=>{if(!g||!g.camera)return;const Z=g.camera,pe=g.getSplatMesh();l.value=X.image_url;const ge=new qe().fromArray(X.matrix),be=new qe;pe?(pe.updateMatrixWorld(),be.copy(pe.matrixWorld).multiply(ge)):be.copy(ge);const Me=new L,Y=new bt,oe=new L;be.decompose(Me,Y,oe);const he=X.fl_y||c.value.fl_y,Fe=X.h||c.value.h;if(he&&Fe){const w=2*Math.atan(Fe/2/he)*(180/Math.PI);Hr.to(Z,{fov:w,duration:1.5,ease:"power3.inOut",onUpdate:()=>Z.updateProjectionMatrix()})}Z.near>.001&&(Z.near=.001,Z.updateProjectionMatrix());const ye=new L(0,0,-1).applyQuaternion(Y),ke=Me.clone().add(ye.multiplyScalar(5));n.value=!1,g.controls&&(g.controls.enabled=!1);const F=Z.position.clone(),B=Z.quaternion.clone(),q={t:0};Hr.killTweensOf(Z.position),Hr.killTweensOf(Z.quaternion),Hr.killTweensOf(q),Hr.to(q,{t:1,duration:1.5,ease:"power3.inOut",onUpdate:()=>{Z.position.lerpVectors(F,Me,q.t),Z.quaternion.slerpQuaternions(B,Y,q.t)},onComplete:()=>{const w=new _i().setFromQuaternion(Z.quaternion,"YXZ");f.value={x:(w.x*180/Math.PI).toFixed(1),y:(w.y*180/Math.PI).toFixed(1),z:(w.z*180/Math.PI).toFixed(1)},x.value={x:0,y:0},m(),g.controls&&(g.controls.target.copy(ke),g.controls.update(),g.controls.enabled=!0)}})},E=()=>{const X=/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);return{rootElement:e.value,cameraUp:[0,1,0],initialCameraPosition:[0,0,5],initialCameraLookAt:[0,0,0],useBuiltInControls:!1,gpuAcceleratedSort:!1,webXRMode:r.value?ys.VR:ys.None,sharedMemoryForWorkers:!1,antialiased:!X}},y=async(X="./models/scene_auto_sync_raw.ply")=>{if(!s.value){s.value=!0;try{g&&(g.renderer.setAnimationLoop(null),g.dispose&&await g.dispose(),g=null),e.value&&(e.value.innerHTML=""),A.isLoaded=!1,A.phase=_.FLY_IN,S.uParticleProgress.value=0,S.uGeoRadius.value=0,S.uColorRadius.value=0;const Z=E();g=new Vr(Z),window.viewer=g,await g.addSplatScene(X,{showLoadingUI:!0,progressiveLoad:!1,rotation:[0,0,0,1]}),window.BrainDanceChannel&&window.BrainDanceChannel.postMessage(JSON.stringify({status:"success",msg:"模型加载完成"})),fetch("./models/webgl_poses_with_tags.json").then(ge=>ge.json()).then(ge=>{ge.frames?(c.value={w:ge.w,h:ge.h,fl_x:ge.fl_x,fl_y:ge.fl_y},o.value=ge.frames.map(be=>({id:be.id,matrix:be.matrix,image_url:be.image_url,tag:be.tag}))):o.value=ge}).catch(ge=>console.error("加载位姿失败:",ge));const pe=g.getSplatMesh();pe.visible=!1,setTimeout(()=>{pe&&(v(pe),b(pe),I(),A.lastFrameTime=Date.now(),A.startTime=Date.now(),A.isLoaded=!0)},200),g.renderer.setAnimationLoop(()=>{if(g.update(),g.render(),!A.isLoaded||A.phase===_.FINISHED)return;const ge=Date.now(),be=(ge-A.lastFrameTime)/1e3||.016;if(A.lastFrameTime=ge,A.phase===_.FLY_IN){const Me=1/A.flyDuration;let Y=S.uParticleProgress.value+be*Me;if(Y>=1.2){Y=1.2;const oe=g.getSplatMesh();oe&&(oe.visible=!0),A.phase=_.DIFFUSION,A.diffuseTime=0}S.uParticleProgress.value=Y}else if(A.phase===_.DIFFUSION){A.diffuseTime+=be;const Me=Math.min(A.diffuseTime/A.diffusionDuration,1),Y=S.uMaxRadius.value;S.uGeoRadius.value=Me*(Y*1.5),p&&p.material&&(p.material.opacity=1-Me),Me>=1&&(p&&(p.visible=!1),S.uGeoRadius.value=99999,A.phase=_.COLORING,A.colorStartTime=ge)}else if(A.phase===_.COLORING){const Me=(ge-A.colorStartTime)/1e3,Y=S.uMaxRadius.value,oe=Me/A.colorDuration;S.uColorRadius.value=oe*(Y*1.5),oe>=1&&(A.phase=_.FINISHED,S.uColorRadius.value=99999)}}),C()}catch(Z){console.error("error:",Z),loadError.value=Z&&(Z.message||String(Z))||"模型加载失败，请检查模型 URL 是否正确可访问"}}},C=()=>{g&&(g.controls&&(g.controls.dispose(),g.controls=null),console.log("Controls explicitly disabled for debugging"))},I=()=>{if(t.value)return;const X=S.uCenter.value,pe=S.uMaxRadius.value/.7*2;g.controls&&(g.controls.target.copy(X),g.controls.update()),g.camera.position.set(X.x,X.y,X.z+pe),g.camera.lookAt(X)},P=async()=>{if(!r.value){alert("需HTTPS");return}t.value?(g.xr&&g.xr.exitVR(),t.value=!1):(g.xr&&g.xr.enterVR(),t.value=!0)},U=()=>{n.value=!n.value},O=()=>{const X=window.location.hostname==="localhost"||window.location.hostname==="127.0.0.1",Z=window.location.protocol==="https:";r.value=X||Z},k=Bn(!1),z={x:0,y:0},W=X=>{k.value=!0,z.x=X.clientX,z.y=X.clientY},H=X=>{if(!k.value||!g||!g.camera)return;const Z=X.clientX-z.x,pe=X.clientY-z.y,ge=.2,be=-Z*ge,Me=-pe*ge;x.value.x+=Me,x.value.y+=be,g.camera.rotateX(Me*Math.PI/180),g.camera.rotateOnWorldAxis(new L(0,1,0),be*Math.PI/180),g.camera.updateProjectionMatrix(),m(),z.x=X.clientX,z.y=X.clientY},$=()=>{k.value=!1};return jm(()=>{window.loadModelFromFlutter=X=>{console.log("准备加载模型: ",X),y(X)},e.value&&(O(),window.BrainDanceChannel?window.BrainDanceChannel.postMessage(JSON.stringify({status:"ready"})):y(),window.addEventListener("mousedown",W),window.addEventListener("mousemove",H),window.addEventListener("mouseup",$))}),$m(async()=>{delete window.loadModelFromFlutter,window.removeEventListener("mousemove",H),window.removeEventListener("mouseup",$),g&&(g.renderer.setAnimationLoop(null),await g.dispose())}),(X,Z)=>(Mn(),Nn("div",{class:"app-container",onMousedown:W,onMousemove:H,onMouseup:$,onMouseleave:$},[Ht("div",{ref_key:"containerRef",ref:e,class:"viewer-container"},null,512),s.value?(Mn(),Nn("div",iw,"正在加载模型...")):fs("",!0),X.loadError?(Mn(),Nn("div",sw,[Z[3]||(Z[3]=Ht("div",{class:"error-icon"},"⚠️",-1)),Z[4]||(Z[4]=Ht("div",{class:"error-title"},"模型加载失败",-1)),Ht("div",rw,hi(X.loadError),1),Ht("button",{class:"error-retry",onClick:Z[0]||(Z[0]=pe=>X.loadError="")},"关闭")])):fs("",!0),Ht("div",ow,[r.value?(Mn(),Nn("button",{key:0,onClick:P,class:Qr({active:t.value})},hi(t.value?"退出 VR":"进入 VR"),3)):fs("",!0),Ht("button",{onClick:U,class:Qr({active:n.value})},hi(n.value?"停止旋转":"自动旋转"),3)]),Ht("div",aw,[r_(Ht("input",{type:"text","onUpdate:modelValue":Z[1]||(Z[1]=pe=>a.value=pe),onKeyup:BA(h,["enter"]),placeholder:"搜索想要的视角 (如: 正面特写...)",class:"search-input"},null,544),[[FA,a.value]]),Ht("button",{onClick:h,class:"search-btn"},"🔍 搜索视角")]),d.value.length>0?(Mn(),Nn("div",lw,[(Mn(!0),Nn(bi,null,M_(d.value,(pe,ge)=>(Mn(),Nn("div",{key:pe.id,class:Qr(["camera-btn",{active:l.value===pe.image_url}]),onClick:be=>M(pe)},[pe.image_url?(Mn(),Nn("img",{key:0,src:pe.image_url,class:"btn-thumb"},null,8,uw)):fs("",!0),pe.tag?(Mn(),Nn("div",fw,[Ht("div",dw,"镜 "+hi(pe.id.split(".")[0].replace("frame_","")),1),Ht("div",hw,hi(pe.tag),1)])):pe.image_url?fs("",!0):(Mn(),Nn("span",pw,"镜头 "+hi(ge+1),1))],10,cw))),128))])):fs("",!0),l.value?(Mn(),Nn("div",{key:3,class:"reference-overlay",onClick:Z[2]||(Z[2]=pe=>l.value="")},[Z[5]||(Z[5]=Ht("div",{class:"ref-title"},"参考原图",-1)),Ht("img",{src:l.value,class:"ref-img"},null,8,mw),c.value.fl_y?(Mn(),Nn("div",gw,[Ht("span",xw,"焦距: "+hi(c.value.fl_y.toFixed(1))+" px",1),Ht("span",_w,"FOV: "+hi((2*Math.atan(c.value.h/(2*c.value.fl_y))*(180/Math.PI)).toFixed(1))+"°",1),Ht("span",Aw,"分辨率: "+hi(c.value.w)+"x"+hi(c.value.h),1)])):fs("",!0),Z[6]||(Z[6]=Ht("div",{class:"ref-hint"},"点击关闭对比",-1))])):fs("",!0)],32))}},vw=nw(Sw,[["__scopeId","data-v-ddaaebc5"]]),yw={__name:"App",setup(i){return(e,t)=>(Mn(),Nn("main",null,[Pi(vw)]))}};NA(yw).mount("#app");
