(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))n(s);new MutationObserver(s=>{for(const r of s)if(r.type==="childList")for(const o of r.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&n(o)}).observe(document,{childList:!0,subtree:!0});function t(s){const r={};return s.integrity&&(r.integrity=s.integrity),s.referrerPolicy&&(r.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?r.credentials="include":s.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function n(s){if(s.ep)return;s.ep=!0;const r=t(s);fetch(s.href,r)}})();function wf(i){const e=Object.create(null);for(const t of i.split(","))e[t]=1;return t=>t in e}const At={},qr=[],Di=()=>{},gm=()=>!1,Hl=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&(i.charCodeAt(2)>122||i.charCodeAt(2)<97),Rf=i=>i.startsWith("onUpdate:"),ln=Object.assign,If=(i,e)=>{const t=i.indexOf(e);t>-1&&i.splice(t,1)},px=Object.prototype.hasOwnProperty,ct=(i,e)=>px.call(i,e),Ke=Array.isArray,Qr=i=>Aa(i)==="[object Map]",xm=i=>Aa(i)==="[object Set]",Qd=i=>Aa(i)==="[object Date]",$e=i=>typeof i=="function",Xt=i=>typeof i=="string",Fi=i=>typeof i=="symbol",mt=i=>i!==null&&typeof i=="object",_m=i=>(mt(i)||$e(i))&&$e(i.then)&&$e(i.catch),Am=Object.prototype.toString,Aa=i=>Am.call(i),mx=i=>Aa(i).slice(8,-1),Sm=i=>Aa(i)==="[object Object]",Df=i=>Xt(i)&&i!=="NaN"&&i[0]!=="-"&&""+parseInt(i,10)===i,No=wf(",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"),Vl=i=>{const e=Object.create(null);return(t=>e[t]||(e[t]=i(t)))},gx=/-\w/g,Ls=Vl(i=>i.replace(gx,e=>e.slice(1).toUpperCase())),xx=/\B([A-Z])/g,ks=Vl(i=>i.replace(xx,"-$1").toLowerCase()),vm=Vl(i=>i.charAt(0).toUpperCase()+i.slice(1)),ac=Vl(i=>i?`on${vm(i)}`:""),Rs=(i,e)=>!Object.is(i,e),ll=(i,...e)=>{for(let t=0;t<i.length;t++)i[t](...e)},ym=(i,e,t,n=!1)=>{Object.defineProperty(i,e,{configurable:!0,enumerable:!1,writable:n,value:t})},Pf=i=>{const e=parseFloat(i);return isNaN(e)?i:e};let Yd;const Gl=()=>Yd||(Yd=typeof globalThis<"u"?globalThis:typeof self<"u"?self:typeof window<"u"?window:typeof global<"u"?global:{});function Ff(i){if(Ke(i)){const e={};for(let t=0;t<i.length;t++){const n=i[t],s=Xt(n)?vx(n):Ff(n);if(s)for(const r in s)e[r]=s[r]}return e}else if(Xt(i)||mt(i))return i}const _x=/;(?![^(]*\))/g,Ax=/:([^]+)/,Sx=/\/\*[^]*?\*\//g;function vx(i){const e={};return i.replace(Sx,"").split(_x).forEach(t=>{if(t){const n=t.split(Ax);n.length>1&&(e[n[0].trim()]=n[1].trim())}}),e}function Yr(i){let e="";if(Xt(i))e=i;else if(Ke(i))for(let t=0;t<i.length;t++){const n=Yr(i[t]);n&&(e+=n+" ")}else if(mt(i))for(const t in i)i[t]&&(e+=t+" ");return e.trim()}const yx="itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly",bx=wf(yx);function bm(i){return!!i||i===""}function Mx(i,e){if(i.length!==e.length)return!1;let t=!0;for(let n=0;t&&n<i.length;n++)t=Lf(i[n],e[n]);return t}function Lf(i,e){if(i===e)return!0;let t=Qd(i),n=Qd(e);if(t||n)return t&&n?i.getTime()===e.getTime():!1;if(t=Fi(i),n=Fi(e),t||n)return i===e;if(t=Ke(i),n=Ke(e),t||n)return t&&n?Mx(i,e):!1;if(t=mt(i),n=mt(e),t||n){if(!t||!n)return!1;const s=Object.keys(i).length,r=Object.keys(e).length;if(s!==r)return!1;for(const o in i){const a=i.hasOwnProperty(o),l=e.hasOwnProperty(o);if(a&&!l||!a&&l||!Lf(i[o],e[o]))return!1}}return String(i)===String(e)}const Mm=i=>!!(i&&i.__v_isRef===!0),hi=i=>Xt(i)?i:i==null?"":Ke(i)||mt(i)&&(i.toString===Am||!$e(i.toString))?Mm(i)?hi(i.value):JSON.stringify(i,Cm,2):String(i),Cm=(i,e)=>Mm(e)?Cm(i,e.value):Qr(e)?{[`Map(${e.size})`]:[...e.entries()].reduce((t,[n,s],r)=>(t[lc(n,r)+" =>"]=s,t),{})}:xm(e)?{[`Set(${e.size})`]:[...e.values()].map(t=>lc(t))}:Fi(e)?lc(e):mt(e)&&!Ke(e)&&!Sm(e)?String(e):e,lc=(i,e="")=>{var t;return Fi(i)?`Symbol(${(t=i.description)!=null?t:e})`:i};let Tn;class Cx{constructor(e=!1){this.detached=e,this._active=!0,this._on=0,this.effects=[],this.cleanups=[],this._isPaused=!1,this.__v_skip=!0,this.parent=Tn,!e&&Tn&&(this.index=(Tn.scopes||(Tn.scopes=[])).push(this)-1)}get active(){return this._active}pause(){if(this._active){this._isPaused=!0;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].pause();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].pause()}}resume(){if(this._active&&this._isPaused){this._isPaused=!1;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].resume();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].resume()}}run(e){if(this._active){const t=Tn;try{return Tn=this,e()}finally{Tn=t}}}on(){++this._on===1&&(this.prevScope=Tn,Tn=this)}off(){this._on>0&&--this._on===0&&(Tn=this.prevScope,this.prevScope=void 0)}stop(e){if(this._active){this._active=!1;let t,n;for(t=0,n=this.effects.length;t<n;t++)this.effects[t].stop();for(this.effects.length=0,t=0,n=this.cleanups.length;t<n;t++)this.cleanups[t]();if(this.cleanups.length=0,this.scopes){for(t=0,n=this.scopes.length;t<n;t++)this.scopes[t].stop(!0);this.scopes.length=0}if(!this.detached&&this.parent&&!e){const s=this.parent.scopes.pop();s&&s!==this&&(this.parent.scopes[this.index]=s,s.index=this.index)}this.parent=void 0}}}function Tx(){return Tn}let St;const cc=new WeakSet;class Tm{constructor(e){this.fn=e,this.deps=void 0,this.depsTail=void 0,this.flags=5,this.next=void 0,this.cleanup=void 0,this.scheduler=void 0,Tn&&Tn.active&&Tn.effects.push(this)}pause(){this.flags|=64}resume(){this.flags&64&&(this.flags&=-65,cc.has(this)&&(cc.delete(this),this.trigger()))}notify(){this.flags&2&&!(this.flags&32)||this.flags&8||wm(this)}run(){if(!(this.flags&1))return this.fn();this.flags|=2,Kd(this),Rm(this);const e=St,t=gi;St=this,gi=!0;try{return this.fn()}finally{Im(this),St=e,gi=t,this.flags&=-3}}stop(){if(this.flags&1){for(let e=this.deps;e;e=e.nextDep)Of(e);this.deps=this.depsTail=void 0,Kd(this),this.onStop&&this.onStop(),this.flags&=-2}}trigger(){this.flags&64?cc.add(this):this.scheduler?this.scheduler():this.runIfDirty()}runIfDirty(){du(this)&&this.run()}get dirty(){return du(this)}}let Em=0,zo,ko;function wm(i,e=!1){if(i.flags|=8,e){i.next=ko,ko=i;return}i.next=zo,zo=i}function Bf(){Em++}function Uf(){if(--Em>0)return;if(ko){let e=ko;for(ko=void 0;e;){const t=e.next;e.next=void 0,e.flags&=-9,e=t}}let i;for(;zo;){let e=zo;for(zo=void 0;e;){const t=e.next;if(e.next=void 0,e.flags&=-9,e.flags&1)try{e.trigger()}catch(n){i||(i=n)}e=t}}if(i)throw i}function Rm(i){for(let e=i.deps;e;e=e.nextDep)e.version=-1,e.prevActiveLink=e.dep.activeLink,e.dep.activeLink=e}function Im(i){let e,t=i.depsTail,n=t;for(;n;){const s=n.prevDep;n.version===-1?(n===t&&(t=s),Of(n),Ex(n)):e=n,n.dep.activeLink=n.prevActiveLink,n.prevActiveLink=void 0,n=s}i.deps=e,i.depsTail=t}function du(i){for(let e=i.deps;e;e=e.nextDep)if(e.dep.version!==e.version||e.dep.computed&&(Dm(e.dep.computed)||e.dep.version!==e.version))return!0;return!!i._dirty}function Dm(i){if(i.flags&4&&!(i.flags&16)||(i.flags&=-17,i.globalVersion===Jo)||(i.globalVersion=Jo,!i.isSSR&&i.flags&128&&(!i.deps&&!i._dirty||!du(i))))return;i.flags|=2;const e=i.dep,t=St,n=gi;St=i,gi=!0;try{Rm(i);const s=i.fn(i._value);(e.version===0||Rs(s,i._value))&&(i.flags|=128,i._value=s,e.version++)}catch(s){throw e.version++,s}finally{St=t,gi=n,Im(i),i.flags&=-3}}function Of(i,e=!1){const{dep:t,prevSub:n,nextSub:s}=i;if(n&&(n.nextSub=s,i.prevSub=void 0),s&&(s.prevSub=n,i.nextSub=void 0),t.subs===i&&(t.subs=n,!n&&t.computed)){t.computed.flags&=-5;for(let r=t.computed.deps;r;r=r.nextDep)Of(r,!0)}!e&&!--t.sc&&t.map&&t.map.delete(t.key)}function Ex(i){const{prevDep:e,nextDep:t}=i;e&&(e.nextDep=t,i.prevDep=void 0),t&&(t.prevDep=e,i.nextDep=void 0)}let gi=!0;const Pm=[];function rs(){Pm.push(gi),gi=!1}function os(){const i=Pm.pop();gi=i===void 0?!0:i}function Kd(i){const{cleanup:e}=i;if(i.cleanup=void 0,e){const t=St;St=void 0;try{e()}finally{St=t}}}let Jo=0;class wx{constructor(e,t){this.sub=e,this.dep=t,this.version=t.version,this.nextDep=this.prevDep=this.nextSub=this.prevSub=this.prevActiveLink=void 0}}class Nf{constructor(e){this.computed=e,this.version=0,this.activeLink=void 0,this.subs=void 0,this.map=void 0,this.key=void 0,this.sc=0,this.__v_skip=!0}track(e){if(!St||!gi||St===this.computed)return;let t=this.activeLink;if(t===void 0||t.sub!==St)t=this.activeLink=new wx(St,this),St.deps?(t.prevDep=St.depsTail,St.depsTail.nextDep=t,St.depsTail=t):St.deps=St.depsTail=t,Fm(t);else if(t.version===-1&&(t.version=this.version,t.nextDep)){const n=t.nextDep;n.prevDep=t.prevDep,t.prevDep&&(t.prevDep.nextDep=n),t.prevDep=St.depsTail,t.nextDep=void 0,St.depsTail.nextDep=t,St.depsTail=t,St.deps===t&&(St.deps=n)}return t}trigger(e){this.version++,Jo++,this.notify(e)}notify(e){Bf();try{for(let t=this.subs;t;t=t.prevSub)t.sub.notify()&&t.sub.dep.notify()}finally{Uf()}}}function Fm(i){if(i.dep.sc++,i.sub.flags&4){const e=i.dep.computed;if(e&&!i.dep.subs){e.flags|=20;for(let n=e.deps;n;n=n.nextDep)Fm(n)}const t=i.dep.subs;t!==i&&(i.prevSub=t,t&&(t.nextSub=i)),i.dep.subs=i}}const hu=new WeakMap,ar=Symbol(""),pu=Symbol(""),ea=Symbol("");function nn(i,e,t){if(gi&&St){let n=hu.get(i);n||hu.set(i,n=new Map);let s=n.get(t);s||(n.set(t,s=new Nf),s.map=n,s.key=t),s.track()}}function Ji(i,e,t,n,s,r){const o=hu.get(i);if(!o){Jo++;return}const a=l=>{l&&l.trigger()};if(Bf(),e==="clear")o.forEach(a);else{const l=Ke(i),c=l&&Df(t);if(l&&t==="length"){const u=Number(n);o.forEach((f,d)=>{(d==="length"||d===ea||!Fi(d)&&d>=u)&&a(f)})}else switch((t!==void 0||o.has(void 0))&&a(o.get(t)),c&&a(o.get(ea)),e){case"add":l?c&&a(o.get("length")):(a(o.get(ar)),Qr(i)&&a(o.get(pu)));break;case"delete":l||(a(o.get(ar)),Qr(i)&&a(o.get(pu)));break;case"set":Qr(i)&&a(o.get(ar));break}}Uf()}function _r(i){const e=lt(i);return e===i?e:(nn(e,"iterate",ea),ai(i)?e:e.map(xi))}function Wl(i){return nn(i=lt(i),"iterate",ea),i}function As(i,e){return as(i)?ro(lr(i)?xi(e):e):xi(e)}const Rx={__proto__:null,[Symbol.iterator](){return uc(this,Symbol.iterator,i=>As(this,i))},concat(...i){return _r(this).concat(...i.map(e=>Ke(e)?_r(e):e))},entries(){return uc(this,"entries",i=>(i[1]=As(this,i[1]),i))},every(i,e){return Ni(this,"every",i,e,void 0,arguments)},filter(i,e){return Ni(this,"filter",i,e,t=>t.map(n=>As(this,n)),arguments)},find(i,e){return Ni(this,"find",i,e,t=>As(this,t),arguments)},findIndex(i,e){return Ni(this,"findIndex",i,e,void 0,arguments)},findLast(i,e){return Ni(this,"findLast",i,e,t=>As(this,t),arguments)},findLastIndex(i,e){return Ni(this,"findLastIndex",i,e,void 0,arguments)},forEach(i,e){return Ni(this,"forEach",i,e,void 0,arguments)},includes(...i){return fc(this,"includes",i)},indexOf(...i){return fc(this,"indexOf",i)},join(i){return _r(this).join(i)},lastIndexOf(...i){return fc(this,"lastIndexOf",i)},map(i,e){return Ni(this,"map",i,e,void 0,arguments)},pop(){return Co(this,"pop")},push(...i){return Co(this,"push",i)},reduce(i,...e){return jd(this,"reduce",i,e)},reduceRight(i,...e){return jd(this,"reduceRight",i,e)},shift(){return Co(this,"shift")},some(i,e){return Ni(this,"some",i,e,void 0,arguments)},splice(...i){return Co(this,"splice",i)},toReversed(){return _r(this).toReversed()},toSorted(i){return _r(this).toSorted(i)},toSpliced(...i){return _r(this).toSpliced(...i)},unshift(...i){return Co(this,"unshift",i)},values(){return uc(this,"values",i=>As(this,i))}};function uc(i,e,t){const n=Wl(i),s=n[e]();return n!==i&&!ai(i)&&(s._next=s.next,s.next=()=>{const r=s._next();return r.done||(r.value=t(r.value)),r}),s}const Ix=Array.prototype;function Ni(i,e,t,n,s,r){const o=Wl(i),a=o!==i&&!ai(i),l=o[e];if(l!==Ix[e]){const f=l.apply(i,r);return a?xi(f):f}let c=t;o!==i&&(a?c=function(f,d){return t.call(this,As(i,f),d,i)}:t.length>2&&(c=function(f,d){return t.call(this,f,d,i)}));const u=l.call(o,c,n);return a&&s?s(u):u}function jd(i,e,t,n){const s=Wl(i);let r=t;return s!==i&&(ai(i)?t.length>3&&(r=function(o,a,l){return t.call(this,o,a,l,i)}):r=function(o,a,l){return t.call(this,o,As(i,a),l,i)}),s[e](r,...n)}function fc(i,e,t){const n=lt(i);nn(n,"iterate",ea);const s=n[e](...t);return(s===-1||s===!1)&&Vf(t[0])?(t[0]=lt(t[0]),n[e](...t)):s}function Co(i,e,t=[]){rs(),Bf();const n=lt(i)[e].apply(i,t);return Uf(),os(),n}const Dx=wf("__proto__,__v_isRef,__isVue"),Lm=new Set(Object.getOwnPropertyNames(Symbol).filter(i=>i!=="arguments"&&i!=="caller").map(i=>Symbol[i]).filter(Fi));function Px(i){Fi(i)||(i=String(i));const e=lt(this);return nn(e,"has",i),e.hasOwnProperty(i)}class Bm{constructor(e=!1,t=!1){this._isReadonly=e,this._isShallow=t}get(e,t,n){if(t==="__v_skip")return e.__v_skip;const s=this._isReadonly,r=this._isShallow;if(t==="__v_isReactive")return!s;if(t==="__v_isReadonly")return s;if(t==="__v_isShallow")return r;if(t==="__v_raw")return n===(s?r?Vx:zm:r?Nm:Om).get(e)||Object.getPrototypeOf(e)===Object.getPrototypeOf(n)?e:void 0;const o=Ke(e);if(!s){let l;if(o&&(l=Rx[t]))return l;if(t==="hasOwnProperty")return Px}const a=Reflect.get(e,t,rn(e)?e:n);if((Fi(t)?Lm.has(t):Dx(t))||(s||nn(e,"get",t),r))return a;if(rn(a)){const l=o&&Df(t)?a:a.value;return s&&mt(l)?gu(l):l}return mt(a)?s?gu(a):kf(a):a}}class Um extends Bm{constructor(e=!1){super(!1,e)}set(e,t,n,s){let r=e[t];const o=Ke(e)&&Df(t);if(!this._isShallow){const c=as(r);if(!ai(n)&&!as(n)&&(r=lt(r),n=lt(n)),!o&&rn(r)&&!rn(n))return c||(r.value=n),!0}const a=o?Number(t)<e.length:ct(e,t),l=Reflect.set(e,t,n,rn(e)?e:s);return e===lt(s)&&(a?Rs(n,r)&&Ji(e,"set",t,n):Ji(e,"add",t,n)),l}deleteProperty(e,t){const n=ct(e,t);e[t];const s=Reflect.deleteProperty(e,t);return s&&n&&Ji(e,"delete",t,void 0),s}has(e,t){const n=Reflect.has(e,t);return(!Fi(t)||!Lm.has(t))&&nn(e,"has",t),n}ownKeys(e){return nn(e,"iterate",Ke(e)?"length":ar),Reflect.ownKeys(e)}}class Fx extends Bm{constructor(e=!1){super(!0,e)}set(e,t){return!0}deleteProperty(e,t){return!0}}const Lx=new Um,Bx=new Fx,Ux=new Um(!0);const mu=i=>i,wa=i=>Reflect.getPrototypeOf(i);function Ox(i,e,t){return function(...n){const s=this.__v_raw,r=lt(s),o=Qr(r),a=i==="entries"||i===Symbol.iterator&&o,l=i==="keys"&&o,c=s[i](...n),u=t?mu:e?ro:xi;return!e&&nn(r,"iterate",l?pu:ar),ln(Object.create(c),{next(){const{value:f,done:d}=c.next();return d?{value:f,done:d}:{value:a?[u(f[0]),u(f[1])]:u(f),done:d}}})}}function Ra(i){return function(...e){return i==="delete"?!1:i==="clear"?void 0:this}}function Nx(i,e){const t={get(s){const r=this.__v_raw,o=lt(r),a=lt(s);i||(Rs(s,a)&&nn(o,"get",s),nn(o,"get",a));const{has:l}=wa(o),c=e?mu:i?ro:xi;if(l.call(o,s))return c(r.get(s));if(l.call(o,a))return c(r.get(a));r!==o&&r.get(s)},get size(){const s=this.__v_raw;return!i&&nn(lt(s),"iterate",ar),s.size},has(s){const r=this.__v_raw,o=lt(r),a=lt(s);return i||(Rs(s,a)&&nn(o,"has",s),nn(o,"has",a)),s===a?r.has(s):r.has(s)||r.has(a)},forEach(s,r){const o=this,a=o.__v_raw,l=lt(a),c=e?mu:i?ro:xi;return!i&&nn(l,"iterate",ar),a.forEach((u,f)=>s.call(r,c(u),c(f),o))}};return ln(t,i?{add:Ra("add"),set:Ra("set"),delete:Ra("delete"),clear:Ra("clear")}:{add(s){!e&&!ai(s)&&!as(s)&&(s=lt(s));const r=lt(this);return wa(r).has.call(r,s)||(r.add(s),Ji(r,"add",s,s)),this},set(s,r){!e&&!ai(r)&&!as(r)&&(r=lt(r));const o=lt(this),{has:a,get:l}=wa(o);let c=a.call(o,s);c||(s=lt(s),c=a.call(o,s));const u=l.call(o,s);return o.set(s,r),c?Rs(r,u)&&Ji(o,"set",s,r):Ji(o,"add",s,r),this},delete(s){const r=lt(this),{has:o,get:a}=wa(r);let l=o.call(r,s);l||(s=lt(s),l=o.call(r,s)),a&&a.call(r,s);const c=r.delete(s);return l&&Ji(r,"delete",s,void 0),c},clear(){const s=lt(this),r=s.size!==0,o=s.clear();return r&&Ji(s,"clear",void 0,void 0),o}}),["keys","values","entries",Symbol.iterator].forEach(s=>{t[s]=Ox(s,i,e)}),t}function zf(i,e){const t=Nx(i,e);return(n,s,r)=>s==="__v_isReactive"?!i:s==="__v_isReadonly"?i:s==="__v_raw"?n:Reflect.get(ct(t,s)&&s in n?t:n,s,r)}const zx={get:zf(!1,!1)},kx={get:zf(!1,!0)},Hx={get:zf(!0,!1)};const Om=new WeakMap,Nm=new WeakMap,zm=new WeakMap,Vx=new WeakMap;function Gx(i){switch(i){case"Object":case"Array":return 1;case"Map":case"Set":case"WeakMap":case"WeakSet":return 2;default:return 0}}function Wx(i){return i.__v_skip||!Object.isExtensible(i)?0:Gx(mx(i))}function kf(i){return as(i)?i:Hf(i,!1,Lx,zx,Om)}function Xx(i){return Hf(i,!1,Ux,kx,Nm)}function gu(i){return Hf(i,!0,Bx,Hx,zm)}function Hf(i,e,t,n,s){if(!mt(i)||i.__v_raw&&!(e&&i.__v_isReactive))return i;const r=Wx(i);if(r===0)return i;const o=s.get(i);if(o)return o;const a=new Proxy(i,r===2?n:t);return s.set(i,a),a}function lr(i){return as(i)?lr(i.__v_raw):!!(i&&i.__v_isReactive)}function as(i){return!!(i&&i.__v_isReadonly)}function ai(i){return!!(i&&i.__v_isShallow)}function Vf(i){return i?!!i.__v_raw:!1}function lt(i){const e=i&&i.__v_raw;return e?lt(e):i}function qx(i){return!ct(i,"__v_skip")&&Object.isExtensible(i)&&ym(i,"__v_skip",!0),i}const xi=i=>mt(i)?kf(i):i,ro=i=>mt(i)?gu(i):i;function rn(i){return i?i.__v_isRef===!0:!1}function yn(i){return Qx(i,!1)}function Qx(i,e){return rn(i)?i:new Yx(i,e)}class Yx{constructor(e,t){this.dep=new Nf,this.__v_isRef=!0,this.__v_isShallow=!1,this._rawValue=t?e:lt(e),this._value=t?e:xi(e),this.__v_isShallow=t}get value(){return this.dep.track(),this._value}set value(e){const t=this._rawValue,n=this.__v_isShallow||ai(e)||as(e);e=n?e:lt(e),Rs(e,t)&&(this._rawValue=e,this._value=n?e:xi(e),this.dep.trigger())}}function Kx(i){return rn(i)?i.value:i}const jx={get:(i,e,t)=>e==="__v_raw"?i:Kx(Reflect.get(i,e,t)),set:(i,e,t,n)=>{const s=i[e];return rn(s)&&!rn(t)?(s.value=t,!0):Reflect.set(i,e,t,n)}};function km(i){return lr(i)?i:new Proxy(i,jx)}class $x{constructor(e,t,n){this.fn=e,this.setter=t,this._value=void 0,this.dep=new Nf(this),this.__v_isRef=!0,this.deps=void 0,this.depsTail=void 0,this.flags=16,this.globalVersion=Jo-1,this.next=void 0,this.effect=this,this.__v_isReadonly=!t,this.isSSR=n}notify(){if(this.flags|=16,!(this.flags&8)&&St!==this)return wm(this,!0),!0}get value(){const e=this.dep.track();return Dm(this),e&&(e.version=this.dep.version),this._value}set value(e){this.setter&&this.setter(e)}}function Zx(i,e,t=!1){let n,s;return $e(i)?n=i:(n=i.get,s=i.set),new $x(n,s,t)}const Ia={},vl=new WeakMap;let er;function Jx(i,e=!1,t=er){if(t){let n=vl.get(t);n||vl.set(t,n=[]),n.push(i)}}function e_(i,e,t=At){const{immediate:n,deep:s,once:r,scheduler:o,augmentJob:a,call:l}=t,c=S=>s?S:ai(S)||s===!1||s===0?es(S,1):es(S);let u,f,d,h,x=!1,p=!1;if(rn(i)?(f=()=>i.value,x=ai(i)):lr(i)?(f=()=>c(i),x=!0):Ke(i)?(p=!0,x=i.some(S=>lr(S)||ai(S)),f=()=>i.map(S=>{if(rn(S))return S.value;if(lr(S))return c(S);if($e(S))return l?l(S,2):S()})):$e(i)?e?f=l?()=>l(i,2):i:f=()=>{if(d){rs();try{d()}finally{os()}}const S=er;er=u;try{return l?l(i,3,[h]):i(h)}finally{er=S}}:f=Di,e&&s){const S=f,v=s===!0?1/0:s;f=()=>es(S(),v)}const g=Tx(),m=()=>{u.stop(),g&&g.active&&If(g.effects,u)};if(r&&e){const S=e;e=(...v)=>{S(...v),m()}}let _=p?new Array(i.length).fill(Ia):Ia;const A=S=>{if(!(!(u.flags&1)||!u.dirty&&!S))if(e){const v=u.run();if(s||x||(p?v.some((y,M)=>Rs(y,_[M])):Rs(v,_))){d&&d();const y=er;er=u;try{const M=[v,_===Ia?void 0:p&&_[0]===Ia?[]:_,h];_=v,l?l(e,3,M):e(...M)}finally{er=y}}}else u.run()};return a&&a(A),u=new Tm(f),u.scheduler=o?()=>o(A,!1):A,h=S=>Jx(S,!1,u),d=u.onStop=()=>{const S=vl.get(u);if(S){if(l)l(S,4);else for(const v of S)v();vl.delete(u)}},e?n?A(!0):_=u.run():o?o(A.bind(null,!0),!0):u.run(),m.pause=u.pause.bind(u),m.resume=u.resume.bind(u),m.stop=m,m}function es(i,e=1/0,t){if(e<=0||!mt(i)||i.__v_skip||(t=t||new Map,(t.get(i)||0)>=e))return i;if(t.set(i,e),e--,rn(i))es(i.value,e,t);else if(Ke(i))for(let n=0;n<i.length;n++)es(i[n],e,t);else if(xm(i)||Qr(i))i.forEach(n=>{es(n,e,t)});else if(Sm(i)){for(const n in i)es(i[n],e,t);for(const n of Object.getOwnPropertySymbols(i))Object.prototype.propertyIsEnumerable.call(i,n)&&es(i[n],e,t)}return i}function Sa(i,e,t,n){try{return n?i(...n):i()}catch(s){Xl(s,e,t)}}function Li(i,e,t,n){if($e(i)){const s=Sa(i,e,t,n);return s&&_m(s)&&s.catch(r=>{Xl(r,e,t)}),s}if(Ke(i)){const s=[];for(let r=0;r<i.length;r++)s.push(Li(i[r],e,t,n));return s}}function Xl(i,e,t,n=!0){const s=e?e.vnode:null,{errorHandler:r,throwUnhandledErrorInProduction:o}=e&&e.appContext.config||At;if(e){let a=e.parent;const l=e.proxy,c=`https://vuejs.org/error-reference/#runtime-${t}`;for(;a;){const u=a.ec;if(u){for(let f=0;f<u.length;f++)if(u[f](i,l,c)===!1)return}a=a.parent}if(r){rs(),Sa(r,null,10,[i,l,c]),os();return}}t_(i,t,s,n,o)}function t_(i,e,t,n=!0,s=!1){if(s)throw i;console.error(i)}const dn=[];let vi=-1;const Kr=[];let Ss=null,zr=0;const Hm=Promise.resolve();let yl=null;function n_(i){const e=yl||Hm;return i?e.then(this?i.bind(this):i):e}function i_(i){let e=vi+1,t=dn.length;for(;e<t;){const n=e+t>>>1,s=dn[n],r=ta(s);r<i||r===i&&s.flags&2?e=n+1:t=n}return e}function Gf(i){if(!(i.flags&1)){const e=ta(i),t=dn[dn.length-1];!t||!(i.flags&2)&&e>=ta(t)?dn.push(i):dn.splice(i_(e),0,i),i.flags|=1,Vm()}}function Vm(){yl||(yl=Hm.then(Wm))}function s_(i){Ke(i)?Kr.push(...i):Ss&&i.id===-1?Ss.splice(zr+1,0,i):i.flags&1||(Kr.push(i),i.flags|=1),Vm()}function $d(i,e,t=vi+1){for(;t<dn.length;t++){const n=dn[t];if(n&&n.flags&2){if(i&&n.id!==i.uid)continue;dn.splice(t,1),t--,n.flags&4&&(n.flags&=-2),n(),n.flags&4||(n.flags&=-2)}}}function Gm(i){if(Kr.length){const e=[...new Set(Kr)].sort((t,n)=>ta(t)-ta(n));if(Kr.length=0,Ss){Ss.push(...e);return}for(Ss=e,zr=0;zr<Ss.length;zr++){const t=Ss[zr];t.flags&4&&(t.flags&=-2),t.flags&8||t(),t.flags&=-2}Ss=null,zr=0}}const ta=i=>i.id==null?i.flags&2?-1:1/0:i.id;function Wm(i){try{for(vi=0;vi<dn.length;vi++){const e=dn[vi];e&&!(e.flags&8)&&(e.flags&4&&(e.flags&=-2),Sa(e,e.i,e.i?15:14),e.flags&4||(e.flags&=-2))}}finally{for(;vi<dn.length;vi++){const e=dn[vi];e&&(e.flags&=-2)}vi=-1,dn.length=0,Gm(),yl=null,(dn.length||Kr.length)&&Wm()}}let ni=null,Xm=null;function bl(i){const e=ni;return ni=i,Xm=i&&i.type.__scopeId||null,e}function r_(i,e=ni,t){if(!e||i._n)return i;const n=(...s)=>{n._d&&lh(-1);const r=bl(e);let o;try{o=i(...s)}finally{bl(r),n._d&&lh(1)}return o};return n._n=!0,n._c=!0,n._d=!0,n}function o_(i,e){if(ni===null)return i;const t=Kl(ni),n=i.dirs||(i.dirs=[]);for(let s=0;s<e.length;s++){let[r,o,a,l=At]=e[s];r&&($e(r)&&(r={mounted:r,updated:r}),r.deep&&es(o),n.push({dir:r,instance:t,value:o,oldValue:void 0,arg:a,modifiers:l}))}return i}function Ws(i,e,t,n){const s=i.dirs,r=e&&e.dirs;for(let o=0;o<s.length;o++){const a=s[o];r&&(a.oldValue=r[o].value);let l=a.dir[n];l&&(rs(),Li(l,t,8,[i.el,a,i,e]),os())}}function a_(i,e){if(pn){let t=pn.provides;const n=pn.parent&&pn.parent.provides;n===t&&(t=pn.provides=Object.create(n)),t[i]=e}}function cl(i,e,t=!1){const n=aA();if(n||jr){let s=jr?jr._context.provides:n?n.parent==null||n.ce?n.vnode.appContext&&n.vnode.appContext.provides:n.parent.provides:void 0;if(s&&i in s)return s[i];if(arguments.length>1)return t&&$e(e)?e.call(n&&n.proxy):e}}const l_=Symbol.for("v-scx"),c_=()=>cl(l_);function dc(i,e,t){return qm(i,e,t)}function qm(i,e,t=At){const{immediate:n,deep:s,flush:r,once:o}=t,a=ln({},t),l=e&&n||!e&&r!=="post";let c;if(ia){if(r==="sync"){const h=c_();c=h.__watcherHandles||(h.__watcherHandles=[])}else if(!l){const h=()=>{};return h.stop=Di,h.resume=Di,h.pause=Di,h}}const u=pn;a.call=(h,x,p)=>Li(h,u,x,p);let f=!1;r==="post"?a.scheduler=h=>{Mn(h,u&&u.suspense)}:r!=="sync"&&(f=!0,a.scheduler=(h,x)=>{x?h():Gf(h)}),a.augmentJob=h=>{e&&(h.flags|=4),f&&(h.flags|=2,u&&(h.id=u.uid,h.i=u))};const d=e_(i,e,a);return ia&&(c?c.push(d):l&&d()),d}function u_(i,e,t){const n=this.proxy,s=Xt(i)?i.includes(".")?Qm(n,i):()=>n[i]:i.bind(n,n);let r;$e(e)?r=e:(r=e.handler,t=e);const o=va(this),a=qm(s,r.bind(n),t);return o(),a}function Qm(i,e){const t=e.split(".");return()=>{let n=i;for(let s=0;s<t.length&&n;s++)n=n[t[s]];return n}}const f_=Symbol("_vte"),d_=i=>i.__isTeleport,h_=Symbol("_leaveCb");function Wf(i,e){i.shapeFlag&6&&i.component?(i.transition=e,Wf(i.component.subTree,e)):i.shapeFlag&128?(i.ssContent.transition=e.clone(i.ssContent),i.ssFallback.transition=e.clone(i.ssFallback)):i.transition=e}function Ym(i){i.ids=[i.ids[0]+i.ids[2]+++"-",0,0]}function Zd(i,e){let t;return!!((t=Object.getOwnPropertyDescriptor(i,e))&&!t.configurable)}const Ml=new WeakMap;function Ho(i,e,t,n,s=!1){if(Ke(i)){i.forEach((p,g)=>Ho(p,e&&(Ke(e)?e[g]:e),t,n,s));return}if(Vo(n)&&!s){n.shapeFlag&512&&n.type.__asyncResolved&&n.component.subTree.component&&Ho(i,e,t,n.component.subTree);return}const r=n.shapeFlag&4?Kl(n.component):n.el,o=s?null:r,{i:a,r:l}=i,c=e&&e.r,u=a.refs===At?a.refs={}:a.refs,f=a.setupState,d=lt(f),h=f===At?gm:p=>Zd(u,p)?!1:ct(d,p),x=(p,g)=>!(g&&Zd(u,g));if(c!=null&&c!==l){if(Jd(e),Xt(c))u[c]=null,h(c)&&(f[c]=null);else if(rn(c)){const p=e;x(c,p.k)&&(c.value=null),p.k&&(u[p.k]=null)}}if($e(l))Sa(l,a,12,[o,u]);else{const p=Xt(l),g=rn(l);if(p||g){const m=()=>{if(i.f){const _=p?h(l)?f[l]:u[l]:x()||!i.k?l.value:u[i.k];if(s)Ke(_)&&If(_,r);else if(Ke(_))_.includes(r)||_.push(r);else if(p)u[l]=[r],h(l)&&(f[l]=u[l]);else{const A=[r];x(l,i.k)&&(l.value=A),i.k&&(u[i.k]=A)}}else p?(u[l]=o,h(l)&&(f[l]=o)):g&&(x(l,i.k)&&(l.value=o),i.k&&(u[i.k]=o))};if(o){const _=()=>{m(),Ml.delete(i)};_.id=-1,Ml.set(i,_),Mn(_,t)}else Jd(i),m()}}}function Jd(i){const e=Ml.get(i);e&&(e.flags|=8,Ml.delete(i))}Gl().requestIdleCallback;Gl().cancelIdleCallback;const Vo=i=>!!i.type.__asyncLoader,Km=i=>i.type.__isKeepAlive;function p_(i,e){jm(i,"a",e)}function m_(i,e){jm(i,"da",e)}function jm(i,e,t=pn){const n=i.__wdc||(i.__wdc=()=>{let s=t;for(;s;){if(s.isDeactivated)return;s=s.parent}return i()});if(ql(e,n,t),t){let s=t.parent;for(;s&&s.parent;)Km(s.parent.vnode)&&g_(n,e,t,s),s=s.parent}}function g_(i,e,t,n){const s=ql(e,i,n,!0);Jm(()=>{If(n[e],s)},t)}function ql(i,e,t=pn,n=!1){if(t){const s=t[i]||(t[i]=[]),r=e.__weh||(e.__weh=(...o)=>{rs();const a=va(t),l=Li(e,t,i,o);return a(),os(),l});return n?s.unshift(r):s.push(r),r}}const us=i=>(e,t=pn)=>{(!ia||i==="sp")&&ql(i,(...n)=>e(...n),t)},x_=us("bm"),$m=us("m"),__=us("bu"),A_=us("u"),Zm=us("bum"),Jm=us("um"),S_=us("sp"),v_=us("rtg"),y_=us("rtc");function b_(i,e=pn){ql("ec",i,e)}const M_=Symbol.for("v-ndc");function C_(i,e,t,n){let s;const r=t,o=Ke(i);if(o||Xt(i)){const a=o&&lr(i);let l=!1,c=!1;a&&(l=!ai(i),c=as(i),i=Wl(i)),s=new Array(i.length);for(let u=0,f=i.length;u<f;u++)s[u]=e(l?c?ro(xi(i[u])):xi(i[u]):i[u],u,void 0,r)}else if(typeof i=="number"){s=new Array(i);for(let a=0;a<i;a++)s[a]=e(a+1,a,void 0,r)}else if(mt(i))if(i[Symbol.iterator])s=Array.from(i,(a,l)=>e(a,l,void 0,r));else{const a=Object.keys(i);s=new Array(a.length);for(let l=0,c=a.length;l<c;l++){const u=a[l];s[l]=e(i[u],u,l,r)}}else s=[];return s}const xu=i=>i?S0(i)?Kl(i):xu(i.parent):null,Go=ln(Object.create(null),{$:i=>i,$el:i=>i.vnode.el,$data:i=>i.data,$props:i=>i.props,$attrs:i=>i.attrs,$slots:i=>i.slots,$refs:i=>i.refs,$parent:i=>xu(i.parent),$root:i=>xu(i.root),$host:i=>i.ce,$emit:i=>i.emit,$options:i=>t0(i),$forceUpdate:i=>i.f||(i.f=()=>{Gf(i.update)}),$nextTick:i=>i.n||(i.n=n_.bind(i.proxy)),$watch:i=>u_.bind(i)}),hc=(i,e)=>i!==At&&!i.__isScriptSetup&&ct(i,e),T_={get({_:i},e){if(e==="__v_skip")return!0;const{ctx:t,setupState:n,data:s,props:r,accessCache:o,type:a,appContext:l}=i;if(e[0]!=="$"){const d=o[e];if(d!==void 0)switch(d){case 1:return n[e];case 2:return s[e];case 4:return t[e];case 3:return r[e]}else{if(hc(n,e))return o[e]=1,n[e];if(s!==At&&ct(s,e))return o[e]=2,s[e];if(ct(r,e))return o[e]=3,r[e];if(t!==At&&ct(t,e))return o[e]=4,t[e];_u&&(o[e]=0)}}const c=Go[e];let u,f;if(c)return e==="$attrs"&&nn(i.attrs,"get",""),c(i);if((u=a.__cssModules)&&(u=u[e]))return u;if(t!==At&&ct(t,e))return o[e]=4,t[e];if(f=l.config.globalProperties,ct(f,e))return f[e]},set({_:i},e,t){const{data:n,setupState:s,ctx:r}=i;return hc(s,e)?(s[e]=t,!0):n!==At&&ct(n,e)?(n[e]=t,!0):ct(i.props,e)||e[0]==="$"&&e.slice(1)in i?!1:(r[e]=t,!0)},has({_:{data:i,setupState:e,accessCache:t,ctx:n,appContext:s,props:r,type:o}},a){let l;return!!(t[a]||i!==At&&a[0]!=="$"&&ct(i,a)||hc(e,a)||ct(r,a)||ct(n,a)||ct(Go,a)||ct(s.config.globalProperties,a)||(l=o.__cssModules)&&l[a])},defineProperty(i,e,t){return t.get!=null?i._.accessCache[e]=0:ct(t,"value")&&this.set(i,e,t.value,null),Reflect.defineProperty(i,e,t)}};function eh(i){return Ke(i)?i.reduce((e,t)=>(e[t]=null,e),{}):i}let _u=!0;function E_(i){const e=t0(i),t=i.proxy,n=i.ctx;_u=!1,e.beforeCreate&&th(e.beforeCreate,i,"bc");const{data:s,computed:r,methods:o,watch:a,provide:l,inject:c,created:u,beforeMount:f,mounted:d,beforeUpdate:h,updated:x,activated:p,deactivated:g,beforeDestroy:m,beforeUnmount:_,destroyed:A,unmounted:S,render:v,renderTracked:y,renderTriggered:M,errorCaptured:E,serverPrefetch:b,expose:C,inheritAttrs:I,components:F,directives:U,filters:O}=e;if(c&&w_(c,n,null),o)for(const V in o){const H=o[V];$e(H)&&(n[V]=H.bind(t))}if(s){const V=s.call(t,t);mt(V)&&(i.data=kf(V))}if(_u=!0,r)for(const V in r){const H=r[V],$=$e(H)?H.bind(t,t):$e(H.get)?H.get.bind(t,t):Di,oe=!$e(H)&&$e(H.set)?H.set.bind(t):Di,Se=y0({get:$,set:oe});Object.defineProperty(n,V,{enumerable:!0,configurable:!0,get:()=>Se.value,set:we=>Se.value=we})}if(a)for(const V in a)e0(a[V],n,t,V);if(l){const V=$e(l)?l.call(t):l;Reflect.ownKeys(V).forEach(H=>{a_(H,V[H])})}u&&th(u,i,"c");function z(V,H){Ke(H)?H.forEach($=>V($.bind(t))):H&&V(H.bind(t))}if(z(x_,f),z($m,d),z(__,h),z(A_,x),z(p_,p),z(m_,g),z(b_,E),z(y_,y),z(v_,M),z(Zm,_),z(Jm,S),z(S_,b),Ke(C))if(C.length){const V=i.exposed||(i.exposed={});C.forEach(H=>{Object.defineProperty(V,H,{get:()=>t[H],set:$=>t[H]=$,enumerable:!0})})}else i.exposed||(i.exposed={});v&&i.render===Di&&(i.render=v),I!=null&&(i.inheritAttrs=I),F&&(i.components=F),U&&(i.directives=U),b&&Ym(i)}function w_(i,e,t=Di){Ke(i)&&(i=Au(i));for(const n in i){const s=i[n];let r;mt(s)?"default"in s?r=cl(s.from||n,s.default,!0):r=cl(s.from||n):r=cl(s),rn(r)?Object.defineProperty(e,n,{enumerable:!0,configurable:!0,get:()=>r.value,set:o=>r.value=o}):e[n]=r}}function th(i,e,t){Li(Ke(i)?i.map(n=>n.bind(e.proxy)):i.bind(e.proxy),e,t)}function e0(i,e,t,n){let s=n.includes(".")?Qm(t,n):()=>t[n];if(Xt(i)){const r=e[i];$e(r)&&dc(s,r)}else if($e(i))dc(s,i.bind(t));else if(mt(i))if(Ke(i))i.forEach(r=>e0(r,e,t,n));else{const r=$e(i.handler)?i.handler.bind(t):e[i.handler];$e(r)&&dc(s,r,i)}}function t0(i){const e=i.type,{mixins:t,extends:n}=e,{mixins:s,optionsCache:r,config:{optionMergeStrategies:o}}=i.appContext,a=r.get(e);let l;return a?l=a:!s.length&&!t&&!n?l=e:(l={},s.length&&s.forEach(c=>Cl(l,c,o,!0)),Cl(l,e,o)),mt(e)&&r.set(e,l),l}function Cl(i,e,t,n=!1){const{mixins:s,extends:r}=e;r&&Cl(i,r,t,!0),s&&s.forEach(o=>Cl(i,o,t,!0));for(const o in e)if(!(n&&o==="expose")){const a=R_[o]||t&&t[o];i[o]=a?a(i[o],e[o]):e[o]}return i}const R_={data:nh,props:ih,emits:ih,methods:Lo,computed:Lo,beforeCreate:un,created:un,beforeMount:un,mounted:un,beforeUpdate:un,updated:un,beforeDestroy:un,beforeUnmount:un,destroyed:un,unmounted:un,activated:un,deactivated:un,errorCaptured:un,serverPrefetch:un,components:Lo,directives:Lo,watch:D_,provide:nh,inject:I_};function nh(i,e){return e?i?function(){return ln($e(i)?i.call(this,this):i,$e(e)?e.call(this,this):e)}:e:i}function I_(i,e){return Lo(Au(i),Au(e))}function Au(i){if(Ke(i)){const e={};for(let t=0;t<i.length;t++)e[i[t]]=i[t];return e}return i}function un(i,e){return i?[...new Set([].concat(i,e))]:e}function Lo(i,e){return i?ln(Object.create(null),i,e):e}function ih(i,e){return i?Ke(i)&&Ke(e)?[...new Set([...i,...e])]:ln(Object.create(null),eh(i),eh(e??{})):e}function D_(i,e){if(!i)return e;if(!e)return i;const t=ln(Object.create(null),i);for(const n in e)t[n]=un(i[n],e[n]);return t}function n0(){return{app:null,config:{isNativeTag:gm,performance:!1,globalProperties:{},optionMergeStrategies:{},errorHandler:void 0,warnHandler:void 0,compilerOptions:{}},mixins:[],components:{},directives:{},provides:Object.create(null),optionsCache:new WeakMap,propsCache:new WeakMap,emitsCache:new WeakMap}}let P_=0;function F_(i,e){return function(n,s=null){$e(n)||(n=ln({},n)),s!=null&&!mt(s)&&(s=null);const r=n0(),o=new WeakSet,a=[];let l=!1;const c=r.app={_uid:P_++,_component:n,_props:s,_container:null,_context:r,_instance:null,version:hA,get config(){return r.config},set config(u){},use(u,...f){return o.has(u)||(u&&$e(u.install)?(o.add(u),u.install(c,...f)):$e(u)&&(o.add(u),u(c,...f))),c},mixin(u){return r.mixins.includes(u)||r.mixins.push(u),c},component(u,f){return f?(r.components[u]=f,c):r.components[u]},directive(u,f){return f?(r.directives[u]=f,c):r.directives[u]},mount(u,f,d){if(!l){const h=c._ceVNode||Pi(n,s);return h.appContext=r,d===!0?d="svg":d===!1&&(d=void 0),i(h,u,d),l=!0,c._container=u,u.__vue_app__=c,Kl(h.component)}},onUnmount(u){a.push(u)},unmount(){l&&(Li(a,c._instance,16),i(null,c._container),delete c._container.__vue_app__)},provide(u,f){return r.provides[u]=f,c},runWithContext(u){const f=jr;jr=c;try{return u()}finally{jr=f}}};return c}}let jr=null;const L_=(i,e)=>e==="modelValue"||e==="model-value"?i.modelModifiers:i[`${e}Modifiers`]||i[`${Ls(e)}Modifiers`]||i[`${ks(e)}Modifiers`];function B_(i,e,...t){if(i.isUnmounted)return;const n=i.vnode.props||At;let s=t;const r=e.startsWith("update:"),o=r&&L_(n,e.slice(7));o&&(o.trim&&(s=t.map(u=>Xt(u)?u.trim():u)),o.number&&(s=t.map(Pf)));let a,l=n[a=ac(e)]||n[a=ac(Ls(e))];!l&&r&&(l=n[a=ac(ks(e))]),l&&Li(l,i,6,s);const c=n[a+"Once"];if(c){if(!i.emitted)i.emitted={};else if(i.emitted[a])return;i.emitted[a]=!0,Li(c,i,6,s)}}const U_=new WeakMap;function i0(i,e,t=!1){const n=t?U_:e.emitsCache,s=n.get(i);if(s!==void 0)return s;const r=i.emits;let o={},a=!1;if(!$e(i)){const l=c=>{const u=i0(c,e,!0);u&&(a=!0,ln(o,u))};!t&&e.mixins.length&&e.mixins.forEach(l),i.extends&&l(i.extends),i.mixins&&i.mixins.forEach(l)}return!r&&!a?(mt(i)&&n.set(i,null),null):(Ke(r)?r.forEach(l=>o[l]=null):ln(o,r),mt(i)&&n.set(i,o),o)}function Ql(i,e){return!i||!Hl(e)?!1:(e=e.slice(2).replace(/Once$/,""),ct(i,e[0].toLowerCase()+e.slice(1))||ct(i,ks(e))||ct(i,e))}function sh(i){const{type:e,vnode:t,proxy:n,withProxy:s,propsOptions:[r],slots:o,attrs:a,emit:l,render:c,renderCache:u,props:f,data:d,setupState:h,ctx:x,inheritAttrs:p}=i,g=bl(i);let m,_;try{if(t.shapeFlag&4){const S=s||n,v=S;m=Mi(c.call(v,S,u,f,h,d,x)),_=a}else{const S=e;m=Mi(S.length>1?S(f,{attrs:a,slots:o,emit:l}):S(f,null)),_=e.props?a:O_(a)}}catch(S){Wo.length=0,Xl(S,i,1),m=Pi(Bs)}let A=m;if(_&&p!==!1){const S=Object.keys(_),{shapeFlag:v}=A;S.length&&v&7&&(r&&S.some(Rf)&&(_=N_(_,r)),A=oo(A,_,!1,!0))}return t.dirs&&(A=oo(A,null,!1,!0),A.dirs=A.dirs?A.dirs.concat(t.dirs):t.dirs),t.transition&&Wf(A,t.transition),m=A,bl(g),m}const O_=i=>{let e;for(const t in i)(t==="class"||t==="style"||Hl(t))&&((e||(e={}))[t]=i[t]);return e},N_=(i,e)=>{const t={};for(const n in i)(!Rf(n)||!(n.slice(9)in e))&&(t[n]=i[n]);return t};function z_(i,e,t){const{props:n,children:s,component:r}=i,{props:o,children:a,patchFlag:l}=e,c=r.emitsOptions;if(e.dirs||e.transition)return!0;if(t&&l>=0){if(l&1024)return!0;if(l&16)return n?rh(n,o,c):!!o;if(l&8){const u=e.dynamicProps;for(let f=0;f<u.length;f++){const d=u[f];if(s0(o,n,d)&&!Ql(c,d))return!0}}}else return(s||a)&&(!a||!a.$stable)?!0:n===o?!1:n?o?rh(n,o,c):!0:!!o;return!1}function rh(i,e,t){const n=Object.keys(e);if(n.length!==Object.keys(i).length)return!0;for(let s=0;s<n.length;s++){const r=n[s];if(s0(e,i,r)&&!Ql(t,r))return!0}return!1}function s0(i,e,t){const n=i[t],s=e[t];return t==="style"&&mt(n)&&mt(s)?!Lf(n,s):n!==s}function k_({vnode:i,parent:e},t){for(;e;){const n=e.subTree;if(n.suspense&&n.suspense.activeBranch===i&&(n.el=i.el),n===i)(i=e.vnode).el=t,e=e.parent;else break}}const r0={},o0=()=>Object.create(r0),a0=i=>Object.getPrototypeOf(i)===r0;function H_(i,e,t,n=!1){const s={},r=o0();i.propsDefaults=Object.create(null),l0(i,e,s,r);for(const o in i.propsOptions[0])o in s||(s[o]=void 0);t?i.props=n?s:Xx(s):i.type.props?i.props=s:i.props=r,i.attrs=r}function V_(i,e,t,n){const{props:s,attrs:r,vnode:{patchFlag:o}}=i,a=lt(s),[l]=i.propsOptions;let c=!1;if((n||o>0)&&!(o&16)){if(o&8){const u=i.vnode.dynamicProps;for(let f=0;f<u.length;f++){let d=u[f];if(Ql(i.emitsOptions,d))continue;const h=e[d];if(l)if(ct(r,d))h!==r[d]&&(r[d]=h,c=!0);else{const x=Ls(d);s[x]=Su(l,a,x,h,i,!1)}else h!==r[d]&&(r[d]=h,c=!0)}}}else{l0(i,e,s,r)&&(c=!0);let u;for(const f in a)(!e||!ct(e,f)&&((u=ks(f))===f||!ct(e,u)))&&(l?t&&(t[f]!==void 0||t[u]!==void 0)&&(s[f]=Su(l,a,f,void 0,i,!0)):delete s[f]);if(r!==a)for(const f in r)(!e||!ct(e,f))&&(delete r[f],c=!0)}c&&Ji(i.attrs,"set","")}function l0(i,e,t,n){const[s,r]=i.propsOptions;let o=!1,a;if(e)for(let l in e){if(No(l))continue;const c=e[l];let u;s&&ct(s,u=Ls(l))?!r||!r.includes(u)?t[u]=c:(a||(a={}))[u]=c:Ql(i.emitsOptions,l)||(!(l in n)||c!==n[l])&&(n[l]=c,o=!0)}if(r){const l=lt(t),c=a||At;for(let u=0;u<r.length;u++){const f=r[u];t[f]=Su(s,l,f,c[f],i,!ct(c,f))}}return o}function Su(i,e,t,n,s,r){const o=i[t];if(o!=null){const a=ct(o,"default");if(a&&n===void 0){const l=o.default;if(o.type!==Function&&!o.skipFactory&&$e(l)){const{propsDefaults:c}=s;if(t in c)n=c[t];else{const u=va(s);n=c[t]=l.call(null,e),u()}}else n=l;s.ce&&s.ce._setProp(t,n)}o[0]&&(r&&!a?n=!1:o[1]&&(n===""||n===ks(t))&&(n=!0))}return n}const G_=new WeakMap;function c0(i,e,t=!1){const n=t?G_:e.propsCache,s=n.get(i);if(s)return s;const r=i.props,o={},a=[];let l=!1;if(!$e(i)){const u=f=>{l=!0;const[d,h]=c0(f,e,!0);ln(o,d),h&&a.push(...h)};!t&&e.mixins.length&&e.mixins.forEach(u),i.extends&&u(i.extends),i.mixins&&i.mixins.forEach(u)}if(!r&&!l)return mt(i)&&n.set(i,qr),qr;if(Ke(r))for(let u=0;u<r.length;u++){const f=Ls(r[u]);oh(f)&&(o[f]=At)}else if(r)for(const u in r){const f=Ls(u);if(oh(f)){const d=r[u],h=o[f]=Ke(d)||$e(d)?{type:d}:ln({},d),x=h.type;let p=!1,g=!0;if(Ke(x))for(let m=0;m<x.length;++m){const _=x[m],A=$e(_)&&_.name;if(A==="Boolean"){p=!0;break}else A==="String"&&(g=!1)}else p=$e(x)&&x.name==="Boolean";h[0]=p,h[1]=g,(p||ct(h,"default"))&&a.push(f)}}const c=[o,a];return mt(i)&&n.set(i,c),c}function oh(i){return i[0]!=="$"&&!No(i)}const Xf=i=>i==="_"||i==="_ctx"||i==="$stable",qf=i=>Ke(i)?i.map(Mi):[Mi(i)],W_=(i,e,t)=>{if(e._n)return e;const n=r_((...s)=>qf(e(...s)),t);return n._c=!1,n},u0=(i,e,t)=>{const n=i._ctx;for(const s in i){if(Xf(s))continue;const r=i[s];if($e(r))e[s]=W_(s,r,n);else if(r!=null){const o=qf(r);e[s]=()=>o}}},f0=(i,e)=>{const t=qf(e);i.slots.default=()=>t},d0=(i,e,t)=>{for(const n in e)(t||!Xf(n))&&(i[n]=e[n])},X_=(i,e,t)=>{const n=i.slots=o0();if(i.vnode.shapeFlag&32){const s=e._;s?(d0(n,e,t),t&&ym(n,"_",s,!0)):u0(e,n)}else e&&f0(i,e)},q_=(i,e,t)=>{const{vnode:n,slots:s}=i;let r=!0,o=At;if(n.shapeFlag&32){const a=e._;a?t&&a===1?r=!1:d0(s,e,t):(r=!e.$stable,u0(e,s)),o=e}else e&&(f0(i,e),o={default:1});if(r)for(const a in s)!Xf(a)&&o[a]==null&&delete s[a]},Mn=$_;function Q_(i){return Y_(i)}function Y_(i,e){const t=Gl();t.__VUE__=!0;const{insert:n,remove:s,patchProp:r,createElement:o,createText:a,createComment:l,setText:c,setElementText:u,parentNode:f,nextSibling:d,setScopeId:h=Di,insertStaticContent:x}=i,p=(P,L,q,w=null,te=null,ie=null,ue=void 0,Z=null,de=!!L.dynamicChildren)=>{if(P===L)return;P&&!To(P,L)&&(w=ee(P),we(P,te,ie,!0),P=null),L.patchFlag===-2&&(de=!1,L.dynamicChildren=null);const{type:ne,ref:ge,shapeFlag:R}=L;switch(ne){case Yl:g(P,L,q,w);break;case Bs:m(P,L,q,w);break;case mc:P==null&&_(L,q,w,ue);break;case bi:F(P,L,q,w,te,ie,ue,Z,de);break;default:R&1?v(P,L,q,w,te,ie,ue,Z,de):R&6?U(P,L,q,w,te,ie,ue,Z,de):(R&64||R&128)&&ne.process(P,L,q,w,te,ie,ue,Z,de,xe)}ge!=null&&te?Ho(ge,P&&P.ref,ie,L||P,!L):ge==null&&P&&P.ref!=null&&Ho(P.ref,null,ie,P,!0)},g=(P,L,q,w)=>{if(P==null)n(L.el=a(L.children),q,w);else{const te=L.el=P.el;L.children!==P.children&&c(te,L.children)}},m=(P,L,q,w)=>{P==null?n(L.el=l(L.children||""),q,w):L.el=P.el},_=(P,L,q,w)=>{[P.el,P.anchor]=x(P.children,L,q,w,P.el,P.anchor)},A=({el:P,anchor:L},q,w)=>{let te;for(;P&&P!==L;)te=d(P),n(P,q,w),P=te;n(L,q,w)},S=({el:P,anchor:L})=>{let q;for(;P&&P!==L;)q=d(P),s(P),P=q;s(L)},v=(P,L,q,w,te,ie,ue,Z,de)=>{if(L.type==="svg"?ue="svg":L.type==="math"&&(ue="mathml"),P==null)y(L,q,w,te,ie,ue,Z,de);else{const ne=P.el&&P.el._isVueCE?P.el:null;try{ne&&ne._beginPatch(),b(P,L,te,ie,ue,Z,de)}finally{ne&&ne._endPatch()}}},y=(P,L,q,w,te,ie,ue,Z)=>{let de,ne;const{props:ge,shapeFlag:R,transition:T,dirs:G}=P;if(de=P.el=o(P.type,ie,ge&&ge.is,ge),R&8?u(de,P.children):R&16&&E(P.children,de,null,w,te,pc(P,ie),ue,Z),G&&Ws(P,null,w,"created"),M(de,P,P.scopeId,ue,w),ge){for(const le in ge)le!=="value"&&!No(le)&&r(de,le,null,ge[le],ie,w);"value"in ge&&r(de,"value",null,ge.value,ie),(ne=ge.onVnodeBeforeMount)&&Si(ne,w,P)}G&&Ws(P,null,w,"beforeMount");const se=K_(te,T);se&&T.beforeEnter(de),n(de,L,q),((ne=ge&&ge.onVnodeMounted)||se||G)&&Mn(()=>{ne&&Si(ne,w,P),se&&T.enter(de),G&&Ws(P,null,w,"mounted")},te)},M=(P,L,q,w,te)=>{if(q&&h(P,q),w)for(let ie=0;ie<w.length;ie++)h(P,w[ie]);if(te){let ie=te.subTree;if(L===ie||g0(ie.type)&&(ie.ssContent===L||ie.ssFallback===L)){const ue=te.vnode;M(P,ue,ue.scopeId,ue.slotScopeIds,te.parent)}}},E=(P,L,q,w,te,ie,ue,Z,de=0)=>{for(let ne=de;ne<P.length;ne++){const ge=P[ne]=Z?ji(P[ne]):Mi(P[ne]);p(null,ge,L,q,w,te,ie,ue,Z)}},b=(P,L,q,w,te,ie,ue)=>{const Z=L.el=P.el;let{patchFlag:de,dynamicChildren:ne,dirs:ge}=L;de|=P.patchFlag&16;const R=P.props||At,T=L.props||At;let G;if(q&&Xs(q,!1),(G=T.onVnodeBeforeUpdate)&&Si(G,q,L,P),ge&&Ws(L,P,q,"beforeUpdate"),q&&Xs(q,!0),(R.innerHTML&&T.innerHTML==null||R.textContent&&T.textContent==null)&&u(Z,""),ne?C(P.dynamicChildren,ne,Z,q,w,pc(L,te),ie):ue||H(P,L,Z,null,q,w,pc(L,te),ie,!1),de>0){if(de&16)I(Z,R,T,q,te);else if(de&2&&R.class!==T.class&&r(Z,"class",null,T.class,te),de&4&&r(Z,"style",R.style,T.style,te),de&8){const se=L.dynamicProps;for(let le=0;le<se.length;le++){const j=se[le],De=R[j],_e=T[j];(_e!==De||j==="value")&&r(Z,j,De,_e,te,q)}}de&1&&P.children!==L.children&&u(Z,L.children)}else!ue&&ne==null&&I(Z,R,T,q,te);((G=T.onVnodeUpdated)||ge)&&Mn(()=>{G&&Si(G,q,L,P),ge&&Ws(L,P,q,"updated")},w)},C=(P,L,q,w,te,ie,ue)=>{for(let Z=0;Z<L.length;Z++){const de=P[Z],ne=L[Z],ge=de.el&&(de.type===bi||!To(de,ne)||de.shapeFlag&198)?f(de.el):q;p(de,ne,ge,null,w,te,ie,ue,!0)}},I=(P,L,q,w,te)=>{if(L!==q){if(L!==At)for(const ie in L)!No(ie)&&!(ie in q)&&r(P,ie,L[ie],null,te,w);for(const ie in q){if(No(ie))continue;const ue=q[ie],Z=L[ie];ue!==Z&&ie!=="value"&&r(P,ie,Z,ue,te,w)}"value"in q&&r(P,"value",L.value,q.value,te)}},F=(P,L,q,w,te,ie,ue,Z,de)=>{const ne=L.el=P?P.el:a(""),ge=L.anchor=P?P.anchor:a("");let{patchFlag:R,dynamicChildren:T,slotScopeIds:G}=L;G&&(Z=Z?Z.concat(G):G),P==null?(n(ne,q,w),n(ge,q,w),E(L.children||[],q,ge,te,ie,ue,Z,de)):R>0&&R&64&&T&&P.dynamicChildren&&P.dynamicChildren.length===T.length?(C(P.dynamicChildren,T,q,te,ie,ue,Z),(L.key!=null||te&&L===te.subTree)&&h0(P,L,!0)):H(P,L,q,ge,te,ie,ue,Z,de)},U=(P,L,q,w,te,ie,ue,Z,de)=>{L.slotScopeIds=Z,P==null?L.shapeFlag&512?te.ctx.activate(L,q,w,ue,de):O(L,q,w,te,ie,ue,de):k(P,L,de)},O=(P,L,q,w,te,ie,ue)=>{const Z=P.component=oA(P,w,te);if(Km(P)&&(Z.ctx.renderer=xe),lA(Z,!1,ue),Z.asyncDep){if(te&&te.registerDep(Z,z,ue),!P.el){const de=Z.subTree=Pi(Bs);m(null,de,L,q),P.placeholder=de.el}}else z(Z,P,L,q,te,ie,ue)},k=(P,L,q)=>{const w=L.component=P.component;if(z_(P,L,q))if(w.asyncDep&&!w.asyncResolved){V(w,L,q);return}else w.next=L,w.update();else L.el=P.el,w.vnode=L},z=(P,L,q,w,te,ie,ue)=>{const Z=()=>{if(P.isMounted){let{next:R,bu:T,u:G,parent:se,vnode:le}=P;{const N=p0(P);if(N){R&&(R.el=le.el,V(P,R,ue)),N.asyncDep.then(()=>{Mn(()=>{P.isUnmounted||ne()},te)});return}}let j=R,De;Xs(P,!1),R?(R.el=le.el,V(P,R,ue)):R=le,T&&ll(T),(De=R.props&&R.props.onVnodeBeforeUpdate)&&Si(De,se,R,le),Xs(P,!0);const _e=sh(P),Ue=P.subTree;P.subTree=_e,p(Ue,_e,f(Ue.el),ee(Ue),P,te,ie),R.el=_e.el,j===null&&k_(P,_e.el),G&&Mn(G,te),(De=R.props&&R.props.onVnodeUpdated)&&Mn(()=>Si(De,se,R,le),te)}else{let R;const{el:T,props:G}=L,{bm:se,m:le,parent:j,root:De,type:_e}=P,Ue=Vo(L);Xs(P,!1),se&&ll(se),!Ue&&(R=G&&G.onVnodeBeforeMount)&&Si(R,j,L),Xs(P,!0);{De.ce&&De.ce._hasShadowRoot()&&De.ce._injectChildStyle(_e);const N=P.subTree=sh(P);p(null,N,q,w,P,te,ie),L.el=N.el}if(le&&Mn(le,te),!Ue&&(R=G&&G.onVnodeMounted)){const N=L;Mn(()=>Si(R,j,N),te)}(L.shapeFlag&256||j&&Vo(j.vnode)&&j.vnode.shapeFlag&256)&&P.a&&Mn(P.a,te),P.isMounted=!0,L=q=w=null}};P.scope.on();const de=P.effect=new Tm(Z);P.scope.off();const ne=P.update=de.run.bind(de),ge=P.job=de.runIfDirty.bind(de);ge.i=P,ge.id=P.uid,de.scheduler=()=>Gf(ge),Xs(P,!0),ne()},V=(P,L,q)=>{L.component=P;const w=P.vnode.props;P.vnode=L,P.next=null,V_(P,L.props,w,q),q_(P,L.children,q),rs(),$d(P),os()},H=(P,L,q,w,te,ie,ue,Z,de=!1)=>{const ne=P&&P.children,ge=P?P.shapeFlag:0,R=L.children,{patchFlag:T,shapeFlag:G}=L;if(T>0){if(T&128){oe(ne,R,q,w,te,ie,ue,Z,de);return}else if(T&256){$(ne,R,q,w,te,ie,ue,Z,de);return}}G&8?(ge&16&&X(ne,te,ie),R!==ne&&u(q,R)):ge&16?G&16?oe(ne,R,q,w,te,ie,ue,Z,de):X(ne,te,ie,!0):(ge&8&&u(q,""),G&16&&E(R,q,w,te,ie,ue,Z,de))},$=(P,L,q,w,te,ie,ue,Z,de)=>{P=P||qr,L=L||qr;const ne=P.length,ge=L.length,R=Math.min(ne,ge);let T;for(T=0;T<R;T++){const G=L[T]=de?ji(L[T]):Mi(L[T]);p(P[T],G,q,null,te,ie,ue,Z,de)}ne>ge?X(P,te,ie,!0,!1,R):E(L,q,w,te,ie,ue,Z,de,R)},oe=(P,L,q,w,te,ie,ue,Z,de)=>{let ne=0;const ge=L.length;let R=P.length-1,T=ge-1;for(;ne<=R&&ne<=T;){const G=P[ne],se=L[ne]=de?ji(L[ne]):Mi(L[ne]);if(To(G,se))p(G,se,q,null,te,ie,ue,Z,de);else break;ne++}for(;ne<=R&&ne<=T;){const G=P[R],se=L[T]=de?ji(L[T]):Mi(L[T]);if(To(G,se))p(G,se,q,null,te,ie,ue,Z,de);else break;R--,T--}if(ne>R){if(ne<=T){const G=T+1,se=G<ge?L[G].el:w;for(;ne<=T;)p(null,L[ne]=de?ji(L[ne]):Mi(L[ne]),q,se,te,ie,ue,Z,de),ne++}}else if(ne>T)for(;ne<=R;)we(P[ne],te,ie,!0),ne++;else{const G=ne,se=ne,le=new Map;for(ne=se;ne<=T;ne++){const Te=L[ne]=de?ji(L[ne]):Mi(L[ne]);Te.key!=null&&le.set(Te.key,ne)}let j,De=0;const _e=T-se+1;let Ue=!1,N=0;const J=new Array(_e);for(ne=0;ne<_e;ne++)J[ne]=0;for(ne=G;ne<=R;ne++){const Te=P[ne];if(De>=_e){we(Te,te,ie,!0);continue}let Pe;if(Te.key!=null)Pe=le.get(Te.key);else for(j=se;j<=T;j++)if(J[j-se]===0&&To(Te,L[j])){Pe=j;break}Pe===void 0?we(Te,te,ie,!0):(J[Pe-se]=ne+1,Pe>=N?N=Pe:Ue=!0,p(Te,L[Pe],q,null,te,ie,ue,Z,de),De++)}const me=Ue?j_(J):qr;for(j=me.length-1,ne=_e-1;ne>=0;ne--){const Te=se+ne,Pe=L[Te],Re=L[Te+1],He=Te+1<ge?Re.el||m0(Re):w;J[ne]===0?p(null,Pe,q,He,te,ie,ue,Z,de):Ue&&(j<0||ne!==me[j]?Se(Pe,q,He,2):j--)}}},Se=(P,L,q,w,te=null)=>{const{el:ie,type:ue,transition:Z,children:de,shapeFlag:ne}=P;if(ne&6){Se(P.component.subTree,L,q,w);return}if(ne&128){P.suspense.move(L,q,w);return}if(ne&64){ue.move(P,L,q,xe);return}if(ue===bi){n(ie,L,q);for(let R=0;R<de.length;R++)Se(de[R],L,q,w);n(P.anchor,L,q);return}if(ue===mc){A(P,L,q);return}if(w!==2&&ne&1&&Z)if(w===0)Z.beforeEnter(ie),n(ie,L,q),Mn(()=>Z.enter(ie),te);else{const{leave:R,delayLeave:T,afterLeave:G}=Z,se=()=>{P.ctx.isUnmounted?s(ie):n(ie,L,q)},le=()=>{ie._isLeaving&&ie[h_](!0),R(ie,()=>{se(),G&&G()})};T?T(ie,se,le):le()}else n(ie,L,q)},we=(P,L,q,w=!1,te=!1)=>{const{type:ie,props:ue,ref:Z,children:de,dynamicChildren:ne,shapeFlag:ge,patchFlag:R,dirs:T,cacheIndex:G}=P;if(R===-2&&(te=!1),Z!=null&&(rs(),Ho(Z,null,q,P,!0),os()),G!=null&&(L.renderCache[G]=void 0),ge&256){L.ctx.deactivate(P);return}const se=ge&1&&T,le=!Vo(P);let j;if(le&&(j=ue&&ue.onVnodeBeforeUnmount)&&Si(j,L,P),ge&6)re(P.component,q,w);else{if(ge&128){P.suspense.unmount(q,w);return}se&&Ws(P,null,L,"beforeUnmount"),ge&64?P.type.remove(P,L,q,xe,w):ne&&!ne.hasOnce&&(ie!==bi||R>0&&R&64)?X(ne,L,q,!1,!0):(ie===bi&&R&384||!te&&ge&16)&&X(de,L,q),w&&Le(P)}(le&&(j=ue&&ue.onVnodeUnmounted)||se)&&Mn(()=>{j&&Si(j,L,P),se&&Ws(P,null,L,"unmounted")},q)},Le=P=>{const{type:L,el:q,anchor:w,transition:te}=P;if(L===bi){fe(q,w);return}if(L===mc){S(P);return}const ie=()=>{s(q),te&&!te.persisted&&te.afterLeave&&te.afterLeave()};if(P.shapeFlag&1&&te&&!te.persisted){const{leave:ue,delayLeave:Z}=te,de=()=>ue(q,ie);Z?Z(P.el,ie,de):de()}else ie()},fe=(P,L)=>{let q;for(;P!==L;)q=d(P),s(P),P=q;s(L)},re=(P,L,q)=>{const{bum:w,scope:te,job:ie,subTree:ue,um:Z,m:de,a:ne}=P;ah(de),ah(ne),w&&ll(w),te.stop(),ie&&(ie.flags|=8,we(ue,P,L,q)),Z&&Mn(Z,L),Mn(()=>{P.isUnmounted=!0},L)},X=(P,L,q,w=!1,te=!1,ie=0)=>{for(let ue=ie;ue<P.length;ue++)we(P[ue],L,q,w,te)},ee=P=>{if(P.shapeFlag&6)return ee(P.component.subTree);if(P.shapeFlag&128)return P.suspense.next();const L=d(P.anchor||P.el),q=L&&L[f_];return q?d(q):L};let pe=!1;const be=(P,L,q)=>{let w;P==null?L._vnode&&(we(L._vnode,null,null,!0),w=L._vnode.component):p(L._vnode||null,P,L,null,null,null,q),L._vnode=P,pe||(pe=!0,$d(w),Gm(),pe=!1)},xe={p,um:we,m:Se,r:Le,mt:O,mc:E,pc:H,pbc:C,n:ee,o:i};return{render:be,hydrate:void 0,createApp:F_(be)}}function pc({type:i,props:e},t){return t==="svg"&&i==="foreignObject"||t==="mathml"&&i==="annotation-xml"&&e&&e.encoding&&e.encoding.includes("html")?void 0:t}function Xs({effect:i,job:e},t){t?(i.flags|=32,e.flags|=4):(i.flags&=-33,e.flags&=-5)}function K_(i,e){return(!i||i&&!i.pendingBranch)&&e&&!e.persisted}function h0(i,e,t=!1){const n=i.children,s=e.children;if(Ke(n)&&Ke(s))for(let r=0;r<n.length;r++){const o=n[r];let a=s[r];a.shapeFlag&1&&!a.dynamicChildren&&((a.patchFlag<=0||a.patchFlag===32)&&(a=s[r]=ji(s[r]),a.el=o.el),!t&&a.patchFlag!==-2&&h0(o,a)),a.type===Yl&&(a.patchFlag===-1&&(a=s[r]=ji(a)),a.el=o.el),a.type===Bs&&!a.el&&(a.el=o.el)}}function j_(i){const e=i.slice(),t=[0];let n,s,r,o,a;const l=i.length;for(n=0;n<l;n++){const c=i[n];if(c!==0){if(s=t[t.length-1],i[s]<c){e[n]=s,t.push(n);continue}for(r=0,o=t.length-1;r<o;)a=r+o>>1,i[t[a]]<c?r=a+1:o=a;c<i[t[r]]&&(r>0&&(e[n]=t[r-1]),t[r]=n)}}for(r=t.length,o=t[r-1];r-- >0;)t[r]=o,o=e[o];return t}function p0(i){const e=i.subTree.component;if(e)return e.asyncDep&&!e.asyncResolved?e:p0(e)}function ah(i){if(i)for(let e=0;e<i.length;e++)i[e].flags|=8}function m0(i){if(i.placeholder)return i.placeholder;const e=i.component;return e?m0(e.subTree):null}const g0=i=>i.__isSuspense;function $_(i,e){e&&e.pendingBranch?Ke(i)?e.effects.push(...i):e.effects.push(i):s_(i)}const bi=Symbol.for("v-fgt"),Yl=Symbol.for("v-txt"),Bs=Symbol.for("v-cmt"),mc=Symbol.for("v-stc"),Wo=[];let Vn=null;function Cn(i=!1){Wo.push(Vn=i?null:[])}function Z_(){Wo.pop(),Vn=Wo[Wo.length-1]||null}let na=1;function lh(i,e=!1){na+=i,i<0&&Vn&&e&&(Vn.hasOnce=!0)}function x0(i){return i.dynamicChildren=na>0?Vn||qr:null,Z_(),na>0&&Vn&&Vn.push(i),i}function Nn(i,e,t,n,s,r){return x0(Ht(i,e,t,n,s,r,!0))}function J_(i,e,t,n,s){return x0(Pi(i,e,t,n,s,!0))}function _0(i){return i?i.__v_isVNode===!0:!1}function To(i,e){return i.type===e.type&&i.key===e.key}const A0=({key:i})=>i??null,ul=({ref:i,ref_key:e,ref_for:t})=>(typeof i=="number"&&(i=""+i),i!=null?Xt(i)||rn(i)||$e(i)?{i:ni,r:i,k:e,f:!!t}:i:null);function Ht(i,e=null,t=null,n=0,s=null,r=i===bi?0:1,o=!1,a=!1){const l={__v_isVNode:!0,__v_skip:!0,type:i,props:e,key:e&&A0(e),ref:e&&ul(e),scopeId:Xm,slotScopeIds:null,children:t,component:null,suspense:null,ssContent:null,ssFallback:null,dirs:null,transition:null,el:null,anchor:null,target:null,targetStart:null,targetAnchor:null,staticCount:0,shapeFlag:r,patchFlag:n,dynamicProps:s,dynamicChildren:null,appContext:null,ctx:ni};return a?(Qf(l,t),r&128&&i.normalize(l)):t&&(l.shapeFlag|=Xt(t)?8:16),na>0&&!o&&Vn&&(l.patchFlag>0||r&6)&&l.patchFlag!==32&&Vn.push(l),l}const Pi=eA;function eA(i,e=null,t=null,n=0,s=null,r=!1){if((!i||i===M_)&&(i=Bs),_0(i)){const a=oo(i,e,!0);return t&&Qf(a,t),na>0&&!r&&Vn&&(a.shapeFlag&6?Vn[Vn.indexOf(i)]=a:Vn.push(a)),a.patchFlag=-2,a}if(dA(i)&&(i=i.__vccOpts),e){e=tA(e);let{class:a,style:l}=e;a&&!Xt(a)&&(e.class=Yr(a)),mt(l)&&(Vf(l)&&!Ke(l)&&(l=ln({},l)),e.style=Ff(l))}const o=Xt(i)?1:g0(i)?128:d_(i)?64:mt(i)?4:$e(i)?2:0;return Ht(i,e,t,n,s,o,r,!0)}function tA(i){return i?Vf(i)||a0(i)?ln({},i):i:null}function oo(i,e,t=!1,n=!1){const{props:s,ref:r,patchFlag:o,children:a,transition:l}=i,c=e?iA(s||{},e):s,u={__v_isVNode:!0,__v_skip:!0,type:i.type,props:c,key:c&&A0(c),ref:e&&e.ref?t&&r?Ke(r)?r.concat(ul(e)):[r,ul(e)]:ul(e):r,scopeId:i.scopeId,slotScopeIds:i.slotScopeIds,children:a,target:i.target,targetStart:i.targetStart,targetAnchor:i.targetAnchor,staticCount:i.staticCount,shapeFlag:i.shapeFlag,patchFlag:e&&i.type!==bi?o===-1?16:o|16:o,dynamicProps:i.dynamicProps,dynamicChildren:i.dynamicChildren,appContext:i.appContext,dirs:i.dirs,transition:l,component:i.component,suspense:i.suspense,ssContent:i.ssContent&&oo(i.ssContent),ssFallback:i.ssFallback&&oo(i.ssFallback),placeholder:i.placeholder,el:i.el,anchor:i.anchor,ctx:i.ctx,ce:i.ce};return l&&n&&Wf(u,l.clone(u)),u}function nA(i=" ",e=0){return Pi(Yl,null,i,e)}function fs(i="",e=!1){return e?(Cn(),J_(Bs,null,i)):Pi(Bs,null,i)}function Mi(i){return i==null||typeof i=="boolean"?Pi(Bs):Ke(i)?Pi(bi,null,i.slice()):_0(i)?ji(i):Pi(Yl,null,String(i))}function ji(i){return i.el===null&&i.patchFlag!==-1||i.memo?i:oo(i)}function Qf(i,e){let t=0;const{shapeFlag:n}=i;if(e==null)e=null;else if(Ke(e))t=16;else if(typeof e=="object")if(n&65){const s=e.default;s&&(s._c&&(s._d=!1),Qf(i,s()),s._c&&(s._d=!0));return}else{t=32;const s=e._;!s&&!a0(e)?e._ctx=ni:s===3&&ni&&(ni.slots._===1?e._=1:(e._=2,i.patchFlag|=1024))}else $e(e)?(e={default:e,_ctx:ni},t=32):(e=String(e),n&64?(t=16,e=[nA(e)]):t=8);i.children=e,i.shapeFlag|=t}function iA(...i){const e={};for(let t=0;t<i.length;t++){const n=i[t];for(const s in n)if(s==="class")e.class!==n.class&&(e.class=Yr([e.class,n.class]));else if(s==="style")e.style=Ff([e.style,n.style]);else if(Hl(s)){const r=e[s],o=n[s];o&&r!==o&&!(Ke(r)&&r.includes(o))&&(e[s]=r?[].concat(r,o):o)}else s!==""&&(e[s]=n[s])}return e}function Si(i,e,t,n=null){Li(i,e,7,[t,n])}const sA=n0();let rA=0;function oA(i,e,t){const n=i.type,s=(e?e.appContext:i.appContext)||sA,r={uid:rA++,vnode:i,type:n,parent:e,appContext:s,root:null,next:null,subTree:null,effect:null,update:null,job:null,scope:new Cx(!0),render:null,proxy:null,exposed:null,exposeProxy:null,withProxy:null,provides:e?e.provides:Object.create(s.provides),ids:e?e.ids:["",0,0],accessCache:null,renderCache:[],components:null,directives:null,propsOptions:c0(n,s),emitsOptions:i0(n,s),emit:null,emitted:null,propsDefaults:At,inheritAttrs:n.inheritAttrs,ctx:At,data:At,props:At,attrs:At,slots:At,refs:At,setupState:At,setupContext:null,suspense:t,suspenseId:t?t.pendingId:0,asyncDep:null,asyncResolved:!1,isMounted:!1,isUnmounted:!1,isDeactivated:!1,bc:null,c:null,bm:null,m:null,bu:null,u:null,um:null,bum:null,da:null,a:null,rtg:null,rtc:null,ec:null,sp:null};return r.ctx={_:r},r.root=e?e.root:r,r.emit=B_.bind(null,r),i.ce&&i.ce(r),r}let pn=null;const aA=()=>pn||ni;let Tl,vu;{const i=Gl(),e=(t,n)=>{let s;return(s=i[t])||(s=i[t]=[]),s.push(n),r=>{s.length>1?s.forEach(o=>o(r)):s[0](r)}};Tl=e("__VUE_INSTANCE_SETTERS__",t=>pn=t),vu=e("__VUE_SSR_SETTERS__",t=>ia=t)}const va=i=>{const e=pn;return Tl(i),i.scope.on(),()=>{i.scope.off(),Tl(e)}},ch=()=>{pn&&pn.scope.off(),Tl(null)};function S0(i){return i.vnode.shapeFlag&4}let ia=!1;function lA(i,e=!1,t=!1){e&&vu(e);const{props:n,children:s}=i.vnode,r=S0(i);H_(i,n,r,e),X_(i,s,t||e);const o=r?cA(i,e):void 0;return e&&vu(!1),o}function cA(i,e){const t=i.type;i.accessCache=Object.create(null),i.proxy=new Proxy(i.ctx,T_);const{setup:n}=t;if(n){rs();const s=i.setupContext=n.length>1?fA(i):null,r=va(i),o=Sa(n,i,0,[i.props,s]),a=_m(o);if(os(),r(),(a||i.sp)&&!Vo(i)&&Ym(i),a){if(o.then(ch,ch),e)return o.then(l=>{uh(i,l)}).catch(l=>{Xl(l,i,0)});i.asyncDep=o}else uh(i,o)}else v0(i)}function uh(i,e,t){$e(e)?i.type.__ssrInlineRender?i.ssrRender=e:i.render=e:mt(e)&&(i.setupState=km(e)),v0(i)}function v0(i,e,t){const n=i.type;i.render||(i.render=n.render||Di);{const s=va(i);rs();try{E_(i)}finally{os(),s()}}}const uA={get(i,e){return nn(i,"get",""),i[e]}};function fA(i){const e=t=>{i.exposed=t||{}};return{attrs:new Proxy(i.attrs,uA),slots:i.slots,emit:i.emit,expose:e}}function Kl(i){return i.exposed?i.exposeProxy||(i.exposeProxy=new Proxy(km(qx(i.exposed)),{get(e,t){if(t in e)return e[t];if(t in Go)return Go[t](i)},has(e,t){return t in e||t in Go}})):i.proxy}function dA(i){return $e(i)&&"__vccOpts"in i}const y0=(i,e)=>Zx(i,e,ia),hA="3.5.28";let yu;const fh=typeof window<"u"&&window.trustedTypes;if(fh)try{yu=fh.createPolicy("vue",{createHTML:i=>i})}catch{}const b0=yu?i=>yu.createHTML(i):i=>i,pA="http://www.w3.org/2000/svg",mA="http://www.w3.org/1998/Math/MathML",Yi=typeof document<"u"?document:null,dh=Yi&&Yi.createElement("template"),gA={insert:(i,e,t)=>{e.insertBefore(i,t||null)},remove:i=>{const e=i.parentNode;e&&e.removeChild(i)},createElement:(i,e,t,n)=>{const s=e==="svg"?Yi.createElementNS(pA,i):e==="mathml"?Yi.createElementNS(mA,i):t?Yi.createElement(i,{is:t}):Yi.createElement(i);return i==="select"&&n&&n.multiple!=null&&s.setAttribute("multiple",n.multiple),s},createText:i=>Yi.createTextNode(i),createComment:i=>Yi.createComment(i),setText:(i,e)=>{i.nodeValue=e},setElementText:(i,e)=>{i.textContent=e},parentNode:i=>i.parentNode,nextSibling:i=>i.nextSibling,querySelector:i=>Yi.querySelector(i),setScopeId(i,e){i.setAttribute(e,"")},insertStaticContent(i,e,t,n,s,r){const o=t?t.previousSibling:e.lastChild;if(s&&(s===r||s.nextSibling))for(;e.insertBefore(s.cloneNode(!0),t),!(s===r||!(s=s.nextSibling)););else{dh.innerHTML=b0(n==="svg"?`<svg>${i}</svg>`:n==="mathml"?`<math>${i}</math>`:i);const a=dh.content;if(n==="svg"||n==="mathml"){const l=a.firstChild;for(;l.firstChild;)a.appendChild(l.firstChild);a.removeChild(l)}e.insertBefore(a,t)}return[o?o.nextSibling:e.firstChild,t?t.previousSibling:e.lastChild]}},xA=Symbol("_vtc");function _A(i,e,t){const n=i[xA];n&&(e=(e?[e,...n]:[...n]).join(" ")),e==null?i.removeAttribute("class"):t?i.setAttribute("class",e):i.className=e}const hh=Symbol("_vod"),AA=Symbol("_vsh"),SA=Symbol(""),vA=/(?:^|;)\s*display\s*:/;function yA(i,e,t){const n=i.style,s=Xt(t);let r=!1;if(t&&!s){if(e)if(Xt(e))for(const o of e.split(";")){const a=o.slice(0,o.indexOf(":")).trim();t[a]==null&&fl(n,a,"")}else for(const o in e)t[o]==null&&fl(n,o,"");for(const o in t)o==="display"&&(r=!0),fl(n,o,t[o])}else if(s){if(e!==t){const o=n[SA];o&&(t+=";"+o),n.cssText=t,r=vA.test(t)}}else e&&i.removeAttribute("style");hh in i&&(i[hh]=r?n.display:"",i[AA]&&(n.display="none"))}const ph=/\s*!important$/;function fl(i,e,t){if(Ke(t))t.forEach(n=>fl(i,e,n));else if(t==null&&(t=""),e.startsWith("--"))i.setProperty(e,t);else{const n=bA(i,e);ph.test(t)?i.setProperty(ks(n),t.replace(ph,""),"important"):i[n]=t}}const mh=["Webkit","Moz","ms"],gc={};function bA(i,e){const t=gc[e];if(t)return t;let n=Ls(e);if(n!=="filter"&&n in i)return gc[e]=n;n=vm(n);for(let s=0;s<mh.length;s++){const r=mh[s]+n;if(r in i)return gc[e]=r}return e}const gh="http://www.w3.org/1999/xlink";function xh(i,e,t,n,s,r=bx(e)){n&&e.startsWith("xlink:")?t==null?i.removeAttributeNS(gh,e.slice(6,e.length)):i.setAttributeNS(gh,e,t):t==null||r&&!bm(t)?i.removeAttribute(e):i.setAttribute(e,r?"":Fi(t)?String(t):t)}function _h(i,e,t,n,s){if(e==="innerHTML"||e==="textContent"){t!=null&&(i[e]=e==="innerHTML"?b0(t):t);return}const r=i.tagName;if(e==="value"&&r!=="PROGRESS"&&!r.includes("-")){const a=r==="OPTION"?i.getAttribute("value")||"":i.value,l=t==null?i.type==="checkbox"?"on":"":String(t);(a!==l||!("_value"in i))&&(i.value=l),t==null&&i.removeAttribute(e),i._value=t;return}let o=!1;if(t===""||t==null){const a=typeof i[e];a==="boolean"?t=bm(t):t==null&&a==="string"?(t="",o=!0):a==="number"&&(t=0,o=!0)}try{i[e]=t}catch{}o&&i.removeAttribute(s||e)}function kr(i,e,t,n){i.addEventListener(e,t,n)}function MA(i,e,t,n){i.removeEventListener(e,t,n)}const Ah=Symbol("_vei");function CA(i,e,t,n,s=null){const r=i[Ah]||(i[Ah]={}),o=r[e];if(n&&o)o.value=n;else{const[a,l]=TA(e);if(n){const c=r[e]=RA(n,s);kr(i,a,c,l)}else o&&(MA(i,a,o,l),r[e]=void 0)}}const Sh=/(?:Once|Passive|Capture)$/;function TA(i){let e;if(Sh.test(i)){e={};let n;for(;n=i.match(Sh);)i=i.slice(0,i.length-n[0].length),e[n[0].toLowerCase()]=!0}return[i[2]===":"?i.slice(3):ks(i.slice(2)),e]}let xc=0;const EA=Promise.resolve(),wA=()=>xc||(EA.then(()=>xc=0),xc=Date.now());function RA(i,e){const t=n=>{if(!n._vts)n._vts=Date.now();else if(n._vts<=t.attached)return;Li(IA(n,t.value),e,5,[n])};return t.value=i,t.attached=wA(),t}function IA(i,e){if(Ke(e)){const t=i.stopImmediatePropagation;return i.stopImmediatePropagation=()=>{t.call(i),i._stopped=!0},e.map(n=>s=>!s._stopped&&n&&n(s))}else return e}const vh=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&i.charCodeAt(2)>96&&i.charCodeAt(2)<123,DA=(i,e,t,n,s,r)=>{const o=s==="svg";e==="class"?_A(i,n,o):e==="style"?yA(i,t,n):Hl(e)?Rf(e)||CA(i,e,t,n,r):(e[0]==="."?(e=e.slice(1),!0):e[0]==="^"?(e=e.slice(1),!1):PA(i,e,n,o))?(_h(i,e,n),!i.tagName.includes("-")&&(e==="value"||e==="checked"||e==="selected")&&xh(i,e,n,o,r,e!=="value")):i._isVueCE&&(/[A-Z]/.test(e)||!Xt(n))?_h(i,Ls(e),n,r,e):(e==="true-value"?i._trueValue=n:e==="false-value"&&(i._falseValue=n),xh(i,e,n,o))};function PA(i,e,t,n){if(n)return!!(e==="innerHTML"||e==="textContent"||e in i&&vh(e)&&$e(t));if(e==="spellcheck"||e==="draggable"||e==="translate"||e==="autocorrect"||e==="sandbox"&&i.tagName==="IFRAME"||e==="form"||e==="list"&&i.tagName==="INPUT"||e==="type"&&i.tagName==="TEXTAREA")return!1;if(e==="width"||e==="height"){const s=i.tagName;if(s==="IMG"||s==="VIDEO"||s==="CANVAS"||s==="SOURCE")return!1}return vh(e)&&Xt(t)?!1:e in i}const yh=i=>{const e=i.props["onUpdate:modelValue"]||!1;return Ke(e)?t=>ll(e,t):e};function FA(i){i.target.composing=!0}function bh(i){const e=i.target;e.composing&&(e.composing=!1,e.dispatchEvent(new Event("input")))}const _c=Symbol("_assign");function Mh(i,e,t){return e&&(i=i.trim()),t&&(i=Pf(i)),i}const LA={created(i,{modifiers:{lazy:e,trim:t,number:n}},s){i[_c]=yh(s);const r=n||s.props&&s.props.type==="number";kr(i,e?"change":"input",o=>{o.target.composing||i[_c](Mh(i.value,t,r))}),(t||r)&&kr(i,"change",()=>{i.value=Mh(i.value,t,r)}),e||(kr(i,"compositionstart",FA),kr(i,"compositionend",bh),kr(i,"change",bh))},mounted(i,{value:e}){i.value=e??""},beforeUpdate(i,{value:e,oldValue:t,modifiers:{lazy:n,trim:s,number:r}},o){if(i[_c]=yh(o),i.composing)return;const a=(r||i.type==="number")&&!/^0\d/.test(i.value)?Pf(i.value):i.value,l=e??"";a!==l&&(document.activeElement===i&&i.type!=="range"&&(n&&e===t||s&&i.value.trim()===l)||(i.value=l))}},BA=["ctrl","shift","alt","meta"],UA={stop:i=>i.stopPropagation(),prevent:i=>i.preventDefault(),self:i=>i.target!==i.currentTarget,ctrl:i=>!i.ctrlKey,shift:i=>!i.shiftKey,alt:i=>!i.altKey,meta:i=>!i.metaKey,left:i=>"button"in i&&i.button!==0,middle:i=>"button"in i&&i.button!==1,right:i=>"button"in i&&i.button!==2,exact:(i,e)=>BA.some(t=>i[`${t}Key`]&&!e.includes(t))},Ar=(i,e)=>{if(!i)return i;const t=i._withMods||(i._withMods={}),n=e.join(".");return t[n]||(t[n]=((s,...r)=>{for(let o=0;o<e.length;o++){const a=UA[e[o]];if(a&&a(s,e))return}return i(s,...r)}))},OA={esc:"escape",space:" ",up:"arrow-up",left:"arrow-left",right:"arrow-right",down:"arrow-down",delete:"backspace"},NA=(i,e)=>{const t=i._withKeys||(i._withKeys={}),n=e.join(".");return t[n]||(t[n]=(s=>{if(!("key"in s))return;const r=ks(s.key);if(e.some(o=>o===r||OA[o]===r))return i(s)}))},zA=ln({patchProp:DA},gA);let Ch;function kA(){return Ch||(Ch=Q_(zA))}const HA=((...i)=>{const e=kA().createApp(...i),{mount:t}=e;return e.mount=n=>{const s=GA(n);if(!s)return;const r=e._component;!$e(r)&&!r.render&&!r.template&&(r.template=s.innerHTML),s.nodeType===1&&(s.textContent="");const o=t(s,!1,VA(s));return s instanceof Element&&(s.removeAttribute("v-cloak"),s.setAttribute("data-v-app","")),o},e});function VA(i){if(i instanceof SVGElement)return"svg";if(typeof MathMLElement=="function"&&i instanceof MathMLElement)return"mathml"}function GA(i){return Xt(i)?document.querySelector(i):i}const Yf="181",Sr={ROTATE:0,DOLLY:1,PAN:2},vr={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},WA=0,Th=1,XA=2,M0=1,qA=2,qi=3,Bi=0,wn=1,ti=2,is=0,Is=1,Eh=2,wh=3,Rh=4,C0=5,nr=100,QA=101,YA=102,KA=103,jA=104,$A=200,ZA=201,JA=202,eS=203,sa=204,ra=205,tS=206,nS=207,iS=208,sS=209,rS=210,oS=211,aS=212,lS=213,cS=214,bu=0,Mu=1,Cu=2,ao=3,Tu=4,Eu=5,wu=6,Ru=7,T0=0,uS=1,fS=2,Ds=0,dS=1,hS=2,pS=3,mS=4,gS=5,xS=6,_S=7,E0=300,lo=301,co=302,Iu=303,Du=304,jl=306,Pu=1e3,ns=1001,Fu=1002,qn=1003,AS=1004,Da=1005,ii=1006,Ac=1007,sr=1008,Ui=1009,w0=1010,R0=1011,oa=1012,Kf=1013,si=1014,mi=1015,pr=1016,jf=1017,$f=1018,aa=1020,I0=35902,D0=35899,P0=1021,F0=1022,gn=1023,uo=1026,la=1027,L0=1028,$l=1029,Zf=1030,Jf=1031,$r=1033,dl=33776,hl=33777,pl=33778,ml=33779,Lu=35840,Bu=35841,Uu=35842,Ou=35843,Nu=36196,zu=37492,ku=37496,Hu=37808,Vu=37809,Gu=37810,Wu=37811,Xu=37812,qu=37813,Qu=37814,Yu=37815,Ku=37816,ju=37817,$u=37818,Zu=37819,Ju=37820,ef=37821,tf=36492,nf=36494,sf=36495,rf=36283,of=36284,af=36285,lf=36286,SS=3200,vS=3201,yS=0,bS=1,bs="",Jn="srgb",fo="srgb-linear",El="linear",ht="srgb",yr=7680,Ih=519,MS=512,CS=513,TS=514,B0=515,ES=516,wS=517,RS=518,IS=519,Dh=35044,DS=35048,Ph="300 es",Ei=2e3,wl=2001;function U0(i){for(let e=i.length-1;e>=0;--e)if(i[e]>=65535)return!0;return!1}function Rl(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function PS(){const i=Rl("canvas");return i.style.display="block",i}const Fh={};function Lh(...i){const e="THREE."+i.shift();console.log(e,...i)}function je(...i){const e="THREE."+i.shift();console.warn(e,...i)}function zt(...i){const e="THREE."+i.shift();console.error(e,...i)}function ca(...i){const e=i.join(" ");e in Fh||(Fh[e]=!0,je(...i))}function FS(i,e,t){return new Promise(function(n,s){function r(){switch(i.clientWaitSync(e,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:s();break;case i.TIMEOUT_EXPIRED:setTimeout(r,t);break;default:n()}}setTimeout(r,t)})}class mr{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[e]===void 0&&(n[e]=[]),n[e].indexOf(t)===-1&&n[e].push(t)}hasEventListener(e,t){const n=this._listeners;return n===void 0?!1:n[e]!==void 0&&n[e].indexOf(t)!==-1}removeEventListener(e,t){const n=this._listeners;if(n===void 0)return;const s=n[e];if(s!==void 0){const r=s.indexOf(t);r!==-1&&s.splice(r,1)}}dispatchEvent(e){const t=this._listeners;if(t===void 0)return;const n=t[e.type];if(n!==void 0){e.target=this;const s=n.slice(0);for(let r=0,o=s.length;r<o;r++)s[r].call(this,e);e.target=null}}}const en=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],gl=Math.PI/180,cf=180/Math.PI;function ya(){const i=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(en[i&255]+en[i>>8&255]+en[i>>16&255]+en[i>>24&255]+"-"+en[e&255]+en[e>>8&255]+"-"+en[e>>16&15|64]+en[e>>24&255]+"-"+en[t&63|128]+en[t>>8&255]+"-"+en[t>>16&255]+en[t>>24&255]+en[n&255]+en[n>>8&255]+en[n>>16&255]+en[n>>24&255]).toLowerCase()}function Je(i,e,t){return Math.max(e,Math.min(t,i))}function LS(i,e){return(i%e+e)%e}function Sc(i,e,t){return(1-t)*i+t*e}function Eo(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return i/4294967295;case Uint16Array:return i/65535;case Uint8Array:return i/255;case Int32Array:return Math.max(i/2147483647,-1);case Int16Array:return Math.max(i/32767,-1);case Int8Array:return Math.max(i/127,-1);default:throw new Error("Invalid component type.")}}function bn(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return Math.round(i*4294967295);case Uint16Array:return Math.round(i*65535);case Uint8Array:return Math.round(i*255);case Int32Array:return Math.round(i*2147483647);case Int16Array:return Math.round(i*32767);case Int8Array:return Math.round(i*127);default:throw new Error("Invalid component type.")}}const O0={DEG2RAD:gl};class ze{constructor(e=0,t=0){ze.prototype.isVector2=!0,this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const t=this.x,n=this.y,s=e.elements;return this.x=s[0]*t+s[3]*n+s[6],this.y=s[1]*t+s[4]*n+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=Je(this.x,e.x,t.x),this.y=Je(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=Je(this.x,e,t),this.y=Je(this.y,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Je(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(Je(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y;return t*t+n*n}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){const n=Math.cos(t),s=Math.sin(t),r=this.x-e.x,o=this.y-e.y;return this.x=r*n-o*s+e.x,this.y=r*s+o*n+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class bt{constructor(e=0,t=0,n=0,s=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=n,this._w=s}static slerpFlat(e,t,n,s,r,o,a){let l=n[s+0],c=n[s+1],u=n[s+2],f=n[s+3],d=r[o+0],h=r[o+1],x=r[o+2],p=r[o+3];if(a<=0){e[t+0]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f;return}if(a>=1){e[t+0]=d,e[t+1]=h,e[t+2]=x,e[t+3]=p;return}if(f!==p||l!==d||c!==h||u!==x){let g=l*d+c*h+u*x+f*p;g<0&&(d=-d,h=-h,x=-x,p=-p,g=-g);let m=1-a;if(g<.9995){const _=Math.acos(g),A=Math.sin(_);m=Math.sin(m*_)/A,a=Math.sin(a*_)/A,l=l*m+d*a,c=c*m+h*a,u=u*m+x*a,f=f*m+p*a}else{l=l*m+d*a,c=c*m+h*a,u=u*m+x*a,f=f*m+p*a;const _=1/Math.sqrt(l*l+c*c+u*u+f*f);l*=_,c*=_,u*=_,f*=_}}e[t]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f}static multiplyQuaternionsFlat(e,t,n,s,r,o){const a=n[s],l=n[s+1],c=n[s+2],u=n[s+3],f=r[o],d=r[o+1],h=r[o+2],x=r[o+3];return e[t]=a*x+u*f+l*h-c*d,e[t+1]=l*x+u*d+c*f-a*h,e[t+2]=c*x+u*h+a*d-l*f,e[t+3]=u*x-a*f-l*d-c*h,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,n,s){return this._x=e,this._y=t,this._z=n,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){const n=e._x,s=e._y,r=e._z,o=e._order,a=Math.cos,l=Math.sin,c=a(n/2),u=a(s/2),f=a(r/2),d=l(n/2),h=l(s/2),x=l(r/2);switch(o){case"XYZ":this._x=d*u*f+c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f-d*h*x;break;case"YXZ":this._x=d*u*f+c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f+d*h*x;break;case"ZXY":this._x=d*u*f-c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f-d*h*x;break;case"ZYX":this._x=d*u*f-c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f+d*h*x;break;case"YZX":this._x=d*u*f+c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f-d*h*x;break;case"XZY":this._x=d*u*f-c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f+d*h*x;break;default:je("Quaternion: .setFromEuler() encountered an unknown order: "+o)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){const n=t/2,s=Math.sin(n);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(e){const t=e.elements,n=t[0],s=t[4],r=t[8],o=t[1],a=t[5],l=t[9],c=t[2],u=t[6],f=t[10],d=n+a+f;if(d>0){const h=.5/Math.sqrt(d+1);this._w=.25/h,this._x=(u-l)*h,this._y=(r-c)*h,this._z=(o-s)*h}else if(n>a&&n>f){const h=2*Math.sqrt(1+n-a-f);this._w=(u-l)/h,this._x=.25*h,this._y=(s+o)/h,this._z=(r+c)/h}else if(a>f){const h=2*Math.sqrt(1+a-n-f);this._w=(r-c)/h,this._x=(s+o)/h,this._y=.25*h,this._z=(l+u)/h}else{const h=2*Math.sqrt(1+f-n-a);this._w=(o-s)/h,this._x=(r+c)/h,this._y=(l+u)/h,this._z=.25*h}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let n=e.dot(t)+1;return n<1e-8?(n=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=n):(this._x=0,this._y=-e.z,this._z=e.y,this._w=n)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=n),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(Je(this.dot(e),-1,1)))}rotateTowards(e,t){const n=this.angleTo(e);if(n===0)return this;const s=Math.min(1,t/n);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){const n=e._x,s=e._y,r=e._z,o=e._w,a=t._x,l=t._y,c=t._z,u=t._w;return this._x=n*u+o*a+s*c-r*l,this._y=s*u+o*l+r*a-n*c,this._z=r*u+o*c+n*l-s*a,this._w=o*u-n*a-s*l-r*c,this._onChangeCallback(),this}slerp(e,t){if(t<=0)return this;if(t>=1)return this.copy(e);let n=e._x,s=e._y,r=e._z,o=e._w,a=this.dot(e);a<0&&(n=-n,s=-s,r=-r,o=-o,a=-a);let l=1-t;if(a<.9995){const c=Math.acos(a),u=Math.sin(c);l=Math.sin(l*c)/u,t=Math.sin(t*c)/u,this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this._onChangeCallback()}else this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this.normalize();return this}slerpQuaternions(e,t,n){return this.copy(e).slerp(t,n)}random(){const e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),n=Math.random(),s=Math.sqrt(1-n),r=Math.sqrt(n);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(t),r*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class B{constructor(e=0,t=0,n=0){B.prototype.isVector3=!0,this.x=e,this.y=t,this.z=n}set(e,t,n){return n===void 0&&(n=this.z),this.x=e,this.y=t,this.z=n,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(Bh.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(Bh.setFromAxisAngle(e,t))}applyMatrix3(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[3]*n+r[6]*s,this.y=r[1]*t+r[4]*n+r[7]*s,this.z=r[2]*t+r[5]*n+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=e.elements,o=1/(r[3]*t+r[7]*n+r[11]*s+r[15]);return this.x=(r[0]*t+r[4]*n+r[8]*s+r[12])*o,this.y=(r[1]*t+r[5]*n+r[9]*s+r[13])*o,this.z=(r[2]*t+r[6]*n+r[10]*s+r[14])*o,this}applyQuaternion(e){const t=this.x,n=this.y,s=this.z,r=e.x,o=e.y,a=e.z,l=e.w,c=2*(o*s-a*n),u=2*(a*t-r*s),f=2*(r*n-o*t);return this.x=t+l*c+o*f-a*u,this.y=n+l*u+a*c-r*f,this.z=s+l*f+r*u-o*c,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[4]*n+r[8]*s,this.y=r[1]*t+r[5]*n+r[9]*s,this.z=r[2]*t+r[6]*n+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=Je(this.x,e.x,t.x),this.y=Je(this.y,e.y,t.y),this.z=Je(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=Je(this.x,e,t),this.y=Je(this.y,e,t),this.z=Je(this.z,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Je(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){const n=e.x,s=e.y,r=e.z,o=t.x,a=t.y,l=t.z;return this.x=s*l-r*a,this.y=r*o-n*l,this.z=n*a-s*o,this}projectOnVector(e){const t=e.lengthSq();if(t===0)return this.set(0,0,0);const n=e.dot(this)/t;return this.copy(e).multiplyScalar(n)}projectOnPlane(e){return vc.copy(this).projectOnVector(e),this.sub(vc)}reflect(e){return this.sub(vc.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(Je(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y,s=this.z-e.z;return t*t+n*n+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,n){const s=Math.sin(t)*e;return this.x=s*Math.sin(n),this.y=Math.cos(t)*e,this.z=s*Math.cos(n),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,n){return this.x=e*Math.sin(t),this.y=n,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){const t=this.setFromMatrixColumn(e,0).length(),n=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=n,this.z=s,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,t=Math.random()*2-1,n=Math.sqrt(1-t*t);return this.x=n*Math.cos(e),this.y=t,this.z=n*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const vc=new B,Bh=new bt;class Qe{constructor(e,t,n,s,r,o,a,l,c){Qe.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c)}set(e,t,n,s,r,o,a,l,c){const u=this.elements;return u[0]=e,u[1]=s,u[2]=a,u[3]=t,u[4]=r,u[5]=l,u[6]=n,u[7]=o,u[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],this}extractBasis(e,t,n){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[3],l=n[6],c=n[1],u=n[4],f=n[7],d=n[2],h=n[5],x=n[8],p=s[0],g=s[3],m=s[6],_=s[1],A=s[4],S=s[7],v=s[2],y=s[5],M=s[8];return r[0]=o*p+a*_+l*v,r[3]=o*g+a*A+l*y,r[6]=o*m+a*S+l*M,r[1]=c*p+u*_+f*v,r[4]=c*g+u*A+f*y,r[7]=c*m+u*S+f*M,r[2]=d*p+h*_+x*v,r[5]=d*g+h*A+x*y,r[8]=d*m+h*S+x*M,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8];return t*o*u-t*a*c-n*r*u+n*a*l+s*r*c-s*o*l}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=u*o-a*c,d=a*l-u*r,h=c*r-o*l,x=t*f+n*d+s*h;if(x===0)return this.set(0,0,0,0,0,0,0,0,0);const p=1/x;return e[0]=f*p,e[1]=(s*c-u*n)*p,e[2]=(a*n-s*o)*p,e[3]=d*p,e[4]=(u*t-s*l)*p,e[5]=(s*r-a*t)*p,e[6]=h*p,e[7]=(n*l-c*t)*p,e[8]=(o*t-n*r)*p,this}transpose(){let e;const t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,n,s,r,o,a){const l=Math.cos(r),c=Math.sin(r);return this.set(n*l,n*c,-n*(l*o+c*a)+o+e,-s*c,s*l,-s*(-c*o+l*a)+a+t,0,0,1),this}scale(e,t){return this.premultiply(yc.makeScale(e,t)),this}rotate(e){return this.premultiply(yc.makeRotation(-e)),this}translate(e,t){return this.premultiply(yc.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,n,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<9;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<9;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e}clone(){return new this.constructor().fromArray(this.elements)}}const yc=new Qe,Uh=new Qe().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Oh=new Qe().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function BS(){const i={enabled:!0,workingColorSpace:fo,spaces:{},convert:function(s,r,o){return this.enabled===!1||r===o||!r||!o||(this.spaces[r].transfer===ht&&(s.r=ss(s.r),s.g=ss(s.g),s.b=ss(s.b)),this.spaces[r].primaries!==this.spaces[o].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[o].fromXYZ)),this.spaces[o].transfer===ht&&(s.r=Zr(s.r),s.g=Zr(s.g),s.b=Zr(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===bs?El:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,o){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[o].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return ca("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),i.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return ca("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),i.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],n=[.3127,.329];return i.define({[fo]:{primaries:e,whitePoint:n,transfer:El,toXYZ:Uh,fromXYZ:Oh,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:Jn},outputColorSpaceConfig:{drawingBufferColorSpace:Jn}},[Jn]:{primaries:e,whitePoint:n,transfer:ht,toXYZ:Uh,fromXYZ:Oh,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:Jn}}}),i}const rt=BS();function ss(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function Zr(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}let br;class US{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let n;if(e instanceof HTMLCanvasElement)n=e;else{br===void 0&&(br=Rl("canvas")),br.width=e.width,br.height=e.height;const s=br.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),n=br}return n.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const t=Rl("canvas");t.width=e.width,t.height=e.height;const n=t.getContext("2d");n.drawImage(e,0,0,e.width,e.height);const s=n.getImageData(0,0,e.width,e.height),r=s.data;for(let o=0;o<r.length;o++)r[o]=ss(r[o]/255)*255;return n.putImageData(s,0,0),t}else if(e.data){const t=e.data.slice(0);for(let n=0;n<t.length;n++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[n]=Math.floor(ss(t[n]/255)*255):t[n]=ss(t[n]);return{data:t,width:e.width,height:e.height}}else return je("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let OS=0;class ed{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:OS++}),this.uuid=ya(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):t instanceof VideoFrame?e.set(t.displayHeight,t.displayWidth,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const n={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let o=0,a=s.length;o<a;o++)s[o].isDataTexture?r.push(bc(s[o].image)):r.push(bc(s[o]))}else r=bc(s);n.url=r}return t||(e.images[this.uuid]=n),n}}function bc(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?US.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(je("Texture: Unable to serialize Texture."),{})}let NS=0;const Mc=new B;class xn extends mr{constructor(e=xn.DEFAULT_IMAGE,t=xn.DEFAULT_MAPPING,n=ns,s=ns,r=ii,o=sr,a=gn,l=Ui,c=xn.DEFAULT_ANISOTROPY,u=bs){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:NS++}),this.uuid=ya(),this.name="",this.source=new ed(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=n,this.wrapT=s,this.magFilter=r,this.minFilter=o,this.anisotropy=c,this.format=a,this.internalFormat=null,this.type=l,this.offset=new ze(0,0),this.repeat=new ze(1,1),this.center=new ze(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Qe,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=u,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0}get width(){return this.source.getSize(Mc).x}get height(){return this.source.getSize(Mc).y}get depth(){return this.source.getSize(Mc).z}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const t in e){const n=e[t];if(n===void 0){je(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){je(`Texture.setValues(): property '${t}' does not exist.`);continue}s&&n&&s.isVector2&&n.isVector2||s&&n&&s.isVector3&&n.isVector3||s&&n&&s.isMatrix3&&n.isMatrix3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const n={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),t||(e.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==E0)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case Pu:e.x=e.x-Math.floor(e.x);break;case ns:e.x=e.x<0?0:1;break;case Fu:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case Pu:e.y=e.y-Math.floor(e.y);break;case ns:e.y=e.y<0?0:1;break;case Fu:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}xn.DEFAULT_IMAGE=null;xn.DEFAULT_MAPPING=E0;xn.DEFAULT_ANISOTROPY=1;class Et{constructor(e=0,t=0,n=0,s=1){Et.prototype.isVector4=!0,this.x=e,this.y=t,this.z=n,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,n,s){return this.x=e,this.y=t,this.z=n,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=this.w,o=e.elements;return this.x=o[0]*t+o[4]*n+o[8]*s+o[12]*r,this.y=o[1]*t+o[5]*n+o[9]*s+o[13]*r,this.z=o[2]*t+o[6]*n+o[10]*s+o[14]*r,this.w=o[3]*t+o[7]*n+o[11]*s+o[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,n,s,r;const l=e.elements,c=l[0],u=l[4],f=l[8],d=l[1],h=l[5],x=l[9],p=l[2],g=l[6],m=l[10];if(Math.abs(u-d)<.01&&Math.abs(f-p)<.01&&Math.abs(x-g)<.01){if(Math.abs(u+d)<.1&&Math.abs(f+p)<.1&&Math.abs(x+g)<.1&&Math.abs(c+h+m-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;const A=(c+1)/2,S=(h+1)/2,v=(m+1)/2,y=(u+d)/4,M=(f+p)/4,E=(x+g)/4;return A>S&&A>v?A<.01?(n=0,s=.707106781,r=.707106781):(n=Math.sqrt(A),s=y/n,r=M/n):S>v?S<.01?(n=.707106781,s=0,r=.707106781):(s=Math.sqrt(S),n=y/s,r=E/s):v<.01?(n=.707106781,s=.707106781,r=0):(r=Math.sqrt(v),n=M/r,s=E/r),this.set(n,s,r,t),this}let _=Math.sqrt((g-x)*(g-x)+(f-p)*(f-p)+(d-u)*(d-u));return Math.abs(_)<.001&&(_=1),this.x=(g-x)/_,this.y=(f-p)/_,this.z=(d-u)/_,this.w=Math.acos((c+h+m-1)/2),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=Je(this.x,e.x,t.x),this.y=Je(this.y,e.y,t.y),this.z=Je(this.z,e.z,t.z),this.w=Je(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=Je(this.x,e,t),this.y=Je(this.y,e,t),this.z=Je(this.z,e,t),this.w=Je(this.w,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Je(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this.w=e.w+(t.w-e.w)*n,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class zS extends mr{constructor(e=1,t=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:ii,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},n),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=n.depth,this.scissor=new Et(0,0,e,t),this.scissorTest=!1,this.viewport=new Et(0,0,e,t);const s={width:e,height:t,depth:n.depth},r=new xn(s);this.textures=[];const o=n.count;for(let a=0;a<o;a++)this.textures[a]=r.clone(),this.textures[a].isRenderTargetTexture=!0,this.textures[a].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview}_setTextureOptions(e={}){const t={minFilter:ii,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let n=0;n<this.textures.length;n++)this.textures[n].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,n=1){if(this.width!==e||this.height!==t||this.depth!==n){this.width=e,this.height=t,this.depth=n;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=t,this.textures[s].image.depth=n,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,n=e.textures.length;t<n;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;const s=Object.assign({},e.textures[t].image);this.textures[t].source=new ed(s)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class Us extends zS{constructor(e=1,t=1,n={}){super(e,t,n),this.isWebGLRenderTarget=!0}}class N0 extends xn{constructor(e=null,t=1,n=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=qn,this.minFilter=qn,this.wrapR=ns,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class kS extends xn{constructor(e=null,t=1,n=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=qn,this.minFilter=qn,this.wrapR=ns,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class wi{constructor(e=new B(1/0,1/0,1/0),t=new B(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t+=3)this.expandByPoint(ui.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,n=e.count;t<n;t++)this.expandByPoint(ui.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){const n=ui.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(n),this.max.copy(e).add(n),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);const n=e.geometry;if(n!==void 0){const r=n.getAttribute("position");if(t===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let o=0,a=r.count;o<a;o++)e.isMesh===!0?e.getVertexPosition(o,ui):ui.fromBufferAttribute(r,o),ui.applyMatrix4(e.matrixWorld),this.expandByPoint(ui);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),Pa.copy(e.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),Pa.copy(n.boundingBox)),Pa.applyMatrix4(e.matrixWorld),this.union(Pa)}const s=e.children;for(let r=0,o=s.length;r<o;r++)this.expandByObject(s[r],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,ui),ui.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,n;return e.normal.x>0?(t=e.normal.x*this.min.x,n=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,n=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,n+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,n+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,n+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,n+=e.normal.z*this.min.z),t<=-e.constant&&n>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(wo),Fa.subVectors(this.max,wo),Mr.subVectors(e.a,wo),Cr.subVectors(e.b,wo),Tr.subVectors(e.c,wo),ds.subVectors(Cr,Mr),hs.subVectors(Tr,Cr),qs.subVectors(Mr,Tr);let t=[0,-ds.z,ds.y,0,-hs.z,hs.y,0,-qs.z,qs.y,ds.z,0,-ds.x,hs.z,0,-hs.x,qs.z,0,-qs.x,-ds.y,ds.x,0,-hs.y,hs.x,0,-qs.y,qs.x,0];return!Cc(t,Mr,Cr,Tr,Fa)||(t=[1,0,0,0,1,0,0,0,1],!Cc(t,Mr,Cr,Tr,Fa))?!1:(La.crossVectors(ds,hs),t=[La.x,La.y,La.z],Cc(t,Mr,Cr,Tr,Fa))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,ui).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(ui).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(zi[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),zi[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),zi[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),zi[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),zi[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),zi[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),zi[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),zi[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(zi),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const zi=[new B,new B,new B,new B,new B,new B,new B,new B],ui=new B,Pa=new wi,Mr=new B,Cr=new B,Tr=new B,ds=new B,hs=new B,qs=new B,wo=new B,Fa=new B,La=new B,Qs=new B;function Cc(i,e,t,n,s){for(let r=0,o=i.length-3;r<=o;r+=3){Qs.fromArray(i,r);const a=s.x*Math.abs(Qs.x)+s.y*Math.abs(Qs.y)+s.z*Math.abs(Qs.z),l=e.dot(Qs),c=t.dot(Qs),u=n.dot(Qs);if(Math.max(-Math.max(l,c,u),Math.min(l,c,u))>a)return!1}return!0}const HS=new wi,Ro=new B,Tc=new B;class Zl{constructor(e=new B,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){const n=this.center;t!==void 0?n.copy(t):HS.setFromPoints(e).getCenter(n);let s=0;for(let r=0,o=e.length;r<o;r++)s=Math.max(s,n.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){const n=this.center.distanceToSquared(e);return t.copy(e),n>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;Ro.subVectors(e,this.center);const t=Ro.lengthSq();if(t>this.radius*this.radius){const n=Math.sqrt(t),s=(n-this.radius)*.5;this.center.addScaledVector(Ro,s/n),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Tc.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(Ro.copy(e.center).add(Tc)),this.expandByPoint(Ro.copy(e.center).sub(Tc))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}const ki=new B,Ec=new B,Ba=new B,ps=new B,wc=new B,Ua=new B,Rc=new B;let td=class{constructor(e=new B,t=new B(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,ki)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);const n=t.dot(this.direction);return n<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const t=ki.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(ki.copy(this.origin).addScaledVector(this.direction,t),ki.distanceToSquared(e))}distanceSqToSegment(e,t,n,s){Ec.copy(e).add(t).multiplyScalar(.5),Ba.copy(t).sub(e).normalize(),ps.copy(this.origin).sub(Ec);const r=e.distanceTo(t)*.5,o=-this.direction.dot(Ba),a=ps.dot(this.direction),l=-ps.dot(Ba),c=ps.lengthSq(),u=Math.abs(1-o*o);let f,d,h,x;if(u>0)if(f=o*l-a,d=o*a-l,x=r*u,f>=0)if(d>=-x)if(d<=x){const p=1/u;f*=p,d*=p,h=f*(f+o*d+2*a)+d*(o*f+d+2*l)+c}else d=r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;else d=-r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;else d<=-x?(f=Math.max(0,-(-o*r+a)),d=f>0?-r:Math.min(Math.max(-r,-l),r),h=-f*f+d*(d+2*l)+c):d<=x?(f=0,d=Math.min(Math.max(-r,-l),r),h=d*(d+2*l)+c):(f=Math.max(0,-(o*r+a)),d=f>0?r:Math.min(Math.max(-r,-l),r),h=-f*f+d*(d+2*l)+c);else d=o>0?-r:r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;return n&&n.copy(this.origin).addScaledVector(this.direction,f),s&&s.copy(Ec).addScaledVector(Ba,d),h}intersectSphere(e,t){ki.subVectors(e.center,this.origin);const n=ki.dot(this.direction),s=ki.dot(ki)-n*n,r=e.radius*e.radius;if(s>r)return null;const o=Math.sqrt(r-s),a=n-o,l=n+o;return l<0?null:a<0?this.at(l,t):this.at(a,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(e.normal)+e.constant)/t;return n>=0?n:null}intersectPlane(e,t){const n=this.distanceToPlane(e);return n===null?null:this.at(n,t)}intersectsPlane(e){const t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let n,s,r,o,a,l;const c=1/this.direction.x,u=1/this.direction.y,f=1/this.direction.z,d=this.origin;return c>=0?(n=(e.min.x-d.x)*c,s=(e.max.x-d.x)*c):(n=(e.max.x-d.x)*c,s=(e.min.x-d.x)*c),u>=0?(r=(e.min.y-d.y)*u,o=(e.max.y-d.y)*u):(r=(e.max.y-d.y)*u,o=(e.min.y-d.y)*u),n>o||r>s||((r>n||isNaN(n))&&(n=r),(o<s||isNaN(s))&&(s=o),f>=0?(a=(e.min.z-d.z)*f,l=(e.max.z-d.z)*f):(a=(e.max.z-d.z)*f,l=(e.min.z-d.z)*f),n>l||a>s)||((a>n||n!==n)&&(n=a),(l<s||s!==s)&&(s=l),s<0)?null:this.at(n>=0?n:s,t)}intersectsBox(e){return this.intersectBox(e,ki)!==null}intersectTriangle(e,t,n,s,r){wc.subVectors(t,e),Ua.subVectors(n,e),Rc.crossVectors(wc,Ua);let o=this.direction.dot(Rc),a;if(o>0){if(s)return null;a=1}else if(o<0)a=-1,o=-o;else return null;ps.subVectors(this.origin,e);const l=a*this.direction.dot(Ua.crossVectors(ps,Ua));if(l<0)return null;const c=a*this.direction.dot(wc.cross(ps));if(c<0||l+c>o)return null;const u=-a*ps.dot(Rc);return u<0?null:this.at(u/o,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}};class qe{constructor(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g){qe.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g)}set(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g){const m=this.elements;return m[0]=e,m[4]=t,m[8]=n,m[12]=s,m[1]=r,m[5]=o,m[9]=a,m[13]=l,m[2]=c,m[6]=u,m[10]=f,m[14]=d,m[3]=h,m[7]=x,m[11]=p,m[15]=g,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new qe().fromArray(this.elements)}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],this}copyPosition(e){const t=this.elements,n=e.elements;return t[12]=n[12],t[13]=n[13],t[14]=n[14],this}setFromMatrix3(e){const t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,n){return e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this}makeBasis(e,t,n){return this.set(e.x,t.x,n.x,0,e.y,t.y,n.y,0,e.z,t.z,n.z,0,0,0,0,1),this}extractRotation(e){const t=this.elements,n=e.elements,s=1/Er.setFromMatrixColumn(e,0).length(),r=1/Er.setFromMatrixColumn(e,1).length(),o=1/Er.setFromMatrixColumn(e,2).length();return t[0]=n[0]*s,t[1]=n[1]*s,t[2]=n[2]*s,t[3]=0,t[4]=n[4]*r,t[5]=n[5]*r,t[6]=n[6]*r,t[7]=0,t[8]=n[8]*o,t[9]=n[9]*o,t[10]=n[10]*o,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){const t=this.elements,n=e.x,s=e.y,r=e.z,o=Math.cos(n),a=Math.sin(n),l=Math.cos(s),c=Math.sin(s),u=Math.cos(r),f=Math.sin(r);if(e.order==="XYZ"){const d=o*u,h=o*f,x=a*u,p=a*f;t[0]=l*u,t[4]=-l*f,t[8]=c,t[1]=h+x*c,t[5]=d-p*c,t[9]=-a*l,t[2]=p-d*c,t[6]=x+h*c,t[10]=o*l}else if(e.order==="YXZ"){const d=l*u,h=l*f,x=c*u,p=c*f;t[0]=d+p*a,t[4]=x*a-h,t[8]=o*c,t[1]=o*f,t[5]=o*u,t[9]=-a,t[2]=h*a-x,t[6]=p+d*a,t[10]=o*l}else if(e.order==="ZXY"){const d=l*u,h=l*f,x=c*u,p=c*f;t[0]=d-p*a,t[4]=-o*f,t[8]=x+h*a,t[1]=h+x*a,t[5]=o*u,t[9]=p-d*a,t[2]=-o*c,t[6]=a,t[10]=o*l}else if(e.order==="ZYX"){const d=o*u,h=o*f,x=a*u,p=a*f;t[0]=l*u,t[4]=x*c-h,t[8]=d*c+p,t[1]=l*f,t[5]=p*c+d,t[9]=h*c-x,t[2]=-c,t[6]=a*l,t[10]=o*l}else if(e.order==="YZX"){const d=o*l,h=o*c,x=a*l,p=a*c;t[0]=l*u,t[4]=p-d*f,t[8]=x*f+h,t[1]=f,t[5]=o*u,t[9]=-a*u,t[2]=-c*u,t[6]=h*f+x,t[10]=d-p*f}else if(e.order==="XZY"){const d=o*l,h=o*c,x=a*l,p=a*c;t[0]=l*u,t[4]=-f,t[8]=c*u,t[1]=d*f+p,t[5]=o*u,t[9]=h*f-x,t[2]=x*f-h,t[6]=a*u,t[10]=p*f+d}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(VS,e,GS)}lookAt(e,t,n){const s=this.elements;return Un.subVectors(e,t),Un.lengthSq()===0&&(Un.z=1),Un.normalize(),ms.crossVectors(n,Un),ms.lengthSq()===0&&(Math.abs(n.z)===1?Un.x+=1e-4:Un.z+=1e-4,Un.normalize(),ms.crossVectors(n,Un)),ms.normalize(),Oa.crossVectors(Un,ms),s[0]=ms.x,s[4]=Oa.x,s[8]=Un.x,s[1]=ms.y,s[5]=Oa.y,s[9]=Un.y,s[2]=ms.z,s[6]=Oa.z,s[10]=Un.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[4],l=n[8],c=n[12],u=n[1],f=n[5],d=n[9],h=n[13],x=n[2],p=n[6],g=n[10],m=n[14],_=n[3],A=n[7],S=n[11],v=n[15],y=s[0],M=s[4],E=s[8],b=s[12],C=s[1],I=s[5],F=s[9],U=s[13],O=s[2],k=s[6],z=s[10],V=s[14],H=s[3],$=s[7],oe=s[11],Se=s[15];return r[0]=o*y+a*C+l*O+c*H,r[4]=o*M+a*I+l*k+c*$,r[8]=o*E+a*F+l*z+c*oe,r[12]=o*b+a*U+l*V+c*Se,r[1]=u*y+f*C+d*O+h*H,r[5]=u*M+f*I+d*k+h*$,r[9]=u*E+f*F+d*z+h*oe,r[13]=u*b+f*U+d*V+h*Se,r[2]=x*y+p*C+g*O+m*H,r[6]=x*M+p*I+g*k+m*$,r[10]=x*E+p*F+g*z+m*oe,r[14]=x*b+p*U+g*V+m*Se,r[3]=_*y+A*C+S*O+v*H,r[7]=_*M+A*I+S*k+v*$,r[11]=_*E+A*F+S*z+v*oe,r[15]=_*b+A*U+S*V+v*Se,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[4],s=e[8],r=e[12],o=e[1],a=e[5],l=e[9],c=e[13],u=e[2],f=e[6],d=e[10],h=e[14],x=e[3],p=e[7],g=e[11],m=e[15];return x*(+r*l*f-s*c*f-r*a*d+n*c*d+s*a*h-n*l*h)+p*(+t*l*h-t*c*d+r*o*d-s*o*h+s*c*u-r*l*u)+g*(+t*c*f-t*a*h-r*o*f+n*o*h+r*a*u-n*c*u)+m*(-s*a*u-t*l*f+t*a*d+s*o*f-n*o*d+n*l*u)}transpose(){const e=this.elements;let t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,n){const s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=t,s[14]=n),this}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=e[9],d=e[10],h=e[11],x=e[12],p=e[13],g=e[14],m=e[15],_=f*g*c-p*d*c+p*l*h-a*g*h-f*l*m+a*d*m,A=x*d*c-u*g*c-x*l*h+o*g*h+u*l*m-o*d*m,S=u*p*c-x*f*c+x*a*h-o*p*h-u*a*m+o*f*m,v=x*f*l-u*p*l-x*a*d+o*p*d+u*a*g-o*f*g,y=t*_+n*A+s*S+r*v;if(y===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const M=1/y;return e[0]=_*M,e[1]=(p*d*r-f*g*r-p*s*h+n*g*h+f*s*m-n*d*m)*M,e[2]=(a*g*r-p*l*r+p*s*c-n*g*c-a*s*m+n*l*m)*M,e[3]=(f*l*r-a*d*r-f*s*c+n*d*c+a*s*h-n*l*h)*M,e[4]=A*M,e[5]=(u*g*r-x*d*r+x*s*h-t*g*h-u*s*m+t*d*m)*M,e[6]=(x*l*r-o*g*r-x*s*c+t*g*c+o*s*m-t*l*m)*M,e[7]=(o*d*r-u*l*r+u*s*c-t*d*c-o*s*h+t*l*h)*M,e[8]=S*M,e[9]=(x*f*r-u*p*r-x*n*h+t*p*h+u*n*m-t*f*m)*M,e[10]=(o*p*r-x*a*r+x*n*c-t*p*c-o*n*m+t*a*m)*M,e[11]=(u*a*r-o*f*r-u*n*c+t*f*c+o*n*h-t*a*h)*M,e[12]=v*M,e[13]=(u*p*s-x*f*s+x*n*d-t*p*d-u*n*g+t*f*g)*M,e[14]=(x*a*s-o*p*s-x*n*l+t*p*l+o*n*g-t*a*g)*M,e[15]=(o*f*s-u*a*s+u*n*l-t*f*l-o*n*d+t*a*d)*M,this}scale(e){const t=this.elements,n=e.x,s=e.y,r=e.z;return t[0]*=n,t[4]*=s,t[8]*=r,t[1]*=n,t[5]*=s,t[9]*=r,t[2]*=n,t[6]*=s,t[10]*=r,t[3]*=n,t[7]*=s,t[11]*=r,this}getMaxScaleOnAxis(){const e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],n=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,n,s))}makeTranslation(e,t,n){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,n,0,0,0,1),this}makeRotationX(e){const t=Math.cos(e),n=Math.sin(e);return this.set(1,0,0,0,0,t,-n,0,0,n,t,0,0,0,0,1),this}makeRotationY(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,0,n,0,0,1,0,0,-n,0,t,0,0,0,0,1),this}makeRotationZ(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,0,n,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){const n=Math.cos(t),s=Math.sin(t),r=1-n,o=e.x,a=e.y,l=e.z,c=r*o,u=r*a;return this.set(c*o+n,c*a-s*l,c*l+s*a,0,c*a+s*l,u*a+n,u*l-s*o,0,c*l-s*a,u*l+s*o,r*l*l+n,0,0,0,0,1),this}makeScale(e,t,n){return this.set(e,0,0,0,0,t,0,0,0,0,n,0,0,0,0,1),this}makeShear(e,t,n,s,r,o){return this.set(1,n,r,0,e,1,o,0,t,s,1,0,0,0,0,1),this}compose(e,t,n){const s=this.elements,r=t._x,o=t._y,a=t._z,l=t._w,c=r+r,u=o+o,f=a+a,d=r*c,h=r*u,x=r*f,p=o*u,g=o*f,m=a*f,_=l*c,A=l*u,S=l*f,v=n.x,y=n.y,M=n.z;return s[0]=(1-(p+m))*v,s[1]=(h+S)*v,s[2]=(x-A)*v,s[3]=0,s[4]=(h-S)*y,s[5]=(1-(d+m))*y,s[6]=(g+_)*y,s[7]=0,s[8]=(x+A)*M,s[9]=(g-_)*M,s[10]=(1-(d+p))*M,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,t,n){const s=this.elements;let r=Er.set(s[0],s[1],s[2]).length();const o=Er.set(s[4],s[5],s[6]).length(),a=Er.set(s[8],s[9],s[10]).length();this.determinant()<0&&(r=-r),e.x=s[12],e.y=s[13],e.z=s[14],fi.copy(this);const c=1/r,u=1/o,f=1/a;return fi.elements[0]*=c,fi.elements[1]*=c,fi.elements[2]*=c,fi.elements[4]*=u,fi.elements[5]*=u,fi.elements[6]*=u,fi.elements[8]*=f,fi.elements[9]*=f,fi.elements[10]*=f,t.setFromRotationMatrix(fi),n.x=r,n.y=o,n.z=a,this}makePerspective(e,t,n,s,r,o,a=Ei,l=!1){const c=this.elements,u=2*r/(t-e),f=2*r/(n-s),d=(t+e)/(t-e),h=(n+s)/(n-s);let x,p;if(l)x=r/(o-r),p=o*r/(o-r);else if(a===Ei)x=-(o+r)/(o-r),p=-2*o*r/(o-r);else if(a===wl)x=-o/(o-r),p=-o*r/(o-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=d,c[12]=0,c[1]=0,c[5]=f,c[9]=h,c[13]=0,c[2]=0,c[6]=0,c[10]=x,c[14]=p,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(e,t,n,s,r,o,a=Ei,l=!1){const c=this.elements,u=2/(t-e),f=2/(n-s),d=-(t+e)/(t-e),h=-(n+s)/(n-s);let x,p;if(l)x=1/(o-r),p=o/(o-r);else if(a===Ei)x=-2/(o-r),p=-(o+r)/(o-r);else if(a===wl)x=-1/(o-r),p=-r/(o-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=0,c[12]=d,c[1]=0,c[5]=f,c[9]=0,c[13]=h,c[2]=0,c[6]=0,c[10]=x,c[14]=p,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<16;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<16;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e[t+9]=n[9],e[t+10]=n[10],e[t+11]=n[11],e[t+12]=n[12],e[t+13]=n[13],e[t+14]=n[14],e[t+15]=n[15],e}}const Er=new B,fi=new qe,VS=new B(0,0,0),GS=new B(1,1,1),ms=new B,Oa=new B,Un=new B,Nh=new qe,zh=new bt;class _i{constructor(e=0,t=0,n=0,s=_i.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=n,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,n,s=this._order){return this._x=e,this._y=t,this._z=n,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,n=!0){const s=e.elements,r=s[0],o=s[4],a=s[8],l=s[1],c=s[5],u=s[9],f=s[2],d=s[6],h=s[10];switch(t){case"XYZ":this._y=Math.asin(Je(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-u,h),this._z=Math.atan2(-o,r)):(this._x=Math.atan2(d,c),this._z=0);break;case"YXZ":this._x=Math.asin(-Je(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(a,h),this._z=Math.atan2(l,c)):(this._y=Math.atan2(-f,r),this._z=0);break;case"ZXY":this._x=Math.asin(Je(d,-1,1)),Math.abs(d)<.9999999?(this._y=Math.atan2(-f,h),this._z=Math.atan2(-o,c)):(this._y=0,this._z=Math.atan2(l,r));break;case"ZYX":this._y=Math.asin(-Je(f,-1,1)),Math.abs(f)<.9999999?(this._x=Math.atan2(d,h),this._z=Math.atan2(l,r)):(this._x=0,this._z=Math.atan2(-o,c));break;case"YZX":this._z=Math.asin(Je(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(-u,c),this._y=Math.atan2(-f,r)):(this._x=0,this._y=Math.atan2(a,h));break;case"XZY":this._z=Math.asin(-Je(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(d,c),this._y=Math.atan2(a,r)):(this._x=Math.atan2(-u,h),this._y=0);break;default:je("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,n===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,n){return Nh.makeRotationFromQuaternion(e),this.setFromRotationMatrix(Nh,t,n)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return zh.setFromEuler(this),this.setFromQuaternion(zh,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}_i.DEFAULT_ORDER="XYZ";class z0{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let WS=0;const kh=new B,wr=new bt,Hi=new qe,Na=new B,Io=new B,XS=new B,qS=new bt,Hh=new B(1,0,0),Vh=new B(0,1,0),Gh=new B(0,0,1),Wh={type:"added"},QS={type:"removed"},Rr={type:"childadded",child:null},Ic={type:"childremoved",child:null};class Wt extends mr{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:WS++}),this.uuid=ya(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=Wt.DEFAULT_UP.clone();const e=new B,t=new _i,n=new bt,s=new B(1,1,1);function r(){n.setFromEuler(t,!1)}function o(){t.setFromQuaternion(n,void 0,!1)}t._onChange(r),n._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new qe},normalMatrix:{value:new Qe}}),this.matrix=new qe,this.matrixWorld=new qe,this.matrixAutoUpdate=Wt.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=Wt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new z0,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return wr.setFromAxisAngle(e,t),this.quaternion.multiply(wr),this}rotateOnWorldAxis(e,t){return wr.setFromAxisAngle(e,t),this.quaternion.premultiply(wr),this}rotateX(e){return this.rotateOnAxis(Hh,e)}rotateY(e){return this.rotateOnAxis(Vh,e)}rotateZ(e){return this.rotateOnAxis(Gh,e)}translateOnAxis(e,t){return kh.copy(e).applyQuaternion(this.quaternion),this.position.add(kh.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(Hh,e)}translateY(e){return this.translateOnAxis(Vh,e)}translateZ(e){return this.translateOnAxis(Gh,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Hi.copy(this.matrixWorld).invert())}lookAt(e,t,n){e.isVector3?Na.copy(e):Na.set(e,t,n);const s=this.parent;this.updateWorldMatrix(!0,!1),Io.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Hi.lookAt(Io,Na,this.up):Hi.lookAt(Na,Io,this.up),this.quaternion.setFromRotationMatrix(Hi),s&&(Hi.extractRotation(s.matrixWorld),wr.setFromRotationMatrix(Hi),this.quaternion.premultiply(wr.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(zt("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(Wh),Rr.child=e,this.dispatchEvent(Rr),Rr.child=null):zt("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(QS),Ic.child=e,this.dispatchEvent(Ic),Ic.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Hi.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Hi.multiply(e.parent.matrixWorld)),e.applyMatrix4(Hi),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(Wh),Rr.child=e,this.dispatchEvent(Rr),Rr.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let n=0,s=this.children.length;n<s;n++){const o=this.children[n].getObjectByProperty(e,t);if(o!==void 0)return o}}getObjectsByProperty(e,t,n=[]){this[e]===t&&n.push(this);const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].getObjectsByProperty(e,t,n);return n}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Io,e,XS),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Io,qS,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverseVisible(e)}traverseAncestors(e){const t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].updateMatrixWorld(e)}updateWorldMatrix(e,t){const n=this.parent;if(e===!0&&n!==null&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),t===!0){const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].updateWorldMatrix(!1,!0)}}toJSON(e){const t=e===void 0||typeof e=="string",n={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(a=>({...a,boundingBox:a.boundingBox?a.boundingBox.toJSON():void 0,boundingSphere:a.boundingSphere?a.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(a=>({...a})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(a,l){return a[l.uuid]===void 0&&(a[l.uuid]=l.toJSON(e)),l.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);const a=this.geometry.parameters;if(a!==void 0&&a.shapes!==void 0){const l=a.shapes;if(Array.isArray(l))for(let c=0,u=l.length;c<u;c++){const f=l[c];r(e.shapes,f)}else r(e.shapes,l)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const a=[];for(let l=0,c=this.material.length;l<c;l++)a.push(r(e.materials,this.material[l]));s.material=a}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let a=0;a<this.children.length;a++)s.children.push(this.children[a].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let a=0;a<this.animations.length;a++){const l=this.animations[a];s.animations.push(r(e.animations,l))}}if(t){const a=o(e.geometries),l=o(e.materials),c=o(e.textures),u=o(e.images),f=o(e.shapes),d=o(e.skeletons),h=o(e.animations),x=o(e.nodes);a.length>0&&(n.geometries=a),l.length>0&&(n.materials=l),c.length>0&&(n.textures=c),u.length>0&&(n.images=u),f.length>0&&(n.shapes=f),d.length>0&&(n.skeletons=d),h.length>0&&(n.animations=h),x.length>0&&(n.nodes=x)}return n.object=s,n;function o(a){const l=[];for(const c in a){const u=a[c];delete u.metadata,l.push(u)}return l}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let n=0;n<e.children.length;n++){const s=e.children[n];this.add(s.clone())}return this}}Wt.DEFAULT_UP=new B(0,1,0);Wt.DEFAULT_MATRIX_AUTO_UPDATE=!0;Wt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const di=new B,Vi=new B,Dc=new B,Gi=new B,Ir=new B,Dr=new B,Xh=new B,Pc=new B,Fc=new B,Lc=new B,Bc=new Et,Uc=new Et,Oc=new Et;class pi{constructor(e=new B,t=new B,n=new B){this.a=e,this.b=t,this.c=n}static getNormal(e,t,n,s){s.subVectors(n,t),di.subVectors(e,t),s.cross(di);const r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,t,n,s,r){di.subVectors(s,t),Vi.subVectors(n,t),Dc.subVectors(e,t);const o=di.dot(di),a=di.dot(Vi),l=di.dot(Dc),c=Vi.dot(Vi),u=Vi.dot(Dc),f=o*c-a*a;if(f===0)return r.set(0,0,0),null;const d=1/f,h=(c*l-a*u)*d,x=(o*u-a*l)*d;return r.set(1-h-x,x,h)}static containsPoint(e,t,n,s){return this.getBarycoord(e,t,n,s,Gi)===null?!1:Gi.x>=0&&Gi.y>=0&&Gi.x+Gi.y<=1}static getInterpolation(e,t,n,s,r,o,a,l){return this.getBarycoord(e,t,n,s,Gi)===null?(l.x=0,l.y=0,"z"in l&&(l.z=0),"w"in l&&(l.w=0),null):(l.setScalar(0),l.addScaledVector(r,Gi.x),l.addScaledVector(o,Gi.y),l.addScaledVector(a,Gi.z),l)}static getInterpolatedAttribute(e,t,n,s,r,o){return Bc.setScalar(0),Uc.setScalar(0),Oc.setScalar(0),Bc.fromBufferAttribute(e,t),Uc.fromBufferAttribute(e,n),Oc.fromBufferAttribute(e,s),o.setScalar(0),o.addScaledVector(Bc,r.x),o.addScaledVector(Uc,r.y),o.addScaledVector(Oc,r.z),o}static isFrontFacing(e,t,n,s){return di.subVectors(n,t),Vi.subVectors(e,t),di.cross(Vi).dot(s)<0}set(e,t,n){return this.a.copy(e),this.b.copy(t),this.c.copy(n),this}setFromPointsAndIndices(e,t,n,s){return this.a.copy(e[t]),this.b.copy(e[n]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,t,n,s){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,n),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return di.subVectors(this.c,this.b),Vi.subVectors(this.a,this.b),di.cross(Vi).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return pi.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return pi.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,n,s,r){return pi.getInterpolation(e,this.a,this.b,this.c,t,n,s,r)}containsPoint(e){return pi.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return pi.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){const n=this.a,s=this.b,r=this.c;let o,a;Ir.subVectors(s,n),Dr.subVectors(r,n),Pc.subVectors(e,n);const l=Ir.dot(Pc),c=Dr.dot(Pc);if(l<=0&&c<=0)return t.copy(n);Fc.subVectors(e,s);const u=Ir.dot(Fc),f=Dr.dot(Fc);if(u>=0&&f<=u)return t.copy(s);const d=l*f-u*c;if(d<=0&&l>=0&&u<=0)return o=l/(l-u),t.copy(n).addScaledVector(Ir,o);Lc.subVectors(e,r);const h=Ir.dot(Lc),x=Dr.dot(Lc);if(x>=0&&h<=x)return t.copy(r);const p=h*c-l*x;if(p<=0&&c>=0&&x<=0)return a=c/(c-x),t.copy(n).addScaledVector(Dr,a);const g=u*x-h*f;if(g<=0&&f-u>=0&&h-x>=0)return Xh.subVectors(r,s),a=(f-u)/(f-u+(h-x)),t.copy(s).addScaledVector(Xh,a);const m=1/(g+p+d);return o=p*m,a=d*m,t.copy(n).addScaledVector(Ir,o).addScaledVector(Dr,a)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}const k0={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},gs={h:0,s:0,l:0},za={h:0,s:0,l:0};function Nc(i,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?i+(e-i)*6*t:t<1/2?e:t<2/3?i+(e-i)*6*(2/3-t):i}class nt{constructor(e,t,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,n)}set(e,t,n){if(t===void 0&&n===void 0){const s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,t,n);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=Jn){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,rt.colorSpaceToWorking(this,t),this}setRGB(e,t,n,s=rt.workingColorSpace){return this.r=e,this.g=t,this.b=n,rt.colorSpaceToWorking(this,s),this}setHSL(e,t,n,s=rt.workingColorSpace){if(e=LS(e,1),t=Je(t,0,1),n=Je(n,0,1),t===0)this.r=this.g=this.b=n;else{const r=n<=.5?n*(1+t):n+t-n*t,o=2*n-r;this.r=Nc(o,r,e+1/3),this.g=Nc(o,r,e),this.b=Nc(o,r,e-1/3)}return rt.colorSpaceToWorking(this,s),this}setStyle(e,t=Jn){function n(r){r!==void 0&&parseFloat(r)<1&&je("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r;const o=s[1],a=s[2];switch(o){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,t);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,t);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,t);break;default:je("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){const r=s[1],o=r.length;if(o===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,t);if(o===6)return this.setHex(parseInt(r,16),t);je("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=Jn){const n=k0[e.toLowerCase()];return n!==void 0?this.setHex(n,t):je("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=ss(e.r),this.g=ss(e.g),this.b=ss(e.b),this}copyLinearToSRGB(e){return this.r=Zr(e.r),this.g=Zr(e.g),this.b=Zr(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Jn){return rt.workingToColorSpace(tn.copy(this),e),Math.round(Je(tn.r*255,0,255))*65536+Math.round(Je(tn.g*255,0,255))*256+Math.round(Je(tn.b*255,0,255))}getHexString(e=Jn){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=rt.workingColorSpace){rt.workingToColorSpace(tn.copy(this),t);const n=tn.r,s=tn.g,r=tn.b,o=Math.max(n,s,r),a=Math.min(n,s,r);let l,c;const u=(a+o)/2;if(a===o)l=0,c=0;else{const f=o-a;switch(c=u<=.5?f/(o+a):f/(2-o-a),o){case n:l=(s-r)/f+(s<r?6:0);break;case s:l=(r-n)/f+2;break;case r:l=(n-s)/f+4;break}l/=6}return e.h=l,e.s=c,e.l=u,e}getRGB(e,t=rt.workingColorSpace){return rt.workingToColorSpace(tn.copy(this),t),e.r=tn.r,e.g=tn.g,e.b=tn.b,e}getStyle(e=Jn){rt.workingToColorSpace(tn.copy(this),e);const t=tn.r,n=tn.g,s=tn.b;return e!==Jn?`color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(n*255)},${Math.round(s*255)})`}offsetHSL(e,t,n){return this.getHSL(gs),this.setHSL(gs.h+e,gs.s+t,gs.l+n)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,n){return this.r=e.r+(t.r-e.r)*n,this.g=e.g+(t.g-e.g)*n,this.b=e.b+(t.b-e.b)*n,this}lerpHSL(e,t){this.getHSL(gs),e.getHSL(za);const n=Sc(gs.h,za.h,t),s=Sc(gs.s,za.s,t),r=Sc(gs.l,za.l,t);return this.setHSL(n,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const t=this.r,n=this.g,s=this.b,r=e.elements;return this.r=r[0]*t+r[3]*n+r[6]*s,this.g=r[1]*t+r[4]*n+r[7]*s,this.b=r[2]*t+r[5]*n+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const tn=new nt;nt.NAMES=k0;let YS=0;class ba extends mr{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:YS++}),this.uuid=ya(),this.name="",this.type="Material",this.blending=Is,this.side=Bi,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=sa,this.blendDst=ra,this.blendEquation=nr,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new nt(0,0,0),this.blendAlpha=0,this.depthFunc=ao,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Ih,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=yr,this.stencilZFail=yr,this.stencilZPass=yr,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const t in e){const n=e[t];if(n===void 0){je(`Material: parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){je(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(n):s&&s.isVector3&&n&&n.isVector3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});const n={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(e).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(e).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(e).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(e).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(e).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==Is&&(n.blending=this.blending),this.side!==Bi&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==sa&&(n.blendSrc=this.blendSrc),this.blendDst!==ra&&(n.blendDst=this.blendDst),this.blendEquation!==nr&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==ao&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==Ih&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==yr&&(n.stencilFail=this.stencilFail),this.stencilZFail!==yr&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==yr&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function s(r){const o=[];for(const a in r){const l=r[a];delete l.metadata,o.push(l)}return o}if(t){const r=s(e.textures),o=s(e.images);r.length>0&&(n.textures=r),o.length>0&&(n.images=o)}return n}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const t=e.clippingPlanes;let n=null;if(t!==null){const s=t.length;n=new Array(s);for(let r=0;r!==s;++r)n[r]=t[r].clone()}return this.clippingPlanes=n,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}class hr extends ba{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new nt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new _i,this.combine=T0,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const ts=KS();function KS(){const i=new ArrayBuffer(4),e=new Float32Array(i),t=new Uint32Array(i),n=new Uint32Array(512),s=new Uint32Array(512);for(let l=0;l<256;++l){const c=l-127;c<-27?(n[l]=0,n[l|256]=32768,s[l]=24,s[l|256]=24):c<-14?(n[l]=1024>>-c-14,n[l|256]=1024>>-c-14|32768,s[l]=-c-1,s[l|256]=-c-1):c<=15?(n[l]=c+15<<10,n[l|256]=c+15<<10|32768,s[l]=13,s[l|256]=13):c<128?(n[l]=31744,n[l|256]=64512,s[l]=24,s[l|256]=24):(n[l]=31744,n[l|256]=64512,s[l]=13,s[l|256]=13)}const r=new Uint32Array(2048),o=new Uint32Array(64),a=new Uint32Array(64);for(let l=1;l<1024;++l){let c=l<<13,u=0;for(;(c&8388608)===0;)c<<=1,u-=8388608;c&=-8388609,u+=947912704,r[l]=c|u}for(let l=1024;l<2048;++l)r[l]=939524096+(l-1024<<13);for(let l=1;l<31;++l)o[l]=l<<23;o[31]=1199570944,o[32]=2147483648;for(let l=33;l<63;++l)o[l]=2147483648+(l-32<<23);o[63]=3347054592;for(let l=1;l<64;++l)l!==32&&(a[l]=1024);return{floatView:e,uint32View:t,baseTable:n,shiftTable:s,mantissaTable:r,exponentTable:o,offsetTable:a}}function jS(i){Math.abs(i)>65504&&je("DataUtils.toHalfFloat(): Value out of range."),i=Je(i,-65504,65504),ts.floatView[0]=i;const e=ts.uint32View[0],t=e>>23&511;return ts.baseTable[t]+((e&8388607)>>ts.shiftTable[t])}function $S(i){const e=i>>10;return ts.uint32View[0]=ts.mantissaTable[ts.offsetTable[e]+(i&1023)]+ts.exponentTable[e],ts.floatView[0]}class ua{static toHalfFloat(e){return jS(e)}static fromHalfFloat(e){return $S(e)}}const kt=new B,ka=new ze;let ZS=0;class li{constructor(e,t,n=!1){if(Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:ZS++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=n,this.usage=Dh,this.updateRanges=[],this.gpuType=mi,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,n){e*=this.itemSize,n*=t.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=t.array[n+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,n=this.count;t<n;t++)ka.fromBufferAttribute(this,t),ka.applyMatrix3(e),this.setXY(t,ka.x,ka.y);else if(this.itemSize===3)for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.applyMatrix3(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}applyMatrix4(e){for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.applyMatrix4(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.applyNormalMatrix(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)kt.fromBufferAttribute(this,t),kt.transformDirection(e),this.setXYZ(t,kt.x,kt.y,kt.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let n=this.array[e*this.itemSize+t];return this.normalized&&(n=Eo(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=bn(n,this.array)),this.array[e*this.itemSize+t]=n,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=Eo(t,this.array)),t}setX(e,t){return this.normalized&&(t=bn(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=Eo(t,this.array)),t}setY(e,t){return this.normalized&&(t=bn(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=Eo(t,this.array)),t}setZ(e,t){return this.normalized&&(t=bn(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=Eo(t,this.array)),t}setW(e,t){return this.normalized&&(t=bn(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,n){return e*=this.itemSize,this.normalized&&(t=bn(t,this.array),n=bn(n,this.array)),this.array[e+0]=t,this.array[e+1]=n,this}setXYZ(e,t,n,s){return e*=this.itemSize,this.normalized&&(t=bn(t,this.array),n=bn(n,this.array),s=bn(s,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this}setXYZW(e,t,n,s,r){return e*=this.itemSize,this.normalized&&(t=bn(t,this.array),n=bn(n,this.array),s=bn(s,this.array),r=bn(r,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==Dh&&(e.usage=this.usage),e}}class H0 extends li{constructor(e,t,n){super(new Uint16Array(e),t,n)}}class V0 extends li{constructor(e,t,n){super(new Uint32Array(e),t,n)}}class on extends li{constructor(e,t,n){super(new Float32Array(e),t,n)}}let JS=0;const $n=new qe,zc=new Wt,Pr=new B,On=new wi,Do=new wi,Kt=new B;class An extends mr{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:JS++}),this.uuid=ya(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(U0(e)?V0:H0)(e,1):this.index=e,this}setIndirect(e){return this.indirect=e,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,n=0){this.groups.push({start:e,count:t,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){const t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const r=new Qe().getNormalMatrix(e);n.applyNormalMatrix(r),n.needsUpdate=!0}const s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return $n.makeRotationFromQuaternion(e),this.applyMatrix4($n),this}rotateX(e){return $n.makeRotationX(e),this.applyMatrix4($n),this}rotateY(e){return $n.makeRotationY(e),this.applyMatrix4($n),this}rotateZ(e){return $n.makeRotationZ(e),this.applyMatrix4($n),this}translate(e,t,n){return $n.makeTranslation(e,t,n),this.applyMatrix4($n),this}scale(e,t,n){return $n.makeScale(e,t,n),this.applyMatrix4($n),this}lookAt(e){return zc.lookAt(e),zc.updateMatrix(),this.applyMatrix4(zc.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Pr).negate(),this.translate(Pr.x,Pr.y,Pr.z),this}setFromPoints(e){const t=this.getAttribute("position");if(t===void 0){const n=[];for(let s=0,r=e.length;s<r;s++){const o=e[s];n.push(o.x,o.y,o.z||0)}this.setAttribute("position",new on(n,3))}else{const n=Math.min(e.length,t.count);for(let s=0;s<n;s++){const r=e[s];t.setXYZ(s,r.x,r.y,r.z||0)}e.length>t.count&&je("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new wi);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){zt("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new B(-1/0,-1/0,-1/0),new B(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let n=0,s=t.length;n<s;n++){const r=t[n];On.setFromBufferAttribute(r),this.morphTargetsRelative?(Kt.addVectors(this.boundingBox.min,On.min),this.boundingBox.expandByPoint(Kt),Kt.addVectors(this.boundingBox.max,On.max),this.boundingBox.expandByPoint(Kt)):(this.boundingBox.expandByPoint(On.min),this.boundingBox.expandByPoint(On.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&zt('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Zl);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){zt("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new B,1/0);return}if(e){const n=this.boundingSphere.center;if(On.setFromBufferAttribute(e),t)for(let r=0,o=t.length;r<o;r++){const a=t[r];Do.setFromBufferAttribute(a),this.morphTargetsRelative?(Kt.addVectors(On.min,Do.min),On.expandByPoint(Kt),Kt.addVectors(On.max,Do.max),On.expandByPoint(Kt)):(On.expandByPoint(Do.min),On.expandByPoint(Do.max))}On.getCenter(n);let s=0;for(let r=0,o=e.count;r<o;r++)Kt.fromBufferAttribute(e,r),s=Math.max(s,n.distanceToSquared(Kt));if(t)for(let r=0,o=t.length;r<o;r++){const a=t[r],l=this.morphTargetsRelative;for(let c=0,u=a.count;c<u;c++)Kt.fromBufferAttribute(a,c),l&&(Pr.fromBufferAttribute(e,c),Kt.add(Pr)),s=Math.max(s,n.distanceToSquared(Kt))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&zt('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){zt("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=t.position,s=t.normal,r=t.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new li(new Float32Array(4*n.count),4));const o=this.getAttribute("tangent"),a=[],l=[];for(let E=0;E<n.count;E++)a[E]=new B,l[E]=new B;const c=new B,u=new B,f=new B,d=new ze,h=new ze,x=new ze,p=new B,g=new B;function m(E,b,C){c.fromBufferAttribute(n,E),u.fromBufferAttribute(n,b),f.fromBufferAttribute(n,C),d.fromBufferAttribute(r,E),h.fromBufferAttribute(r,b),x.fromBufferAttribute(r,C),u.sub(c),f.sub(c),h.sub(d),x.sub(d);const I=1/(h.x*x.y-x.x*h.y);isFinite(I)&&(p.copy(u).multiplyScalar(x.y).addScaledVector(f,-h.y).multiplyScalar(I),g.copy(f).multiplyScalar(h.x).addScaledVector(u,-x.x).multiplyScalar(I),a[E].add(p),a[b].add(p),a[C].add(p),l[E].add(g),l[b].add(g),l[C].add(g))}let _=this.groups;_.length===0&&(_=[{start:0,count:e.count}]);for(let E=0,b=_.length;E<b;++E){const C=_[E],I=C.start,F=C.count;for(let U=I,O=I+F;U<O;U+=3)m(e.getX(U+0),e.getX(U+1),e.getX(U+2))}const A=new B,S=new B,v=new B,y=new B;function M(E){v.fromBufferAttribute(s,E),y.copy(v);const b=a[E];A.copy(b),A.sub(v.multiplyScalar(v.dot(b))).normalize(),S.crossVectors(y,b);const I=S.dot(l[E])<0?-1:1;o.setXYZW(E,A.x,A.y,A.z,I)}for(let E=0,b=_.length;E<b;++E){const C=_[E],I=C.start,F=C.count;for(let U=I,O=I+F;U<O;U+=3)M(e.getX(U+0)),M(e.getX(U+1)),M(e.getX(U+2))}}computeVertexNormals(){const e=this.index,t=this.getAttribute("position");if(t!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new li(new Float32Array(t.count*3),3),this.setAttribute("normal",n);else for(let d=0,h=n.count;d<h;d++)n.setXYZ(d,0,0,0);const s=new B,r=new B,o=new B,a=new B,l=new B,c=new B,u=new B,f=new B;if(e)for(let d=0,h=e.count;d<h;d+=3){const x=e.getX(d+0),p=e.getX(d+1),g=e.getX(d+2);s.fromBufferAttribute(t,x),r.fromBufferAttribute(t,p),o.fromBufferAttribute(t,g),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),a.fromBufferAttribute(n,x),l.fromBufferAttribute(n,p),c.fromBufferAttribute(n,g),a.add(u),l.add(u),c.add(u),n.setXYZ(x,a.x,a.y,a.z),n.setXYZ(p,l.x,l.y,l.z),n.setXYZ(g,c.x,c.y,c.z)}else for(let d=0,h=t.count;d<h;d+=3)s.fromBufferAttribute(t,d+0),r.fromBufferAttribute(t,d+1),o.fromBufferAttribute(t,d+2),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),n.setXYZ(d+0,u.x,u.y,u.z),n.setXYZ(d+1,u.x,u.y,u.z),n.setXYZ(d+2,u.x,u.y,u.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let t=0,n=e.count;t<n;t++)Kt.fromBufferAttribute(e,t),Kt.normalize(),e.setXYZ(t,Kt.x,Kt.y,Kt.z)}toNonIndexed(){function e(a,l){const c=a.array,u=a.itemSize,f=a.normalized,d=new c.constructor(l.length*u);let h=0,x=0;for(let p=0,g=l.length;p<g;p++){a.isInterleavedBufferAttribute?h=l[p]*a.data.stride+a.offset:h=l[p]*u;for(let m=0;m<u;m++)d[x++]=c[h++]}return new li(d,u,f)}if(this.index===null)return je("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const t=new An,n=this.index.array,s=this.attributes;for(const a in s){const l=s[a],c=e(l,n);t.setAttribute(a,c)}const r=this.morphAttributes;for(const a in r){const l=[],c=r[a];for(let u=0,f=c.length;u<f;u++){const d=c[u],h=e(d,n);l.push(h)}t.morphAttributes[a]=l}t.morphTargetsRelative=this.morphTargetsRelative;const o=this.groups;for(let a=0,l=o.length;a<l;a++){const c=o[a];t.addGroup(c.start,c.count,c.materialIndex)}return t}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const l=this.parameters;for(const c in l)l[c]!==void 0&&(e[c]=l[c]);return e}e.data={attributes:{}};const t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});const n=this.attributes;for(const l in n){const c=n[l];e.data.attributes[l]=c.toJSON(e.data)}const s={};let r=!1;for(const l in this.morphAttributes){const c=this.morphAttributes[l],u=[];for(let f=0,d=c.length;f<d;f++){const h=c[f];u.push(h.toJSON(e.data))}u.length>0&&(s[l]=u,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);const o=this.groups;o.length>0&&(e.data.groups=JSON.parse(JSON.stringify(o)));const a=this.boundingSphere;return a!==null&&(e.data.boundingSphere=a.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const t={};this.name=e.name;const n=e.index;n!==null&&this.setIndex(n.clone());const s=e.attributes;for(const c in s){const u=s[c];this.setAttribute(c,u.clone(t))}const r=e.morphAttributes;for(const c in r){const u=[],f=r[c];for(let d=0,h=f.length;d<h;d++)u.push(f[d].clone(t));this.morphAttributes[c]=u}this.morphTargetsRelative=e.morphTargetsRelative;const o=e.groups;for(let c=0,u=o.length;c<u;c++){const f=o[c];this.addGroup(f.start,f.count,f.materialIndex)}const a=e.boundingBox;a!==null&&(this.boundingBox=a.clone());const l=e.boundingSphere;return l!==null&&(this.boundingSphere=l.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const qh=new qe,Ys=new td,Ha=new Zl,Qh=new B,Va=new B,Ga=new B,Wa=new B,kc=new B,Xa=new B,Yh=new B,qa=new B;class Vt extends Wt{constructor(e=new An,t=new hr){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}getVertexPosition(e,t){const n=this.geometry,s=n.attributes.position,r=n.morphAttributes.position,o=n.morphTargetsRelative;t.fromBufferAttribute(s,e);const a=this.morphTargetInfluences;if(r&&a){Xa.set(0,0,0);for(let l=0,c=r.length;l<c;l++){const u=a[l],f=r[l];u!==0&&(kc.fromBufferAttribute(f,e),o?Xa.addScaledVector(kc,u):Xa.addScaledVector(kc.sub(t),u))}t.add(Xa)}return t}raycast(e,t){const n=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),Ha.copy(n.boundingSphere),Ha.applyMatrix4(r),Ys.copy(e.ray).recast(e.near),!(Ha.containsPoint(Ys.origin)===!1&&(Ys.intersectSphere(Ha,Qh)===null||Ys.origin.distanceToSquared(Qh)>(e.far-e.near)**2))&&(qh.copy(r).invert(),Ys.copy(e.ray).applyMatrix4(qh),!(n.boundingBox!==null&&Ys.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(e,t,Ys)))}_computeIntersections(e,t,n){let s;const r=this.geometry,o=this.material,a=r.index,l=r.attributes.position,c=r.attributes.uv,u=r.attributes.uv1,f=r.attributes.normal,d=r.groups,h=r.drawRange;if(a!==null)if(Array.isArray(o))for(let x=0,p=d.length;x<p;x++){const g=d[x],m=o[g.materialIndex],_=Math.max(g.start,h.start),A=Math.min(a.count,Math.min(g.start+g.count,h.start+h.count));for(let S=_,v=A;S<v;S+=3){const y=a.getX(S),M=a.getX(S+1),E=a.getX(S+2);s=Qa(this,m,e,n,c,u,f,y,M,E),s&&(s.faceIndex=Math.floor(S/3),s.face.materialIndex=g.materialIndex,t.push(s))}}else{const x=Math.max(0,h.start),p=Math.min(a.count,h.start+h.count);for(let g=x,m=p;g<m;g+=3){const _=a.getX(g),A=a.getX(g+1),S=a.getX(g+2);s=Qa(this,o,e,n,c,u,f,_,A,S),s&&(s.faceIndex=Math.floor(g/3),t.push(s))}}else if(l!==void 0)if(Array.isArray(o))for(let x=0,p=d.length;x<p;x++){const g=d[x],m=o[g.materialIndex],_=Math.max(g.start,h.start),A=Math.min(l.count,Math.min(g.start+g.count,h.start+h.count));for(let S=_,v=A;S<v;S+=3){const y=S,M=S+1,E=S+2;s=Qa(this,m,e,n,c,u,f,y,M,E),s&&(s.faceIndex=Math.floor(S/3),s.face.materialIndex=g.materialIndex,t.push(s))}}else{const x=Math.max(0,h.start),p=Math.min(l.count,h.start+h.count);for(let g=x,m=p;g<m;g+=3){const _=g,A=g+1,S=g+2;s=Qa(this,o,e,n,c,u,f,_,A,S),s&&(s.faceIndex=Math.floor(g/3),t.push(s))}}}}function ev(i,e,t,n,s,r,o,a){let l;if(e.side===wn?l=n.intersectTriangle(o,r,s,!0,a):l=n.intersectTriangle(s,r,o,e.side===Bi,a),l===null)return null;qa.copy(a),qa.applyMatrix4(i.matrixWorld);const c=t.ray.origin.distanceTo(qa);return c<t.near||c>t.far?null:{distance:c,point:qa.clone(),object:i}}function Qa(i,e,t,n,s,r,o,a,l,c){i.getVertexPosition(a,Va),i.getVertexPosition(l,Ga),i.getVertexPosition(c,Wa);const u=ev(i,e,t,n,Va,Ga,Wa,Yh);if(u){const f=new B;pi.getBarycoord(Yh,Va,Ga,Wa,f),s&&(u.uv=pi.getInterpolatedAttribute(s,a,l,c,f,new ze)),r&&(u.uv1=pi.getInterpolatedAttribute(r,a,l,c,f,new ze)),o&&(u.normal=pi.getInterpolatedAttribute(o,a,l,c,f,new B),u.normal.dot(n.direction)>0&&u.normal.multiplyScalar(-1));const d={a,b:l,c,normal:new B,materialIndex:0};pi.getNormal(Va,Ga,Wa,d.normal),u.face=d,u.barycoord=f}return u}class yo extends An{constructor(e=1,t=1,n=1,s=1,r=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:n,widthSegments:s,heightSegments:r,depthSegments:o};const a=this;s=Math.floor(s),r=Math.floor(r),o=Math.floor(o);const l=[],c=[],u=[],f=[];let d=0,h=0;x("z","y","x",-1,-1,n,t,e,o,r,0),x("z","y","x",1,-1,n,t,-e,o,r,1),x("x","z","y",1,1,e,n,t,s,o,2),x("x","z","y",1,-1,e,n,-t,s,o,3),x("x","y","z",1,-1,e,t,n,s,r,4),x("x","y","z",-1,-1,e,t,-n,s,r,5),this.setIndex(l),this.setAttribute("position",new on(c,3)),this.setAttribute("normal",new on(u,3)),this.setAttribute("uv",new on(f,2));function x(p,g,m,_,A,S,v,y,M,E,b){const C=S/M,I=v/E,F=S/2,U=v/2,O=y/2,k=M+1,z=E+1;let V=0,H=0;const $=new B;for(let oe=0;oe<z;oe++){const Se=oe*I-U;for(let we=0;we<k;we++){const Le=we*C-F;$[p]=Le*_,$[g]=Se*A,$[m]=O,c.push($.x,$.y,$.z),$[p]=0,$[g]=0,$[m]=y>0?1:-1,u.push($.x,$.y,$.z),f.push(we/M),f.push(1-oe/E),V+=1}}for(let oe=0;oe<E;oe++)for(let Se=0;Se<M;Se++){const we=d+Se+k*oe,Le=d+Se+k*(oe+1),fe=d+(Se+1)+k*(oe+1),re=d+(Se+1)+k*oe;l.push(we,Le,re),l.push(Le,fe,re),H+=6}a.addGroup(h,H,b),h+=H,d+=V}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new yo(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}function ho(i){const e={};for(const t in i){e[t]={};for(const n in i[t]){const s=i[t][n];s&&(s.isColor||s.isMatrix3||s.isMatrix4||s.isVector2||s.isVector3||s.isVector4||s.isTexture||s.isQuaternion)?s.isRenderTargetTexture?(je("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][n]=null):e[t][n]=s.clone():Array.isArray(s)?e[t][n]=s.slice():e[t][n]=s}}return e}function fn(i){const e={};for(let t=0;t<i.length;t++){const n=ho(i[t]);for(const s in n)e[s]=n[s]}return e}function tv(i){const e=[];for(let t=0;t<i.length;t++)e.push(i[t].clone());return e}function G0(i){const e=i.getRenderTarget();return e===null?i.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:rt.workingColorSpace}const nv={clone:ho,merge:fn};var iv=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,sv=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class _n extends ba{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=iv,this.fragmentShader=sv,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=ho(e.uniforms),this.uniformsGroups=tv(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this}toJSON(e){const t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(const s in this.uniforms){const o=this.uniforms[s].value;o&&o.isTexture?t.uniforms[s]={type:"t",value:o.toJSON(e).uuid}:o&&o.isColor?t.uniforms[s]={type:"c",value:o.getHex()}:o&&o.isVector2?t.uniforms[s]={type:"v2",value:o.toArray()}:o&&o.isVector3?t.uniforms[s]={type:"v3",value:o.toArray()}:o&&o.isVector4?t.uniforms[s]={type:"v4",value:o.toArray()}:o&&o.isMatrix3?t.uniforms[s]={type:"m3",value:o.toArray()}:o&&o.isMatrix4?t.uniforms[s]={type:"m4",value:o.toArray()}:t.uniforms[s]={value:o}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;const n={};for(const s in this.extensions)this.extensions[s]===!0&&(n[s]=!0);return Object.keys(n).length>0&&(t.extensions=n),t}}class W0 extends Wt{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new qe,this.projectionMatrix=new qe,this.projectionMatrixInverse=new qe,this.coordinateSystem=Ei,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const xs=new B,Kh=new ze,jh=new ze;class ei extends W0{constructor(e=50,t=1,n=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=n,this.far=s,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const t=.5*this.getFilmHeight()/e;this.fov=cf*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(gl*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return cf*2*Math.atan(Math.tan(gl*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,n){xs.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(xs.x,xs.y).multiplyScalar(-e/xs.z),xs.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(xs.x,xs.y).multiplyScalar(-e/xs.z)}getViewSize(e,t){return this.getViewBounds(e,Kh,jh),t.subVectors(jh,Kh)}setViewOffset(e,t,n,s,r,o){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let t=e*Math.tan(gl*.5*this.fov)/this.zoom,n=2*t,s=this.aspect*n,r=-.5*s;const o=this.view;if(this.view!==null&&this.view.enabled){const l=o.fullWidth,c=o.fullHeight;r+=o.offsetX*s/l,t-=o.offsetY*n/c,s*=o.width/l,n*=o.height/c}const a=this.filmOffset;a!==0&&(r+=e*a/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,t,t-n,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}}const Fr=-90,Lr=1;class rv extends Wt{constructor(e,t,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const s=new ei(Fr,Lr,e,t);s.layers=this.layers,this.add(s);const r=new ei(Fr,Lr,e,t);r.layers=this.layers,this.add(r);const o=new ei(Fr,Lr,e,t);o.layers=this.layers,this.add(o);const a=new ei(Fr,Lr,e,t);a.layers=this.layers,this.add(a);const l=new ei(Fr,Lr,e,t);l.layers=this.layers,this.add(l);const c=new ei(Fr,Lr,e,t);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){const e=this.coordinateSystem,t=this.children.concat(),[n,s,r,o,a,l]=t;for(const c of t)this.remove(c);if(e===Ei)n.up.set(0,1,0),n.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),o.up.set(0,0,1),o.lookAt(0,-1,0),a.up.set(0,1,0),a.lookAt(0,0,1),l.up.set(0,1,0),l.lookAt(0,0,-1);else if(e===wl)n.up.set(0,-1,0),n.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),o.up.set(0,0,-1),o.lookAt(0,-1,0),a.up.set(0,-1,0),a.lookAt(0,0,1),l.up.set(0,-1,0),l.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const c of t)this.add(c),c.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[r,o,a,l,c,u]=this.children,f=e.getRenderTarget(),d=e.getActiveCubeFace(),h=e.getActiveMipmapLevel(),x=e.xr.enabled;e.xr.enabled=!1;const p=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,e.setRenderTarget(n,0,s),e.render(t,r),e.setRenderTarget(n,1,s),e.render(t,o),e.setRenderTarget(n,2,s),e.render(t,a),e.setRenderTarget(n,3,s),e.render(t,l),e.setRenderTarget(n,4,s),e.render(t,c),n.texture.generateMipmaps=p,e.setRenderTarget(n,5,s),e.render(t,u),e.setRenderTarget(f,d,h),e.xr.enabled=x,n.texture.needsPMREMUpdate=!0}}class X0 extends xn{constructor(e=[],t=lo,n,s,r,o,a,l,c,u){super(e,t,n,s,r,o,a,l,c,u),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class ov extends Us{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;const n={width:e,height:e,depth:1},s=[n,n,n,n,n,n];this.texture=new X0(s),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

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
			`},s=new yo(5,5,5),r=new _n({name:"CubemapFromEquirect",uniforms:ho(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:wn,blending:is});r.uniforms.tEquirect.value=t;const o=new Vt(s,r),a=t.minFilter;return t.minFilter===sr&&(t.minFilter=ii),new rv(1,10,this).update(e,o),t.minFilter=a,o.geometry.dispose(),o.material.dispose(),this}clear(e,t=!0,n=!0,s=!0){const r=e.getRenderTarget();for(let o=0;o<6;o++)e.setRenderTarget(this,o),e.clear(t,n,s);e.setRenderTarget(r)}}class Ya extends Wt{constructor(){super(),this.isGroup=!0,this.type="Group"}}const av={type:"move"};class Hc{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Ya,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Ya,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new B,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new B),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Ya,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new B,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new B),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const t=this._hand;if(t)for(const n of e.hand.values())this._getHandJoint(t,n)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,n){let s=null,r=null,o=null;const a=this._targetRay,l=this._grip,c=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(c&&e.hand){o=!0;for(const p of e.hand.values()){const g=t.getJointPose(p,n),m=this._getHandJoint(c,p);g!==null&&(m.matrix.fromArray(g.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,m.jointRadius=g.radius),m.visible=g!==null}const u=c.joints["index-finger-tip"],f=c.joints["thumb-tip"],d=u.position.distanceTo(f.position),h=.02,x=.005;c.inputState.pinching&&d>h+x?(c.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!c.inputState.pinching&&d<=h-x&&(c.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else l!==null&&e.gripSpace&&(r=t.getPose(e.gripSpace,n),r!==null&&(l.matrix.fromArray(r.transform.matrix),l.matrix.decompose(l.position,l.rotation,l.scale),l.matrixWorldNeedsUpdate=!0,r.linearVelocity?(l.hasLinearVelocity=!0,l.linearVelocity.copy(r.linearVelocity)):l.hasLinearVelocity=!1,r.angularVelocity?(l.hasAngularVelocity=!0,l.angularVelocity.copy(r.angularVelocity)):l.hasAngularVelocity=!1));a!==null&&(s=t.getPose(e.targetRaySpace,n),s===null&&r!==null&&(s=r),s!==null&&(a.matrix.fromArray(s.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),a.matrixWorldNeedsUpdate=!0,s.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(s.linearVelocity)):a.hasLinearVelocity=!1,s.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(s.angularVelocity)):a.hasAngularVelocity=!1,this.dispatchEvent(av)))}return a!==null&&(a.visible=s!==null),l!==null&&(l.visible=r!==null),c!==null&&(c.visible=o!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){const n=new Ya;n.matrixAutoUpdate=!1,n.visible=!1,e.joints[t.jointName]=n,e.add(n)}return e.joints[t.jointName]}}class lv extends Wt{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new _i,this.environmentIntensity=1,this.environmentRotation=new _i,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}}class Qi extends xn{constructor(e=null,t=1,n=1,s,r,o,a,l,c=qn,u=qn,f,d){super(null,o,a,l,c,u,s,r,f,d),this.isDataTexture=!0,this.image={data:e,width:t,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class cv extends li{constructor(e,t,n,s=1){super(e,t,n),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=s}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}toJSON(){const e=super.toJSON();return e.meshPerAttribute=this.meshPerAttribute,e.isInstancedBufferAttribute=!0,e}}const Vc=new B,uv=new B,fv=new Qe;class vs{constructor(e=new B(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,n,s){return this.normal.set(e,t,n),this.constant=s,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,n){const s=Vc.subVectors(n,t).cross(uv.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t){const n=e.delta(Vc),s=this.normal.dot(n);if(s===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;const r=-(e.start.dot(this.normal)+this.constant)/s;return r<0||r>1?null:t.copy(e.start).addScaledVector(n,r)}intersectsLine(e){const t=this.distanceToPoint(e.start),n=this.distanceToPoint(e.end);return t<0&&n>0||n<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){const n=t||fv.getNormalMatrix(e),s=this.coplanarPoint(Vc).applyMatrix4(e),r=this.normal.applyMatrix3(n).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const Ks=new Zl,dv=new ze(.5,.5),Ka=new B;class q0{constructor(e=new vs,t=new vs,n=new vs,s=new vs,r=new vs,o=new vs){this.planes=[e,t,n,s,r,o]}set(e,t,n,s,r,o){const a=this.planes;return a[0].copy(e),a[1].copy(t),a[2].copy(n),a[3].copy(s),a[4].copy(r),a[5].copy(o),this}copy(e){const t=this.planes;for(let n=0;n<6;n++)t[n].copy(e.planes[n]);return this}setFromProjectionMatrix(e,t=Ei,n=!1){const s=this.planes,r=e.elements,o=r[0],a=r[1],l=r[2],c=r[3],u=r[4],f=r[5],d=r[6],h=r[7],x=r[8],p=r[9],g=r[10],m=r[11],_=r[12],A=r[13],S=r[14],v=r[15];if(s[0].setComponents(c-o,h-u,m-x,v-_).normalize(),s[1].setComponents(c+o,h+u,m+x,v+_).normalize(),s[2].setComponents(c+a,h+f,m+p,v+A).normalize(),s[3].setComponents(c-a,h-f,m-p,v-A).normalize(),n)s[4].setComponents(l,d,g,S).normalize(),s[5].setComponents(c-l,h-d,m-g,v-S).normalize();else if(s[4].setComponents(c-l,h-d,m-g,v-S).normalize(),t===Ei)s[5].setComponents(c+l,h+d,m+g,v+S).normalize();else if(t===wl)s[5].setComponents(l,d,g,S).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),Ks.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),Ks.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(Ks)}intersectsSprite(e){Ks.center.set(0,0,0);const t=dv.distanceTo(e.center);return Ks.radius=.7071067811865476+t,Ks.applyMatrix4(e.matrixWorld),this.intersectsSphere(Ks)}intersectsSphere(e){const t=this.planes,n=e.center,s=-e.radius;for(let r=0;r<6;r++)if(t[r].distanceToPoint(n)<s)return!1;return!0}intersectsBox(e){const t=this.planes;for(let n=0;n<6;n++){const s=t[n];if(Ka.x=s.normal.x>0?e.max.x:e.min.x,Ka.y=s.normal.y>0?e.max.y:e.min.y,Ka.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(Ka)<0)return!1}return!0}containsPoint(e){const t=this.planes;for(let n=0;n<6;n++)if(t[n].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class hv extends ba{constructor(e){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new nt(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.size=e.size,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}}const $h=new qe,uf=new td,ja=new Zl,$a=new B;class pv extends Wt{constructor(e=new An,t=new hv){super(),this.isPoints=!0,this.type="Points",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}raycast(e,t){const n=this.geometry,s=this.matrixWorld,r=e.params.Points.threshold,o=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),ja.copy(n.boundingSphere),ja.applyMatrix4(s),ja.radius+=r,e.ray.intersectsSphere(ja)===!1)return;$h.copy(s).invert(),uf.copy(e.ray).applyMatrix4($h);const a=r/((this.scale.x+this.scale.y+this.scale.z)/3),l=a*a,c=n.index,f=n.attributes.position;if(c!==null){const d=Math.max(0,o.start),h=Math.min(c.count,o.start+o.count);for(let x=d,p=h;x<p;x++){const g=c.getX(x);$a.fromBufferAttribute(f,g),Zh($a,g,l,s,e,t,this)}}else{const d=Math.max(0,o.start),h=Math.min(f.count,o.start+o.count);for(let x=d,p=h;x<p;x++)$a.fromBufferAttribute(f,x),Zh($a,x,l,s,e,t,this)}}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}}function Zh(i,e,t,n,s,r,o){const a=uf.distanceSqToPoint(i);if(a<t){const l=new B;uf.closestPointToPoint(i,l),l.applyMatrix4(n);const c=s.ray.origin.distanceTo(l);if(c<s.near||c>s.far)return;r.push({distance:c,distanceToRay:Math.sqrt(a),point:l,index:e,face:null,faceIndex:null,barycoord:null,object:o})}}class nd extends xn{constructor(e,t,n=si,s,r,o,a=qn,l=qn,c,u=uo,f=1){if(u!==uo&&u!==la)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const d={width:e,height:t,depth:f};super(d,s,r,o,a,l,u,n,c),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new ed(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}}class Q0 extends xn{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class fa extends An{constructor(e=1,t=1,n=1,s=32,r=1,o=!1,a=0,l=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:n,radialSegments:s,heightSegments:r,openEnded:o,thetaStart:a,thetaLength:l};const c=this;s=Math.floor(s),r=Math.floor(r);const u=[],f=[],d=[],h=[];let x=0;const p=[],g=n/2;let m=0;_(),o===!1&&(e>0&&A(!0),t>0&&A(!1)),this.setIndex(u),this.setAttribute("position",new on(f,3)),this.setAttribute("normal",new on(d,3)),this.setAttribute("uv",new on(h,2));function _(){const S=new B,v=new B;let y=0;const M=(t-e)/n;for(let E=0;E<=r;E++){const b=[],C=E/r,I=C*(t-e)+e;for(let F=0;F<=s;F++){const U=F/s,O=U*l+a,k=Math.sin(O),z=Math.cos(O);v.x=I*k,v.y=-C*n+g,v.z=I*z,f.push(v.x,v.y,v.z),S.set(k,M,z).normalize(),d.push(S.x,S.y,S.z),h.push(U,1-C),b.push(x++)}p.push(b)}for(let E=0;E<s;E++)for(let b=0;b<r;b++){const C=p[b][E],I=p[b+1][E],F=p[b+1][E+1],U=p[b][E+1];(e>0||b!==0)&&(u.push(C,I,U),y+=3),(t>0||b!==r-1)&&(u.push(I,F,U),y+=3)}c.addGroup(m,y,0),m+=y}function A(S){const v=x,y=new ze,M=new B;let E=0;const b=S===!0?e:t,C=S===!0?1:-1;for(let F=1;F<=s;F++)f.push(0,g*C,0),d.push(0,C,0),h.push(.5,.5),x++;const I=x;for(let F=0;F<=s;F++){const O=F/s*l+a,k=Math.cos(O),z=Math.sin(O);M.x=b*z,M.y=g*C,M.z=b*k,f.push(M.x,M.y,M.z),d.push(0,C,0),y.x=k*.5+.5,y.y=z*.5*C+.5,h.push(y.x,y.y),x++}for(let F=0;F<s;F++){const U=v+F,O=I+F;S===!0?u.push(O,O+1,U):u.push(O+1,O,U),E+=3}c.addGroup(m,E,S===!0?1:2),m+=E}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new fa(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class id extends fa{constructor(e=1,t=1,n=32,s=1,r=!1,o=0,a=Math.PI*2){super(0,e,t,n,s,r,o,a),this.type="ConeGeometry",this.parameters={radius:e,height:t,radialSegments:n,heightSegments:s,openEnded:r,thetaStart:o,thetaLength:a}}static fromJSON(e){return new id(e.radius,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class po extends An{constructor(e=1,t=1,n=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:n,heightSegments:s};const r=e/2,o=t/2,a=Math.floor(n),l=Math.floor(s),c=a+1,u=l+1,f=e/a,d=t/l,h=[],x=[],p=[],g=[];for(let m=0;m<u;m++){const _=m*d-o;for(let A=0;A<c;A++){const S=A*f-r;x.push(S,-_,0),p.push(0,0,1),g.push(A/a),g.push(1-m/l)}}for(let m=0;m<l;m++)for(let _=0;_<a;_++){const A=_+c*m,S=_+c*(m+1),v=_+1+c*(m+1),y=_+1+c*m;h.push(A,S,y),h.push(S,v,y)}this.setIndex(h),this.setAttribute("position",new on(x,3)),this.setAttribute("normal",new on(p,3)),this.setAttribute("uv",new on(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new po(e.width,e.height,e.widthSegments,e.heightSegments)}}class Il extends An{constructor(e=1,t=32,n=16,s=0,r=Math.PI*2,o=0,a=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:e,widthSegments:t,heightSegments:n,phiStart:s,phiLength:r,thetaStart:o,thetaLength:a},t=Math.max(3,Math.floor(t)),n=Math.max(2,Math.floor(n));const l=Math.min(o+a,Math.PI);let c=0;const u=[],f=new B,d=new B,h=[],x=[],p=[],g=[];for(let m=0;m<=n;m++){const _=[],A=m/n;let S=0;m===0&&o===0?S=.5/t:m===n&&l===Math.PI&&(S=-.5/t);for(let v=0;v<=t;v++){const y=v/t;f.x=-e*Math.cos(s+y*r)*Math.sin(o+A*a),f.y=e*Math.cos(o+A*a),f.z=e*Math.sin(s+y*r)*Math.sin(o+A*a),x.push(f.x,f.y,f.z),d.copy(f).normalize(),p.push(d.x,d.y,d.z),g.push(y+S,1-A),_.push(c++)}u.push(_)}for(let m=0;m<n;m++)for(let _=0;_<t;_++){const A=u[m][_+1],S=u[m][_],v=u[m+1][_],y=u[m+1][_+1];(m!==0||o>0)&&h.push(A,S,y),(m!==n-1||l<Math.PI)&&h.push(S,v,y)}this.setIndex(h),this.setAttribute("position",new on(x,3)),this.setAttribute("normal",new on(p,3)),this.setAttribute("uv",new on(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Il(e.radius,e.widthSegments,e.heightSegments,e.phiStart,e.phiLength,e.thetaStart,e.thetaLength)}}class mv extends ba{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=SS,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class gv extends ba{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}class sd extends W0{constructor(e=-1,t=1,n=1,s=-1,r=.1,o=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=n,this.bottom=s,this.near=r,this.far=o,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,n,s,r,o){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,s=(this.top+this.bottom)/2;let r=n-e,o=n+e,a=s+t,l=s-t;if(this.view!==null&&this.view.enabled){const c=(this.right-this.left)/this.view.fullWidth/this.zoom,u=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=c*this.view.offsetX,o=r+c*this.view.width,a-=u*this.view.offsetY,l=a-u*this.view.height}this.projectionMatrix.makeOrthographic(r,o,a,l,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}}class xv extends An{constructor(){super(),this.isInstancedBufferGeometry=!0,this.type="InstancedBufferGeometry",this.instanceCount=1/0}copy(e){return super.copy(e),this.instanceCount=e.instanceCount,this}toJSON(){const e=super.toJSON();return e.instanceCount=this.instanceCount,e.isInstancedBufferGeometry=!0,e}}class _v extends ei{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}class Jh{constructor(e=1,t=0,n=0){this.radius=e,this.phi=t,this.theta=n}set(e,t,n){return this.radius=e,this.phi=t,this.theta=n,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=Je(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,n){return this.radius=Math.sqrt(e*e+t*t+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,n),this.phi=Math.acos(Je(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}function ep(i,e,t,n){const s=Av(n);switch(t){case P0:return i*e;case L0:return i*e/s.components*s.byteLength;case $l:return i*e/s.components*s.byteLength;case Zf:return i*e*2/s.components*s.byteLength;case Jf:return i*e*2/s.components*s.byteLength;case F0:return i*e*3/s.components*s.byteLength;case gn:return i*e*4/s.components*s.byteLength;case $r:return i*e*4/s.components*s.byteLength;case dl:case hl:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case pl:case ml:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Bu:case Ou:return Math.max(i,16)*Math.max(e,8)/4;case Lu:case Uu:return Math.max(i,8)*Math.max(e,8)/2;case Nu:case zu:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case ku:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Hu:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Vu:return Math.floor((i+4)/5)*Math.floor((e+3)/4)*16;case Gu:return Math.floor((i+4)/5)*Math.floor((e+4)/5)*16;case Wu:return Math.floor((i+5)/6)*Math.floor((e+4)/5)*16;case Xu:return Math.floor((i+5)/6)*Math.floor((e+5)/6)*16;case qu:return Math.floor((i+7)/8)*Math.floor((e+4)/5)*16;case Qu:return Math.floor((i+7)/8)*Math.floor((e+5)/6)*16;case Yu:return Math.floor((i+7)/8)*Math.floor((e+7)/8)*16;case Ku:return Math.floor((i+9)/10)*Math.floor((e+4)/5)*16;case ju:return Math.floor((i+9)/10)*Math.floor((e+5)/6)*16;case $u:return Math.floor((i+9)/10)*Math.floor((e+7)/8)*16;case Zu:return Math.floor((i+9)/10)*Math.floor((e+9)/10)*16;case Ju:return Math.floor((i+11)/12)*Math.floor((e+9)/10)*16;case ef:return Math.floor((i+11)/12)*Math.floor((e+11)/12)*16;case tf:case nf:case sf:return Math.ceil(i/4)*Math.ceil(e/4)*16;case rf:case of:return Math.ceil(i/4)*Math.ceil(e/4)*8;case af:case lf:return Math.ceil(i/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function Av(i){switch(i){case Ui:case w0:return{byteLength:1,components:1};case oa:case R0:case pr:return{byteLength:2,components:1};case jf:case $f:return{byteLength:2,components:4};case si:case Kf:case mi:return{byteLength:4,components:1};case I0:case D0:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${i}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:Yf}}));typeof window<"u"&&(window.__THREE__?je("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=Yf);function Y0(){let i=null,e=!1,t=null,n=null;function s(r,o){t(r,o),n=i.requestAnimationFrame(s)}return{start:function(){e!==!0&&t!==null&&(n=i.requestAnimationFrame(s),e=!0)},stop:function(){i.cancelAnimationFrame(n),e=!1},setAnimationLoop:function(r){t=r},setContext:function(r){i=r}}}function Sv(i){const e=new WeakMap;function t(a,l){const c=a.array,u=a.usage,f=c.byteLength,d=i.createBuffer();i.bindBuffer(l,d),i.bufferData(l,c,u),a.onUploadCallback();let h;if(c instanceof Float32Array)h=i.FLOAT;else if(typeof Float16Array<"u"&&c instanceof Float16Array)h=i.HALF_FLOAT;else if(c instanceof Uint16Array)a.isFloat16BufferAttribute?h=i.HALF_FLOAT:h=i.UNSIGNED_SHORT;else if(c instanceof Int16Array)h=i.SHORT;else if(c instanceof Uint32Array)h=i.UNSIGNED_INT;else if(c instanceof Int32Array)h=i.INT;else if(c instanceof Int8Array)h=i.BYTE;else if(c instanceof Uint8Array)h=i.UNSIGNED_BYTE;else if(c instanceof Uint8ClampedArray)h=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+c);return{buffer:d,type:h,bytesPerElement:c.BYTES_PER_ELEMENT,version:a.version,size:f}}function n(a,l,c){const u=l.array,f=l.updateRanges;if(i.bindBuffer(c,a),f.length===0)i.bufferSubData(c,0,u);else{f.sort((h,x)=>h.start-x.start);let d=0;for(let h=1;h<f.length;h++){const x=f[d],p=f[h];p.start<=x.start+x.count+1?x.count=Math.max(x.count,p.start+p.count-x.start):(++d,f[d]=p)}f.length=d+1;for(let h=0,x=f.length;h<x;h++){const p=f[h];i.bufferSubData(c,p.start*u.BYTES_PER_ELEMENT,u,p.start,p.count)}l.clearUpdateRanges()}l.onUploadCallback()}function s(a){return a.isInterleavedBufferAttribute&&(a=a.data),e.get(a)}function r(a){a.isInterleavedBufferAttribute&&(a=a.data);const l=e.get(a);l&&(i.deleteBuffer(l.buffer),e.delete(a))}function o(a,l){if(a.isInterleavedBufferAttribute&&(a=a.data),a.isGLBufferAttribute){const u=e.get(a);(!u||u.version<a.version)&&e.set(a,{buffer:a.buffer,type:a.type,bytesPerElement:a.elementSize,version:a.version});return}const c=e.get(a);if(c===void 0)e.set(a,t(a,l));else if(c.version<a.version){if(c.size!==a.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(c.buffer,a,l),c.version=a.version}}return{get:s,remove:r,update:o}}var vv=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,yv=`#ifdef USE_ALPHAHASH
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
#endif`,bv=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,Mv=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Cv=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,Tv=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,Ev=`#ifdef USE_AOMAP
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
#endif`,wv=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,Rv=`#ifdef USE_BATCHING
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
#endif`,Iv=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,Dv=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Pv=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,Fv=`float G_BlinnPhong_Implicit( ) {
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
} // validated`,Lv=`#ifdef USE_IRIDESCENCE
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
#endif`,Bv=`#ifdef USE_BUMPMAP
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
#endif`,Uv=`#if NUM_CLIPPING_PLANES > 0
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
#endif`,Ov=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,Nv=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,zv=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,kv=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,Hv=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,Vv=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,Gv=`#if defined( USE_COLOR_ALPHA )
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
#endif`,Wv=`#define PI 3.141592653589793
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
} // validated`,Xv=`#ifdef ENVMAP_TYPE_CUBE_UV
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
#endif`,qv=`vec3 transformedNormal = objectNormal;
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
#endif`,Qv=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,Yv=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,Kv=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,jv=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,$v="gl_FragColor = linearToOutputTexel( gl_FragColor );",Zv=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,Jv=`#ifdef USE_ENVMAP
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
#endif`,ey=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,ty=`#ifdef USE_ENVMAP
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
#endif`,ny=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,iy=`#ifdef USE_ENVMAP
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
#endif`,sy=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,ry=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,oy=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,ay=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,ly=`#ifdef USE_GRADIENTMAP
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
}`,cy=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,uy=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,fy=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,dy=`uniform bool receiveShadow;
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
#endif`,hy=`#ifdef USE_ENVMAP
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
#endif`,py=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,my=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,gy=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,xy=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,_y=`PhysicalMaterial material;
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
#endif`,Ay=`uniform sampler2D dfgLUT;
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
}`,Sy=`
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
#endif`,vy=`#if defined( RE_IndirectDiffuse )
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
#endif`,yy=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,by=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,My=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Cy=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Ty=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,Ey=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,wy=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,Ry=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
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
#endif`,Iy=`#if defined( USE_POINTS_UV )
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
#endif`,Dy=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Py=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,Fy=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,Ly=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,By=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Uy=`#ifdef USE_MORPHTARGETS
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
#endif`,Oy=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Ny=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
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
vec3 nonPerturbedNormal = normal;`,zy=`#ifdef USE_NORMALMAP_OBJECTSPACE
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
#endif`,ky=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Hy=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Vy=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,Gy=`#ifdef USE_NORMALMAP
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
#endif`,Wy=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,Xy=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,qy=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,Qy=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,Yy=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,Ky=`vec3 packNormalToRGB( const in vec3 normal ) {
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
}`,jy=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,$y=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,Zy=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,Jy=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,eb=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,tb=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,nb=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,ib=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,sb=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
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
#endif`,rb=`float getShadowMask() {
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
}`,ob=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,ab=`#ifdef USE_SKINNING
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
#endif`,lb=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,cb=`#ifdef USE_SKINNING
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
#endif`,ub=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,fb=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,db=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,hb=`#ifndef saturate
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
vec3 CustomToneMapping( vec3 color ) { return color; }`,pb=`#ifdef USE_TRANSMISSION
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
#endif`,mb=`#ifdef USE_TRANSMISSION
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
#endif`,gb=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,xb=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,_b=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,Ab=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const Sb=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,vb=`uniform sampler2D t2D;
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
}`,yb=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,bb=`#ifdef ENVMAP_TYPE_CUBE
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
}`,Mb=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Cb=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Tb=`#include <common>
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
}`,Eb=`#if DEPTH_PACKING == 3200
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
}`,wb=`#define DISTANCE
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
}`,Rb=`#define DISTANCE
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
}`,Ib=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Db=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Pb=`uniform float scale;
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
}`,Fb=`uniform vec3 diffuse;
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
}`,Lb=`#include <common>
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
}`,Bb=`uniform vec3 diffuse;
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
}`,Ub=`#define LAMBERT
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
}`,Ob=`#define LAMBERT
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
}`,Nb=`#define MATCAP
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
}`,zb=`#define MATCAP
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
}`,kb=`#define NORMAL
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
}`,Hb=`#define NORMAL
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
}`,Vb=`#define PHONG
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
}`,Gb=`#define PHONG
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
}`,Wb=`#define STANDARD
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
}`,Xb=`#define STANDARD
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
}`,qb=`#define TOON
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
}`,Qb=`#define TOON
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
}`,Yb=`uniform float size;
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
}`,Kb=`uniform vec3 diffuse;
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
}`,jb=`#include <common>
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
}`,$b=`uniform vec3 color;
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
}`,Zb=`uniform float rotation;
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
}`,Jb=`uniform vec3 diffuse;
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
}`,Ze={alphahash_fragment:vv,alphahash_pars_fragment:yv,alphamap_fragment:bv,alphamap_pars_fragment:Mv,alphatest_fragment:Cv,alphatest_pars_fragment:Tv,aomap_fragment:Ev,aomap_pars_fragment:wv,batching_pars_vertex:Rv,batching_vertex:Iv,begin_vertex:Dv,beginnormal_vertex:Pv,bsdfs:Fv,iridescence_fragment:Lv,bumpmap_pars_fragment:Bv,clipping_planes_fragment:Uv,clipping_planes_pars_fragment:Ov,clipping_planes_pars_vertex:Nv,clipping_planes_vertex:zv,color_fragment:kv,color_pars_fragment:Hv,color_pars_vertex:Vv,color_vertex:Gv,common:Wv,cube_uv_reflection_fragment:Xv,defaultnormal_vertex:qv,displacementmap_pars_vertex:Qv,displacementmap_vertex:Yv,emissivemap_fragment:Kv,emissivemap_pars_fragment:jv,colorspace_fragment:$v,colorspace_pars_fragment:Zv,envmap_fragment:Jv,envmap_common_pars_fragment:ey,envmap_pars_fragment:ty,envmap_pars_vertex:ny,envmap_physical_pars_fragment:hy,envmap_vertex:iy,fog_vertex:sy,fog_pars_vertex:ry,fog_fragment:oy,fog_pars_fragment:ay,gradientmap_pars_fragment:ly,lightmap_pars_fragment:cy,lights_lambert_fragment:uy,lights_lambert_pars_fragment:fy,lights_pars_begin:dy,lights_toon_fragment:py,lights_toon_pars_fragment:my,lights_phong_fragment:gy,lights_phong_pars_fragment:xy,lights_physical_fragment:_y,lights_physical_pars_fragment:Ay,lights_fragment_begin:Sy,lights_fragment_maps:vy,lights_fragment_end:yy,logdepthbuf_fragment:by,logdepthbuf_pars_fragment:My,logdepthbuf_pars_vertex:Cy,logdepthbuf_vertex:Ty,map_fragment:Ey,map_pars_fragment:wy,map_particle_fragment:Ry,map_particle_pars_fragment:Iy,metalnessmap_fragment:Dy,metalnessmap_pars_fragment:Py,morphinstance_vertex:Fy,morphcolor_vertex:Ly,morphnormal_vertex:By,morphtarget_pars_vertex:Uy,morphtarget_vertex:Oy,normal_fragment_begin:Ny,normal_fragment_maps:zy,normal_pars_fragment:ky,normal_pars_vertex:Hy,normal_vertex:Vy,normalmap_pars_fragment:Gy,clearcoat_normal_fragment_begin:Wy,clearcoat_normal_fragment_maps:Xy,clearcoat_pars_fragment:qy,iridescence_pars_fragment:Qy,opaque_fragment:Yy,packing:Ky,premultiplied_alpha_fragment:jy,project_vertex:$y,dithering_fragment:Zy,dithering_pars_fragment:Jy,roughnessmap_fragment:eb,roughnessmap_pars_fragment:tb,shadowmap_pars_fragment:nb,shadowmap_pars_vertex:ib,shadowmap_vertex:sb,shadowmask_pars_fragment:rb,skinbase_vertex:ob,skinning_pars_vertex:ab,skinning_vertex:lb,skinnormal_vertex:cb,specularmap_fragment:ub,specularmap_pars_fragment:fb,tonemapping_fragment:db,tonemapping_pars_fragment:hb,transmission_fragment:pb,transmission_pars_fragment:mb,uv_pars_fragment:gb,uv_pars_vertex:xb,uv_vertex:_b,worldpos_vertex:Ab,background_vert:Sb,background_frag:vb,backgroundCube_vert:yb,backgroundCube_frag:bb,cube_vert:Mb,cube_frag:Cb,depth_vert:Tb,depth_frag:Eb,distanceRGBA_vert:wb,distanceRGBA_frag:Rb,equirect_vert:Ib,equirect_frag:Db,linedashed_vert:Pb,linedashed_frag:Fb,meshbasic_vert:Lb,meshbasic_frag:Bb,meshlambert_vert:Ub,meshlambert_frag:Ob,meshmatcap_vert:Nb,meshmatcap_frag:zb,meshnormal_vert:kb,meshnormal_frag:Hb,meshphong_vert:Vb,meshphong_frag:Gb,meshphysical_vert:Wb,meshphysical_frag:Xb,meshtoon_vert:qb,meshtoon_frag:Qb,points_vert:Yb,points_frag:Kb,shadow_vert:jb,shadow_frag:$b,sprite_vert:Zb,sprite_frag:Jb},Fe={common:{diffuse:{value:new nt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Qe},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Qe}},envmap:{envMap:{value:null},envMapRotation:{value:new Qe},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Qe}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Qe}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Qe},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Qe},normalScale:{value:new ze(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Qe},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Qe}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Qe}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Qe}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new nt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new nt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0},uvTransform:{value:new Qe}},sprite:{diffuse:{value:new nt(16777215)},opacity:{value:1},center:{value:new ze(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Qe},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0}}},Ci={basic:{uniforms:fn([Fe.common,Fe.specularmap,Fe.envmap,Fe.aomap,Fe.lightmap,Fe.fog]),vertexShader:Ze.meshbasic_vert,fragmentShader:Ze.meshbasic_frag},lambert:{uniforms:fn([Fe.common,Fe.specularmap,Fe.envmap,Fe.aomap,Fe.lightmap,Fe.emissivemap,Fe.bumpmap,Fe.normalmap,Fe.displacementmap,Fe.fog,Fe.lights,{emissive:{value:new nt(0)}}]),vertexShader:Ze.meshlambert_vert,fragmentShader:Ze.meshlambert_frag},phong:{uniforms:fn([Fe.common,Fe.specularmap,Fe.envmap,Fe.aomap,Fe.lightmap,Fe.emissivemap,Fe.bumpmap,Fe.normalmap,Fe.displacementmap,Fe.fog,Fe.lights,{emissive:{value:new nt(0)},specular:{value:new nt(1118481)},shininess:{value:30}}]),vertexShader:Ze.meshphong_vert,fragmentShader:Ze.meshphong_frag},standard:{uniforms:fn([Fe.common,Fe.envmap,Fe.aomap,Fe.lightmap,Fe.emissivemap,Fe.bumpmap,Fe.normalmap,Fe.displacementmap,Fe.roughnessmap,Fe.metalnessmap,Fe.fog,Fe.lights,{emissive:{value:new nt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:Ze.meshphysical_vert,fragmentShader:Ze.meshphysical_frag},toon:{uniforms:fn([Fe.common,Fe.aomap,Fe.lightmap,Fe.emissivemap,Fe.bumpmap,Fe.normalmap,Fe.displacementmap,Fe.gradientmap,Fe.fog,Fe.lights,{emissive:{value:new nt(0)}}]),vertexShader:Ze.meshtoon_vert,fragmentShader:Ze.meshtoon_frag},matcap:{uniforms:fn([Fe.common,Fe.bumpmap,Fe.normalmap,Fe.displacementmap,Fe.fog,{matcap:{value:null}}]),vertexShader:Ze.meshmatcap_vert,fragmentShader:Ze.meshmatcap_frag},points:{uniforms:fn([Fe.points,Fe.fog]),vertexShader:Ze.points_vert,fragmentShader:Ze.points_frag},dashed:{uniforms:fn([Fe.common,Fe.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:Ze.linedashed_vert,fragmentShader:Ze.linedashed_frag},depth:{uniforms:fn([Fe.common,Fe.displacementmap]),vertexShader:Ze.depth_vert,fragmentShader:Ze.depth_frag},normal:{uniforms:fn([Fe.common,Fe.bumpmap,Fe.normalmap,Fe.displacementmap,{opacity:{value:1}}]),vertexShader:Ze.meshnormal_vert,fragmentShader:Ze.meshnormal_frag},sprite:{uniforms:fn([Fe.sprite,Fe.fog]),vertexShader:Ze.sprite_vert,fragmentShader:Ze.sprite_frag},background:{uniforms:{uvTransform:{value:new Qe},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:Ze.background_vert,fragmentShader:Ze.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Qe}},vertexShader:Ze.backgroundCube_vert,fragmentShader:Ze.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:Ze.cube_vert,fragmentShader:Ze.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:Ze.equirect_vert,fragmentShader:Ze.equirect_frag},distanceRGBA:{uniforms:fn([Fe.common,Fe.displacementmap,{referencePosition:{value:new B},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:Ze.distanceRGBA_vert,fragmentShader:Ze.distanceRGBA_frag},shadow:{uniforms:fn([Fe.lights,Fe.fog,{color:{value:new nt(0)},opacity:{value:1}}]),vertexShader:Ze.shadow_vert,fragmentShader:Ze.shadow_frag}};Ci.physical={uniforms:fn([Ci.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Qe},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Qe},clearcoatNormalScale:{value:new ze(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Qe},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Qe},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Qe},sheen:{value:0},sheenColor:{value:new nt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Qe},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Qe},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Qe},transmissionSamplerSize:{value:new ze},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Qe},attenuationDistance:{value:0},attenuationColor:{value:new nt(0)},specularColor:{value:new nt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Qe},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Qe},anisotropyVector:{value:new ze},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Qe}}]),vertexShader:Ze.meshphysical_vert,fragmentShader:Ze.meshphysical_frag};const Za={r:0,b:0,g:0},js=new _i,eM=new qe;function tM(i,e,t,n,s,r,o){const a=new nt(0);let l=r===!0?0:1,c,u,f=null,d=0,h=null;function x(A){let S=A.isScene===!0?A.background:null;return S&&S.isTexture&&(S=(A.backgroundBlurriness>0?t:e).get(S)),S}function p(A){let S=!1;const v=x(A);v===null?m(a,l):v&&v.isColor&&(m(v,1),S=!0);const y=i.xr.getEnvironmentBlendMode();y==="additive"?n.buffers.color.setClear(0,0,0,1,o):y==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,o),(i.autoClear||S)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function g(A,S){const v=x(S);v&&(v.isCubeTexture||v.mapping===jl)?(u===void 0&&(u=new Vt(new yo(1,1,1),new _n({name:"BackgroundCubeMaterial",uniforms:ho(Ci.backgroundCube.uniforms),vertexShader:Ci.backgroundCube.vertexShader,fragmentShader:Ci.backgroundCube.fragmentShader,side:wn,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),u.geometry.deleteAttribute("normal"),u.geometry.deleteAttribute("uv"),u.onBeforeRender=function(y,M,E){this.matrixWorld.copyPosition(E.matrixWorld)},Object.defineProperty(u.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),s.update(u)),js.copy(S.backgroundRotation),js.x*=-1,js.y*=-1,js.z*=-1,v.isCubeTexture&&v.isRenderTargetTexture===!1&&(js.y*=-1,js.z*=-1),u.material.uniforms.envMap.value=v,u.material.uniforms.flipEnvMap.value=v.isCubeTexture&&v.isRenderTargetTexture===!1?-1:1,u.material.uniforms.backgroundBlurriness.value=S.backgroundBlurriness,u.material.uniforms.backgroundIntensity.value=S.backgroundIntensity,u.material.uniforms.backgroundRotation.value.setFromMatrix4(eM.makeRotationFromEuler(js)),u.material.toneMapped=rt.getTransfer(v.colorSpace)!==ht,(f!==v||d!==v.version||h!==i.toneMapping)&&(u.material.needsUpdate=!0,f=v,d=v.version,h=i.toneMapping),u.layers.enableAll(),A.unshift(u,u.geometry,u.material,0,0,null)):v&&v.isTexture&&(c===void 0&&(c=new Vt(new po(2,2),new _n({name:"BackgroundMaterial",uniforms:ho(Ci.background.uniforms),vertexShader:Ci.background.vertexShader,fragmentShader:Ci.background.fragmentShader,side:Bi,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),s.update(c)),c.material.uniforms.t2D.value=v,c.material.uniforms.backgroundIntensity.value=S.backgroundIntensity,c.material.toneMapped=rt.getTransfer(v.colorSpace)!==ht,v.matrixAutoUpdate===!0&&v.updateMatrix(),c.material.uniforms.uvTransform.value.copy(v.matrix),(f!==v||d!==v.version||h!==i.toneMapping)&&(c.material.needsUpdate=!0,f=v,d=v.version,h=i.toneMapping),c.layers.enableAll(),A.unshift(c,c.geometry,c.material,0,0,null))}function m(A,S){A.getRGB(Za,G0(i)),n.buffers.color.setClear(Za.r,Za.g,Za.b,S,o)}function _(){u!==void 0&&(u.geometry.dispose(),u.material.dispose(),u=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return a},setClearColor:function(A,S=1){a.set(A),l=S,m(a,l)},getClearAlpha:function(){return l},setClearAlpha:function(A){l=A,m(a,l)},render:p,addToRenderList:g,dispose:_}}function nM(i,e){const t=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},s=d(null);let r=s,o=!1;function a(C,I,F,U,O){let k=!1;const z=f(U,F,I);r!==z&&(r=z,c(r.object)),k=h(C,U,F,O),k&&x(C,U,F,O),O!==null&&e.update(O,i.ELEMENT_ARRAY_BUFFER),(k||o)&&(o=!1,S(C,I,F,U),O!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,e.get(O).buffer))}function l(){return i.createVertexArray()}function c(C){return i.bindVertexArray(C)}function u(C){return i.deleteVertexArray(C)}function f(C,I,F){const U=F.wireframe===!0;let O=n[C.id];O===void 0&&(O={},n[C.id]=O);let k=O[I.id];k===void 0&&(k={},O[I.id]=k);let z=k[U];return z===void 0&&(z=d(l()),k[U]=z),z}function d(C){const I=[],F=[],U=[];for(let O=0;O<t;O++)I[O]=0,F[O]=0,U[O]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:I,enabledAttributes:F,attributeDivisors:U,object:C,attributes:{},index:null}}function h(C,I,F,U){const O=r.attributes,k=I.attributes;let z=0;const V=F.getAttributes();for(const H in V)if(V[H].location>=0){const oe=O[H];let Se=k[H];if(Se===void 0&&(H==="instanceMatrix"&&C.instanceMatrix&&(Se=C.instanceMatrix),H==="instanceColor"&&C.instanceColor&&(Se=C.instanceColor)),oe===void 0||oe.attribute!==Se||Se&&oe.data!==Se.data)return!0;z++}return r.attributesNum!==z||r.index!==U}function x(C,I,F,U){const O={},k=I.attributes;let z=0;const V=F.getAttributes();for(const H in V)if(V[H].location>=0){let oe=k[H];oe===void 0&&(H==="instanceMatrix"&&C.instanceMatrix&&(oe=C.instanceMatrix),H==="instanceColor"&&C.instanceColor&&(oe=C.instanceColor));const Se={};Se.attribute=oe,oe&&oe.data&&(Se.data=oe.data),O[H]=Se,z++}r.attributes=O,r.attributesNum=z,r.index=U}function p(){const C=r.newAttributes;for(let I=0,F=C.length;I<F;I++)C[I]=0}function g(C){m(C,0)}function m(C,I){const F=r.newAttributes,U=r.enabledAttributes,O=r.attributeDivisors;F[C]=1,U[C]===0&&(i.enableVertexAttribArray(C),U[C]=1),O[C]!==I&&(i.vertexAttribDivisor(C,I),O[C]=I)}function _(){const C=r.newAttributes,I=r.enabledAttributes;for(let F=0,U=I.length;F<U;F++)I[F]!==C[F]&&(i.disableVertexAttribArray(F),I[F]=0)}function A(C,I,F,U,O,k,z){z===!0?i.vertexAttribIPointer(C,I,F,O,k):i.vertexAttribPointer(C,I,F,U,O,k)}function S(C,I,F,U){p();const O=U.attributes,k=F.getAttributes(),z=I.defaultAttributeValues;for(const V in k){const H=k[V];if(H.location>=0){let $=O[V];if($===void 0&&(V==="instanceMatrix"&&C.instanceMatrix&&($=C.instanceMatrix),V==="instanceColor"&&C.instanceColor&&($=C.instanceColor)),$!==void 0){const oe=$.normalized,Se=$.itemSize,we=e.get($);if(we===void 0)continue;const Le=we.buffer,fe=we.type,re=we.bytesPerElement,X=fe===i.INT||fe===i.UNSIGNED_INT||$.gpuType===Kf;if($.isInterleavedBufferAttribute){const ee=$.data,pe=ee.stride,be=$.offset;if(ee.isInstancedInterleavedBuffer){for(let xe=0;xe<H.locationSize;xe++)m(H.location+xe,ee.meshPerAttribute);C.isInstancedMesh!==!0&&U._maxInstanceCount===void 0&&(U._maxInstanceCount=ee.meshPerAttribute*ee.count)}else for(let xe=0;xe<H.locationSize;xe++)g(H.location+xe);i.bindBuffer(i.ARRAY_BUFFER,Le);for(let xe=0;xe<H.locationSize;xe++)A(H.location+xe,Se/H.locationSize,fe,oe,pe*re,(be+Se/H.locationSize*xe)*re,X)}else{if($.isInstancedBufferAttribute){for(let ee=0;ee<H.locationSize;ee++)m(H.location+ee,$.meshPerAttribute);C.isInstancedMesh!==!0&&U._maxInstanceCount===void 0&&(U._maxInstanceCount=$.meshPerAttribute*$.count)}else for(let ee=0;ee<H.locationSize;ee++)g(H.location+ee);i.bindBuffer(i.ARRAY_BUFFER,Le);for(let ee=0;ee<H.locationSize;ee++)A(H.location+ee,Se/H.locationSize,fe,oe,Se*re,Se/H.locationSize*ee*re,X)}}else if(z!==void 0){const oe=z[V];if(oe!==void 0)switch(oe.length){case 2:i.vertexAttrib2fv(H.location,oe);break;case 3:i.vertexAttrib3fv(H.location,oe);break;case 4:i.vertexAttrib4fv(H.location,oe);break;default:i.vertexAttrib1fv(H.location,oe)}}}}_()}function v(){E();for(const C in n){const I=n[C];for(const F in I){const U=I[F];for(const O in U)u(U[O].object),delete U[O];delete I[F]}delete n[C]}}function y(C){if(n[C.id]===void 0)return;const I=n[C.id];for(const F in I){const U=I[F];for(const O in U)u(U[O].object),delete U[O];delete I[F]}delete n[C.id]}function M(C){for(const I in n){const F=n[I];if(F[C.id]===void 0)continue;const U=F[C.id];for(const O in U)u(U[O].object),delete U[O];delete F[C.id]}}function E(){b(),o=!0,r!==s&&(r=s,c(r.object))}function b(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:a,reset:E,resetDefaultState:b,dispose:v,releaseStatesOfGeometry:y,releaseStatesOfProgram:M,initAttributes:p,enableAttribute:g,disableUnusedAttributes:_}}function iM(i,e,t){let n;function s(c){n=c}function r(c,u){i.drawArrays(n,c,u),t.update(u,n,1)}function o(c,u,f){f!==0&&(i.drawArraysInstanced(n,c,u,f),t.update(u,n,f))}function a(c,u,f){if(f===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,c,0,u,0,f);let h=0;for(let x=0;x<f;x++)h+=u[x];t.update(h,n,1)}function l(c,u,f,d){if(f===0)return;const h=e.get("WEBGL_multi_draw");if(h===null)for(let x=0;x<c.length;x++)o(c[x],u[x],d[x]);else{h.multiDrawArraysInstancedWEBGL(n,c,0,u,0,d,0,f);let x=0;for(let p=0;p<f;p++)x+=u[p]*d[p];t.update(x,n,1)}}this.setMode=s,this.render=r,this.renderInstances=o,this.renderMultiDraw=a,this.renderMultiDrawInstances=l}function sM(i,e,t,n){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){const M=e.get("EXT_texture_filter_anisotropic");s=i.getParameter(M.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function o(M){return!(M!==gn&&n.convert(M)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function a(M){const E=M===pr&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(M!==Ui&&n.convert(M)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE)&&M!==mi&&!E)}function l(M){if(M==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";M="mediump"}return M==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let c=t.precision!==void 0?t.precision:"highp";const u=l(c);u!==c&&(je("WebGLRenderer:",c,"not supported, using",u,"instead."),c=u);const f=t.logarithmicDepthBuffer===!0,d=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),h=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),x=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),p=i.getParameter(i.MAX_TEXTURE_SIZE),g=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),m=i.getParameter(i.MAX_VERTEX_ATTRIBS),_=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),A=i.getParameter(i.MAX_VARYING_VECTORS),S=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),v=x>0,y=i.getParameter(i.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:l,textureFormatReadable:o,textureTypeReadable:a,precision:c,logarithmicDepthBuffer:f,reversedDepthBuffer:d,maxTextures:h,maxVertexTextures:x,maxTextureSize:p,maxCubemapSize:g,maxAttributes:m,maxVertexUniforms:_,maxVaryings:A,maxFragmentUniforms:S,vertexTextures:v,maxSamples:y}}function rM(i){const e=this;let t=null,n=0,s=!1,r=!1;const o=new vs,a=new Qe,l={value:null,needsUpdate:!1};this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(f,d){const h=f.length!==0||d||n!==0||s;return s=d,n=f.length,h},this.beginShadows=function(){r=!0,u(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(f,d){t=u(f,d,0)},this.setState=function(f,d,h){const x=f.clippingPlanes,p=f.clipIntersection,g=f.clipShadows,m=i.get(f);if(!s||x===null||x.length===0||r&&!g)r?u(null):c();else{const _=r?0:n,A=_*4;let S=m.clippingState||null;l.value=S,S=u(x,d,A,h);for(let v=0;v!==A;++v)S[v]=t[v];m.clippingState=S,this.numIntersection=p?this.numPlanes:0,this.numPlanes+=_}};function c(){l.value!==t&&(l.value=t,l.needsUpdate=n>0),e.numPlanes=n,e.numIntersection=0}function u(f,d,h,x){const p=f!==null?f.length:0;let g=null;if(p!==0){if(g=l.value,x!==!0||g===null){const m=h+p*4,_=d.matrixWorldInverse;a.getNormalMatrix(_),(g===null||g.length<m)&&(g=new Float32Array(m));for(let A=0,S=h;A!==p;++A,S+=4)o.copy(f[A]).applyMatrix4(_,a),o.normal.toArray(g,S),g[S+3]=o.constant}l.value=g,l.needsUpdate=!0}return e.numPlanes=p,e.numIntersection=0,g}}function oM(i){let e=new WeakMap;function t(o,a){return a===Iu?o.mapping=lo:a===Du&&(o.mapping=co),o}function n(o){if(o&&o.isTexture){const a=o.mapping;if(a===Iu||a===Du)if(e.has(o)){const l=e.get(o).texture;return t(l,o.mapping)}else{const l=o.image;if(l&&l.height>0){const c=new ov(l.height);return c.fromEquirectangularTexture(i,o),e.set(o,c),o.addEventListener("dispose",s),t(c.texture,o.mapping)}else return null}}return o}function s(o){const a=o.target;a.removeEventListener("dispose",s);const l=e.get(a);l!==void 0&&(e.delete(a),l.dispose())}function r(){e=new WeakMap}return{get:n,dispose:r}}const Ms=4,tp=[.125,.215,.35,.446,.526,.582],ir=20,aM=256,Po=new sd,np=new nt;let Gc=null,Wc=0,Xc=0,qc=!1;const lM=new B;class ip{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,n=.1,s=100,r={}){const{size:o=256,position:a=lM}=r;Gc=this._renderer.getRenderTarget(),Wc=this._renderer.getActiveCubeFace(),Xc=this._renderer.getActiveMipmapLevel(),qc=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(o);const l=this._allocateTargets();return l.depthBuffer=!0,this._sceneToCubeUV(e,n,s,l,a),t>0&&this._blur(l,0,0,t),this._applyPMREM(l),this._cleanup(l),l}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=op(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=rp(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(Gc,Wc,Xc),this._renderer.xr.enabled=qc,e.scissorTest=!1,Br(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===lo||e.mapping===co?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),Gc=this._renderer.getRenderTarget(),Wc=this._renderer.getActiveCubeFace(),Xc=this._renderer.getActiveMipmapLevel(),qc=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const n=t||this._allocateTargets();return this._textureToCubeUV(e,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,n={magFilter:ii,minFilter:ii,generateMipmaps:!1,type:pr,format:gn,colorSpace:fo,depthBuffer:!1},s=sp(e,t,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=sp(e,t,n);const{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=cM(r)),this._blurMaterial=fM(r,e,t),this._ggxMaterial=uM(r,e,t)}return s}_compileMaterial(e){const t=new Vt(new An,e);this._renderer.compile(t,Po)}_sceneToCubeUV(e,t,n,s,r){const l=new ei(90,1,t,n),c=[1,-1,1,1,1,1],u=[1,1,1,-1,-1,-1],f=this._renderer,d=f.autoClear,h=f.toneMapping;f.getClearColor(np),f.toneMapping=Ds,f.autoClear=!1,f.state.buffers.depth.getReversed()&&(f.setRenderTarget(s),f.clearDepth(),f.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new Vt(new yo,new hr({name:"PMREM.Background",side:wn,depthWrite:!1,depthTest:!1})));const p=this._backgroundBox,g=p.material;let m=!1;const _=e.background;_?_.isColor&&(g.color.copy(_),e.background=null,m=!0):(g.color.copy(np),m=!0);for(let A=0;A<6;A++){const S=A%3;S===0?(l.up.set(0,c[A],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x+u[A],r.y,r.z)):S===1?(l.up.set(0,0,c[A]),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y+u[A],r.z)):(l.up.set(0,c[A],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y,r.z+u[A]));const v=this._cubeSize;Br(s,S*v,A>2?v:0,v,v),f.setRenderTarget(s),m&&f.render(p,l),f.render(e,l)}f.toneMapping=h,f.autoClear=d,e.background=_}_textureToCubeUV(e,t){const n=this._renderer,s=e.mapping===lo||e.mapping===co;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=op()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=rp());const r=s?this._cubemapMaterial:this._equirectMaterial,o=this._lodMeshes[0];o.material=r;const a=r.uniforms;a.envMap.value=e;const l=this._cubeSize;Br(t,0,0,3*l,2*l),n.setRenderTarget(t),n.render(o,Po)}_applyPMREM(e){const t=this._renderer,n=t.autoClear;t.autoClear=!1;const s=this._lodMeshes.length;for(let r=1;r<s;r++)this._applyGGXFilter(e,r-1,r);t.autoClear=n}_applyGGXFilter(e,t,n){const s=this._renderer,r=this._pingPongRenderTarget,o=this._ggxMaterial,a=this._lodMeshes[n];a.material=o;const l=o.uniforms,c=n/(this._lodMeshes.length-1),u=t/(this._lodMeshes.length-1),f=Math.sqrt(c*c-u*u),d=.05+c*.95,h=f*d,{_lodMax:x}=this,p=this._sizeLods[n],g=3*p*(n>x-Ms?n-x+Ms:0),m=4*(this._cubeSize-p);l.envMap.value=e.texture,l.roughness.value=h,l.mipInt.value=x-t,Br(r,g,m,3*p,2*p),s.setRenderTarget(r),s.render(a,Po),l.envMap.value=r.texture,l.roughness.value=0,l.mipInt.value=x-n,Br(e,g,m,3*p,2*p),s.setRenderTarget(e),s.render(a,Po)}_blur(e,t,n,s,r){const o=this._pingPongRenderTarget;this._halfBlur(e,o,t,n,s,"latitudinal",r),this._halfBlur(o,e,n,n,s,"longitudinal",r)}_halfBlur(e,t,n,s,r,o,a){const l=this._renderer,c=this._blurMaterial;o!=="latitudinal"&&o!=="longitudinal"&&zt("blur direction must be either latitudinal or longitudinal!");const u=3,f=this._lodMeshes[s];f.material=c;const d=c.uniforms,h=this._sizeLods[n]-1,x=isFinite(r)?Math.PI/(2*h):2*Math.PI/(2*ir-1),p=r/x,g=isFinite(r)?1+Math.floor(u*p):ir;g>ir&&je(`sigmaRadians, ${r}, is too large and will clip, as it requested ${g} samples when the maximum is set to ${ir}`);const m=[];let _=0;for(let M=0;M<ir;++M){const E=M/p,b=Math.exp(-E*E/2);m.push(b),M===0?_+=b:M<g&&(_+=2*b)}for(let M=0;M<m.length;M++)m[M]=m[M]/_;d.envMap.value=e.texture,d.samples.value=g,d.weights.value=m,d.latitudinal.value=o==="latitudinal",a&&(d.poleAxis.value=a);const{_lodMax:A}=this;d.dTheta.value=x,d.mipInt.value=A-n;const S=this._sizeLods[s],v=3*S*(s>A-Ms?s-A+Ms:0),y=4*(this._cubeSize-S);Br(t,v,y,3*S,2*S),l.setRenderTarget(t),l.render(f,Po)}}function cM(i){const e=[],t=[],n=[];let s=i;const r=i-Ms+1+tp.length;for(let o=0;o<r;o++){const a=Math.pow(2,s);e.push(a);let l=1/a;o>i-Ms?l=tp[o-i+Ms-1]:o===0&&(l=0),t.push(l);const c=1/(a-2),u=-c,f=1+c,d=[u,u,f,u,f,f,u,u,f,f,u,f],h=6,x=6,p=3,g=2,m=1,_=new Float32Array(p*x*h),A=new Float32Array(g*x*h),S=new Float32Array(m*x*h);for(let y=0;y<h;y++){const M=y%3*2/3-1,E=y>2?0:-1,b=[M,E,0,M+2/3,E,0,M+2/3,E+1,0,M,E,0,M+2/3,E+1,0,M,E+1,0];_.set(b,p*x*y),A.set(d,g*x*y);const C=[y,y,y,y,y,y];S.set(C,m*x*y)}const v=new An;v.setAttribute("position",new li(_,p)),v.setAttribute("uv",new li(A,g)),v.setAttribute("faceIndex",new li(S,m)),n.push(new Vt(v,null)),s>Ms&&s--}return{lodMeshes:n,sizeLods:e,sigmas:t}}function sp(i,e,t){const n=new Us(i,e,t);return n.texture.mapping=jl,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function Br(i,e,t,n,s){i.viewport.set(e,t,n,s),i.scissor.set(e,t,n,s)}function uM(i,e,t){return new _n({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:aM,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:Jl(),fragmentShader:`

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
		`,blending:is,depthTest:!1,depthWrite:!1})}function fM(i,e,t){const n=new Float32Array(ir),s=new B(0,1,0);return new _n({name:"SphericalGaussianBlur",defines:{n:ir,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:Jl(),fragmentShader:`

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
		`,blending:is,depthTest:!1,depthWrite:!1})}function rp(){return new _n({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Jl(),fragmentShader:`

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
		`,blending:is,depthTest:!1,depthWrite:!1})}function op(){return new _n({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Jl(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:is,depthTest:!1,depthWrite:!1})}function Jl(){return`

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
	`}function dM(i){let e=new WeakMap,t=null;function n(a){if(a&&a.isTexture){const l=a.mapping,c=l===Iu||l===Du,u=l===lo||l===co;if(c||u){let f=e.get(a);const d=f!==void 0?f.texture.pmremVersion:0;if(a.isRenderTargetTexture&&a.pmremVersion!==d)return t===null&&(t=new ip(i)),f=c?t.fromEquirectangular(a,f):t.fromCubemap(a,f),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),f.texture;if(f!==void 0)return f.texture;{const h=a.image;return c&&h&&h.height>0||u&&h&&s(h)?(t===null&&(t=new ip(i)),f=c?t.fromEquirectangular(a):t.fromCubemap(a),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),a.addEventListener("dispose",r),f.texture):null}}}return a}function s(a){let l=0;const c=6;for(let u=0;u<c;u++)a[u]!==void 0&&l++;return l===c}function r(a){const l=a.target;l.removeEventListener("dispose",r);const c=e.get(l);c!==void 0&&(e.delete(l),c.dispose())}function o(){e=new WeakMap,t!==null&&(t.dispose(),t=null)}return{get:n,dispose:o}}function hM(i){const e={};function t(n){if(e[n]!==void 0)return e[n];const s=i.getExtension(n);return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(n){const s=t(n);return s===null&&ca("WebGLRenderer: "+n+" extension not supported."),s}}}function pM(i,e,t,n){const s={},r=new WeakMap;function o(f){const d=f.target;d.index!==null&&e.remove(d.index);for(const x in d.attributes)e.remove(d.attributes[x]);d.removeEventListener("dispose",o),delete s[d.id];const h=r.get(d);h&&(e.remove(h),r.delete(d)),n.releaseStatesOfGeometry(d),d.isInstancedBufferGeometry===!0&&delete d._maxInstanceCount,t.memory.geometries--}function a(f,d){return s[d.id]===!0||(d.addEventListener("dispose",o),s[d.id]=!0,t.memory.geometries++),d}function l(f){const d=f.attributes;for(const h in d)e.update(d[h],i.ARRAY_BUFFER)}function c(f){const d=[],h=f.index,x=f.attributes.position;let p=0;if(h!==null){const _=h.array;p=h.version;for(let A=0,S=_.length;A<S;A+=3){const v=_[A+0],y=_[A+1],M=_[A+2];d.push(v,y,y,M,M,v)}}else if(x!==void 0){const _=x.array;p=x.version;for(let A=0,S=_.length/3-1;A<S;A+=3){const v=A+0,y=A+1,M=A+2;d.push(v,y,y,M,M,v)}}else return;const g=new(U0(d)?V0:H0)(d,1);g.version=p;const m=r.get(f);m&&e.remove(m),r.set(f,g)}function u(f){const d=r.get(f);if(d){const h=f.index;h!==null&&d.version<h.version&&c(f)}else c(f);return r.get(f)}return{get:a,update:l,getWireframeAttribute:u}}function mM(i,e,t){let n;function s(d){n=d}let r,o;function a(d){r=d.type,o=d.bytesPerElement}function l(d,h){i.drawElements(n,h,r,d*o),t.update(h,n,1)}function c(d,h,x){x!==0&&(i.drawElementsInstanced(n,h,r,d*o,x),t.update(h,n,x))}function u(d,h,x){if(x===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,h,0,r,d,0,x);let g=0;for(let m=0;m<x;m++)g+=h[m];t.update(g,n,1)}function f(d,h,x,p){if(x===0)return;const g=e.get("WEBGL_multi_draw");if(g===null)for(let m=0;m<d.length;m++)c(d[m]/o,h[m],p[m]);else{g.multiDrawElementsInstancedWEBGL(n,h,0,r,d,0,p,0,x);let m=0;for(let _=0;_<x;_++)m+=h[_]*p[_];t.update(m,n,1)}}this.setMode=s,this.setIndex=a,this.render=l,this.renderInstances=c,this.renderMultiDraw=u,this.renderMultiDrawInstances=f}function gM(i){const e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function n(r,o,a){switch(t.calls++,o){case i.TRIANGLES:t.triangles+=a*(r/3);break;case i.LINES:t.lines+=a*(r/2);break;case i.LINE_STRIP:t.lines+=a*(r-1);break;case i.LINE_LOOP:t.lines+=a*r;break;case i.POINTS:t.points+=a*r;break;default:zt("WebGLInfo: Unknown draw mode:",o);break}}function s(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:s,update:n}}function xM(i,e,t){const n=new WeakMap,s=new Et;function r(o,a,l){const c=o.morphTargetInfluences,u=a.morphAttributes.position||a.morphAttributes.normal||a.morphAttributes.color,f=u!==void 0?u.length:0;let d=n.get(a);if(d===void 0||d.count!==f){let C=function(){E.dispose(),n.delete(a),a.removeEventListener("dispose",C)};var h=C;d!==void 0&&d.texture.dispose();const x=a.morphAttributes.position!==void 0,p=a.morphAttributes.normal!==void 0,g=a.morphAttributes.color!==void 0,m=a.morphAttributes.position||[],_=a.morphAttributes.normal||[],A=a.morphAttributes.color||[];let S=0;x===!0&&(S=1),p===!0&&(S=2),g===!0&&(S=3);let v=a.attributes.position.count*S,y=1;v>e.maxTextureSize&&(y=Math.ceil(v/e.maxTextureSize),v=e.maxTextureSize);const M=new Float32Array(v*y*4*f),E=new N0(M,v,y,f);E.type=mi,E.needsUpdate=!0;const b=S*4;for(let I=0;I<f;I++){const F=m[I],U=_[I],O=A[I],k=v*y*4*I;for(let z=0;z<F.count;z++){const V=z*b;x===!0&&(s.fromBufferAttribute(F,z),M[k+V+0]=s.x,M[k+V+1]=s.y,M[k+V+2]=s.z,M[k+V+3]=0),p===!0&&(s.fromBufferAttribute(U,z),M[k+V+4]=s.x,M[k+V+5]=s.y,M[k+V+6]=s.z,M[k+V+7]=0),g===!0&&(s.fromBufferAttribute(O,z),M[k+V+8]=s.x,M[k+V+9]=s.y,M[k+V+10]=s.z,M[k+V+11]=O.itemSize===4?s.w:1)}}d={count:f,texture:E,size:new ze(v,y)},n.set(a,d),a.addEventListener("dispose",C)}if(o.isInstancedMesh===!0&&o.morphTexture!==null)l.getUniforms().setValue(i,"morphTexture",o.morphTexture,t);else{let x=0;for(let g=0;g<c.length;g++)x+=c[g];const p=a.morphTargetsRelative?1:1-x;l.getUniforms().setValue(i,"morphTargetBaseInfluence",p),l.getUniforms().setValue(i,"morphTargetInfluences",c)}l.getUniforms().setValue(i,"morphTargetsTexture",d.texture,t),l.getUniforms().setValue(i,"morphTargetsTextureSize",d.size)}return{update:r}}function _M(i,e,t,n){let s=new WeakMap;function r(l){const c=n.render.frame,u=l.geometry,f=e.get(l,u);if(s.get(f)!==c&&(e.update(f),s.set(f,c)),l.isInstancedMesh&&(l.hasEventListener("dispose",a)===!1&&l.addEventListener("dispose",a),s.get(l)!==c&&(t.update(l.instanceMatrix,i.ARRAY_BUFFER),l.instanceColor!==null&&t.update(l.instanceColor,i.ARRAY_BUFFER),s.set(l,c))),l.isSkinnedMesh){const d=l.skeleton;s.get(d)!==c&&(d.update(),s.set(d,c))}return f}function o(){s=new WeakMap}function a(l){const c=l.target;c.removeEventListener("dispose",a),t.remove(c.instanceMatrix),c.instanceColor!==null&&t.remove(c.instanceColor)}return{update:r,dispose:o}}const K0=new xn,ap=new nd(1,1),j0=new N0,$0=new kS,Z0=new X0,lp=[],cp=[],up=new Float32Array(16),fp=new Float32Array(9),dp=new Float32Array(4);function bo(i,e,t){const n=i[0];if(n<=0||n>0)return i;const s=e*t;let r=lp[s];if(r===void 0&&(r=new Float32Array(s),lp[s]=r),e!==0){n.toArray(r,0);for(let o=1,a=0;o!==e;++o)a+=t,i[o].toArray(r,a)}return r}function Qt(i,e){if(i.length!==e.length)return!1;for(let t=0,n=i.length;t<n;t++)if(i[t]!==e[t])return!1;return!0}function Yt(i,e){for(let t=0,n=e.length;t<n;t++)i[t]=e[t]}function ec(i,e){let t=cp[e];t===void 0&&(t=new Int32Array(e),cp[e]=t);for(let n=0;n!==e;++n)t[n]=i.allocateTextureUnit();return t}function AM(i,e){const t=this.cache;t[0]!==e&&(i.uniform1f(this.addr,e),t[0]=e)}function SM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Qt(t,e))return;i.uniform2fv(this.addr,e),Yt(t,e)}}function vM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(i.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(Qt(t,e))return;i.uniform3fv(this.addr,e),Yt(t,e)}}function yM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Qt(t,e))return;i.uniform4fv(this.addr,e),Yt(t,e)}}function bM(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Qt(t,e))return;i.uniformMatrix2fv(this.addr,!1,e),Yt(t,e)}else{if(Qt(t,n))return;dp.set(n),i.uniformMatrix2fv(this.addr,!1,dp),Yt(t,n)}}function MM(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Qt(t,e))return;i.uniformMatrix3fv(this.addr,!1,e),Yt(t,e)}else{if(Qt(t,n))return;fp.set(n),i.uniformMatrix3fv(this.addr,!1,fp),Yt(t,n)}}function CM(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Qt(t,e))return;i.uniformMatrix4fv(this.addr,!1,e),Yt(t,e)}else{if(Qt(t,n))return;up.set(n),i.uniformMatrix4fv(this.addr,!1,up),Yt(t,n)}}function TM(i,e){const t=this.cache;t[0]!==e&&(i.uniform1i(this.addr,e),t[0]=e)}function EM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Qt(t,e))return;i.uniform2iv(this.addr,e),Yt(t,e)}}function wM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Qt(t,e))return;i.uniform3iv(this.addr,e),Yt(t,e)}}function RM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Qt(t,e))return;i.uniform4iv(this.addr,e),Yt(t,e)}}function IM(i,e){const t=this.cache;t[0]!==e&&(i.uniform1ui(this.addr,e),t[0]=e)}function DM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Qt(t,e))return;i.uniform2uiv(this.addr,e),Yt(t,e)}}function PM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Qt(t,e))return;i.uniform3uiv(this.addr,e),Yt(t,e)}}function FM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Qt(t,e))return;i.uniform4uiv(this.addr,e),Yt(t,e)}}function LM(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s);let r;this.type===i.SAMPLER_2D_SHADOW?(ap.compareFunction=B0,r=ap):r=K0,t.setTexture2D(e||r,s)}function BM(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture3D(e||$0,s)}function UM(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTextureCube(e||Z0,s)}function OM(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture2DArray(e||j0,s)}function NM(i){switch(i){case 5126:return AM;case 35664:return SM;case 35665:return vM;case 35666:return yM;case 35674:return bM;case 35675:return MM;case 35676:return CM;case 5124:case 35670:return TM;case 35667:case 35671:return EM;case 35668:case 35672:return wM;case 35669:case 35673:return RM;case 5125:return IM;case 36294:return DM;case 36295:return PM;case 36296:return FM;case 35678:case 36198:case 36298:case 36306:case 35682:return LM;case 35679:case 36299:case 36307:return BM;case 35680:case 36300:case 36308:case 36293:return UM;case 36289:case 36303:case 36311:case 36292:return OM}}function zM(i,e){i.uniform1fv(this.addr,e)}function kM(i,e){const t=bo(e,this.size,2);i.uniform2fv(this.addr,t)}function HM(i,e){const t=bo(e,this.size,3);i.uniform3fv(this.addr,t)}function VM(i,e){const t=bo(e,this.size,4);i.uniform4fv(this.addr,t)}function GM(i,e){const t=bo(e,this.size,4);i.uniformMatrix2fv(this.addr,!1,t)}function WM(i,e){const t=bo(e,this.size,9);i.uniformMatrix3fv(this.addr,!1,t)}function XM(i,e){const t=bo(e,this.size,16);i.uniformMatrix4fv(this.addr,!1,t)}function qM(i,e){i.uniform1iv(this.addr,e)}function QM(i,e){i.uniform2iv(this.addr,e)}function YM(i,e){i.uniform3iv(this.addr,e)}function KM(i,e){i.uniform4iv(this.addr,e)}function jM(i,e){i.uniform1uiv(this.addr,e)}function $M(i,e){i.uniform2uiv(this.addr,e)}function ZM(i,e){i.uniform3uiv(this.addr,e)}function JM(i,e){i.uniform4uiv(this.addr,e)}function eC(i,e,t){const n=this.cache,s=e.length,r=ec(t,s);Qt(n,r)||(i.uniform1iv(this.addr,r),Yt(n,r));for(let o=0;o!==s;++o)t.setTexture2D(e[o]||K0,r[o])}function tC(i,e,t){const n=this.cache,s=e.length,r=ec(t,s);Qt(n,r)||(i.uniform1iv(this.addr,r),Yt(n,r));for(let o=0;o!==s;++o)t.setTexture3D(e[o]||$0,r[o])}function nC(i,e,t){const n=this.cache,s=e.length,r=ec(t,s);Qt(n,r)||(i.uniform1iv(this.addr,r),Yt(n,r));for(let o=0;o!==s;++o)t.setTextureCube(e[o]||Z0,r[o])}function iC(i,e,t){const n=this.cache,s=e.length,r=ec(t,s);Qt(n,r)||(i.uniform1iv(this.addr,r),Yt(n,r));for(let o=0;o!==s;++o)t.setTexture2DArray(e[o]||j0,r[o])}function sC(i){switch(i){case 5126:return zM;case 35664:return kM;case 35665:return HM;case 35666:return VM;case 35674:return GM;case 35675:return WM;case 35676:return XM;case 5124:case 35670:return qM;case 35667:case 35671:return QM;case 35668:case 35672:return YM;case 35669:case 35673:return KM;case 5125:return jM;case 36294:return $M;case 36295:return ZM;case 36296:return JM;case 35678:case 36198:case 36298:case 36306:case 35682:return eC;case 35679:case 36299:case 36307:return tC;case 35680:case 36300:case 36308:case 36293:return nC;case 36289:case 36303:case 36311:case 36292:return iC}}class rC{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.setValue=NM(t.type)}}class oC{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=sC(t.type)}}class aC{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,n){const s=this.seq;for(let r=0,o=s.length;r!==o;++r){const a=s[r];a.setValue(e,t[a.id],n)}}}const Qc=/(\w+)(\])?(\[|\.)?/g;function hp(i,e){i.seq.push(e),i.map[e.id]=e}function lC(i,e,t){const n=i.name,s=n.length;for(Qc.lastIndex=0;;){const r=Qc.exec(n),o=Qc.lastIndex;let a=r[1];const l=r[2]==="]",c=r[3];if(l&&(a=a|0),c===void 0||c==="["&&o+2===s){hp(t,c===void 0?new rC(a,i,e):new oC(a,i,e));break}else{let f=t.map[a];f===void 0&&(f=new aC(a),hp(t,f)),t=f}}}class xl{constructor(e,t){this.seq=[],this.map={};const n=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let s=0;s<n;++s){const r=e.getActiveUniform(t,s),o=e.getUniformLocation(t,r.name);lC(r,o,this)}}setValue(e,t,n,s){const r=this.map[t];r!==void 0&&r.setValue(e,n,s)}setOptional(e,t,n){const s=t[n];s!==void 0&&this.setValue(e,n,s)}static upload(e,t,n,s){for(let r=0,o=t.length;r!==o;++r){const a=t[r],l=n[a.id];l.needsUpdate!==!1&&a.setValue(e,l.value,s)}}static seqWithValue(e,t){const n=[];for(let s=0,r=e.length;s!==r;++s){const o=e[s];o.id in t&&n.push(o)}return n}}function pp(i,e,t){const n=i.createShader(e);return i.shaderSource(n,t),i.compileShader(n),n}const cC=37297;let uC=0;function fC(i,e){const t=i.split(`
`),n=[],s=Math.max(e-6,0),r=Math.min(e+6,t.length);for(let o=s;o<r;o++){const a=o+1;n.push(`${a===e?">":" "} ${a}: ${t[o]}`)}return n.join(`
`)}const mp=new Qe;function dC(i){rt._getMatrix(mp,rt.workingColorSpace,i);const e=`mat3( ${mp.elements.map(t=>t.toFixed(4))} )`;switch(rt.getTransfer(i)){case El:return[e,"LinearTransferOETF"];case ht:return[e,"sRGBTransferOETF"];default:return je("WebGLProgram: Unsupported color space: ",i),[e,"LinearTransferOETF"]}}function gp(i,e,t){const n=i.getShaderParameter(e,i.COMPILE_STATUS),r=(i.getShaderInfoLog(e)||"").trim();if(n&&r==="")return"";const o=/ERROR: 0:(\d+)/.exec(r);if(o){const a=parseInt(o[1]);return t.toUpperCase()+`

`+r+`

`+fC(i.getShaderSource(e),a)}else return r}function hC(i,e){const t=dC(e);return[`vec4 ${i}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}function pC(i,e){let t;switch(e){case dS:t="Linear";break;case hS:t="Reinhard";break;case pS:t="Cineon";break;case mS:t="ACESFilmic";break;case xS:t="AgX";break;case _S:t="Neutral";break;case gS:t="Custom";break;default:je("WebGLProgram: Unsupported toneMapping:",e),t="Linear"}return"vec3 "+i+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}const Ja=new B;function mC(){rt.getLuminanceCoefficients(Ja);const i=Ja.x.toFixed(4),e=Ja.y.toFixed(4),t=Ja.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function gC(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(Bo).join(`
`)}function xC(i){const e=[];for(const t in i){const n=i[t];n!==!1&&e.push("#define "+t+" "+n)}return e.join(`
`)}function _C(i,e){const t={},n=i.getProgramParameter(e,i.ACTIVE_ATTRIBUTES);for(let s=0;s<n;s++){const r=i.getActiveAttrib(e,s),o=r.name;let a=1;r.type===i.FLOAT_MAT2&&(a=2),r.type===i.FLOAT_MAT3&&(a=3),r.type===i.FLOAT_MAT4&&(a=4),t[o]={type:r.type,location:i.getAttribLocation(e,o),locationSize:a}}return t}function Bo(i){return i!==""}function xp(i,e){const t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return i.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function _p(i,e){return i.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const AC=/^[ \t]*#include +<([\w\d./]+)>/gm;function ff(i){return i.replace(AC,vC)}const SC=new Map;function vC(i,e){let t=Ze[e];if(t===void 0){const n=SC.get(e);if(n!==void 0)t=Ze[n],je('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,n);else throw new Error("Can not resolve #include <"+e+">")}return ff(t)}const yC=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function Ap(i){return i.replace(yC,bC)}function bC(i,e,t,n){let s="";for(let r=parseInt(e);r<parseInt(t);r++)s+=n.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function Sp(i){let e=`precision ${i.precision} float;
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
#define LOW_PRECISION`),e}function MC(i){let e="SHADOWMAP_TYPE_BASIC";return i.shadowMapType===M0?e="SHADOWMAP_TYPE_PCF":i.shadowMapType===qA?e="SHADOWMAP_TYPE_PCF_SOFT":i.shadowMapType===qi&&(e="SHADOWMAP_TYPE_VSM"),e}function CC(i){let e="ENVMAP_TYPE_CUBE";if(i.envMap)switch(i.envMapMode){case lo:case co:e="ENVMAP_TYPE_CUBE";break;case jl:e="ENVMAP_TYPE_CUBE_UV";break}return e}function TC(i){let e="ENVMAP_MODE_REFLECTION";return i.envMap&&i.envMapMode===co&&(e="ENVMAP_MODE_REFRACTION"),e}function EC(i){let e="ENVMAP_BLENDING_NONE";if(i.envMap)switch(i.combine){case T0:e="ENVMAP_BLENDING_MULTIPLY";break;case uS:e="ENVMAP_BLENDING_MIX";break;case fS:e="ENVMAP_BLENDING_ADD";break}return e}function wC(i){const e=i.envMapCubeUVHeight;if(e===null)return null;const t=Math.log2(e)-2,n=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:n,maxMip:t}}function RC(i,e,t,n){const s=i.getContext(),r=t.defines;let o=t.vertexShader,a=t.fragmentShader;const l=MC(t),c=CC(t),u=TC(t),f=EC(t),d=wC(t),h=gC(t),x=xC(r),p=s.createProgram();let g,m,_=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(g=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x].filter(Bo).join(`
`),g.length>0&&(g+=`
`),m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x].filter(Bo).join(`
`),m.length>0&&(m+=`
`)):(g=[Sp(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+u:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(Bo).join(`
`),m=[Sp(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+c:"",t.envMap?"#define "+u:"",t.envMap?"#define "+f:"",d?"#define CUBEUV_TEXEL_WIDTH "+d.texelWidth:"",d?"#define CUBEUV_TEXEL_HEIGHT "+d.texelHeight:"",d?"#define CUBEUV_MAX_MIP "+d.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor||t.batchingColor?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==Ds?"#define TONE_MAPPING":"",t.toneMapping!==Ds?Ze.tonemapping_pars_fragment:"",t.toneMapping!==Ds?pC("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",Ze.colorspace_pars_fragment,hC("linearToOutputTexel",t.outputColorSpace),mC(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(Bo).join(`
`)),o=ff(o),o=xp(o,t),o=_p(o,t),a=ff(a),a=xp(a,t),a=_p(a,t),o=Ap(o),a=Ap(a),t.isRawShaderMaterial!==!0&&(_=`#version 300 es
`,g=[h,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+g,m=["#define varying in",t.glslVersion===Ph?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===Ph?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+m);const A=_+g+o,S=_+m+a,v=pp(s,s.VERTEX_SHADER,A),y=pp(s,s.FRAGMENT_SHADER,S);s.attachShader(p,v),s.attachShader(p,y),t.index0AttributeName!==void 0?s.bindAttribLocation(p,0,t.index0AttributeName):t.morphTargets===!0&&s.bindAttribLocation(p,0,"position"),s.linkProgram(p);function M(I){if(i.debug.checkShaderErrors){const F=s.getProgramInfoLog(p)||"",U=s.getShaderInfoLog(v)||"",O=s.getShaderInfoLog(y)||"",k=F.trim(),z=U.trim(),V=O.trim();let H=!0,$=!0;if(s.getProgramParameter(p,s.LINK_STATUS)===!1)if(H=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(s,p,v,y);else{const oe=gp(s,v,"vertex"),Se=gp(s,y,"fragment");zt("THREE.WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(p,s.VALIDATE_STATUS)+`

Material Name: `+I.name+`
Material Type: `+I.type+`

Program Info Log: `+k+`
`+oe+`
`+Se)}else k!==""?je("WebGLProgram: Program Info Log:",k):(z===""||V==="")&&($=!1);$&&(I.diagnostics={runnable:H,programLog:k,vertexShader:{log:z,prefix:g},fragmentShader:{log:V,prefix:m}})}s.deleteShader(v),s.deleteShader(y),E=new xl(s,p),b=_C(s,p)}let E;this.getUniforms=function(){return E===void 0&&M(this),E};let b;this.getAttributes=function(){return b===void 0&&M(this),b};let C=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return C===!1&&(C=s.getProgramParameter(p,cC)),C},this.destroy=function(){n.releaseStatesOfProgram(this),s.deleteProgram(p),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=uC++,this.cacheKey=e,this.usedTimes=1,this.program=p,this.vertexShader=v,this.fragmentShader=y,this}let IC=0;class DC{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const t=e.vertexShader,n=e.fragmentShader,s=this._getShaderStage(t),r=this._getShaderStage(n),o=this._getShaderCacheForMaterial(e);return o.has(s)===!1&&(o.add(s),s.usedTimes++),o.has(r)===!1&&(o.add(r),r.usedTimes++),this}remove(e){const t=this.materialCache.get(e);for(const n of t)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const t=this.materialCache;let n=t.get(e);return n===void 0&&(n=new Set,t.set(e,n)),n}_getShaderStage(e){const t=this.shaderCache;let n=t.get(e);return n===void 0&&(n=new PC(e),t.set(e,n)),n}}class PC{constructor(e){this.id=IC++,this.code=e,this.usedTimes=0}}function FC(i,e,t,n,s,r,o){const a=new z0,l=new DC,c=new Set,u=[],f=s.logarithmicDepthBuffer,d=s.vertexTextures;let h=s.precision;const x={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function p(b){return c.add(b),b===0?"uv":`uv${b}`}function g(b,C,I,F,U){const O=F.fog,k=U.geometry,z=b.isMeshStandardMaterial?F.environment:null,V=(b.isMeshStandardMaterial?t:e).get(b.envMap||z),H=V&&V.mapping===jl?V.image.height:null,$=x[b.type];b.precision!==null&&(h=s.getMaxPrecision(b.precision),h!==b.precision&&je("WebGLProgram.getParameters:",b.precision,"not supported, using",h,"instead."));const oe=k.morphAttributes.position||k.morphAttributes.normal||k.morphAttributes.color,Se=oe!==void 0?oe.length:0;let we=0;k.morphAttributes.position!==void 0&&(we=1),k.morphAttributes.normal!==void 0&&(we=2),k.morphAttributes.color!==void 0&&(we=3);let Le,fe,re,X;if($){const ut=Ci[$];Le=ut.vertexShader,fe=ut.fragmentShader}else Le=b.vertexShader,fe=b.fragmentShader,l.update(b),re=l.getVertexShaderID(b),X=l.getFragmentShaderID(b);const ee=i.getRenderTarget(),pe=i.state.buffers.depth.getReversed(),be=U.isInstancedMesh===!0,xe=U.isBatchedMesh===!0,Ce=!!b.map,P=!!b.matcap,L=!!V,q=!!b.aoMap,w=!!b.lightMap,te=!!b.bumpMap,ie=!!b.normalMap,ue=!!b.displacementMap,Z=!!b.emissiveMap,de=!!b.metalnessMap,ne=!!b.roughnessMap,ge=b.anisotropy>0,R=b.clearcoat>0,T=b.dispersion>0,G=b.iridescence>0,se=b.sheen>0,le=b.transmission>0,j=ge&&!!b.anisotropyMap,De=R&&!!b.clearcoatMap,_e=R&&!!b.clearcoatNormalMap,Ue=R&&!!b.clearcoatRoughnessMap,N=G&&!!b.iridescenceMap,J=G&&!!b.iridescenceThicknessMap,me=se&&!!b.sheenColorMap,Te=se&&!!b.sheenRoughnessMap,Pe=!!b.specularMap,Re=!!b.specularColorMap,He=!!b.specularIntensityMap,W=le&&!!b.transmissionMap,Ie=le&&!!b.thicknessMap,ve=!!b.gradientMap,ye=!!b.alphaMap,Ae=b.alphaTest>0,he=!!b.alphaHash,Oe=!!b.extensions;let We=Ds;b.toneMapped&&(ee===null||ee.isXRRenderTarget===!0)&&(We=i.toneMapping);const vt={shaderID:$,shaderType:b.type,shaderName:b.name,vertexShader:Le,fragmentShader:fe,defines:b.defines,customVertexShaderID:re,customFragmentShaderID:X,isRawShaderMaterial:b.isRawShaderMaterial===!0,glslVersion:b.glslVersion,precision:h,batching:xe,batchingColor:xe&&U._colorsTexture!==null,instancing:be,instancingColor:be&&U.instanceColor!==null,instancingMorph:be&&U.morphTexture!==null,supportsVertexTextures:d,outputColorSpace:ee===null?i.outputColorSpace:ee.isXRRenderTarget===!0?ee.texture.colorSpace:fo,alphaToCoverage:!!b.alphaToCoverage,map:Ce,matcap:P,envMap:L,envMapMode:L&&V.mapping,envMapCubeUVHeight:H,aoMap:q,lightMap:w,bumpMap:te,normalMap:ie,displacementMap:d&&ue,emissiveMap:Z,normalMapObjectSpace:ie&&b.normalMapType===bS,normalMapTangentSpace:ie&&b.normalMapType===yS,metalnessMap:de,roughnessMap:ne,anisotropy:ge,anisotropyMap:j,clearcoat:R,clearcoatMap:De,clearcoatNormalMap:_e,clearcoatRoughnessMap:Ue,dispersion:T,iridescence:G,iridescenceMap:N,iridescenceThicknessMap:J,sheen:se,sheenColorMap:me,sheenRoughnessMap:Te,specularMap:Pe,specularColorMap:Re,specularIntensityMap:He,transmission:le,transmissionMap:W,thicknessMap:Ie,gradientMap:ve,opaque:b.transparent===!1&&b.blending===Is&&b.alphaToCoverage===!1,alphaMap:ye,alphaTest:Ae,alphaHash:he,combine:b.combine,mapUv:Ce&&p(b.map.channel),aoMapUv:q&&p(b.aoMap.channel),lightMapUv:w&&p(b.lightMap.channel),bumpMapUv:te&&p(b.bumpMap.channel),normalMapUv:ie&&p(b.normalMap.channel),displacementMapUv:ue&&p(b.displacementMap.channel),emissiveMapUv:Z&&p(b.emissiveMap.channel),metalnessMapUv:de&&p(b.metalnessMap.channel),roughnessMapUv:ne&&p(b.roughnessMap.channel),anisotropyMapUv:j&&p(b.anisotropyMap.channel),clearcoatMapUv:De&&p(b.clearcoatMap.channel),clearcoatNormalMapUv:_e&&p(b.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Ue&&p(b.clearcoatRoughnessMap.channel),iridescenceMapUv:N&&p(b.iridescenceMap.channel),iridescenceThicknessMapUv:J&&p(b.iridescenceThicknessMap.channel),sheenColorMapUv:me&&p(b.sheenColorMap.channel),sheenRoughnessMapUv:Te&&p(b.sheenRoughnessMap.channel),specularMapUv:Pe&&p(b.specularMap.channel),specularColorMapUv:Re&&p(b.specularColorMap.channel),specularIntensityMapUv:He&&p(b.specularIntensityMap.channel),transmissionMapUv:W&&p(b.transmissionMap.channel),thicknessMapUv:Ie&&p(b.thicknessMap.channel),alphaMapUv:ye&&p(b.alphaMap.channel),vertexTangents:!!k.attributes.tangent&&(ie||ge),vertexColors:b.vertexColors,vertexAlphas:b.vertexColors===!0&&!!k.attributes.color&&k.attributes.color.itemSize===4,pointsUvs:U.isPoints===!0&&!!k.attributes.uv&&(Ce||ye),fog:!!O,useFog:b.fog===!0,fogExp2:!!O&&O.isFogExp2,flatShading:b.flatShading===!0&&b.wireframe===!1,sizeAttenuation:b.sizeAttenuation===!0,logarithmicDepthBuffer:f,reversedDepthBuffer:pe,skinning:U.isSkinnedMesh===!0,morphTargets:k.morphAttributes.position!==void 0,morphNormals:k.morphAttributes.normal!==void 0,morphColors:k.morphAttributes.color!==void 0,morphTargetsCount:Se,morphTextureStride:we,numDirLights:C.directional.length,numPointLights:C.point.length,numSpotLights:C.spot.length,numSpotLightMaps:C.spotLightMap.length,numRectAreaLights:C.rectArea.length,numHemiLights:C.hemi.length,numDirLightShadows:C.directionalShadowMap.length,numPointLightShadows:C.pointShadowMap.length,numSpotLightShadows:C.spotShadowMap.length,numSpotLightShadowsWithMaps:C.numSpotLightShadowsWithMaps,numLightProbes:C.numLightProbes,numClippingPlanes:o.numPlanes,numClipIntersection:o.numIntersection,dithering:b.dithering,shadowMapEnabled:i.shadowMap.enabled&&I.length>0,shadowMapType:i.shadowMap.type,toneMapping:We,decodeVideoTexture:Ce&&b.map.isVideoTexture===!0&&rt.getTransfer(b.map.colorSpace)===ht,decodeVideoTextureEmissive:Z&&b.emissiveMap.isVideoTexture===!0&&rt.getTransfer(b.emissiveMap.colorSpace)===ht,premultipliedAlpha:b.premultipliedAlpha,doubleSided:b.side===ti,flipSided:b.side===wn,useDepthPacking:b.depthPacking>=0,depthPacking:b.depthPacking||0,index0AttributeName:b.index0AttributeName,extensionClipCullDistance:Oe&&b.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Oe&&b.extensions.multiDraw===!0||xe)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:b.customProgramCacheKey()};return vt.vertexUv1s=c.has(1),vt.vertexUv2s=c.has(2),vt.vertexUv3s=c.has(3),c.clear(),vt}function m(b){const C=[];if(b.shaderID?C.push(b.shaderID):(C.push(b.customVertexShaderID),C.push(b.customFragmentShaderID)),b.defines!==void 0)for(const I in b.defines)C.push(I),C.push(b.defines[I]);return b.isRawShaderMaterial===!1&&(_(C,b),A(C,b),C.push(i.outputColorSpace)),C.push(b.customProgramCacheKey),C.join()}function _(b,C){b.push(C.precision),b.push(C.outputColorSpace),b.push(C.envMapMode),b.push(C.envMapCubeUVHeight),b.push(C.mapUv),b.push(C.alphaMapUv),b.push(C.lightMapUv),b.push(C.aoMapUv),b.push(C.bumpMapUv),b.push(C.normalMapUv),b.push(C.displacementMapUv),b.push(C.emissiveMapUv),b.push(C.metalnessMapUv),b.push(C.roughnessMapUv),b.push(C.anisotropyMapUv),b.push(C.clearcoatMapUv),b.push(C.clearcoatNormalMapUv),b.push(C.clearcoatRoughnessMapUv),b.push(C.iridescenceMapUv),b.push(C.iridescenceThicknessMapUv),b.push(C.sheenColorMapUv),b.push(C.sheenRoughnessMapUv),b.push(C.specularMapUv),b.push(C.specularColorMapUv),b.push(C.specularIntensityMapUv),b.push(C.transmissionMapUv),b.push(C.thicknessMapUv),b.push(C.combine),b.push(C.fogExp2),b.push(C.sizeAttenuation),b.push(C.morphTargetsCount),b.push(C.morphAttributeCount),b.push(C.numDirLights),b.push(C.numPointLights),b.push(C.numSpotLights),b.push(C.numSpotLightMaps),b.push(C.numHemiLights),b.push(C.numRectAreaLights),b.push(C.numDirLightShadows),b.push(C.numPointLightShadows),b.push(C.numSpotLightShadows),b.push(C.numSpotLightShadowsWithMaps),b.push(C.numLightProbes),b.push(C.shadowMapType),b.push(C.toneMapping),b.push(C.numClippingPlanes),b.push(C.numClipIntersection),b.push(C.depthPacking)}function A(b,C){a.disableAll(),C.supportsVertexTextures&&a.enable(0),C.instancing&&a.enable(1),C.instancingColor&&a.enable(2),C.instancingMorph&&a.enable(3),C.matcap&&a.enable(4),C.envMap&&a.enable(5),C.normalMapObjectSpace&&a.enable(6),C.normalMapTangentSpace&&a.enable(7),C.clearcoat&&a.enable(8),C.iridescence&&a.enable(9),C.alphaTest&&a.enable(10),C.vertexColors&&a.enable(11),C.vertexAlphas&&a.enable(12),C.vertexUv1s&&a.enable(13),C.vertexUv2s&&a.enable(14),C.vertexUv3s&&a.enable(15),C.vertexTangents&&a.enable(16),C.anisotropy&&a.enable(17),C.alphaHash&&a.enable(18),C.batching&&a.enable(19),C.dispersion&&a.enable(20),C.batchingColor&&a.enable(21),C.gradientMap&&a.enable(22),b.push(a.mask),a.disableAll(),C.fog&&a.enable(0),C.useFog&&a.enable(1),C.flatShading&&a.enable(2),C.logarithmicDepthBuffer&&a.enable(3),C.reversedDepthBuffer&&a.enable(4),C.skinning&&a.enable(5),C.morphTargets&&a.enable(6),C.morphNormals&&a.enable(7),C.morphColors&&a.enable(8),C.premultipliedAlpha&&a.enable(9),C.shadowMapEnabled&&a.enable(10),C.doubleSided&&a.enable(11),C.flipSided&&a.enable(12),C.useDepthPacking&&a.enable(13),C.dithering&&a.enable(14),C.transmission&&a.enable(15),C.sheen&&a.enable(16),C.opaque&&a.enable(17),C.pointsUvs&&a.enable(18),C.decodeVideoTexture&&a.enable(19),C.decodeVideoTextureEmissive&&a.enable(20),C.alphaToCoverage&&a.enable(21),b.push(a.mask)}function S(b){const C=x[b.type];let I;if(C){const F=Ci[C];I=nv.clone(F.uniforms)}else I=b.uniforms;return I}function v(b,C){let I;for(let F=0,U=u.length;F<U;F++){const O=u[F];if(O.cacheKey===C){I=O,++I.usedTimes;break}}return I===void 0&&(I=new RC(i,C,b,r),u.push(I)),I}function y(b){if(--b.usedTimes===0){const C=u.indexOf(b);u[C]=u[u.length-1],u.pop(),b.destroy()}}function M(b){l.remove(b)}function E(){l.dispose()}return{getParameters:g,getProgramCacheKey:m,getUniforms:S,acquireProgram:v,releaseProgram:y,releaseShaderCache:M,programs:u,dispose:E}}function LC(){let i=new WeakMap;function e(o){return i.has(o)}function t(o){let a=i.get(o);return a===void 0&&(a={},i.set(o,a)),a}function n(o){i.delete(o)}function s(o,a,l){i.get(o)[a]=l}function r(){i=new WeakMap}return{has:e,get:t,remove:n,update:s,dispose:r}}function BC(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.material.id!==e.material.id?i.material.id-e.material.id:i.z!==e.z?i.z-e.z:i.id-e.id}function vp(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.z!==e.z?e.z-i.z:i.id-e.id}function yp(){const i=[];let e=0;const t=[],n=[],s=[];function r(){e=0,t.length=0,n.length=0,s.length=0}function o(f,d,h,x,p,g){let m=i[e];return m===void 0?(m={id:f.id,object:f,geometry:d,material:h,groupOrder:x,renderOrder:f.renderOrder,z:p,group:g},i[e]=m):(m.id=f.id,m.object=f,m.geometry=d,m.material=h,m.groupOrder=x,m.renderOrder=f.renderOrder,m.z=p,m.group=g),e++,m}function a(f,d,h,x,p,g){const m=o(f,d,h,x,p,g);h.transmission>0?n.push(m):h.transparent===!0?s.push(m):t.push(m)}function l(f,d,h,x,p,g){const m=o(f,d,h,x,p,g);h.transmission>0?n.unshift(m):h.transparent===!0?s.unshift(m):t.unshift(m)}function c(f,d){t.length>1&&t.sort(f||BC),n.length>1&&n.sort(d||vp),s.length>1&&s.sort(d||vp)}function u(){for(let f=e,d=i.length;f<d;f++){const h=i[f];if(h.id===null)break;h.id=null,h.object=null,h.geometry=null,h.material=null,h.group=null}}return{opaque:t,transmissive:n,transparent:s,init:r,push:a,unshift:l,finish:u,sort:c}}function UC(){let i=new WeakMap;function e(n,s){const r=i.get(n);let o;return r===void 0?(o=new yp,i.set(n,[o])):s>=r.length?(o=new yp,r.push(o)):o=r[s],o}function t(){i=new WeakMap}return{get:e,dispose:t}}function OC(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new B,color:new nt};break;case"SpotLight":t={position:new B,direction:new B,color:new nt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new B,color:new nt,distance:0,decay:0};break;case"HemisphereLight":t={direction:new B,skyColor:new nt,groundColor:new nt};break;case"RectAreaLight":t={color:new nt,position:new B,halfWidth:new B,halfHeight:new B};break}return i[e.id]=t,t}}}function NC(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new ze};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new ze};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new ze,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[e.id]=t,t}}}let zC=0;function kC(i,e){return(e.castShadow?2:0)-(i.castShadow?2:0)+(e.map?1:0)-(i.map?1:0)}function HC(i){const e=new OC,t=NC(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)n.probe.push(new B);const s=new B,r=new qe,o=new qe;function a(c){let u=0,f=0,d=0;for(let b=0;b<9;b++)n.probe[b].set(0,0,0);let h=0,x=0,p=0,g=0,m=0,_=0,A=0,S=0,v=0,y=0,M=0;c.sort(kC);for(let b=0,C=c.length;b<C;b++){const I=c[b],F=I.color,U=I.intensity,O=I.distance,k=I.shadow&&I.shadow.map?I.shadow.map.texture:null;if(I.isAmbientLight)u+=F.r*U,f+=F.g*U,d+=F.b*U;else if(I.isLightProbe){for(let z=0;z<9;z++)n.probe[z].addScaledVector(I.sh.coefficients[z],U);M++}else if(I.isDirectionalLight){const z=e.get(I);if(z.color.copy(I.color).multiplyScalar(I.intensity),I.castShadow){const V=I.shadow,H=t.get(I);H.shadowIntensity=V.intensity,H.shadowBias=V.bias,H.shadowNormalBias=V.normalBias,H.shadowRadius=V.radius,H.shadowMapSize=V.mapSize,n.directionalShadow[h]=H,n.directionalShadowMap[h]=k,n.directionalShadowMatrix[h]=I.shadow.matrix,_++}n.directional[h]=z,h++}else if(I.isSpotLight){const z=e.get(I);z.position.setFromMatrixPosition(I.matrixWorld),z.color.copy(F).multiplyScalar(U),z.distance=O,z.coneCos=Math.cos(I.angle),z.penumbraCos=Math.cos(I.angle*(1-I.penumbra)),z.decay=I.decay,n.spot[p]=z;const V=I.shadow;if(I.map&&(n.spotLightMap[v]=I.map,v++,V.updateMatrices(I),I.castShadow&&y++),n.spotLightMatrix[p]=V.matrix,I.castShadow){const H=t.get(I);H.shadowIntensity=V.intensity,H.shadowBias=V.bias,H.shadowNormalBias=V.normalBias,H.shadowRadius=V.radius,H.shadowMapSize=V.mapSize,n.spotShadow[p]=H,n.spotShadowMap[p]=k,S++}p++}else if(I.isRectAreaLight){const z=e.get(I);z.color.copy(F).multiplyScalar(U),z.halfWidth.set(I.width*.5,0,0),z.halfHeight.set(0,I.height*.5,0),n.rectArea[g]=z,g++}else if(I.isPointLight){const z=e.get(I);if(z.color.copy(I.color).multiplyScalar(I.intensity),z.distance=I.distance,z.decay=I.decay,I.castShadow){const V=I.shadow,H=t.get(I);H.shadowIntensity=V.intensity,H.shadowBias=V.bias,H.shadowNormalBias=V.normalBias,H.shadowRadius=V.radius,H.shadowMapSize=V.mapSize,H.shadowCameraNear=V.camera.near,H.shadowCameraFar=V.camera.far,n.pointShadow[x]=H,n.pointShadowMap[x]=k,n.pointShadowMatrix[x]=I.shadow.matrix,A++}n.point[x]=z,x++}else if(I.isHemisphereLight){const z=e.get(I);z.skyColor.copy(I.color).multiplyScalar(U),z.groundColor.copy(I.groundColor).multiplyScalar(U),n.hemi[m]=z,m++}}g>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=Fe.LTC_FLOAT_1,n.rectAreaLTC2=Fe.LTC_FLOAT_2):(n.rectAreaLTC1=Fe.LTC_HALF_1,n.rectAreaLTC2=Fe.LTC_HALF_2)),n.ambient[0]=u,n.ambient[1]=f,n.ambient[2]=d;const E=n.hash;(E.directionalLength!==h||E.pointLength!==x||E.spotLength!==p||E.rectAreaLength!==g||E.hemiLength!==m||E.numDirectionalShadows!==_||E.numPointShadows!==A||E.numSpotShadows!==S||E.numSpotMaps!==v||E.numLightProbes!==M)&&(n.directional.length=h,n.spot.length=p,n.rectArea.length=g,n.point.length=x,n.hemi.length=m,n.directionalShadow.length=_,n.directionalShadowMap.length=_,n.pointShadow.length=A,n.pointShadowMap.length=A,n.spotShadow.length=S,n.spotShadowMap.length=S,n.directionalShadowMatrix.length=_,n.pointShadowMatrix.length=A,n.spotLightMatrix.length=S+v-y,n.spotLightMap.length=v,n.numSpotLightShadowsWithMaps=y,n.numLightProbes=M,E.directionalLength=h,E.pointLength=x,E.spotLength=p,E.rectAreaLength=g,E.hemiLength=m,E.numDirectionalShadows=_,E.numPointShadows=A,E.numSpotShadows=S,E.numSpotMaps=v,E.numLightProbes=M,n.version=zC++)}function l(c,u){let f=0,d=0,h=0,x=0,p=0;const g=u.matrixWorldInverse;for(let m=0,_=c.length;m<_;m++){const A=c[m];if(A.isDirectionalLight){const S=n.directional[f];S.direction.setFromMatrixPosition(A.matrixWorld),s.setFromMatrixPosition(A.target.matrixWorld),S.direction.sub(s),S.direction.transformDirection(g),f++}else if(A.isSpotLight){const S=n.spot[h];S.position.setFromMatrixPosition(A.matrixWorld),S.position.applyMatrix4(g),S.direction.setFromMatrixPosition(A.matrixWorld),s.setFromMatrixPosition(A.target.matrixWorld),S.direction.sub(s),S.direction.transformDirection(g),h++}else if(A.isRectAreaLight){const S=n.rectArea[x];S.position.setFromMatrixPosition(A.matrixWorld),S.position.applyMatrix4(g),o.identity(),r.copy(A.matrixWorld),r.premultiply(g),o.extractRotation(r),S.halfWidth.set(A.width*.5,0,0),S.halfHeight.set(0,A.height*.5,0),S.halfWidth.applyMatrix4(o),S.halfHeight.applyMatrix4(o),x++}else if(A.isPointLight){const S=n.point[d];S.position.setFromMatrixPosition(A.matrixWorld),S.position.applyMatrix4(g),d++}else if(A.isHemisphereLight){const S=n.hemi[p];S.direction.setFromMatrixPosition(A.matrixWorld),S.direction.transformDirection(g),p++}}}return{setup:a,setupView:l,state:n}}function bp(i){const e=new HC(i),t=[],n=[];function s(u){c.camera=u,t.length=0,n.length=0}function r(u){t.push(u)}function o(u){n.push(u)}function a(){e.setup(t)}function l(u){e.setupView(t,u)}const c={lightsArray:t,shadowsArray:n,camera:null,lights:e,transmissionRenderTarget:{}};return{init:s,state:c,setupLights:a,setupLightsView:l,pushLight:r,pushShadow:o}}function VC(i){let e=new WeakMap;function t(s,r=0){const o=e.get(s);let a;return o===void 0?(a=new bp(i),e.set(s,[a])):r>=o.length?(a=new bp(i),o.push(a)):a=o[r],a}function n(){e=new WeakMap}return{get:t,dispose:n}}const GC=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,WC=`uniform sampler2D shadow_pass;
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
}`;function XC(i,e,t){let n=new q0;const s=new ze,r=new ze,o=new Et,a=new mv({depthPacking:vS}),l=new gv,c={},u=t.maxTextureSize,f={[Bi]:wn,[wn]:Bi,[ti]:ti},d=new _n({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new ze},radius:{value:4}},vertexShader:GC,fragmentShader:WC}),h=d.clone();h.defines.HORIZONTAL_PASS=1;const x=new An;x.setAttribute("position",new li(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const p=new Vt(x,d),g=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=M0;let m=this.type;this.render=function(y,M,E){if(g.enabled===!1||g.autoUpdate===!1&&g.needsUpdate===!1||y.length===0)return;const b=i.getRenderTarget(),C=i.getActiveCubeFace(),I=i.getActiveMipmapLevel(),F=i.state;F.setBlending(is),F.buffers.depth.getReversed()===!0?F.buffers.color.setClear(0,0,0,0):F.buffers.color.setClear(1,1,1,1),F.buffers.depth.setTest(!0),F.setScissorTest(!1);const U=m!==qi&&this.type===qi,O=m===qi&&this.type!==qi;for(let k=0,z=y.length;k<z;k++){const V=y[k],H=V.shadow;if(H===void 0){je("WebGLShadowMap:",V,"has no shadow.");continue}if(H.autoUpdate===!1&&H.needsUpdate===!1)continue;s.copy(H.mapSize);const $=H.getFrameExtents();if(s.multiply($),r.copy(H.mapSize),(s.x>u||s.y>u)&&(s.x>u&&(r.x=Math.floor(u/$.x),s.x=r.x*$.x,H.mapSize.x=r.x),s.y>u&&(r.y=Math.floor(u/$.y),s.y=r.y*$.y,H.mapSize.y=r.y)),H.map===null||U===!0||O===!0){const Se=this.type!==qi?{minFilter:qn,magFilter:qn}:{};H.map!==null&&H.map.dispose(),H.map=new Us(s.x,s.y,Se),H.map.texture.name=V.name+".shadowMap",H.camera.updateProjectionMatrix()}i.setRenderTarget(H.map),i.clear();const oe=H.getViewportCount();for(let Se=0;Se<oe;Se++){const we=H.getViewport(Se);o.set(r.x*we.x,r.y*we.y,r.x*we.z,r.y*we.w),F.viewport(o),H.updateMatrices(V,Se),n=H.getFrustum(),S(M,E,H.camera,V,this.type)}H.isPointLightShadow!==!0&&this.type===qi&&_(H,E),H.needsUpdate=!1}m=this.type,g.needsUpdate=!1,i.setRenderTarget(b,C,I)};function _(y,M){const E=e.update(p);d.defines.VSM_SAMPLES!==y.blurSamples&&(d.defines.VSM_SAMPLES=y.blurSamples,h.defines.VSM_SAMPLES=y.blurSamples,d.needsUpdate=!0,h.needsUpdate=!0),y.mapPass===null&&(y.mapPass=new Us(s.x,s.y)),d.uniforms.shadow_pass.value=y.map.texture,d.uniforms.resolution.value=y.mapSize,d.uniforms.radius.value=y.radius,i.setRenderTarget(y.mapPass),i.clear(),i.renderBufferDirect(M,null,E,d,p,null),h.uniforms.shadow_pass.value=y.mapPass.texture,h.uniforms.resolution.value=y.mapSize,h.uniforms.radius.value=y.radius,i.setRenderTarget(y.map),i.clear(),i.renderBufferDirect(M,null,E,h,p,null)}function A(y,M,E,b){let C=null;const I=E.isPointLight===!0?y.customDistanceMaterial:y.customDepthMaterial;if(I!==void 0)C=I;else if(C=E.isPointLight===!0?l:a,i.localClippingEnabled&&M.clipShadows===!0&&Array.isArray(M.clippingPlanes)&&M.clippingPlanes.length!==0||M.displacementMap&&M.displacementScale!==0||M.alphaMap&&M.alphaTest>0||M.map&&M.alphaTest>0||M.alphaToCoverage===!0){const F=C.uuid,U=M.uuid;let O=c[F];O===void 0&&(O={},c[F]=O);let k=O[U];k===void 0&&(k=C.clone(),O[U]=k,M.addEventListener("dispose",v)),C=k}if(C.visible=M.visible,C.wireframe=M.wireframe,b===qi?C.side=M.shadowSide!==null?M.shadowSide:M.side:C.side=M.shadowSide!==null?M.shadowSide:f[M.side],C.alphaMap=M.alphaMap,C.alphaTest=M.alphaToCoverage===!0?.5:M.alphaTest,C.map=M.map,C.clipShadows=M.clipShadows,C.clippingPlanes=M.clippingPlanes,C.clipIntersection=M.clipIntersection,C.displacementMap=M.displacementMap,C.displacementScale=M.displacementScale,C.displacementBias=M.displacementBias,C.wireframeLinewidth=M.wireframeLinewidth,C.linewidth=M.linewidth,E.isPointLight===!0&&C.isMeshDistanceMaterial===!0){const F=i.properties.get(C);F.light=E}return C}function S(y,M,E,b,C){if(y.visible===!1)return;if(y.layers.test(M.layers)&&(y.isMesh||y.isLine||y.isPoints)&&(y.castShadow||y.receiveShadow&&C===qi)&&(!y.frustumCulled||n.intersectsObject(y))){y.modelViewMatrix.multiplyMatrices(E.matrixWorldInverse,y.matrixWorld);const U=e.update(y),O=y.material;if(Array.isArray(O)){const k=U.groups;for(let z=0,V=k.length;z<V;z++){const H=k[z],$=O[H.materialIndex];if($&&$.visible){const oe=A(y,$,b,C);y.onBeforeShadow(i,y,M,E,U,oe,H),i.renderBufferDirect(E,null,U,oe,y,H),y.onAfterShadow(i,y,M,E,U,oe,H)}}}else if(O.visible){const k=A(y,O,b,C);y.onBeforeShadow(i,y,M,E,U,k,null),i.renderBufferDirect(E,null,U,k,y,null),y.onAfterShadow(i,y,M,E,U,k,null)}}const F=y.children;for(let U=0,O=F.length;U<O;U++)S(F[U],M,E,b,C)}function v(y){y.target.removeEventListener("dispose",v);for(const E in c){const b=c[E],C=y.target.uuid;C in b&&(b[C].dispose(),delete b[C])}}}const qC={[bu]:Mu,[Cu]:wu,[Tu]:Ru,[ao]:Eu,[Mu]:bu,[wu]:Cu,[Ru]:Tu,[Eu]:ao};function QC(i,e){function t(){let W=!1;const Ie=new Et;let ve=null;const ye=new Et(0,0,0,0);return{setMask:function(Ae){ve!==Ae&&!W&&(i.colorMask(Ae,Ae,Ae,Ae),ve=Ae)},setLocked:function(Ae){W=Ae},setClear:function(Ae,he,Oe,We,vt){vt===!0&&(Ae*=We,he*=We,Oe*=We),Ie.set(Ae,he,Oe,We),ye.equals(Ie)===!1&&(i.clearColor(Ae,he,Oe,We),ye.copy(Ie))},reset:function(){W=!1,ve=null,ye.set(-1,0,0,0)}}}function n(){let W=!1,Ie=!1,ve=null,ye=null,Ae=null;return{setReversed:function(he){if(Ie!==he){const Oe=e.get("EXT_clip_control");he?Oe.clipControlEXT(Oe.LOWER_LEFT_EXT,Oe.ZERO_TO_ONE_EXT):Oe.clipControlEXT(Oe.LOWER_LEFT_EXT,Oe.NEGATIVE_ONE_TO_ONE_EXT),Ie=he;const We=Ae;Ae=null,this.setClear(We)}},getReversed:function(){return Ie},setTest:function(he){he?ee(i.DEPTH_TEST):pe(i.DEPTH_TEST)},setMask:function(he){ve!==he&&!W&&(i.depthMask(he),ve=he)},setFunc:function(he){if(Ie&&(he=qC[he]),ye!==he){switch(he){case bu:i.depthFunc(i.NEVER);break;case Mu:i.depthFunc(i.ALWAYS);break;case Cu:i.depthFunc(i.LESS);break;case ao:i.depthFunc(i.LEQUAL);break;case Tu:i.depthFunc(i.EQUAL);break;case Eu:i.depthFunc(i.GEQUAL);break;case wu:i.depthFunc(i.GREATER);break;case Ru:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}ye=he}},setLocked:function(he){W=he},setClear:function(he){Ae!==he&&(Ie&&(he=1-he),i.clearDepth(he),Ae=he)},reset:function(){W=!1,ve=null,ye=null,Ae=null,Ie=!1}}}function s(){let W=!1,Ie=null,ve=null,ye=null,Ae=null,he=null,Oe=null,We=null,vt=null;return{setTest:function(ut){W||(ut?ee(i.STENCIL_TEST):pe(i.STENCIL_TEST))},setMask:function(ut){Ie!==ut&&!W&&(i.stencilMask(ut),Ie=ut)},setFunc:function(ut,Ai,ci){(ve!==ut||ye!==Ai||Ae!==ci)&&(i.stencilFunc(ut,Ai,ci),ve=ut,ye=Ai,Ae=ci)},setOp:function(ut,Ai,ci){(he!==ut||Oe!==Ai||We!==ci)&&(i.stencilOp(ut,Ai,ci),he=ut,Oe=Ai,We=ci)},setLocked:function(ut){W=ut},setClear:function(ut){vt!==ut&&(i.clearStencil(ut),vt=ut)},reset:function(){W=!1,Ie=null,ve=null,ye=null,Ae=null,he=null,Oe=null,We=null,vt=null}}}const r=new t,o=new n,a=new s,l=new WeakMap,c=new WeakMap;let u={},f={},d=new WeakMap,h=[],x=null,p=!1,g=null,m=null,_=null,A=null,S=null,v=null,y=null,M=new nt(0,0,0),E=0,b=!1,C=null,I=null,F=null,U=null,O=null;const k=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let z=!1,V=0;const H=i.getParameter(i.VERSION);H.indexOf("WebGL")!==-1?(V=parseFloat(/^WebGL (\d)/.exec(H)[1]),z=V>=1):H.indexOf("OpenGL ES")!==-1&&(V=parseFloat(/^OpenGL ES (\d)/.exec(H)[1]),z=V>=2);let $=null,oe={};const Se=i.getParameter(i.SCISSOR_BOX),we=i.getParameter(i.VIEWPORT),Le=new Et().fromArray(Se),fe=new Et().fromArray(we);function re(W,Ie,ve,ye){const Ae=new Uint8Array(4),he=i.createTexture();i.bindTexture(W,he),i.texParameteri(W,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(W,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let Oe=0;Oe<ve;Oe++)W===i.TEXTURE_3D||W===i.TEXTURE_2D_ARRAY?i.texImage3D(Ie,0,i.RGBA,1,1,ye,0,i.RGBA,i.UNSIGNED_BYTE,Ae):i.texImage2D(Ie+Oe,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,Ae);return he}const X={};X[i.TEXTURE_2D]=re(i.TEXTURE_2D,i.TEXTURE_2D,1),X[i.TEXTURE_CUBE_MAP]=re(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),X[i.TEXTURE_2D_ARRAY]=re(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),X[i.TEXTURE_3D]=re(i.TEXTURE_3D,i.TEXTURE_3D,1,1),r.setClear(0,0,0,1),o.setClear(1),a.setClear(0),ee(i.DEPTH_TEST),o.setFunc(ao),te(!1),ie(Th),ee(i.CULL_FACE),q(is);function ee(W){u[W]!==!0&&(i.enable(W),u[W]=!0)}function pe(W){u[W]!==!1&&(i.disable(W),u[W]=!1)}function be(W,Ie){return f[W]!==Ie?(i.bindFramebuffer(W,Ie),f[W]=Ie,W===i.DRAW_FRAMEBUFFER&&(f[i.FRAMEBUFFER]=Ie),W===i.FRAMEBUFFER&&(f[i.DRAW_FRAMEBUFFER]=Ie),!0):!1}function xe(W,Ie){let ve=h,ye=!1;if(W){ve=d.get(Ie),ve===void 0&&(ve=[],d.set(Ie,ve));const Ae=W.textures;if(ve.length!==Ae.length||ve[0]!==i.COLOR_ATTACHMENT0){for(let he=0,Oe=Ae.length;he<Oe;he++)ve[he]=i.COLOR_ATTACHMENT0+he;ve.length=Ae.length,ye=!0}}else ve[0]!==i.BACK&&(ve[0]=i.BACK,ye=!0);ye&&i.drawBuffers(ve)}function Ce(W){return x!==W?(i.useProgram(W),x=W,!0):!1}const P={[nr]:i.FUNC_ADD,[QA]:i.FUNC_SUBTRACT,[YA]:i.FUNC_REVERSE_SUBTRACT};P[KA]=i.MIN,P[jA]=i.MAX;const L={[$A]:i.ZERO,[ZA]:i.ONE,[JA]:i.SRC_COLOR,[sa]:i.SRC_ALPHA,[rS]:i.SRC_ALPHA_SATURATE,[iS]:i.DST_COLOR,[tS]:i.DST_ALPHA,[eS]:i.ONE_MINUS_SRC_COLOR,[ra]:i.ONE_MINUS_SRC_ALPHA,[sS]:i.ONE_MINUS_DST_COLOR,[nS]:i.ONE_MINUS_DST_ALPHA,[oS]:i.CONSTANT_COLOR,[aS]:i.ONE_MINUS_CONSTANT_COLOR,[lS]:i.CONSTANT_ALPHA,[cS]:i.ONE_MINUS_CONSTANT_ALPHA};function q(W,Ie,ve,ye,Ae,he,Oe,We,vt,ut){if(W===is){p===!0&&(pe(i.BLEND),p=!1);return}if(p===!1&&(ee(i.BLEND),p=!0),W!==C0){if(W!==g||ut!==b){if((m!==nr||S!==nr)&&(i.blendEquation(i.FUNC_ADD),m=nr,S=nr),ut)switch(W){case Is:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Eh:i.blendFunc(i.ONE,i.ONE);break;case wh:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case Rh:i.blendFuncSeparate(i.DST_COLOR,i.ONE_MINUS_SRC_ALPHA,i.ZERO,i.ONE);break;default:zt("WebGLState: Invalid blending: ",W);break}else switch(W){case Is:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Eh:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE,i.ONE,i.ONE);break;case wh:zt("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case Rh:zt("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:zt("WebGLState: Invalid blending: ",W);break}_=null,A=null,v=null,y=null,M.set(0,0,0),E=0,g=W,b=ut}return}Ae=Ae||Ie,he=he||ve,Oe=Oe||ye,(Ie!==m||Ae!==S)&&(i.blendEquationSeparate(P[Ie],P[Ae]),m=Ie,S=Ae),(ve!==_||ye!==A||he!==v||Oe!==y)&&(i.blendFuncSeparate(L[ve],L[ye],L[he],L[Oe]),_=ve,A=ye,v=he,y=Oe),(We.equals(M)===!1||vt!==E)&&(i.blendColor(We.r,We.g,We.b,vt),M.copy(We),E=vt),g=W,b=!1}function w(W,Ie){W.side===ti?pe(i.CULL_FACE):ee(i.CULL_FACE);let ve=W.side===wn;Ie&&(ve=!ve),te(ve),W.blending===Is&&W.transparent===!1?q(is):q(W.blending,W.blendEquation,W.blendSrc,W.blendDst,W.blendEquationAlpha,W.blendSrcAlpha,W.blendDstAlpha,W.blendColor,W.blendAlpha,W.premultipliedAlpha),o.setFunc(W.depthFunc),o.setTest(W.depthTest),o.setMask(W.depthWrite),r.setMask(W.colorWrite);const ye=W.stencilWrite;a.setTest(ye),ye&&(a.setMask(W.stencilWriteMask),a.setFunc(W.stencilFunc,W.stencilRef,W.stencilFuncMask),a.setOp(W.stencilFail,W.stencilZFail,W.stencilZPass)),Z(W.polygonOffset,W.polygonOffsetFactor,W.polygonOffsetUnits),W.alphaToCoverage===!0?ee(i.SAMPLE_ALPHA_TO_COVERAGE):pe(i.SAMPLE_ALPHA_TO_COVERAGE)}function te(W){C!==W&&(W?i.frontFace(i.CW):i.frontFace(i.CCW),C=W)}function ie(W){W!==WA?(ee(i.CULL_FACE),W!==I&&(W===Th?i.cullFace(i.BACK):W===XA?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):pe(i.CULL_FACE),I=W}function ue(W){W!==F&&(z&&i.lineWidth(W),F=W)}function Z(W,Ie,ve){W?(ee(i.POLYGON_OFFSET_FILL),(U!==Ie||O!==ve)&&(i.polygonOffset(Ie,ve),U=Ie,O=ve)):pe(i.POLYGON_OFFSET_FILL)}function de(W){W?ee(i.SCISSOR_TEST):pe(i.SCISSOR_TEST)}function ne(W){W===void 0&&(W=i.TEXTURE0+k-1),$!==W&&(i.activeTexture(W),$=W)}function ge(W,Ie,ve){ve===void 0&&($===null?ve=i.TEXTURE0+k-1:ve=$);let ye=oe[ve];ye===void 0&&(ye={type:void 0,texture:void 0},oe[ve]=ye),(ye.type!==W||ye.texture!==Ie)&&($!==ve&&(i.activeTexture(ve),$=ve),i.bindTexture(W,Ie||X[W]),ye.type=W,ye.texture=Ie)}function R(){const W=oe[$];W!==void 0&&W.type!==void 0&&(i.bindTexture(W.type,null),W.type=void 0,W.texture=void 0)}function T(){try{i.compressedTexImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function G(){try{i.compressedTexImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function se(){try{i.texSubImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function le(){try{i.texSubImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function j(){try{i.compressedTexSubImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function De(){try{i.compressedTexSubImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function _e(){try{i.texStorage2D(...arguments)}catch(W){W("WebGLState:",W)}}function Ue(){try{i.texStorage3D(...arguments)}catch(W){W("WebGLState:",W)}}function N(){try{i.texImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function J(){try{i.texImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function me(W){Le.equals(W)===!1&&(i.scissor(W.x,W.y,W.z,W.w),Le.copy(W))}function Te(W){fe.equals(W)===!1&&(i.viewport(W.x,W.y,W.z,W.w),fe.copy(W))}function Pe(W,Ie){let ve=c.get(Ie);ve===void 0&&(ve=new WeakMap,c.set(Ie,ve));let ye=ve.get(W);ye===void 0&&(ye=i.getUniformBlockIndex(Ie,W.name),ve.set(W,ye))}function Re(W,Ie){const ye=c.get(Ie).get(W);l.get(Ie)!==ye&&(i.uniformBlockBinding(Ie,ye,W.__bindingPointIndex),l.set(Ie,ye))}function He(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),o.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),u={},$=null,oe={},f={},d=new WeakMap,h=[],x=null,p=!1,g=null,m=null,_=null,A=null,S=null,v=null,y=null,M=new nt(0,0,0),E=0,b=!1,C=null,I=null,F=null,U=null,O=null,Le.set(0,0,i.canvas.width,i.canvas.height),fe.set(0,0,i.canvas.width,i.canvas.height),r.reset(),o.reset(),a.reset()}return{buffers:{color:r,depth:o,stencil:a},enable:ee,disable:pe,bindFramebuffer:be,drawBuffers:xe,useProgram:Ce,setBlending:q,setMaterial:w,setFlipSided:te,setCullFace:ie,setLineWidth:ue,setPolygonOffset:Z,setScissorTest:de,activeTexture:ne,bindTexture:ge,unbindTexture:R,compressedTexImage2D:T,compressedTexImage3D:G,texImage2D:N,texImage3D:J,updateUBOMapping:Pe,uniformBlockBinding:Re,texStorage2D:_e,texStorage3D:Ue,texSubImage2D:se,texSubImage3D:le,compressedTexSubImage2D:j,compressedTexSubImage3D:De,scissor:me,viewport:Te,reset:He}}function YC(i,e,t,n,s,r,o){const a=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new ze,u=new WeakMap;let f;const d=new WeakMap;let h=!1;try{h=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function x(R,T){return h?new OffscreenCanvas(R,T):Rl("canvas")}function p(R,T,G){let se=1;const le=ge(R);if((le.width>G||le.height>G)&&(se=G/Math.max(le.width,le.height)),se<1)if(typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&R instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&R instanceof ImageBitmap||typeof VideoFrame<"u"&&R instanceof VideoFrame){const j=Math.floor(se*le.width),De=Math.floor(se*le.height);f===void 0&&(f=x(j,De));const _e=T?x(j,De):f;return _e.width=j,_e.height=De,_e.getContext("2d").drawImage(R,0,0,j,De),je("WebGLRenderer: Texture has been resized from ("+le.width+"x"+le.height+") to ("+j+"x"+De+")."),_e}else return"data"in R&&je("WebGLRenderer: Image in DataTexture is too big ("+le.width+"x"+le.height+")."),R;return R}function g(R){return R.generateMipmaps}function m(R){i.generateMipmap(R)}function _(R){return R.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:R.isWebGL3DRenderTarget?i.TEXTURE_3D:R.isWebGLArrayRenderTarget||R.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function A(R,T,G,se,le=!1){if(R!==null){if(i[R]!==void 0)return i[R];je("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+R+"'")}let j=T;if(T===i.RED&&(G===i.FLOAT&&(j=i.R32F),G===i.HALF_FLOAT&&(j=i.R16F),G===i.UNSIGNED_BYTE&&(j=i.R8)),T===i.RED_INTEGER&&(G===i.UNSIGNED_BYTE&&(j=i.R8UI),G===i.UNSIGNED_SHORT&&(j=i.R16UI),G===i.UNSIGNED_INT&&(j=i.R32UI),G===i.BYTE&&(j=i.R8I),G===i.SHORT&&(j=i.R16I),G===i.INT&&(j=i.R32I)),T===i.RG&&(G===i.FLOAT&&(j=i.RG32F),G===i.HALF_FLOAT&&(j=i.RG16F),G===i.UNSIGNED_BYTE&&(j=i.RG8)),T===i.RG_INTEGER&&(G===i.UNSIGNED_BYTE&&(j=i.RG8UI),G===i.UNSIGNED_SHORT&&(j=i.RG16UI),G===i.UNSIGNED_INT&&(j=i.RG32UI),G===i.BYTE&&(j=i.RG8I),G===i.SHORT&&(j=i.RG16I),G===i.INT&&(j=i.RG32I)),T===i.RGB_INTEGER&&(G===i.UNSIGNED_BYTE&&(j=i.RGB8UI),G===i.UNSIGNED_SHORT&&(j=i.RGB16UI),G===i.UNSIGNED_INT&&(j=i.RGB32UI),G===i.BYTE&&(j=i.RGB8I),G===i.SHORT&&(j=i.RGB16I),G===i.INT&&(j=i.RGB32I)),T===i.RGBA_INTEGER&&(G===i.UNSIGNED_BYTE&&(j=i.RGBA8UI),G===i.UNSIGNED_SHORT&&(j=i.RGBA16UI),G===i.UNSIGNED_INT&&(j=i.RGBA32UI),G===i.BYTE&&(j=i.RGBA8I),G===i.SHORT&&(j=i.RGBA16I),G===i.INT&&(j=i.RGBA32I)),T===i.RGB&&(G===i.UNSIGNED_INT_5_9_9_9_REV&&(j=i.RGB9_E5),G===i.UNSIGNED_INT_10F_11F_11F_REV&&(j=i.R11F_G11F_B10F)),T===i.RGBA){const De=le?El:rt.getTransfer(se);G===i.FLOAT&&(j=i.RGBA32F),G===i.HALF_FLOAT&&(j=i.RGBA16F),G===i.UNSIGNED_BYTE&&(j=De===ht?i.SRGB8_ALPHA8:i.RGBA8),G===i.UNSIGNED_SHORT_4_4_4_4&&(j=i.RGBA4),G===i.UNSIGNED_SHORT_5_5_5_1&&(j=i.RGB5_A1)}return(j===i.R16F||j===i.R32F||j===i.RG16F||j===i.RG32F||j===i.RGBA16F||j===i.RGBA32F)&&e.get("EXT_color_buffer_float"),j}function S(R,T){let G;return R?T===null||T===si||T===aa?G=i.DEPTH24_STENCIL8:T===mi?G=i.DEPTH32F_STENCIL8:T===oa&&(G=i.DEPTH24_STENCIL8,je("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):T===null||T===si||T===aa?G=i.DEPTH_COMPONENT24:T===mi?G=i.DEPTH_COMPONENT32F:T===oa&&(G=i.DEPTH_COMPONENT16),G}function v(R,T){return g(R)===!0||R.isFramebufferTexture&&R.minFilter!==qn&&R.minFilter!==ii?Math.log2(Math.max(T.width,T.height))+1:R.mipmaps!==void 0&&R.mipmaps.length>0?R.mipmaps.length:R.isCompressedTexture&&Array.isArray(R.image)?T.mipmaps.length:1}function y(R){const T=R.target;T.removeEventListener("dispose",y),E(T),T.isVideoTexture&&u.delete(T)}function M(R){const T=R.target;T.removeEventListener("dispose",M),C(T)}function E(R){const T=n.get(R);if(T.__webglInit===void 0)return;const G=R.source,se=d.get(G);if(se){const le=se[T.__cacheKey];le.usedTimes--,le.usedTimes===0&&b(R),Object.keys(se).length===0&&d.delete(G)}n.remove(R)}function b(R){const T=n.get(R);i.deleteTexture(T.__webglTexture);const G=R.source,se=d.get(G);delete se[T.__cacheKey],o.memory.textures--}function C(R){const T=n.get(R);if(R.depthTexture&&(R.depthTexture.dispose(),n.remove(R.depthTexture)),R.isWebGLCubeRenderTarget)for(let se=0;se<6;se++){if(Array.isArray(T.__webglFramebuffer[se]))for(let le=0;le<T.__webglFramebuffer[se].length;le++)i.deleteFramebuffer(T.__webglFramebuffer[se][le]);else i.deleteFramebuffer(T.__webglFramebuffer[se]);T.__webglDepthbuffer&&i.deleteRenderbuffer(T.__webglDepthbuffer[se])}else{if(Array.isArray(T.__webglFramebuffer))for(let se=0;se<T.__webglFramebuffer.length;se++)i.deleteFramebuffer(T.__webglFramebuffer[se]);else i.deleteFramebuffer(T.__webglFramebuffer);if(T.__webglDepthbuffer&&i.deleteRenderbuffer(T.__webglDepthbuffer),T.__webglMultisampledFramebuffer&&i.deleteFramebuffer(T.__webglMultisampledFramebuffer),T.__webglColorRenderbuffer)for(let se=0;se<T.__webglColorRenderbuffer.length;se++)T.__webglColorRenderbuffer[se]&&i.deleteRenderbuffer(T.__webglColorRenderbuffer[se]);T.__webglDepthRenderbuffer&&i.deleteRenderbuffer(T.__webglDepthRenderbuffer)}const G=R.textures;for(let se=0,le=G.length;se<le;se++){const j=n.get(G[se]);j.__webglTexture&&(i.deleteTexture(j.__webglTexture),o.memory.textures--),n.remove(G[se])}n.remove(R)}let I=0;function F(){I=0}function U(){const R=I;return R>=s.maxTextures&&je("WebGLTextures: Trying to use "+R+" texture units while this GPU supports only "+s.maxTextures),I+=1,R}function O(R){const T=[];return T.push(R.wrapS),T.push(R.wrapT),T.push(R.wrapR||0),T.push(R.magFilter),T.push(R.minFilter),T.push(R.anisotropy),T.push(R.internalFormat),T.push(R.format),T.push(R.type),T.push(R.generateMipmaps),T.push(R.premultiplyAlpha),T.push(R.flipY),T.push(R.unpackAlignment),T.push(R.colorSpace),T.join()}function k(R,T){const G=n.get(R);if(R.isVideoTexture&&de(R),R.isRenderTargetTexture===!1&&R.isExternalTexture!==!0&&R.version>0&&G.__version!==R.version){const se=R.image;if(se===null)je("WebGLRenderer: Texture marked for update but no image data found.");else if(se.complete===!1)je("WebGLRenderer: Texture marked for update but image is incomplete");else{X(G,R,T);return}}else R.isExternalTexture&&(G.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(i.TEXTURE_2D,G.__webglTexture,i.TEXTURE0+T)}function z(R,T){const G=n.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&G.__version!==R.version){X(G,R,T);return}else R.isExternalTexture&&(G.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(i.TEXTURE_2D_ARRAY,G.__webglTexture,i.TEXTURE0+T)}function V(R,T){const G=n.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&G.__version!==R.version){X(G,R,T);return}t.bindTexture(i.TEXTURE_3D,G.__webglTexture,i.TEXTURE0+T)}function H(R,T){const G=n.get(R);if(R.version>0&&G.__version!==R.version){ee(G,R,T);return}t.bindTexture(i.TEXTURE_CUBE_MAP,G.__webglTexture,i.TEXTURE0+T)}const $={[Pu]:i.REPEAT,[ns]:i.CLAMP_TO_EDGE,[Fu]:i.MIRRORED_REPEAT},oe={[qn]:i.NEAREST,[AS]:i.NEAREST_MIPMAP_NEAREST,[Da]:i.NEAREST_MIPMAP_LINEAR,[ii]:i.LINEAR,[Ac]:i.LINEAR_MIPMAP_NEAREST,[sr]:i.LINEAR_MIPMAP_LINEAR},Se={[MS]:i.NEVER,[IS]:i.ALWAYS,[CS]:i.LESS,[B0]:i.LEQUAL,[TS]:i.EQUAL,[RS]:i.GEQUAL,[ES]:i.GREATER,[wS]:i.NOTEQUAL};function we(R,T){if(T.type===mi&&e.has("OES_texture_float_linear")===!1&&(T.magFilter===ii||T.magFilter===Ac||T.magFilter===Da||T.magFilter===sr||T.minFilter===ii||T.minFilter===Ac||T.minFilter===Da||T.minFilter===sr)&&je("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(R,i.TEXTURE_WRAP_S,$[T.wrapS]),i.texParameteri(R,i.TEXTURE_WRAP_T,$[T.wrapT]),(R===i.TEXTURE_3D||R===i.TEXTURE_2D_ARRAY)&&i.texParameteri(R,i.TEXTURE_WRAP_R,$[T.wrapR]),i.texParameteri(R,i.TEXTURE_MAG_FILTER,oe[T.magFilter]),i.texParameteri(R,i.TEXTURE_MIN_FILTER,oe[T.minFilter]),T.compareFunction&&(i.texParameteri(R,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(R,i.TEXTURE_COMPARE_FUNC,Se[T.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(T.magFilter===qn||T.minFilter!==Da&&T.minFilter!==sr||T.type===mi&&e.has("OES_texture_float_linear")===!1)return;if(T.anisotropy>1||n.get(T).__currentAnisotropy){const G=e.get("EXT_texture_filter_anisotropic");i.texParameterf(R,G.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(T.anisotropy,s.getMaxAnisotropy())),n.get(T).__currentAnisotropy=T.anisotropy}}}function Le(R,T){let G=!1;R.__webglInit===void 0&&(R.__webglInit=!0,T.addEventListener("dispose",y));const se=T.source;let le=d.get(se);le===void 0&&(le={},d.set(se,le));const j=O(T);if(j!==R.__cacheKey){le[j]===void 0&&(le[j]={texture:i.createTexture(),usedTimes:0},o.memory.textures++,G=!0),le[j].usedTimes++;const De=le[R.__cacheKey];De!==void 0&&(le[R.__cacheKey].usedTimes--,De.usedTimes===0&&b(T)),R.__cacheKey=j,R.__webglTexture=le[j].texture}return G}function fe(R,T,G){return Math.floor(Math.floor(R/G)/T)}function re(R,T,G,se){const j=R.updateRanges;if(j.length===0)t.texSubImage2D(i.TEXTURE_2D,0,0,0,T.width,T.height,G,se,T.data);else{j.sort((J,me)=>J.start-me.start);let De=0;for(let J=1;J<j.length;J++){const me=j[De],Te=j[J],Pe=me.start+me.count,Re=fe(Te.start,T.width,4),He=fe(me.start,T.width,4);Te.start<=Pe+1&&Re===He&&fe(Te.start+Te.count-1,T.width,4)===Re?me.count=Math.max(me.count,Te.start+Te.count-me.start):(++De,j[De]=Te)}j.length=De+1;const _e=i.getParameter(i.UNPACK_ROW_LENGTH),Ue=i.getParameter(i.UNPACK_SKIP_PIXELS),N=i.getParameter(i.UNPACK_SKIP_ROWS);i.pixelStorei(i.UNPACK_ROW_LENGTH,T.width);for(let J=0,me=j.length;J<me;J++){const Te=j[J],Pe=Math.floor(Te.start/4),Re=Math.ceil(Te.count/4),He=Pe%T.width,W=Math.floor(Pe/T.width),Ie=Re,ve=1;i.pixelStorei(i.UNPACK_SKIP_PIXELS,He),i.pixelStorei(i.UNPACK_SKIP_ROWS,W),t.texSubImage2D(i.TEXTURE_2D,0,He,W,Ie,ve,G,se,T.data)}R.clearUpdateRanges(),i.pixelStorei(i.UNPACK_ROW_LENGTH,_e),i.pixelStorei(i.UNPACK_SKIP_PIXELS,Ue),i.pixelStorei(i.UNPACK_SKIP_ROWS,N)}}function X(R,T,G){let se=i.TEXTURE_2D;(T.isDataArrayTexture||T.isCompressedArrayTexture)&&(se=i.TEXTURE_2D_ARRAY),T.isData3DTexture&&(se=i.TEXTURE_3D);const le=Le(R,T),j=T.source;t.bindTexture(se,R.__webglTexture,i.TEXTURE0+G);const De=n.get(j);if(j.version!==De.__version||le===!0){t.activeTexture(i.TEXTURE0+G);const _e=rt.getPrimaries(rt.workingColorSpace),Ue=T.colorSpace===bs?null:rt.getPrimaries(T.colorSpace),N=T.colorSpace===bs||_e===Ue?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,T.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,T.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,N);let J=p(T.image,!1,s.maxTextureSize);J=ne(T,J);const me=r.convert(T.format,T.colorSpace),Te=r.convert(T.type);let Pe=A(T.internalFormat,me,Te,T.colorSpace,T.isVideoTexture);we(se,T);let Re;const He=T.mipmaps,W=T.isVideoTexture!==!0,Ie=De.__version===void 0||le===!0,ve=j.dataReady,ye=v(T,J);if(T.isDepthTexture)Pe=S(T.format===la,T.type),Ie&&(W?t.texStorage2D(i.TEXTURE_2D,1,Pe,J.width,J.height):t.texImage2D(i.TEXTURE_2D,0,Pe,J.width,J.height,0,me,Te,null));else if(T.isDataTexture)if(He.length>0){W&&Ie&&t.texStorage2D(i.TEXTURE_2D,ye,Pe,He[0].width,He[0].height);for(let Ae=0,he=He.length;Ae<he;Ae++)Re=He[Ae],W?ve&&t.texSubImage2D(i.TEXTURE_2D,Ae,0,0,Re.width,Re.height,me,Te,Re.data):t.texImage2D(i.TEXTURE_2D,Ae,Pe,Re.width,Re.height,0,me,Te,Re.data);T.generateMipmaps=!1}else W?(Ie&&t.texStorage2D(i.TEXTURE_2D,ye,Pe,J.width,J.height),ve&&re(T,J,me,Te)):t.texImage2D(i.TEXTURE_2D,0,Pe,J.width,J.height,0,me,Te,J.data);else if(T.isCompressedTexture)if(T.isCompressedArrayTexture){W&&Ie&&t.texStorage3D(i.TEXTURE_2D_ARRAY,ye,Pe,He[0].width,He[0].height,J.depth);for(let Ae=0,he=He.length;Ae<he;Ae++)if(Re=He[Ae],T.format!==gn)if(me!==null)if(W){if(ve)if(T.layerUpdates.size>0){const Oe=ep(Re.width,Re.height,T.format,T.type);for(const We of T.layerUpdates){const vt=Re.data.subarray(We*Oe/Re.data.BYTES_PER_ELEMENT,(We+1)*Oe/Re.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,Ae,0,0,We,Re.width,Re.height,1,me,vt)}T.clearLayerUpdates()}else t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,Ae,0,0,0,Re.width,Re.height,J.depth,me,Re.data)}else t.compressedTexImage3D(i.TEXTURE_2D_ARRAY,Ae,Pe,Re.width,Re.height,J.depth,0,Re.data,0,0);else je("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else W?ve&&t.texSubImage3D(i.TEXTURE_2D_ARRAY,Ae,0,0,0,Re.width,Re.height,J.depth,me,Te,Re.data):t.texImage3D(i.TEXTURE_2D_ARRAY,Ae,Pe,Re.width,Re.height,J.depth,0,me,Te,Re.data)}else{W&&Ie&&t.texStorage2D(i.TEXTURE_2D,ye,Pe,He[0].width,He[0].height);for(let Ae=0,he=He.length;Ae<he;Ae++)Re=He[Ae],T.format!==gn?me!==null?W?ve&&t.compressedTexSubImage2D(i.TEXTURE_2D,Ae,0,0,Re.width,Re.height,me,Re.data):t.compressedTexImage2D(i.TEXTURE_2D,Ae,Pe,Re.width,Re.height,0,Re.data):je("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):W?ve&&t.texSubImage2D(i.TEXTURE_2D,Ae,0,0,Re.width,Re.height,me,Te,Re.data):t.texImage2D(i.TEXTURE_2D,Ae,Pe,Re.width,Re.height,0,me,Te,Re.data)}else if(T.isDataArrayTexture)if(W){if(Ie&&t.texStorage3D(i.TEXTURE_2D_ARRAY,ye,Pe,J.width,J.height,J.depth),ve)if(T.layerUpdates.size>0){const Ae=ep(J.width,J.height,T.format,T.type);for(const he of T.layerUpdates){const Oe=J.data.subarray(he*Ae/J.data.BYTES_PER_ELEMENT,(he+1)*Ae/J.data.BYTES_PER_ELEMENT);t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,he,J.width,J.height,1,me,Te,Oe)}T.clearLayerUpdates()}else t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,J.width,J.height,J.depth,me,Te,J.data)}else t.texImage3D(i.TEXTURE_2D_ARRAY,0,Pe,J.width,J.height,J.depth,0,me,Te,J.data);else if(T.isData3DTexture)W?(Ie&&t.texStorage3D(i.TEXTURE_3D,ye,Pe,J.width,J.height,J.depth),ve&&t.texSubImage3D(i.TEXTURE_3D,0,0,0,0,J.width,J.height,J.depth,me,Te,J.data)):t.texImage3D(i.TEXTURE_3D,0,Pe,J.width,J.height,J.depth,0,me,Te,J.data);else if(T.isFramebufferTexture){if(Ie)if(W)t.texStorage2D(i.TEXTURE_2D,ye,Pe,J.width,J.height);else{let Ae=J.width,he=J.height;for(let Oe=0;Oe<ye;Oe++)t.texImage2D(i.TEXTURE_2D,Oe,Pe,Ae,he,0,me,Te,null),Ae>>=1,he>>=1}}else if(He.length>0){if(W&&Ie){const Ae=ge(He[0]);t.texStorage2D(i.TEXTURE_2D,ye,Pe,Ae.width,Ae.height)}for(let Ae=0,he=He.length;Ae<he;Ae++)Re=He[Ae],W?ve&&t.texSubImage2D(i.TEXTURE_2D,Ae,0,0,me,Te,Re):t.texImage2D(i.TEXTURE_2D,Ae,Pe,me,Te,Re);T.generateMipmaps=!1}else if(W){if(Ie){const Ae=ge(J);t.texStorage2D(i.TEXTURE_2D,ye,Pe,Ae.width,Ae.height)}ve&&t.texSubImage2D(i.TEXTURE_2D,0,0,0,me,Te,J)}else t.texImage2D(i.TEXTURE_2D,0,Pe,me,Te,J);g(T)&&m(se),De.__version=j.version,T.onUpdate&&T.onUpdate(T)}R.__version=T.version}function ee(R,T,G){if(T.image.length!==6)return;const se=Le(R,T),le=T.source;t.bindTexture(i.TEXTURE_CUBE_MAP,R.__webglTexture,i.TEXTURE0+G);const j=n.get(le);if(le.version!==j.__version||se===!0){t.activeTexture(i.TEXTURE0+G);const De=rt.getPrimaries(rt.workingColorSpace),_e=T.colorSpace===bs?null:rt.getPrimaries(T.colorSpace),Ue=T.colorSpace===bs||De===_e?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,T.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,T.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,Ue);const N=T.isCompressedTexture||T.image[0].isCompressedTexture,J=T.image[0]&&T.image[0].isDataTexture,me=[];for(let he=0;he<6;he++)!N&&!J?me[he]=p(T.image[he],!0,s.maxCubemapSize):me[he]=J?T.image[he].image:T.image[he],me[he]=ne(T,me[he]);const Te=me[0],Pe=r.convert(T.format,T.colorSpace),Re=r.convert(T.type),He=A(T.internalFormat,Pe,Re,T.colorSpace),W=T.isVideoTexture!==!0,Ie=j.__version===void 0||se===!0,ve=le.dataReady;let ye=v(T,Te);we(i.TEXTURE_CUBE_MAP,T);let Ae;if(N){W&&Ie&&t.texStorage2D(i.TEXTURE_CUBE_MAP,ye,He,Te.width,Te.height);for(let he=0;he<6;he++){Ae=me[he].mipmaps;for(let Oe=0;Oe<Ae.length;Oe++){const We=Ae[Oe];T.format!==gn?Pe!==null?W?ve&&t.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,Oe,0,0,We.width,We.height,Pe,We.data):t.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,Oe,He,We.width,We.height,0,We.data):je("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):W?ve&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,Oe,0,0,We.width,We.height,Pe,Re,We.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,Oe,He,We.width,We.height,0,Pe,Re,We.data)}}}else{if(Ae=T.mipmaps,W&&Ie){Ae.length>0&&ye++;const he=ge(me[0]);t.texStorage2D(i.TEXTURE_CUBE_MAP,ye,He,he.width,he.height)}for(let he=0;he<6;he++)if(J){W?ve&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,0,0,0,me[he].width,me[he].height,Pe,Re,me[he].data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,0,He,me[he].width,me[he].height,0,Pe,Re,me[he].data);for(let Oe=0;Oe<Ae.length;Oe++){const vt=Ae[Oe].image[he].image;W?ve&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,Oe+1,0,0,vt.width,vt.height,Pe,Re,vt.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,Oe+1,He,vt.width,vt.height,0,Pe,Re,vt.data)}}else{W?ve&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,0,0,0,Pe,Re,me[he]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,0,He,Pe,Re,me[he]);for(let Oe=0;Oe<Ae.length;Oe++){const We=Ae[Oe];W?ve&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,Oe+1,0,0,Pe,Re,We.image[he]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+he,Oe+1,He,Pe,Re,We.image[he])}}}g(T)&&m(i.TEXTURE_CUBE_MAP),j.__version=le.version,T.onUpdate&&T.onUpdate(T)}R.__version=T.version}function pe(R,T,G,se,le,j){const De=r.convert(G.format,G.colorSpace),_e=r.convert(G.type),Ue=A(G.internalFormat,De,_e,G.colorSpace),N=n.get(T),J=n.get(G);if(J.__renderTarget=T,!N.__hasExternalTextures){const me=Math.max(1,T.width>>j),Te=Math.max(1,T.height>>j);le===i.TEXTURE_3D||le===i.TEXTURE_2D_ARRAY?t.texImage3D(le,j,Ue,me,Te,T.depth,0,De,_e,null):t.texImage2D(le,j,Ue,me,Te,0,De,_e,null)}t.bindFramebuffer(i.FRAMEBUFFER,R),Z(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,se,le,J.__webglTexture,0,ue(T)):(le===i.TEXTURE_2D||le>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&le<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,se,le,J.__webglTexture,j),t.bindFramebuffer(i.FRAMEBUFFER,null)}function be(R,T,G){if(i.bindRenderbuffer(i.RENDERBUFFER,R),T.depthBuffer){const se=T.depthTexture,le=se&&se.isDepthTexture?se.type:null,j=S(T.stencilBuffer,le),De=T.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,_e=ue(T);Z(T)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,_e,j,T.width,T.height):G?i.renderbufferStorageMultisample(i.RENDERBUFFER,_e,j,T.width,T.height):i.renderbufferStorage(i.RENDERBUFFER,j,T.width,T.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,De,i.RENDERBUFFER,R)}else{const se=T.textures;for(let le=0;le<se.length;le++){const j=se[le],De=r.convert(j.format,j.colorSpace),_e=r.convert(j.type),Ue=A(j.internalFormat,De,_e,j.colorSpace),N=ue(T);G&&Z(T)===!1?i.renderbufferStorageMultisample(i.RENDERBUFFER,N,Ue,T.width,T.height):Z(T)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,N,Ue,T.width,T.height):i.renderbufferStorage(i.RENDERBUFFER,Ue,T.width,T.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function xe(R,T){if(T&&T.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(t.bindFramebuffer(i.FRAMEBUFFER,R),!(T.depthTexture&&T.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const se=n.get(T.depthTexture);se.__renderTarget=T,(!se.__webglTexture||T.depthTexture.image.width!==T.width||T.depthTexture.image.height!==T.height)&&(T.depthTexture.image.width=T.width,T.depthTexture.image.height=T.height,T.depthTexture.needsUpdate=!0),k(T.depthTexture,0);const le=se.__webglTexture,j=ue(T);if(T.depthTexture.format===uo)Z(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,le,0,j):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,le,0);else if(T.depthTexture.format===la)Z(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,le,0,j):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,le,0);else throw new Error("Unknown depthTexture format")}function Ce(R){const T=n.get(R),G=R.isWebGLCubeRenderTarget===!0;if(T.__boundDepthTexture!==R.depthTexture){const se=R.depthTexture;if(T.__depthDisposeCallback&&T.__depthDisposeCallback(),se){const le=()=>{delete T.__boundDepthTexture,delete T.__depthDisposeCallback,se.removeEventListener("dispose",le)};se.addEventListener("dispose",le),T.__depthDisposeCallback=le}T.__boundDepthTexture=se}if(R.depthTexture&&!T.__autoAllocateDepthBuffer){if(G)throw new Error("target.depthTexture not supported in Cube render targets");const se=R.texture.mipmaps;se&&se.length>0?xe(T.__webglFramebuffer[0],R):xe(T.__webglFramebuffer,R)}else if(G){T.__webglDepthbuffer=[];for(let se=0;se<6;se++)if(t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer[se]),T.__webglDepthbuffer[se]===void 0)T.__webglDepthbuffer[se]=i.createRenderbuffer(),be(T.__webglDepthbuffer[se],R,!1);else{const le=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,j=T.__webglDepthbuffer[se];i.bindRenderbuffer(i.RENDERBUFFER,j),i.framebufferRenderbuffer(i.FRAMEBUFFER,le,i.RENDERBUFFER,j)}}else{const se=R.texture.mipmaps;if(se&&se.length>0?t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer[0]):t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer),T.__webglDepthbuffer===void 0)T.__webglDepthbuffer=i.createRenderbuffer(),be(T.__webglDepthbuffer,R,!1);else{const le=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,j=T.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,j),i.framebufferRenderbuffer(i.FRAMEBUFFER,le,i.RENDERBUFFER,j)}}t.bindFramebuffer(i.FRAMEBUFFER,null)}function P(R,T,G){const se=n.get(R);T!==void 0&&pe(se.__webglFramebuffer,R,R.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),G!==void 0&&Ce(R)}function L(R){const T=R.texture,G=n.get(R),se=n.get(T);R.addEventListener("dispose",M);const le=R.textures,j=R.isWebGLCubeRenderTarget===!0,De=le.length>1;if(De||(se.__webglTexture===void 0&&(se.__webglTexture=i.createTexture()),se.__version=T.version,o.memory.textures++),j){G.__webglFramebuffer=[];for(let _e=0;_e<6;_e++)if(T.mipmaps&&T.mipmaps.length>0){G.__webglFramebuffer[_e]=[];for(let Ue=0;Ue<T.mipmaps.length;Ue++)G.__webglFramebuffer[_e][Ue]=i.createFramebuffer()}else G.__webglFramebuffer[_e]=i.createFramebuffer()}else{if(T.mipmaps&&T.mipmaps.length>0){G.__webglFramebuffer=[];for(let _e=0;_e<T.mipmaps.length;_e++)G.__webglFramebuffer[_e]=i.createFramebuffer()}else G.__webglFramebuffer=i.createFramebuffer();if(De)for(let _e=0,Ue=le.length;_e<Ue;_e++){const N=n.get(le[_e]);N.__webglTexture===void 0&&(N.__webglTexture=i.createTexture(),o.memory.textures++)}if(R.samples>0&&Z(R)===!1){G.__webglMultisampledFramebuffer=i.createFramebuffer(),G.__webglColorRenderbuffer=[],t.bindFramebuffer(i.FRAMEBUFFER,G.__webglMultisampledFramebuffer);for(let _e=0;_e<le.length;_e++){const Ue=le[_e];G.__webglColorRenderbuffer[_e]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,G.__webglColorRenderbuffer[_e]);const N=r.convert(Ue.format,Ue.colorSpace),J=r.convert(Ue.type),me=A(Ue.internalFormat,N,J,Ue.colorSpace,R.isXRRenderTarget===!0),Te=ue(R);i.renderbufferStorageMultisample(i.RENDERBUFFER,Te,me,R.width,R.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+_e,i.RENDERBUFFER,G.__webglColorRenderbuffer[_e])}i.bindRenderbuffer(i.RENDERBUFFER,null),R.depthBuffer&&(G.__webglDepthRenderbuffer=i.createRenderbuffer(),be(G.__webglDepthRenderbuffer,R,!0)),t.bindFramebuffer(i.FRAMEBUFFER,null)}}if(j){t.bindTexture(i.TEXTURE_CUBE_MAP,se.__webglTexture),we(i.TEXTURE_CUBE_MAP,T);for(let _e=0;_e<6;_e++)if(T.mipmaps&&T.mipmaps.length>0)for(let Ue=0;Ue<T.mipmaps.length;Ue++)pe(G.__webglFramebuffer[_e][Ue],R,T,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+_e,Ue);else pe(G.__webglFramebuffer[_e],R,T,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+_e,0);g(T)&&m(i.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(De){for(let _e=0,Ue=le.length;_e<Ue;_e++){const N=le[_e],J=n.get(N);let me=i.TEXTURE_2D;(R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(me=R.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(me,J.__webglTexture),we(me,N),pe(G.__webglFramebuffer,R,N,i.COLOR_ATTACHMENT0+_e,me,0),g(N)&&m(me)}t.unbindTexture()}else{let _e=i.TEXTURE_2D;if((R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(_e=R.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(_e,se.__webglTexture),we(_e,T),T.mipmaps&&T.mipmaps.length>0)for(let Ue=0;Ue<T.mipmaps.length;Ue++)pe(G.__webglFramebuffer[Ue],R,T,i.COLOR_ATTACHMENT0,_e,Ue);else pe(G.__webglFramebuffer,R,T,i.COLOR_ATTACHMENT0,_e,0);g(T)&&m(_e),t.unbindTexture()}R.depthBuffer&&Ce(R)}function q(R){const T=R.textures;for(let G=0,se=T.length;G<se;G++){const le=T[G];if(g(le)){const j=_(R),De=n.get(le).__webglTexture;t.bindTexture(j,De),m(j),t.unbindTexture()}}}const w=[],te=[];function ie(R){if(R.samples>0){if(Z(R)===!1){const T=R.textures,G=R.width,se=R.height;let le=i.COLOR_BUFFER_BIT;const j=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,De=n.get(R),_e=T.length>1;if(_e)for(let N=0;N<T.length;N++)t.bindFramebuffer(i.FRAMEBUFFER,De.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+N,i.RENDERBUFFER,null),t.bindFramebuffer(i.FRAMEBUFFER,De.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+N,i.TEXTURE_2D,null,0);t.bindFramebuffer(i.READ_FRAMEBUFFER,De.__webglMultisampledFramebuffer);const Ue=R.texture.mipmaps;Ue&&Ue.length>0?t.bindFramebuffer(i.DRAW_FRAMEBUFFER,De.__webglFramebuffer[0]):t.bindFramebuffer(i.DRAW_FRAMEBUFFER,De.__webglFramebuffer);for(let N=0;N<T.length;N++){if(R.resolveDepthBuffer&&(R.depthBuffer&&(le|=i.DEPTH_BUFFER_BIT),R.stencilBuffer&&R.resolveStencilBuffer&&(le|=i.STENCIL_BUFFER_BIT)),_e){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,De.__webglColorRenderbuffer[N]);const J=n.get(T[N]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,J,0)}i.blitFramebuffer(0,0,G,se,0,0,G,se,le,i.NEAREST),l===!0&&(w.length=0,te.length=0,w.push(i.COLOR_ATTACHMENT0+N),R.depthBuffer&&R.resolveDepthBuffer===!1&&(w.push(j),te.push(j),i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,te)),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,w))}if(t.bindFramebuffer(i.READ_FRAMEBUFFER,null),t.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),_e)for(let N=0;N<T.length;N++){t.bindFramebuffer(i.FRAMEBUFFER,De.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+N,i.RENDERBUFFER,De.__webglColorRenderbuffer[N]);const J=n.get(T[N]).__webglTexture;t.bindFramebuffer(i.FRAMEBUFFER,De.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+N,i.TEXTURE_2D,J,0)}t.bindFramebuffer(i.DRAW_FRAMEBUFFER,De.__webglMultisampledFramebuffer)}else if(R.depthBuffer&&R.resolveDepthBuffer===!1&&l){const T=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[T])}}}function ue(R){return Math.min(s.maxSamples,R.samples)}function Z(R){const T=n.get(R);return R.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&T.__useRenderToTexture!==!1}function de(R){const T=o.render.frame;u.get(R)!==T&&(u.set(R,T),R.update())}function ne(R,T){const G=R.colorSpace,se=R.format,le=R.type;return R.isCompressedTexture===!0||R.isVideoTexture===!0||G!==fo&&G!==bs&&(rt.getTransfer(G)===ht?(se!==gn||le!==Ui)&&je("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):zt("WebGLTextures: Unsupported texture color space:",G)),T}function ge(R){return typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement?(c.width=R.naturalWidth||R.width,c.height=R.naturalHeight||R.height):typeof VideoFrame<"u"&&R instanceof VideoFrame?(c.width=R.displayWidth,c.height=R.displayHeight):(c.width=R.width,c.height=R.height),c}this.allocateTextureUnit=U,this.resetTextureUnits=F,this.setTexture2D=k,this.setTexture2DArray=z,this.setTexture3D=V,this.setTextureCube=H,this.rebindTextures=P,this.setupRenderTarget=L,this.updateRenderTargetMipmap=q,this.updateMultisampleRenderTarget=ie,this.setupDepthRenderbuffer=Ce,this.setupFrameBufferTexture=pe,this.useMultisampledRTT=Z}function J0(i,e){function t(n,s=bs){let r;const o=rt.getTransfer(s);if(n===Ui)return i.UNSIGNED_BYTE;if(n===jf)return i.UNSIGNED_SHORT_4_4_4_4;if(n===$f)return i.UNSIGNED_SHORT_5_5_5_1;if(n===I0)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===D0)return i.UNSIGNED_INT_10F_11F_11F_REV;if(n===w0)return i.BYTE;if(n===R0)return i.SHORT;if(n===oa)return i.UNSIGNED_SHORT;if(n===Kf)return i.INT;if(n===si)return i.UNSIGNED_INT;if(n===mi)return i.FLOAT;if(n===pr)return i.HALF_FLOAT;if(n===P0)return i.ALPHA;if(n===F0)return i.RGB;if(n===gn)return i.RGBA;if(n===uo)return i.DEPTH_COMPONENT;if(n===la)return i.DEPTH_STENCIL;if(n===L0)return i.RED;if(n===$l)return i.RED_INTEGER;if(n===Zf)return i.RG;if(n===Jf)return i.RG_INTEGER;if(n===$r)return i.RGBA_INTEGER;if(n===dl||n===hl||n===pl||n===ml)if(o===ht)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(n===dl)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===hl)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===pl)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===ml)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(n===dl)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===hl)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===pl)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===ml)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===Lu||n===Bu||n===Uu||n===Ou)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(n===Lu)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===Bu)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===Uu)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===Ou)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===Nu||n===zu||n===ku)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(n===Nu||n===zu)return o===ht?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(n===ku)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(n===Hu||n===Vu||n===Gu||n===Wu||n===Xu||n===qu||n===Qu||n===Yu||n===Ku||n===ju||n===$u||n===Zu||n===Ju||n===ef)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(n===Hu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===Vu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===Gu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===Wu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===Xu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===qu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===Qu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===Yu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===Ku)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===ju)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===$u)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===Zu)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===Ju)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===ef)return o===ht?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===tf||n===nf||n===sf)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(n===tf)return o===ht?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===nf)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===sf)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===rf||n===of||n===af||n===lf)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(n===rf)return r.COMPRESSED_RED_RGTC1_EXT;if(n===of)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===af)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===lf)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===aa?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:t}}const KC=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,jC=`
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

}`;class $C{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){const n=new Q0(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=n}}getMesh(e){if(this.texture!==null&&this.mesh===null){const t=e.cameras[0].viewport,n=new _n({vertexShader:KC,fragmentShader:jC,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new Vt(new po(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class ZC extends mr{constructor(e,t){super();const n=this;let s=null,r=1,o=null,a="local-floor",l=1,c=null,u=null,f=null,d=null,h=null,x=null;const p=typeof XRWebGLBinding<"u",g=new $C,m={},_=t.getContextAttributes();let A=null,S=null;const v=[],y=[],M=new ze;let E=null;const b=new ei;b.viewport=new Et;const C=new ei;C.viewport=new Et;const I=[b,C],F=new _v;let U=null,O=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(X){let ee=v[X];return ee===void 0&&(ee=new Hc,v[X]=ee),ee.getTargetRaySpace()},this.getControllerGrip=function(X){let ee=v[X];return ee===void 0&&(ee=new Hc,v[X]=ee),ee.getGripSpace()},this.getHand=function(X){let ee=v[X];return ee===void 0&&(ee=new Hc,v[X]=ee),ee.getHandSpace()};function k(X){const ee=y.indexOf(X.inputSource);if(ee===-1)return;const pe=v[ee];pe!==void 0&&(pe.update(X.inputSource,X.frame,c||o),pe.dispatchEvent({type:X.type,data:X.inputSource}))}function z(){s.removeEventListener("select",k),s.removeEventListener("selectstart",k),s.removeEventListener("selectend",k),s.removeEventListener("squeeze",k),s.removeEventListener("squeezestart",k),s.removeEventListener("squeezeend",k),s.removeEventListener("end",z),s.removeEventListener("inputsourceschange",V);for(let X=0;X<v.length;X++){const ee=y[X];ee!==null&&(y[X]=null,v[X].disconnect(ee))}U=null,O=null,g.reset();for(const X in m)delete m[X];e.setRenderTarget(A),h=null,d=null,f=null,s=null,S=null,re.stop(),n.isPresenting=!1,e.setPixelRatio(E),e.setSize(M.width,M.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(X){r=X,n.isPresenting===!0&&je("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(X){a=X,n.isPresenting===!0&&je("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return c||o},this.setReferenceSpace=function(X){c=X},this.getBaseLayer=function(){return d!==null?d:h},this.getBinding=function(){return f===null&&p&&(f=new XRWebGLBinding(s,t)),f},this.getFrame=function(){return x},this.getSession=function(){return s},this.setSession=async function(X){if(s=X,s!==null){if(A=e.getRenderTarget(),s.addEventListener("select",k),s.addEventListener("selectstart",k),s.addEventListener("selectend",k),s.addEventListener("squeeze",k),s.addEventListener("squeezestart",k),s.addEventListener("squeezeend",k),s.addEventListener("end",z),s.addEventListener("inputsourceschange",V),_.xrCompatible!==!0&&await t.makeXRCompatible(),E=e.getPixelRatio(),e.getSize(M),p&&"createProjectionLayer"in XRWebGLBinding.prototype){let pe=null,be=null,xe=null;_.depth&&(xe=_.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,pe=_.stencil?la:uo,be=_.stencil?aa:si);const Ce={colorFormat:t.RGBA8,depthFormat:xe,scaleFactor:r};f=this.getBinding(),d=f.createProjectionLayer(Ce),s.updateRenderState({layers:[d]}),e.setPixelRatio(1),e.setSize(d.textureWidth,d.textureHeight,!1),S=new Us(d.textureWidth,d.textureHeight,{format:gn,type:Ui,depthTexture:new nd(d.textureWidth,d.textureHeight,be,void 0,void 0,void 0,void 0,void 0,void 0,pe),stencilBuffer:_.stencil,colorSpace:e.outputColorSpace,samples:_.antialias?4:0,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1})}else{const pe={antialias:_.antialias,alpha:!0,depth:_.depth,stencil:_.stencil,framebufferScaleFactor:r};h=new XRWebGLLayer(s,t,pe),s.updateRenderState({baseLayer:h}),e.setPixelRatio(1),e.setSize(h.framebufferWidth,h.framebufferHeight,!1),S=new Us(h.framebufferWidth,h.framebufferHeight,{format:gn,type:Ui,colorSpace:e.outputColorSpace,stencilBuffer:_.stencil,resolveDepthBuffer:h.ignoreDepthValues===!1,resolveStencilBuffer:h.ignoreDepthValues===!1})}S.isXRRenderTarget=!0,this.setFoveation(l),c=null,o=await s.requestReferenceSpace(a),re.setContext(s),re.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return g.getDepthTexture()};function V(X){for(let ee=0;ee<X.removed.length;ee++){const pe=X.removed[ee],be=y.indexOf(pe);be>=0&&(y[be]=null,v[be].disconnect(pe))}for(let ee=0;ee<X.added.length;ee++){const pe=X.added[ee];let be=y.indexOf(pe);if(be===-1){for(let Ce=0;Ce<v.length;Ce++)if(Ce>=y.length){y.push(pe),be=Ce;break}else if(y[Ce]===null){y[Ce]=pe,be=Ce;break}if(be===-1)break}const xe=v[be];xe&&xe.connect(pe)}}const H=new B,$=new B;function oe(X,ee,pe){H.setFromMatrixPosition(ee.matrixWorld),$.setFromMatrixPosition(pe.matrixWorld);const be=H.distanceTo($),xe=ee.projectionMatrix.elements,Ce=pe.projectionMatrix.elements,P=xe[14]/(xe[10]-1),L=xe[14]/(xe[10]+1),q=(xe[9]+1)/xe[5],w=(xe[9]-1)/xe[5],te=(xe[8]-1)/xe[0],ie=(Ce[8]+1)/Ce[0],ue=P*te,Z=P*ie,de=be/(-te+ie),ne=de*-te;if(ee.matrixWorld.decompose(X.position,X.quaternion,X.scale),X.translateX(ne),X.translateZ(de),X.matrixWorld.compose(X.position,X.quaternion,X.scale),X.matrixWorldInverse.copy(X.matrixWorld).invert(),xe[10]===-1)X.projectionMatrix.copy(ee.projectionMatrix),X.projectionMatrixInverse.copy(ee.projectionMatrixInverse);else{const ge=P+de,R=L+de,T=ue-ne,G=Z+(be-ne),se=q*L/R*ge,le=w*L/R*ge;X.projectionMatrix.makePerspective(T,G,se,le,ge,R),X.projectionMatrixInverse.copy(X.projectionMatrix).invert()}}function Se(X,ee){ee===null?X.matrixWorld.copy(X.matrix):X.matrixWorld.multiplyMatrices(ee.matrixWorld,X.matrix),X.matrixWorldInverse.copy(X.matrixWorld).invert()}this.updateCamera=function(X){if(s===null)return;let ee=X.near,pe=X.far;g.texture!==null&&(g.depthNear>0&&(ee=g.depthNear),g.depthFar>0&&(pe=g.depthFar)),F.near=C.near=b.near=ee,F.far=C.far=b.far=pe,(U!==F.near||O!==F.far)&&(s.updateRenderState({depthNear:F.near,depthFar:F.far}),U=F.near,O=F.far),F.layers.mask=X.layers.mask|6,b.layers.mask=F.layers.mask&3,C.layers.mask=F.layers.mask&5;const be=X.parent,xe=F.cameras;Se(F,be);for(let Ce=0;Ce<xe.length;Ce++)Se(xe[Ce],be);xe.length===2?oe(F,b,C):F.projectionMatrix.copy(b.projectionMatrix),we(X,F,be)};function we(X,ee,pe){pe===null?X.matrix.copy(ee.matrixWorld):(X.matrix.copy(pe.matrixWorld),X.matrix.invert(),X.matrix.multiply(ee.matrixWorld)),X.matrix.decompose(X.position,X.quaternion,X.scale),X.updateMatrixWorld(!0),X.projectionMatrix.copy(ee.projectionMatrix),X.projectionMatrixInverse.copy(ee.projectionMatrixInverse),X.isPerspectiveCamera&&(X.fov=cf*2*Math.atan(1/X.projectionMatrix.elements[5]),X.zoom=1)}this.getCamera=function(){return F},this.getFoveation=function(){if(!(d===null&&h===null))return l},this.setFoveation=function(X){l=X,d!==null&&(d.fixedFoveation=X),h!==null&&h.fixedFoveation!==void 0&&(h.fixedFoveation=X)},this.hasDepthSensing=function(){return g.texture!==null},this.getDepthSensingMesh=function(){return g.getMesh(F)},this.getCameraTexture=function(X){return m[X]};let Le=null;function fe(X,ee){if(u=ee.getViewerPose(c||o),x=ee,u!==null){const pe=u.views;h!==null&&(e.setRenderTargetFramebuffer(S,h.framebuffer),e.setRenderTarget(S));let be=!1;pe.length!==F.cameras.length&&(F.cameras.length=0,be=!0);for(let L=0;L<pe.length;L++){const q=pe[L];let w=null;if(h!==null)w=h.getViewport(q);else{const ie=f.getViewSubImage(d,q);w=ie.viewport,L===0&&(e.setRenderTargetTextures(S,ie.colorTexture,ie.depthStencilTexture),e.setRenderTarget(S))}let te=I[L];te===void 0&&(te=new ei,te.layers.enable(L),te.viewport=new Et,I[L]=te),te.matrix.fromArray(q.transform.matrix),te.matrix.decompose(te.position,te.quaternion,te.scale),te.projectionMatrix.fromArray(q.projectionMatrix),te.projectionMatrixInverse.copy(te.projectionMatrix).invert(),te.viewport.set(w.x,w.y,w.width,w.height),L===0&&(F.matrix.copy(te.matrix),F.matrix.decompose(F.position,F.quaternion,F.scale)),be===!0&&F.cameras.push(te)}const xe=s.enabledFeatures;if(xe&&xe.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&p){f=n.getBinding();const L=f.getDepthInformation(pe[0]);L&&L.isValid&&L.texture&&g.init(L,s.renderState)}if(xe&&xe.includes("camera-access")&&p){e.state.unbindTexture(),f=n.getBinding();for(let L=0;L<pe.length;L++){const q=pe[L].camera;if(q){let w=m[q];w||(w=new Q0,m[q]=w);const te=f.getCameraImage(q);w.sourceTexture=te}}}}for(let pe=0;pe<v.length;pe++){const be=y[pe],xe=v[pe];be!==null&&xe!==void 0&&xe.update(be,ee,c||o)}Le&&Le(X,ee),ee.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:ee}),x=null}const re=new Y0;re.setAnimationLoop(fe),this.setAnimationLoop=function(X){Le=X},this.dispose=function(){}}}const $s=new _i,JC=new qe;function eT(i,e){function t(g,m){g.matrixAutoUpdate===!0&&g.updateMatrix(),m.value.copy(g.matrix)}function n(g,m){m.color.getRGB(g.fogColor.value,G0(i)),m.isFog?(g.fogNear.value=m.near,g.fogFar.value=m.far):m.isFogExp2&&(g.fogDensity.value=m.density)}function s(g,m,_,A,S){m.isMeshBasicMaterial||m.isMeshLambertMaterial?r(g,m):m.isMeshToonMaterial?(r(g,m),f(g,m)):m.isMeshPhongMaterial?(r(g,m),u(g,m)):m.isMeshStandardMaterial?(r(g,m),d(g,m),m.isMeshPhysicalMaterial&&h(g,m,S)):m.isMeshMatcapMaterial?(r(g,m),x(g,m)):m.isMeshDepthMaterial?r(g,m):m.isMeshDistanceMaterial?(r(g,m),p(g,m)):m.isMeshNormalMaterial?r(g,m):m.isLineBasicMaterial?(o(g,m),m.isLineDashedMaterial&&a(g,m)):m.isPointsMaterial?l(g,m,_,A):m.isSpriteMaterial?c(g,m):m.isShadowMaterial?(g.color.value.copy(m.color),g.opacity.value=m.opacity):m.isShaderMaterial&&(m.uniformsNeedUpdate=!1)}function r(g,m){g.opacity.value=m.opacity,m.color&&g.diffuse.value.copy(m.color),m.emissive&&g.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity),m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.bumpMap&&(g.bumpMap.value=m.bumpMap,t(m.bumpMap,g.bumpMapTransform),g.bumpScale.value=m.bumpScale,m.side===wn&&(g.bumpScale.value*=-1)),m.normalMap&&(g.normalMap.value=m.normalMap,t(m.normalMap,g.normalMapTransform),g.normalScale.value.copy(m.normalScale),m.side===wn&&g.normalScale.value.negate()),m.displacementMap&&(g.displacementMap.value=m.displacementMap,t(m.displacementMap,g.displacementMapTransform),g.displacementScale.value=m.displacementScale,g.displacementBias.value=m.displacementBias),m.emissiveMap&&(g.emissiveMap.value=m.emissiveMap,t(m.emissiveMap,g.emissiveMapTransform)),m.specularMap&&(g.specularMap.value=m.specularMap,t(m.specularMap,g.specularMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest);const _=e.get(m),A=_.envMap,S=_.envMapRotation;A&&(g.envMap.value=A,$s.copy(S),$s.x*=-1,$s.y*=-1,$s.z*=-1,A.isCubeTexture&&A.isRenderTargetTexture===!1&&($s.y*=-1,$s.z*=-1),g.envMapRotation.value.setFromMatrix4(JC.makeRotationFromEuler($s)),g.flipEnvMap.value=A.isCubeTexture&&A.isRenderTargetTexture===!1?-1:1,g.reflectivity.value=m.reflectivity,g.ior.value=m.ior,g.refractionRatio.value=m.refractionRatio),m.lightMap&&(g.lightMap.value=m.lightMap,g.lightMapIntensity.value=m.lightMapIntensity,t(m.lightMap,g.lightMapTransform)),m.aoMap&&(g.aoMap.value=m.aoMap,g.aoMapIntensity.value=m.aoMapIntensity,t(m.aoMap,g.aoMapTransform))}function o(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform))}function a(g,m){g.dashSize.value=m.dashSize,g.totalSize.value=m.dashSize+m.gapSize,g.scale.value=m.scale}function l(g,m,_,A){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.size.value=m.size*_,g.scale.value=A*.5,m.map&&(g.map.value=m.map,t(m.map,g.uvTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function c(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.rotation.value=m.rotation,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function u(g,m){g.specular.value.copy(m.specular),g.shininess.value=Math.max(m.shininess,1e-4)}function f(g,m){m.gradientMap&&(g.gradientMap.value=m.gradientMap)}function d(g,m){g.metalness.value=m.metalness,m.metalnessMap&&(g.metalnessMap.value=m.metalnessMap,t(m.metalnessMap,g.metalnessMapTransform)),g.roughness.value=m.roughness,m.roughnessMap&&(g.roughnessMap.value=m.roughnessMap,t(m.roughnessMap,g.roughnessMapTransform)),m.envMap&&(g.envMapIntensity.value=m.envMapIntensity)}function h(g,m,_){g.ior.value=m.ior,m.sheen>0&&(g.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen),g.sheenRoughness.value=m.sheenRoughness,m.sheenColorMap&&(g.sheenColorMap.value=m.sheenColorMap,t(m.sheenColorMap,g.sheenColorMapTransform)),m.sheenRoughnessMap&&(g.sheenRoughnessMap.value=m.sheenRoughnessMap,t(m.sheenRoughnessMap,g.sheenRoughnessMapTransform))),m.clearcoat>0&&(g.clearcoat.value=m.clearcoat,g.clearcoatRoughness.value=m.clearcoatRoughness,m.clearcoatMap&&(g.clearcoatMap.value=m.clearcoatMap,t(m.clearcoatMap,g.clearcoatMapTransform)),m.clearcoatRoughnessMap&&(g.clearcoatRoughnessMap.value=m.clearcoatRoughnessMap,t(m.clearcoatRoughnessMap,g.clearcoatRoughnessMapTransform)),m.clearcoatNormalMap&&(g.clearcoatNormalMap.value=m.clearcoatNormalMap,t(m.clearcoatNormalMap,g.clearcoatNormalMapTransform),g.clearcoatNormalScale.value.copy(m.clearcoatNormalScale),m.side===wn&&g.clearcoatNormalScale.value.negate())),m.dispersion>0&&(g.dispersion.value=m.dispersion),m.iridescence>0&&(g.iridescence.value=m.iridescence,g.iridescenceIOR.value=m.iridescenceIOR,g.iridescenceThicknessMinimum.value=m.iridescenceThicknessRange[0],g.iridescenceThicknessMaximum.value=m.iridescenceThicknessRange[1],m.iridescenceMap&&(g.iridescenceMap.value=m.iridescenceMap,t(m.iridescenceMap,g.iridescenceMapTransform)),m.iridescenceThicknessMap&&(g.iridescenceThicknessMap.value=m.iridescenceThicknessMap,t(m.iridescenceThicknessMap,g.iridescenceThicknessMapTransform))),m.transmission>0&&(g.transmission.value=m.transmission,g.transmissionSamplerMap.value=_.texture,g.transmissionSamplerSize.value.set(_.width,_.height),m.transmissionMap&&(g.transmissionMap.value=m.transmissionMap,t(m.transmissionMap,g.transmissionMapTransform)),g.thickness.value=m.thickness,m.thicknessMap&&(g.thicknessMap.value=m.thicknessMap,t(m.thicknessMap,g.thicknessMapTransform)),g.attenuationDistance.value=m.attenuationDistance,g.attenuationColor.value.copy(m.attenuationColor)),m.anisotropy>0&&(g.anisotropyVector.value.set(m.anisotropy*Math.cos(m.anisotropyRotation),m.anisotropy*Math.sin(m.anisotropyRotation)),m.anisotropyMap&&(g.anisotropyMap.value=m.anisotropyMap,t(m.anisotropyMap,g.anisotropyMapTransform))),g.specularIntensity.value=m.specularIntensity,g.specularColor.value.copy(m.specularColor),m.specularColorMap&&(g.specularColorMap.value=m.specularColorMap,t(m.specularColorMap,g.specularColorMapTransform)),m.specularIntensityMap&&(g.specularIntensityMap.value=m.specularIntensityMap,t(m.specularIntensityMap,g.specularIntensityMapTransform))}function x(g,m){m.matcap&&(g.matcap.value=m.matcap)}function p(g,m){const _=e.get(m).light;g.referencePosition.value.setFromMatrixPosition(_.matrixWorld),g.nearDistance.value=_.shadow.camera.near,g.farDistance.value=_.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:s}}function tT(i,e,t,n){let s={},r={},o=[];const a=i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);function l(_,A){const S=A.program;n.uniformBlockBinding(_,S)}function c(_,A){let S=s[_.id];S===void 0&&(x(_),S=u(_),s[_.id]=S,_.addEventListener("dispose",g));const v=A.program;n.updateUBOMapping(_,v);const y=e.render.frame;r[_.id]!==y&&(d(_),r[_.id]=y)}function u(_){const A=f();_.__bindingPointIndex=A;const S=i.createBuffer(),v=_.__size,y=_.usage;return i.bindBuffer(i.UNIFORM_BUFFER,S),i.bufferData(i.UNIFORM_BUFFER,v,y),i.bindBuffer(i.UNIFORM_BUFFER,null),i.bindBufferBase(i.UNIFORM_BUFFER,A,S),S}function f(){for(let _=0;_<a;_++)if(o.indexOf(_)===-1)return o.push(_),_;return zt("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function d(_){const A=s[_.id],S=_.uniforms,v=_.__cache;i.bindBuffer(i.UNIFORM_BUFFER,A);for(let y=0,M=S.length;y<M;y++){const E=Array.isArray(S[y])?S[y]:[S[y]];for(let b=0,C=E.length;b<C;b++){const I=E[b];if(h(I,y,b,v)===!0){const F=I.__offset,U=Array.isArray(I.value)?I.value:[I.value];let O=0;for(let k=0;k<U.length;k++){const z=U[k],V=p(z);typeof z=="number"||typeof z=="boolean"?(I.__data[0]=z,i.bufferSubData(i.UNIFORM_BUFFER,F+O,I.__data)):z.isMatrix3?(I.__data[0]=z.elements[0],I.__data[1]=z.elements[1],I.__data[2]=z.elements[2],I.__data[3]=0,I.__data[4]=z.elements[3],I.__data[5]=z.elements[4],I.__data[6]=z.elements[5],I.__data[7]=0,I.__data[8]=z.elements[6],I.__data[9]=z.elements[7],I.__data[10]=z.elements[8],I.__data[11]=0):(z.toArray(I.__data,O),O+=V.storage/Float32Array.BYTES_PER_ELEMENT)}i.bufferSubData(i.UNIFORM_BUFFER,F,I.__data)}}}i.bindBuffer(i.UNIFORM_BUFFER,null)}function h(_,A,S,v){const y=_.value,M=A+"_"+S;if(v[M]===void 0)return typeof y=="number"||typeof y=="boolean"?v[M]=y:v[M]=y.clone(),!0;{const E=v[M];if(typeof y=="number"||typeof y=="boolean"){if(E!==y)return v[M]=y,!0}else if(E.equals(y)===!1)return E.copy(y),!0}return!1}function x(_){const A=_.uniforms;let S=0;const v=16;for(let M=0,E=A.length;M<E;M++){const b=Array.isArray(A[M])?A[M]:[A[M]];for(let C=0,I=b.length;C<I;C++){const F=b[C],U=Array.isArray(F.value)?F.value:[F.value];for(let O=0,k=U.length;O<k;O++){const z=U[O],V=p(z),H=S%v,$=H%V.boundary,oe=H+$;S+=$,oe!==0&&v-oe<V.storage&&(S+=v-oe),F.__data=new Float32Array(V.storage/Float32Array.BYTES_PER_ELEMENT),F.__offset=S,S+=V.storage}}}const y=S%v;return y>0&&(S+=v-y),_.__size=S,_.__cache={},this}function p(_){const A={boundary:0,storage:0};return typeof _=="number"||typeof _=="boolean"?(A.boundary=4,A.storage=4):_.isVector2?(A.boundary=8,A.storage=8):_.isVector3||_.isColor?(A.boundary=16,A.storage=12):_.isVector4?(A.boundary=16,A.storage=16):_.isMatrix3?(A.boundary=48,A.storage=48):_.isMatrix4?(A.boundary=64,A.storage=64):_.isTexture?je("WebGLRenderer: Texture samplers can not be part of an uniforms group."):je("WebGLRenderer: Unsupported uniform value type.",_),A}function g(_){const A=_.target;A.removeEventListener("dispose",g);const S=o.indexOf(A.__bindingPointIndex);o.splice(S,1),i.deleteBuffer(s[A.id]),delete s[A.id],delete r[A.id]}function m(){for(const _ in s)i.deleteBuffer(s[_]);o=[],s={},r={}}return{bind:l,update:c,dispose:m}}const nT=new Uint16Array([11481,15204,11534,15171,11808,15015,12385,14843,12894,14716,13396,14600,13693,14483,13976,14366,14237,14171,14405,13961,14511,13770,14605,13598,14687,13444,14760,13305,14822,13066,14876,12857,14923,12675,14963,12517,14997,12379,15025,12230,15049,12023,15070,11843,15086,11687,15100,11551,15111,11433,15120,11330,15127,11217,15132,11060,15135,10922,15138,10801,15139,10695,15139,10600,13012,14923,13020,14917,13064,14886,13176,14800,13349,14666,13513,14526,13724,14398,13960,14230,14200,14020,14383,13827,14488,13651,14583,13491,14667,13348,14740,13132,14803,12908,14856,12713,14901,12542,14938,12394,14968,12241,14992,12017,15010,11822,15024,11654,15034,11507,15041,11380,15044,11269,15044,11081,15042,10913,15037,10764,15031,10635,15023,10520,15014,10419,15003,10330,13657,14676,13658,14673,13670,14660,13698,14622,13750,14547,13834,14442,13956,14317,14112,14093,14291,13889,14407,13704,14499,13538,14586,13389,14664,13201,14733,12966,14792,12758,14842,12577,14882,12418,14915,12272,14940,12033,14959,11826,14972,11646,14980,11490,14983,11355,14983,11212,14979,11008,14971,10830,14961,10675,14950,10540,14936,10420,14923,10315,14909,10204,14894,10041,14089,14460,14090,14459,14096,14452,14112,14431,14141,14388,14186,14305,14252,14130,14341,13941,14399,13756,14467,13585,14539,13430,14610,13272,14677,13026,14737,12808,14790,12617,14833,12449,14869,12303,14896,12065,14916,11845,14929,11655,14937,11490,14939,11347,14936,11184,14930,10970,14921,10783,14912,10621,14900,10480,14885,10356,14867,10247,14848,10062,14827,9894,14805,9745,14400,14208,14400,14206,14402,14198,14406,14174,14415,14122,14427,14035,14444,13913,14469,13767,14504,13613,14548,13463,14598,13324,14651,13082,14704,12858,14752,12658,14795,12483,14831,12330,14860,12106,14881,11875,14895,11675,14903,11501,14905,11351,14903,11178,14900,10953,14892,10757,14880,10589,14865,10442,14847,10313,14827,10162,14805,9965,14782,9792,14757,9642,14731,9507,14562,13883,14562,13883,14563,13877,14566,13862,14570,13830,14576,13773,14584,13689,14595,13582,14613,13461,14637,13336,14668,13120,14704,12897,14741,12695,14776,12516,14808,12358,14835,12150,14856,11910,14870,11701,14878,11519,14882,11361,14884,11187,14880,10951,14871,10748,14858,10572,14842,10418,14823,10286,14801,10099,14777,9897,14751,9722,14725,9567,14696,9430,14666,9309,14702,13604,14702,13604,14702,13600,14703,13591,14705,13570,14707,13533,14709,13477,14712,13400,14718,13305,14727,13106,14743,12907,14762,12716,14784,12539,14807,12380,14827,12190,14844,11943,14855,11727,14863,11539,14870,11376,14871,11204,14868,10960,14858,10748,14845,10565,14829,10406,14809,10269,14786,10058,14761,9852,14734,9671,14705,9512,14674,9374,14641,9253,14608,9076,14821,13366,14821,13365,14821,13364,14821,13358,14821,13344,14821,13320,14819,13252,14817,13145,14815,13011,14814,12858,14817,12698,14823,12539,14832,12389,14841,12214,14850,11968,14856,11750,14861,11558,14866,11390,14867,11226,14862,10972,14853,10754,14840,10565,14823,10401,14803,10259,14780,10032,14754,9820,14725,9635,14694,9473,14661,9333,14627,9203,14593,8988,14557,8798,14923,13014,14922,13014,14922,13012,14922,13004,14920,12987,14919,12957,14915,12907,14909,12834,14902,12738,14894,12623,14888,12498,14883,12370,14880,12203,14878,11970,14875,11759,14873,11569,14874,11401,14872,11243,14865,10986,14855,10762,14842,10568,14825,10401,14804,10255,14781,10017,14754,9799,14725,9611,14692,9445,14658,9301,14623,9139,14587,8920,14548,8729,14509,8562,15008,12672,15008,12672,15008,12671,15007,12667,15005,12656,15001,12637,14997,12605,14989,12556,14978,12490,14966,12407,14953,12313,14940,12136,14927,11934,14914,11742,14903,11563,14896,11401,14889,11247,14879,10992,14866,10767,14851,10570,14833,10400,14812,10252,14789,10007,14761,9784,14731,9592,14698,9424,14663,9279,14627,9088,14588,8868,14548,8676,14508,8508,14467,8360,15080,12386,15080,12386,15079,12385,15078,12383,15076,12378,15072,12367,15066,12347,15057,12315,15045,12253,15030,12138,15012,11998,14993,11845,14972,11685,14951,11530,14935,11383,14920,11228,14904,10981,14887,10762,14870,10567,14850,10397,14827,10248,14803,9997,14774,9771,14743,9578,14710,9407,14674,9259,14637,9048,14596,8826,14555,8632,14514,8464,14471,8317,14427,8182,15139,12008,15139,12008,15138,12008,15137,12007,15135,12003,15130,11990,15124,11969,15115,11929,15102,11872,15086,11794,15064,11693,15041,11581,15013,11459,14987,11336,14966,11170,14944,10944,14921,10738,14898,10552,14875,10387,14850,10239,14824,9983,14794,9758,14762,9563,14728,9392,14692,9244,14653,9014,14611,8791,14569,8597,14526,8427,14481,8281,14436,8110,14391,7885,15188,11617,15188,11617,15187,11617,15186,11618,15183,11617,15179,11612,15173,11601,15163,11581,15150,11546,15133,11495,15110,11427,15083,11346,15051,11246,15024,11057,14996,10868,14967,10687,14938,10517,14911,10362,14882,10206,14853,9956,14821,9737,14787,9543,14752,9375,14715,9228,14675,8980,14632,8760,14589,8565,14544,8395,14498,8248,14451,8049,14404,7824,14357,7630,15228,11298,15228,11298,15227,11299,15226,11301,15223,11303,15219,11302,15213,11299,15204,11290,15191,11271,15174,11217,15150,11129,15119,11015,15087,10886,15057,10744,15024,10599,14990,10455,14957,10318,14924,10143,14891,9911,14856,9701,14820,9516,14782,9352,14744,9200,14703,8946,14659,8725,14615,8533,14568,8366,14521,8220,14472,7992,14423,7770,14374,7578,14315,7408,15260,10819,15260,10819,15259,10822,15258,10826,15256,10832,15251,10836,15246,10841,15237,10838,15225,10821,15207,10788,15183,10734,15151,10660,15120,10571,15087,10469,15049,10359,15012,10249,14974,10041,14937,9837,14900,9647,14860,9475,14820,9320,14779,9147,14736,8902,14691,8688,14646,8499,14598,8335,14549,8189,14499,7940,14448,7720,14397,7529,14347,7363,14256,7218,15285,10410,15285,10411,15285,10413,15284,10418,15282,10425,15278,10434,15272,10442,15264,10449,15252,10445,15235,10433,15210,10403,15179,10358,15149,10301,15113,10218,15073,10059,15033,9894,14991,9726,14951,9565,14909,9413,14865,9273,14822,9073,14777,8845,14730,8641,14682,8459,14633,8300,14583,8129,14531,7883,14479,7670,14426,7482,14373,7321,14305,7176,14201,6939,15305,9939,15305,9940,15305,9945,15304,9955,15302,9967,15298,9989,15293,10010,15286,10033,15274,10044,15258,10045,15233,10022,15205,9975,15174,9903,15136,9808,15095,9697,15053,9578,15009,9451,14965,9327,14918,9198,14871,8973,14825,8766,14775,8579,14725,8408,14675,8259,14622,8058,14569,7821,14515,7615,14460,7435,14405,7276,14350,7108,14256,6866,14149,6653,15321,9444,15321,9445,15321,9448,15320,9458,15317,9470,15314,9490,15310,9515,15302,9540,15292,9562,15276,9579,15251,9577,15226,9559,15195,9519,15156,9463,15116,9389,15071,9304,15025,9208,14978,9023,14927,8838,14878,8661,14827,8496,14774,8344,14722,8206,14667,7973,14612,7749,14556,7555,14499,7382,14443,7229,14385,7025,14322,6791,14210,6588,14100,6409,15333,8920,15333,8921,15332,8927,15332,8943,15329,8965,15326,9002,15322,9048,15316,9106,15307,9162,15291,9204,15267,9221,15244,9221,15212,9196,15175,9134,15133,9043,15088,8930,15040,8801,14990,8665,14938,8526,14886,8391,14830,8261,14775,8087,14719,7866,14661,7664,14603,7482,14544,7322,14485,7178,14426,6936,14367,6713,14281,6517,14166,6348,14054,6198,15341,8360,15341,8361,15341,8366,15341,8379,15339,8399,15336,8431,15332,8473,15326,8527,15318,8585,15302,8632,15281,8670,15258,8690,15227,8690,15191,8664,15149,8612,15104,8543,15055,8456,15001,8360,14948,8259,14892,8122,14834,7923,14776,7734,14716,7558,14656,7397,14595,7250,14534,7070,14472,6835,14410,6628,14350,6443,14243,6283,14125,6135,14010,5889,15348,7715,15348,7717,15348,7725,15347,7745,15345,7780,15343,7836,15339,7905,15334,8e3,15326,8103,15310,8193,15293,8239,15270,8270,15240,8287,15204,8283,15163,8260,15118,8223,15067,8143,15014,8014,14958,7873,14899,7723,14839,7573,14778,7430,14715,7293,14652,7164,14588,6931,14524,6720,14460,6531,14396,6362,14330,6210,14207,6015,14086,5781,13969,5576,15352,7114,15352,7116,15352,7128,15352,7159,15350,7195,15348,7237,15345,7299,15340,7374,15332,7457,15317,7544,15301,7633,15280,7703,15251,7754,15216,7775,15176,7767,15131,7733,15079,7670,15026,7588,14967,7492,14906,7387,14844,7278,14779,7171,14714,6965,14648,6770,14581,6587,14515,6420,14448,6269,14382,6123,14299,5881,14172,5665,14049,5477,13929,5310,15355,6329,15355,6330,15355,6339,15355,6362,15353,6410,15351,6472,15349,6572,15344,6688,15337,6835,15323,6985,15309,7142,15287,7220,15260,7277,15226,7310,15188,7326,15142,7318,15090,7285,15036,7239,14976,7177,14914,7045,14849,6892,14782,6736,14714,6581,14645,6433,14576,6293,14506,6164,14438,5946,14369,5733,14270,5540,14140,5369,14014,5216,13892,5043,15357,5483,15357,5484,15357,5496,15357,5528,15356,5597,15354,5692,15351,5835,15347,6011,15339,6195,15328,6317,15314,6446,15293,6566,15268,6668,15235,6746,15197,6796,15152,6811,15101,6790,15046,6748,14985,6673,14921,6583,14854,6479,14785,6371,14714,6259,14643,6149,14571,5946,14499,5750,14428,5567,14358,5401,14242,5250,14109,5111,13980,4870,13856,4657,15359,4555,15359,4557,15358,4573,15358,4633,15357,4715,15355,4841,15353,5061,15349,5216,15342,5391,15331,5577,15318,5770,15299,5967,15274,6150,15243,6223,15206,6280,15161,6310,15111,6317,15055,6300,14994,6262,14928,6208,14860,6141,14788,5994,14715,5838,14641,5684,14566,5529,14492,5384,14418,5247,14346,5121,14216,4892,14079,4682,13948,4496,13822,4330,15359,3498,15359,3501,15359,3520,15359,3598,15358,3719,15356,3860,15355,4137,15351,4305,15344,4563,15334,4809,15321,5116,15303,5273,15280,5418,15250,5547,15214,5653,15170,5722,15120,5761,15064,5763,15002,5733,14935,5673,14865,5597,14792,5504,14716,5400,14640,5294,14563,5185,14486,5041,14410,4841,14335,4655,14191,4482,14051,4325,13918,4183,13790,4012,15360,2282,15360,2285,15360,2306,15360,2401,15359,2547,15357,2748,15355,3103,15352,3349,15345,3675,15336,4020,15324,4272,15307,4496,15285,4716,15255,4908,15220,5086,15178,5170,15128,5214,15072,5234,15010,5231,14943,5206,14871,5166,14796,5102,14718,4971,14639,4833,14559,4687,14480,4541,14402,4401,14315,4268,14167,4142,14025,3958,13888,3747,13759,3556,15360,923,15360,925,15360,946,15360,1052,15359,1214,15357,1494,15356,1892,15352,2274,15346,2663,15338,3099,15326,3393,15309,3679,15288,3980,15260,4183,15226,4325,15185,4437,15136,4517,15080,4570,15018,4591,14950,4581,14877,4545,14800,4485,14720,4411,14638,4325,14556,4231,14475,4136,14395,3988,14297,3803,14145,3628,13999,3465,13861,3314,13729,3177,15360,263,15360,264,15360,272,15360,325,15359,407,15358,548,15356,780,15352,1144,15347,1580,15339,2099,15328,2425,15312,2795,15292,3133,15264,3329,15232,3517,15191,3689,15143,3819,15088,3923,15025,3978,14956,3999,14882,3979,14804,3931,14722,3855,14639,3756,14554,3645,14470,3529,14388,3409,14279,3289,14124,3173,13975,3055,13834,2848,13701,2658,15360,49,15360,49,15360,52,15360,75,15359,111,15358,201,15356,283,15353,519,15348,726,15340,1045,15329,1415,15314,1795,15295,2173,15269,2410,15237,2649,15197,2866,15150,3054,15095,3140,15032,3196,14963,3228,14888,3236,14808,3224,14725,3191,14639,3146,14553,3088,14466,2976,14382,2836,14262,2692,14103,2549,13952,2409,13808,2278,13674,2154,15360,4,15360,4,15360,4,15360,13,15359,33,15358,59,15357,112,15353,199,15348,302,15341,456,15331,628,15316,827,15297,1082,15272,1332,15241,1601,15202,1851,15156,2069,15101,2172,15039,2256,14970,2314,14894,2348,14813,2358,14728,2344,14640,2311,14551,2263,14463,2203,14376,2133,14247,2059,14084,1915,13930,1761,13784,1609,13648,1464,15360,0,15360,0,15360,0,15360,3,15359,18,15358,26,15357,53,15354,80,15348,97,15341,165,15332,238,15318,326,15299,427,15275,529,15245,654,15207,771,15161,885,15108,994,15046,1089,14976,1170,14900,1229,14817,1266,14731,1284,14641,1282,14550,1260,14460,1223,14370,1174,14232,1116,14066,1050,13909,981,13761,910,13623,839]);let Wi=null;function iT(){return Wi===null&&(Wi=new Qi(nT,32,32,Zf,pr),Wi.minFilter=ii,Wi.magFilter=ii,Wi.wrapS=ns,Wi.wrapT=ns,Wi.generateMipmaps=!1,Wi.needsUpdate=!0),Wi}class sT{constructor(e={}){const{canvas:t=PS(),context:n=null,depth:s=!0,stencil:r=!1,alpha:o=!1,antialias:a=!1,premultipliedAlpha:l=!0,preserveDrawingBuffer:c=!1,powerPreference:u="default",failIfMajorPerformanceCaveat:f=!1,reversedDepthBuffer:d=!1}=e;this.isWebGLRenderer=!0;let h;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");h=n.getContextAttributes().alpha}else h=o;const x=new Set([$r,Jf,$l]),p=new Set([Ui,si,oa,aa,jf,$f]),g=new Uint32Array(4),m=new Int32Array(4);let _=null,A=null;const S=[],v=[];this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=Ds,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const y=this;let M=!1;this._outputColorSpace=Jn;let E=0,b=0,C=null,I=-1,F=null;const U=new Et,O=new Et;let k=null;const z=new nt(0);let V=0,H=t.width,$=t.height,oe=1,Se=null,we=null;const Le=new Et(0,0,H,$),fe=new Et(0,0,H,$);let re=!1;const X=new q0;let ee=!1,pe=!1;const be=new qe,xe=new B,Ce=new Et,P={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let L=!1;function q(){return C===null?oe:1}let w=n;function te(D,Q){return t.getContext(D,Q)}try{const D={alpha:!0,depth:s,stencil:r,antialias:a,premultipliedAlpha:l,preserveDrawingBuffer:c,powerPreference:u,failIfMajorPerformanceCaveat:f};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${Yf}`),t.addEventListener("webglcontextlost",Ae,!1),t.addEventListener("webglcontextrestored",he,!1),t.addEventListener("webglcontextcreationerror",Oe,!1),w===null){const Q="webgl2";if(w=te(Q,D),w===null)throw te(Q)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(D){throw D("WebGLRenderer: "+D.message),D}let ie,ue,Z,de,ne,ge,R,T,G,se,le,j,De,_e,Ue,N,J,me,Te,Pe,Re,He,W,Ie;function ve(){ie=new hM(w),ie.init(),He=new J0(w,ie),ue=new sM(w,ie,e,He),Z=new QC(w,ie),ue.reversedDepthBuffer&&d&&Z.buffers.depth.setReversed(!0),de=new gM(w),ne=new LC,ge=new YC(w,ie,Z,ne,ue,He,de),R=new oM(y),T=new dM(y),G=new Sv(w),W=new nM(w,G),se=new pM(w,G,de,W),le=new _M(w,se,G,de),Te=new xM(w,ue,ge),N=new rM(ne),j=new FC(y,R,T,ie,ue,W,N),De=new eT(y,ne),_e=new UC,Ue=new VC(ie),me=new tM(y,R,T,Z,le,h,l),J=new XC(y,le,ue),Ie=new tT(w,de,ue,Z),Pe=new iM(w,ie,de),Re=new mM(w,ie,de),de.programs=j.programs,y.capabilities=ue,y.extensions=ie,y.properties=ne,y.renderLists=_e,y.shadowMap=J,y.state=Z,y.info=de}ve();const ye=new ZC(y,w);this.xr=ye,this.getContext=function(){return w},this.getContextAttributes=function(){return w.getContextAttributes()},this.forceContextLoss=function(){const D=ie.get("WEBGL_lose_context");D&&D.loseContext()},this.forceContextRestore=function(){const D=ie.get("WEBGL_lose_context");D&&D.restoreContext()},this.getPixelRatio=function(){return oe},this.setPixelRatio=function(D){D!==void 0&&(oe=D,this.setSize(H,$,!1))},this.getSize=function(D){return D.set(H,$)},this.setSize=function(D,Q,ae=!0){if(ye.isPresenting){je("WebGLRenderer: Can't change size while VR device is presenting.");return}H=D,$=Q,t.width=Math.floor(D*oe),t.height=Math.floor(Q*oe),ae===!0&&(t.style.width=D+"px",t.style.height=Q+"px"),this.setViewport(0,0,D,Q)},this.getDrawingBufferSize=function(D){return D.set(H*oe,$*oe).floor()},this.setDrawingBufferSize=function(D,Q,ae){H=D,$=Q,oe=ae,t.width=Math.floor(D*ae),t.height=Math.floor(Q*ae),this.setViewport(0,0,D,Q)},this.getCurrentViewport=function(D){return D.copy(U)},this.getViewport=function(D){return D.copy(Le)},this.setViewport=function(D,Q,ae,ce){D.isVector4?Le.set(D.x,D.y,D.z,D.w):Le.set(D,Q,ae,ce),Z.viewport(U.copy(Le).multiplyScalar(oe).round())},this.getScissor=function(D){return D.copy(fe)},this.setScissor=function(D,Q,ae,ce){D.isVector4?fe.set(D.x,D.y,D.z,D.w):fe.set(D,Q,ae,ce),Z.scissor(O.copy(fe).multiplyScalar(oe).round())},this.getScissorTest=function(){return re},this.setScissorTest=function(D){Z.setScissorTest(re=D)},this.setOpaqueSort=function(D){Se=D},this.setTransparentSort=function(D){we=D},this.getClearColor=function(D){return D.copy(me.getClearColor())},this.setClearColor=function(){me.setClearColor(...arguments)},this.getClearAlpha=function(){return me.getClearAlpha()},this.setClearAlpha=function(){me.setClearAlpha(...arguments)},this.clear=function(D=!0,Q=!0,ae=!0){let ce=0;if(D){let Y=!1;if(C!==null){const Me=C.texture.format;Y=x.has(Me)}if(Y){const Me=C.texture.type,Be=p.has(Me),ke=me.getClearColor(),Ne=me.getClearAlpha(),Xe=ke.r,Ye=ke.g,Ve=ke.b;Be?(g[0]=Xe,g[1]=Ye,g[2]=Ve,g[3]=Ne,w.clearBufferuiv(w.COLOR,0,g)):(m[0]=Xe,m[1]=Ye,m[2]=Ve,m[3]=Ne,w.clearBufferiv(w.COLOR,0,m))}else ce|=w.COLOR_BUFFER_BIT}Q&&(ce|=w.DEPTH_BUFFER_BIT),ae&&(ce|=w.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),w.clear(ce)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){t.removeEventListener("webglcontextlost",Ae,!1),t.removeEventListener("webglcontextrestored",he,!1),t.removeEventListener("webglcontextcreationerror",Oe,!1),me.dispose(),_e.dispose(),Ue.dispose(),ne.dispose(),R.dispose(),T.dispose(),le.dispose(),W.dispose(),Ie.dispose(),j.dispose(),ye.dispose(),ye.removeEventListener("sessionstart",kd),ye.removeEventListener("sessionend",Hd),Vs.stop()};function Ae(D){D.preventDefault(),Lh("WebGLRenderer: Context Lost."),M=!0}function he(){Lh("WebGLRenderer: Context Restored."),M=!1;const D=de.autoReset,Q=J.enabled,ae=J.autoUpdate,ce=J.needsUpdate,Y=J.type;ve(),de.autoReset=D,J.enabled=Q,J.autoUpdate=ae,J.needsUpdate=ce,J.type=Y}function Oe(D){zt("WebGLRenderer: A WebGL context could not be created. Reason: ",D.statusMessage)}function We(D){const Q=D.target;Q.removeEventListener("dispose",We),vt(Q)}function vt(D){ut(D),ne.remove(D)}function ut(D){const Q=ne.get(D).programs;Q!==void 0&&(Q.forEach(function(ae){j.releaseProgram(ae)}),D.isShaderMaterial&&j.releaseShaderCache(D))}this.renderBufferDirect=function(D,Q,ae,ce,Y,Me){Q===null&&(Q=P);const Be=Y.isMesh&&Y.matrixWorld.determinant()<0,ke=lx(D,Q,ae,ce,Y);Z.setMaterial(ce,Be);let Ne=ae.index,Xe=1;if(ce.wireframe===!0){if(Ne=se.getWireframeAttribute(ae),Ne===void 0)return;Xe=2}const Ye=ae.drawRange,Ve=ae.attributes.position;let et=Ye.start*Xe,ft=(Ye.start+Ye.count)*Xe;Me!==null&&(et=Math.max(et,Me.start*Xe),ft=Math.min(ft,(Me.start+Me.count)*Xe)),Ne!==null?(et=Math.max(et,0),ft=Math.min(ft,Ne.count)):Ve!=null&&(et=Math.max(et,0),ft=Math.min(ft,Ve.count));const Lt=ft-et;if(Lt<0||Lt===1/0)return;W.setup(Y,ce,ke,ae,Ne);let Bt,gt=Pe;if(Ne!==null&&(Bt=G.get(Ne),gt=Re,gt.setIndex(Bt)),Y.isMesh)ce.wireframe===!0?(Z.setLineWidth(ce.wireframeLinewidth*q()),gt.setMode(w.LINES)):gt.setMode(w.TRIANGLES);else if(Y.isLine){let Ge=ce.linewidth;Ge===void 0&&(Ge=1),Z.setLineWidth(Ge*q()),Y.isLineSegments?gt.setMode(w.LINES):Y.isLineLoop?gt.setMode(w.LINE_LOOP):gt.setMode(w.LINE_STRIP)}else Y.isPoints?gt.setMode(w.POINTS):Y.isSprite&&gt.setMode(w.TRIANGLES);if(Y.isBatchedMesh)if(Y._multiDrawInstances!==null)ca("WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."),gt.renderMultiDrawInstances(Y._multiDrawStarts,Y._multiDrawCounts,Y._multiDrawCount,Y._multiDrawInstances);else if(ie.get("WEBGL_multi_draw"))gt.renderMultiDraw(Y._multiDrawStarts,Y._multiDrawCounts,Y._multiDrawCount);else{const Ge=Y._multiDrawStarts,wt=Y._multiDrawCounts,it=Y._multiDrawCount,Ln=Ne?G.get(Ne).bytesPerElement:1,xr=ne.get(ce).currentProgram.getUniforms();for(let Bn=0;Bn<it;Bn++)xr.setValue(w,"_gl_DrawID",Bn),gt.render(Ge[Bn]/Ln,wt[Bn])}else if(Y.isInstancedMesh)gt.renderInstances(et,Lt,Y.count);else if(ae.isInstancedBufferGeometry){const Ge=ae._maxInstanceCount!==void 0?ae._maxInstanceCount:1/0,wt=Math.min(ae.instanceCount,Ge);gt.renderInstances(et,Lt,wt)}else gt.render(et,Lt)};function Ai(D,Q,ae){D.transparent===!0&&D.side===ti&&D.forceSinglePass===!1?(D.side=wn,D.needsUpdate=!0,Ea(D,Q,ae),D.side=Bi,D.needsUpdate=!0,Ea(D,Q,ae),D.side=ti):Ea(D,Q,ae)}this.compile=function(D,Q,ae=null){ae===null&&(ae=D),A=Ue.get(ae),A.init(Q),v.push(A),ae.traverseVisible(function(Y){Y.isLight&&Y.layers.test(Q.layers)&&(A.pushLight(Y),Y.castShadow&&A.pushShadow(Y))}),D!==ae&&D.traverseVisible(function(Y){Y.isLight&&Y.layers.test(Q.layers)&&(A.pushLight(Y),Y.castShadow&&A.pushShadow(Y))}),A.setupLights();const ce=new Set;return D.traverse(function(Y){if(!(Y.isMesh||Y.isPoints||Y.isLine||Y.isSprite))return;const Me=Y.material;if(Me)if(Array.isArray(Me))for(let Be=0;Be<Me.length;Be++){const ke=Me[Be];Ai(ke,ae,Y),ce.add(ke)}else Ai(Me,ae,Y),ce.add(Me)}),A=v.pop(),ce},this.compileAsync=function(D,Q,ae=null){const ce=this.compile(D,Q,ae);return new Promise(Y=>{function Me(){if(ce.forEach(function(Be){ne.get(Be).currentProgram.isReady()&&ce.delete(Be)}),ce.size===0){Y(D);return}setTimeout(Me,10)}ie.get("KHR_parallel_shader_compile")!==null?Me():setTimeout(Me,10)})};let ci=null;function ax(D){ci&&ci(D)}function kd(){Vs.stop()}function Hd(){Vs.start()}const Vs=new Y0;Vs.setAnimationLoop(ax),typeof self<"u"&&Vs.setContext(self),this.setAnimationLoop=function(D){ci=D,ye.setAnimationLoop(D),D===null?Vs.stop():Vs.start()},ye.addEventListener("sessionstart",kd),ye.addEventListener("sessionend",Hd),this.render=function(D,Q){if(Q!==void 0&&Q.isCamera!==!0){zt("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(M===!0)return;if(D.matrixWorldAutoUpdate===!0&&D.updateMatrixWorld(),Q.parent===null&&Q.matrixWorldAutoUpdate===!0&&Q.updateMatrixWorld(),ye.enabled===!0&&ye.isPresenting===!0&&(ye.cameraAutoUpdate===!0&&ye.updateCamera(Q),Q=ye.getCamera()),D.isScene===!0&&D.onBeforeRender(y,D,Q,C),A=Ue.get(D,v.length),A.init(Q),v.push(A),be.multiplyMatrices(Q.projectionMatrix,Q.matrixWorldInverse),X.setFromProjectionMatrix(be,Ei,Q.reversedDepth),pe=this.localClippingEnabled,ee=N.init(this.clippingPlanes,pe),_=_e.get(D,S.length),_.init(),S.push(_),ye.enabled===!0&&ye.isPresenting===!0){const Me=y.xr.getDepthSensingMesh();Me!==null&&rc(Me,Q,-1/0,y.sortObjects)}rc(D,Q,0,y.sortObjects),_.finish(),y.sortObjects===!0&&_.sort(Se,we),L=ye.enabled===!1||ye.isPresenting===!1||ye.hasDepthSensing()===!1,L&&me.addToRenderList(_,D),this.info.render.frame++,ee===!0&&N.beginShadows();const ae=A.state.shadowsArray;J.render(ae,D,Q),ee===!0&&N.endShadows(),this.info.autoReset===!0&&this.info.reset();const ce=_.opaque,Y=_.transmissive;if(A.setupLights(),Q.isArrayCamera){const Me=Q.cameras;if(Y.length>0)for(let Be=0,ke=Me.length;Be<ke;Be++){const Ne=Me[Be];Gd(ce,Y,D,Ne)}L&&me.render(D);for(let Be=0,ke=Me.length;Be<ke;Be++){const Ne=Me[Be];Vd(_,D,Ne,Ne.viewport)}}else Y.length>0&&Gd(ce,Y,D,Q),L&&me.render(D),Vd(_,D,Q);C!==null&&b===0&&(ge.updateMultisampleRenderTarget(C),ge.updateRenderTargetMipmap(C)),D.isScene===!0&&D.onAfterRender(y,D,Q),W.resetDefaultState(),I=-1,F=null,v.pop(),v.length>0?(A=v[v.length-1],ee===!0&&N.setGlobalState(y.clippingPlanes,A.state.camera)):A=null,S.pop(),S.length>0?_=S[S.length-1]:_=null};function rc(D,Q,ae,ce){if(D.visible===!1)return;if(D.layers.test(Q.layers)){if(D.isGroup)ae=D.renderOrder;else if(D.isLOD)D.autoUpdate===!0&&D.update(Q);else if(D.isLight)A.pushLight(D),D.castShadow&&A.pushShadow(D);else if(D.isSprite){if(!D.frustumCulled||X.intersectsSprite(D)){ce&&Ce.setFromMatrixPosition(D.matrixWorld).applyMatrix4(be);const Be=le.update(D),ke=D.material;ke.visible&&_.push(D,Be,ke,ae,Ce.z,null)}}else if((D.isMesh||D.isLine||D.isPoints)&&(!D.frustumCulled||X.intersectsObject(D))){const Be=le.update(D),ke=D.material;if(ce&&(D.boundingSphere!==void 0?(D.boundingSphere===null&&D.computeBoundingSphere(),Ce.copy(D.boundingSphere.center)):(Be.boundingSphere===null&&Be.computeBoundingSphere(),Ce.copy(Be.boundingSphere.center)),Ce.applyMatrix4(D.matrixWorld).applyMatrix4(be)),Array.isArray(ke)){const Ne=Be.groups;for(let Xe=0,Ye=Ne.length;Xe<Ye;Xe++){const Ve=Ne[Xe],et=ke[Ve.materialIndex];et&&et.visible&&_.push(D,Be,et,ae,Ce.z,Ve)}}else ke.visible&&_.push(D,Be,ke,ae,Ce.z,null)}}const Me=D.children;for(let Be=0,ke=Me.length;Be<ke;Be++)rc(Me[Be],Q,ae,ce)}function Vd(D,Q,ae,ce){const{opaque:Y,transmissive:Me,transparent:Be}=D;A.setupLightsView(ae),ee===!0&&N.setGlobalState(y.clippingPlanes,ae),ce&&Z.viewport(U.copy(ce)),Y.length>0&&Ta(Y,Q,ae),Me.length>0&&Ta(Me,Q,ae),Be.length>0&&Ta(Be,Q,ae),Z.buffers.depth.setTest(!0),Z.buffers.depth.setMask(!0),Z.buffers.color.setMask(!0),Z.setPolygonOffset(!1)}function Gd(D,Q,ae,ce){if((ae.isScene===!0?ae.overrideMaterial:null)!==null)return;A.state.transmissionRenderTarget[ce.id]===void 0&&(A.state.transmissionRenderTarget[ce.id]=new Us(1,1,{generateMipmaps:!0,type:ie.has("EXT_color_buffer_half_float")||ie.has("EXT_color_buffer_float")?pr:Ui,minFilter:sr,samples:4,stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:rt.workingColorSpace}));const Me=A.state.transmissionRenderTarget[ce.id],Be=ce.viewport||U;Me.setSize(Be.z*y.transmissionResolutionScale,Be.w*y.transmissionResolutionScale);const ke=y.getRenderTarget(),Ne=y.getActiveCubeFace(),Xe=y.getActiveMipmapLevel();y.setRenderTarget(Me),y.getClearColor(z),V=y.getClearAlpha(),V<1&&y.setClearColor(16777215,.5),y.clear(),L&&me.render(ae);const Ye=y.toneMapping;y.toneMapping=Ds;const Ve=ce.viewport;if(ce.viewport!==void 0&&(ce.viewport=void 0),A.setupLightsView(ce),ee===!0&&N.setGlobalState(y.clippingPlanes,ce),Ta(D,ae,ce),ge.updateMultisampleRenderTarget(Me),ge.updateRenderTargetMipmap(Me),ie.has("WEBGL_multisampled_render_to_texture")===!1){let et=!1;for(let ft=0,Lt=Q.length;ft<Lt;ft++){const Bt=Q[ft],{object:gt,geometry:Ge,material:wt,group:it}=Bt;if(wt.side===ti&&gt.layers.test(ce.layers)){const Ln=wt.side;wt.side=wn,wt.needsUpdate=!0,Wd(gt,ae,ce,Ge,wt,it),wt.side=Ln,wt.needsUpdate=!0,et=!0}}et===!0&&(ge.updateMultisampleRenderTarget(Me),ge.updateRenderTargetMipmap(Me))}y.setRenderTarget(ke,Ne,Xe),y.setClearColor(z,V),Ve!==void 0&&(ce.viewport=Ve),y.toneMapping=Ye}function Ta(D,Q,ae){const ce=Q.isScene===!0?Q.overrideMaterial:null;for(let Y=0,Me=D.length;Y<Me;Y++){const Be=D[Y],{object:ke,geometry:Ne,group:Xe}=Be;let Ye=Be.material;Ye.allowOverride===!0&&ce!==null&&(Ye=ce),ke.layers.test(ae.layers)&&Wd(ke,Q,ae,Ne,Ye,Xe)}}function Wd(D,Q,ae,ce,Y,Me){D.onBeforeRender(y,Q,ae,ce,Y,Me),D.modelViewMatrix.multiplyMatrices(ae.matrixWorldInverse,D.matrixWorld),D.normalMatrix.getNormalMatrix(D.modelViewMatrix),Y.onBeforeRender(y,Q,ae,ce,D,Me),Y.transparent===!0&&Y.side===ti&&Y.forceSinglePass===!1?(Y.side=wn,Y.needsUpdate=!0,y.renderBufferDirect(ae,Q,ce,Y,D,Me),Y.side=Bi,Y.needsUpdate=!0,y.renderBufferDirect(ae,Q,ce,Y,D,Me),Y.side=ti):y.renderBufferDirect(ae,Q,ce,Y,D,Me),D.onAfterRender(y,Q,ae,ce,Y,Me)}function Ea(D,Q,ae){Q.isScene!==!0&&(Q=P);const ce=ne.get(D),Y=A.state.lights,Me=A.state.shadowsArray,Be=Y.state.version,ke=j.getParameters(D,Y.state,Me,Q,ae),Ne=j.getProgramCacheKey(ke);let Xe=ce.programs;ce.environment=D.isMeshStandardMaterial?Q.environment:null,ce.fog=Q.fog,ce.envMap=(D.isMeshStandardMaterial?T:R).get(D.envMap||ce.environment),ce.envMapRotation=ce.environment!==null&&D.envMap===null?Q.environmentRotation:D.envMapRotation,Xe===void 0&&(D.addEventListener("dispose",We),Xe=new Map,ce.programs=Xe);let Ye=Xe.get(Ne);if(Ye!==void 0){if(ce.currentProgram===Ye&&ce.lightsStateVersion===Be)return qd(D,ke),Ye}else ke.uniforms=j.getUniforms(D),D.onBeforeCompile(ke,y),Ye=j.acquireProgram(ke,Ne),Xe.set(Ne,Ye),ce.uniforms=ke.uniforms;const Ve=ce.uniforms;return(!D.isShaderMaterial&&!D.isRawShaderMaterial||D.clipping===!0)&&(Ve.clippingPlanes=N.uniform),qd(D,ke),ce.needsLights=ux(D),ce.lightsStateVersion=Be,ce.needsLights&&(Ve.ambientLightColor.value=Y.state.ambient,Ve.lightProbe.value=Y.state.probe,Ve.directionalLights.value=Y.state.directional,Ve.directionalLightShadows.value=Y.state.directionalShadow,Ve.spotLights.value=Y.state.spot,Ve.spotLightShadows.value=Y.state.spotShadow,Ve.rectAreaLights.value=Y.state.rectArea,Ve.ltc_1.value=Y.state.rectAreaLTC1,Ve.ltc_2.value=Y.state.rectAreaLTC2,Ve.pointLights.value=Y.state.point,Ve.pointLightShadows.value=Y.state.pointShadow,Ve.hemisphereLights.value=Y.state.hemi,Ve.directionalShadowMap.value=Y.state.directionalShadowMap,Ve.directionalShadowMatrix.value=Y.state.directionalShadowMatrix,Ve.spotShadowMap.value=Y.state.spotShadowMap,Ve.spotLightMatrix.value=Y.state.spotLightMatrix,Ve.spotLightMap.value=Y.state.spotLightMap,Ve.pointShadowMap.value=Y.state.pointShadowMap,Ve.pointShadowMatrix.value=Y.state.pointShadowMatrix),ce.currentProgram=Ye,ce.uniformsList=null,Ye}function Xd(D){if(D.uniformsList===null){const Q=D.currentProgram.getUniforms();D.uniformsList=xl.seqWithValue(Q.seq,D.uniforms)}return D.uniformsList}function qd(D,Q){const ae=ne.get(D);ae.outputColorSpace=Q.outputColorSpace,ae.batching=Q.batching,ae.batchingColor=Q.batchingColor,ae.instancing=Q.instancing,ae.instancingColor=Q.instancingColor,ae.instancingMorph=Q.instancingMorph,ae.skinning=Q.skinning,ae.morphTargets=Q.morphTargets,ae.morphNormals=Q.morphNormals,ae.morphColors=Q.morphColors,ae.morphTargetsCount=Q.morphTargetsCount,ae.numClippingPlanes=Q.numClippingPlanes,ae.numIntersection=Q.numClipIntersection,ae.vertexAlphas=Q.vertexAlphas,ae.vertexTangents=Q.vertexTangents,ae.toneMapping=Q.toneMapping}function lx(D,Q,ae,ce,Y){Q.isScene!==!0&&(Q=P),ge.resetTextureUnits();const Me=Q.fog,Be=ce.isMeshStandardMaterial?Q.environment:null,ke=C===null?y.outputColorSpace:C.isXRRenderTarget===!0?C.texture.colorSpace:fo,Ne=(ce.isMeshStandardMaterial?T:R).get(ce.envMap||Be),Xe=ce.vertexColors===!0&&!!ae.attributes.color&&ae.attributes.color.itemSize===4,Ye=!!ae.attributes.tangent&&(!!ce.normalMap||ce.anisotropy>0),Ve=!!ae.morphAttributes.position,et=!!ae.morphAttributes.normal,ft=!!ae.morphAttributes.color;let Lt=Ds;ce.toneMapped&&(C===null||C.isXRRenderTarget===!0)&&(Lt=y.toneMapping);const Bt=ae.morphAttributes.position||ae.morphAttributes.normal||ae.morphAttributes.color,gt=Bt!==void 0?Bt.length:0,Ge=ne.get(ce),wt=A.state.lights;if(ee===!0&&(pe===!0||D!==F)){const cn=D===F&&ce.id===I;N.setState(ce,D,cn)}let it=!1;ce.version===Ge.__version?(Ge.needsLights&&Ge.lightsStateVersion!==wt.state.version||Ge.outputColorSpace!==ke||Y.isBatchedMesh&&Ge.batching===!1||!Y.isBatchedMesh&&Ge.batching===!0||Y.isBatchedMesh&&Ge.batchingColor===!0&&Y.colorTexture===null||Y.isBatchedMesh&&Ge.batchingColor===!1&&Y.colorTexture!==null||Y.isInstancedMesh&&Ge.instancing===!1||!Y.isInstancedMesh&&Ge.instancing===!0||Y.isSkinnedMesh&&Ge.skinning===!1||!Y.isSkinnedMesh&&Ge.skinning===!0||Y.isInstancedMesh&&Ge.instancingColor===!0&&Y.instanceColor===null||Y.isInstancedMesh&&Ge.instancingColor===!1&&Y.instanceColor!==null||Y.isInstancedMesh&&Ge.instancingMorph===!0&&Y.morphTexture===null||Y.isInstancedMesh&&Ge.instancingMorph===!1&&Y.morphTexture!==null||Ge.envMap!==Ne||ce.fog===!0&&Ge.fog!==Me||Ge.numClippingPlanes!==void 0&&(Ge.numClippingPlanes!==N.numPlanes||Ge.numIntersection!==N.numIntersection)||Ge.vertexAlphas!==Xe||Ge.vertexTangents!==Ye||Ge.morphTargets!==Ve||Ge.morphNormals!==et||Ge.morphColors!==ft||Ge.toneMapping!==Lt||Ge.morphTargetsCount!==gt)&&(it=!0):(it=!0,Ge.__version=ce.version);let Ln=Ge.currentProgram;it===!0&&(Ln=Ea(ce,Q,Y));let xr=!1,Bn=!1,Mo=!1;const Rt=Ln.getUniforms(),Sn=Ge.uniforms;if(Z.useProgram(Ln.program)&&(xr=!0,Bn=!0,Mo=!0),ce.id!==I&&(I=ce.id,Bn=!0),xr||F!==D){Z.buffers.depth.getReversed()&&D.reversedDepth!==!0&&(D._reversedDepth=!0,D.updateProjectionMatrix()),Rt.setValue(w,"projectionMatrix",D.projectionMatrix),Rt.setValue(w,"viewMatrix",D.matrixWorldInverse);const vn=Rt.map.cameraPosition;vn!==void 0&&vn.setValue(w,xe.setFromMatrixPosition(D.matrixWorld)),ue.logarithmicDepthBuffer&&Rt.setValue(w,"logDepthBufFC",2/(Math.log(D.far+1)/Math.LN2)),(ce.isMeshPhongMaterial||ce.isMeshToonMaterial||ce.isMeshLambertMaterial||ce.isMeshBasicMaterial||ce.isMeshStandardMaterial||ce.isShaderMaterial)&&Rt.setValue(w,"isOrthographic",D.isOrthographicCamera===!0),F!==D&&(F=D,Bn=!0,Mo=!0)}if(Y.isSkinnedMesh){Rt.setOptional(w,Y,"bindMatrix"),Rt.setOptional(w,Y,"bindMatrixInverse");const cn=Y.skeleton;cn&&(cn.boneTexture===null&&cn.computeBoneTexture(),Rt.setValue(w,"boneTexture",cn.boneTexture,ge))}Y.isBatchedMesh&&(Rt.setOptional(w,Y,"batchingTexture"),Rt.setValue(w,"batchingTexture",Y._matricesTexture,ge),Rt.setOptional(w,Y,"batchingIdTexture"),Rt.setValue(w,"batchingIdTexture",Y._indirectTexture,ge),Rt.setOptional(w,Y,"batchingColorTexture"),Y._colorsTexture!==null&&Rt.setValue(w,"batchingColorTexture",Y._colorsTexture,ge));const jn=ae.morphAttributes;if((jn.position!==void 0||jn.normal!==void 0||jn.color!==void 0)&&Te.update(Y,ae,Ln),(Bn||Ge.receiveShadow!==Y.receiveShadow)&&(Ge.receiveShadow=Y.receiveShadow,Rt.setValue(w,"receiveShadow",Y.receiveShadow)),ce.isMeshGouraudMaterial&&ce.envMap!==null&&(Sn.envMap.value=Ne,Sn.flipEnvMap.value=Ne.isCubeTexture&&Ne.isRenderTargetTexture===!1?-1:1),ce.isMeshStandardMaterial&&ce.envMap===null&&Q.environment!==null&&(Sn.envMapIntensity.value=Q.environmentIntensity),Sn.dfgLUT!==void 0&&(Sn.dfgLUT.value=iT()),Bn&&(Rt.setValue(w,"toneMappingExposure",y.toneMappingExposure),Ge.needsLights&&cx(Sn,Mo),Me&&ce.fog===!0&&De.refreshFogUniforms(Sn,Me),De.refreshMaterialUniforms(Sn,ce,oe,$,A.state.transmissionRenderTarget[D.id]),xl.upload(w,Xd(Ge),Sn,ge)),ce.isShaderMaterial&&ce.uniformsNeedUpdate===!0&&(xl.upload(w,Xd(Ge),Sn,ge),ce.uniformsNeedUpdate=!1),ce.isSpriteMaterial&&Rt.setValue(w,"center",Y.center),Rt.setValue(w,"modelViewMatrix",Y.modelViewMatrix),Rt.setValue(w,"normalMatrix",Y.normalMatrix),Rt.setValue(w,"modelMatrix",Y.matrixWorld),ce.isShaderMaterial||ce.isRawShaderMaterial){const cn=ce.uniformsGroups;for(let vn=0,oc=cn.length;vn<oc;vn++){const Gs=cn[vn];Ie.update(Gs,Ln),Ie.bind(Gs,Ln)}}return Ln}function cx(D,Q){D.ambientLightColor.needsUpdate=Q,D.lightProbe.needsUpdate=Q,D.directionalLights.needsUpdate=Q,D.directionalLightShadows.needsUpdate=Q,D.pointLights.needsUpdate=Q,D.pointLightShadows.needsUpdate=Q,D.spotLights.needsUpdate=Q,D.spotLightShadows.needsUpdate=Q,D.rectAreaLights.needsUpdate=Q,D.hemisphereLights.needsUpdate=Q}function ux(D){return D.isMeshLambertMaterial||D.isMeshToonMaterial||D.isMeshPhongMaterial||D.isMeshStandardMaterial||D.isShadowMaterial||D.isShaderMaterial&&D.lights===!0}this.getActiveCubeFace=function(){return E},this.getActiveMipmapLevel=function(){return b},this.getRenderTarget=function(){return C},this.setRenderTargetTextures=function(D,Q,ae){const ce=ne.get(D);ce.__autoAllocateDepthBuffer=D.resolveDepthBuffer===!1,ce.__autoAllocateDepthBuffer===!1&&(ce.__useRenderToTexture=!1),ne.get(D.texture).__webglTexture=Q,ne.get(D.depthTexture).__webglTexture=ce.__autoAllocateDepthBuffer?void 0:ae,ce.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(D,Q){const ae=ne.get(D);ae.__webglFramebuffer=Q,ae.__useDefaultFramebuffer=Q===void 0};const fx=w.createFramebuffer();this.setRenderTarget=function(D,Q=0,ae=0){C=D,E=Q,b=ae;let ce=!0,Y=null,Me=!1,Be=!1;if(D){const Ne=ne.get(D);if(Ne.__useDefaultFramebuffer!==void 0)Z.bindFramebuffer(w.FRAMEBUFFER,null),ce=!1;else if(Ne.__webglFramebuffer===void 0)ge.setupRenderTarget(D);else if(Ne.__hasExternalTextures)ge.rebindTextures(D,ne.get(D.texture).__webglTexture,ne.get(D.depthTexture).__webglTexture);else if(D.depthBuffer){const Ve=D.depthTexture;if(Ne.__boundDepthTexture!==Ve){if(Ve!==null&&ne.has(Ve)&&(D.width!==Ve.image.width||D.height!==Ve.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");ge.setupDepthRenderbuffer(D)}}const Xe=D.texture;(Xe.isData3DTexture||Xe.isDataArrayTexture||Xe.isCompressedArrayTexture)&&(Be=!0);const Ye=ne.get(D).__webglFramebuffer;D.isWebGLCubeRenderTarget?(Array.isArray(Ye[Q])?Y=Ye[Q][ae]:Y=Ye[Q],Me=!0):D.samples>0&&ge.useMultisampledRTT(D)===!1?Y=ne.get(D).__webglMultisampledFramebuffer:Array.isArray(Ye)?Y=Ye[ae]:Y=Ye,U.copy(D.viewport),O.copy(D.scissor),k=D.scissorTest}else U.copy(Le).multiplyScalar(oe).floor(),O.copy(fe).multiplyScalar(oe).floor(),k=re;if(ae!==0&&(Y=fx),Z.bindFramebuffer(w.FRAMEBUFFER,Y)&&ce&&Z.drawBuffers(D,Y),Z.viewport(U),Z.scissor(O),Z.setScissorTest(k),Me){const Ne=ne.get(D.texture);w.framebufferTexture2D(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_CUBE_MAP_POSITIVE_X+Q,Ne.__webglTexture,ae)}else if(Be){const Ne=Q;for(let Xe=0;Xe<D.textures.length;Xe++){const Ye=ne.get(D.textures[Xe]);w.framebufferTextureLayer(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0+Xe,Ye.__webglTexture,ae,Ne)}}else if(D!==null&&ae!==0){const Ne=ne.get(D.texture);w.framebufferTexture2D(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,Ne.__webglTexture,ae)}I=-1},this.readRenderTargetPixels=function(D,Q,ae,ce,Y,Me,Be,ke=0){if(!(D&&D.isWebGLRenderTarget)){zt("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let Ne=ne.get(D).__webglFramebuffer;if(D.isWebGLCubeRenderTarget&&Be!==void 0&&(Ne=Ne[Be]),Ne){Z.bindFramebuffer(w.FRAMEBUFFER,Ne);try{const Xe=D.textures[ke],Ye=Xe.format,Ve=Xe.type;if(!ue.textureFormatReadable(Ye)){zt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!ue.textureTypeReadable(Ve)){zt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}Q>=0&&Q<=D.width-ce&&ae>=0&&ae<=D.height-Y&&(D.textures.length>1&&w.readBuffer(w.COLOR_ATTACHMENT0+ke),w.readPixels(Q,ae,ce,Y,He.convert(Ye),He.convert(Ve),Me))}finally{const Xe=C!==null?ne.get(C).__webglFramebuffer:null;Z.bindFramebuffer(w.FRAMEBUFFER,Xe)}}},this.readRenderTargetPixelsAsync=async function(D,Q,ae,ce,Y,Me,Be,ke=0){if(!(D&&D.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let Ne=ne.get(D).__webglFramebuffer;if(D.isWebGLCubeRenderTarget&&Be!==void 0&&(Ne=Ne[Be]),Ne)if(Q>=0&&Q<=D.width-ce&&ae>=0&&ae<=D.height-Y){Z.bindFramebuffer(w.FRAMEBUFFER,Ne);const Xe=D.textures[ke],Ye=Xe.format,Ve=Xe.type;if(!ue.textureFormatReadable(Ye))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!ue.textureTypeReadable(Ve))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const et=w.createBuffer();w.bindBuffer(w.PIXEL_PACK_BUFFER,et),w.bufferData(w.PIXEL_PACK_BUFFER,Me.byteLength,w.STREAM_READ),D.textures.length>1&&w.readBuffer(w.COLOR_ATTACHMENT0+ke),w.readPixels(Q,ae,ce,Y,He.convert(Ye),He.convert(Ve),0);const ft=C!==null?ne.get(C).__webglFramebuffer:null;Z.bindFramebuffer(w.FRAMEBUFFER,ft);const Lt=w.fenceSync(w.SYNC_GPU_COMMANDS_COMPLETE,0);return w.flush(),await FS(w,Lt,4),w.bindBuffer(w.PIXEL_PACK_BUFFER,et),w.getBufferSubData(w.PIXEL_PACK_BUFFER,0,Me),w.deleteBuffer(et),w.deleteSync(Lt),Me}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(D,Q=null,ae=0){const ce=Math.pow(2,-ae),Y=Math.floor(D.image.width*ce),Me=Math.floor(D.image.height*ce),Be=Q!==null?Q.x:0,ke=Q!==null?Q.y:0;ge.setTexture2D(D,0),w.copyTexSubImage2D(w.TEXTURE_2D,ae,0,0,Be,ke,Y,Me),Z.unbindTexture()};const dx=w.createFramebuffer(),hx=w.createFramebuffer();this.copyTextureToTexture=function(D,Q,ae=null,ce=null,Y=0,Me=null){Me===null&&(Y!==0?(ca("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."),Me=Y,Y=0):Me=0);let Be,ke,Ne,Xe,Ye,Ve,et,ft,Lt;const Bt=D.isCompressedTexture?D.mipmaps[Me]:D.image;if(ae!==null)Be=ae.max.x-ae.min.x,ke=ae.max.y-ae.min.y,Ne=ae.isBox3?ae.max.z-ae.min.z:1,Xe=ae.min.x,Ye=ae.min.y,Ve=ae.isBox3?ae.min.z:0;else{const jn=Math.pow(2,-Y);Be=Math.floor(Bt.width*jn),ke=Math.floor(Bt.height*jn),D.isDataArrayTexture?Ne=Bt.depth:D.isData3DTexture?Ne=Math.floor(Bt.depth*jn):Ne=1,Xe=0,Ye=0,Ve=0}ce!==null?(et=ce.x,ft=ce.y,Lt=ce.z):(et=0,ft=0,Lt=0);const gt=He.convert(Q.format),Ge=He.convert(Q.type);let wt;Q.isData3DTexture?(ge.setTexture3D(Q,0),wt=w.TEXTURE_3D):Q.isDataArrayTexture||Q.isCompressedArrayTexture?(ge.setTexture2DArray(Q,0),wt=w.TEXTURE_2D_ARRAY):(ge.setTexture2D(Q,0),wt=w.TEXTURE_2D),w.pixelStorei(w.UNPACK_FLIP_Y_WEBGL,Q.flipY),w.pixelStorei(w.UNPACK_PREMULTIPLY_ALPHA_WEBGL,Q.premultiplyAlpha),w.pixelStorei(w.UNPACK_ALIGNMENT,Q.unpackAlignment);const it=w.getParameter(w.UNPACK_ROW_LENGTH),Ln=w.getParameter(w.UNPACK_IMAGE_HEIGHT),xr=w.getParameter(w.UNPACK_SKIP_PIXELS),Bn=w.getParameter(w.UNPACK_SKIP_ROWS),Mo=w.getParameter(w.UNPACK_SKIP_IMAGES);w.pixelStorei(w.UNPACK_ROW_LENGTH,Bt.width),w.pixelStorei(w.UNPACK_IMAGE_HEIGHT,Bt.height),w.pixelStorei(w.UNPACK_SKIP_PIXELS,Xe),w.pixelStorei(w.UNPACK_SKIP_ROWS,Ye),w.pixelStorei(w.UNPACK_SKIP_IMAGES,Ve);const Rt=D.isDataArrayTexture||D.isData3DTexture,Sn=Q.isDataArrayTexture||Q.isData3DTexture;if(D.isDepthTexture){const jn=ne.get(D),cn=ne.get(Q),vn=ne.get(jn.__renderTarget),oc=ne.get(cn.__renderTarget);Z.bindFramebuffer(w.READ_FRAMEBUFFER,vn.__webglFramebuffer),Z.bindFramebuffer(w.DRAW_FRAMEBUFFER,oc.__webglFramebuffer);for(let Gs=0;Gs<Ne;Gs++)Rt&&(w.framebufferTextureLayer(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,ne.get(D).__webglTexture,Y,Ve+Gs),w.framebufferTextureLayer(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,ne.get(Q).__webglTexture,Me,Lt+Gs)),w.blitFramebuffer(Xe,Ye,Be,ke,et,ft,Be,ke,w.DEPTH_BUFFER_BIT,w.NEAREST);Z.bindFramebuffer(w.READ_FRAMEBUFFER,null),Z.bindFramebuffer(w.DRAW_FRAMEBUFFER,null)}else if(Y!==0||D.isRenderTargetTexture||ne.has(D)){const jn=ne.get(D),cn=ne.get(Q);Z.bindFramebuffer(w.READ_FRAMEBUFFER,dx),Z.bindFramebuffer(w.DRAW_FRAMEBUFFER,hx);for(let vn=0;vn<Ne;vn++)Rt?w.framebufferTextureLayer(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,jn.__webglTexture,Y,Ve+vn):w.framebufferTexture2D(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,jn.__webglTexture,Y),Sn?w.framebufferTextureLayer(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,cn.__webglTexture,Me,Lt+vn):w.framebufferTexture2D(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,cn.__webglTexture,Me),Y!==0?w.blitFramebuffer(Xe,Ye,Be,ke,et,ft,Be,ke,w.COLOR_BUFFER_BIT,w.NEAREST):Sn?w.copyTexSubImage3D(wt,Me,et,ft,Lt+vn,Xe,Ye,Be,ke):w.copyTexSubImage2D(wt,Me,et,ft,Xe,Ye,Be,ke);Z.bindFramebuffer(w.READ_FRAMEBUFFER,null),Z.bindFramebuffer(w.DRAW_FRAMEBUFFER,null)}else Sn?D.isDataTexture||D.isData3DTexture?w.texSubImage3D(wt,Me,et,ft,Lt,Be,ke,Ne,gt,Ge,Bt.data):Q.isCompressedArrayTexture?w.compressedTexSubImage3D(wt,Me,et,ft,Lt,Be,ke,Ne,gt,Bt.data):w.texSubImage3D(wt,Me,et,ft,Lt,Be,ke,Ne,gt,Ge,Bt):D.isDataTexture?w.texSubImage2D(w.TEXTURE_2D,Me,et,ft,Be,ke,gt,Ge,Bt.data):D.isCompressedTexture?w.compressedTexSubImage2D(w.TEXTURE_2D,Me,et,ft,Bt.width,Bt.height,gt,Bt.data):w.texSubImage2D(w.TEXTURE_2D,Me,et,ft,Be,ke,gt,Ge,Bt);w.pixelStorei(w.UNPACK_ROW_LENGTH,it),w.pixelStorei(w.UNPACK_IMAGE_HEIGHT,Ln),w.pixelStorei(w.UNPACK_SKIP_PIXELS,xr),w.pixelStorei(w.UNPACK_SKIP_ROWS,Bn),w.pixelStorei(w.UNPACK_SKIP_IMAGES,Mo),Me===0&&Q.generateMipmaps&&w.generateMipmap(wt),Z.unbindTexture()},this.initRenderTarget=function(D){ne.get(D).__webglFramebuffer===void 0&&ge.setupRenderTarget(D)},this.initTexture=function(D){D.isCubeTexture?ge.setTextureCube(D,0):D.isData3DTexture?ge.setTexture3D(D,0):D.isDataArrayTexture||D.isCompressedArrayTexture?ge.setTexture2DArray(D,0):ge.setTexture2D(D,0),Z.unbindTexture()},this.resetState=function(){E=0,b=0,C=null,Z.reset(),W.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Ei}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const t=this.getContext();t.drawingBufferColorSpace=rt._getDrawingBufferColorSpace(e),t.unpackColorSpace=rt._getUnpackColorSpace()}}class Cs{static idGen=0;constructor(e,t){let n,s;this.promise=new Promise((c,u)=>{n=c,s=u});const r=n.bind(this),o=s.bind(this),a=(...c)=>{r(...c)},l=c=>{o(c)};e(a.bind(this),l.bind(this)),this.abortHandler=t,this.id=Cs.idGen++}then(e){return new Cs((t,n)=>{this.promise=this.promise.then((...s)=>{const r=e(...s);r instanceof Promise||r instanceof Cs?r.then((...o)=>{t(...o)}):t(r)}).catch(s=>{n(s)})},this.abortHandler)}catch(e){return new Cs(t=>{this.promise=this.promise.then((...n)=>{t(...n)}).catch(e)},this.abortHandler)}abort(e){this.abortHandler&&this.abortHandler(e)}}class eg extends Error{constructor(e){super(e)}}(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){i[0]=t;const n=e[0];let s=n>>16&32768,r=n>>12&2047;const o=n>>23&255;return o<103?s:o>142?(s|=31744,s|=(o==255?0:1)&&n&8388607,s):o<113?(r|=2048,s|=(r>>114-o)+(r>>113-o&1),s):(s|=o-112<<10|r>>1,s+=r&1,s)}})();const Yc=(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){return i[0]=t,e[0]}})(),rT=function(i,e){return i[e]+(i[e+1]<<8)+(i[e+2]<<16)+(i[e+3]<<24)},tc=function(i,e,t=!0,n){const s=new AbortController,r=s.signal;let o=!1;const a=u=>{s.abort(u),o=!0};let l=!1;const c=(u,f,d,h)=>{e&&!l&&(e(u,f,d,h),u===100&&(l=!0))};return new Cs((u,f)=>{const d={signal:r};n&&(d.headers=n),fetch(i,d).then(async h=>{if(!h.ok){const A=await h.text();f(new Error(`Fetch failed: ${h.status} ${h.statusText} ${A}`));return}const x=h.body.getReader();let p=0,g=h.headers.get("Content-Length"),m=g?parseInt(g):void 0;const _=[];for(;!o;)try{const{value:A,done:S}=await x.read();if(S){if(c(100,"100%",A,m),t){const M=new Blob(_).arrayBuffer();u(M)}else u();break}p+=A.length;let v,y;m!==void 0&&(v=p/m*100,y=`${v.toFixed(2)}%`),t&&_.push(A),c(v,y,A,m)}catch(A){f(A);return}}).catch(h=>{f(new eg(h))})},a)},Ct=function(i,e,t){return Math.max(Math.min(i,t),e)},Ur=function(){return performance.now()/1e3},Hr=i=>{if(i.geometry&&(i.geometry.dispose(),i.geometry=null),i.material&&(i.material.dispose(),i.material=null),i.children)for(let e of i.children)Hr(e)},Gn=(i,e)=>new Promise(t=>{window.setTimeout(()=>{t(i?i():void 0)},e?1:50)}),Jr=(i=0)=>{let e=0;if(i===1)e=9;else if(i===2)e=24;else if(i===3)e=45;else if(i>3)throw new Error("getSphericalHarmonicsComponentCountForDegree() -> Invalid spherical harmonics degree");return e},rd=()=>{let i,e;return{promise:new Promise((n,s)=>{i=n,e=s}),resolve:i,reject:e}},Kc=i=>{let e,t;return i||(i=()=>{}),{promise:new Cs((s,r)=>{e=s,t=r},i),resolve:e,reject:t}};class oT{constructor(e,t,n){this.major=e,this.minor=t,this.patch=n}toString(){return`${this.major}_${this.minor}_${this.patch}`}}function od(){const i=navigator.userAgent;return i.indexOf("iPhone")>0||i.indexOf("iPad")>0}function tg(){if(od()){const i=navigator.userAgent.match(/OS (\d+)_(\d+)_?(\d+)?/);return new oT(parseInt(i[1]||0,10),parseInt(i[2]||0,10),parseInt(i[3]||0,10))}else return null}const aT=14;class Ee{static OFFSET={X:0,Y:1,Z:2,SCALE0:3,SCALE1:4,SCALE2:5,ROTATION0:6,ROTATION1:7,ROTATION2:8,ROTATION3:9,FDC0:10,FDC1:11,FDC2:12,OPACITY:13,FRC0:14,FRC1:15,FRC2:16,FRC3:17,FRC4:18,FRC5:19,FRC6:20,FRC7:21,FRC8:22,FRC9:23,FRC10:24,FRC11:25,FRC12:26,FRC13:27,FRC14:28,FRC15:29,FRC16:30,FRC17:31,FRC18:32,FRC19:33,FRC20:34,FRC21:35,FRC22:36,FRC23:37};constructor(e=0){this.sphericalHarmonicsDegree=e,this.sphericalHarmonicsCount=Jr(this.sphericalHarmonicsDegree),this.componentCount=this.sphericalHarmonicsCount+aT,this.defaultSphericalHarmonics=new Array(this.sphericalHarmonicsCount).fill(0),this.splats=[],this.splatCount=0}static createSplat(e=0){const t=[0,0,0,1,1,1,1,0,0,0,0,0,0,0];let n=Jr(e);for(let s=0;s<n;s++)t.push(0);return t}addSplat(e){this.splats.push(e),this.splatCount++}getSplat(e){return this.splats[e]}addDefaultSplat(){const e=Ee.createSplat(this.sphericalHarmonicsDegree);return this.addSplat(e),e}addSplatFromComonents(e,t,n,s,r,o,a,l,c,u,f,d,h,x,...p){const g=[e,t,n,s,r,o,a,l,c,u,f,d,h,x,...this.defaultSphericalHarmonics];for(let m=0;m<p.length&&m<this.sphericalHarmonicsCount;m++)g[m]=p[m];return this.addSplat(g),g}addSplatFromArray(e,t){const n=e.splats[t],s=Ee.createSplat(this.sphericalHarmonicsDegree);for(let r=0;r<this.componentCount&&r<n.length;r++)s[r]=n[r];this.addSplat(s)}}class pt{static DefaultSplatSortDistanceMapPrecision=16;static MemoryPageSize=65536;static BytesPerFloat=4;static BytesPerInt=4;static MaxScenes=32;static ProgressiveLoadSectionSize=262144;static ProgressiveLoadSectionDelayDuration=15;static SphericalHarmonics8BitCompressionRange=3}const lT=pt.SphericalHarmonics8BitCompressionRange,_s=lT/2,qt=ua.toHalfFloat.bind(ua),ad=ua.fromHalfFloat.bind(ua),Mt=(i,e,t=!1,n,s)=>{if(e===0)return i;if(e===1||e===2&&!t)return ua.fromHalfFloat(i);if(e===2)return ld(i,n,s)},Xo=(i,e,t)=>{i=Ct(i,e,t);const n=t-e;return Ct(Math.floor((i-e)/n*255),0,255)},ld=(i,e,t)=>{const n=t-e;return i/255*n+e},ng=(i,e,t)=>Xo(ad(i,e,t)),cT=(i,e,t)=>qt(ld(i,e,t)),at=(i,e,t,n=!1)=>t===0?i.getFloat32(e*4,!0):t===1||t===2&&!n?i.getUint16(e*2,!0):i.getUint8(e,!0),uT=(function(){const i=e=>e;return function(e,t,n,s=!1){if(t===n)return e;let r=i;return t===2&&s?n===1?r=cT:n==0&&(r=ld):t===2||t===1?n===0?r=ad:n==2&&(s?r=ng:r=i):t===0&&(n===1?r=qt:n==2&&(s?r=Xo:r=qt)),r(e)}})(),Or=(i,e,t,n,s=0)=>{const r=new Uint8Array(i,e),o=new Uint8Array(t,n);for(let a=0;a<s;a++)o[a]=r[a]};class K{static CurrentMajorVersion=0;static CurrentMinorVersion=1;static CenterComponentCount=3;static ScaleComponentCount=3;static RotationComponentCount=4;static ColorComponentCount=4;static CovarianceComponentCount=6;static SplatScaleOffsetFloat=3;static SplatRotationOffsetFloat=6;static CompressionLevels={0:{BytesPerCenter:12,BytesPerScale:12,BytesPerRotation:16,BytesPerColor:4,ScaleOffsetBytes:12,RotationffsetBytes:24,ColorOffsetBytes:40,SphericalHarmonicsOffsetBytes:44,ScaleRange:1,BytesPerSphericalHarmonicsComponent:4,SphericalHarmonicsOffsetFloat:11,SphericalHarmonicsDegrees:{0:{BytesPerSplat:44},1:{BytesPerSplat:80},2:{BytesPerSplat:140}}},1:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:2,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:42},2:{BytesPerSplat:72}}},2:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:1,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:33},2:{BytesPerSplat:48}}}};static CovarianceSizeFloats=6;static HeaderSizeBytes=4096;static SectionHeaderSizeBytes=1024;static BucketStorageSizeBytes=12;static BucketStorageSizeFloats=3;static BucketBlockSize=5;static BucketSize=256;constructor(e,t=!0){this.constructFromBuffer(e,t)}getSplatCount(){return this.splatCount}getMaxSplatCount(){return this.maxSplatCount}getMinSphericalHarmonicsDegree(){let e=0;for(let t=0;t<this.sections.length;t++){const n=this.sections[t];(t===0||n.sphericalHarmonicsDegree<e)&&(e=n.sphericalHarmonicsDegree)}return e}getBucketIndex(e,t){let n;const s=e.fullBucketCount*e.bucketSize;if(t<s)n=Math.floor(t/e.bucketSize);else{let r=s;n=e.fullBucketCount;let o=0;for(;r<e.splatCount;){let a=e.partiallyFilledBucketLengths[o];if(t>=r&&t<r+a)break;r+=a,n++,o++}}return n}getSplatCenter(e,t,n){const s=this.globalSplatIndexToSectionMap[e],r=this.sections[s],o=e-r.splatCountOffset,a=r.bytesPerSplat*o,l=new DataView(this.bufferData,r.dataBase+a),c=at(l,0,this.compressionLevel),u=at(l,1,this.compressionLevel),f=at(l,2,this.compressionLevel);if(this.compressionLevel>=1){const h=this.getBucketIndex(r,o)*K.BucketStorageSizeFloats,x=r.compressionScaleFactor,p=r.compressionScaleRange;t.x=(c-p)*x+r.bucketArray[h],t.y=(u-p)*x+r.bucketArray[h+1],t.z=(f-p)*x+r.bucketArray[h+2]}else t.x=c,t.y=u,t.z=f;n&&t.applyMatrix4(n)}getSplatScaleAndRotation=(function(){const e=new qe,t=new qe,n=new qe,s=new B,r=new B,o=new bt;return function(a,l,c,u,f){const d=this.globalSplatIndexToSectionMap[a],h=this.sections[d],x=a-h.splatCountOffset,p=h.bytesPerSplat*x+K.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,g=new DataView(this.bufferData,h.dataBase+p);r.set(Mt(at(g,0,this.compressionLevel),this.compressionLevel),Mt(at(g,1,this.compressionLevel),this.compressionLevel),Mt(at(g,2,this.compressionLevel),this.compressionLevel)),f&&(f.x!==void 0&&(r.x=f.x),f.y!==void 0&&(r.y=f.y),f.z!==void 0&&(r.z=f.z)),o.set(Mt(at(g,4,this.compressionLevel),this.compressionLevel),Mt(at(g,5,this.compressionLevel),this.compressionLevel),Mt(at(g,6,this.compressionLevel),this.compressionLevel),Mt(at(g,3,this.compressionLevel),this.compressionLevel)),u?(e.makeScale(r.x,r.y,r.z),t.makeRotationFromQuaternion(o),n.copy(e).multiply(t).multiply(u),n.decompose(s,c,l)):(l.copy(r),c.copy(o))}})();getSplatColor(e,t){const n=this.globalSplatIndexToSectionMap[e],s=this.sections[n],r=e-s.splatCountOffset,o=s.bytesPerSplat*r+K.CompressionLevels[this.compressionLevel].ColorOffsetBytes,a=new Uint8Array(this.bufferData,s.dataBase+o,4);t.set(a[0],a[1],a[2],a[3])}fillSplatCenterArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);const a=new B;for(let l=n;l<=s;l++){const c=this.globalSplatIndexToSectionMap[l],u=this.sections[c],f=l-u.splatCountOffset,d=(l-n+r)*K.CenterComponentCount,h=u.bytesPerSplat*f,x=new DataView(this.bufferData,u.dataBase+h),p=at(x,0,this.compressionLevel),g=at(x,1,this.compressionLevel),m=at(x,2,this.compressionLevel);if(this.compressionLevel>=1){const A=this.getBucketIndex(u,f)*K.BucketStorageSizeFloats,S=u.compressionScaleFactor,v=u.compressionScaleRange;a.x=(p-v)*S+u.bucketArray[A],a.y=(g-v)*S+u.bucketArray[A+1],a.z=(m-v)*S+u.bucketArray[A+2]}else a.x=p,a.y=g,a.z=m;t&&a.applyMatrix4(t),e[d]=a.x,e[d+1]=a.y,e[d+2]=a.z}}fillSplatScaleRotationArray=(function(){const e=new qe,t=new qe,n=new qe,s=new B,r=new bt,o=new B,a=l=>{const c=l.w<0?-1:1;l.x*=c,l.y*=c,l.z*=c,l.w*=c};return function(l,c,u,f,d,h,x,p){const g=this.splatCount;f=f||0,d=d||g-1,h===void 0&&(h=f);const m=(_,A)=>uT(_,A,x);for(let _=f;_<=d;_++){const A=this.globalSplatIndexToSectionMap[_],S=this.sections[A],v=_-S.splatCountOffset,y=S.bytesPerSplat*v+K.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,M=(_-f+h)*K.ScaleComponentCount,E=(_-f+h)*K.RotationComponentCount,b=new DataView(this.bufferData,S.dataBase+y),C=p&&p.x!==void 0?p.x:at(b,0,this.compressionLevel),I=p&&p.y!==void 0?p.y:at(b,1,this.compressionLevel),F=p&&p.z!==void 0?p.z:at(b,2,this.compressionLevel),U=at(b,3,this.compressionLevel),O=at(b,4,this.compressionLevel),k=at(b,5,this.compressionLevel),z=at(b,6,this.compressionLevel);s.set(Mt(C,this.compressionLevel),Mt(I,this.compressionLevel),Mt(F,this.compressionLevel)),r.set(Mt(O,this.compressionLevel),Mt(k,this.compressionLevel),Mt(z,this.compressionLevel),Mt(U,this.compressionLevel)).normalize(),u&&(o.set(0,0,0),e.makeScale(s.x,s.y,s.z),t.makeRotationFromQuaternion(r),n.identity().premultiply(e).premultiply(t),n.premultiply(u),n.decompose(o,r,s),r.normalize()),a(r),l&&(l[M]=m(s.x,0),l[M+1]=m(s.y,0),l[M+2]=m(s.z,0)),c&&(c[E]=m(r.x,0),c[E+1]=m(r.y,0),c[E+2]=m(r.z,0),c[E+3]=m(r.w,0))}}})();static computeCovariance=(function(){const e=new qe,t=new Qe,n=new Qe,s=new Qe,r=new Qe,o=new Qe,a=new Qe;return function(l,c,u,f,d=0,h){e.makeScale(l.x,l.y,l.z),t.setFromMatrix4(e),e.makeRotationFromQuaternion(c),n.setFromMatrix4(e),s.copy(n).multiply(t),r.copy(s).transpose().premultiply(s),u&&(o.setFromMatrix4(u),a.copy(o).transpose(),r.multiply(a),r.premultiply(o)),h>=1?(f[d]=qt(r.elements[0]),f[d+1]=qt(r.elements[3]),f[d+2]=qt(r.elements[6]),f[d+3]=qt(r.elements[4]),f[d+4]=qt(r.elements[7]),f[d+5]=qt(r.elements[8])):(f[d]=r.elements[0],f[d+1]=r.elements[3],f[d+2]=r.elements[6],f[d+3]=r.elements[4],f[d+4]=r.elements[7],f[d+5]=r.elements[8])}})();fillSplatCovarianceArray(e,t,n,s,r,o){const a=this.splatCount,l=new B,c=new bt;n=n||0,s=s||a-1,r===void 0&&(r=n);for(let u=n;u<=s;u++){const f=this.globalSplatIndexToSectionMap[u],d=this.sections[f],h=u-d.splatCountOffset,x=(u-n+r)*K.CovarianceComponentCount,p=d.bytesPerSplat*h+K.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,g=new DataView(this.bufferData,d.dataBase+p);l.set(Mt(at(g,0,this.compressionLevel),this.compressionLevel),Mt(at(g,1,this.compressionLevel),this.compressionLevel),Mt(at(g,2,this.compressionLevel),this.compressionLevel)),c.set(Mt(at(g,4,this.compressionLevel),this.compressionLevel),Mt(at(g,5,this.compressionLevel),this.compressionLevel),Mt(at(g,6,this.compressionLevel),this.compressionLevel),Mt(at(g,3,this.compressionLevel),this.compressionLevel)),K.computeCovariance(l,c,t,e,x,o)}}fillSplatColorArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);for(let a=n;a<=s;a++){const l=this.globalSplatIndexToSectionMap[a],c=this.sections[l],u=a-c.splatCountOffset,f=(a-n+r)*K.ColorComponentCount,d=c.bytesPerSplat*u+K.CompressionLevels[this.compressionLevel].ColorOffsetBytes,h=new Uint8Array(this.bufferData,c.dataBase+d);let x=h[3];x=x>=t?x:0,e[f]=h[0],e[f+1]=h[1],e[f+2]=h[2],e[f+3]=x}}fillSphericalHarmonicsArray=(function(){for(let O=0;O<15;O++)new B;const e=new Qe,t=new qe,n=new B,s=new B,r=new bt,o=[],a=[],l=[],c=[],u=[],f=[],d=[],h=[],x=[],p=[],g=[],m=[],_=[],A=[],S=[],v=[],y=[],M=[],E=O=>O,b=(O,k,z,V)=>{O[0]=k,O[1]=z,O[2]=V},C=(O,k,z,V,H)=>{O[0]=at(k,V,H,!0),O[1]=at(k,V+z,H,!0),O[2]=at(k,V+z+z,H,!0)},I=(O,k)=>{k[0]=O[0],k[1]=O[1],k[2]=O[2]},F=(O,k,z,V)=>{k[z]=V(O[0]),k[z+1]=V(O[1]),k[z+2]=V(O[2])},U=(O,k,z,V,H)=>(k[0]=Mt(O[0],z,!0,V,H),k[1]=Mt(O[1],z,!0,V,H),k[2]=Mt(O[2],z,!0,V,H),k);return function(O,k,z,V,H,$,oe){const Se=this.splatCount;V=V||0,H=H||Se-1,$===void 0&&($=V),z&&k>=1&&(t.copy(z),t.decompose(n,r,s),r.normalize(),t.makeRotationFromQuaternion(r),e.setFromMatrix4(t),b(o,e.elements[4],-e.elements[7],e.elements[1]),b(a,-e.elements[5],e.elements[8],-e.elements[2]),b(l,e.elements[3],-e.elements[6],e.elements[0]));const we=fe=>ng(fe,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff),Le=fe=>Xo(fe,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff);for(let fe=V;fe<=H;fe++){const re=this.globalSplatIndexToSectionMap[fe],X=this.sections[re];k=Math.min(k,X.sphericalHarmonicsDegree);const ee=Jr(k),pe=fe-X.splatCountOffset,be=X.bytesPerSplat*pe+K.CompressionLevels[this.compressionLevel].SphericalHarmonicsOffsetBytes,xe=new DataView(this.bufferData,X.dataBase+be),Ce=(fe-V+$)*ee;let P=z?0:this.compressionLevel,L=E;P!==oe&&(P===1?oe===0?L=ad:oe==2&&(L=we):P===0&&(oe===1?L=qt:oe==2&&(L=Le)));const q=this.minSphericalHarmonicsCoeff,w=this.maxSphericalHarmonicsCoeff;k>=1&&(C(x,xe,3,0,this.compressionLevel),C(p,xe,3,1,this.compressionLevel),C(g,xe,3,2,this.compressionLevel),z?(U(x,x,this.compressionLevel,q,w),U(p,p,this.compressionLevel,q,w),U(g,g,this.compressionLevel,q,w),K.rotateSphericalHarmonics3(x,p,g,o,a,l,A,S,v)):(I(x,A),I(p,S),I(g,v)),F(A,O,Ce,L),F(S,O,Ce+3,L),F(v,O,Ce+6,L),k>=2&&(C(x,xe,5,9,this.compressionLevel),C(p,xe,5,10,this.compressionLevel),C(g,xe,5,11,this.compressionLevel),C(m,xe,5,12,this.compressionLevel),C(_,xe,5,13,this.compressionLevel),z?(U(x,x,this.compressionLevel,q,w),U(p,p,this.compressionLevel,q,w),U(g,g,this.compressionLevel,q,w),U(m,m,this.compressionLevel,q,w),U(_,_,this.compressionLevel,q,w),K.rotateSphericalHarmonics5(x,p,g,m,_,o,a,l,c,u,f,d,h,A,S,v,y,M)):(I(x,A),I(p,S),I(g,v),I(m,y),I(_,M)),F(A,O,Ce+9,L),F(S,O,Ce+12,L),F(v,O,Ce+15,L),F(y,O,Ce+18,L),F(M,O,Ce+21,L)))}}})();static dot3=(e,t,n,s,r)=>{r[0]=r[1]=r[2]=0;const o=s[0],a=s[1],l=s[2];K.addInto3(e[0]*o,e[1]*o,e[2]*o,r),K.addInto3(t[0]*a,t[1]*a,t[2]*a,r),K.addInto3(n[0]*l,n[1]*l,n[2]*l,r)};static addInto3=(e,t,n,s)=>{s[0]=s[0]+e,s[1]=s[1]+t,s[2]=s[2]+n};static dot5=(e,t,n,s,r,o,a)=>{a[0]=a[1]=a[2]=0;const l=o[0],c=o[1],u=o[2],f=o[3],d=o[4];K.addInto3(e[0]*l,e[1]*l,e[2]*l,a),K.addInto3(t[0]*c,t[1]*c,t[2]*c,a),K.addInto3(n[0]*u,n[1]*u,n[2]*u,a),K.addInto3(s[0]*f,s[1]*f,s[2]*f,a),K.addInto3(r[0]*d,r[1]*d,r[2]*d,a)};static rotateSphericalHarmonics3=(e,t,n,s,r,o,a,l,c)=>{K.dot3(e,t,n,s,a),K.dot3(e,t,n,r,l),K.dot3(e,t,n,o,c)};static rotateSphericalHarmonics5=(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g,m,_)=>{const A=Math.sqrt(.25),S=Math.sqrt(3/4),v=Math.sqrt(1/3),y=Math.sqrt(4/3),M=Math.sqrt(1/12);c[0]=A*(l[2]*o[0]+l[0]*o[2]+(o[2]*l[0]+o[0]*l[2])),c[1]=l[1]*o[0]+o[1]*l[0],c[2]=S*(l[1]*o[1]+o[1]*l[1]),c[3]=l[1]*o[2]+o[1]*l[2],c[4]=A*(l[2]*o[2]-l[0]*o[0]+(o[2]*l[2]-o[0]*l[0])),K.dot5(e,t,n,s,r,c,x),u[0]=A*(a[2]*o[0]+a[0]*o[2]+(o[2]*a[0]+o[0]*a[2])),u[1]=a[1]*o[0]+o[1]*a[0],u[2]=S*(a[1]*o[1]+o[1]*a[1]),u[3]=a[1]*o[2]+o[1]*a[2],u[4]=A*(a[2]*o[2]-a[0]*o[0]+(o[2]*a[2]-o[0]*a[0])),K.dot5(e,t,n,s,r,u,p),f[0]=v*(a[2]*a[0]+a[0]*a[2])+-M*(l[2]*l[0]+l[0]*l[2]+(o[2]*o[0]+o[0]*o[2])),f[1]=y*a[1]*a[0]+-v*(l[1]*l[0]+o[1]*o[0]),f[2]=a[1]*a[1]+-A*(l[1]*l[1]+o[1]*o[1]),f[3]=y*a[1]*a[2]+-v*(l[1]*l[2]+o[1]*o[2]),f[4]=v*(a[2]*a[2]-a[0]*a[0])+-M*(l[2]*l[2]-l[0]*l[0]+(o[2]*o[2]-o[0]*o[0])),K.dot5(e,t,n,s,r,f,g),d[0]=A*(a[2]*l[0]+a[0]*l[2]+(l[2]*a[0]+l[0]*a[2])),d[1]=a[1]*l[0]+l[1]*a[0],d[2]=S*(a[1]*l[1]+l[1]*a[1]),d[3]=a[1]*l[2]+l[1]*a[2],d[4]=A*(a[2]*l[2]-a[0]*l[0]+(l[2]*a[2]-l[0]*a[0])),K.dot5(e,t,n,s,r,d,m),h[0]=A*(l[2]*l[0]+l[0]*l[2]-(o[2]*o[0]+o[0]*o[2])),h[1]=l[1]*l[0]-o[1]*o[0],h[2]=S*(l[1]*l[1]-o[1]*o[1]),h[3]=l[1]*l[2]-o[1]*o[2],h[4]=A*(l[2]*l[2]-l[0]*l[0]-(o[2]*o[2]-o[0]*o[0])),K.dot5(e,t,n,s,r,h,_)};static parseHeader(e){const t=new Uint8Array(e,0,K.HeaderSizeBytes),n=new Uint16Array(e,0,K.HeaderSizeBytes/2),s=new Uint32Array(e,0,K.HeaderSizeBytes/4),r=new Float32Array(e,0,K.HeaderSizeBytes/4),o=t[0],a=t[1],l=s[1],c=s[2],u=s[3],f=s[4],d=n[10],h=new B(r[6],r[7],r[8]),x=r[9]||-_s,p=r[10]||_s;return{versionMajor:o,versionMinor:a,maxSectionCount:l,sectionCount:c,maxSplatCount:u,splatCount:f,compressionLevel:d,sceneCenter:h,minSphericalHarmonicsCoeff:x,maxSphericalHarmonicsCoeff:p}}static writeHeaderCountsToBuffer(e,t,n){const s=new Uint32Array(n,0,K.HeaderSizeBytes/4);s[2]=e,s[4]=t}static writeHeaderToBuffer(e,t){const n=new Uint8Array(t,0,K.HeaderSizeBytes),s=new Uint16Array(t,0,K.HeaderSizeBytes/2),r=new Uint32Array(t,0,K.HeaderSizeBytes/4),o=new Float32Array(t,0,K.HeaderSizeBytes/4);n[0]=e.versionMajor,n[1]=e.versionMinor,n[2]=0,n[3]=0,r[1]=e.maxSectionCount,r[2]=e.sectionCount,r[3]=e.maxSplatCount,r[4]=e.splatCount,s[10]=e.compressionLevel,o[6]=e.sceneCenter.x,o[7]=e.sceneCenter.y,o[8]=e.sceneCenter.z,o[9]=e.minSphericalHarmonicsCoeff||-_s,o[10]=e.maxSphericalHarmonicsCoeff||_s}static parseSectionHeaders(e,t,n=0,s){const r=e.compressionLevel,o=e.maxSectionCount,a=new Uint16Array(t,n,o*K.SectionHeaderSizeBytes/2),l=new Uint32Array(t,n,o*K.SectionHeaderSizeBytes/4),c=new Float32Array(t,n,o*K.SectionHeaderSizeBytes/4),u=[];let f=0,d=f/2,h=f/4,x=K.HeaderSizeBytes+e.maxSectionCount*K.SectionHeaderSizeBytes,p=0;for(let g=0;g<o;g++){const m=l[h+1],_=l[h+2],A=l[h+3],S=c[h+4],v=S/2,y=a[d+10],M=l[h+6]||K.CompressionLevels[r].ScaleRange,E=l[h+8],b=l[h+9],C=b*4,I=y*A+C,F=a[d+20],{bytesPerSplat:U}=K.calculateComponentStorage(r,F),O=U*m,k=O+I,z={bytesPerSplat:U,splatCountOffset:p,splatCount:s?m:0,maxSplatCount:m,bucketSize:_,bucketCount:A,bucketBlockSize:S,halfBucketBlockSize:v,bucketStorageSizeBytes:y,bucketsStorageSizeBytes:I,splatDataStorageSizeBytes:O,storageSizeBytes:k,compressionScaleRange:M,compressionScaleFactor:v/M,base:x,bucketsBase:x+C,dataBase:x+I,fullBucketCount:E,partiallyFilledBucketCount:b,sphericalHarmonicsDegree:F};u[g]=z,x+=k,f+=K.SectionHeaderSizeBytes,d=f/2,h=f/4,p+=m}return u}static writeSectionHeaderToBuffer(e,t,n,s=0){const r=new Uint16Array(n,s,K.SectionHeaderSizeBytes/2),o=new Uint32Array(n,s,K.SectionHeaderSizeBytes/4),a=new Float32Array(n,s,K.SectionHeaderSizeBytes/4);o[0]=e.splatCount,o[1]=e.maxSplatCount,o[2]=t>=1?e.bucketSize:0,o[3]=t>=1?e.bucketCount:0,a[4]=t>=1?e.bucketBlockSize:0,r[10]=t>=1?K.BucketStorageSizeBytes:0,o[6]=t>=1?e.compressionScaleRange:0,o[7]=e.storageSizeBytes,o[8]=t>=1?e.fullBucketCount:0,o[9]=t>=1?e.partiallyFilledBucketCount:0,r[20]=e.sphericalHarmonicsDegree}static writeSectionHeaderSplatCountToBuffer(e,t,n=0){const s=new Uint32Array(t,n,K.SectionHeaderSizeBytes/4);s[0]=e}constructFromBuffer(e,t){this.bufferData=e,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSectionMap=[];const n=K.parseHeader(this.bufferData);this.versionMajor=n.versionMajor,this.versionMinor=n.versionMinor,this.maxSectionCount=n.maxSectionCount,this.sectionCount=t?n.maxSectionCount:0,this.maxSplatCount=n.maxSplatCount,this.splatCount=t?n.maxSplatCount:0,this.compressionLevel=n.compressionLevel,this.sceneCenter=new B().copy(n.sceneCenter),this.minSphericalHarmonicsCoeff=n.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff=n.maxSphericalHarmonicsCoeff,this.sections=K.parseSectionHeaders(n,this.bufferData,K.HeaderSizeBytes,t),this.linkBufferArrays(),this.buildMaps()}static calculateComponentStorage(e,t){const n=K.CompressionLevels[e].BytesPerCenter,s=K.CompressionLevels[e].BytesPerScale,r=K.CompressionLevels[e].BytesPerRotation,o=K.CompressionLevels[e].BytesPerColor,a=Jr(t),l=K.CompressionLevels[e].BytesPerSphericalHarmonicsComponent*a,c=n+s+r+o+l;return{bytesPerCenter:n,bytesPerScale:s,bytesPerRotation:r,bytesPerColor:o,sphericalHarmonicsComponentsPerSplat:a,sphericalHarmonicsBytesPerSplat:l,bytesPerSplat:c}}linkBufferArrays(){for(let e=0;e<this.maxSectionCount;e++){const t=this.sections[e];t.bucketArray=new Float32Array(this.bufferData,t.bucketsBase,t.bucketCount*K.BucketStorageSizeFloats),t.partiallyFilledBucketCount>0&&(t.partiallyFilledBucketLengths=new Uint32Array(this.bufferData,t.base,t.partiallyFilledBucketCount))}}buildMaps(){let e=0;for(let t=0;t<this.maxSectionCount;t++){const n=this.sections[t];for(let s=0;s<n.maxSplatCount;s++){const r=e+s;this.globalSplatIndexToLocalSplatIndexMap[r]=s,this.globalSplatIndexToSectionMap[r]=t}e+=n.maxSplatCount}}updateLoadedCounts(e,t){K.writeHeaderCountsToBuffer(e,t,this.bufferData),this.sectionCount=e,this.splatCount=t}updateSectionLoadedCounts(e,t){const n=K.HeaderSizeBytes+K.SectionHeaderSizeBytes*e;K.writeSectionHeaderSplatCountToBuffer(t,this.bufferData,n),this.sections[e].splatCount=t}static writeSplatDataToSectionBuffer=(function(){const e=new ArrayBuffer(12),t=new ArrayBuffer(12),n=new ArrayBuffer(16),s=new ArrayBuffer(4),r=new ArrayBuffer(256),o=new bt,a=new B,l=new B,{X:c,Y:u,Z:f,SCALE0:d,SCALE1:h,SCALE2:x,ROTATION0:p,ROTATION1:g,ROTATION2:m,ROTATION3:_,FDC0:A,FDC1:S,FDC2:v,OPACITY:y,FRC0:M,FRC9:E}=Ee.OFFSET,b=(C,I,F)=>{const U=F*2+1;return C=Math.round(C*I)+F,Ct(C,0,U)};return function(C,I,F,U,O,k,z,V,H=-_s,$=_s){const oe=Jr(O),Se=K.CompressionLevels[U].BytesPerCenter,we=K.CompressionLevels[U].BytesPerScale,Le=K.CompressionLevels[U].BytesPerRotation,fe=K.CompressionLevels[U].BytesPerColor,re=F,X=re+Se,ee=X+we,pe=ee+Le,be=pe+fe;if(C[p]!==void 0?(o.set(C[p],C[g],C[m],C[_]),o.normalize()):o.set(1,0,0,0),C[d]!==void 0?a.set(C[d]||0,C[h]||0,C[x]||0):a.set(0,0,0),U===0){const Ce=new Float32Array(I,re,K.CenterComponentCount),P=new Float32Array(I,ee,K.RotationComponentCount),L=new Float32Array(I,X,K.ScaleComponentCount);if(P.set([o.x,o.y,o.z,o.w]),L.set([a.x,a.y,a.z]),Ce.set([C[c],C[u],C[f]]),O>0){const q=new Float32Array(I,be,oe);if(O>=1){for(let w=0;w<9;w++)q[w]=C[M+w]||0;if(O>=2)for(let w=0;w<15;w++)q[w+9]=C[E+w]||0}}}else{const Ce=new Uint16Array(e,0,K.CenterComponentCount),P=new Uint16Array(n,0,K.RotationComponentCount),L=new Uint16Array(t,0,K.ScaleComponentCount);if(P.set([qt(o.x),qt(o.y),qt(o.z),qt(o.w)]),L.set([qt(a.x),qt(a.y),qt(a.z)]),l.set(C[c],C[u],C[f]).sub(k),l.x=b(l.x,z,V),l.y=b(l.y,z,V),l.z=b(l.z,z,V),Ce.set([l.x,l.y,l.z]),O>0){const q=U===1?Uint16Array:Uint8Array,w=U===1?2:1,te=new q(r,0,oe);if(O>=1){for(let ue=0;ue<9;ue++){const Z=C[M+ue]||0;te[ue]=U===1?qt(Z):Xo(Z,H,$)}const ie=9*w;if(Or(te.buffer,0,I,be,ie),O>=2){for(let ue=0;ue<15;ue++){const Z=C[E+ue]||0;te[ue+9]=U===1?qt(Z):Xo(Z,H,$)}Or(te.buffer,ie,I,be+ie,15*w)}}}Or(Ce.buffer,0,I,re,6),Or(L.buffer,0,I,X,6),Or(P.buffer,0,I,ee,8)}const xe=new Uint8ClampedArray(s,0,4);xe.set([C[A]||0,C[S]||0,C[v]||0]),xe[3]=C[y]||0,Or(xe.buffer,0,I,pe,4)}})();static generateFromUncompressedSplatArrays(e,t,n,s,r,o,a=[]){let l=0;for(let v=0;v<e.length;v++){const y=e[v];l=Math.max(y.sphericalHarmonicsDegree,l)}let c,u;for(let v=0;v<e.length;v++){const y=e[v];for(let M=0;M<y.splats.length;M++){const E=y.splats[M];for(let b=Ee.OFFSET.FRC0;b<Ee.OFFSET.FRC23&&b<E.length;b++)(!c||E[b]<c)&&(c=E[b]),(!u||E[b]>u)&&(u=E[b])}}c=c||-_s,u=u||_s;const{bytesPerSplat:f}=K.calculateComponentStorage(n,l),d=K.CompressionLevels[n].ScaleRange,h=[],x=[];let p=0;for(let v=0;v<e.length;v++){const y=e[v],M=new Ee(l);for(let re=0;re<y.splatCount;re++){const X=y.splats[re];(X[Ee.OFFSET.OPACITY]||0)>=t&&M.addSplat(X)}const E=a[v]||{},b=(E.blockSizeFactor||1)*(r||K.BucketBlockSize),C=Math.ceil((E.bucketSizeFactor||1)*(o||K.BucketSize)),I=K.computeBucketsForUncompressedSplatArray(M,b,C),F=I.fullBuckets.length,U=I.partiallyFullBuckets.map(re=>re.splats.length),O=U.length,k=[...I.fullBuckets,...I.partiallyFullBuckets],z=M.splats.length*f,V=O*4,H=n>=1?k.length*K.BucketStorageSizeBytes+V:0,$=z+H,oe=new ArrayBuffer($),Se=d/(b*.5),we=new B;let Le=0;for(let re=0;re<k.length;re++){const X=k[re];we.fromArray(X.center);for(let ee=0;ee<X.splats.length;ee++){let pe=X.splats[ee];const be=M.splats[pe],xe=H+Le*f;K.writeSplatDataToSectionBuffer(be,oe,xe,n,l,we,Se,d,c,u),Le++}}if(p+=Le,n>=1){const re=new Uint32Array(oe,0,U.length*4);for(let ee=0;ee<U.length;ee++)re[ee]=U[ee];const X=new Float32Array(oe,V,k.length*K.BucketStorageSizeFloats);for(let ee=0;ee<k.length;ee++){const pe=k[ee],be=ee*3;X[be]=pe.center[0],X[be+1]=pe.center[1],X[be+2]=pe.center[2]}}h.push(oe);const fe=new ArrayBuffer(K.SectionHeaderSizeBytes);K.writeSectionHeaderToBuffer({maxSplatCount:Le,splatCount:Le,bucketSize:C,bucketCount:k.length,bucketBlockSize:b,compressionScaleRange:d,storageSizeBytes:$,fullBucketCount:F,partiallyFilledBucketCount:O,sphericalHarmonicsDegree:l},n,fe,0),x.push(fe)}let g=0;for(let v of h)g+=v.byteLength;const m=K.HeaderSizeBytes+K.SectionHeaderSizeBytes*h.length+g,_=new ArrayBuffer(m);K.writeHeaderToBuffer({versionMajor:0,versionMinor:1,maxSectionCount:h.length,sectionCount:h.length,maxSplatCount:p,splatCount:p,compressionLevel:n,sceneCenter:s,minSphericalHarmonicsCoeff:c,maxSphericalHarmonicsCoeff:u},_);let A=K.HeaderSizeBytes;for(let v of x)new Uint8Array(_,A,K.SectionHeaderSizeBytes).set(new Uint8Array(v)),A+=K.SectionHeaderSizeBytes;for(let v of h)new Uint8Array(_,A,v.byteLength).set(new Uint8Array(v)),A+=v.byteLength;return new K(_)}static computeBucketsForUncompressedSplatArray(e,t,n){let s=e.splatCount;const r=t/2,o=new B,a=new B;for(let p=0;p<s;p++){const g=e.splats[p],m=[g[Ee.OFFSET.X],g[Ee.OFFSET.Y],g[Ee.OFFSET.Z]];(p===0||m[0]<o.x)&&(o.x=m[0]),(p===0||m[0]>a.x)&&(a.x=m[0]),(p===0||m[1]<o.y)&&(o.y=m[1]),(p===0||m[1]>a.y)&&(a.y=m[1]),(p===0||m[2]<o.z)&&(o.z=m[2]),(p===0||m[2]>a.z)&&(a.z=m[2])}const l=new B().copy(a).sub(o),c=Math.ceil(l.y/t),u=Math.ceil(l.z/t),f=new B,d=[],h={};for(let p=0;p<s;p++){const g=e.splats[p],m=[g[Ee.OFFSET.X],g[Ee.OFFSET.Y],g[Ee.OFFSET.Z]],_=Math.floor((m[0]-o.x)/t),A=Math.floor((m[1]-o.y)/t),S=Math.floor((m[2]-o.z)/t);f.x=_*t+o.x+r,f.y=A*t+o.y+r,f.z=S*t+o.z+r;const v=_*(c*u)+A*u+S;let y=h[v];y||(h[v]=y={splats:[],center:f.toArray()}),y.splats.push(p),y.splats.length>=n&&(d.push(y),h[v]=null)}const x=[];for(let p in h)if(h.hasOwnProperty(p)){const g=h[p];g&&x.push(g)}return{fullBuckets:d,partiallyFullBuckets:x}}static preallocateUncompressed(e,t){const n=K.CompressionLevels[0].SphericalHarmonicsDegrees[t],s=K.HeaderSizeBytes+K.SectionHeaderSizeBytes,r=s+n.BytesPerSplat*e,o=new ArrayBuffer(r);return K.writeHeaderToBuffer({versionMajor:K.CurrentMajorVersion,versionMinor:K.CurrentMinorVersion,maxSectionCount:1,sectionCount:1,maxSplatCount:e,splatCount:e,compressionLevel:0,sceneCenter:new B},o),K.writeSectionHeaderToBuffer({maxSplatCount:e,splatCount:e,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:t},0,o,K.HeaderSizeBytes),{splatBuffer:new K(o,!0),splatBufferDataOffsetBytes:s}}}const Mp=new Uint8Array([112,108,121,10]),Cp=new Uint8Array([10,101,110,100,95,104,101,97,100,101,114,10]),jc="end_header",$c=new Map([["char",Int8Array],["uchar",Uint8Array],["short",Int16Array],["ushort",Uint16Array],["int",Int32Array],["uint",Uint32Array],["float",Float32Array],["double",Float64Array]]),Ri=(i,e)=>{const t=(1<<e)-1;return(i&t)/t},Tp=(i,e)=>{i.x=Ri(e>>>21,11),i.y=Ri(e>>>11,10),i.z=Ri(e,11)},fT=(i,e)=>{i.x=Ri(e>>>24,8),i.y=Ri(e>>>16,8),i.z=Ri(e>>>8,8),i.w=Ri(e,8)},dT=(i,e)=>{const t=1/(Math.sqrt(2)*.5),n=(Ri(e>>>20,10)-.5)*t,s=(Ri(e>>>10,10)-.5)*t,r=(Ri(e,10)-.5)*t,o=Math.sqrt(1-(n*n+s*s+r*r));switch(e>>>30){case 0:i.set(o,n,s,r);break;case 1:i.set(n,o,s,r);break;case 2:i.set(n,s,o,r);break;case 3:i.set(n,s,r,o);break}},Xi=(i,e,t)=>i*(1-t)+e*t,It=(i,e)=>i.properties.find(t=>t.name===e&&t.storage)?.storage;class st{static decodeHeaderText(e){let t,n,s,r;const o=e.split(`
`).filter(f=>!f.startsWith("comment "));let a=0,l=!1;for(let f=1;f<o.length;++f){const d=o[f].split(" ");switch(d[0]){case"format":if(d[1]!=="binary_little_endian")throw new Error("Unsupported ply format");break;case"element":t={name:d[1],count:parseInt(d[2],10),properties:[],storageSizeBytes:0},t.name==="chunk"?n=t:t.name==="vertex"?s=t:t.name==="sh"&&(r=t);break;case"property":{if(!$c.has(d[1]))throw new Error(`Unrecognized property data type '${d[1]}' in ply header`);const h=$c.get(d[1]),x=h.BYTES_PER_ELEMENT*t.count;t.name==="vertex"&&(a+=h.BYTES_PER_ELEMENT),t.properties.push({type:d[1],name:d[2],storage:null,byteSize:h.BYTES_PER_ELEMENT,storageSizeByes:x}),t.storageSizeBytes+=x;break}case jc:l=!0;break;default:throw new Error(`Unrecognized header value '${d[0]}' in ply header`)}if(l)break}let c=0,u=0;return r&&(u=r.properties.length,r.properties.length>=45?c=3:r.properties.length>=24?c=2:r.properties.length>=9&&(c=1)),{chunkElement:n,vertexElement:s,shElement:r,bytesPerSplat:a,headerSizeBytes:e.indexOf(jc)+jc.length+1,sphericalHarmonicsDegree:c,sphericalHarmonicsPerSplat:u}}static decodeHeader(e){const t=(h,x)=>{const p=h.length-x.length;let g,m;for(g=0;g<=p;++g){for(m=0;m<x.length&&h[g+m]===x[m];++m);if(m===x.length)return g}return-1},n=(h,x)=>{if(h.length<x.length)return!1;for(let p=0;p<x.length;++p)if(h[p]!==x[p])return!1;return!0};let s=new Uint8Array(e),r;if(s.length>=Mp.length&&!n(s,Mp))throw new Error("Invalid PLY header");if(r=t(s,Cp),r===-1)throw new Error("End of PLY header not found");const o=new TextDecoder("ascii").decode(s.slice(0,r)),{chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f,bytesPerSplat:d}=st.decodeHeaderText(o);return{headerSizeBytes:r+Cp.length,bytesPerSplat:d,chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f}}static readElementData(e,t,n,s,r,o=null){let a=t instanceof DataView?t:new DataView(t);s=s||0,r=r||e.count-1;for(let l=s;l<=r;++l)for(let c=0;c<e.properties.length;++c){const u=e.properties[c],f=$c.get(u.type),d=f.BYTES_PER_ELEMENT*e.count;if((!u.storage||u.storage.byteLength<d)&&(!o||o(u.name))&&(u.storage=new f(e.count)),u.storage)switch(u.type){case"char":u.storage[l]=a.getInt8(n);break;case"uchar":u.storage[l]=a.getUint8(n);break;case"short":u.storage[l]=a.getInt16(n,!0);break;case"ushort":u.storage[l]=a.getUint16(n,!0);break;case"int":u.storage[l]=a.getInt32(n,!0);break;case"uint":u.storage[l]=a.getUint32(n,!0);break;case"float":u.storage[l]=a.getFloat32(n,!0);break;case"double":u.storage[l]=a.getFloat64(n,!0);break}n+=u.byteSize}return n}static readPly(e,t=null){const n=st.decodeHeader(e);let s=st.readElementData(n.chunkElement,e,n.headerSizeBytes,null,null,t);return s=st.readElementData(n.vertexElement,e,s,null,null,t),st.readElementData(n.shElement,e,s,null,null,t),{chunkElement:n.chunkElement,vertexElement:n.vertexElement,shElement:n.shElement,sphericalHarmonicsDegree:n.sphericalHarmonicsDegree,sphericalHarmonicsPerSplat:n.sphericalHarmonicsPerSplat}}static getElementStorageArrays(e,t,n){const s={};if(t){const r=It(e,"min_r"),o=It(e,"min_g"),a=It(e,"min_b"),l=It(e,"max_r"),c=It(e,"max_g"),u=It(e,"max_b"),f=It(e,"min_x"),d=It(e,"min_y"),h=It(e,"min_z"),x=It(e,"max_x"),p=It(e,"max_y"),g=It(e,"max_z"),m=It(e,"min_scale_x"),_=It(e,"min_scale_y"),A=It(e,"min_scale_z"),S=It(e,"max_scale_x"),v=It(e,"max_scale_y"),y=It(e,"max_scale_z"),M=It(t,"packed_position"),E=It(t,"packed_rotation"),b=It(t,"packed_scale"),C=It(t,"packed_color");s.colorExtremes={minR:r,maxR:l,minG:o,maxG:c,minB:a,maxB:u},s.positionExtremes={minX:f,maxX:x,minY:d,maxY:p,minZ:h,maxZ:g},s.scaleExtremes={minScaleX:m,maxScaleX:S,minScaleY:_,maxScaleY:v,minScaleZ:A,maxScaleZ:y},s.position=M,s.rotation=E,s.scale=b,s.color=C}if(n){const r={};for(let o=0;o<45;o++){const a=`f_rest_${o}`,l=It(n,a);if(l)r[a]=l;else break}s.sh=r}return s}static decompressBaseSplat=(function(){const e=new B,t=new bt,n=new B,s=new Et,r=Ee.OFFSET;return function(o,a,l,c,u,f,d,h,x,p){p=p||Ee.createSplat();const g=Math.floor((a+o)/256);return Tp(e,l[o]),dT(t,d[o]),Tp(n,u[o]),fT(s,x[o]),p[r.X]=Xi(c.minX[g],c.maxX[g],e.x),p[r.Y]=Xi(c.minY[g],c.maxY[g],e.y),p[r.Z]=Xi(c.minZ[g],c.maxZ[g],e.z),p[r.ROTATION0]=t.x,p[r.ROTATION1]=t.y,p[r.ROTATION2]=t.z,p[r.ROTATION3]=t.w,p[r.SCALE0]=Math.exp(Xi(f.minScaleX[g],f.maxScaleX[g],n.x)),p[r.SCALE1]=Math.exp(Xi(f.minScaleY[g],f.maxScaleY[g],n.y)),p[r.SCALE2]=Math.exp(Xi(f.minScaleZ[g],f.maxScaleZ[g],n.z)),h.minR&&h.maxR?p[r.FDC0]=Ct(Math.round(Xi(h.minR[g],h.maxR[g],s.x)*255),0,255):p[r.FDC0]=Ct(Math.floor(s.x*255),0,255),h.minG&&h.maxG?p[r.FDC1]=Ct(Math.round(Xi(h.minG[g],h.maxG[g],s.y)*255),0,255):p[r.FDC1]=Ct(Math.floor(s.y*255),0,255),h.minB&&h.maxB?p[r.FDC2]=Ct(Math.round(Xi(h.minB[g],h.maxB[g],s.z)*255),0,255):p[r.FDC2]=Ct(Math.floor(s.z*255),0,255),p[r.OPACITY]=Ct(Math.floor(s.w*255),0,255),p}})();static decompressSphericalHarmonics=(function(){const e=[0,3,8,15],t=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(n,s,r,o,a){a=a||Ee.createSplat();let l=e[r],c=e[o];for(let u=0;u<3;++u)for(let f=0;f<15;++f){const d=t[u*15+f];f<l&&f<c&&(a[Ee.OFFSET.FRC0+d]=s[u*c+f][n]*(8/255)-4)}return a}})();static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l,c=null){st.readElementData(t,o,0,n,s,c);const u=K.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,{positionExtremes:f,scaleExtremes:d,colorExtremes:h,position:x,rotation:p,scale:g,color:m}=st.getElementStorageArrays(e,t),_=Ee.createSplat();for(let A=n;A<=s;++A){st.decompressBaseSplat(A,r,x,f,g,d,p,h,m,_);const S=A*u+l;K.writeSplatDataToSectionBuffer(_,a,S,0,0)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a,l=null){st.readElementData(t,o,0,n,s,l);const{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:d,rotation:h,scale:x,color:p}=st.getElementStorageArrays(e,t);for(let g=n;g<=s;++g){const m=Ee.createSplat();st.decompressBaseSplat(g,r,d,c,x,u,h,f,p,m),a.addSplat(m)}}static parseSphericalHarmonicsToUncompressedSplatArraySection(e,t,n,s,r,o,a,l,c,u=null){st.readElementData(t,r,o,n,s,u);const{sh:f}=st.getElementStorageArrays(e,void 0,t),d=Object.values(f);for(let h=n;h<=s;++h)st.decompressSphericalHarmonics(h,d,a,l,c.splats[h])}static parseToUncompressedSplatArray(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=st.readPly(e);t=Math.min(t,o);const a=new Ee(t),{positionExtremes:l,scaleExtremes:c,colorExtremes:u,position:f,rotation:d,scale:h,color:x}=st.getElementStorageArrays(n,s);let p;if(t>0){const{sh:g}=st.getElementStorageArrays(n,void 0,r);p=Object.values(g)}for(let g=0;g<s.count;++g){a.addDefaultSplat();const m=a.getSplat(a.splatCount-1);st.decompressBaseSplat(g,0,f,l,h,c,d,u,x,m),t>0&&st.decompressSphericalHarmonics(g,p,t,o,m)}return a}static parseToUncompressedSplatBuffer(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=st.readPly(e);t=Math.min(t,o);const{splatBuffer:a,splatBufferDataOffsetBytes:l}=K.preallocateUncompressed(s.count,t),{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:d,rotation:h,scale:x,color:p}=st.getElementStorageArrays(n,s);let g;if(t>0){const{sh:A}=st.getElementStorageArrays(n,void 0,r);g=Object.values(A)}const m=K.CompressionLevels[0].SphericalHarmonicsDegrees[t].BytesPerSplat,_=Ee.createSplat(t);for(let A=0;A<s.count;++A){st.decompressBaseSplat(A,0,d,c,x,u,h,f,p,_),t>0&&st.decompressSphericalHarmonics(A,g,t,o,_);const S=A*m+l;K.writeSplatDataToSectionBuffer(_,a.bufferData,S,0,t)}return a}}const hn={INRIAV1:0,INRIAV2:1,PlayCanvasCompressed:2},[ig,cd,ud,fd,dd,hd,pd]=[0,1,2,3,4,5,6],Ep={double:ig,int:cd,uint:ud,float:fd,short:dd,ushort:hd,uchar:pd},hT={[ig]:8,[cd]:4,[ud]:4,[fd]:4,[dd]:2,[hd]:2,[pd]:1};class ot{static HeaderEndToken="end_header";static decodeSectionHeader(e,t,n=0){const s=[];let r=!1,o=-1,a=0,l=!1,c=null;const u=[],f=[],d=[],h={};for(let m=n;m<e.length;m++){const _=e[m].trim();if(_.startsWith("element"))if(r){o--;break}else{r=!0,n=m,o=m;const A=_.split(" ");let S=0;for(let v of A){const y=v.trim();y.length>0&&(S++,S===2?c=y:S===3&&(a=parseInt(y)))}}else if(_.startsWith("property")){const A=_.match(/(\w+)\s+(\w+)\s+(\w+)/);if(A){const S=A[2],v=A[3];d.push(v);const y=t[v];h[v]=S;const M=Ep[S];y!==void 0&&(u.push(y),f[y]=M)}}if(_===ot.HeaderEndToken){l=!0;break}r&&(s.push(_),o++)}const x=[];let p=0;for(let m of d){const _=h[m];if(h.hasOwnProperty(m)){const A=t[m];A!==void 0&&(x[A]=p)}p+=hT[Ep[_]]}const g=ot.decodeSphericalHarmonicsFromSectionHeader(d,t);return{headerLines:s,headerStartLine:n,headerEndLine:o,fieldTypes:f,fieldIds:u,fieldOffsets:x,bytesPerVertex:p,vertexCount:a,dataSizeBytes:p*a,endOfHeader:l,sectionName:c,sphericalHarmonicsDegree:g.degree,sphericalHarmonicsCoefficientsPerChannel:g.coefficientsPerChannel,sphericalHarmonicsDegree1Fields:g.degree1Fields,sphericalHarmonicsDegree2Fields:g.degree2Fields}}static decodeSphericalHarmonicsFromSectionHeader(e,t){let n=0,s=0;for(let l of e)l.startsWith("f_rest")&&n++;s=n/3;let r=0;s>=3&&(r=1),s>=8&&(r=2);let o=[],a=[];for(let l=0;l<3;l++){if(r>=1)for(let c=0;c<3;c++)o.push(t["f_rest_"+(c+s*l)]);if(r>=2)for(let c=0;c<5;c++)a.push(t["f_rest_"+(c+s*l+3)])}return{degree:r,coefficientsPerChannel:s,degree1Fields:o,degree2Fields:a}}static getHeaderSectionNames(e){const t=[];for(let n of e)if(n.startsWith("element")){const s=n.split(" ");let r=0;for(let o of s){const a=o.trim();a.length>0&&(r++,r===2&&t.push(a))}}return t}static checkTextForEndHeader(e){return!!e.includes(ot.HeaderEndToken)}static checkBufferForEndHeader(e,t,n,s){const r=new Uint8Array(e,Math.max(0,t-n),n),o=s.decode(r);return ot.checkTextForEndHeader(o)}static extractHeaderFromBufferToText(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,ot.checkBufferForEndHeader(e,n,r*2,t))break}return s}static readHeaderFromBuffer(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,ot.checkBufferForEndHeader(e,n,r*2,t))break}return s}static convertHeaderTextToLines(e){const t=e.split(`
`),n=[];for(let s=0;s<t.length;s++){const r=t[s].trim();if(n.push(r),r===ot.HeaderEndToken)break}return n}static determineHeaderFormatFromHeaderText(e){const t=ot.convertHeaderTextToLines(e);let n=hn.INRIAV1;for(let s=0;s<t.length;s++){const r=t[s].trim();if(r.startsWith("element chunk")||r.match(/[A-Za-z]*packed_[A-Za-z]*/))n=hn.PlayCanvasCompressed;else if(r.startsWith("element codebook_centers"))n=hn.INRIAV2;else if(r===ot.HeaderEndToken)break}return n}static determineHeaderFormatFromPlyBuffer(e){const t=ot.extractHeaderFromBufferToText(e);return ot.determineHeaderFormatFromHeaderText(t)}static readVertex(e,t,n,s,r,o,a=!0){const l=n*t.bytesPerVertex+s,c=t.fieldOffsets,u=t.fieldTypes;for(let f of r){const d=u[f];d===fd?o[f]=e.getFloat32(l+c[f],!0):d===dd?o[f]=e.getInt16(l+c[f],!0):d===hd?o[f]=e.getUint16(l+c[f],!0):d===cd?o[f]=e.getInt32(l+c[f],!0):d===ud?o[f]=e.getUint32(l+c[f],!0):d===pd&&(a?o[f]=e.getUint8(l+c[f])/255:o[f]=e.getUint8(l+c[f]))}}}const sg=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0"],pT=sg.map((i,e)=>e),[wp,mT,gT,xT,_T,AT,ST,vT,yT,bT,Rp,MT,CT,Ip,Dp,TT,ET,wT]=pT;class Zt{static decodeHeaderLines(e){let t=0;e.forEach(u=>{u.includes("f_rest_")&&t++});let n=0;t>=45?n=45:t>=24?n=24:t>=9&&(n=9);let r=Array.from(Array(Math.max(n-1,0))).map((u,f)=>`f_rest_${f+1}`);const o=[...sg,...r],a=o.map((u,f)=>f),l=a.reduce((u,f)=>(u[o[f]]=f,u),{}),c=ot.decodeSectionHeader(e,l,0);return c.splatCount=c.vertexCount,c.bytesPerSplat=c.bytesPerVertex,c.fieldsToReadIndexes=a,c}static decodeHeaderText(e){const t=ot.convertHeaderTextToLines(e),n=Zt.decodeHeaderLines(t);return n.headerText=e,n.headerSizeBytes=e.indexOf(ot.HeaderEndToken)+ot.HeaderEndToken.length+1,n}static decodeHeaderFromBuffer(e){const t=ot.readHeaderFromBuffer(e);return Zt.decodeHeaderText(t)}static findSplatData(e,t){return new DataView(e,t.headerSizeBytes)}static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l=0){l=Math.min(l,e.sphericalHarmonicsDegree);const c=K.CompressionLevels[0].SphericalHarmonicsDegrees[l].BytesPerSplat;for(let u=t;u<=n;u++){const f=Zt.parseToUncompressedSplat(s,u,e,r,l),d=u*c+a;K.writeSplatDataToSectionBuffer(f,o,d,0,l)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a=0){a=Math.min(a,e.sphericalHarmonicsDegree);for(let l=t;l<=n;l++){const c=Zt.parseToUncompressedSplat(s,l,e,r,a);o.addSplat(c)}}static decodeSectionSplatData(e,t,n,s,r=!0){if(s=Math.min(s,n.sphericalHarmonicsDegree),r){const o=new Ee(s);for(let a=0;a<t;a++){const l=Zt.parseToUncompressedSplat(e,a,n,0,s);o.addSplat(l)}return o}else{const{splatBuffer:o,splatBufferDataOffsetBytes:a}=K.preallocateUncompressed(t,s);return Zt.parseToUncompressedSplatBufferSection(n,0,t-1,e,0,o.bufferData,a,s),o}}static parseToUncompressedSplat=(function(){let e=[];const t=new bt,n=Ee.OFFSET.X,s=Ee.OFFSET.Y,r=Ee.OFFSET.Z,o=Ee.OFFSET.SCALE0,a=Ee.OFFSET.SCALE1,l=Ee.OFFSET.SCALE2,c=Ee.OFFSET.ROTATION0,u=Ee.OFFSET.ROTATION1,f=Ee.OFFSET.ROTATION2,d=Ee.OFFSET.ROTATION3,h=Ee.OFFSET.FDC0,x=Ee.OFFSET.FDC1,p=Ee.OFFSET.FDC2,g=Ee.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=Ee.OFFSET.FRC0+_;return function(_,A,S,v=0,y=0){y=Math.min(y,S.sphericalHarmonicsDegree),Zt.readSplat(_,S,A,v,e);const M=Ee.createSplat(y);if(e[wp]!==void 0?(M[o]=Math.exp(e[wp]),M[a]=Math.exp(e[mT]),M[l]=Math.exp(e[gT])):(M[o]=.01,M[a]=.01,M[l]=.01),e[Rp]!==void 0){const E=.28209479177387814;M[h]=(.5+E*e[Rp])*255,M[x]=(.5+E*e[MT])*255,M[p]=(.5+E*e[CT])*255}else e[Dp]!==void 0?(M[h]=e[Dp]*255,M[x]=e[TT]*255,M[p]=e[ET]*255):(M[h]=0,M[x]=0,M[p]=0);if(e[Ip]!==void 0&&(M[g]=1/(1+Math.exp(-e[Ip]))*255),M[h]=Ct(Math.floor(M[h]),0,255),M[x]=Ct(Math.floor(M[x]),0,255),M[p]=Ct(Math.floor(M[p]),0,255),M[g]=Ct(Math.floor(M[g]),0,255),y>=1&&e[wT]!==void 0){for(let E=0;E<9;E++)M[m[E]]=e[S.sphericalHarmonicsDegree1Fields[E]];if(y>=2)for(let E=0;E<15;E++)M[m[9+E]]=e[S.sphericalHarmonicsDegree2Fields[E]]}return t.set(e[xT],e[_T],e[AT],e[ST]),t.normalize(),M[c]=t.x,M[u]=t.y,M[f]=t.z,M[d]=t.w,M[n]=e[vT],M[s]=e[yT],M[r]=e[bT],M}})();static readSplat(e,t,n,s,r){return ot.readVertex(e,t,n,s,t.fieldsToReadIndexes,r,!0)}static parseToUncompressedSplatArray(e,t=0){const{header:n,splatCount:s,splatData:r}=Pp(e);return Zt.decodeSectionSplatData(r,s,n,t,!0)}static parseToUncompressedSplatBuffer(e,t=0){const{header:n,splatCount:s,splatData:r}=Pp(e);return Zt.decodeSectionSplatData(r,s,n,t,!1)}}function Pp(i){const e=Zt.decodeHeaderFromBuffer(i),t=e.splatCount,n=Zt.findSplatData(i,e);return{header:e,splatCount:t,splatData:n}}const rg=["features_dc","features_rest_0","features_rest_1","features_rest_2","features_rest_3","features_rest_4","features_rest_5","features_rest_6","features_rest_7","features_rest_8","features_rest_9","features_rest_10","features_rest_11","features_rest_12","features_rest_13","features_rest_14","opacity","scaling","rotation_re","rotation_im"],el=rg.map((i,e)=>e),[tl,RT,IT,Fp,nl,DT,Zc]=[0,1,4,16,17,18,19],og=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0","f_rest_1","f_rest_2","f_rest_3","f_rest_4","f_rest_5","f_rest_6","f_rest_7","f_rest_8","f_rest_9","f_rest_10","f_rest_11","f_rest_12","f_rest_13","f_rest_14","f_rest_15","f_rest_16","f_rest_17","f_rest_18","f_rest_19","f_rest_20","f_rest_21","f_rest_22","f_rest_23","f_rest_24","f_rest_25","f_rest_26","f_rest_27","f_rest_28","f_rest_29","f_rest_30","f_rest_31","f_rest_32","f_rest_33","f_rest_34","f_rest_35","f_rest_36","f_rest_37","f_rest_38","f_rest_39","f_rest_40","f_rest_41","f_rest_42","f_rest_43","f_rest_44","f_rest_45"],df=og.map((i,e)=>e),[Lp,PT,FT,LT,BT,UT,OT,NT,zT,kT,hf,ag,lg,Bp]=df,Up=hf,HT=ag,VT=lg,il=i=>{const e=(31744&i)>>10,t=1023&i;return(i>>15?-1:1)*(e?e===31?t?NaN:1/0:Math.pow(2,e-15)*(1+t/1024):t/1024*6103515625e-14)};class zn{static decodeSectionHeadersFromHeaderLines(e){const t=df.reduce((u,f)=>(u[og[f]]=f,u),{}),n=el.reduce((u,f)=>(u[rg[f]]=f,u),{}),s=ot.getHeaderSectionNames(e);let r;for(let u=0;u<s.length;u++)s[u]==="codebook_centers"&&(r=u);let o=0,a=!1;const l=[];let c=0;for(;!a;){let u;c===r?u=ot.decodeSectionHeader(e,n,o):u=ot.decodeSectionHeader(e,t,o),a=u.endOfHeader,o=u.headerEndLine+1,a||(u.splatCount=u.vertexCount,u.bytesPerSplat=u.bytesPerVertex),l.push(u),c++}return l}static decodeSectionHeadersFromHeaderText(e){const t=ot.convertHeaderTextToLines(e);return zn.decodeSectionHeadersFromHeaderLines(t)}static getSplatCountFromSectionHeaders(e){let t=0;for(let n of e)n.sectionName!=="codebook_centers"&&(t+=n.vertexCount);return t}static decodeHeaderFromHeaderText(e){const t=e.indexOf(ot.HeaderEndToken)+ot.HeaderEndToken.length+1,n=zn.decodeSectionHeadersFromHeaderText(e),s=zn.getSplatCountFromSectionHeaders(n);return{headerSizeBytes:t,sectionHeaders:n,splatCount:s}}static decodeHeaderFromBuffer(e){const t=ot.readHeaderFromBuffer(e);return zn.decodeHeaderFromHeaderText(t)}static findVertexData(e,t,n){let s=t.headerSizeBytes;for(let r=0;r<n&&r<t.sectionHeaders.length;r++){const o=t.sectionHeaders[r];s+=o.dataSizeBytes}return new DataView(e,s,t.sectionHeaders[n].dataSizeBytes)}static decodeCodeBook(e,t){const n=[],s=[];for(let r=0;r<t.vertexCount;r++){ot.readVertex(e,t,r,0,el,n);for(let o of el){const a=el[o];let l=s[a];l||(s[a]=l=[]),l.push(n[o])}}for(let r=0;r<s.length;r++){const o=s[r],a=.28209479177387814;for(let l=0;l<o.length;l++){const c=il(o[l]);r===Fp?o[l]=Math.round(1/(1+Math.exp(-c))*255):r===tl?o[l]=Math.round((.5+a*c)*255):r===nl?o[l]=Math.exp(c):o[l]=c}}return s}static decodeSectionSplatData(e,t,n,s,r){r=Math.min(r,n.sphericalHarmonicsDegree);const o=new Ee(r);for(let a=0;a<t;a++){const l=zn.parseToUncompressedSplat(e,a,n,s,0,r);o.addSplat(l)}return o}static parseToUncompressedSplat=(function(){let e=[];const t=new bt,n=Ee.OFFSET.X,s=Ee.OFFSET.Y,r=Ee.OFFSET.Z,o=Ee.OFFSET.SCALE0,a=Ee.OFFSET.SCALE1,l=Ee.OFFSET.SCALE2,c=Ee.OFFSET.ROTATION0,u=Ee.OFFSET.ROTATION1,f=Ee.OFFSET.ROTATION2,d=Ee.OFFSET.ROTATION3,h=Ee.OFFSET.FDC0,x=Ee.OFFSET.FDC1,p=Ee.OFFSET.FDC2,g=Ee.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=Ee.OFFSET.FRC0+_;return function(_,A,S,v,y=0,M=0){M=Math.min(M,S.sphericalHarmonicsDegree),zn.readSplat(_,S,A,y,e);const E=Ee.createSplat(M);if(e[Lp]!==void 0?(E[o]=v[nl][e[Lp]],E[a]=v[nl][e[PT]],E[l]=v[nl][e[FT]]):(E[o]=.01,E[a]=.01,E[l]=.01),e[hf]!==void 0?(E[h]=v[tl][e[hf]],E[x]=v[tl][e[ag]],E[p]=v[tl][e[lg]]):e[Up]!==void 0?(E[h]=e[Up]*255,E[x]=e[HT]*255,E[p]=e[VT]*255):(E[h]=0,E[x]=0,E[p]=0),e[Bp]!==void 0&&(E[g]=v[Fp][e[Bp]]),E[h]=Ct(Math.floor(E[h]),0,255),E[x]=Ct(Math.floor(E[x]),0,255),E[p]=Ct(Math.floor(E[p]),0,255),E[g]=Ct(Math.floor(E[g]),0,255),M>=1&&S.sphericalHarmonicsDegree>=1){for(let U=0;U<9;U++){const O=v[RT+U%3];E[m[U]]=O[e[S.sphericalHarmonicsDegree1Fields[U]]]}if(M>=2&&S.sphericalHarmonicsDegree>=2)for(let U=0;U<15;U++){const O=v[IT+U%5];E[m[9+U]]=O[e[S.sphericalHarmonicsDegree2Fields[U]]]}}const b=v[DT][e[LT]],C=v[Zc][e[BT]],I=v[Zc][e[UT]],F=v[Zc][e[OT]];return t.set(b,C,I,F),t.normalize(),E[c]=t.x,E[u]=t.y,E[f]=t.z,E[d]=t.w,E[n]=il(e[NT]),E[s]=il(e[zT]),E[r]=il(e[kT]),E}})();static readSplat(e,t,n,s,r){return ot.readVertex(e,t,n,s,df,r,!1)}static parseToUncompressedSplatArray(e,t=0){const n=[],s=zn.decodeHeaderFromBuffer(e,t);let r;for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName==="codebook_centers"){const c=zn.findVertexData(e,s,a);r=zn.decodeCodeBook(c,l)}}for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName!=="codebook_centers"){const c=l.vertexCount,u=zn.findVertexData(e,s,a),f=zn.decodeSectionSplatData(u,c,l,r,t);n.push(f)}}const o=new Ee(t);for(let a of n)for(let l of a.splats)o.addSplat(l);return o}}class Op{static parseToUncompressedSplatArray(e,t=0){const n=ot.determineHeaderFormatFromPlyBuffer(e);if(n===hn.PlayCanvasCompressed)return st.parseToUncompressedSplatArray(e,t);if(n===hn.INRIAV1)return Zt.parseToUncompressedSplatArray(e,t);if(n===hn.INRIAV2)return zn.parseToUncompressedSplatArray(e,t)}static parseToUncompressedSplatBuffer(e,t=0){const n=ot.determineHeaderFormatFromPlyBuffer(e);if(n===hn.PlayCanvasCompressed)return st.parseToUncompressedSplatBuffer(e,t);if(n===hn.INRIAV1)return Zt.parseToUncompressedSplatBuffer(e,t);if(n===hn.INRIAV2)throw new Error("parseToUncompressedSplatBuffer() is not implemented for INRIA V2 PLY files")}}class md{constructor(e,t,n,s){this.sectionCount=e,this.sectionFilters=t,this.groupingParameters=n,this.partitionGenerator=s}partitionUncompressedSplatArray(e){let t,n,s;if(this.partitionGenerator){const o=this.partitionGenerator(e);t=o.groupingParameters,n=o.sectionCount,s=o.sectionFilters}else t=this.groupingParameters,n=this.sectionCount,s=this.sectionFilters;const r=[];for(let o=0;o<n;o++){const a=new Ee(e.sphericalHarmonicsDegree),l=s[o];for(let c=0;c<e.splatCount;c++)l(c)&&a.addSplat(e.splats[c]);r.push(a)}return{splatArrays:r,parameters:t}}static getStandardPartitioner(e=0,t=new B,n=K.BucketBlockSize,s=K.BucketSize){const r=o=>{const a=Ee.OFFSET.X,l=Ee.OFFSET.Y,c=Ee.OFFSET.Z;e<=0&&(e=o.splatCount);const u=new B,f=.5,d=m=>{m.x=Math.floor(m.x/f)*f,m.y=Math.floor(m.y/f)*f,m.z=Math.floor(m.z/f)*f};o.splats.forEach(m=>{u.set(m[a],m[l],m[c]).sub(t),d(u),m.centerDist=u.lengthSq()}),o.splats.sort((m,_)=>{let A=m.centerDist,S=_.centerDist;return A>S?1:-1});const h=[],x=[];e=Math.min(o.splatCount,e);const p=Math.ceil(o.splatCount/e);let g=0;for(let m=0;m<p;m++){let _=g;h.push(A=>A>=_&&A<_+e),x.push({blocksSize:n,bucketSize:s}),g+=e}return{sectionCount:h.length,sectionFilters:h,groupingParameters:x}};return new md(void 0,void 0,void 0,r)}}class Ma{constructor(e,t,n,s,r,o,a){this.splatPartitioner=e,this.alphaRemovalThreshold=t,this.compressionLevel=n,this.sectionSize=s,this.sceneCenter=r?new B().copy(r):void 0,this.blockSize=o,this.bucketSize=a}generateFromUncompressedSplatArray(e){const t=this.splatPartitioner.partitionUncompressedSplatArray(e);return K.generateFromUncompressedSplatArrays(t.splatArrays,this.alphaRemovalThreshold,this.compressionLevel,this.sceneCenter,this.blockSize,this.bucketSize,t.parameters)}static getStandardGenerator(e=1,t=1,n=0,s=new B,r=K.BucketBlockSize,o=K.BucketSize){const a=md.getStandardPartitioner(n,s,r,o);return new Ma(a,e,t,n,s,r,o)}}const Nt={Downloading:0,Processing:1,Done:2};class Dl extends Error{constructor(e){super(e)}}const yt={ProgressiveToSplatBuffer:0,ProgressiveToSplatArray:1,DownloadBeforeProcessing:2};function Np(i,e){let t=0;for(let s of i)t+=s.sizeBytes;(!e||e.byteLength<t)&&(e=new ArrayBuffer(t));let n=0;for(let s of i)new Uint8Array(e,n,s.sizeBytes).set(s.data),n+=s.sizeBytes;return e}function zp(i,e,t,n,s,r,o,a){return e?Ma.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):K.generateFromUncompressedSplatArrays([i],t,0,new B)}class gd{static loadFromURL(e,t,n,s,r,o,a=!0,l=0,c,u,f,d,h){let x;!n&&!a?x=yt.DownloadBeforeProcessing:a?x=yt.ProgressiveToSplatArray:x=yt.ProgressiveToSplatBuffer;const p=pt.ProgressiveLoadSectionSize,g=K.HeaderSizeBytes+K.SectionHeaderSizeBytes,m=1;let _,A,S,v,y,M=0,E=0,b=0,C=!1,I=!1,F=!1;const U=rd();let O=0,k=0,z=0,V=0,H="",$=null,oe=[],Se;const we=new TextDecoder,Le=(fe,re,X)=>{const ee=fe>=100;if(X&&(oe.push({data:X,sizeBytes:X.byteLength,startBytes:z,endBytes:z+X.byteLength}),z+=X.byteLength),x===yt.DownloadBeforeProcessing)ee&&U.resolve(oe);else{if(C){if(_===hn.PlayCanvasCompressed&&!I){const pe=$.headerSizeBytes+$.chunkElement.storageSizeBytes;y=Np(oe,y),y.byteLength>=pe&&(st.readElementData($.chunkElement,y,$.headerSizeBytes),O=pe,k=pe,I=!0)}}else if(H+=we.decode(X),ot.checkTextForEndHeader(H)){if(_=ot.determineHeaderFormatFromHeaderText(H),_===hn.INRIAV1)$=Zt.decodeHeaderText(H),l=Math.min(l,$.sphericalHarmonicsDegree),M=$.splatCount,I=!0,V=$.headerSizeBytes+$.bytesPerSplat*M;else if(_===hn.PlayCanvasCompressed){if($=st.decodeHeaderText(H),l=Math.min(l,$.sphericalHarmonicsDegree),x===yt.ProgressiveToSplatBuffer&&l>0)throw new Dl("PlyLoader.loadFromURL() -> Selected PLY format has spherical harmonics data that cannot be progressively loaded.");M=$.vertexElement.count,V=$.headerSizeBytes+$.bytesPerSplat*M+$.chunkElement.storageSizeBytes}else{if(x===yt.ProgressiveToSplatBuffer)throw new Dl("PlyLoader.loadFromURL() -> Selected PLY format cannot be progressively loaded.");x=yt.DownloadBeforeProcessing;return}if(x===yt.ProgressiveToSplatBuffer){const pe=K.CompressionLevels[0].SphericalHarmonicsDegrees[l],be=g+pe.BytesPerSplat*M;S=new ArrayBuffer(be),K.writeHeaderToBuffer({versionMajor:K.CurrentMajorVersion,versionMinor:K.CurrentMinorVersion,maxSectionCount:m,sectionCount:m,maxSplatCount:M,splatCount:0,compressionLevel:0,sceneCenter:new B},S)}else Se=new Ee(l);O=$.headerSizeBytes,k=$.headerSizeBytes,C=!0}if(C&&I&&oe.length>0&&(A=Np(oe,A),z-O>p||z>=V&&!F||ee)){const be=F?$.sphericalHarmonicsPerSplat:$.bytesPerSplat,Ce=(F?z:Math.min(V,z))-k,P=Math.floor(Ce/be),L=P*be,q=z-k-L,w=k-oe[0].startBytes,te=new DataView(A,w,L);if(F)_===hn.PlayCanvasCompressed&&x===yt.ProgressiveToSplatArray&&(st.parseSphericalHarmonicsToUncompressedSplatArraySection($.chunkElement,$.shElement,b,b+P-1,te,0,l,$.sphericalHarmonicsDegree,Se),b+=P);else{if(x===yt.ProgressiveToSplatBuffer){const ie=K.CompressionLevels[0].SphericalHarmonicsDegrees[l],ue=E*ie.BytesPerSplat+g;_===hn.PlayCanvasCompressed?st.parseToUncompressedSplatBufferSection($.chunkElement,$.vertexElement,0,P-1,E,te,S,ue):Zt.parseToUncompressedSplatBufferSection($,0,P-1,te,0,S,ue,l)}else _===hn.PlayCanvasCompressed?st.parseToUncompressedSplatArraySection($.chunkElement,$.vertexElement,0,P-1,E,te,Se):Zt.parseToUncompressedSplatArraySection($,0,P-1,te,0,Se,l);E+=P,x===yt.ProgressiveToSplatBuffer&&(v||(K.writeSectionHeaderToBuffer({maxSplatCount:M,splatCount:E,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:l},0,S,K.HeaderSizeBytes),v=new K(S,!1)),v.updateLoadedCounts(1,E)),z>=V&&(F=!0)}if(q===0)oe=[];else{let ie=[],ue=0;for(let Z=oe.length-1;Z>=0;Z--){const de=oe[Z];if(ue+=de.sizeBytes,ie.unshift(de),ue>=q)break}oe=ie}O+=p,k+=L}s&&v&&s(v,ee),ee&&(x===yt.ProgressiveToSplatBuffer?U.resolve(v):U.resolve(Se))}t&&t(fe,re,Nt.Downloading)};return t&&t(0,"0%",Nt.Downloading),tc(e,Le,!1,c).then(()=>(t&&t(0,"0%",Nt.Processing),U.promise.then(fe=>{if(t&&t(100,"100%",Nt.Done),x===yt.DownloadBeforeProcessing){const re=oe.map(X=>X.data);return new Blob(re).arrayBuffer().then(X=>gd.loadFromFileData(X,r,o,a,l,u,f,d,h))}else return x===yt.ProgressiveToSplatBuffer?fe:Gn(()=>zp(fe,a,r,o,u,f,d,h))})))}static loadFromFileData(e,t,n,s,r=0,o,a,l,c){return s?Gn(()=>Op.parseToUncompressedSplatArray(e,r)).then(u=>zp(u,s,t,n,o,a,l,c)):Gn(()=>Op.parseToUncompressedSplatBuffer(e,r))}}const GT=i=>new ReadableStream({async start(e){e.enqueue(i),e.close()}});async function WT(i){try{const e=GT(i);if(!e)throw new Error("Failed to create stream from data");return await XT(e)}catch(e){throw console.error("Error decompressing gzipped data:",e),e}}async function XT(i){const e=i.pipeThrough(new DecompressionStream("gzip")),n=await new Response(e).arrayBuffer();return new Uint8Array(n)}const qT=1347635022,QT=1,YT=.15;function KT(i){const e=i>>15&1,t=i>>10&31,n=i&1023,s=e===1?-1:1;return t===0?s*Math.pow(2,-14)*n/1024:t===31?n!==0?NaN:s*(1/0):s*Math.pow(2,t-15)*(1+n/1024)}function jT(i){return(i-128)/128}function rr(i){switch(i){case 0:return 0;case 1:return 3;case 2:return 8;case 3:return 15;default:return console.error(`[SPZ: ERROR] Unsupported SH degree: ${i}`),0}}const $T=(function(){let i=[];const e=new bt,t=Ee.OFFSET.X,n=Ee.OFFSET.Y,s=Ee.OFFSET.Z,r=Ee.OFFSET.SCALE0,o=Ee.OFFSET.SCALE1,a=Ee.OFFSET.SCALE2,l=Ee.OFFSET.ROTATION0,c=Ee.OFFSET.ROTATION1,u=Ee.OFFSET.ROTATION2,f=Ee.OFFSET.ROTATION3,d=Ee.OFFSET.FDC0,h=Ee.OFFSET.FDC1,x=Ee.OFFSET.FDC2,p=Ee.OFFSET.OPACITY,g=[rr(0),rr(1),rr(2),rr(3)],m=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(_,A,S){S=Math.min(A,S);const v=Ee.createSplat(S);_.scale[0]!==void 0?(v[r]=_.scale[0],v[o]=_.scale[1],v[a]=_.scale[2]):(v[r]=.01,v[o]=.01,v[a]=.01),_.color[0]!==void 0?(v[d]=_.color[0],v[h]=_.color[1],v[x]=_.color[2]):i[RED]!==void 0?(v[d]=i[RED]*255,v[h]=i[GREEN]*255,v[x]=i[BLUE]*255):(v[d]=0,v[h]=0,v[x]=0),_.alpha!==void 0&&(v[p]=_.alpha),v[d]=Ct(Math.floor(v[d]),0,255),v[h]=Ct(Math.floor(v[h]),0,255),v[x]=Ct(Math.floor(v[x]),0,255),v[p]=Ct(Math.floor(v[p]),0,255);let y=g[S],M=g[A];for(let E=0;E<3;++E)for(let b=0;b<15;++b){const C=m[E*15+b];b<y&&b<M&&(v[Ee.OFFSET.FRC0+C]=_.sh[E*M+b])}return e.set(_.rotation[3],_.rotation[0],_.rotation[1],_.rotation[2]),e.normalize(),v[l]=e.x,v[c]=e.y,v[u]=e.z,v[f]=e.w,v[t]=_.position[0],v[n]=_.position[1],v[s]=_.position[2],v}})();function ZT(i,e,t,n){return!(i.positions.length!==e*3*(n?2:3)||i.scales.length!==e*3||i.rotations.length!==e*3||i.alphas.length!==e||i.colors.length!==e*3||i.sh.length!==e*t*3)}function kp(i,e,t,n,s){e=Math.min(e,i.shDegree);const r=i.numPoints,o=rr(i.shDegree),a=i.positions.length===r*3*2;if(!ZT(i,r,o,a))return null;const l={position:[],scale:[],rotation:[],alpha:void 0,color:[],sh:[]};let c;a&&(c=new Uint16Array(i.positions.buffer,i.positions.byteOffset,r*3));const u=1/(1<<i.fractionalBits),f=rr(i.shDegree),d=.28209479177387814;for(let h=0;h<r;h++){if(a)for(let _=0;_<3;_++)l.position[_]=KT(c[h*3+_]);else for(let _=0;_<3;_++){const A=h*9+_*3;let S=i.positions[A];S|=i.positions[A+1]<<8,S|=i.positions[A+2]<<16,S|=S&8388608?4278190080:0,l.position[_]=S*u}for(let _=0;_<3;_++)l.scale[_]=Math.exp(i.scales[h*3+_]/16-10);const x=i.rotations.subarray(h*3,h*3+3),p=[x[0]/127.5-1,x[1]/127.5-1,x[2]/127.5-1];l.rotation[0]=p[0],l.rotation[1]=p[1],l.rotation[2]=p[2];const g=p[0]*p[0]+p[1]*p[1]+p[2]*p[2];l.rotation[3]=Math.sqrt(Math.max(0,1-g)),l.alpha=Math.floor(i.alphas[h]);for(let _=0;_<3;_++)l.color[_]=Math.floor(((i.colors[h*3+_]/255-.5)/YT*d+.5)*255);for(let _=0;_<3;_++)for(let A=0;A<f;A++)l.sh[_*f+A]=jT(i.sh[f*3*h+A*3+_]);const m=$T(l,i.shDegree,e);if(t){const _=K.CompressionLevels[0].SphericalHarmonicsDegrees[e].BytesPerSplat,A=h*_+s;K.writeSplatDataToSectionBuffer(m,n,A,0,e)}else n.addSplat(m)}}const JT=16,eE=1e7;function tE(i){const e=new DataView(i);let t=0;const n={magic:e.getUint32(t,!0),version:e.getUint32(t+4,!0),numPoints:e.getUint32(t+8,!0),shDegree:e.getUint8(t+12),fractionalBits:e.getUint8(t+13),flags:e.getUint8(t+14),reserved:e.getUint8(t+15)};if(t+=JT,n.magic!==qT)return console.error("[SPZ ERROR] deserializePackedGaussians: header not found"),null;if(n.version<1||n.version>2)return console.error(`[SPZ ERROR] deserializePackedGaussians: version not supported: ${n.version}`),null;if(n.numPoints>eE)return console.error(`[SPZ ERROR] deserializePackedGaussians: Too many points: ${n.numPoints}`),null;if(n.shDegree>3)return console.error(`[SPZ ERROR] deserializePackedGaussians: Unsupported SH degree: ${n.shDegree}`),null;const s=n.numPoints,r=rr(n.shDegree),o=n.version===1,a={numPoints:s,shDegree:n.shDegree,fractionalBits:n.fractionalBits,antialiased:(n.flags&QT)!==0,positions:new Uint8Array(s*3*(o?2:3)),scales:new Uint8Array(s*3),rotations:new Uint8Array(s*3),alphas:new Uint8Array(s),colors:new Uint8Array(s*3),sh:new Uint8Array(s*r*3)};try{const l=new Uint8Array(i);let c=a.positions.length,u=t;if(a.positions.set(l.slice(u,u+c)),u+=c,a.alphas.set(l.slice(u,u+a.alphas.length)),u+=a.alphas.length,a.colors.set(l.slice(u,u+a.colors.length)),u+=a.colors.length,a.scales.set(l.slice(u,u+a.scales.length)),u+=a.scales.length,a.rotations.set(l.slice(u,u+a.rotations.length)),u+=a.rotations.length,a.sh.set(l.slice(u,u+a.sh.length)),u+a.sh.length!==i.byteLength)return console.error("[SPZ ERROR] deserializePackedGaussians: incorrect buffer size"),null}catch(l){return console.error("[SPZ ERROR] deserializePackedGaussians: read error",l),null}return a}async function nE(i){try{const e=await WT(i);return tE(e.buffer)}catch(e){return console.error("[SPZ ERROR] loadSpzPacked: decompression error",e),null}}class xd{static loadFromURL(e,t,n,s,r=!0,o=0,a,l,c,u,f){return t&&t(0,"0%",Nt.Downloading),tc(e,t,!0,a).then(d=>(t&&t(0,"0%",Nt.Processing),xd.loadFromFileData(d,n,s,r,o,l,c,u,f)))}static async loadFromFileData(e,t,n,s,r=0,o,a,l,c){await Gn();const u=await nE(e);r=Math.min(u.shDegree,r);const f=new Ee(r);if(s)return kp(u,r,!1,f,0),Ma.getStandardGenerator(t,n,o,a,l,c).generateFromUncompressedSplatArray(f);{const{splatBuffer:d,splatBufferDataOffsetBytes:h}=K.preallocateUncompressed(u.numPoints,r);return kp(u,r,!0,d.bufferData,h),d}}}class dt{static RowSizeBytes=32;static CenterSizeBytes=12;static ScaleSizeBytes=12;static RotationSizeBytes=4;static ColorSizeBytes=4;static parseToUncompressedSplatBufferSection(e,t,n,s,r,o){const a=K.CompressionLevels[0].BytesPerCenter,l=K.CompressionLevels[0].BytesPerScale,c=K.CompressionLevels[0].BytesPerRotation,u=K.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat;for(let f=e;f<=t;f++){const d=f*dt.RowSizeBytes+s,h=new Float32Array(n,d,3),x=new Float32Array(n,d+dt.CenterSizeBytes,3),p=new Uint8Array(n,d+dt.CenterSizeBytes+dt.ScaleSizeBytes,4),g=new Uint8Array(n,d+dt.CenterSizeBytes+dt.ScaleSizeBytes+dt.RotationSizeBytes,4),m=new bt((g[1]-128)/128,(g[2]-128)/128,(g[3]-128)/128,(g[0]-128)/128);m.normalize();const _=f*u+o,A=new Float32Array(r,_,3),S=new Float32Array(r,_+a,3),v=new Float32Array(r,_+a+l,4),y=new Uint8Array(r,_+a+l+c,4);A[0]=h[0],A[1]=h[1],A[2]=h[2],S[0]=x[0],S[1]=x[1],S[2]=x[2],v[0]=m.w,v[1]=m.x,v[2]=m.y,v[3]=m.z,y[0]=p[0],y[1]=p[1],y[2]=p[2],y[3]=p[3]}}static parseToUncompressedSplatArraySection(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*dt.RowSizeBytes+s,l=new Float32Array(n,a,3),c=new Float32Array(n,a+dt.CenterSizeBytes,3),u=new Uint8Array(n,a+dt.CenterSizeBytes+dt.ScaleSizeBytes,4),f=new Uint8Array(n,a+dt.CenterSizeBytes+dt.ScaleSizeBytes+dt.RotationSizeBytes,4),d=new bt((f[1]-128)/128,(f[2]-128)/128,(f[3]-128)/128,(f[0]-128)/128);d.normalize(),r.addSplatFromComonents(l[0],l[1],l[2],c[0],c[1],c[2],d.w,d.x,d.y,d.z,u[0],u[1],u[2],u[3])}}static parseStandardSplatToUncompressedSplatArray(e){const t=e.byteLength/dt.RowSizeBytes,n=new Ee;for(let s=0;s<t;s++){const r=s*dt.RowSizeBytes,o=new Float32Array(e,r,3),a=new Float32Array(e,r+dt.CenterSizeBytes,3),l=new Uint8Array(e,r+dt.CenterSizeBytes+dt.ScaleSizeBytes,4),c=new Uint8Array(e,r+dt.CenterSizeBytes+dt.ScaleSizeBytes+dt.ColorSizeBytes,4),u=new bt((c[1]-128)/128,(c[2]-128)/128,(c[3]-128)/128,(c[0]-128)/128);u.normalize(),n.addSplatFromComonents(o[0],o[1],o[2],a[0],a[1],a[2],u.w,u.x,u.y,u.z,l[0],l[1],l[2],l[3])}return n}}function Hp(i,e,t,n,s,r,o,a){return e?Ma.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):K.generateFromUncompressedSplatArrays([i],t,0,new B)}class _d{static loadFromURL(e,t,n,s,r,o,a=!0,l,c,u,f,d){let h=n?yt.ProgressiveToSplatBuffer:yt.ProgressiveToSplatArray;a&&(h=yt.ProgressiveToSplatArray);const x=K.HeaderSizeBytes+K.SectionHeaderSizeBytes,p=pt.ProgressiveLoadSectionSize,g=1;let m,_,A,S=0,v=0,y;const M=rd();let E=0,b=0,C=[];const I=(F,U,O,k)=>{const z=F>=100;if(O&&C.push(O),h===yt.DownloadBeforeProcessing){z&&M.resolve(C);return}if(!k){if(n)throw new Dl("Cannon directly load .splat because no file size info is available.");h=yt.DownloadBeforeProcessing;return}if(!m){S=k/dt.RowSizeBytes,m=new ArrayBuffer(k);const V=K.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,H=x+V*S;h===yt.ProgressiveToSplatBuffer?(_=new ArrayBuffer(H),K.writeHeaderToBuffer({versionMajor:K.CurrentMajorVersion,versionMinor:K.CurrentMinorVersion,maxSectionCount:g,sectionCount:g,maxSplatCount:S,splatCount:v,compressionLevel:0,sceneCenter:new B},_)):y=new Ee(0)}if(O){new Uint8Array(m,b,O.byteLength).set(new Uint8Array(O)),b+=O.byteLength;const V=b-E;if(V>p||z){const $=(z?V:p)/dt.RowSizeBytes,oe=v+$;h===yt.ProgressiveToSplatBuffer?dt.parseToUncompressedSplatBufferSection(v,oe-1,m,0,_,x):dt.parseToUncompressedSplatArraySection(v,oe-1,m,0,y),v=oe,h===yt.ProgressiveToSplatBuffer&&(A||(K.writeSectionHeaderToBuffer({maxSplatCount:S,splatCount:v,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0},0,_,K.HeaderSizeBytes),A=new K(_,!1)),A.updateLoadedCounts(1,v),s&&s(A,z)),E+=p}}z&&(h===yt.ProgressiveToSplatBuffer?M.resolve(A):M.resolve(y)),t&&t(F,U,Nt.Downloading)};return t&&t(0,"0%",Nt.Downloading),tc(e,I,!1,l).then(()=>(t&&t(0,"0%",Nt.Processing),M.promise.then(F=>(t&&t(100,"100%",Nt.Done),h===yt.DownloadBeforeProcessing?new Blob(C).arrayBuffer().then(U=>_d.loadFromFileData(U,r,o,a,c,u,f,d)):h===yt.ProgressiveToSplatBuffer?F:Gn(()=>Hp(F,a,r,o,c,u,f,d))))))}static loadFromFileData(e,t,n,s,r,o,a,l){return Gn(()=>{const c=dt.parseStandardSplatToUncompressedSplatArray(e);return Hp(c,s,t,n,r,o,a,l)})}}class qo{static checkVersion(e){const t=K.CurrentMajorVersion,n=K.CurrentMinorVersion,s=K.parseHeader(e);if(s.versionMajor===t&&s.versionMinor>=n||s.versionMajor>t)return!0;throw new Error(`KSplat version not supported: v${s.versionMajor}.${s.versionMinor}. Minimum required: v${t}.${n}`)}static loadFromURL(e,t,n,s,r){let o,a,l,c,u=!1,f=!1,d,h=[],x=!1,p=!1,g=0,m=0,_=0,A=!1,S=!1,v=!1,y=[];const M=rd(),E=()=>{!u&&!f&&g>=K.HeaderSizeBytes&&(f=!0,new Blob(y).arrayBuffer().then(k=>{l=new ArrayBuffer(K.HeaderSizeBytes),new Uint8Array(l).set(new Uint8Array(k,0,K.HeaderSizeBytes)),qo.checkVersion(l),f=!1,u=!0,c=K.parseHeader(l),window.setTimeout(()=>{I()},1)}))};let b=0;const C=()=>{b===0&&(b++,window.setTimeout(()=>{b--,F()},1))},I=()=>{const O=()=>{p=!0,new Blob(y).arrayBuffer().then(z=>{p=!1,x=!0,d=new ArrayBuffer(c.maxSectionCount*K.SectionHeaderSizeBytes),new Uint8Array(d).set(new Uint8Array(z,K.HeaderSizeBytes,c.maxSectionCount*K.SectionHeaderSizeBytes)),h=K.parseSectionHeaders(c,d,0,!1);let V=0;for(let $=0;$<c.maxSectionCount;$++)V+=h[$].storageSizeBytes;const H=K.HeaderSizeBytes+c.maxSectionCount*K.SectionHeaderSizeBytes+V;if(!o){o=new ArrayBuffer(H);let $=0;for(let oe=0;oe<y.length;oe++){const Se=y[oe];new Uint8Array(o,$,Se.byteLength).set(new Uint8Array(Se)),$+=Se.byteLength}}_=K.HeaderSizeBytes+K.SectionHeaderSizeBytes*c.maxSectionCount;for(let $=0;$<=h.length&&$<c.maxSectionCount;$++)_+=h[$].storageSizeBytes;C()})};!p&&!x&&u&&g>=K.HeaderSizeBytes+K.SectionHeaderSizeBytes*c.maxSectionCount&&O()},F=()=>{if(v)return;v=!0;const O=()=>{if(v=!1,x){if(S)return;if(A=g>=_,g-m>pt.ProgressiveLoadSectionSize||A){m+=pt.ProgressiveLoadSectionSize,S=m>=_,a||(a=new K(o,!1));const z=K.HeaderSizeBytes+K.SectionHeaderSizeBytes*c.maxSectionCount;let V=0,H=0,$=0;for(let we=0;we<c.maxSectionCount;we++){const Le=h[we],fe=V+Le.partiallyFilledBucketCount*4+Le.bucketStorageSizeBytes*Le.bucketCount,re=z+fe;if(m>=re){H++;const X=m-re,be=K.CompressionLevels[c.compressionLevel].SphericalHarmonicsDegrees[Le.sphericalHarmonicsDegree].BytesPerSplat;let xe=Math.floor(X/be);xe=Math.min(xe,Le.maxSplatCount),$+=xe,a.updateLoadedCounts(H,$),a.updateSectionLoadedCounts(we,xe)}else break;V+=Le.storageSizeBytes}s(a,S);const oe=m/_*100,Se=oe.toFixed(2)+"%";t&&t(oe,Se,Nt.Downloading),S?M.resolve(a):F()}}};window.setTimeout(O,pt.ProgressiveLoadSectionDelayDuration)};return tc(e,(O,k,z)=>{z&&(y.push(z),o&&new Uint8Array(o,g,z.byteLength).set(new Uint8Array(z)),g+=z.byteLength),n?(E(),I(),F()):t&&t(O,k,Nt.Downloading)},!n,r).then(O=>(t&&t(0,"0%",Nt.Processing),(n?M.promise:qo.loadFromFileData(O)).then(z=>(t&&t(100,"100%",Nt.Done),z))))}static loadFromFileData(e){return Gn(()=>(qo.checkVersion(e),new K(e)))}static downloadFile=(function(){let e;return function(t,n){const s=new Blob([t.bufferData],{type:"application/octet-stream"});e||(e=document.createElement("a"),document.body.appendChild(e)),e.download=n,e.href=URL.createObjectURL(s),e.click()}})()}const En={Splat:0,KSplat:1,Ply:2,Spz:3},Vp=i=>i.endsWith(".ply")?En.Ply:i.endsWith(".splat")?En.Splat:i.endsWith(".ksplat")?En.KSplat:i.endsWith(".spz")?En.Spz:null,Gp={type:"change"},Jc={type:"start"},Wp={type:"end"},sl=new td,Xp=new vs,iE=Math.cos(70*O0.DEG2RAD);class rl extends mr{constructor(e,t){super(),this.object=e,this.domElement=t,this.domElement.style.touchAction="none",this.enabled=!0,this.target=new B,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"KeyA",UP:"KeyW",RIGHT:"KeyD",BOTTOM:"KeyS"},this.mouseButtons={LEFT:Sr.ROTATE,MIDDLE:Sr.DOLLY,RIGHT:Sr.PAN},this.touches={ONE:vr.ROTATE,TWO:vr.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this.getPolarAngle=function(){return a.phi},this.getAzimuthalAngle=function(){return a.theta},this.getDistance=function(){return this.object.position.distanceTo(this.target)},this.listenToKeyEvents=function(N){N.addEventListener("keydown",T),this._domElementKeyEvents=N},this.stopListenToKeyEvents=function(){this._domElementKeyEvents.removeEventListener("keydown",T),this._domElementKeyEvents=null},this.saveState=function(){n.target0.copy(n.target),n.position0.copy(n.object.position),n.zoom0=n.object.zoom},this.reset=function(){n.target.copy(n.target0),n.object.position.copy(n.position0),n.object.zoom=n.zoom0,this.clearDampedRotation(),this.clearDampedPan(),n.object.updateProjectionMatrix(),n.dispatchEvent(Gp),n.update(),r=s.NONE},this.clearDampedRotation=function(){l.theta=0,l.phi=0},this.clearDampedPan=function(){u.set(0,0,0)},this.update=(function(){const N=new B,J=new bt().setFromUnitVectors(e.up,new B(0,1,0)),me=J.clone().invert(),Te=new B,Pe=new bt,Re=new B,He=2*Math.PI;return function(){J.setFromUnitVectors(e.up,new B(0,1,0)),me.copy(J).invert();const Ie=n.object.position;N.copy(Ie).sub(n.target),N.applyQuaternion(J),a.setFromVector3(N),n.autoRotate&&r===s.NONE&&I(b()),n.enableDamping?(a.theta+=l.theta*n.dampingFactor,a.phi+=l.phi*n.dampingFactor):(a.theta+=l.theta,a.phi+=l.phi);let ve=n.minAzimuthAngle,ye=n.maxAzimuthAngle;isFinite(ve)&&isFinite(ye)&&(ve<-Math.PI?ve+=He:ve>Math.PI&&(ve-=He),ye<-Math.PI?ye+=He:ye>Math.PI&&(ye-=He),ve<=ye?a.theta=Math.max(ve,Math.min(ye,a.theta)):a.theta=a.theta>(ve+ye)/2?Math.max(ve,a.theta):Math.min(ye,a.theta)),a.phi=Math.max(n.minPolarAngle,Math.min(n.maxPolarAngle,a.phi)),a.makeSafe(),n.enableDamping===!0?n.target.addScaledVector(u,n.dampingFactor):n.target.add(u),n.zoomToCursor&&y||n.object.isOrthographicCamera?a.radius=$(a.radius):a.radius=$(a.radius*c),N.setFromSpherical(a),N.applyQuaternion(me),Ie.copy(n.target).add(N),n.object.lookAt(n.target),n.enableDamping===!0?(l.theta*=1-n.dampingFactor,l.phi*=1-n.dampingFactor,u.multiplyScalar(1-n.dampingFactor)):(l.set(0,0,0),u.set(0,0,0));let Ae=!1;if(n.zoomToCursor&&y){let he=null;if(n.object.isPerspectiveCamera){const Oe=N.length();he=$(Oe*c);const We=Oe-he;n.object.position.addScaledVector(S,We),n.object.updateMatrixWorld()}else if(n.object.isOrthographicCamera){const Oe=new B(v.x,v.y,0);Oe.unproject(n.object),n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),Ae=!0;const We=new B(v.x,v.y,0);We.unproject(n.object),n.object.position.sub(We).add(Oe),n.object.updateMatrixWorld(),he=N.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),n.zoomToCursor=!1;he!==null&&(this.screenSpacePanning?n.target.set(0,0,-1).transformDirection(n.object.matrix).multiplyScalar(he).add(n.object.position):(sl.origin.copy(n.object.position),sl.direction.set(0,0,-1).transformDirection(n.object.matrix),Math.abs(n.object.up.dot(sl.direction))<iE?e.lookAt(n.target):(Xp.setFromNormalAndCoplanarPoint(n.object.up,n.target),sl.intersectPlane(Xp,n.target))))}else n.object.isOrthographicCamera&&(n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),Ae=!0);return c=1,y=!1,Ae||Te.distanceToSquared(n.object.position)>o||8*(1-Pe.dot(n.object.quaternion))>o||Re.distanceToSquared(n.target)>0?(n.dispatchEvent(Gp),Te.copy(n.object.position),Pe.copy(n.object.quaternion),Re.copy(n.target),Ae=!1,!0):!1}})(),this.dispose=function(){n.domElement.removeEventListener("contextmenu",le),n.domElement.removeEventListener("pointerdown",ue),n.domElement.removeEventListener("pointercancel",de),n.domElement.removeEventListener("wheel",R),n.domElement.removeEventListener("pointermove",Z),n.domElement.removeEventListener("pointerup",de),n._domElementKeyEvents!==null&&(n._domElementKeyEvents.removeEventListener("keydown",T),n._domElementKeyEvents=null)};const n=this,s={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6};let r=s.NONE;const o=1e-6,a=new Jh,l=new Jh;let c=1;const u=new B,f=new ze,d=new ze,h=new ze,x=new ze,p=new ze,g=new ze,m=new ze,_=new ze,A=new ze,S=new B,v=new ze;let y=!1;const M=[],E={};function b(){return 2*Math.PI/60/60*n.autoRotateSpeed}function C(){return Math.pow(.95,n.zoomSpeed)}function I(N){l.theta-=N}function F(N){l.phi-=N}const U=(function(){const N=new B;return function(me,Te){N.setFromMatrixColumn(Te,0),N.multiplyScalar(-me),u.add(N)}})(),O=(function(){const N=new B;return function(me,Te){n.screenSpacePanning===!0?N.setFromMatrixColumn(Te,1):(N.setFromMatrixColumn(Te,0),N.crossVectors(n.object.up,N)),N.multiplyScalar(me),u.add(N)}})(),k=(function(){const N=new B;return function(me,Te){const Pe=n.domElement;if(n.object.isPerspectiveCamera){const Re=n.object.position;N.copy(Re).sub(n.target);let He=N.length();He*=Math.tan(n.object.fov/2*Math.PI/180),U(2*me*He/Pe.clientHeight,n.object.matrix),O(2*Te*He/Pe.clientHeight,n.object.matrix)}else n.object.isOrthographicCamera?(U(me*(n.object.right-n.object.left)/n.object.zoom/Pe.clientWidth,n.object.matrix),O(Te*(n.object.top-n.object.bottom)/n.object.zoom/Pe.clientHeight,n.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),n.enablePan=!1)}})();function z(N){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c/=N:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function V(N){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c*=N:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function H(N){if(!n.zoomToCursor)return;y=!0;const J=n.domElement.getBoundingClientRect(),me=N.clientX-J.left,Te=N.clientY-J.top,Pe=J.width,Re=J.height;v.x=me/Pe*2-1,v.y=-(Te/Re)*2+1,S.set(v.x,v.y,1).unproject(e).sub(e.position).normalize()}function $(N){return Math.max(n.minDistance,Math.min(n.maxDistance,N))}function oe(N){f.set(N.clientX,N.clientY)}function Se(N){H(N),m.set(N.clientX,N.clientY)}function we(N){x.set(N.clientX,N.clientY)}function Le(N){d.set(N.clientX,N.clientY),h.subVectors(d,f).multiplyScalar(n.rotateSpeed);const J=n.domElement;I(2*Math.PI*h.x/J.clientHeight),F(2*Math.PI*h.y/J.clientHeight),f.copy(d),n.update()}function fe(N){_.set(N.clientX,N.clientY),A.subVectors(_,m),A.y>0?z(C()):A.y<0&&V(C()),m.copy(_),n.update()}function re(N){p.set(N.clientX,N.clientY),g.subVectors(p,x).multiplyScalar(n.panSpeed),k(g.x,g.y),x.copy(p),n.update()}function X(N){H(N),N.deltaY<0?V(C()):N.deltaY>0&&z(C()),n.update()}function ee(N){let J=!1;switch(N.code){case n.keys.UP:N.ctrlKey||N.metaKey||N.shiftKey?F(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):k(0,n.keyPanSpeed),J=!0;break;case n.keys.BOTTOM:N.ctrlKey||N.metaKey||N.shiftKey?F(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):k(0,-n.keyPanSpeed),J=!0;break;case n.keys.LEFT:N.ctrlKey||N.metaKey||N.shiftKey?I(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):k(n.keyPanSpeed,0),J=!0;break;case n.keys.RIGHT:N.ctrlKey||N.metaKey||N.shiftKey?I(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):k(-n.keyPanSpeed,0),J=!0;break}J&&(N.preventDefault(),n.update())}function pe(){if(M.length===1)f.set(M[0].pageX,M[0].pageY);else{const N=.5*(M[0].pageX+M[1].pageX),J=.5*(M[0].pageY+M[1].pageY);f.set(N,J)}}function be(){if(M.length===1)x.set(M[0].pageX,M[0].pageY);else{const N=.5*(M[0].pageX+M[1].pageX),J=.5*(M[0].pageY+M[1].pageY);x.set(N,J)}}function xe(){const N=M[0].pageX-M[1].pageX,J=M[0].pageY-M[1].pageY,me=Math.sqrt(N*N+J*J);m.set(0,me)}function Ce(){n.enableZoom&&xe(),n.enablePan&&be()}function P(){n.enableZoom&&xe(),n.enableRotate&&pe()}function L(N){if(M.length==1)d.set(N.pageX,N.pageY);else{const me=Ue(N),Te=.5*(N.pageX+me.x),Pe=.5*(N.pageY+me.y);d.set(Te,Pe)}h.subVectors(d,f).multiplyScalar(n.rotateSpeed);const J=n.domElement;I(2*Math.PI*h.x/J.clientHeight),F(2*Math.PI*h.y/J.clientHeight),f.copy(d)}function q(N){if(M.length===1)p.set(N.pageX,N.pageY);else{const J=Ue(N),me=.5*(N.pageX+J.x),Te=.5*(N.pageY+J.y);p.set(me,Te)}g.subVectors(p,x).multiplyScalar(n.panSpeed),k(g.x,g.y),x.copy(p)}function w(N){const J=Ue(N),me=N.pageX-J.x,Te=N.pageY-J.y,Pe=Math.sqrt(me*me+Te*Te);_.set(0,Pe),A.set(0,Math.pow(_.y/m.y,n.zoomSpeed)),z(A.y),m.copy(_)}function te(N){n.enableZoom&&w(N),n.enablePan&&q(N)}function ie(N){n.enableZoom&&w(N),n.enableRotate&&L(N)}function ue(N){n.enabled!==!1&&(M.length===0&&(n.domElement.setPointerCapture(N.pointerId),n.domElement.addEventListener("pointermove",Z),n.domElement.addEventListener("pointerup",de)),j(N),N.pointerType==="touch"?G(N):ne(N))}function Z(N){n.enabled!==!1&&(N.pointerType==="touch"?se(N):ge(N))}function de(N){De(N),M.length===0&&(n.domElement.releasePointerCapture(N.pointerId),n.domElement.removeEventListener("pointermove",Z),n.domElement.removeEventListener("pointerup",de)),n.dispatchEvent(Wp),r=s.NONE}function ne(N){let J;switch(N.button){case 0:J=n.mouseButtons.LEFT;break;case 1:J=n.mouseButtons.MIDDLE;break;case 2:J=n.mouseButtons.RIGHT;break;default:J=-1}switch(J){case Sr.DOLLY:if(n.enableZoom===!1)return;Se(N),r=s.DOLLY;break;case Sr.ROTATE:if(N.ctrlKey||N.metaKey||N.shiftKey){if(n.enablePan===!1)return;we(N),r=s.PAN}else{if(n.enableRotate===!1)return;oe(N),r=s.ROTATE}break;case Sr.PAN:if(N.ctrlKey||N.metaKey||N.shiftKey){if(n.enableRotate===!1)return;oe(N),r=s.ROTATE}else{if(n.enablePan===!1)return;we(N),r=s.PAN}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(Jc)}function ge(N){switch(r){case s.ROTATE:if(n.enableRotate===!1)return;Le(N);break;case s.DOLLY:if(n.enableZoom===!1)return;fe(N);break;case s.PAN:if(n.enablePan===!1)return;re(N);break}}function R(N){n.enabled===!1||n.enableZoom===!1||r!==s.NONE||(N.preventDefault(),n.dispatchEvent(Jc),X(N),n.dispatchEvent(Wp))}function T(N){n.enabled===!1||n.enablePan===!1||ee(N)}function G(N){switch(_e(N),M.length){case 1:switch(n.touches.ONE){case vr.ROTATE:if(n.enableRotate===!1)return;pe(),r=s.TOUCH_ROTATE;break;case vr.PAN:if(n.enablePan===!1)return;be(),r=s.TOUCH_PAN;break;default:r=s.NONE}break;case 2:switch(n.touches.TWO){case vr.DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;Ce(),r=s.TOUCH_DOLLY_PAN;break;case vr.DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;P(),r=s.TOUCH_DOLLY_ROTATE;break;default:r=s.NONE}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(Jc)}function se(N){switch(_e(N),r){case s.TOUCH_ROTATE:if(n.enableRotate===!1)return;L(N),n.update();break;case s.TOUCH_PAN:if(n.enablePan===!1)return;q(N),n.update();break;case s.TOUCH_DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;te(N),n.update();break;case s.TOUCH_DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;ie(N),n.update();break;default:r=s.NONE}}function le(N){n.enabled!==!1&&N.preventDefault()}function j(N){M.push(N)}function De(N){delete E[N.pointerId];for(let J=0;J<M.length;J++)if(M[J].pointerId==N.pointerId){M.splice(J,1);return}}function _e(N){let J=E[N.pointerId];J===void 0&&(J=new ze,E[N.pointerId]=J),J.set(N.pageX,N.pageY)}function Ue(N){const J=N.pointerId===M[0].pointerId?M[1]:M[0];return E[J.pointerId]}n.domElement.addEventListener("contextmenu",le),n.domElement.addEventListener("pointerdown",ue),n.domElement.addEventListener("pointercancel",de),n.domElement.addEventListener("wheel",R,{passive:!1}),this.update()}}const sE=(i,e,t,n,s)=>{const r=performance.now();let o=i.style.display==="none"?0:parseFloat(i.style.opacity);isNaN(o)&&(o=1);const a=window.setInterval(()=>{const c=performance.now()-r;let u=Math.min(c/n,1);u>.999&&(u=1);let f;e?(f=(1-u)*o,f<1e-4&&(f=0)):f=(1-o)*u+o,f>0?(i.style.display=t,i.style.opacity=f):i.style.display="none",u>=1&&(s&&s(),window.clearInterval(a))},16);return a},rE=500;class Ad{static elementIDGen=0;constructor(e,t){this.taskIDGen=0,this.elementID=Ad.elementIDGen++,this.tasks=[],this.message=e||"Loading...",this.container=t||document.body,this.spinnerContainerOuter=document.createElement("div"),this.spinnerContainerOuter.className=`spinnerOuterContainer${this.elementID}`,this.spinnerContainerOuter.style.display="none",this.spinnerContainerPrimary=document.createElement("div"),this.spinnerContainerPrimary.className=`spinnerContainerPrimary${this.elementID}`,this.spinnerPrimary=document.createElement("div"),this.spinnerPrimary.classList.add(`spinner${this.elementID}`,`spinnerPrimary${this.elementID}`),this.messageContainerPrimary=document.createElement("div"),this.messageContainerPrimary.classList.add(`messageContainer${this.elementID}`,`messageContainerPrimary${this.elementID}`),this.messageContainerPrimary.innerHTML=this.message,this.spinnerContainerMin=document.createElement("div"),this.spinnerContainerMin.className=`spinnerContainerMin${this.elementID}`,this.spinnerMin=document.createElement("div"),this.spinnerMin.classList.add(`spinner${this.elementID}`,`spinnerMin${this.elementID}`),this.messageContainerMin=document.createElement("div"),this.messageContainerMin.classList.add(`messageContainer${this.elementID}`,`messageContainerMin${this.elementID}`),this.messageContainerMin.innerHTML=this.message,this.spinnerContainerPrimary.appendChild(this.spinnerPrimary),this.spinnerContainerPrimary.appendChild(this.messageContainerPrimary),this.spinnerContainerOuter.appendChild(this.spinnerContainerPrimary),this.spinnerContainerMin.appendChild(this.spinnerMin),this.spinnerContainerMin.appendChild(this.messageContainerMin),this.spinnerContainerOuter.appendChild(this.spinnerContainerMin);const n=document.createElement("style");n.innerHTML=`

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

        `,this.spinnerContainerOuter.appendChild(n),this.container.appendChild(this.spinnerContainerOuter),this.setMinimized(!1,!0),this.fadeTransitions=[]}addTask(e){const t={message:e,id:this.taskIDGen++};return this.tasks.push(t),this.update(),t.id}removeTask(e){let t=0;for(let n of this.tasks){if(n.id===e){this.tasks.splice(t,1);break}t++}this.update()}removeAllTasks(){this.tasks=[],this.update()}setMessageForTask(e,t){for(let n of this.tasks)if(n.id===e){n.message=t;break}this.update()}update(){this.tasks.length>0?(this.show(),this.setMessage(this.tasks[this.tasks.length-1].message)):this.hide()}show(){this.spinnerContainerOuter.style.display="block",this.visible=!0}hide(){this.spinnerContainerOuter.style.display="none",this.visible=!1}setContainer(e){this.container&&this.spinnerContainerOuter.parentElement===this.container&&this.container.removeChild(this.spinnerContainerOuter),e&&(this.container=e,this.container.appendChild(this.spinnerContainerOuter),this.spinnerContainerOuter.style.zIndex=this.container.style.zIndex+1)}setMinimized(e,t){const n=(s,r,o,a,l)=>{o?s.style.display=r?a:"none":this.fadeTransitions[l]=sE(s,!r,a,rE,()=>{this.fadeTransitions[l]=null})};n(this.spinnerContainerPrimary,!e,t,"block",0),n(this.spinnerContainerMin,e,t,"flex",1),this.minimized=e}setMessage(e){this.messageContainerPrimary.innerHTML=e,this.messageContainerMin.innerHTML=e}}class oE{constructor(e){this.idGen=0,this.tasks=[],this.container=e||document.body,this.progressBarContainerOuter=document.createElement("div"),this.progressBarContainerOuter.className="progressBarOuterContainer",this.progressBarContainerOuter.style.display="none",this.progressBarBox=document.createElement("div"),this.progressBarBox.className="progressBarBox",this.progressBarBackground=document.createElement("div"),this.progressBarBackground.className="progressBarBackground",this.progressBar=document.createElement("div"),this.progressBar.className="progressBar",this.progressBarBackground.appendChild(this.progressBar),this.progressBarBox.appendChild(this.progressBarBackground),this.progressBarContainerOuter.appendChild(this.progressBarBox);const t=document.createElement("style");t.innerHTML=`

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

        `,this.progressBarContainerOuter.appendChild(t),this.container.appendChild(this.progressBarContainerOuter)}show(){this.progressBarContainerOuter.style.display="block"}hide(){this.progressBarContainerOuter.style.display="none"}setProgress(e){this.progressBar.style.width=e+"%"}setContainer(e){this.container&&this.progressBarContainerOuter.parentElement===this.container&&this.container.removeChild(this.progressBarContainerOuter),e&&(this.container=e,this.container.appendChild(this.progressBarContainerOuter),this.progressBarContainerOuter.style.zIndex=this.container.style.zIndex+1)}}class aE{constructor(e){this.container=e||document.body,this.infoCells={};const t=[["Camera position","cameraPosition"],["Camera look-at","cameraLookAt"],["Camera up","cameraUp"],["Camera mode","orthographicCamera"],["Cursor position","cursorPosition"],["FPS","fps"],["Rendering:","renderSplatCount"],["Sort time","sortTime"],["Render window","renderWindow"],["Focal adjustment","focalAdjustment"],["Splat scale","splatScale"],["Point cloud mode","pointCloudMode"]];this.infoPanelContainer=document.createElement("div");const n=document.createElement("style");n.innerHTML=`

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

        `,this.infoPanelContainer.append(n),this.infoPanel=document.createElement("div"),this.infoPanel.className="infoPanel";const s=document.createElement("div");s.style.display="table";for(let r of t){const o=document.createElement("div");o.style.display="table-row",o.className="info-panel-row";const a=document.createElement("div");a.style.display="table-cell",a.innerHTML=`${r[0]}: `,a.classList.add("info-panel-cell","label-cell");const l=document.createElement("div");l.style.display="table-cell",l.style.width="10px",l.innerHTML=" ",l.className="info-panel-cell";const c=document.createElement("div");c.style.display="table-cell",c.innerHTML="",c.className="info-panel-cell",this.infoCells[r[1]]=c,o.appendChild(a),o.appendChild(l),o.appendChild(c),s.appendChild(o)}this.infoPanel.appendChild(s),this.infoPanelContainer.append(this.infoPanel),this.infoPanelContainer.style.display="none",this.container.appendChild(this.infoPanelContainer),this.visible=!1}update=function(e,t,n,s,r,o,a,l,c,u,f,d,h,x){const p=`${t.x.toFixed(5)}, ${t.y.toFixed(5)}, ${t.z.toFixed(5)}`;if(this.infoCells.cameraPosition.innerHTML!==p&&(this.infoCells.cameraPosition.innerHTML=p),n){const m=n,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cameraLookAt.innerHTML!==_&&(this.infoCells.cameraLookAt.innerHTML=_)}const g=`${s.x.toFixed(5)}, ${s.y.toFixed(5)}, ${s.z.toFixed(5)}`;if(this.infoCells.cameraUp.innerHTML!==g&&(this.infoCells.cameraUp.innerHTML=g),this.infoCells.orthographicCamera.innerHTML=r?"Orthographic":"Perspective",o){const m=o,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cursorPosition.innerHTML=_}else this.infoCells.cursorPosition.innerHTML="N/A";this.infoCells.fps.innerHTML=a,this.infoCells.renderWindow.innerHTML=`${e.x} x ${e.y}`,this.infoCells.renderSplatCount.innerHTML=`${c} splats out of ${l} (${u.toFixed(2)}%)`,this.infoCells.sortTime.innerHTML=`${f.toFixed(3)} ms`,this.infoCells.focalAdjustment.innerHTML=`${d.toFixed(3)}`,this.infoCells.splatScale.innerHTML=`${h.toFixed(3)}`,this.infoCells.pointCloudMode.innerHTML=`${x}`};setContainer(e){this.container&&this.infoPanelContainer.parentElement===this.container&&this.container.removeChild(this.infoPanelContainer),e&&(this.container=e,this.container.appendChild(this.infoPanelContainer),this.infoPanelContainer.style.zIndex=this.container.style.zIndex+1)}show(){this.infoPanelContainer.style.display="block",this.visible=!0}hide(){this.infoPanelContainer.style.display="none",this.visible=!1}}const qp=new B;class lE extends Wt{constructor(e=new B(0,0,1),t=new B(0,0,0),n=1,s=.1,r=16776960,o=n*.2,a=o*.2){super(),this.type="ArrowHelper";const l=new fa(s,s,n,32);l.translate(0,n/2,0);const c=new fa(0,a,o,32);c.translate(0,n,0),this.position.copy(t),this.line=new Vt(l,new hr({color:r,toneMapped:!1})),this.line.matrixAutoUpdate=!1,this.add(this.line),this.cone=new Vt(c,new hr({color:r,toneMapped:!1})),this.cone.matrixAutoUpdate=!1,this.add(this.cone),this.setDirection(e)}setDirection(e){if(e.y>.99999)this.quaternion.set(0,0,0,1);else if(e.y<-.99999)this.quaternion.set(1,0,0,0);else{qp.set(e.z,0,-e.x).normalize();const t=Math.acos(e.y);this.quaternion.setFromAxisAngle(qp,t)}}setColor(e){this.line.material.color.set(e),this.cone.material.color.set(e)}copy(e){return super.copy(e,!1),this.line.copy(e.line),this.cone.copy(e.cone),this}dispose(){this.line.geometry.dispose(),this.line.material.dispose(),this.cone.geometry.dispose(),this.cone.material.dispose()}}class Qo{constructor(e){this.threeScene=e,this.splatRenderTarget=null,this.renderTargetCopyQuad=null,this.renderTargetCopyCamera=null,this.meshCursor=null,this.focusMarker=null,this.controlPlane=null,this.debugRoot=null,this.secondaryDebugRoot=null}updateSplatRenderTargetForRenderDimensions(e,t){this.destroySplatRendertarget(),this.splatRenderTarget=new Us(e,t,{format:gn,stencilBuffer:!1,depthBuffer:!0}),this.splatRenderTarget.depthTexture=new nd(e,t),this.splatRenderTarget.depthTexture.format=uo,this.splatRenderTarget.depthTexture.type=si}destroySplatRendertarget(){this.splatRenderTarget&&(this.splatRenderTarget=null)}setupRenderTargetCopyObjects(){const e={sourceColorTexture:{type:"t",value:null},sourceDepthTexture:{type:"t",value:null}},t=new _n({vertexShader:`
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
            `,uniforms:e,depthWrite:!1,depthTest:!1,transparent:!0,blending:C0,blendSrc:sa,blendSrcAlpha:sa,blendDst:ra,blendDstAlpha:ra});t.extensions.fragDepth=!0,this.renderTargetCopyQuad=new Vt(new po(2,2),t),this.renderTargetCopyCamera=new sd(-1,1,1,-1,0,1)}destroyRenderTargetCopyObjects(){this.renderTargetCopyQuad&&(Hr(this.renderTargetCopyQuad),this.renderTargetCopyQuad=null)}setupMeshCursor(){if(!this.meshCursor){const e=new id(.5,1.5,32),t=new hr({color:16777215}),n=new Vt(e,t);n.rotation.set(0,0,Math.PI),n.position.set(0,1,0);const s=new Vt(e,t);s.position.set(0,-1,0);const r=new Vt(e,t);r.rotation.set(0,0,Math.PI/2),r.position.set(1,0,0);const o=new Vt(e,t);o.rotation.set(0,0,-Math.PI/2),o.position.set(-1,0,0),this.meshCursor=new Wt,this.meshCursor.add(n),this.meshCursor.add(s),this.meshCursor.add(r),this.meshCursor.add(o),this.meshCursor.scale.set(.1,.1,.1),this.threeScene.add(this.meshCursor),this.meshCursor.visible=!1}}destroyMeshCursor(){this.meshCursor&&(Hr(this.meshCursor),this.threeScene.remove(this.meshCursor),this.meshCursor=null)}setMeshCursorVisibility(e){this.meshCursor.visible=e}getMeschCursorVisibility(){return this.meshCursor.visible}setMeshCursorPosition(e){this.meshCursor.position.copy(e)}positionAndOrientMeshCursor(e,t){this.meshCursor.position.copy(e),this.meshCursor.up.copy(t.up),this.meshCursor.lookAt(t.position)}setupFocusMarker(){if(!this.focusMarker){const e=new Il(.5,32,32),t=Qo.buildFocusMarkerMaterial();t.depthTest=!1,t.depthWrite=!1,t.transparent=!0,this.focusMarker=new Vt(e,t)}}destroyFocusMarker(){this.focusMarker&&(Hr(this.focusMarker),this.focusMarker=null)}updateFocusMarker=(function(){const e=new B,t=new qe,n=new B;return function(s,r,o){t.copy(r.matrixWorld).invert(),e.copy(s).applyMatrix4(t),e.normalize().multiplyScalar(10),e.applyMatrix4(r.matrixWorld),n.copy(r.position).sub(s);const a=n.length();this.focusMarker.position.copy(s),this.focusMarker.scale.set(a,a,a),this.focusMarker.material.uniforms.realFocusPosition.value.copy(s),this.focusMarker.material.uniforms.viewport.value.copy(o),this.focusMarker.material.uniformsNeedUpdate=!0}})();setFocusMarkerVisibility(e){this.focusMarker.visible=e}setFocusMarkerOpacity(e){this.focusMarker.material.uniforms.opacity.value=e,this.focusMarker.material.uniformsNeedUpdate=!0}getFocusMarkerOpacity(){return this.focusMarker.material.uniforms.opacity.value}setupControlPlane(){if(!this.controlPlane){const e=new po(1,1);e.rotateX(-Math.PI/2);const t=new hr({color:16777215});t.transparent=!0,t.opacity=.6,t.depthTest=!1,t.depthWrite=!1,t.side=ti;const n=new Vt(e,t),s=new B(0,1,0);s.normalize();const r=new B(0,0,0),o=.5,a=.01,l=56576,c=new lE(s,r,o,a,l,.1,.03);this.controlPlane=new Wt,this.controlPlane.add(n),this.controlPlane.add(c)}}destroyControlPlane(){this.controlPlane&&(Hr(this.controlPlane),this.controlPlane=null)}setControlPlaneVisibility(e){this.controlPlane.visible=e}positionAndOrientControlPlane=(function(){const e=new bt,t=new B(0,1,0);return function(n,s){e.setFromUnitVectors(t,s),this.controlPlane.position.copy(n),this.controlPlane.quaternion.copy(e)}})();addDebugMeshes(){this.debugRoot=this.createDebugMeshes(),this.secondaryDebugRoot=this.createSecondaryDebugMeshes(),this.threeScene.add(this.debugRoot),this.threeScene.add(this.secondaryDebugRoot)}destroyDebugMeshes(){for(let e of[this.debugRoot,this.secondaryDebugRoot])e&&(Hr(e),this.threeScene.remove(e));this.debugRoot=null,this.secondaryDebugRoot=null}createDebugMeshes(e){const t=new Il(1,32,32),n=new Wt,s=(r,o)=>{let a=new Vt(t,Qo.buildDebugMaterial(r));a.renderOrder=e,n.add(a),a.position.fromArray(o)};return s(16711680,[-50,0,0]),s(16711680,[50,0,0]),s(65280,[0,0,-50]),s(65280,[0,0,50]),s(16755200,[5,0,5]),n}createSecondaryDebugMeshes(e){const t=new yo(3,3,3),n=new Wt;let s=12303291;const r=a=>{let l=new Vt(t,Qo.buildDebugMaterial(s));l.renderOrder=e,n.add(l),l.position.fromArray(a)};let o=10;return r([-o,0,-o]),r([-o,0,o]),r([o,0,-o]),r([o,0,o]),n}static buildDebugMaterial(e){const t=`
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
        `,s={color:{type:"v3",value:new nt(e)},realFocusPosition:{type:"v3",value:new B},viewport:{type:"v2",value:new ze},opacity:{value:0}};return new _n({uniforms:s,vertexShader:t,fragmentShader:n,transparent:!0,depthTest:!1,depthWrite:!1,side:Bi})}dispose(){this.destroyMeshCursor(),this.destroyFocusMarker(),this.destroyDebugMeshes(),this.destroyControlPlane(),this.destroyRenderTargetCopyObjects(),this.destroySplatRendertarget()}}const cE=new B(1,0,0),uE=new B(0,1,0),fE=new B(0,0,1);class eu{constructor(e=new B,t=new B){this.origin=new B,this.direction=new B,this.setParameters(e,t)}setParameters(e,t){this.origin.copy(e),this.direction.copy(t).normalize()}boxContainsPoint(e,t,n){return!(t.x<e.min.x-n||t.x>e.max.x+n||t.y<e.min.y-n||t.y>e.max.y+n||t.z<e.min.z-n||t.z>e.max.z+n)}intersectBox=(function(){const e=new B,t=[],n=[],s=[];return function(r,o){if(n[0]=this.origin.x,n[1]=this.origin.y,n[2]=this.origin.z,s[0]=this.direction.x,s[1]=this.direction.y,s[2]=this.direction.z,this.boxContainsPoint(r,this.origin,1e-4))return o&&(o.origin.copy(this.origin),o.normal.set(0,0,0),o.distance=-1),!0;for(let a=0;a<3;a++){if(s[a]==0)continue;const l=a==0?cE:a==1?uE:fE,c=s[a]<0?r.max:r.min;let u=-Math.sign(s[a]);t[0]=a==0?c.x:a==1?c.y:c.z;let f=t[0]-n[a];if(f*u<0){const d=(a+1)%3,h=(a+2)%3;if(t[2]=s[d]/s[a]*f+n[d],t[1]=s[h]/s[a]*f+n[h],e.set(t[a],t[h],t[d]),this.boxContainsPoint(r,e,1e-4))return o&&(o.origin.copy(e),o.normal.copy(l).multiplyScalar(u),o.distance=e.sub(this.origin).length()),!0}}return!1}})();intersectSphere=(function(){const e=new B;return function(t,n,s){e.copy(t).sub(this.origin);const r=e.dot(this.direction),o=r*r,l=e.dot(e)-o,c=n*n;if(l>c)return!1;const u=Math.sqrt(c-l),f=r-u,d=r+u;if(d<0)return!1;let h=f<0?d:f;return s&&(s.origin.copy(this.origin).addScaledVector(this.direction,h),s.normal.copy(s.origin).sub(t).normalize(),s.distance=h),!0}})()}class Sd{constructor(){this.origin=new B,this.normal=new B,this.distance=0,this.splatIndex=0}set(e,t,n,s){this.origin.copy(e),this.normal.copy(t),this.distance=n,this.splatIndex=s}clone(){const e=new Sd;return e.origin.copy(this.origin),e.normal.copy(this.normal),e.distance=this.distance,e.splatIndex=this.splatIndex,e}}const $i={ThreeD:0,TwoD:1};class dE{constructor(e,t,n=!1){this.ray=new eu(e,t),this.raycastAgainstTrueSplatEllipsoid=n}setFromCameraAndScreenPosition=(function(){const e=new ze;return function(t,n,s){if(e.x=n.x/s.x*2-1,e.y=(s.y-n.y)/s.y*2-1,t.isPerspectiveCamera)this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(e.x,e.y,.5).unproject(t).sub(this.ray.origin).normalize(),this.camera=t;else if(t.isOrthographicCamera)this.ray.origin.set(e.x,e.y,(t.near+t.far)/(t.near-t.far)).unproject(t),this.ray.direction.set(0,0,-1).transformDirection(t.matrixWorld),this.camera=t;else throw new Error("Raycaster::setFromCameraAndScreenPosition() -> Unsupported camera type")}})();intersectSplatMesh=(function(){const e=new qe,t=new qe,n=new qe,s=new eu,r=new B;return function(o,a=[]){const l=o.getSplatTree();if(l){for(let c=0;c<l.subTrees.length;c++){const u=l.subTrees[c];t.copy(o.matrixWorld),o.dynamicMode&&(o.getSceneTransform(c,n),t.multiply(n)),e.copy(t).invert(),s.origin.copy(this.ray.origin).applyMatrix4(e),s.direction.copy(this.ray.origin).add(this.ray.direction),s.direction.applyMatrix4(e).sub(s.origin).normalize();const f=[];u.rootNode&&this.castRayAtSplatTreeNode(s,l,u.rootNode,f),f.forEach(d=>{d.origin.applyMatrix4(t),d.normal.applyMatrix4(t).normalize(),d.distance=r.copy(d.origin).sub(this.ray.origin).length()}),a.push(...f)}return a.sort((c,u)=>c.distance>u.distance?1:-1),a}}})();castRayAtSplatTreeNode=(function(){const e=new Et,t=new B,n=new B,s=new bt,r=new Sd,o=1e-7,a=new B(0,0,0),l=new qe,c=new qe,u=new qe,f=new qe,d=new qe,h=new eu;return function(x,p,g,m=[]){if(x.intersectBox(g.boundingBox)){if(g.data&&g.data.indexes&&g.data.indexes.length>0)for(let _=0;_<g.data.indexes.length;_++){const A=g.data.indexes[_],S=p.splatMesh.getSceneIndexForSplat(A);if(p.splatMesh.getScene(S).visible&&(p.splatMesh.getSplatColor(A,e),p.splatMesh.getSplatCenter(A,t),p.splatMesh.getSplatScaleAndRotation(A,n,s),!(n.x<=o||n.y<=o||p.splatMesh.splatRenderMode===$i.ThreeD&&n.z<=o)))if(this.raycastAgainstTrueSplatEllipsoid){c.makeScale(n.x,n.y,n.z),u.makeRotationFromQuaternion(s);const y=Math.log10(e.w)*2;if(l.makeScale(y,y,y),d.copy(l).multiply(u).multiply(c),f.copy(d).invert(),h.origin.copy(x.origin).sub(t).applyMatrix4(f),h.direction.copy(x.origin).add(x.direction).sub(t),h.direction.applyMatrix4(f).sub(h.origin).normalize(),h.intersectSphere(a,1,r)){const M=r.clone();M.splatIndex=A,M.origin.applyMatrix4(d).add(t),m.push(M)}}else{let y=n.x+n.y,M=2;if(p.splatMesh.splatRenderMode===$i.ThreeD&&(y+=n.z,M=3),y=y/M,x.intersectSphere(t,y,r)){const E=r.clone();E.splatIndex=A,m.push(E)}}}if(g.children&&g.children.length>0)for(let _ of g.children)this.castRayAtSplatTreeNode(x,p,_,m);return m}}})()}class eo{static buildVertexShaderBase(e=!1,t=!1,n=0,s=""){let r=`
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
        `}static getUniforms(e=!1,t=!1,n=0,s=1,r=!1){const o={sceneCenter:{type:"v3",value:new B},fadeInComplete:{type:"i",value:0},orthographicMode:{type:"i",value:0},visibleRegionFadeStartRadius:{type:"f",value:0},visibleRegionRadius:{type:"f",value:0},currentTime:{type:"f",value:0},firstRenderTime:{type:"f",value:0},centersColorsTexture:{type:"t",value:null},sphericalHarmonicsTexture:{type:"t",value:null},sphericalHarmonicsTextureR:{type:"t",value:null},sphericalHarmonicsTextureG:{type:"t",value:null},sphericalHarmonicsTextureB:{type:"t",value:null},sphericalHarmonics8BitCompressionRangeMin:{type:"f",value:[]},sphericalHarmonics8BitCompressionRangeMax:{type:"f",value:[]},focal:{type:"v2",value:new ze},orthoZoom:{type:"f",value:1},inverseFocalAdjustment:{type:"f",value:1},viewport:{type:"v2",value:new ze},basisViewport:{type:"v2",value:new ze},debugColor:{type:"v3",value:new nt},centersColorsTextureSize:{type:"v2",value:new ze(1024,1024)},sphericalHarmonicsDegree:{type:"i",value:n},sphericalHarmonicsTextureSize:{type:"v2",value:new ze(1024,1024)},sphericalHarmonics8BitMode:{type:"i",value:0},sphericalHarmonicsMultiTextureMode:{type:"i",value:0},splatScale:{type:"f",value:s},pointCloudModeEnabled:{type:"i",value:r?1:0},sceneIndexesTexture:{type:"t",value:null},sceneIndexesTextureSize:{type:"v2",value:new ze(1024,1024)},sceneCount:{type:"i",value:1}};for(let a=0;a<pt.MaxScenes;a++)o.sphericalHarmonics8BitCompressionRangeMin.value.push(-3/2),o.sphericalHarmonics8BitCompressionRangeMax.value.push(pt.SphericalHarmonics8BitCompressionRange/2);if(t){const a=[];for(let c=0;c<pt.MaxScenes;c++)a.push(1);o.sceneOpacity={type:"f",value:a};const l=[];for(let c=0;c<pt.MaxScenes;c++)l.push(1);o.sceneVisibility={type:"i",value:l}}if(e){const a=[];for(let l=0;l<pt.MaxScenes;l++)a.push(new qe);o.transforms={type:"mat4",value:a}}return o}}class Pl{static build(e=!1,t=!1,n=!1,s=2048,r=1,o=!1,a=0,l=.3){let u=eo.buildVertexShaderBase(e,t,a,`
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
        `);u+=Pl.buildVertexShaderProjection(n,t,s,l);const f=Pl.buildFragmentShader(),d=eo.getUniforms(e,t,a,r,o);return d.covariancesTextureSize={type:"v2",value:new ze(1024,1024)},d.covariancesTexture={type:"t",value:null},d.covariancesTextureHalfFloat={type:"t",value:null},d.covariancesAreHalfFloat={type:"i",value:0},new _n({uniforms:d,vertexShader:u,fragmentShader:f,transparent:!0,alphaTest:1,blending:Is,depthTest:!0,depthWrite:!1,side:ti})}static buildVertexShaderProjection(e,t,n,s){let r=`

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
        `,r+=eo.getVertexShaderFadeIn(),r+="}",r}static buildFragmentShader(){let e=`
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
        `,e}}class Fl{static build(e=!1,t=!1,n=1,s=!1,r=0){let a=eo.buildVertexShaderBase(e,t,r,`
            uniform vec2 scaleRotationsTextureSize;
            uniform highp sampler2D scaleRotationsTexture;
            varying mat3 vT;
            varying vec2 vQuadCenter;
            varying vec2 vFragCoord;
        `);a+=Fl.buildVertexShaderProjection();const l=Fl.buildFragmentShader(),c=eo.getUniforms(e,t,r,n,s);return c.scaleRotationsTexture={type:"t",value:null},c.scaleRotationsTextureSize={type:"v2",value:new ze(1024,1024)},new _n({uniforms:c,vertexShader:a,fragmentShader:l,transparent:!0,alphaTest:1,blending:Is,depthTest:!0,depthWrite:!1,side:ti})}static buildVertexShaderProjection(){let e=`

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
            `,e+=eo.getVertexShaderFadeIn(),e+="}",e}static buildFragmentShader(){return`
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
        `}}class hE{static build(e){const t=new An;t.setIndex([0,1,2,0,2,3]);const n=new Float32Array(12),s=new li(n,3);t.setAttribute("position",s),s.setXYZ(0,-1,-1,0),s.setXYZ(1,-1,1,0),s.setXYZ(2,1,1,0),s.setXYZ(3,1,-1,0),s.needsUpdate=!0;const r=new xv().copy(t),o=new Uint32Array(e),a=new cv(o,1,!1);return a.setUsage(DS),r.setAttribute("splatIndex",a),r.instanceCount=0,r}}class pE extends Wt{constructor(e,t=new B,n=new bt,s=new B(1,1,1),r=1,o=1,a=!0){super(),this.splatBuffer=e,this.position.copy(t),this.quaternion.copy(n),this.scale.copy(s),this.transform=new qe,this.minimumAlpha=r,this.opacity=o,this.visible=a}copyTransformData(e){this.position.copy(e.position),this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.transform.copy(e.transform)}updateTransform(e){e?(this.matrixWorldAutoUpdate&&this.updateWorldMatrix(!0,!1),this.transform.copy(this.matrixWorld)):(this.matrixAutoUpdate&&this.updateMatrix(),this.transform.copy(this.matrix))}}class vd{static idGen=0;constructor(e,t,n,s){this.min=new B().copy(e),this.max=new B().copy(t),this.boundingBox=new wi(this.min,this.max),this.center=new B().copy(this.max).sub(this.min).multiplyScalar(.5).add(this.min),this.depth=n,this.children=[],this.data=null,this.id=s||vd.idGen++}}class Yo{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.sceneDimensions=new B,this.sceneMin=new B,this.sceneMax=new B,this.rootNode=null,this.nodesWithIndexes=[],this.splatMesh=null}static convertWorkerSubTreeNode(e){const t=new B().fromArray(e.min),n=new B().fromArray(e.max),s=new vd(t,n,e.depth,e.id);if(e.data.indexes){s.data={indexes:[]};for(let r of e.data.indexes)s.data.indexes.push(r)}if(e.children)for(let r of e.children)s.children.push(Yo.convertWorkerSubTreeNode(r));return s}static convertWorkerSubTree(e,t){const n=new Yo(e.maxDepth,e.maxCentersPerNode);n.sceneMin=new B().fromArray(e.sceneMin),n.sceneMax=new B().fromArray(e.sceneMax),n.splatMesh=t,n.rootNode=Yo.convertWorkerSubTreeNode(e.rootNode);const s=(r,o)=>{r.children.length===0&&o(r);for(let a of r.children)s(a,o)};return n.nodesWithIndexes=[],s(n.rootNode,r=>{r.data&&r.data.indexes&&r.data.indexes.length>0&&n.nodesWithIndexes.push(r)}),n}}function mE(i){let e=0;class t{constructor(l,c){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]]}containsPoint(l){return l[0]>=this.min[0]&&l[0]<=this.max[0]&&l[1]>=this.min[1]&&l[1]<=this.max[1]&&l[2]>=this.min[2]&&l[2]<=this.max[2]}}class n{constructor(l,c){this.maxDepth=l,this.maxCentersPerNode=c,this.sceneDimensions=[],this.sceneMin=[],this.sceneMax=[],this.rootNode=null,this.addedIndexes={},this.nodesWithIndexes=[],this.splatMesh=null,this.disposed=!1}}class s{constructor(l,c,u,f){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]],this.center=[(c[0]-l[0])*.5+l[0],(c[1]-l[1])*.5+l[1],(c[2]-l[2])*.5+l[2]],this.depth=u,this.children=[],this.data=null,this.id=f||e++}}processSplatTreeNode=function(a,l,c,u){const f=l.data.indexes.length;if(f<a.maxCentersPerNode||l.depth>a.maxDepth){const _=[];for(let A=0;A<l.data.indexes.length;A++)a.addedIndexes[l.data.indexes[A]]||(_.push(l.data.indexes[A]),a.addedIndexes[l.data.indexes[A]]=!0);l.data.indexes=_,l.data.indexes.sort((A,S)=>A>S?1:-1),a.nodesWithIndexes.push(l);return}const d=[l.max[0]-l.min[0],l.max[1]-l.min[1],l.max[2]-l.min[2]],h=[d[0]*.5,d[1]*.5,d[2]*.5],x=[l.min[0]+h[0],l.min[1]+h[1],l.min[2]+h[2]],p=[new t([x[0]-h[0],x[1],x[2]-h[2]],[x[0],x[1]+h[1],x[2]]),new t([x[0],x[1],x[2]-h[2]],[x[0]+h[0],x[1]+h[1],x[2]]),new t([x[0],x[1],x[2]],[x[0]+h[0],x[1]+h[1],x[2]+h[2]]),new t([x[0]-h[0],x[1],x[2]],[x[0],x[1]+h[1],x[2]+h[2]]),new t([x[0]-h[0],x[1]-h[1],x[2]-h[2]],[x[0],x[1],x[2]]),new t([x[0],x[1]-h[1],x[2]-h[2]],[x[0]+h[0],x[1],x[2]]),new t([x[0],x[1]-h[1],x[2]],[x[0]+h[0],x[1],x[2]+h[2]]),new t([x[0]-h[0],x[1]-h[1],x[2]],[x[0],x[1],x[2]+h[2]])],g=[];for(let _=0;_<p.length;_++)g[_]=[];const m=[0,0,0];for(let _=0;_<f;_++){const A=l.data.indexes[_],S=c[A];m[0]=u[S],m[1]=u[S+1],m[2]=u[S+2];for(let v=0;v<p.length;v++)p[v].containsPoint(m)&&g[v].push(A)}for(let _=0;_<p.length;_++){const A=new s(p[_].min,p[_].max,l.depth+1);A.data={indexes:g[_]},l.children.push(A)}l.data={};for(let _ of l.children)processSplatTreeNode(a,_,c,u)};const r=(a,l,c)=>{const u=[0,0,0],f=[0,0,0],d=[],h=Math.floor(a.length/4);for(let p=0;p<h;p++){const g=p*4,m=a[g],_=a[g+1],A=a[g+2],S=Math.round(a[g+3]);(p===0||m<u[0])&&(u[0]=m),(p===0||m>f[0])&&(f[0]=m),(p===0||_<u[1])&&(u[1]=_),(p===0||_>f[1])&&(f[1]=_),(p===0||A<u[2])&&(u[2]=A),(p===0||A>f[2])&&(f[2]=A),d.push(S)}const x=new n(l,c);return x.sceneMin=u,x.sceneMax=f,x.rootNode=new s(x.sceneMin,x.sceneMax,0),x.rootNode.data={indexes:d},x};function o(a,l,c){const u=[];for(let d of a){const h=Math.floor(d.length/4);for(let x=0;x<h;x++){const p=x*4,g=Math.round(d[p+3]);u[g]=p}}const f=[];for(let d of a){const h=r(d,l,c);f.push(h),processSplatTreeNode(h,h.rootNode,u,d)}i.postMessage({subTrees:f})}i.onmessage=a=>{a.data.process&&o(a.data.process.centers,a.data.process.maxDepth,a.data.process.maxCentersPerNode)}}function gE(i,e,t,n,s){i.postMessage({process:{centers:e,maxDepth:n,maxCentersPerNode:s}},t)}function xE(){return new Worker(URL.createObjectURL(new Blob(["(",mE.toString(),")(self)"],{type:"application/javascript"})))}class _E{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.subTrees=[],this.splatMesh=null}dispose(){this.diposeSplatTreeWorker(),this.disposed=!0}diposeSplatTreeWorker(){this.splatTreeWorker&&this.splatTreeWorker.terminate(),this.splatTreeWorker=null}processSplatMesh=function(e,t=()=>!0,n,s){this.splatTreeWorker||(this.splatTreeWorker=xE()),this.splatMesh=e,this.subTrees=[];const r=new B,o=(a,l)=>{const c=new Float32Array(l*4);let u=0;for(let f=0;f<l;f++){const d=f+a;if(t(d)){e.getSplatCenter(d,r);const h=u*4;c[h]=r.x,c[h+1]=r.y,c[h+2]=r.z,c[h+3]=d,u++}}return c};return new Promise(a=>{const l=()=>this.disposed?(this.diposeSplatTreeWorker(),a(),!0):!1;n&&n(!1),Gn(()=>{if(l())return;const c=[];if(e.dynamicMode){let u=0;for(let f=0;f<e.scenes.length;f++){const h=e.getScene(f).splatBuffer.getSplatCount(),x=o(u,h);c.push(x),u+=h}}else{const u=o(0,e.getSplatCount());c.push(u)}this.splatTreeWorker.onmessage=u=>{l()||u.data.subTrees&&(s&&s(!1),Gn(()=>{if(!l()){for(let f of u.data.subTrees){const d=Yo.convertWorkerSubTree(f,e);this.subTrees.push(d)}this.diposeSplatTreeWorker(),s&&s(!0),Gn(()=>{a()})}}))},Gn(()=>{if(l())return;n&&n(!0);const u=c.map(f=>f.buffer);gE(this.splatTreeWorker,c,u,this.maxDepth,this.maxCentersPerNode)})})})};countLeaves(){let e=0;return this.visitLeaves(()=>{e++}),e}visitLeaves(e){const t=(n,s)=>{n.children.length===0&&s(n);for(let r of n.children)t(r,s)};for(let n of this.subTrees)t(n.rootNode,e)}}function AE(i){const e={};function t(n){if(e[n]!==void 0)return e[n];let s;switch(n){case"WEBGL_depth_texture":s=i.getExtension("WEBGL_depth_texture")||i.getExtension("MOZ_WEBGL_depth_texture")||i.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":s=i.getExtension("EXT_texture_filter_anisotropic")||i.getExtension("MOZ_EXT_texture_filter_anisotropic")||i.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":s=i.getExtension("WEBGL_compressed_texture_s3tc")||i.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":s=i.getExtension("WEBGL_compressed_texture_pvrtc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:s=i.getExtension(n)}return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(n){n.isWebGL2?(t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance")):(t("WEBGL_depth_texture"),t("OES_texture_float"),t("OES_texture_half_float"),t("OES_texture_half_float_linear"),t("OES_standard_derivatives"),t("OES_element_index_uint"),t("OES_vertex_array_object"),t("ANGLE_instanced_arrays")),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture")},get:function(n){const s=t(n);return s===null&&console.warn("THREE.WebGLRenderer: "+n+" extension not supported."),s}}}function SE(i,e,t){let n;function s(){if(n!==void 0)return n;if(e.has("EXT_texture_filter_anisotropic")===!0){const M=e.get("EXT_texture_filter_anisotropic");n=i.getParameter(M.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else n=0;return n}function r(M){if(M==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";M="mediump"}return M==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}const o=typeof WebGL2RenderingContext<"u"&&i.constructor.name==="WebGL2RenderingContext";let a=t.precision!==void 0?t.precision:"highp";const l=r(a);l!==a&&(console.warn("THREE.WebGLRenderer:",a,"not supported, using",l,"instead."),a=l);const c=o||e.has("WEBGL_draw_buffers"),u=t.logarithmicDepthBuffer===!0,f=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),d=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),h=i.getParameter(i.MAX_TEXTURE_SIZE),x=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),p=i.getParameter(i.MAX_VERTEX_ATTRIBS),g=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),m=i.getParameter(i.MAX_VARYING_VECTORS),_=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),A=d>0,S=o||e.has("OES_texture_float"),v=A&&S,y=o?i.getParameter(i.MAX_SAMPLES):0;return{isWebGL2:o,drawBuffers:c,getMaxAnisotropy:s,getMaxPrecision:r,precision:a,logarithmicDepthBuffer:u,maxTextures:f,maxVertexTextures:d,maxTextureSize:h,maxCubemapSize:x,maxAttributes:p,maxVertexUniforms:g,maxVaryings:m,maxFragmentUniforms:_,vertexTextures:A,floatFragmentTextures:S,floatVertexTextures:v,maxSamples:y}}const Ko={Default:0,Instant:2},to={None:0,Info:3},Qp=new An,vE=new hr,ol=6,yE=4,bE=4,ME=4,CE=6,TE=8,tu=4,nu=4,Yp=1,EE=.012,wE=.003,Kp=1,jp=16777216;class $t extends Vt{constructor(e=$i.ThreeD,t=!1,n=!1,s=!1,r=1,o=!0,a=!1,l=!1,c=1024,u=to.None,f=0,d=1,h=.3){super(Qp,vE),this.renderer=void 0,this.splatRenderMode=e,this.dynamicMode=t,this.enableOptionalEffects=n,this.halfPrecisionCovariancesOnGPU=s,this.devicePixelRatio=r,this.enableDistancesComputationOnGPU=o,this.integerBasedDistancesComputation=a,this.antialiased=l,this.kernel2DSize=h,this.maxScreenSpaceSplatSize=c,this.logLevel=u,this.sphericalHarmonicsDegree=f,this.minSphericalHarmonicsDegree=0,this.sceneFadeInRateMultiplier=d,this.scenes=[],this.splatTree=null,this.baseSplatTree=null,this.splatDataTextures={},this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new wi,this.calculatedSceneCenter=new B,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!1,this.lastRenderer=null,this.visible=!1}static buildScenes(e,t,n){const s=[];s.length=t.length;for(let r=0;r<t.length;r++){const o=t[r],a=n[r]||{};let l=a.position||[0,0,0],c=a.rotation||[0,0,0,1],u=a.scale||[1,1,1];const f=new B().fromArray(l),d=new bt().fromArray(c),h=new B().fromArray(u),x=$t.createScene(o,f,d,h,a.splatAlphaRemovalThreshold||1,a.opacity,a.visible);e.add(x),s[r]=x}return s}static createScene(e,t,n,s,r,o=1,a=!0){return new pE(e,t,n,s,r,o,a)}static buildSplatIndexMaps(e){const t=[],n=[];let s=0;for(let r=0;r<e.length;r++){const a=e[r].getMaxSplatCount();for(let l=0;l<a;l++)t[s]=l,n[s]=r,s++}return{localSplatIndexMap:t,sceneIndexMap:n}}buildSplatTree=function(e=[],t,n){return new Promise(s=>{this.disposeSplatTree(),this.baseSplatTree=new _E(8,1e3);const r=performance.now(),o=new Et;this.baseSplatTree.processSplatMesh(this,a=>{this.getSplatColor(a,o);const l=this.getSceneIndexForSplat(a),c=e[l]||1;return o.w>=c},t,n).then(()=>{const a=performance.now()-r;if(this.logLevel>=to.Info&&console.log("SplatTree build: "+a+" ms"),this.disposed)s();else{this.splatTree=this.baseSplatTree,this.baseSplatTree=null;let l=0,c=0,u=0;this.splatTree.visitLeaves(f=>{const d=f.data.indexes.length;d>0&&(c+=d,u++,l++)}),this.logLevel>=to.Info&&(console.log(`SplatTree leaves: ${this.splatTree.countLeaves()}`),console.log(`SplatTree leaves with splats:${l}`),c=c/u,console.log(`Avg splat count per node: ${c}`),console.log(`Total splat count: ${this.getSplatCount()}`)),s()}})})};build(e,t,n=!0,s=!1,r,o,a=!0){this.sceneOptions=t,this.finalBuild=s;const l=$t.getTotalMaxSplatCountForSplatBuffers(e),c=$t.buildScenes(this,e,t);if(n)for(let p=0;p<this.scenes.length&&p<c.length;p++){const g=c[p],m=this.getScene(p);g.copyTransformData(m)}this.scenes=c;let u=3;for(let p of e){const g=p.getMinSphericalHarmonicsDegree();g<u&&(u=g)}this.minSphericalHarmonicsDegree=Math.min(u,this.sphericalHarmonicsDegree);let f=!1;if(e.length!==this.lastBuildScenes.length)f=!0;else for(let p=0;p<e.length;p++)if(e[p]!==this.lastBuildScenes[p].splatBuffer){f=!0;break}let d=!0;if((this.scenes.length!==1||this.lastBuildSceneCount!==this.scenes.length||this.lastBuildMaxSplatCount!==l||f)&&(d=!1),!d){this.boundingBox=new wi,a||(this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.firstRenderTime=-1),this.lastBuildScenes=[],this.lastBuildSplatCount=0,this.lastBuildMaxSplatCount=0,this.disposeMeshData(),this.geometry=hE.build(l),this.splatRenderMode===$i.ThreeD?this.material=Pl.build(this.dynamicMode,this.enableOptionalEffects,this.antialiased,this.maxScreenSpaceSplatSize,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree,this.kernel2DSize):this.material=Fl.build(this.dynamicMode,this.enableOptionalEffects,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree);const p=$t.buildSplatIndexMaps(e);this.globalSplatIndexToLocalSplatIndexMap=p.localSplatIndexMap,this.globalSplatIndexToSceneIndexMap=p.sceneIndexMap}const h=this.getSplatCount(!0);this.enableDistancesComputationOnGPU&&this.setupDistancesComputationTransformFeedback();const x=this.refreshGPUDataFromSplatBuffers(d);for(let p=0;p<this.scenes.length;p++)this.lastBuildScenes[p]=this.scenes[p];return this.lastBuildSplatCount=h,this.lastBuildMaxSplatCount=this.getMaxSplatCount(),this.lastBuildSceneCount=this.scenes.length,s&&this.scenes.length>0&&this.buildSplatTree(t.map(p=>p.splatAlphaRemovalThreshold||1),r,o).then(()=>{this.onSplatTreeReadyCallback&&this.onSplatTreeReadyCallback(this.splatTree),this.onSplatTreeReadyCallback=null}),this.visible=this.scenes.length>0,x}freeIntermediateSplatData(){const e=t=>{delete t.source.data,delete t.image,t.onUpdate=null};delete this.splatDataTextures.baseData.covariances,delete this.splatDataTextures.baseData.centers,delete this.splatDataTextures.baseData.colors,delete this.splatDataTextures.baseData.sphericalHarmonics,delete this.splatDataTextures.centerColors.data,delete this.splatDataTextures.covariances.data,this.splatDataTextures.sphericalHarmonics&&delete this.splatDataTextures.sphericalHarmonics.data,this.splatDataTextures.sceneIndexes&&delete this.splatDataTextures.sceneIndexes.data,this.splatDataTextures.centerColors.texture.needsUpdate=!0,this.splatDataTextures.centerColors.texture.onUpdate=()=>{e(this.splatDataTextures.centerColors.texture)},this.splatDataTextures.covariances.texture.needsUpdate=!0,this.splatDataTextures.covariances.texture.onUpdate=()=>{e(this.splatDataTextures.covariances.texture)},this.splatDataTextures.sphericalHarmonics&&(this.splatDataTextures.sphericalHarmonics.texture?(this.splatDataTextures.sphericalHarmonics.texture.needsUpdate=!0,this.splatDataTextures.sphericalHarmonics.texture.onUpdate=()=>{e(this.splatDataTextures.sphericalHarmonics.texture)}):this.splatDataTextures.sphericalHarmonics.textures.forEach(t=>{t.needsUpdate=!0,t.onUpdate=()=>{e(t)}})),this.splatDataTextures.sceneIndexes&&(this.splatDataTextures.sceneIndexes.texture.needsUpdate=!0,this.splatDataTextures.sceneIndexes.texture.onUpdate=()=>{e(this.splatDataTextures.sceneIndexes.texture)})}dispose(){this.disposeMeshData(),this.disposeTextures(),this.disposeSplatTree(),this.enableDistancesComputationOnGPU&&(this.computeDistancesOnGPUSyncTimeout&&(clearTimeout(this.computeDistancesOnGPUSyncTimeout),this.computeDistancesOnGPUSyncTimeout=null),this.disposeDistancesComputationGPUResources()),this.scenes=[],this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.renderer=null,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new wi,this.calculatedSceneCenter=new B,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!0,this.lastRenderer=null,this.visible=!1}disposeMeshData(){this.geometry&&this.geometry!==Qp&&(this.geometry.dispose(),this.geometry=null),this.material&&(this.material.dispose(),this.material=null)}disposeTextures(){for(let e in this.splatDataTextures)if(this.splatDataTextures.hasOwnProperty(e)){const t=this.splatDataTextures[e];t.texture&&(t.texture.dispose(),t.texture=null)}this.splatDataTextures=null}disposeSplatTree(){this.splatTree&&(this.splatTree.dispose(),this.splatTree=null),this.baseSplatTree&&(this.baseSplatTree.dispose(),this.baseSplatTree=null)}getSplatTree(){return this.splatTree}onSplatTreeReady(e){this.onSplatTreeReadyCallback=e}getDataForDistancesComputation(e,t){const n=this.integerBasedDistancesComputation?this.getIntegerCenters(e,t,!0):this.getFloatCenters(e,t,!0),s=this.getSceneIndexes(e,t);return{centers:n,sceneIndexes:s}}refreshGPUDataFromSplatBuffers(e){const t=this.getSplatCount(!0);this.refreshDataTexturesFromSplatBuffers(e);const n=e?this.lastBuildSplatCount:0,{centers:s,sceneIndexes:r}=this.getDataForDistancesComputation(n,t-1);return this.enableDistancesComputationOnGPU&&this.refreshGPUBuffersForDistancesComputation(s,r,e),{from:n,to:t-1,count:t-n,centers:s,sceneIndexes:r}}refreshGPUBuffersForDistancesComputation(e,t,n=!1){const s=n?this.lastBuildSplatCount:0;this.updateGPUCentersBufferForDistancesComputation(n,e,s),this.updateGPUTransformIndexesBufferForDistancesComputation(n,t,s)}refreshDataTexturesFromSplatBuffers(e){const t=this.getSplatCount(!0),n=this.lastBuildSplatCount,s=t-1;e?this.updateBaseDataFromSplatBuffers(n,s):(this.setupDataTextures(),this.updateBaseDataFromSplatBuffers()),this.updateDataTexturesFromBaseData(n,s),this.updateVisibleRegion(e)}setupDataTextures(){const e=this.getMaxSplatCount(),t=this.getSplatCount(!0);this.disposeTextures();const n=(M,E)=>{const b=new ze(4096,1024);for(;b.x*b.y*M<e*E;)b.y*=2;return b},s=M=>M>=1?CE:bE,r=M=>{const E=s(M),b=n(E,6);return{elementsPerTexelStored:E,texSize:b}};let o=this.getTargetCovarianceCompressionLevel();const a=0,l=this.getTargetSphericalHarmonicsCompressionLevel();let c,u,f;if(this.splatRenderMode===$i.ThreeD){const M=r(o);M.texSize.x*M.texSize.y>jp&&o===0&&(o=1),c=new Float32Array(e*ol)}else u=new Float32Array(e*3),f=new Float32Array(e*4);const d=new Float32Array(e*3),h=new Uint8Array(e*4);let x=Float32Array;l===1?x=Uint16Array:l===2&&(x=Uint8Array);const p=Jr(this.minSphericalHarmonicsDegree),g=this.minSphericalHarmonicsDegree?new x(e*p):void 0,m=n(nu,4),_=new Uint32Array(m.x*m.y*nu);$t.updateCenterColorsPaddedData(0,t-1,d,h,_);const A=new Qi(_,m.x,m.y,$r,si);if(A.internalFormat="RGBA32UI",A.needsUpdate=!0,this.material.uniforms.centersColorsTexture.value=A,this.material.uniforms.centersColorsTextureSize.value.copy(m),this.material.uniformsNeedUpdate=!0,this.splatDataTextures={baseData:{covariances:c,scales:u,rotations:f,centers:d,colors:h,sphericalHarmonics:g},centerColors:{data:_,texture:A,size:m}},this.splatRenderMode===$i.ThreeD){const M=r(o),E=M.elementsPerTexelStored,b=M.texSize;let C=o>=1?Uint32Array:Float32Array;const I=o>=1?TE:ME,F=new C(b.x*b.y*I);o===0?F.set(c):$t.updatePaddedCompressedCovariancesTextureData(c,F,0,0,c.length);let U;if(o>=1)U=new Qi(F,b.x,b.y,$r,si),U.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=U;else{U=new Qi(F,b.x,b.y,gn,mi),this.material.uniforms.covariancesTexture.value=U;const O=new Qi(new Uint32Array(32),2,2,$r,si);O.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=O,O.needsUpdate=!0}U.needsUpdate=!0,this.material.uniforms.covariancesAreHalfFloat.value=o>=1?1:0,this.material.uniforms.covariancesTextureSize.value.copy(b),this.splatDataTextures.covariances={data:F,texture:U,size:b,compressionLevel:o,elementsPerTexelStored:E,elementsPerTexelAllocated:I}}else{const E=n(tu,6);let b=Float32Array,C=mi;const I=new b(E.x*E.y*tu);$t.updateScaleRotationsPaddedData(0,t-1,u,f,I);const F=new Qi(I,E.x,E.y,gn,C);F.needsUpdate=!0,this.material.uniforms.scaleRotationsTexture.value=F,this.material.uniforms.scaleRotationsTextureSize.value.copy(E),this.splatDataTextures.scaleRotations={data:I,texture:F,size:E,compressionLevel:a}}if(g){const M=l===2?Ui:pr;let E=p;E%2!==0&&E++;const b=4,C=gn;let I=n(b,E);if(I.x*I.y<=jp){const F=I.x*I.y*b,U=new x(F);for(let k=0;k<t;k++){const z=p*k,V=E*k;for(let H=0;H<p;H++)U[V+H]=g[z+H]}const O=new Qi(U,I.x,I.y,C,M);O.needsUpdate=!0,this.material.uniforms.sphericalHarmonicsTexture.value=O,this.splatDataTextures.sphericalHarmonics={componentCount:p,paddedComponentCount:E,data:U,textureCount:1,texture:O,size:I,compressionLevel:l,elementsPerTexel:b}}else{const F=p/3;E=F,E%2!==0&&E++,I=n(b,E);const U=I.x*I.y*b,O=[this.material.uniforms.sphericalHarmonicsTextureR,this.material.uniforms.sphericalHarmonicsTextureG,this.material.uniforms.sphericalHarmonicsTextureB],k=[],z=[];for(let V=0;V<3;V++){const H=new x(U);k.push(H);for(let oe=0;oe<t;oe++){const Se=p*oe,we=E*oe;if(F>=3){for(let Le=0;Le<3;Le++)H[we+Le]=g[Se+V*3+Le];if(F>=8)for(let Le=0;Le<5;Le++)H[we+3+Le]=g[Se+9+V*5+Le]}}const $=new Qi(H,I.x,I.y,C,M);z.push($),$.needsUpdate=!0,O[V].value=$}this.material.uniforms.sphericalHarmonicsMultiTextureMode.value=1,this.splatDataTextures.sphericalHarmonics={componentCount:p,componentCountPerChannel:F,paddedComponentCount:E,data:k,textureCount:3,textures:z,size:I,compressionLevel:l,elementsPerTexel:b}}this.material.uniforms.sphericalHarmonicsTextureSize.value.copy(I),this.material.uniforms.sphericalHarmonics8BitMode.value=l===2?1:0;for(let F=0;F<this.scenes.length;F++){const U=this.scenes[F].splatBuffer;this.material.uniforms.sphericalHarmonics8BitCompressionRangeMin.value[F]=U.minSphericalHarmonicsCoeff,this.material.uniforms.sphericalHarmonics8BitCompressionRangeMax.value[F]=U.maxSphericalHarmonicsCoeff}this.material.uniformsNeedUpdate=!0}const S=n(Yp,4),v=new Uint32Array(S.x*S.y*Yp);for(let M=0;M<t;M++)v[M]=this.globalSplatIndexToSceneIndexMap[M];const y=new Qi(v,S.x,S.y,$l,si);y.internalFormat="R32UI",y.needsUpdate=!0,this.material.uniforms.sceneIndexesTexture.value=y,this.material.uniforms.sceneIndexesTextureSize.value.copy(S),this.material.uniformsNeedUpdate=!0,this.splatDataTextures.sceneIndexes={data:v,texture:y,size:S},this.material.uniforms.sceneCount.value=this.scenes.length}updateBaseDataFromSplatBuffers(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0;this.fillSplatDataArrays(this.splatDataTextures.baseData.covariances,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,this.splatDataTextures.baseData.sphericalHarmonics,void 0,s,o,l,e,t,e)}updateDataTexturesFromBaseData(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0,c=this.splatDataTextures.centerColors,u=c.data,f=c.texture;$t.updateCenterColorsPaddedData(e,t,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,u);const d=this.renderer?this.renderer.properties.get(f):null;if(!d||!d.__webglTexture?f.needsUpdate=!0:this.updateDataTexture(u,c.texture,c.size,d,nu,yE,4,e,t),n){const _=n.texture,A=e*ol,S=t*ol;if(s===0)for(let y=A;y<=S;y++){const M=this.splatDataTextures.baseData.covariances[y];n.data[y]=M}else $t.updatePaddedCompressedCovariancesTextureData(this.splatDataTextures.baseData.covariances,n.data,e*n.elementsPerTexelAllocated,A,S);const v=this.renderer?this.renderer.properties.get(_):null;!v||!v.__webglTexture?_.needsUpdate=!0:s===0?this.updateDataTexture(n.data,n.texture,n.size,v,n.elementsPerTexelStored,ol,4,e,t):this.updateDataTexture(n.data,n.texture,n.size,v,n.elementsPerTexelAllocated,n.elementsPerTexelAllocated,2,e,t)}if(r){const _=r.data,A=r.texture,S=6,v=o===0?4:2;$t.updateScaleRotationsPaddedData(e,t,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,_);const y=this.renderer?this.renderer.properties.get(A):null;!y||!y.__webglTexture?A.needsUpdate=!0:this.updateDataTexture(_,r.texture,r.size,y,tu,S,v,e,t)}const h=this.splatDataTextures.baseData.sphericalHarmonics;if(h){let _=4;l===1?_=2:l===2&&(_=1);const A=(y,M,E,b,C)=>{const I=this.renderer?this.renderer.properties.get(y):null;!I||!I.__webglTexture?y.needsUpdate=!0:this.updateDataTexture(b,y,M,I,E,C,_,e,t)},S=a.componentCount,v=a.paddedComponentCount;if(a.textureCount===1){const y=a.data;for(let M=e;M<=t;M++){const E=S*M,b=v*M;for(let C=0;C<S;C++)y[b+C]=h[E+C]}A(a.texture,a.size,a.elementsPerTexel,y,v)}else{const y=a.componentCountPerChannel;for(let M=0;M<3;M++){const E=a.data[M];for(let b=e;b<=t;b++){const C=S*b,I=v*b;if(y>=3){for(let F=0;F<3;F++)E[I+F]=h[C+M*3+F];if(y>=8)for(let F=0;F<5;F++)E[I+3+F]=h[C+9+M*5+F]}}A(a.textures[M],a.size,a.elementsPerTexel,E,v)}}}const x=this.splatDataTextures.sceneIndexes,p=x.data;for(let _=this.lastBuildSplatCount;_<=t;_++)p[_]=this.globalSplatIndexToSceneIndexMap[_];const g=x.texture,m=this.renderer?this.renderer.properties.get(g):null;!m||!m.__webglTexture?g.needsUpdate=!0:this.updateDataTexture(p,x.texture,x.size,m,1,1,1,this.lastBuildSplatCount,t)}getTargetCovarianceCompressionLevel(){return this.halfPrecisionCovariancesOnGPU?1:0}getTargetSphericalHarmonicsCompressionLevel(){return Math.max(1,this.getMaximumSplatBufferCompressionLevel())}getMaximumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel>e)&&(e=s.compressionLevel)}return e}getMinimumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel<e)&&(e=s.compressionLevel)}return e}static computeTextureUpdateRegion(e,t,n,s,r){const o=r/s,a=e*o,l=Math.floor(a/n),c=l*n*s,u=t*o,f=Math.floor(u/n),d=f*n*s+n*s;return{dataStart:c,dataEnd:d,startRow:l,endRow:f}}updateDataTexture(e,t,n,s,r,o,a,l,c){const u=this.renderer.getContext(),f=$t.computeTextureUpdateRegion(l,c,n.x,r,o),d=f.dataEnd-f.dataStart,h=new e.constructor(e.buffer,f.dataStart*a,d),x=f.endRow-f.startRow+1,p=this.webGLUtils.convert(t.type),g=this.webGLUtils.convert(t.format,t.colorSpace),m=u.getParameter(u.TEXTURE_BINDING_2D);u.bindTexture(u.TEXTURE_2D,s.__webglTexture),u.texSubImage2D(u.TEXTURE_2D,0,0,f.startRow,n.x,x,g,p,h),u.bindTexture(u.TEXTURE_2D,m)}static updatePaddedCompressedCovariancesTextureData(e,t,n,s,r){let o=new DataView(t.buffer),a=n,l=0;for(let c=s;c<=r;c+=2)o.setUint16(a*2,e[c],!0),o.setUint16(a*2+2,e[c+1],!0),a+=2,l++,l>=3&&(a+=2,l=0)}static updateCenterColorsPaddedData(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*4,l=o*3,c=o*4;r[c]=rT(s,a),r[c+1]=Yc(n[l]),r[c+2]=Yc(n[l+1]),r[c+3]=Yc(n[l+2])}}static updateScaleRotationsPaddedData(e,t,n,s,r){for(let a=e;a<=t;a++){const l=a*3,c=a*4,u=a*6;r[u]=n[l],r[u+1]=n[l+1],r[u+2]=n[l+2],r[u+3]=s[c],r[u+4]=s[c+1],r[u+5]=s[c+2]}}updateVisibleRegion(e){const t=this.getSplatCount(!0),n=new B;if(!e){const r=new B;this.scenes.forEach(o=>{r.add(o.splatBuffer.sceneCenter)}),r.multiplyScalar(1/this.scenes.length),this.calculatedSceneCenter.copy(r),this.material.uniforms.sceneCenter.value.copy(this.calculatedSceneCenter),this.material.uniformsNeedUpdate=!0}const s=e?this.lastBuildSplatCount:0;for(let r=s;r<t;r++){this.getSplatCenter(r,n,!0);const o=n.sub(this.calculatedSceneCenter).length();o>this.maxSplatDistanceFromSceneCenter&&(this.maxSplatDistanceFromSceneCenter=o)}this.maxSplatDistanceFromSceneCenter-this.visibleRegionBufferRadius>Kp&&(this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter,this.visibleRegionRadius=Math.max(this.visibleRegionBufferRadius-Kp,0)),this.finalBuild&&(this.visibleRegionRadius=this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter),this.updateVisibleRegionFadeDistance()}updateVisibleRegionFadeDistance(e=Ko.Default){const t=EE*this.sceneFadeInRateMultiplier,n=wE*this.sceneFadeInRateMultiplier,s=this.finalBuild?t:n,r=e===Ko.Default?s:n;this.visibleRegionFadeStartRadius=(this.visibleRegionRadius-this.visibleRegionFadeStartRadius)*r+this.visibleRegionFadeStartRadius;const a=(this.visibleRegionBufferRadius>0?this.visibleRegionFadeStartRadius/this.visibleRegionBufferRadius:0)>.99,l=a||e===Ko.Instant?1:0;this.material.uniforms.visibleRegionFadeStartRadius.value=this.visibleRegionFadeStartRadius,this.material.uniforms.visibleRegionRadius.value=this.visibleRegionRadius,this.material.uniforms.firstRenderTime.value=this.firstRenderTime,this.material.uniforms.currentTime.value=performance.now(),this.material.uniforms.fadeInComplete.value=l,this.material.uniformsNeedUpdate=!0,this.visibleRegionChanging=!a}updateRenderIndexes(e,t){const n=this.geometry;n.attributes.splatIndex.set(e),n.attributes.splatIndex.needsUpdate=!0,t>0&&this.firstRenderTime===-1&&(this.firstRenderTime=performance.now()),n.instanceCount=t,n.setDrawRange(0,t)}updateTransforms(){for(let e=0;e<this.scenes.length;e++)this.getScene(e).updateTransform(this.dynamicMode)}updateUniforms=(function(){const e=new ze;return function(t,n,s,r,o,a){if(this.getSplatCount()>0){if(e.set(t.x*this.devicePixelRatio,t.y*this.devicePixelRatio),this.material.uniforms.viewport.value.copy(e),this.material.uniforms.basisViewport.value.set(1/e.x,1/e.y),this.material.uniforms.focal.value.set(n,s),this.material.uniforms.orthographicMode.value=r?1:0,this.material.uniforms.orthoZoom.value=o,this.material.uniforms.inverseFocalAdjustment.value=a,this.dynamicMode)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.transforms.value[c].copy(this.getScene(c).transform);if(this.enableOptionalEffects)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.sceneOpacity.value[c]=Ct(this.getScene(c).opacity,0,1),this.material.uniforms.sceneVisibility.value[c]=this.getScene(c).visible?1:0,this.material.uniformsNeedUpdate=!0;this.material.uniformsNeedUpdate=!0}}})();setSplatScale(e=1){this.splatScale=e,this.material.uniforms.splatScale.value=e,this.material.uniformsNeedUpdate=!0}getSplatScale(){return this.splatScale}setPointCloudModeEnabled(e){this.pointCloudModeEnabled=e,this.material.uniforms.pointCloudModeEnabled.value=e?1:0,this.material.uniformsNeedUpdate=!0}getPointCloudModeEnabled(){return this.pointCloudModeEnabled}getSplatDataTextures(){return this.splatDataTextures}getSplatCount(e=!1){return e?$t.getTotalSplatCountForScenes(this.scenes):this.lastBuildSplatCount}static getTotalSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getSplatCount());return t}static getTotalSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getSplatCount();return t}getMaxSplatCount(){return $t.getTotalMaxSplatCountForScenes(this.scenes)}static getTotalMaxSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getMaxSplatCount());return t}static getTotalMaxSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getMaxSplatCount();return t}disposeDistancesComputationGPUResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.vao&&(e.deleteVertexArray(this.distancesTransformFeedback.vao),this.distancesTransformFeedback.vao=null),this.distancesTransformFeedback.program&&(e.deleteProgram(this.distancesTransformFeedback.program),e.deleteShader(this.distancesTransformFeedback.vertexShader),e.deleteShader(this.distancesTransformFeedback.fragmentShader),this.distancesTransformFeedback.program=null,this.distancesTransformFeedback.vertexShader=null,this.distancesTransformFeedback.fragmentShader=null),this.disposeDistancesComputationGPUBufferResources(),this.distancesTransformFeedback.id&&(e.deleteTransformFeedback(this.distancesTransformFeedback.id),this.distancesTransformFeedback.id=null)}disposeDistancesComputationGPUBufferResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.centersBuffer&&(this.distancesTransformFeedback.centersBuffer=null,e.deleteBuffer(this.distancesTransformFeedback.centersBuffer)),this.distancesTransformFeedback.outDistancesBuffer&&(e.deleteBuffer(this.distancesTransformFeedback.outDistancesBuffer),this.distancesTransformFeedback.outDistancesBuffer=null)}setRenderer(e){if(e!==this.renderer){this.renderer=e;const t=this.renderer.getContext(),n=new AE(t),s=new SE(t,n,{});if(n.init(s),this.webGLUtils=new J0(t,n),this.enableDistancesComputationOnGPU&&this.getSplatCount()>0){this.setupDistancesComputationTransformFeedback();const{centers:r,sceneIndexes:o}=this.getDataForDistancesComputation(0,this.getSplatCount()-1);this.refreshGPUBuffersForDistancesComputation(r,o)}}}setupDistancesComputationTransformFeedback=(function(){let e;return function(){const t=this.getMaxSplatCount();if(!this.renderer)return;const n=this.lastRenderer!==this.renderer,s=e!==t;if(!n&&!s)return;n?this.disposeDistancesComputationGPUResources():s&&this.disposeDistancesComputationGPUBufferResources();const r=this.renderer.getContext(),o=(d,h,x)=>{const p=d.createShader(h);if(!p)return console.error("Fatal error: gl could not create a shader object."),null;if(d.shaderSource(p,x),d.compileShader(p),!d.getShaderParameter(p,d.COMPILE_STATUS)){let m="unknown";h===d.VERTEX_SHADER?m="vertex shader":h===d.FRAGMENT_SHADER&&(m="fragement shader");const _=d.getShaderInfoLog(p);return console.error("Failed to compile "+m+" with these errors:"+_),d.deleteShader(p),null}return p};let a;this.integerBasedDistancesComputation?(a=`#version 300 es
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
            `,c=r.getParameter(r.VERTEX_ARRAY_BINDING),u=r.getParameter(r.CURRENT_PROGRAM),f=u?r.getProgramParameter(u,r.DELETE_STATUS):!1;if(n&&(this.distancesTransformFeedback.vao=r.createVertexArray()),r.bindVertexArray(this.distancesTransformFeedback.vao),n){const d=r.createProgram(),h=o(r,r.VERTEX_SHADER,a),x=o(r,r.FRAGMENT_SHADER,l);if(!h||!x)throw new Error("Could not compile shaders for distances computation on GPU.");if(r.attachShader(d,h),r.attachShader(d,x),r.transformFeedbackVaryings(d,["distance"],r.SEPARATE_ATTRIBS),r.linkProgram(d),!r.getProgramParameter(d,r.LINK_STATUS)){const g=r.getProgramInfoLog(d);throw console.error("Fatal error: Failed to link program: "+g),r.deleteProgram(d),r.deleteShader(x),r.deleteShader(h),new Error("Could not link shaders for distances computation on GPU.")}this.distancesTransformFeedback.program=d,this.distancesTransformFeedback.vertexShader=h,this.distancesTransformFeedback.vertexShader=x}if(r.useProgram(this.distancesTransformFeedback.program),this.distancesTransformFeedback.centersLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"center"),this.dynamicMode){this.distancesTransformFeedback.sceneIndexesLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"sceneIndex");for(let d=0;d<this.scenes.length;d++)this.distancesTransformFeedback.transformsLocs[d]=r.getUniformLocation(this.distancesTransformFeedback.program,`transforms[${d}]`)}else this.distancesTransformFeedback.modelViewProjLoc=r.getUniformLocation(this.distancesTransformFeedback.program,"modelViewProj");(n||s)&&(this.distancesTransformFeedback.centersBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?r.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,r.INT,0,0):r.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,r.FLOAT,!1,0,0),this.dynamicMode&&(this.distancesTransformFeedback.sceneIndexesBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),r.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,r.UNSIGNED_INT,0,0))),(n||s)&&(this.distancesTransformFeedback.outDistancesBuffer=r.createBuffer()),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),r.bufferData(r.ARRAY_BUFFER,t*4,r.STATIC_READ),n&&(this.distancesTransformFeedback.id=r.createTransformFeedback()),r.bindTransformFeedback(r.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),r.bindBufferBase(r.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),u&&f!==!0&&r.useProgram(u),c&&r.bindVertexArray(c),this.lastRenderer=this.renderer,e=t}})();updateGPUCentersBufferForDistancesComputation(e,t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=this.integerBasedDistancesComputation?Uint32Array:Float32Array,a=16,l=n*a;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,l,t);else{const c=new o(this.getMaxSplatCount()*a);c.set(t),s.bufferData(s.ARRAY_BUFFER,c,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}updateGPUTransformIndexesBufferForDistancesComputation(e,t,n){if(!this.renderer||!this.dynamicMode)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=n*4;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,o,t);else{const a=new Uint32Array(this.getMaxSplatCount()*4);a.set(t),s.bufferData(s.ARRAY_BUFFER,a,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}getSceneIndexes(e,t){let n;const s=t-e+1;n=new Uint32Array(s);for(let r=e;r<=t;r++)n[r]=this.globalSplatIndexToSceneIndexMap[r];return n}fillTransformsArray=(function(){const e=[];return function(t){e.length!==t.length&&(e.length=t.length);for(let n=0;n<this.scenes.length;n++){const r=this.getScene(n).transform.elements;for(let o=0;o<16;o++)e[n*16+o]=r[o]}t.set(e)}})();computeDistancesOnGPU=(function(){const e=new qe;return function(t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING),o=s.getParameter(s.CURRENT_PROGRAM),a=o?s.getProgramParameter(o,s.DELETE_STATUS):!1;if(s.bindVertexArray(this.distancesTransformFeedback.vao),s.useProgram(this.distancesTransformFeedback.program),s.enable(s.RASTERIZER_DISCARD),this.dynamicMode)for(let u=0;u<this.scenes.length;u++)if(e.copy(this.getScene(u).transform),e.premultiply(t),this.integerBasedDistancesComputation){const f=$t.getIntegerMatrixArray(e),d=[f[2],f[6],f[10],f[14]];s.uniform4i(this.distancesTransformFeedback.transformsLocs[u],d[0],d[1],d[2],d[3])}else s.uniformMatrix4fv(this.distancesTransformFeedback.transformsLocs[u],!1,e.elements);else if(this.integerBasedDistancesComputation){const u=$t.getIntegerMatrixArray(t),f=[u[2],u[6],u[10]];s.uniform3i(this.distancesTransformFeedback.modelViewProjLoc,f[0],f[1],f[2])}else{const u=[t.elements[2],t.elements[6],t.elements[10]];s.uniform3f(this.distancesTransformFeedback.modelViewProjLoc,u[0],u[1],u[2])}s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?s.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,s.INT,0,0):s.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,s.FLOAT,!1,0,0),this.dynamicMode&&(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),s.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,s.UNSIGNED_INT,0,0)),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),s.beginTransformFeedback(s.POINTS),s.drawArrays(s.POINTS,0,this.getSplatCount()),s.endTransformFeedback(),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,null),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,null),s.disable(s.RASTERIZER_DISCARD);const l=s.fenceSync(s.SYNC_GPU_COMMANDS_COMPLETE,0);s.flush();const c=new Promise(u=>{const f=()=>{if(this.disposed)u();else switch(s.clientWaitSync(l,0,0)){case s.TIMEOUT_EXPIRED:return this.computeDistancesOnGPUSyncTimeout=setTimeout(f),this.computeDistancesOnGPUSyncTimeout;case s.WAIT_FAILED:throw new Error("should never get here");default:this.computeDistancesOnGPUSyncTimeout=null,s.deleteSync(l);const p=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao),s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),s.getBufferSubData(s.ARRAY_BUFFER,0,n),s.bindBuffer(s.ARRAY_BUFFER,null),p&&s.bindVertexArray(p),u()}};this.computeDistancesOnGPUSyncTimeout=setTimeout(f)});return o&&a!==!0&&s.useProgram(o),r&&s.bindVertexArray(r),c}})();getLocalSplatParameters(e,t,n){n==null&&(n=!this.dynamicMode),t.splatBuffer=this.getSplatBufferForSplat(e),t.localIndex=this.getSplatLocalIndex(e),t.sceneTransform=n?this.getSceneTransformForSplat(e):null}fillSplatDataArrays(e,t,n,s,r,o,a,l=0,c=0,u=1,f,d,h=0,x){const p=new B;p.x=void 0,p.y=void 0,this.splatRenderMode===$i.ThreeD?p.z=void 0:p.z=1;const g=new qe;let m=0,_=this.scenes.length-1;x!=null&&x>=0&&x<=this.scenes.length&&(m=x,_=x);for(let A=m;A<=_;A++){a==null&&(a=!this.dynamicMode);const S=this.getScene(A),v=S.splatBuffer;let y;if(a&&(this.getSceneTransform(A,g),y=g),e&&v.fillSplatCovarianceArray(e,y,f,d,h,l),t||n){if(!t||!n)throw new Error('SplatMesh::fillSplatDataArrays() -> "scales" and "rotations" must both be valid.');v.fillSplatScaleRotationArray(t,n,y,f,d,h,c,p)}s&&v.fillSplatCenterArray(s,y,f,d,h),r&&v.fillSplatColorArray(r,S.minimumAlpha,f,d,h),o&&v.fillSphericalHarmonicsArray(o,this.minSphericalHarmonicsDegree,y,f,d,h,u),h+=v.getSplatCount()}}getIntegerCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e);let o,a=n?4:3;o=new Int32Array(s*a);for(let l=0;l<s;l++){for(let c=0;c<3;c++)o[l*a+c]=Math.round(r[l*3+c]*1e3);n&&(o[l*a+3]=1e3)}return o}getFloatCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);if(this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e),!n)return r;let o=new Float32Array(s*4);for(let a=0;a<s;a++){for(let l=0;l<3;l++)o[a*4+l]=r[a*3+l];o[a*4+3]=1}return o}getSplatCenter=(function(){const e={};return function(t,n,s){this.getLocalSplatParameters(t,e,s),e.splatBuffer.getSplatCenter(e.localIndex,n,e.sceneTransform)}})();getSplatScaleAndRotation=(function(){const e={},t=new B;return function(n,s,r,o){this.getLocalSplatParameters(n,e,o),t.x=void 0,t.y=void 0,t.z=void 0,this.splatRenderMode===$i.TwoD&&(t.z=0),e.splatBuffer.getSplatScaleAndRotation(e.localIndex,s,r,e.sceneTransform,t)}})();getSplatColor=(function(){const e={};return function(t,n){this.getLocalSplatParameters(t,e),e.splatBuffer.getSplatColor(e.localIndex,n)}})();getSceneTransform(e,t){const n=this.getScene(e);n.updateTransform(this.dynamicMode),t.copy(n.transform)}getScene(e){if(e<0||e>=this.scenes.length)throw new Error("SplatMesh::getScene() -> Invalid scene index.");return this.scenes[e]}getSceneCount(){return this.scenes.length}getSplatBufferForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).splatBuffer}getSceneIndexForSplat(e){return this.globalSplatIndexToSceneIndexMap[e]}getSceneTransformForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).transform}getSplatLocalIndex(e){return this.globalSplatIndexToLocalSplatIndexMap[e]}static getIntegerMatrixArray(e){const t=e.elements,n=[];for(let s=0;s<16;s++)n[s]=Math.round(t[s]*1e3);return n}computeBoundingBox(e=!1,t){let n=this.getSplatCount();if(t!=null){if(t<0||t>=this.scenes.length)throw new Error("SplatMesh::computeBoundingBox() -> Invalid scene index.");n=this.scenes[t].splatBuffer.getSplatCount()}const s=new Float32Array(n*3);this.fillSplatDataArrays(null,null,null,s,null,null,e,void 0,void 0,void 0,void 0,t);const r=new B,o=new B;for(let a=0;a<n;a++){const l=a*3,c=s[l],u=s[l+1],f=s[l+2];(a===0||c<r.x)&&(r.x=c),(a===0||u<r.y)&&(r.y=u),(a===0||f<r.z)&&(r.z=f),(a===0||c>o.x)&&(o.x=c),(a===0||u>o.y)&&(o.y=u),(a===0||f>o.z)&&(o.z=f)}return new wi(r,o)}}var RE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEbA2AAAGAQf39/f39/f39/f39/f39/fwBgAAF/AhIBA2VudgZtZW1vcnkCAwCAgAQDBAMAAQIHVAQRX193YXNtX2NhbGxfY3RvcnMAABhfX3dhc21fYXBwbHlfZGF0YV9yZWxvY3MAAAtzb3J0SW5kZXhlcwABE2Vtc2NyaXB0ZW5fdGxzX2luaXQAAgqWEAMDAAELihAEAXwDewN/A30gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEBA0AgAyABQQJ0IgVqIAIgACAFaigCAEECdGooAgAiBTYCACAFIAogBSAKSBshCiAFIA0gBSANShshDSABQQFqIgEgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiFWooAgAiFkECdGooAgAiFEcEQAJ/IAX9CQI4IAggFEEGdGoiDv0JAgwgDioCHP0gASAOKgIs/SACIA4qAjz9IAP95gEgBf0JAiggDv0JAgggDioCGP0gASAOKgIo/SACIA4qAjj9IAP95gEgBf0JAgggDv0JAgAgDioCEP0gASAOKgIg/SACIA4qAjD9IAP95gEgBf0JAhggDv0JAgQgDioCFP0gASAOKgIk/SACIA4qAjT9IAP95gH95AH95AH95AEiEf1f/QwAAAAAAECPQAAAAAAAQI9AIhL98gEiE/0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBP9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/REgDv0cAQJ/IBEgEf0NCAkKCwwNDg8AAAAAAAAAAP1fIBL98gEiEf0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9HAICfyAR/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAyESIBQhDwsgAyAVaiABIBZBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmogEf0bA2oiDjYCACAOIAogCiAOShshCiAOIA0gDSAOSBshDSACQQFqIgIgC0cNAAsMAwsCfyAFKgIIu/0UIAUqAhi7/SIB/QwAAAAAAECPQAAAAAAAQI9A/fIBIhH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIQ4CfyAR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyECAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQIgAv0RIA79HAEgBf0cAiESIAwhBQNAIAMgBUECdCICaiABIAAgAmooAgBBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmoiAjYCACACIAogAiAKSBshCiACIA0gAiANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEBA0AgAyABQQJ0IgVqAn8gAiAAIAVqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAFBAWoiASALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIRcgBSoCGCEYIAUqAgghGUH4////ByEKQYiAgIB4IQ0gDCEFA0ACfyAXIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCAZIAIqAgCUIBggAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIUaigCAEECdCIVaigCACIORwRAIAX9CQI4IAggDkEGdGoiD/0JAgwgDyoCHP0gASAPKgIs/SACIA8qAjz9IAP95gEgBf0JAiggD/0JAgggDyoCGP0gASAPKgIo/SACIA8qAjj9IAP95gEgBf0JAgggD/0JAgAgDyoCEP0gASAPKgIg/SACIA8qAjD9IAP95gEgBf0JAhggD/0JAgQgDyoCFP0gASAPKgIk/SACIA8qAjT9IAP95gH95AH95AH95AEhESAOIQ8LIAMgFGoCfyAR/R8DIAEgFUECdCIOQQxyaioCAJQgEf0fAiABIA5BCHJqKgIAlCAR/R8AIAEgDmoqAgCUIBH9HwEgASAOQQRyaioCAJSSkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSACQQFqIgIgC0cNAAsMAQtBiICAgHghDUH4////ByEKCyALIAxLBEAgCUEBa7MgDbIgCrKTlSEXIAwhDQNAAn8gFyADIA1BAnRqIgEoAgAgCmuylCIYi0MAAABPXQRAIBioDAELQYCAgIB4CyEOIAEgDjYCACAEIA5BAnRqIgEgASgCAEEBajYCACANQQFqIg0gC0cNAAsLIAlBAk8EQCAEKAIAIQ1BASEKA0AgBCAKQQJ0aiIBIAEoAgAgDWoiDTYCACAKQQFqIgogCUcNAAsLIAxBAEoEQCAMIQoDQCAGIApBAWsiAUECdCICaiAAIAJqKAIANgIAIApBAUshAiABIQogAg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCwsEAEEACw==",$p="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACEgEDZW52Bm1lbW9yeQIDAICABAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=",IE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQrrDwICAAvlDwQBfAN7B30DfyALIAprIQwCQAJAIA4EQCANBEBB+P///wchCkGIgICAeCENIAsgDE0NAyAMIQUDQCADIAVBAnQiAWogAiAAIAFqKAIAQQJ0aigCACIBNgIAIAEgCiABIApIGyEKIAEgDSABIA1KGyENIAVBAWoiBSALRw0ACwwDCyAPBEAgCyAMTQ0CQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIcaigCACIdQQJ0aigCACIbRwRAAn8gBf0JAjggCCAbQQZ0aiIO/QkCDCAOKgIc/SABIA4qAiz9IAIgDioCPP0gA/3mASAF/QkCKCAO/QkCCCAOKgIY/SABIA4qAij9IAIgDioCOP0gA/3mASAF/QkCCCAO/QkCACAOKgIQ/SABIA4qAiD9IAIgDioCMP0gA/3mASAF/QkCGCAO/QkCBCAOKgIU/SABIA4qAiT9IAIgDioCNP0gA/3mAf3kAf3kAf3kASIR/V/9DAAAAAAAQI9AAAAAAABAj0AiEv3yASIT/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOAn8gE/0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9ESAO/RwBAn8gESAR/Q0ICQoLDA0ODwABAgMAAQID/V8gEv3yASIR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAgJ/IBH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/RwDIRIgGyEPCyADIBxqIAEgHUEEdGr9AAAAIBL9tQEiEf0bACAR/RsBaiAR/RsCaiAR/RsDaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAgi7/RQgBSoCGLv9IgH9DAAAAAAAQI9AAAAAAABAj0D98gEiEf0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBH9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQL9ESAO/RwBIAX9HAIhEiAMIQUDQCADIAVBAnQiAmogASAAIAJqKAIAQQR0av0AAAAgEv21ASIR/RsAIBH9GwFqIBH9GwJqIgI2AgAgAiAKIAIgCkgbIQogAiANIAIgDUobIQ0gBUEBaiIFIAtHDQALDAILIA0EQEH4////ByEKQYiAgIB4IQ0gCyAMTQ0CIAwhBQNAIAMgBUECdCIBagJ/IAIgACABaigCAEECdGoqAgC7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgD0UEQCALIAxNDQEgBSoCKCEUIAUqAhghFSAFKgIIIRZB+P///wchCkGIgICAeCENIAwhBQNAAn8gFCABIAAgBUECdCIHaigCAEEEdGoiAioCCJQgFiACKgIAlCAVIAIqAgSUkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDiADIAdqIA42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gBUEBaiIFIAtHDQALDAILIAsgDE0NAEF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiG2ooAgBBAnQiHGooAgAiDkcEQCAFKgI4IhQgCCAOQQZ0aiIPKgI8lCAFKgIoIhUgDyoCOJQgBSoCCCIWIA8qAjCUIAUqAhgiFyAPKgI0lJKSkiEYIBQgDyoCLJQgFSAPKgIolCAWIA8qAiCUIBcgDyoCJJSSkpIhGSAUIA8qAhyUIBUgDyoCGJQgFiAPKgIQlCAXIA8qAhSUkpKSIRogFCAPKgIMlCAVIA8qAgiUIBYgDyoCAJQgFyAPKgIElJKSkiEUIA4hDwsgAyAbagJ/IBggASAcQQJ0aiIOKgIMlCAZIA4qAgiUIBQgDioCAJQgGiAOKgIElJKSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAJBAWoiAiALRw0ACwwBC0GIgICAeCENQfj///8HIQoLIAsgDEsEQCAJQQFrsyANsiAKspOVIRQgDCENA0ACfyAUIAMgDUECdGoiASgCACAKa7KUIhWLQwAAAE9dBEAgFagMAQtBgICAgHgLIQ4gASAONgIAIAQgDkECdGoiASABKAIAQQFqNgIAIA1BAWoiDSALRw0ACwsgCUECTwRAIAQoAgAhDUEBIQoDQCAEIApBAnRqIgEgASgCACANaiINNgIAIApBAWoiCiAJRw0ACwsgDEEASgRAIAwhCgNAIAYgCkEBayIBQQJ0IgJqIAAgAmooAgA2AgAgCkEBSyABIQoNAAsLIAsgDEoEQCALIQoDQCAGIAsgBCADIApBAWsiCkECdCIBaigCAEECdGoiAigCACIFa0ECdGogACABaigCADYCACACIAVBAWs2AgAgCiAMSg0ACwsL",DE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=";function PE(i){let e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g,m,_,A,S;function v(y,M,E,b,C,I,F){const U=performance.now();if(!n&&(new Uint32Array(t,a,C.byteLength/S.BytesPerInt).set(C),new Float32Array(t,u,F.byteLength/S.BytesPerFloat).set(F),b)){let H;s?H=new Int32Array(t,f,I.byteLength/S.BytesPerInt):H=new Float32Array(t,f,I.byteLength/S.BytesPerFloat),H.set(I)}g||(g=new Uint32Array(_)),new Float32Array(t,p,16).set(E),new Uint32Array(t,h,_).set(g),e.exports.sortIndexes(a,x,f,d,h,p,l,c,u,_,y,M,o,b,s,r);const O={sortDone:!0,splatSortCount:y,splatRenderCount:M,sortTime:0};if(!n){const z=new Uint32Array(t,l,M);(!m||m.length<M)&&(m=new Uint32Array(M)),m.set(z),O.sortedIndexes=m}const k=performance.now();O.sortTime=k-U,i.postMessage(O)}i.onmessage=y=>{if(y.data.centers)centers=y.data.centers,sceneIndexes=y.data.sceneIndexes,s?new Int32Array(t,x+y.data.range.from*S.BytesPerInt*4,y.data.range.count*4).set(new Int32Array(centers)):new Float32Array(t,x+y.data.range.from*S.BytesPerFloat*4,y.data.range.count*4).set(new Float32Array(centers)),r&&new Uint32Array(t,c+y.data.range.from*4,y.data.range.count).set(new Uint32Array(sceneIndexes)),A=y.data.range.from+y.data.range.count;else if(y.data.sort){const M=Math.min(y.data.sort.splatRenderCount||0,A),E=Math.min(y.data.sort.splatSortCount||0,A),b=y.data.sort.usePrecomputedDistances;let C,I,F;n||(C=y.data.sort.indexesToSort,F=y.data.sort.transforms,b&&(I=y.data.sort.precomputedDistances)),v(E,M,y.data.sort.modelViewProj,b,C,I,F)}else if(y.data.init){S=y.data.init.Constants,o=y.data.init.splatCount,n=y.data.init.useSharedMemory,s=y.data.init.integerBasedSort,r=y.data.init.dynamicMode,_=y.data.init.distanceMapRange,A=0;const M=s?S.BytesPerInt*4:S.BytesPerFloat*4,E=new Uint8Array(y.data.init.sorterWasmBytes),b=16*S.BytesPerFloat,C=o*S.BytesPerInt,I=o*M,F=b,U=s?o*S.BytesPerInt:o*S.BytesPerFloat,O=o*S.BytesPerInt,k=o*S.BytesPerInt,z=s?_*S.BytesPerInt*2:_*S.BytesPerFloat*2,V=r?o*S.BytesPerInt:0,H=r?S.MaxScenes*b:0,$=S.MemoryPageSize*32,oe=C+I+F+U+O+z+k+V+H+$,Se=Math.floor(oe/S.MemoryPageSize)+1,we={module:{},env:{memory:new WebAssembly.Memory({initial:Se,maximum:Se,shared:!0})}};WebAssembly.compile(E).then(Le=>WebAssembly.instantiate(Le,we)).then(Le=>{e=Le,a=0,x=a+C,p=x+I,f=p+F,d=f+U,h=d+O,l=h+z,c=l+k,u=c+V,t=we.env.memory.buffer,n?i.postMessage({sortSetupPhase1Complete:!0,indexesToSortBuffer:t,indexesToSortOffset:a,sortedIndexesBuffer:t,sortedIndexesOffset:l,precomputedDistancesBuffer:t,precomputedDistancesOffset:f,transformsBuffer:t,transformsOffset:u}):i.postMessage({sortSetupPhase1Complete:!0})})}}}function FE(i,e,t,n,s,r=pt.DefaultSplatSortDistanceMapPrecision){const o=new Worker(URL.createObjectURL(new Blob(["(",PE.toString(),")(self)"],{type:"application/javascript"})));let a=RE;const l=od()?tg():null;!t&&!e?(a=$p,l&&l.major<=16&&l.minor<4&&(a=DE)):t?e||l&&l.major<=16&&l.minor<4&&(a=IE):a=$p;const c=atob(a),u=new Uint8Array(c.length);for(let f=0;f<c.length;f++)u[f]=c.charCodeAt(f);return o.postMessage({init:{sorterWasmBytes:u.buffer,splatCount:i,useSharedMemory:e,integerBasedSort:n,dynamicMode:s,distanceMapRange:1<<r,Constants:{BytesPerFloat:pt.BytesPerFloat,BytesPerInt:pt.BytesPerInt,MemoryPageSize:pt.MemoryPageSize,MaxScenes:pt.MaxScenes}}}),o}const ys={None:0,VR:1,AR:2};class mo{static createButton(e,t={}){const n=document.createElement("button");function s(){let c=null;async function u(h){h.addEventListener("end",f),await e.xr.setSession(h),n.textContent="EXIT VR",c=h}function f(){c.removeEventListener("end",f),n.textContent="ENTER VR",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="ENTER VR";const d={...t,optionalFeatures:["local-floor","bounded-floor","layers",...t.optionalFeatures||[]]};n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-vr",d).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",d).then(u).catch(h=>{console.warn(h)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",d).then(u).catch(h=>{console.warn(h)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="VR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="VR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="VRButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-vr").then(function(c){c?s():o(),c&&mo.xrSessionIsGranted&&n.click()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}static registerSessionGrantedListener(){if(typeof navigator<"u"&&"xr"in navigator){if(/WebXRViewer\//i.test(navigator.userAgent))return;navigator.xr.addEventListener("sessiongranted",()=>{mo.xrSessionIsGranted=!0})}}}mo.xrSessionIsGranted=!1;mo.registerSessionGrantedListener();class LE{static createButton(e,t={}){const n=document.createElement("button");function s(){if(t.domOverlay===void 0){const d=document.createElement("div");d.style.display="none",document.body.appendChild(d);const h=document.createElementNS("http://www.w3.org/2000/svg","svg");h.setAttribute("width",38),h.setAttribute("height",38),h.style.position="absolute",h.style.right="20px",h.style.top="20px",h.addEventListener("click",function(){c.end()}),d.appendChild(h);const x=document.createElementNS("http://www.w3.org/2000/svg","path");x.setAttribute("d","M 12,12 L 28,28 M 28,12 12,28"),x.setAttribute("stroke","#fff"),x.setAttribute("stroke-width",2),h.appendChild(x),t.optionalFeatures===void 0&&(t.optionalFeatures=[]),t.optionalFeatures.push("dom-overlay"),t.domOverlay={root:d}}let c=null;async function u(d){d.addEventListener("end",f),e.xr.setReferenceSpaceType("local"),await e.xr.setSession(d),n.textContent="STOP AR",t.domOverlay.root.style.display="",c=d}function f(){c.removeEventListener("end",f),n.textContent="START AR",t.domOverlay.root.style.display="none",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="START AR",n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-ar",t).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(d=>{console.warn(d)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(d=>{console.warn(d)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="AR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="AR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="ARButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-ar").then(function(c){c?s():o()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}}const iu={Always:0,Never:2},BE=50,UE=.75,OE=15e5,NE=10,zE=2.5,kE=60;class Gr{constructor(e={}){if(e.cameraUp||(e.cameraUp=[0,1,0]),this.cameraUp=new B().fromArray(e.cameraUp),e.initialCameraPosition||(e.initialCameraPosition=[0,10,15]),this.initialCameraPosition=new B().fromArray(e.initialCameraPosition),e.initialCameraLookAt||(e.initialCameraLookAt=[0,0,0]),this.initialCameraLookAt=new B().fromArray(e.initialCameraLookAt),this.dropInMode=e.dropInMode||!1,(e.selfDrivenMode===void 0||e.selfDrivenMode===null)&&(e.selfDrivenMode=!0),this.selfDrivenMode=e.selfDrivenMode&&!this.dropInMode,this.selfDrivenUpdateFunc=this.selfDrivenUpdate.bind(this),e.useBuiltInControls===void 0&&(e.useBuiltInControls=!0),this.useBuiltInControls=e.useBuiltInControls,this.rootElement=e.rootElement,this.ignoreDevicePixelRatio=e.ignoreDevicePixelRatio||!1,this.devicePixelRatio=this.ignoreDevicePixelRatio?1:window.devicePixelRatio||1,this.halfPrecisionCovariancesOnGPU=e.halfPrecisionCovariancesOnGPU||!1,this.threeScene=e.threeScene,this.renderer=e.renderer,this.camera=e.camera,this.gpuAcceleratedSort=e.gpuAcceleratedSort||!1,(e.integerBasedSort===void 0||e.integerBasedSort===null)&&(e.integerBasedSort=!0),this.integerBasedSort=e.integerBasedSort,(e.sharedMemoryForWorkers===void 0||e.sharedMemoryForWorkers===null)&&(e.sharedMemoryForWorkers=!0),this.sharedMemoryForWorkers=e.sharedMemoryForWorkers,this.dynamicScene=!!e.dynamicScene,this.antialiased=e.antialiased||!1,this.kernel2DSize=e.kernel2DSize===void 0?.3:e.kernel2DSize,this.webXRMode=e.webXRMode||ys.None,this.webXRMode!==ys.None&&(this.gpuAcceleratedSort=!1),this.webXRActive=!1,this.webXRSessionInit=e.webXRSessionInit||{},this.renderMode=e.renderMode||iu.Always,this.sceneRevealMode=e.sceneRevealMode||Ko.Default,this.focalAdjustment=e.focalAdjustment||1,this.maxScreenSpaceSplatSize=e.maxScreenSpaceSplatSize||1024,this.logLevel=e.logLevel||to.None,this.sphericalHarmonicsDegree=e.sphericalHarmonicsDegree||0,this.enableOptionalEffects=e.enableOptionalEffects||!1,(e.enableSIMDInSort===void 0||e.enableSIMDInSort===null)&&(e.enableSIMDInSort=!0),this.enableSIMDInSort=e.enableSIMDInSort,(e.inMemoryCompressionLevel===void 0||e.inMemoryCompressionLevel===null)&&(e.inMemoryCompressionLevel=0),this.inMemoryCompressionLevel=e.inMemoryCompressionLevel,(e.optimizeSplatData===void 0||e.optimizeSplatData===null)&&(e.optimizeSplatData=!0),this.optimizeSplatData=e.optimizeSplatData,(e.freeIntermediateSplatData===void 0||e.freeIntermediateSplatData===null)&&(e.freeIntermediateSplatData=!1),this.freeIntermediateSplatData=e.freeIntermediateSplatData,od()){const n=tg();n.major<17&&(this.enableSIMDInSort=!1),n.major<16&&(this.sharedMemoryForWorkers=!1)}(e.splatRenderMode===void 0||e.splatRenderMode===null)&&(e.splatRenderMode=$i.ThreeD),this.splatRenderMode=e.splatRenderMode,this.sceneFadeInRateMultiplier=e.sceneFadeInRateMultiplier||1,this.splatSortDistanceMapPrecision=e.splatSortDistanceMapPrecision||pt.DefaultSplatSortDistanceMapPrecision;const t=this.integerBasedSort?20:24;this.splatSortDistanceMapPrecision=Ct(this.splatSortDistanceMapPrecision,10,t),this.onSplatMeshChangedCallback=null,this.createSplatMesh(),this.controls=null,this.perspectiveControls=null,this.orthographicControls=null,this.orthographicCamera=null,this.perspectiveCamera=null,this.showMeshCursor=!1,this.showControlPlane=!1,this.showInfo=!1,this.sceneHelper=null,this.sortWorker=null,this.sortRunning=!1,this.splatRenderCount=0,this.splatSortCount=0,this.lastSplatSortCount=0,this.sortWorkerIndexesToSort=null,this.sortWorkerSortedIndexes=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.preSortMessages=[],this.runAfterNextSort=[],this.selfDrivenModeRunning=!1,this.splatRenderReady=!1,this.raycaster=new dE,this.infoPanel=null,this.startInOrthographicMode=!1,this.currentFPS=0,this.lastSortTime=0,this.consecutiveRenderFrames=0,this.previousCameraTarget=new B,this.nextCameraTarget=new B,this.mousePosition=new ze,this.mouseDownPosition=new ze,this.mouseDownTime=null,this.resizeObserver=null,this.mouseMoveListener=null,this.mouseDownListener=null,this.mouseUpListener=null,this.keyDownListener=null,this.sortPromise=null,this.sortPromiseResolver=null,this.splatSceneDownloadPromises={},this.splatSceneDownloadAndBuildPromise=null,this.splatSceneRemovalPromise=null,this.loadingSpinner=new Ad(null,this.rootElement||document.body),this.loadingSpinner.hide(),this.loadingProgressBar=new oE(this.rootElement||document.body),this.loadingProgressBar.hide(),this.infoPanel=new aE(this.rootElement||document.body),this.infoPanel.hide(),this.usingExternalCamera=!!(this.dropInMode||this.camera),this.usingExternalRenderer=!!(this.dropInMode||this.renderer),this.initialized=!1,this.disposing=!1,this.disposed=!1,this.disposePromise=null,this.dropInMode||this.init()}createSplatMesh(){this.splatMesh=new $t(this.splatRenderMode,this.dynamicScene,this.enableOptionalEffects,this.halfPrecisionCovariancesOnGPU,this.devicePixelRatio,this.gpuAcceleratedSort,this.integerBasedSort,this.antialiased,this.maxScreenSpaceSplatSize,this.logLevel,this.sphericalHarmonicsDegree,this.sceneFadeInRateMultiplier,this.kernel2DSize),this.splatMesh.frustumCulled=!1,this.onSplatMeshChangedCallback&&this.onSplatMeshChangedCallback()}init(){this.initialized||(this.rootElement||(this.usingExternalRenderer?this.rootElement=this.renderer.domElement||document.body:(this.rootElement=document.createElement("div"),this.rootElement.style.width="100%",this.rootElement.style.height="100%",this.rootElement.style.position="absolute",document.body.appendChild(this.rootElement))),this.setupCamera(),this.setupRenderer(),this.setupWebXR(this.webXRSessionInit),this.setupControls(),this.setupEventHandlers(),this.threeScene=this.threeScene||new lv,this.sceneHelper=new Qo(this.threeScene),this.sceneHelper.setupMeshCursor(),this.sceneHelper.setupFocusMarker(),this.sceneHelper.setupControlPlane(),this.loadingProgressBar.setContainer(this.rootElement),this.loadingSpinner.setContainer(this.rootElement),this.infoPanel.setContainer(this.rootElement),this.initialized=!0)}setupCamera(){if(!this.usingExternalCamera){const e=new ze;this.getRenderDimensions(e),this.perspectiveCamera=new ei(BE,e.x/e.y,.1,1e3),this.orthographicCamera=new sd(e.x/-2,e.x/2,e.y/2,e.y/-2,.1,1e3),this.camera=this.startInOrthographicMode?this.orthographicCamera:this.perspectiveCamera,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt)}}setupRenderer(){if(!this.usingExternalRenderer){const e=new ze;this.getRenderDimensions(e),this.renderer=new sT({antialias:!1,precision:"highp"}),this.renderer.setPixelRatio(this.devicePixelRatio),this.renderer.autoClear=!0,this.renderer.setClearColor(new nt(0),0),this.renderer.setSize(e.x,e.y),this.resizeObserver=new ResizeObserver(()=>{this.getRenderDimensions(e),this.renderer.setSize(e.x,e.y),this.forceRenderNextFrame()}),this.resizeObserver.observe(this.rootElement),this.rootElement.appendChild(this.renderer.domElement)}}setupWebXR(e){this.webXRMode&&(this.webXRMode===ys.VR?this.rootElement.appendChild(mo.createButton(this.renderer,e)):this.webXRMode===ys.AR&&this.rootElement.appendChild(LE.createButton(this.renderer,e)),this.renderer.xr.addEventListener("sessionstart",t=>{this.webXRActive=!0}),this.renderer.xr.addEventListener("sessionend",t=>{this.webXRActive=!1}),this.renderer.xr.enabled=!0,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt))}setupControls(){if(this.useBuiltInControls&&this.webXRMode===ys.None){this.usingExternalCamera?this.camera.isOrthographicCamera?this.orthographicControls=new rl(this.camera,this.renderer.domElement):this.perspectiveControls=new rl(this.camera,this.renderer.domElement):(this.perspectiveControls=new rl(this.perspectiveCamera,this.renderer.domElement),this.orthographicControls=new rl(this.orthographicCamera,this.renderer.domElement));for(let e of[this.orthographicControls,this.perspectiveControls])e&&(e.listenToKeyEvents(window),e.rotateSpeed=.5,e.maxPolarAngle=Math.PI*.75,e.minPolarAngle=.1,e.enableDamping=!0,e.dampingFactor=.05,e.target.copy(this.initialCameraLookAt),e.update());this.controls=this.camera.isOrthographicCamera?this.orthographicControls:this.perspectiveControls,this.controls.update()}}setupEventHandlers(){this.useBuiltInControls&&this.webXRMode===ys.None&&(this.mouseMoveListener=this.onMouseMove.bind(this),this.renderer.domElement.addEventListener("pointermove",this.mouseMoveListener,!1),this.mouseDownListener=this.onMouseDown.bind(this),this.renderer.domElement.addEventListener("pointerdown",this.mouseDownListener,!1),this.mouseUpListener=this.onMouseUp.bind(this),this.renderer.domElement.addEventListener("pointerup",this.mouseUpListener,!1),this.keyDownListener=this.onKeyDown.bind(this),window.addEventListener("keydown",this.keyDownListener,!1))}removeEventHandlers(){this.useBuiltInControls&&(this.renderer.domElement.removeEventListener("pointermove",this.mouseMoveListener),this.mouseMoveListener=null,this.renderer.domElement.removeEventListener("pointerdown",this.mouseDownListener),this.mouseDownListener=null,this.renderer.domElement.removeEventListener("pointerup",this.mouseUpListener),this.mouseUpListener=null,window.removeEventListener("keydown",this.keyDownListener),this.keyDownListener=null)}setRenderMode(e){this.renderMode=e}setActiveSphericalHarmonicsDegrees(e){this.splatMesh.material.uniforms.sphericalHarmonicsDegree.value=e,this.splatMesh.material.uniformsNeedUpdate=!0}onSplatMeshChanged(e){this.onSplatMeshChangedCallback=e}onKeyDown=(function(){const e=new B,t=new qe,n=new qe;return function(s){switch(e.set(0,0,-1),e.transformDirection(this.camera.matrixWorld),t.makeRotationAxis(e,Math.PI/128),n.makeRotationAxis(e,-Math.PI/128),s.code){case"KeyG":this.focalAdjustment+=.02,this.forceRenderNextFrame();break;case"KeyF":this.focalAdjustment-=.02,this.forceRenderNextFrame();break;case"ArrowLeft":this.camera.up.transformDirection(t);break;case"ArrowRight":this.camera.up.transformDirection(n);break;case"KeyC":this.showMeshCursor=!this.showMeshCursor;break;case"KeyU":this.showControlPlane=!this.showControlPlane;break;case"KeyI":this.showInfo=!this.showInfo,this.showInfo?this.infoPanel.show():this.infoPanel.hide();break;case"KeyO":this.usingExternalCamera||this.setOrthographicMode(!this.camera.isOrthographicCamera);break;case"KeyP":this.usingExternalCamera||this.splatMesh.setPointCloudModeEnabled(!this.splatMesh.getPointCloudModeEnabled());break;case"Equal":this.usingExternalCamera||this.splatMesh.setSplatScale(this.splatMesh.getSplatScale()+.05);break;case"Minus":this.usingExternalCamera||this.splatMesh.setSplatScale(Math.max(this.splatMesh.getSplatScale()-.05,0));break}}})();onMouseMove(e){this.mousePosition.set(e.offsetX,e.offsetY)}onMouseDown(){this.mouseDownPosition.copy(this.mousePosition),this.mouseDownTime=Ur()}onMouseUp=(function(){const e=new ze;return function(t){e.copy(this.mousePosition).sub(this.mouseDownPosition),Ur()-this.mouseDownTime<.5&&e.length()<2&&this.onMouseClick(t)}})();onMouseClick(e){this.mousePosition.set(e.offsetX,e.offsetY),this.checkForFocalPointChange()}checkForFocalPointChange=(function(){const e=new ze,t=new B,n=[];return function(){if(!this.transitioningCameraTarget&&(this.getRenderDimensions(e),n.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,e),this.raycaster.intersectSplatMesh(this.splatMesh,n),n.length>0)){const r=n[0].origin;t.copy(r).sub(this.camera.position),t.length()>UE&&(this.previousCameraTarget.copy(this.controls.target),this.nextCameraTarget.copy(r),this.transitioningCameraTarget=!0,this.transitioningCameraTargetStartTime=Ur())}}})();getRenderDimensions(e){this.rootElement?(e.x=this.rootElement.offsetWidth,e.y=this.rootElement.offsetHeight):this.renderer.getSize(e)}setOrthographicMode(e){if(e===this.camera.isOrthographicCamera)return;const t=this.camera,n=e?this.orthographicCamera:this.perspectiveCamera;if(n.position.copy(t.position),n.up.copy(t.up),n.rotation.copy(t.rotation),n.quaternion.copy(t.quaternion),n.matrix.copy(t.matrix),this.camera=n,this.controls){const s=a=>{a.saveState(),a.reset()},r=this.controls,o=e?this.orthographicControls:this.perspectiveControls;s(o),s(r),o.target.copy(r.target),e?Gr.setCameraZoomFromPosition(n,t,r):Gr.setCameraPositionFromZoom(n,t,o),this.controls=o,this.camera.lookAt(this.controls.target)}}static setCameraPositionFromZoom=(function(){const e=new B;return function(t,n,s){const r=1/(n.zoom*.001);e.copy(s.target).sub(t.position).normalize().multiplyScalar(r).negate(),t.position.copy(s.target).add(e)}})();static setCameraZoomFromPosition=(function(){const e=new B;return function(t,n,s){const r=e.copy(s.target).sub(n.position).length();t.zoom=1/(r*.001)}})();updateSplatMesh=(function(){const e=new ze;return function(){if(!this.splatMesh)return;if(this.splatMesh.getSplatCount()>0){this.splatMesh.updateVisibleRegionFadeDistance(this.sceneRevealMode),this.splatMesh.updateTransforms(),this.getRenderDimensions(e);const n=this.camera.projectionMatrix.elements[0]*.5*this.devicePixelRatio*e.x,s=this.camera.projectionMatrix.elements[5]*.5*this.devicePixelRatio*e.y,r=this.camera.isOrthographicCamera?1/this.devicePixelRatio:1,o=this.focalAdjustment*r,a=1/o;this.adjustForWebXRStereo(e),this.splatMesh.updateUniforms(e,n*o,s*o,this.camera.isOrthographicCamera,this.camera.zoom||1,a)}}})();adjustForWebXRStereo(e){if(this.camera&&this.webXRActive){const n=this.renderer.xr.getCamera().projectionMatrix.elements[0],s=this.camera.projectionMatrix.elements[0];e.x*=s/n}}isLoadingOrUnloading(){return Object.keys(this.splatSceneDownloadPromises).length>0||this.splatSceneDownloadAndBuildPromise!==null||this.splatSceneRemovalPromise!==null}isDisposingOrDisposed(){return this.disposing||this.disposed}addSplatSceneDownloadPromise(e){this.splatSceneDownloadPromises[e.id]=e}removeSplatSceneDownloadPromise(e){delete this.splatSceneDownloadPromises[e.id]}setSplatSceneDownloadAndBuildPromise(e){this.splatSceneDownloadAndBuildPromise=e}clearSplatSceneDownloadAndBuildPromise(){this.splatSceneDownloadAndBuildPromise=null}addSplatScene(e,t={}){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");t.progressiveLoad&&this.splatMesh.scenes&&this.splatMesh.scenes.length>0&&(console.log('addSplatScene(): "progressiveLoad" option ignore because there are multiple splat scenes'),t.progressiveLoad=!1);const n=t.format!==void 0&&t.format!==null?t.format:Vp(e),s=Gr.isProgressivelyLoadable(n)&&t.progressiveLoad,r=t.showLoadingUI!==void 0&&t.showLoadingUI!==null?t.showLoadingUI:!0;let o=null;r&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=()=>{this.loadingProgressBar.hide(),this.loadingSpinner.removeAllTasks()},l=(p,g,m)=>{if(r)if(m===Nt.Downloading)if(p==100)this.loadingSpinner.setMessageForTask(o,"Download complete!");else if(s)this.loadingSpinner.setMessageForTask(o,"Downloading splats...");else{const _=g?`: ${g}`:"...";this.loadingSpinner.setMessageForTask(o,`Downloading${_}`)}else m===Nt.Processing&&this.loadingSpinner.setMessageForTask(o,"Processing splats...")};let c=!1,u=0;const f=(p,g)=>{r&&((p&&s||g&&!s)&&(this.loadingSpinner.removeTask(o),!g&&!c&&this.loadingProgressBar.show()),s&&(g?(c=!0,this.loadingProgressBar.hide()):this.loadingProgressBar.setProgress(u)))},d=(p,g,m)=>{u=p,l(p,g,m),t.onProgress&&t.onProgress(p,g,m)},h=(p,g,m)=>{!s&&t.onProgress&&t.onProgress(0,"0%",Nt.Processing);const _={rotation:t.rotation||t.orientation,position:t.position,scale:t.scale,splatAlphaRemovalThreshold:t.splatAlphaRemovalThreshold};return this.addSplatBuffers([p],[_],m,g&&r,r,s,s).then(()=>{!s&&t.onProgress&&t.onProgress(100,"100%",Nt.Processing),f(g,m)})};return(s?this.downloadAndBuildSingleSplatSceneProgressiveLoad.bind(this):this.downloadAndBuildSingleSplatSceneStandardLoad.bind(this))(e,n,t.splatAlphaRemovalThreshold,h.bind(this),d,a.bind(this),t.headers)}downloadAndBuildSingleSplatSceneStandardLoad(e,t,n,s,r,o,a){const l=this.downloadSplatSceneToSplatBuffer(e,n,r,!1,void 0,t,a),c=Kc(l.abortHandler);return l.then(u=>(this.removeSplatSceneDownloadPromise(l),s(u,!0,!0).then(()=>{c.resolve(),this.clearSplatSceneDownloadAndBuildPromise()}))).catch(u=>{o&&o(),this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(l),c.reject(this.updateError(u,`Viewer::addSplatScene -> Could not load file ${e}`))}),this.addSplatSceneDownloadPromise(l),this.setSplatSceneDownloadAndBuildPromise(c.promise),c.promise}downloadAndBuildSingleSplatSceneProgressiveLoad(e,t,n,s,r,o,a){let l=0,c=!1;const u=[],f=()=>{if(u.length>0&&!c&&!this.isDisposingOrDisposed()){c=!0;const g=u.shift();s(g.splatBuffer,g.firstBuild,g.finalBuild).then(()=>{c=!1,g.firstBuild?x.resolve():g.finalBuild&&(p.resolve(),this.clearSplatSceneDownloadAndBuildPromise()),u.length>0&&Gn(()=>f())})}},d=(g,m)=>{this.isDisposingOrDisposed()||(m||u.length===0||g.getSplatCount()>u[0].splatBuffer.getSplatCount())&&(u.push({splatBuffer:g,firstBuild:l===0,finalBuild:m}),l++,f())},h=this.downloadSplatSceneToSplatBuffer(e,n,r,!0,d,t,a),x=Kc(h.abortHandler),p=Kc();return this.addSplatSceneDownloadPromise(h),this.setSplatSceneDownloadAndBuildPromise(p.promise),h.then(()=>{this.removeSplatSceneDownloadPromise(h)}).catch(g=>{this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(h);const m=this.updateError(g,"Viewer::addSplatScene -> Could not load one or more scenes");x.reject(m),o&&o(m)}),x.promise}addSplatScenes(e,t=!0,n=void 0){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");const s=e.length,r=[];let o;t&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=(f,d,h,x)=>{r[f]=d;let p=0;for(let g=0;g<s;g++)p+=r[g]||0;p=p/s,h=`${p.toFixed(2)}%`,t&&x===Nt.Downloading&&this.loadingSpinner.setMessageForTask(o,p==100?"Download complete!":`Downloading: ${h}`),n&&n(p,h,x)},l=[],c=[];for(let f=0;f<e.length;f++){const d=e[f],h=d.format!==void 0&&d.format!==null?d.format:Vp(d.path),x=this.downloadSplatSceneToSplatBuffer(d.path,d.splatAlphaRemovalThreshold,a.bind(this,f),!1,void 0,h,d.headers);l.push(x),c.push(x.promise)}const u=new Cs((f,d)=>{Promise.all(c).then(h=>{t&&this.loadingSpinner.removeTask(o),n&&n(0,"0%",Nt.Processing),this.addSplatBuffers(h,e,!0,t,t,!1,!1).then(()=>{n&&n(100,"100%",Nt.Processing),this.clearSplatSceneDownloadAndBuildPromise(),f()})}).catch(h=>{t&&this.loadingSpinner.removeTask(o),this.clearSplatSceneDownloadAndBuildPromise(),d(this.updateError(h,"Viewer::addSplatScenes -> Could not load one or more splat scenes."))}).finally(()=>{this.removeSplatSceneDownloadPromise(u)})},f=>{for(let d of l)d.abort(f)});return this.addSplatSceneDownloadPromise(u),this.setSplatSceneDownloadAndBuildPromise(u),u}downloadSplatSceneToSplatBuffer(e,t=1,n=void 0,s=!1,r=void 0,o,a){try{if(o===En.Splat||o===En.KSplat||o===En.Ply){const l=s?!1:this.optimizeSplatData;if(o===En.Splat)return _d.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,a);if(o===En.KSplat)return qo.loadFromURL(e,n,s,r,a);if(o===En.Ply)return gd.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,this.sphericalHarmonicsDegree,a)}else if(o===En.Spz)return xd.loadFromURL(e,n,t,this.inMemoryCompressionLevel,this.optimizeSplatData,this.sphericalHarmonicsDegree,a)}catch(l){throw this.updateError(l,null)}throw new Error(`Viewer::downloadSplatSceneToSplatBuffer -> File format not supported: ${e}`)}static isProgressivelyLoadable(e){return e===En.Splat||e===En.KSplat||e===En.Ply}addSplatBuffers=(function(){return function(e,t=[],n=!0,s=!0,r=!0,o=!1,a=!1,l=!0){if(this.isDisposingOrDisposed())return Promise.resolve();let c=null;const u=()=>{c!==null&&(this.loadingSpinner.removeTask(c),c=null)};return this.splatRenderReady=!1,new Promise(f=>{s&&(c=this.loadingSpinner.addTask("Processing splats...")),Gn(()=>{if(this.isDisposingOrDisposed())f();else{const d=this.addSplatBuffersToMesh(e,t,n,r,o,l),h=this.splatMesh.getMaxSplatCount();this.sortWorker&&this.sortWorker.maxSplatCount!==h&&this.disposeSortWorker(),this.gpuAcceleratedSort||this.preSortMessages.push({centers:d.centers.buffer,sceneIndexes:d.sceneIndexes.buffer,range:{from:d.from,to:d.to,count:d.count}}),(!this.sortWorker&&h>0?this.setupSortWorker(this.splatMesh):Promise.resolve()).then(()=>{this.isDisposingOrDisposed()||this.runSplatSort(!0,!0).then(p=>{!this.sortWorker||!p?(this.splatRenderReady=!0,u(),f()):(a?this.splatRenderReady=!0:this.runAfterNextSort.push(()=>{this.splatRenderReady=!0}),this.runAfterNextSort.push(()=>{u(),f()}))})})}},!0)})}})();addSplatBuffersToMesh=(function(){let e;return function(t,n,s=!0,r=!1,o=!1,a=!0){if(this.isDisposingOrDisposed())return;let l=[],c=[];o||(l=this.splatMesh.scenes.map(h=>h.splatBuffer)||[],c=this.splatMesh.sceneOptions?this.splatMesh.sceneOptions.map(h=>h):[]),l.push(...t),c.push(...n),this.renderer&&this.splatMesh.setRenderer(this.renderer);const u=h=>{if(this.isDisposingOrDisposed())return;const x=this.splatMesh.getSplatCount();r&&x>=OE&&!h&&!e&&(this.loadingSpinner.setMinimized(!0,!0),e=this.loadingSpinner.addTask("Optimizing data structures..."))},f=h=>{this.isDisposingOrDisposed()||h&&e&&(this.loadingSpinner.removeTask(e),e=null)},d=this.splatMesh.build(l,c,!0,s,u,f,a);return s&&this.freeIntermediateSplatData&&this.splatMesh.freeIntermediateSplatData(),d}})();setupSortWorker(e){if(!this.isDisposingOrDisposed())return new Promise(t=>{const n=this.integerBasedSort?Int32Array:Float32Array,s=e.getSplatCount(),r=e.getMaxSplatCount();this.sortWorker=FE(r,this.sharedMemoryForWorkers,this.enableSIMDInSort,this.integerBasedSort,this.splatMesh.dynamicMode,this.splatSortDistanceMapPrecision),this.sortWorker.onmessage=o=>{if(o.data.sortDone){if(this.sortRunning=!1,this.sharedMemoryForWorkers)this.splatMesh.updateRenderIndexes(this.sortWorkerSortedIndexes,o.data.splatRenderCount);else{const a=new Uint32Array(o.data.sortedIndexes.buffer,0,o.data.splatRenderCount);this.splatMesh.updateRenderIndexes(a,o.data.splatRenderCount)}this.lastSplatSortCount=this.splatSortCount,this.lastSortTime=o.data.sortTime,this.sortPromiseResolver(),this.sortPromiseResolver=null,this.forceRenderNextFrame(),this.runAfterNextSort.length>0&&(this.runAfterNextSort.forEach(a=>{a()}),this.runAfterNextSort.length=0)}else if(o.data.sortCanceled)this.sortRunning=!1;else if(o.data.sortSetupPhase1Complete){this.logLevel>=to.Info&&console.log("Sorting web worker WASM setup complete."),this.sharedMemoryForWorkers?(this.sortWorkerSortedIndexes=new Uint32Array(o.data.sortedIndexesBuffer,o.data.sortedIndexesOffset,r),this.sortWorkerIndexesToSort=new Uint32Array(o.data.indexesToSortBuffer,o.data.indexesToSortOffset,r),this.sortWorkerPrecomputedDistances=new n(o.data.precomputedDistancesBuffer,o.data.precomputedDistancesOffset,r),this.sortWorkerTransforms=new Float32Array(o.data.transformsBuffer,o.data.transformsOffset,pt.MaxScenes*16)):(this.sortWorkerIndexesToSort=new Uint32Array(r),this.sortWorkerPrecomputedDistances=new n(r),this.sortWorkerTransforms=new Float32Array(pt.MaxScenes*16));for(let a=0;a<s;a++)this.sortWorkerIndexesToSort[a]=a;if(this.sortWorker.maxSplatCount=r,this.logLevel>=to.Info){console.log("Sorting web worker ready.");const a=this.splatMesh.getSplatDataTextures(),l=a.covariances.size,c=a.centerColors.size;console.log("Covariances texture size: "+l.x+" x "+l.y),console.log("Centers/colors texture size: "+c.x+" x "+c.y)}t()}}})}updateError(e,t){return e instanceof eg?e:e instanceof Dl?new Error("File type or server does not support progressive loading."):t?new Error(t):e}disposeSortWorker(){this.sortWorker&&this.sortWorker.terminate(),this.sortWorker=null,this.sortPromise=null,this.sortPromiseResolver&&(this.sortPromiseResolver(),this.sortPromiseResolver=null),this.preSortMessages=[],this.sortRunning=!1}removeSplatScene(e,t=!0){return this.removeSplatScenes([e],t)}removeSplatScenes(e,t=!0){if(this.isLoadingOrUnloading())throw new Error("Cannot remove splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot remove splat scene after dispose() is called.");let n;return this.splatSceneRemovalPromise=new Promise((s,r)=>{let o;t&&(this.loadingSpinner.removeAllTasks(),this.loadingSpinner.show(),o=this.loadingSpinner.addTask("Removing splat scene..."));const a=()=>{t&&(this.loadingSpinner.hide(),this.loadingSpinner.removeTask(o))},l=u=>{a(),this.splatSceneRemovalPromise=null,u?r(u):s()},c=()=>this.isDisposingOrDisposed()?(l(),!0):!1;n=this.sortPromise||Promise.resolve(),n.then(()=>{if(c())return;const u=[],f=[],d=[];for(let h=0;h<this.splatMesh.scenes.length;h++){let x=!1;for(let p of e)if(p===h){x=!0;break}if(!x){const p=this.splatMesh.scenes[h];u.push(p.splatBuffer),f.push(this.splatMesh.sceneOptions[h]),d.push({position:p.position.clone(),quaternion:p.quaternion.clone(),scale:p.scale.clone()})}}this.disposeSortWorker(),this.splatMesh.dispose(),this.sceneRevealMode=Ko.Instant,this.createSplatMesh(),this.addSplatBuffers(u,f,!0,!1,!0).then(()=>{c()||(a(),this.splatMesh.scenes.forEach((h,x)=>{h.position.copy(d[x].position),h.quaternion.copy(d[x].quaternion),h.scale.copy(d[x].scale)}),this.splatMesh.updateTransforms(),this.splatRenderReady=!1,this.runSplatSort(!0).then(()=>{if(c()){this.splatRenderReady=!0;return}n=this.sortPromise||Promise.resolve(),n.then(()=>{this.splatRenderReady=!0,l()})}))}).catch(h=>{l(h)})})}),this.splatSceneRemovalPromise}start(){if(this.selfDrivenMode)this.webXRMode?this.renderer.setAnimationLoop(this.selfDrivenUpdateFunc):this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc),this.selfDrivenModeRunning=!0;else throw new Error("Cannot start viewer unless it is in self driven mode.")}stop(){this.selfDrivenMode&&this.selfDrivenModeRunning&&(this.webXRMode?this.renderer.setAnimationLoop(null):cancelAnimationFrame(this.requestFrameId),this.selfDrivenModeRunning=!1)}async dispose(){if(this.isDisposingOrDisposed())return this.disposePromise;let e=[],t=[];for(let n in this.splatSceneDownloadPromises)if(this.splatSceneDownloadPromises.hasOwnProperty(n)){const s=this.splatSceneDownloadPromises[n];t.push(s),e.push(s.promise)}return this.sortPromise&&e.push(this.sortPromise),this.disposing=!0,this.disposePromise=Promise.all(e).finally(()=>{this.stop(),this.orthographicControls&&(this.orthographicControls.dispose(),this.orthographicControls=null),this.perspectiveControls&&(this.perspectiveControls.dispose(),this.perspectiveControls=null),this.controls=null,this.splatMesh&&(this.splatMesh.dispose(),this.splatMesh=null),this.sceneHelper&&(this.sceneHelper.dispose(),this.sceneHelper=null),this.resizeObserver&&(this.resizeObserver.unobserve(this.rootElement),this.resizeObserver=null),this.disposeSortWorker(),this.removeEventHandlers(),this.loadingSpinner.removeAllTasks(),this.loadingSpinner.setContainer(null),this.loadingProgressBar.hide(),this.loadingProgressBar.setContainer(null),this.infoPanel.setContainer(null),this.camera=null,this.threeScene=null,this.splatRenderReady=!1,this.initialized=!1,this.renderer&&(this.usingExternalRenderer||(this.rootElement.removeChild(this.renderer.domElement),this.renderer.dispose()),this.renderer=null),this.usingExternalRenderer||document.body.removeChild(this.rootElement),this.sortWorkerSortedIndexes=null,this.sortWorkerIndexesToSort=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.disposed=!0,this.disposing=!1,this.disposePromise=null}),t.forEach(n=>{n.abort("Scene disposed")}),this.disposePromise}selfDrivenUpdate(){this.selfDrivenMode&&!this.webXRMode&&(this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc)),this.update(),this.shouldRender()?(this.render(),this.consecutiveRenderFrames++):this.consecutiveRenderFrames=0,this.renderNextFrame=!1}forceRenderNextFrame(){this.renderNextFrame=!0}shouldRender=(function(){let e=0;const t=new B,n=new bt,s=1e-4;return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return!1;let r=!1,o=!1;if(this.camera){const a=this.camera.position,l=this.camera.quaternion;o=Math.abs(a.x-t.x)>s||Math.abs(a.y-t.y)>s||Math.abs(a.z-t.z)>s||Math.abs(l.x-n.x)>s||Math.abs(l.y-n.y)>s||Math.abs(l.z-n.z)>s||Math.abs(l.w-n.w)>s}return r=this.renderMode!==iu.Never&&(e===0||this.splatMesh.visibleRegionChanging||o||this.renderMode===iu.Always||this.dynamicMode===!0||this.renderNextFrame),this.camera&&(t.copy(this.camera.position),n.copy(this.camera.quaternion)),e++,r}})();render=(function(){return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return;const e=n=>{for(let s of n.children)if(s.visible)return!0;return!1},t=this.renderer.autoClear;e(this.threeScene)&&(this.renderer.render(this.threeScene,this.camera),this.renderer.autoClear=!1),this.renderer.render(this.splatMesh,this.camera),this.renderer.autoClear=!1,this.sceneHelper.getFocusMarkerOpacity()>0&&this.renderer.render(this.sceneHelper.focusMarker,this.camera),this.showControlPlane&&this.renderer.render(this.sceneHelper.controlPlane,this.camera),this.renderer.autoClear=t}})();update(e,t){this.dropInMode&&this.updateForDropInMode(e,t),!(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())&&(this.controls&&(this.controls.update(),this.camera.isOrthographicCamera&&!this.usingExternalCamera&&Gr.setCameraPositionFromZoom(this.camera,this.camera,this.controls)),this.runSplatSort(),this.updateForRendererSizeChanges(),this.updateSplatMesh(),this.updateMeshCursor(),this.updateFPS(),this.timingSensitiveUpdates(),this.updateInfoPanel(),this.updateControlPlane())}updateForDropInMode(e,t){this.renderer=e,this.splatMesh&&this.splatMesh.setRenderer(this.renderer),this.camera=t,this.controls&&(this.controls.object=t),this.init()}updateFPS=(function(){let e=Ur(),t=0;return function(){if(this.consecutiveRenderFrames>kE){const n=Ur();n-e>=1?(this.currentFPS=t,t=0,e=n):t++}else this.currentFPS=null}})();updateForRendererSizeChanges=(function(){const e=new ze,t=new ze;let n;return function(){this.usingExternalCamera||(this.renderer.getSize(t),(n===void 0||n!==this.camera.isOrthographicCamera||t.x!==e.x||t.y!==e.y)&&(this.camera.isOrthographicCamera?(this.camera.left=-t.x/2,this.camera.right=t.x/2,this.camera.top=t.y/2,this.camera.bottom=-t.y/2):this.camera.aspect=t.x/t.y,this.camera.updateProjectionMatrix(),e.copy(t),n=this.camera.isOrthographicCamera))}})();timingSensitiveUpdates=(function(){let e;return function(){const t=Ur();e||(e=t);const n=t-e;this.updateCameraTransition(t),this.updateFocusMarker(n),e=t}})();updateCameraTransition=(function(){let e=new B,t=new B,n=new B;return function(s){if(this.transitioningCameraTarget){t.copy(this.previousCameraTarget).sub(this.camera.position).normalize(),n.copy(this.nextCameraTarget).sub(this.camera.position).normalize();const r=Math.acos(t.dot(n)),a=(r/(Math.PI/3)*.65+.3)/r*(s-this.transitioningCameraTargetStartTime);e.copy(this.previousCameraTarget).lerp(this.nextCameraTarget,a),this.camera.lookAt(e),this.controls.target.copy(e),a>=1&&(this.transitioningCameraTarget=!1)}}})();updateFocusMarker=(function(){const e=new ze;let t=!1;return function(n){if(this.getRenderDimensions(e),this.transitioningCameraTarget){this.sceneHelper.setFocusMarkerVisibility(!0);const s=Math.max(this.sceneHelper.getFocusMarkerOpacity(),0);let r=Math.min(s+NE*n,1);this.sceneHelper.setFocusMarkerOpacity(r),this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e),t=!0,this.forceRenderNextFrame()}else{let s;if(t?s=1:s=Math.min(this.sceneHelper.getFocusMarkerOpacity(),1),s>0){this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e);let r=Math.max(s-zE*n,0);this.sceneHelper.setFocusMarkerOpacity(r),r===0&&this.sceneHelper.setFocusMarkerVisibility(!1)}s>0&&this.forceRenderNextFrame(),t=!1}}})();updateMeshCursor=(function(){const e=[],t=new ze;return function(){this.showMeshCursor?(this.forceRenderNextFrame(),this.getRenderDimensions(t),e.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,t),this.raycaster.intersectSplatMesh(this.splatMesh,e),e.length>0?(this.sceneHelper.setMeshCursorVisibility(!0),this.sceneHelper.positionAndOrientMeshCursor(e[0].origin,this.camera)):this.sceneHelper.setMeshCursorVisibility(!1)):(this.sceneHelper.getMeschCursorVisibility()&&this.forceRenderNextFrame(),this.sceneHelper.setMeshCursorVisibility(!1))}})();updateInfoPanel=(function(){const e=new ze;return function(){if(!this.showInfo)return;const t=this.splatMesh.getSplatCount();this.getRenderDimensions(e);const n=this.controls?this.controls.target:null,s=this.showMeshCursor?this.sceneHelper.meshCursor.position:null,r=t>0?this.splatRenderCount/t*100:0;this.infoPanel.update(e,this.camera.position,n,this.camera.up,this.camera.isOrthographicCamera,s,this.currentFPS||"N/A",t,this.splatRenderCount,r,this.lastSortTime,this.focalAdjustment,this.splatMesh.getSplatScale(),this.splatMesh.getPointCloudModeEnabled())}})();updateControlPlane(){this.showControlPlane?(this.sceneHelper.setControlPlaneVisibility(!0),this.sceneHelper.positionAndOrientControlPlane(this.controls.target,this.camera.up)):this.sceneHelper.setControlPlaneVisibility(!1)}runSplatSort=(function(){const e=new qe,t=[],n=new B(0,0,-1),s=new B(0,0,-1),r=new B,o=new B,a=[],l=[{angleThreshold:.55,sortFractions:[.125,.33333,.75]},{angleThreshold:.65,sortFractions:[.33333,.66667]},{angleThreshold:.8,sortFractions:[.5]}];return function(c=!1,u=!1){if(!this.initialized)return Promise.resolve(!1);if(this.sortRunning)return Promise.resolve(!0);if(this.splatMesh.getSplatCount()<=0)return this.splatRenderCount=0,Promise.resolve(!1);let f=0,d=0,h=!1,x=!1;if(s.set(0,0,-1).applyQuaternion(this.camera.quaternion),f=s.dot(n),d=o.copy(this.camera.position).sub(r).length(),!c&&!this.splatMesh.dynamicMode&&a.length===0&&(f<=.99&&(h=!0),d>=1&&(x=!0),!h&&!x))return Promise.resolve(!1);this.sortRunning=!0;let{splatRenderCount:p,shouldSortAll:g}=this.gatherSceneNodesForSort();g=g||u,this.splatRenderCount=p,e.copy(this.camera.matrixWorld).invert();const m=this.perspectiveCamera||this.camera;e.premultiply(m.projectionMatrix),this.splatMesh.dynamicMode||e.multiply(this.splatMesh.matrixWorld);let _=Promise.resolve(!0);return this.gpuAcceleratedSort&&(a.length<=1||a.length%2===0)&&(_=this.splatMesh.computeDistancesOnGPU(e,this.sortWorkerPrecomputedDistances)),_.then(()=>{if(a.length===0)if(this.splatMesh.dynamicMode||g)a.push(this.splatRenderCount);else{for(let v of l)if(f<v.angleThreshold){for(let y of v.sortFractions)a.push(Math.floor(this.splatRenderCount*y));break}a.push(this.splatRenderCount)}let A=Math.min(a.shift(),this.splatRenderCount);this.splatSortCount=A,t[0]=this.camera.position.x,t[1]=this.camera.position.y,t[2]=this.camera.position.z;const S={modelViewProj:e.elements,cameraPosition:t,splatRenderCount:this.splatRenderCount,splatSortCount:A,usePrecomputedDistances:this.gpuAcceleratedSort};return this.splatMesh.dynamicMode&&this.splatMesh.fillTransformsArray(this.sortWorkerTransforms),this.sharedMemoryForWorkers||(S.indexesToSort=this.sortWorkerIndexesToSort,S.transforms=this.sortWorkerTransforms,this.gpuAcceleratedSort&&(S.precomputedDistances=this.sortWorkerPrecomputedDistances)),this.sortPromise=new Promise(v=>{this.sortPromiseResolver=v}),this.preSortMessages.length>0&&(this.preSortMessages.forEach(v=>{this.sortWorker.postMessage(v)}),this.preSortMessages=[]),this.sortWorker.postMessage({sort:S}),a.length===0&&(r.copy(this.camera.position),n.copy(s)),!0}),_}})();gatherSceneNodesForSort=(function(){const e=[];let t=null;const n=new B,s=new B,r=new B,o=new qe,a=new qe,l=new qe,c=new B,u=new B(0,0,-1),f=new B,d=h=>f.copy(h.max).sub(h.min).length();return function(h=!1){this.getRenderDimensions(c);const x=c.y/2/Math.tan(this.camera.fov/2*O0.DEG2RAD),p=Math.atan(c.x/2/x),g=Math.atan(c.y/2/x),m=Math.cos(p),_=Math.cos(g),A=this.splatMesh.getSplatTree();if(A){a.copy(this.camera.matrixWorld).invert(),this.splatMesh.dynamicMode||a.multiply(this.splatMesh.matrixWorld);let S=0,v=0;for(let M=0;M<A.subTrees.length;M++){const E=A.subTrees[M];o.copy(a),this.splatMesh.dynamicMode&&(this.splatMesh.getSceneTransform(M,l),o.multiply(l));const b=E.nodesWithIndexes.length;for(let C=0;C<b;C++){const I=E.nodesWithIndexes[C];if(!I.data||!I.data.indexes||I.data.indexes.length===0)continue;r.copy(I.center).applyMatrix4(o);const F=r.length();r.normalize(),n.copy(r).setX(0).normalize(),s.copy(r).setY(0).normalize();const U=u.dot(s),O=u.dot(n),k=d(I),z=O<_-.6,V=U<m-.6;!h&&(V||z)&&F>k||(v+=I.data.indexes.length,e[S]=I,I.data.distanceToNode=F,S++)}}e.length=S,e.sort((M,E)=>M.data.distanceToNode<E.data.distanceToNode?-1:1);let y=v*pt.BytesPerInt;for(let M=0;M<S;M++){const E=e[M],b=E.data.indexes.length,C=b*pt.BytesPerInt;new Uint32Array(this.sortWorkerIndexesToSort.buffer,y-C,b).set(E.data.indexes),y-=C}return{splatRenderCount:v,shouldSortAll:!1}}else{const S=this.splatMesh.getSplatCount();if(!t||t.length!==S){t=new Uint32Array(S);for(let v=0;v<S;v++)t[v]=v}return this.sortWorkerIndexesToSort.set(t),{splatRenderCount:S,shouldSortAll:!0}}}})();getSplatMesh(){return this.splatMesh}getSplatScene(e){return this.splatMesh.getScene(e)}getSceneCount(){return this.splatMesh.getSceneCount()}isMobile(){return navigator.userAgent.includes("Mobi")}}function Ki(i){if(i===void 0)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return i}function cg(i,e){i.prototype=Object.create(e.prototype),i.prototype.constructor=i,i.__proto__=e}var Qn={autoSleep:120,force3D:"auto",nullTargetWarn:1,units:{lineHeight:""}},go={duration:.5,overwrite:!1,delay:0},yd,Jt,Tt,ri=1e8,_t=1/ri,pf=Math.PI*2,HE=pf/4,VE=0,ug=Math.sqrt,GE=Math.cos,WE=Math.sin,jt=function(e){return typeof e=="string"},Ut=function(e){return typeof e=="function"},ls=function(e){return typeof e=="number"},bd=function(e){return typeof e>"u"},Oi=function(e){return typeof e=="object"},Rn=function(e){return e!==!1},Md=function(){return typeof window<"u"},al=function(e){return Ut(e)||jt(e)},fg=typeof ArrayBuffer=="function"&&ArrayBuffer.isView||function(){},an=Array.isArray,XE=/random\([^)]+\)/g,qE=/,\s*/g,Zp=/(?:-?\.?\d|\.)+/gi,dg=/[-+=.]*\d+[.e\-+]*\d*[e\-+]*\d*/g,Wr=/[-+=.]*\d+[.e-]*\d*[a-z%]*/g,su=/[-+=.]*\d+\.?\d*(?:e-|e\+)?\d*/gi,hg=/[+-]=-?[.\d]+/,QE=/[^,'"\[\]\s]+/gi,YE=/^[+\-=e\s\d]*\d+[.\d]*([a-z]*|%)\s*$/i,Pt,yi,mf,Cd,Yn={},Ll={},pg,mg=function(e){return(Ll=xo(e,Yn))&&Fn},Td=function(e,t){return console.warn("Invalid property",e,"set to",t,"Missing plugin? gsap.registerPlugin()")},da=function(e,t){return!t&&console.warn(e)},gg=function(e,t){return e&&(Yn[e]=t)&&Ll&&(Ll[e]=t)||Yn},ha=function(){return 0},KE={suppressEvents:!0,isStart:!0,kill:!1},_l={suppressEvents:!0,kill:!1},jE={suppressEvents:!0},Ed={},Ps=[],gf={},xg,kn={},ru={},Jp=30,Al=[],wd="",Rd=function(e){var t=e[0],n,s;if(Oi(t)||Ut(t)||(e=[e]),!(n=(t._gsap||{}).harness)){for(s=Al.length;s--&&!Al[s].targetTest(t););n=Al[s]}for(s=e.length;s--;)e[s]&&(e[s]._gsap||(e[s]._gsap=new Hg(e[s],n)))||e.splice(s,1);return e},cr=function(e){return e._gsap||Rd(oi(e))[0]._gsap},_g=function(e,t,n){return(n=e[t])&&Ut(n)?e[t]():bd(n)&&e.getAttribute&&e.getAttribute(t)||n},In=function(e,t){return(e=e.split(",")).forEach(t)||e},Ot=function(e){return Math.round(e*1e5)/1e5||0},Dt=function(e){return Math.round(e*1e7)/1e7||0},no=function(e,t){var n=t.charAt(0),s=parseFloat(t.substr(2));return e=parseFloat(e),n==="+"?e+s:n==="-"?e-s:n==="*"?e*s:e/s},$E=function(e,t){for(var n=t.length,s=0;e.indexOf(t[s])<0&&++s<n;);return s<n},Bl=function(){var e=Ps.length,t=Ps.slice(0),n,s;for(gf={},Ps.length=0,n=0;n<e;n++)s=t[n],s&&s._lazy&&(s.render(s._lazy[0],s._lazy[1],!0)._lazy=0)},Id=function(e){return!!(e._initted||e._startAt||e.add)},Ag=function(e,t,n,s){Ps.length&&!Jt&&Bl(),e.render(t,n,!!(Jt&&t<0&&Id(e))),Ps.length&&!Jt&&Bl()},Sg=function(e){var t=parseFloat(e);return(t||t===0)&&(e+"").match(QE).length<2?t:jt(e)?e.trim():e},vg=function(e){return e},Kn=function(e,t){for(var n in t)n in e||(e[n]=t[n]);return e},ZE=function(e){return function(t,n){for(var s in n)s in t||s==="duration"&&e||s==="ease"||(t[s]=n[s])}},xo=function(e,t){for(var n in t)e[n]=t[n];return e},em=function i(e,t){for(var n in t)n!=="__proto__"&&n!=="constructor"&&n!=="prototype"&&(e[n]=Oi(t[n])?i(e[n]||(e[n]={}),t[n]):t[n]);return e},Ul=function(e,t){var n={},s;for(s in e)s in t||(n[s]=e[s]);return n},jo=function(e){var t=e.parent||Pt,n=e.keyframes?ZE(an(e.keyframes)):Kn;if(Rn(e.inherit))for(;t;)n(e,t.vars.defaults),t=t.parent||t._dp;return e},JE=function(e,t){for(var n=e.length,s=n===t.length;s&&n--&&e[n]===t[n];);return n<0},yg=function(e,t,n,s,r){var o=e[s],a;if(r)for(a=t[r];o&&o[r]>a;)o=o._prev;return o?(t._next=o._next,o._next=t):(t._next=e[n],e[n]=t),t._next?t._next._prev=t:e[s]=t,t._prev=o,t.parent=t._dp=e,t},nc=function(e,t,n,s){n===void 0&&(n="_first"),s===void 0&&(s="_last");var r=t._prev,o=t._next;r?r._next=o:e[n]===t&&(e[n]=o),o?o._prev=r:e[s]===t&&(e[s]=r),t._next=t._prev=t.parent=null},Os=function(e,t){e.parent&&(!t||e.parent.autoRemoveChildren)&&e.parent.remove&&e.parent.remove(e),e._act=0},ur=function(e,t){if(e&&(!t||t._end>e._dur||t._start<0))for(var n=e;n;)n._dirty=1,n=n.parent;return e},e1=function(e){for(var t=e.parent;t&&t.parent;)t._dirty=1,t.totalDuration(),t=t.parent;return e},xf=function(e,t,n,s){return e._startAt&&(Jt?e._startAt.revert(_l):e.vars.immediateRender&&!e.vars.autoRevert||e._startAt.render(t,!0,s))},t1=function i(e){return!e||e._ts&&i(e.parent)},tm=function(e){return e._repeat?_o(e._tTime,e=e.duration()+e._rDelay)*e:0},_o=function(e,t){var n=Math.floor(e=Dt(e/t));return e&&n===e?n-1:n},Ol=function(e,t){return(e-t._start)*t._ts+(t._ts>=0?0:t._dirty?t.totalDuration():t._tDur)},ic=function(e){return e._end=Dt(e._start+(e._tDur/Math.abs(e._ts||e._rts||_t)||0))},sc=function(e,t){var n=e._dp;return n&&n.smoothChildTiming&&e._ts&&(e._start=Dt(n._time-(e._ts>0?t/e._ts:((e._dirty?e.totalDuration():e._tDur)-t)/-e._ts)),ic(e),n._dirty||ur(n,e)),e},bg=function(e,t){var n;if((t._time||!t._dur&&t._initted||t._start<e._time&&(t._dur||!t.add))&&(n=Ol(e.rawTime(),t),(!t._dur||Ca(0,t.totalDuration(),n)-t._tTime>_t)&&t.render(n,!0)),ur(e,t)._dp&&e._initted&&e._time>=e._dur&&e._ts){if(e._dur<e.duration())for(n=e;n._dp;)n.rawTime()>=0&&n.totalTime(n._tTime),n=n._dp;e._zTime=-_t}},Ti=function(e,t,n,s){return t.parent&&Os(t),t._start=Dt((ls(n)?n:n||e!==Pt?Zn(e,n,t):e._time)+t._delay),t._end=Dt(t._start+(t.totalDuration()/Math.abs(t.timeScale())||0)),yg(e,t,"_first","_last",e._sort?"_start":0),_f(t)||(e._recent=t),s||bg(e,t),e._ts<0&&sc(e,e._tTime),e},Mg=function(e,t){return(Yn.ScrollTrigger||Td("scrollTrigger",t))&&Yn.ScrollTrigger.create(t,e)},Cg=function(e,t,n,s,r){if(Pd(e,t,r),!e._initted)return 1;if(!n&&e._pt&&!Jt&&(e._dur&&e.vars.lazy!==!1||!e._dur&&e.vars.lazy)&&xg!==Hn.frame)return Ps.push(e),e._lazy=[r,s],1},n1=function i(e){var t=e.parent;return t&&t._ts&&t._initted&&!t._lock&&(t.rawTime()<0||i(t))},_f=function(e){var t=e.data;return t==="isFromStart"||t==="isStart"},i1=function(e,t,n,s){var r=e.ratio,o=t<0||!t&&(!e._start&&n1(e)&&!(!e._initted&&_f(e))||(e._ts<0||e._dp._ts<0)&&!_f(e))?0:1,a=e._rDelay,l=0,c,u,f;if(a&&e._repeat&&(l=Ca(0,e._tDur,t),u=_o(l,a),e._yoyo&&u&1&&(o=1-o),u!==_o(e._tTime,a)&&(r=1-o,e.vars.repeatRefresh&&e._initted&&e.invalidate())),o!==r||Jt||s||e._zTime===_t||!t&&e._zTime){if(!e._initted&&Cg(e,t,s,n,l))return;for(f=e._zTime,e._zTime=t||(n?_t:0),n||(n=t&&!f),e.ratio=o,e._from&&(o=1-o),e._time=0,e._tTime=l,c=e._pt;c;)c.r(o,c.d),c=c._next;t<0&&xf(e,t,n,!0),e._onUpdate&&!n&&Wn(e,"onUpdate"),l&&e._repeat&&!n&&e.parent&&Wn(e,"onRepeat"),(t>=e._tDur||t<0)&&e.ratio===o&&(o&&Os(e,1),!n&&!Jt&&(Wn(e,o?"onComplete":"onReverseComplete",!0),e._prom&&e._prom()))}else e._zTime||(e._zTime=t)},s1=function(e,t,n){var s;if(n>t)for(s=e._first;s&&s._start<=n;){if(s.data==="isPause"&&s._start>t)return s;s=s._next}else for(s=e._last;s&&s._start>=n;){if(s.data==="isPause"&&s._start<t)return s;s=s._prev}},Ao=function(e,t,n,s){var r=e._repeat,o=Dt(t)||0,a=e._tTime/e._tDur;return a&&!s&&(e._time*=o/e._dur),e._dur=o,e._tDur=r?r<0?1e10:Dt(o*(r+1)+e._rDelay*r):o,a>0&&!s&&sc(e,e._tTime=e._tDur*a),e.parent&&ic(e),n||ur(e.parent,e),e},nm=function(e){return e instanceof mn?ur(e):Ao(e,e._dur)},r1={_start:0,endTime:ha,totalDuration:ha},Zn=function i(e,t,n){var s=e.labels,r=e._recent||r1,o=e.duration()>=ri?r.endTime(!1):e._dur,a,l,c;return jt(t)&&(isNaN(t)||t in s)?(l=t.charAt(0),c=t.substr(-1)==="%",a=t.indexOf("="),l==="<"||l===">"?(a>=0&&(t=t.replace(/=/,"")),(l==="<"?r._start:r.endTime(r._repeat>=0))+(parseFloat(t.substr(1))||0)*(c?(a<0?r:n).totalDuration()/100:1)):a<0?(t in s||(s[t]=o),s[t]):(l=parseFloat(t.charAt(a-1)+t.substr(a+1)),c&&n&&(l=l/100*(an(n)?n[0]:n).totalDuration()),a>1?i(e,t.substr(0,a-1),n)+l:o+l)):t==null?o:+t},$o=function(e,t,n){var s=ls(t[1]),r=(s?2:1)+(e<2?0:1),o=t[r],a,l;if(s&&(o.duration=t[1]),o.parent=n,e){for(a=o,l=n;l&&!("immediateRender"in a);)a=l.vars.defaults||{},l=Rn(l.vars.inherit)&&l.parent;o.immediateRender=Rn(a.immediateRender),e<2?o.runBackwards=1:o.startAt=t[r-1]}return new Gt(t[0],o,t[r+1])},Hs=function(e,t){return e||e===0?t(e):t},Ca=function(e,t,n){return n<e?e:n>t?t:n},sn=function(e,t){return!jt(e)||!(t=YE.exec(e))?"":t[1]},o1=function(e,t,n){return Hs(n,function(s){return Ca(e,t,s)})},Af=[].slice,Tg=function(e,t){return e&&Oi(e)&&"length"in e&&(!t&&!e.length||e.length-1 in e&&Oi(e[0]))&&!e.nodeType&&e!==yi},a1=function(e,t,n){return n===void 0&&(n=[]),e.forEach(function(s){var r;return jt(s)&&!t||Tg(s,1)?(r=n).push.apply(r,oi(s)):n.push(s)})||n},oi=function(e,t,n){return Tt&&!t&&Tt.selector?Tt.selector(e):jt(e)&&!n&&(mf||!So())?Af.call((t||Cd).querySelectorAll(e),0):an(e)?a1(e,n):Tg(e)?Af.call(e,0):e?[e]:[]},Sf=function(e){return e=oi(e)[0]||da("Invalid scope")||{},function(t){var n=e.current||e.nativeElement||e;return oi(t,n.querySelectorAll?n:n===e?da("Invalid scope")||Cd.createElement("div"):e)}},Eg=function(e){return e.sort(function(){return .5-Math.random()})},wg=function(e){if(Ut(e))return e;var t=Oi(e)?e:{each:e},n=fr(t.ease),s=t.from||0,r=parseFloat(t.base)||0,o={},a=s>0&&s<1,l=isNaN(s)||a,c=t.axis,u=s,f=s;return jt(s)?u=f={center:.5,edges:.5,end:1}[s]||0:!a&&l&&(u=s[0],f=s[1]),function(d,h,x){var p=(x||t).length,g=o[p],m,_,A,S,v,y,M,E,b;if(!g){if(b=t.grid==="auto"?0:(t.grid||[1,ri])[1],!b){for(M=-ri;M<(M=x[b++].getBoundingClientRect().left)&&b<p;);b<p&&b--}for(g=o[p]=[],m=l?Math.min(b,p)*u-.5:s%b,_=b===ri?0:l?p*f/b-.5:s/b|0,M=0,E=ri,y=0;y<p;y++)A=y%b-m,S=_-(y/b|0),g[y]=v=c?Math.abs(c==="y"?S:A):ug(A*A+S*S),v>M&&(M=v),v<E&&(E=v);s==="random"&&Eg(g),g.max=M-E,g.min=E,g.v=p=(parseFloat(t.amount)||parseFloat(t.each)*(b>p?p-1:c?c==="y"?p/b:b:Math.max(b,p/b))||0)*(s==="edges"?-1:1),g.b=p<0?r-p:r,g.u=sn(t.amount||t.each)||0,n=n&&p<0?Ng(n):n}return p=(g[d]-g.min)/g.max||0,Dt(g.b+(n?n(p):p)*g.v)+g.u}},vf=function(e){var t=Math.pow(10,((e+"").split(".")[1]||"").length);return function(n){var s=Dt(Math.round(parseFloat(n)/e)*e*t);return(s-s%1)/t+(ls(n)?0:sn(n))}},Rg=function(e,t){var n=an(e),s,r;return!n&&Oi(e)&&(s=n=e.radius||ri,e.values?(e=oi(e.values),(r=!ls(e[0]))&&(s*=s)):e=vf(e.increment)),Hs(t,n?Ut(e)?function(o){return r=e(o),Math.abs(r-o)<=s?r:o}:function(o){for(var a=parseFloat(r?o.x:o),l=parseFloat(r?o.y:0),c=ri,u=0,f=e.length,d,h;f--;)r?(d=e[f].x-a,h=e[f].y-l,d=d*d+h*h):d=Math.abs(e[f]-a),d<c&&(c=d,u=f);return u=!s||c<=s?e[u]:o,r||u===o||ls(o)?u:u+sn(o)}:vf(e))},Ig=function(e,t,n,s){return Hs(an(e)?!t:n===!0?!!(n=0):!s,function(){return an(e)?e[~~(Math.random()*e.length)]:(n=n||1e-5)&&(s=n<1?Math.pow(10,(n+"").length-2):1)&&Math.floor(Math.round((e-n/2+Math.random()*(t-e+n*.99))/n)*n*s)/s})},l1=function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(s){return t.reduce(function(r,o){return o(r)},s)}},c1=function(e,t){return function(n){return e(parseFloat(n))+(t||sn(n))}},u1=function(e,t,n){return Pg(e,t,0,1,n)},Dg=function(e,t,n){return Hs(n,function(s){return e[~~t(s)]})},f1=function i(e,t,n){var s=t-e;return an(e)?Dg(e,i(0,e.length),t):Hs(n,function(r){return(s+(r-e)%s)%s+e})},d1=function i(e,t,n){var s=t-e,r=s*2;return an(e)?Dg(e,i(0,e.length-1),t):Hs(n,function(o){return o=(r+(o-e)%r)%r||0,e+(o>s?r-o:o)})},pa=function(e){return e.replace(XE,function(t){var n=t.indexOf("[")+1,s=t.substring(n||7,n?t.indexOf("]"):t.length-1).split(qE);return Ig(n?s:+s[0],n?0:+s[1],+s[2]||1e-5)})},Pg=function(e,t,n,s,r){var o=t-e,a=s-n;return Hs(r,function(l){return n+((l-e)/o*a||0)})},h1=function i(e,t,n,s){var r=isNaN(e+t)?0:function(h){return(1-h)*e+h*t};if(!r){var o=jt(e),a={},l,c,u,f,d;if(n===!0&&(s=1)&&(n=null),o)e={p:e},t={p:t};else if(an(e)&&!an(t)){for(u=[],f=e.length,d=f-2,c=1;c<f;c++)u.push(i(e[c-1],e[c]));f--,r=function(x){x*=f;var p=Math.min(d,~~x);return u[p](x-p)},n=t}else s||(e=xo(an(e)?[]:{},e));if(!u){for(l in t)Dd.call(a,e,l,"get",t[l]);r=function(x){return Bd(x,a)||(o?e.p:e)}}}return Hs(n,r)},im=function(e,t,n){var s=e.labels,r=ri,o,a,l;for(o in s)a=s[o]-t,a<0==!!n&&a&&r>(a=Math.abs(a))&&(l=o,r=a);return l},Wn=function(e,t,n){var s=e.vars,r=s[t],o=Tt,a=e._ctx,l,c,u;if(r)return l=s[t+"Params"],c=s.callbackScope||e,n&&Ps.length&&Bl(),a&&(Tt=a),u=l?r.apply(c,l):r.call(c),Tt=o,u},Uo=function(e){return Os(e),e.scrollTrigger&&e.scrollTrigger.kill(!!Jt),e.progress()<1&&Wn(e,"onInterrupt"),e},Xr,Fg=[],Lg=function(e){if(e)if(e=!e.name&&e.default||e,Md()||e.headless){var t=e.name,n=Ut(e),s=t&&!n&&e.init?function(){this._props=[]}:e,r={init:ha,render:Bd,add:Dd,kill:R1,modifier:w1,rawVars:0},o={targetTest:0,get:0,getSetter:Ld,aliases:{},register:0};if(So(),e!==s){if(kn[t])return;Kn(s,Kn(Ul(e,r),o)),xo(s.prototype,xo(r,Ul(e,o))),kn[s.prop=t]=s,e.targetTest&&(Al.push(s),Ed[t]=1),t=(t==="css"?"CSS":t.charAt(0).toUpperCase()+t.substr(1))+"Plugin"}gg(t,s),e.register&&e.register(Fn,s,Dn)}else Fg.push(e)},xt=255,Oo={aqua:[0,xt,xt],lime:[0,xt,0],silver:[192,192,192],black:[0,0,0],maroon:[128,0,0],teal:[0,128,128],blue:[0,0,xt],navy:[0,0,128],white:[xt,xt,xt],olive:[128,128,0],yellow:[xt,xt,0],orange:[xt,165,0],gray:[128,128,128],purple:[128,0,128],green:[0,128,0],red:[xt,0,0],pink:[xt,192,203],cyan:[0,xt,xt],transparent:[xt,xt,xt,0]},ou=function(e,t,n){return e+=e<0?1:e>1?-1:0,(e*6<1?t+(n-t)*e*6:e<.5?n:e*3<2?t+(n-t)*(2/3-e)*6:t)*xt+.5|0},Bg=function(e,t,n){var s=e?ls(e)?[e>>16,e>>8&xt,e&xt]:0:Oo.black,r,o,a,l,c,u,f,d,h,x;if(!s){if(e.substr(-1)===","&&(e=e.substr(0,e.length-1)),Oo[e])s=Oo[e];else if(e.charAt(0)==="#"){if(e.length<6&&(r=e.charAt(1),o=e.charAt(2),a=e.charAt(3),e="#"+r+r+o+o+a+a+(e.length===5?e.charAt(4)+e.charAt(4):"")),e.length===9)return s=parseInt(e.substr(1,6),16),[s>>16,s>>8&xt,s&xt,parseInt(e.substr(7),16)/255];e=parseInt(e.substr(1),16),s=[e>>16,e>>8&xt,e&xt]}else if(e.substr(0,3)==="hsl"){if(s=x=e.match(Zp),!t)l=+s[0]%360/360,c=+s[1]/100,u=+s[2]/100,o=u<=.5?u*(c+1):u+c-u*c,r=u*2-o,s.length>3&&(s[3]*=1),s[0]=ou(l+1/3,r,o),s[1]=ou(l,r,o),s[2]=ou(l-1/3,r,o);else if(~e.indexOf("="))return s=e.match(dg),n&&s.length<4&&(s[3]=1),s}else s=e.match(Zp)||Oo.transparent;s=s.map(Number)}return t&&!x&&(r=s[0]/xt,o=s[1]/xt,a=s[2]/xt,f=Math.max(r,o,a),d=Math.min(r,o,a),u=(f+d)/2,f===d?l=c=0:(h=f-d,c=u>.5?h/(2-f-d):h/(f+d),l=f===r?(o-a)/h+(o<a?6:0):f===o?(a-r)/h+2:(r-o)/h+4,l*=60),s[0]=~~(l+.5),s[1]=~~(c*100+.5),s[2]=~~(u*100+.5)),n&&s.length<4&&(s[3]=1),s},Ug=function(e){var t=[],n=[],s=-1;return e.split(Fs).forEach(function(r){var o=r.match(Wr)||[];t.push.apply(t,o),n.push(s+=o.length+1)}),t.c=n,t},sm=function(e,t,n){var s="",r=(e+s).match(Fs),o=t?"hsla(":"rgba(",a=0,l,c,u,f;if(!r)return e;if(r=r.map(function(d){return(d=Bg(d,t,1))&&o+(t?d[0]+","+d[1]+"%,"+d[2]+"%,"+d[3]:d.join(","))+")"}),n&&(u=Ug(e),l=n.c,l.join(s)!==u.c.join(s)))for(c=e.replace(Fs,"1").split(Wr),f=c.length-1;a<f;a++)s+=c[a]+(~l.indexOf(a)?r.shift()||o+"0,0,0,0)":(u.length?u:r.length?r:n).shift());if(!c)for(c=e.split(Fs),f=c.length-1;a<f;a++)s+=c[a]+r[a];return s+c[f]},Fs=(function(){var i="(?:\\b(?:(?:rgb|rgba|hsl|hsla)\\(.+?\\))|\\B#(?:[0-9a-f]{3,4}){1,2}\\b",e;for(e in Oo)i+="|"+e+"\\b";return new RegExp(i+")","gi")})(),p1=/hsl[a]?\(/,Og=function(e){var t=e.join(" "),n;if(Fs.lastIndex=0,Fs.test(t))return n=p1.test(t),e[1]=sm(e[1],n),e[0]=sm(e[0],n,Ug(e[1])),!0},ma,Hn=(function(){var i=Date.now,e=500,t=33,n=i(),s=n,r=1e3/240,o=r,a=[],l,c,u,f,d,h,x=function p(g){var m=i()-s,_=g===!0,A,S,v,y;if((m>e||m<0)&&(n+=m-t),s+=m,v=s-n,A=v-o,(A>0||_)&&(y=++f.frame,d=v-f.time*1e3,f.time=v=v/1e3,o+=A+(A>=r?4:r-A),S=1),_||(l=c(p)),S)for(h=0;h<a.length;h++)a[h](v,d,y,g)};return f={time:0,frame:0,tick:function(){x(!0)},deltaRatio:function(g){return d/(1e3/(g||60))},wake:function(){pg&&(!mf&&Md()&&(yi=mf=window,Cd=yi.document||{},Yn.gsap=Fn,(yi.gsapVersions||(yi.gsapVersions=[])).push(Fn.version),mg(Ll||yi.GreenSockGlobals||!yi.gsap&&yi||{}),Fg.forEach(Lg)),u=typeof requestAnimationFrame<"u"&&requestAnimationFrame,l&&f.sleep(),c=u||function(g){return setTimeout(g,o-f.time*1e3+1|0)},ma=1,x(2))},sleep:function(){(u?cancelAnimationFrame:clearTimeout)(l),ma=0,c=ha},lagSmoothing:function(g,m){e=g||1/0,t=Math.min(m||33,e)},fps:function(g){r=1e3/(g||240),o=f.time*1e3+r},add:function(g,m,_){var A=m?function(S,v,y,M){g(S,v,y,M),f.remove(A)}:g;return f.remove(g),a[_?"unshift":"push"](A),So(),A},remove:function(g,m){~(m=a.indexOf(g))&&a.splice(m,1)&&h>=m&&h--},_listeners:a},f})(),So=function(){return!ma&&Hn.wake()},tt={},m1=/^[\d.\-M][\d.\-,\s]/,g1=/["']/g,x1=function(e){for(var t={},n=e.substr(1,e.length-3).split(":"),s=n[0],r=1,o=n.length,a,l,c;r<o;r++)l=n[r],a=r!==o-1?l.lastIndexOf(","):l.length,c=l.substr(0,a),t[s]=isNaN(c)?c.replace(g1,"").trim():+c,s=l.substr(a+1).trim();return t},_1=function(e){var t=e.indexOf("(")+1,n=e.indexOf(")"),s=e.indexOf("(",t);return e.substring(t,~s&&s<n?e.indexOf(")",n+1):n)},A1=function(e){var t=(e+"").split("("),n=tt[t[0]];return n&&t.length>1&&n.config?n.config.apply(null,~e.indexOf("{")?[x1(t[1])]:_1(e).split(",").map(Sg)):tt._CE&&m1.test(e)?tt._CE("",e):n},Ng=function(e){return function(t){return 1-e(1-t)}},zg=function i(e,t){for(var n=e._first,s;n;)n instanceof mn?i(n,t):n.vars.yoyoEase&&(!n._yoyo||!n._repeat)&&n._yoyo!==t&&(n.timeline?i(n.timeline,t):(s=n._ease,n._ease=n._yEase,n._yEase=s,n._yoyo=t)),n=n._next},fr=function(e,t){return e&&(Ut(e)?e:tt[e]||A1(e))||t},gr=function(e,t,n,s){n===void 0&&(n=function(l){return 1-t(1-l)}),s===void 0&&(s=function(l){return l<.5?t(l*2)/2:1-t((1-l)*2)/2});var r={easeIn:t,easeOut:n,easeInOut:s},o;return In(e,function(a){tt[a]=Yn[a]=r,tt[o=a.toLowerCase()]=n;for(var l in r)tt[o+(l==="easeIn"?".in":l==="easeOut"?".out":".inOut")]=tt[a+"."+l]=r[l]}),r},kg=function(e){return function(t){return t<.5?(1-e(1-t*2))/2:.5+e((t-.5)*2)/2}},au=function i(e,t,n){var s=t>=1?t:1,r=(n||(e?.3:.45))/(t<1?t:1),o=r/pf*(Math.asin(1/s)||0),a=function(u){return u===1?1:s*Math.pow(2,-10*u)*WE((u-o)*r)+1},l=e==="out"?a:e==="in"?function(c){return 1-a(1-c)}:kg(a);return r=pf/r,l.config=function(c,u){return i(e,c,u)},l},lu=function i(e,t){t===void 0&&(t=1.70158);var n=function(o){return o?--o*o*((t+1)*o+t)+1:0},s=e==="out"?n:e==="in"?function(r){return 1-n(1-r)}:kg(n);return s.config=function(r){return i(e,r)},s};In("Linear,Quad,Cubic,Quart,Quint,Strong",function(i,e){var t=e<5?e+1:e;gr(i+",Power"+(t-1),e?function(n){return Math.pow(n,t)}:function(n){return n},function(n){return 1-Math.pow(1-n,t)},function(n){return n<.5?Math.pow(n*2,t)/2:1-Math.pow((1-n)*2,t)/2})});tt.Linear.easeNone=tt.none=tt.Linear.easeIn;gr("Elastic",au("in"),au("out"),au());(function(i,e){var t=1/e,n=2*t,s=2.5*t,r=function(a){return a<t?i*a*a:a<n?i*Math.pow(a-1.5/e,2)+.75:a<s?i*(a-=2.25/e)*a+.9375:i*Math.pow(a-2.625/e,2)+.984375};gr("Bounce",function(o){return 1-r(1-o)},r)})(7.5625,2.75);gr("Expo",function(i){return Math.pow(2,10*(i-1))*i+i*i*i*i*i*i*(1-i)});gr("Circ",function(i){return-(ug(1-i*i)-1)});gr("Sine",function(i){return i===1?1:-GE(i*HE)+1});gr("Back",lu("in"),lu("out"),lu());tt.SteppedEase=tt.steps=Yn.SteppedEase={config:function(e,t){e===void 0&&(e=1);var n=1/e,s=e+(t?0:1),r=t?1:0,o=1-_t;return function(a){return((s*Ca(0,o,a)|0)+r)*n}}};go.ease=tt["quad.out"];In("onComplete,onUpdate,onStart,onRepeat,onReverseComplete,onInterrupt",function(i){return wd+=i+","+i+"Params,"});var Hg=function(e,t){this.id=VE++,e._gsap=this,this.target=e,this.harness=t,this.get=t?t.get:_g,this.set=t?t.getSetter:Ld},ga=(function(){function i(t){this.vars=t,this._delay=+t.delay||0,(this._repeat=t.repeat===1/0?-2:t.repeat||0)&&(this._rDelay=t.repeatDelay||0,this._yoyo=!!t.yoyo||!!t.yoyoEase),this._ts=1,Ao(this,+t.duration,1,1),this.data=t.data,Tt&&(this._ctx=Tt,Tt.data.push(this)),ma||Hn.wake()}var e=i.prototype;return e.delay=function(n){return n||n===0?(this.parent&&this.parent.smoothChildTiming&&this.startTime(this._start+n-this._delay),this._delay=n,this):this._delay},e.duration=function(n){return arguments.length?this.totalDuration(this._repeat>0?n+(n+this._rDelay)*this._repeat:n):this.totalDuration()&&this._dur},e.totalDuration=function(n){return arguments.length?(this._dirty=0,Ao(this,this._repeat<0?n:(n-this._repeat*this._rDelay)/(this._repeat+1))):this._tDur},e.totalTime=function(n,s){if(So(),!arguments.length)return this._tTime;var r=this._dp;if(r&&r.smoothChildTiming&&this._ts){for(sc(this,n),!r._dp||r.parent||bg(r,this);r&&r.parent;)r.parent._time!==r._start+(r._ts>=0?r._tTime/r._ts:(r.totalDuration()-r._tTime)/-r._ts)&&r.totalTime(r._tTime,!0),r=r.parent;!this.parent&&this._dp.autoRemoveChildren&&(this._ts>0&&n<this._tDur||this._ts<0&&n>0||!this._tDur&&!n)&&Ti(this._dp,this,this._start-this._delay)}return(this._tTime!==n||!this._dur&&!s||this._initted&&Math.abs(this._zTime)===_t||!this._initted&&this._dur&&n||!n&&!this._initted&&(this.add||this._ptLookup))&&(this._ts||(this._pTime=n),Ag(this,n,s)),this},e.time=function(n,s){return arguments.length?this.totalTime(Math.min(this.totalDuration(),n+tm(this))%(this._dur+this._rDelay)||(n?this._dur:0),s):this._time},e.totalProgress=function(n,s){return arguments.length?this.totalTime(this.totalDuration()*n,s):this.totalDuration()?Math.min(1,this._tTime/this._tDur):this.rawTime()>=0&&this._initted?1:0},e.progress=function(n,s){return arguments.length?this.totalTime(this.duration()*(this._yoyo&&!(this.iteration()&1)?1-n:n)+tm(this),s):this.duration()?Math.min(1,this._time/this._dur):this.rawTime()>0?1:0},e.iteration=function(n,s){var r=this.duration()+this._rDelay;return arguments.length?this.totalTime(this._time+(n-1)*r,s):this._repeat?_o(this._tTime,r)+1:1},e.timeScale=function(n,s){if(!arguments.length)return this._rts===-_t?0:this._rts;if(this._rts===n)return this;var r=this.parent&&this._ts?Ol(this.parent._time,this):this._tTime;return this._rts=+n||0,this._ts=this._ps||n===-_t?0:this._rts,this.totalTime(Ca(-Math.abs(this._delay),this.totalDuration(),r),s!==!1),ic(this),e1(this)},e.paused=function(n){return arguments.length?(this._ps!==n&&(this._ps=n,n?(this._pTime=this._tTime||Math.max(-this._delay,this.rawTime()),this._ts=this._act=0):(So(),this._ts=this._rts,this.totalTime(this.parent&&!this.parent.smoothChildTiming?this.rawTime():this._tTime||this._pTime,this.progress()===1&&Math.abs(this._zTime)!==_t&&(this._tTime-=_t)))),this):this._ps},e.startTime=function(n){if(arguments.length){this._start=Dt(n);var s=this.parent||this._dp;return s&&(s._sort||!this.parent)&&Ti(s,this,this._start-this._delay),this}return this._start},e.endTime=function(n){return this._start+(Rn(n)?this.totalDuration():this.duration())/Math.abs(this._ts||1)},e.rawTime=function(n){var s=this.parent||this._dp;return s?n&&(!this._ts||this._repeat&&this._time&&this.totalProgress()<1)?this._tTime%(this._dur+this._rDelay):this._ts?Ol(s.rawTime(n),this):this._tTime:this._tTime},e.revert=function(n){n===void 0&&(n=jE);var s=Jt;return Jt=n,Id(this)&&(this.timeline&&this.timeline.revert(n),this.totalTime(-.01,n.suppressEvents)),this.data!=="nested"&&n.kill!==!1&&this.kill(),Jt=s,this},e.globalTime=function(n){for(var s=this,r=arguments.length?n:s.rawTime();s;)r=s._start+r/(Math.abs(s._ts)||1),s=s._dp;return!this.parent&&this._sat?this._sat.globalTime(n):r},e.repeat=function(n){return arguments.length?(this._repeat=n===1/0?-2:n,nm(this)):this._repeat===-2?1/0:this._repeat},e.repeatDelay=function(n){if(arguments.length){var s=this._time;return this._rDelay=n,nm(this),s?this.time(s):this}return this._rDelay},e.yoyo=function(n){return arguments.length?(this._yoyo=n,this):this._yoyo},e.seek=function(n,s){return this.totalTime(Zn(this,n),Rn(s))},e.restart=function(n,s){return this.play().totalTime(n?-this._delay:0,Rn(s)),this._dur||(this._zTime=-_t),this},e.play=function(n,s){return n!=null&&this.seek(n,s),this.reversed(!1).paused(!1)},e.reverse=function(n,s){return n!=null&&this.seek(n||this.totalDuration(),s),this.reversed(!0).paused(!1)},e.pause=function(n,s){return n!=null&&this.seek(n,s),this.paused(!0)},e.resume=function(){return this.paused(!1)},e.reversed=function(n){return arguments.length?(!!n!==this.reversed()&&this.timeScale(-this._rts||(n?-_t:0)),this):this._rts<0},e.invalidate=function(){return this._initted=this._act=0,this._zTime=-_t,this},e.isActive=function(){var n=this.parent||this._dp,s=this._start,r;return!!(!n||this._ts&&this._initted&&n.isActive()&&(r=n.rawTime(!0))>=s&&r<this.endTime(!0)-_t)},e.eventCallback=function(n,s,r){var o=this.vars;return arguments.length>1?(s?(o[n]=s,r&&(o[n+"Params"]=r),n==="onUpdate"&&(this._onUpdate=s)):delete o[n],this):o[n]},e.then=function(n){var s=this,r=s._prom;return new Promise(function(o){var a=Ut(n)?n:vg,l=function(){var u=s.then;s.then=null,r&&r(),Ut(a)&&(a=a(s))&&(a.then||a===s)&&(s.then=u),o(a),s.then=u};s._initted&&s.totalProgress()===1&&s._ts>=0||!s._tTime&&s._ts<0?l():s._prom=l})},e.kill=function(){Uo(this)},i})();Kn(ga.prototype,{_time:0,_start:0,_end:0,_tTime:0,_tDur:0,_dirty:0,_repeat:0,_yoyo:!1,parent:null,_initted:!1,_rDelay:0,_ts:1,_dp:0,ratio:0,_zTime:-_t,_prom:0,_ps:!1,_rts:1});var mn=(function(i){cg(e,i);function e(n,s){var r;return n===void 0&&(n={}),r=i.call(this,n)||this,r.labels={},r.smoothChildTiming=!!n.smoothChildTiming,r.autoRemoveChildren=!!n.autoRemoveChildren,r._sort=Rn(n.sortChildren),Pt&&Ti(n.parent||Pt,Ki(r),s),n.reversed&&r.reverse(),n.paused&&r.paused(!0),n.scrollTrigger&&Mg(Ki(r),n.scrollTrigger),r}var t=e.prototype;return t.to=function(s,r,o){return $o(0,arguments,this),this},t.from=function(s,r,o){return $o(1,arguments,this),this},t.fromTo=function(s,r,o,a){return $o(2,arguments,this),this},t.set=function(s,r,o){return r.duration=0,r.parent=this,jo(r).repeatDelay||(r.repeat=0),r.immediateRender=!!r.immediateRender,new Gt(s,r,Zn(this,o),1),this},t.call=function(s,r,o){return Ti(this,Gt.delayedCall(0,s,r),o)},t.staggerTo=function(s,r,o,a,l,c,u){return o.duration=r,o.stagger=o.stagger||a,o.onComplete=c,o.onCompleteParams=u,o.parent=this,new Gt(s,o,Zn(this,l)),this},t.staggerFrom=function(s,r,o,a,l,c,u){return o.runBackwards=1,jo(o).immediateRender=Rn(o.immediateRender),this.staggerTo(s,r,o,a,l,c,u)},t.staggerFromTo=function(s,r,o,a,l,c,u,f){return a.startAt=o,jo(a).immediateRender=Rn(a.immediateRender),this.staggerTo(s,r,a,l,c,u,f)},t.render=function(s,r,o){var a=this._time,l=this._dirty?this.totalDuration():this._tDur,c=this._dur,u=s<=0?0:Dt(s),f=this._zTime<0!=s<0&&(this._initted||!c),d,h,x,p,g,m,_,A,S,v,y,M;if(this!==Pt&&u>l&&s>=0&&(u=l),u!==this._tTime||o||f){if(a!==this._time&&c&&(u+=this._time-a,s+=this._time-a),d=u,S=this._start,A=this._ts,m=!A,f&&(c||(a=this._zTime),(s||!r)&&(this._zTime=s)),this._repeat){if(y=this._yoyo,g=c+this._rDelay,this._repeat<-1&&s<0)return this.totalTime(g*100+s,r,o);if(d=Dt(u%g),u===l?(p=this._repeat,d=c):(v=Dt(u/g),p=~~v,p&&p===v&&(d=c,p--),d>c&&(d=c)),v=_o(this._tTime,g),!a&&this._tTime&&v!==p&&this._tTime-v*g-this._dur<=0&&(v=p),y&&p&1&&(d=c-d,M=1),p!==v&&!this._lock){var E=y&&v&1,b=E===(y&&p&1);if(p<v&&(E=!E),a=E?0:u%c?c:u,this._lock=1,this.render(a||(M?0:Dt(p*g)),r,!c)._lock=0,this._tTime=u,!r&&this.parent&&Wn(this,"onRepeat"),this.vars.repeatRefresh&&!M&&(this.invalidate()._lock=1,v=p),a&&a!==this._time||m!==!this._ts||this.vars.onRepeat&&!this.parent&&!this._act)return this;if(c=this._dur,l=this._tDur,b&&(this._lock=2,a=E?c:-1e-4,this.render(a,!0),this.vars.repeatRefresh&&!M&&this.invalidate()),this._lock=0,!this._ts&&!m)return this;zg(this,M)}}if(this._hasPause&&!this._forcing&&this._lock<2&&(_=s1(this,Dt(a),Dt(d)),_&&(u-=d-(d=_._start))),this._tTime=u,this._time=d,this._act=!A,this._initted||(this._onUpdate=this.vars.onUpdate,this._initted=1,this._zTime=s,a=0),!a&&u&&c&&!r&&!v&&(Wn(this,"onStart"),this._tTime!==u))return this;if(d>=a&&s>=0)for(h=this._first;h;){if(x=h._next,(h._act||d>=h._start)&&h._ts&&_!==h){if(h.parent!==this)return this.render(s,r,o);if(h.render(h._ts>0?(d-h._start)*h._ts:(h._dirty?h.totalDuration():h._tDur)+(d-h._start)*h._ts,r,o),d!==this._time||!this._ts&&!m){_=0,x&&(u+=this._zTime=-_t);break}}h=x}else{h=this._last;for(var C=s<0?s:d;h;){if(x=h._prev,(h._act||C<=h._end)&&h._ts&&_!==h){if(h.parent!==this)return this.render(s,r,o);if(h.render(h._ts>0?(C-h._start)*h._ts:(h._dirty?h.totalDuration():h._tDur)+(C-h._start)*h._ts,r,o||Jt&&Id(h)),d!==this._time||!this._ts&&!m){_=0,x&&(u+=this._zTime=C?-_t:_t);break}}h=x}}if(_&&!r&&(this.pause(),_.render(d>=a?0:-_t)._zTime=d>=a?1:-1,this._ts))return this._start=S,ic(this),this.render(s,r,o);this._onUpdate&&!r&&Wn(this,"onUpdate",!0),(u===l&&this._tTime>=this.totalDuration()||!u&&a)&&(S===this._start||Math.abs(A)!==Math.abs(this._ts))&&(this._lock||((s||!c)&&(u===l&&this._ts>0||!u&&this._ts<0)&&Os(this,1),!r&&!(s<0&&!a)&&(u||a||!l)&&(Wn(this,u===l&&s>=0?"onComplete":"onReverseComplete",!0),this._prom&&!(u<l&&this.timeScale()>0)&&this._prom())))}return this},t.add=function(s,r){var o=this;if(ls(r)||(r=Zn(this,r,s)),!(s instanceof ga)){if(an(s))return s.forEach(function(a){return o.add(a,r)}),this;if(jt(s))return this.addLabel(s,r);if(Ut(s))s=Gt.delayedCall(0,s);else return this}return this!==s?Ti(this,s,r):this},t.getChildren=function(s,r,o,a){s===void 0&&(s=!0),r===void 0&&(r=!0),o===void 0&&(o=!0),a===void 0&&(a=-ri);for(var l=[],c=this._first;c;)c._start>=a&&(c instanceof Gt?r&&l.push(c):(o&&l.push(c),s&&l.push.apply(l,c.getChildren(!0,r,o)))),c=c._next;return l},t.getById=function(s){for(var r=this.getChildren(1,1,1),o=r.length;o--;)if(r[o].vars.id===s)return r[o]},t.remove=function(s){return jt(s)?this.removeLabel(s):Ut(s)?this.killTweensOf(s):(s.parent===this&&nc(this,s),s===this._recent&&(this._recent=this._last),ur(this))},t.totalTime=function(s,r){return arguments.length?(this._forcing=1,!this._dp&&this._ts&&(this._start=Dt(Hn.time-(this._ts>0?s/this._ts:(this.totalDuration()-s)/-this._ts))),i.prototype.totalTime.call(this,s,r),this._forcing=0,this):this._tTime},t.addLabel=function(s,r){return this.labels[s]=Zn(this,r),this},t.removeLabel=function(s){return delete this.labels[s],this},t.addPause=function(s,r,o){var a=Gt.delayedCall(0,r||ha,o);return a.data="isPause",this._hasPause=1,Ti(this,a,Zn(this,s))},t.removePause=function(s){var r=this._first;for(s=Zn(this,s);r;)r._start===s&&r.data==="isPause"&&Os(r),r=r._next},t.killTweensOf=function(s,r,o){for(var a=this.getTweensOf(s,o),l=a.length;l--;)Ts!==a[l]&&a[l].kill(s,r);return this},t.getTweensOf=function(s,r){for(var o=[],a=oi(s),l=this._first,c=ls(r),u;l;)l instanceof Gt?$E(l._targets,a)&&(c?(!Ts||l._initted&&l._ts)&&l.globalTime(0)<=r&&l.globalTime(l.totalDuration())>r:!r||l.isActive())&&o.push(l):(u=l.getTweensOf(a,r)).length&&o.push.apply(o,u),l=l._next;return o},t.tweenTo=function(s,r){r=r||{};var o=this,a=Zn(o,s),l=r,c=l.startAt,u=l.onStart,f=l.onStartParams,d=l.immediateRender,h,x=Gt.to(o,Kn({ease:r.ease||"none",lazy:!1,immediateRender:!1,time:a,overwrite:"auto",duration:r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale())||_t,onStart:function(){if(o.pause(),!h){var g=r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale());x._dur!==g&&Ao(x,g,0,1).render(x._time,!0,!0),h=1}u&&u.apply(x,f||[])}},r));return d?x.render(0):x},t.tweenFromTo=function(s,r,o){return this.tweenTo(r,Kn({startAt:{time:Zn(this,s)}},o))},t.recent=function(){return this._recent},t.nextLabel=function(s){return s===void 0&&(s=this._time),im(this,Zn(this,s))},t.previousLabel=function(s){return s===void 0&&(s=this._time),im(this,Zn(this,s),1)},t.currentLabel=function(s){return arguments.length?this.seek(s,!0):this.previousLabel(this._time+_t)},t.shiftChildren=function(s,r,o){o===void 0&&(o=0);var a=this._first,l=this.labels,c;for(s=Dt(s);a;)a._start>=o&&(a._start+=s,a._end+=s),a=a._next;if(r)for(c in l)l[c]>=o&&(l[c]+=s);return ur(this)},t.invalidate=function(s){var r=this._first;for(this._lock=0;r;)r.invalidate(s),r=r._next;return i.prototype.invalidate.call(this,s)},t.clear=function(s){s===void 0&&(s=!0);for(var r=this._first,o;r;)o=r._next,this.remove(r),r=o;return this._dp&&(this._time=this._tTime=this._pTime=0),s&&(this.labels={}),ur(this)},t.totalDuration=function(s){var r=0,o=this,a=o._last,l=ri,c,u,f;if(arguments.length)return o.timeScale((o._repeat<0?o.duration():o.totalDuration())/(o.reversed()?-s:s));if(o._dirty){for(f=o.parent;a;)c=a._prev,a._dirty&&a.totalDuration(),u=a._start,u>l&&o._sort&&a._ts&&!o._lock?(o._lock=1,Ti(o,a,u-a._delay,1)._lock=0):l=u,u<0&&a._ts&&(r-=u,(!f&&!o._dp||f&&f.smoothChildTiming)&&(o._start+=Dt(u/o._ts),o._time-=u,o._tTime-=u),o.shiftChildren(-u,!1,-1/0),l=0),a._end>r&&a._ts&&(r=a._end),a=c;Ao(o,o===Pt&&o._time>r?o._time:r,1,1),o._dirty=0}return o._tDur},e.updateRoot=function(s){if(Pt._ts&&(Ag(Pt,Ol(s,Pt)),xg=Hn.frame),Hn.frame>=Jp){Jp+=Qn.autoSleep||120;var r=Pt._first;if((!r||!r._ts)&&Qn.autoSleep&&Hn._listeners.length<2){for(;r&&!r._ts;)r=r._next;r||Hn.sleep()}}},e})(ga);Kn(mn.prototype,{_lock:0,_hasPause:0,_forcing:0});var S1=function(e,t,n,s,r,o,a){var l=new Dn(this._pt,e,t,0,1,Qg,null,r),c=0,u=0,f,d,h,x,p,g,m,_;for(l.b=n,l.e=s,n+="",s+="",(m=~s.indexOf("random("))&&(s=pa(s)),o&&(_=[n,s],o(_,e,t),n=_[0],s=_[1]),d=n.match(su)||[];f=su.exec(s);)x=f[0],p=s.substring(c,f.index),h?h=(h+1)%5:p.substr(-5)==="rgba("&&(h=1),x!==d[u++]&&(g=parseFloat(d[u-1])||0,l._pt={_next:l._pt,p:p||u===1?p:",",s:g,c:x.charAt(1)==="="?no(g,x)-g:parseFloat(x)-g,m:h&&h<4?Math.round:0},c=su.lastIndex);return l.c=c<s.length?s.substring(c,s.length):"",l.fp=a,(hg.test(s)||m)&&(l.e=0),this._pt=l,l},Dd=function(e,t,n,s,r,o,a,l,c,u){Ut(s)&&(s=s(r||0,e,o));var f=e[t],d=n!=="get"?n:Ut(f)?c?e[t.indexOf("set")||!Ut(e["get"+t.substr(3)])?t:"get"+t.substr(3)](c):e[t]():f,h=Ut(f)?c?C1:Xg:Fd,x;if(jt(s)&&(~s.indexOf("random(")&&(s=pa(s)),s.charAt(1)==="="&&(x=no(d,s)+(sn(d)||0),(x||x===0)&&(s=x))),!u||d!==s||yf)return!isNaN(d*s)&&s!==""?(x=new Dn(this._pt,e,t,+d||0,s-(d||0),typeof f=="boolean"?E1:qg,0,h),c&&(x.fp=c),a&&x.modifier(a,this,e),this._pt=x):(!f&&!(t in e)&&Td(t,s),S1.call(this,e,t,d,s,h,l||Qn.stringFilter,c))},v1=function(e,t,n,s,r){if(Ut(e)&&(e=Zo(e,r,t,n,s)),!Oi(e)||e.style&&e.nodeType||an(e)||fg(e))return jt(e)?Zo(e,r,t,n,s):e;var o={},a;for(a in e)o[a]=Zo(e[a],r,t,n,s);return o},Vg=function(e,t,n,s,r,o){var a,l,c,u;if(kn[e]&&(a=new kn[e]).init(r,a.rawVars?t[e]:v1(t[e],s,r,o,n),n,s,o)!==!1&&(n._pt=l=new Dn(n._pt,r,e,0,1,a.render,a,0,a.priority),n!==Xr))for(c=n._ptLookup[n._targets.indexOf(r)],u=a._props.length;u--;)c[a._props[u]]=l;return a},Ts,yf,Pd=function i(e,t,n){var s=e.vars,r=s.ease,o=s.startAt,a=s.immediateRender,l=s.lazy,c=s.onUpdate,u=s.runBackwards,f=s.yoyoEase,d=s.keyframes,h=s.autoRevert,x=e._dur,p=e._startAt,g=e._targets,m=e.parent,_=m&&m.data==="nested"?m.vars.targets:g,A=e._overwrite==="auto"&&!yd,S=e.timeline,v,y,M,E,b,C,I,F,U,O,k,z,V;if(S&&(!d||!r)&&(r="none"),e._ease=fr(r,go.ease),e._yEase=f?Ng(fr(f===!0?r:f,go.ease)):0,f&&e._yoyo&&!e._repeat&&(f=e._yEase,e._yEase=e._ease,e._ease=f),e._from=!S&&!!s.runBackwards,!S||d&&!s.stagger){if(F=g[0]?cr(g[0]).harness:0,z=F&&s[F.prop],v=Ul(s,Ed),p&&(p._zTime<0&&p.progress(1),t<0&&u&&a&&!h?p.render(-1,!0):p.revert(u&&x?_l:KE),p._lazy=0),o){if(Os(e._startAt=Gt.set(g,Kn({data:"isStart",overwrite:!1,parent:m,immediateRender:!0,lazy:!p&&Rn(l),startAt:null,delay:0,onUpdate:c&&function(){return Wn(e,"onUpdate")},stagger:0},o))),e._startAt._dp=0,e._startAt._sat=e,t<0&&(Jt||!a&&!h)&&e._startAt.revert(_l),a&&x&&t<=0&&n<=0){t&&(e._zTime=t);return}}else if(u&&x&&!p){if(t&&(a=!1),M=Kn({overwrite:!1,data:"isFromStart",lazy:a&&!p&&Rn(l),immediateRender:a,stagger:0,parent:m},v),z&&(M[F.prop]=z),Os(e._startAt=Gt.set(g,M)),e._startAt._dp=0,e._startAt._sat=e,t<0&&(Jt?e._startAt.revert(_l):e._startAt.render(-1,!0)),e._zTime=t,!a)i(e._startAt,_t,_t);else if(!t)return}for(e._pt=e._ptCache=0,l=x&&Rn(l)||l&&!x,y=0;y<g.length;y++){if(b=g[y],I=b._gsap||Rd(g)[y]._gsap,e._ptLookup[y]=O={},gf[I.id]&&Ps.length&&Bl(),k=_===g?y:_.indexOf(b),F&&(U=new F).init(b,z||v,e,k,_)!==!1&&(e._pt=E=new Dn(e._pt,b,U.name,0,1,U.render,U,0,U.priority),U._props.forEach(function(H){O[H]=E}),U.priority&&(C=1)),!F||z)for(M in v)kn[M]&&(U=Vg(M,v,e,k,b,_))?U.priority&&(C=1):O[M]=E=Dd.call(e,b,M,"get",v[M],k,_,0,s.stringFilter);e._op&&e._op[y]&&e.kill(b,e._op[y]),A&&e._pt&&(Ts=e,Pt.killTweensOf(b,O,e.globalTime(t)),V=!e.parent,Ts=0),e._pt&&l&&(gf[I.id]=1)}C&&Yg(e),e._onInit&&e._onInit(e)}e._onUpdate=c,e._initted=(!e._op||e._pt)&&!V,d&&t<=0&&S.render(ri,!0,!0)},y1=function(e,t,n,s,r,o,a,l){var c=(e._pt&&e._ptCache||(e._ptCache={}))[t],u,f,d,h;if(!c)for(c=e._ptCache[t]=[],d=e._ptLookup,h=e._targets.length;h--;){if(u=d[h][t],u&&u.d&&u.d._pt)for(u=u.d._pt;u&&u.p!==t&&u.fp!==t;)u=u._next;if(!u)return yf=1,e.vars[t]="+=0",Pd(e,a),yf=0,l?da(t+" not eligible for reset"):1;c.push(u)}for(h=c.length;h--;)f=c[h],u=f._pt||f,u.s=(s||s===0)&&!r?s:u.s+(s||0)+o*u.c,u.c=n-u.s,f.e&&(f.e=Ot(n)+sn(f.e)),f.b&&(f.b=u.s+sn(f.b))},b1=function(e,t){var n=e[0]?cr(e[0]).harness:0,s=n&&n.aliases,r,o,a,l;if(!s)return t;r=xo({},t);for(o in s)if(o in r)for(l=s[o].split(","),a=l.length;a--;)r[l[a]]=r[o];return r},M1=function(e,t,n,s){var r=t.ease||s||"power1.inOut",o,a;if(an(t))a=n[e]||(n[e]=[]),t.forEach(function(l,c){return a.push({t:c/(t.length-1)*100,v:l,e:r})});else for(o in t)a=n[o]||(n[o]=[]),o==="ease"||a.push({t:parseFloat(e),v:t[o],e:r})},Zo=function(e,t,n,s,r){return Ut(e)?e.call(t,n,s,r):jt(e)&&~e.indexOf("random(")?pa(e):e},Gg=wd+"repeat,repeatDelay,yoyo,repeatRefresh,yoyoEase,autoRevert",Wg={};In(Gg+",id,stagger,delay,duration,paused,scrollTrigger",function(i){return Wg[i]=1});var Gt=(function(i){cg(e,i);function e(n,s,r,o){var a;typeof s=="number"&&(r.duration=s,s=r,r=null),a=i.call(this,o?s:jo(s))||this;var l=a.vars,c=l.duration,u=l.delay,f=l.immediateRender,d=l.stagger,h=l.overwrite,x=l.keyframes,p=l.defaults,g=l.scrollTrigger,m=l.yoyoEase,_=s.parent||Pt,A=(an(n)||fg(n)?ls(n[0]):"length"in s)?[n]:oi(n),S,v,y,M,E,b,C,I;if(a._targets=A.length?Rd(A):da("GSAP target "+n+" not found. https://gsap.com",!Qn.nullTargetWarn)||[],a._ptLookup=[],a._overwrite=h,x||d||al(c)||al(u)){if(s=a.vars,S=a.timeline=new mn({data:"nested",defaults:p||{},targets:_&&_.data==="nested"?_.vars.targets:A}),S.kill(),S.parent=S._dp=Ki(a),S._start=0,d||al(c)||al(u)){if(M=A.length,C=d&&wg(d),Oi(d))for(E in d)~Gg.indexOf(E)&&(I||(I={}),I[E]=d[E]);for(v=0;v<M;v++)y=Ul(s,Wg),y.stagger=0,m&&(y.yoyoEase=m),I&&xo(y,I),b=A[v],y.duration=+Zo(c,Ki(a),v,b,A),y.delay=(+Zo(u,Ki(a),v,b,A)||0)-a._delay,!d&&M===1&&y.delay&&(a._delay=u=y.delay,a._start+=u,y.delay=0),S.to(b,y,C?C(v,b,A):0),S._ease=tt.none;S.duration()?c=u=0:a.timeline=0}else if(x){jo(Kn(S.vars.defaults,{ease:"none"})),S._ease=fr(x.ease||s.ease||"none");var F=0,U,O,k;if(an(x))x.forEach(function(z){return S.to(A,z,">")}),S.duration();else{y={};for(E in x)E==="ease"||E==="easeEach"||M1(E,x[E],y,x.easeEach);for(E in y)for(U=y[E].sort(function(z,V){return z.t-V.t}),F=0,v=0;v<U.length;v++)O=U[v],k={ease:O.e,duration:(O.t-(v?U[v-1].t:0))/100*c},k[E]=O.v,S.to(A,k,F),F+=k.duration;S.duration()<c&&S.to({},{duration:c-S.duration()})}}c||a.duration(c=S.duration())}else a.timeline=0;return h===!0&&!yd&&(Ts=Ki(a),Pt.killTweensOf(A),Ts=0),Ti(_,Ki(a),r),s.reversed&&a.reverse(),s.paused&&a.paused(!0),(f||!c&&!x&&a._start===Dt(_._time)&&Rn(f)&&t1(Ki(a))&&_.data!=="nested")&&(a._tTime=-_t,a.render(Math.max(0,-u)||0)),g&&Mg(Ki(a),g),a}var t=e.prototype;return t.render=function(s,r,o){var a=this._time,l=this._tDur,c=this._dur,u=s<0,f=s>l-_t&&!u?l:s<_t?0:s,d,h,x,p,g,m,_,A,S;if(!c)i1(this,s,r,o);else if(f!==this._tTime||!s||o||!this._initted&&this._tTime||this._startAt&&this._zTime<0!==u||this._lazy){if(d=f,A=this.timeline,this._repeat){if(p=c+this._rDelay,this._repeat<-1&&u)return this.totalTime(p*100+s,r,o);if(d=Dt(f%p),f===l?(x=this._repeat,d=c):(g=Dt(f/p),x=~~g,x&&x===g?(d=c,x--):d>c&&(d=c)),m=this._yoyo&&x&1,m&&(S=this._yEase,d=c-d),g=_o(this._tTime,p),d===a&&!o&&this._initted&&x===g)return this._tTime=f,this;x!==g&&(A&&this._yEase&&zg(A,m),this.vars.repeatRefresh&&!m&&!this._lock&&d!==p&&this._initted&&(this._lock=o=1,this.render(Dt(p*x),!0).invalidate()._lock=0))}if(!this._initted){if(Cg(this,u?s:d,o,r,f))return this._tTime=0,this;if(a!==this._time&&!(o&&this.vars.repeatRefresh&&x!==g))return this;if(c!==this._dur)return this.render(s,r,o)}if(this._tTime=f,this._time=d,!this._act&&this._ts&&(this._act=1,this._lazy=0),this.ratio=_=(S||this._ease)(d/c),this._from&&(this.ratio=_=1-_),!a&&f&&!r&&!g&&(Wn(this,"onStart"),this._tTime!==f))return this;for(h=this._pt;h;)h.r(_,h.d),h=h._next;A&&A.render(s<0?s:A._dur*A._ease(d/this._dur),r,o)||this._startAt&&(this._zTime=s),this._onUpdate&&!r&&(u&&xf(this,s,r,o),Wn(this,"onUpdate")),this._repeat&&x!==g&&this.vars.onRepeat&&!r&&this.parent&&Wn(this,"onRepeat"),(f===this._tDur||!f)&&this._tTime===f&&(u&&!this._onUpdate&&xf(this,s,!0,!0),(s||!c)&&(f===this._tDur&&this._ts>0||!f&&this._ts<0)&&Os(this,1),!r&&!(u&&!a)&&(f||a||m)&&(Wn(this,f===l?"onComplete":"onReverseComplete",!0),this._prom&&!(f<l&&this.timeScale()>0)&&this._prom()))}return this},t.targets=function(){return this._targets},t.invalidate=function(s){return(!s||!this.vars.runBackwards)&&(this._startAt=0),this._pt=this._op=this._onUpdate=this._lazy=this.ratio=0,this._ptLookup=[],this.timeline&&this.timeline.invalidate(s),i.prototype.invalidate.call(this,s)},t.resetTo=function(s,r,o,a,l){ma||Hn.wake(),this._ts||this.play();var c=Math.min(this._dur,(this._dp._time-this._start)*this._ts),u;return this._initted||Pd(this,c),u=this._ease(c/this._dur),y1(this,s,r,o,a,u,c,l)?this.resetTo(s,r,o,a,1):(sc(this,0),this.parent||yg(this._dp,this,"_first","_last",this._dp._sort?"_start":0),this.render(0))},t.kill=function(s,r){if(r===void 0&&(r="all"),!s&&(!r||r==="all"))return this._lazy=this._pt=0,this.parent?Uo(this):this.scrollTrigger&&this.scrollTrigger.kill(!!Jt),this;if(this.timeline){var o=this.timeline.totalDuration();return this.timeline.killTweensOf(s,r,Ts&&Ts.vars.overwrite!==!0)._first||Uo(this),this.parent&&o!==this.timeline.totalDuration()&&Ao(this,this._dur*this.timeline._tDur/o,0,1),this}var a=this._targets,l=s?oi(s):a,c=this._ptLookup,u=this._pt,f,d,h,x,p,g,m;if((!r||r==="all")&&JE(a,l))return r==="all"&&(this._pt=0),Uo(this);for(f=this._op=this._op||[],r!=="all"&&(jt(r)&&(p={},In(r,function(_){return p[_]=1}),r=p),r=b1(a,r)),m=a.length;m--;)if(~l.indexOf(a[m])){d=c[m],r==="all"?(f[m]=r,x=d,h={}):(h=f[m]=f[m]||{},x=r);for(p in x)g=d&&d[p],g&&((!("kill"in g.d)||g.d.kill(p)===!0)&&nc(this,g,"_pt"),delete d[p]),h!=="all"&&(h[p]=1)}return this._initted&&!this._pt&&u&&Uo(this),this},e.to=function(s,r){return new e(s,r,arguments[2])},e.from=function(s,r){return $o(1,arguments)},e.delayedCall=function(s,r,o,a){return new e(r,0,{immediateRender:!1,lazy:!1,overwrite:!1,delay:s,onComplete:r,onReverseComplete:r,onCompleteParams:o,onReverseCompleteParams:o,callbackScope:a})},e.fromTo=function(s,r,o){return $o(2,arguments)},e.set=function(s,r){return r.duration=0,r.repeatDelay||(r.repeat=0),new e(s,r)},e.killTweensOf=function(s,r,o){return Pt.killTweensOf(s,r,o)},e})(ga);Kn(Gt.prototype,{_targets:[],_lazy:0,_startAt:0,_op:0,_onInit:0});In("staggerTo,staggerFrom,staggerFromTo",function(i){Gt[i]=function(){var e=new mn,t=Af.call(arguments,0);return t.splice(i==="staggerFromTo"?5:4,0,0),e[i].apply(e,t)}});var Fd=function(e,t,n){return e[t]=n},Xg=function(e,t,n){return e[t](n)},C1=function(e,t,n,s){return e[t](s.fp,n)},T1=function(e,t,n){return e.setAttribute(t,n)},Ld=function(e,t){return Ut(e[t])?Xg:bd(e[t])&&e.setAttribute?T1:Fd},qg=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e6)/1e6,t)},E1=function(e,t){return t.set(t.t,t.p,!!(t.s+t.c*e),t)},Qg=function(e,t){var n=t._pt,s="";if(!e&&t.b)s=t.b;else if(e===1&&t.e)s=t.e;else{for(;n;)s=n.p+(n.m?n.m(n.s+n.c*e):Math.round((n.s+n.c*e)*1e4)/1e4)+s,n=n._next;s+=t.c}t.set(t.t,t.p,s,t)},Bd=function(e,t){for(var n=t._pt;n;)n.r(e,n.d),n=n._next},w1=function(e,t,n,s){for(var r=this._pt,o;r;)o=r._next,r.p===s&&r.modifier(e,t,n),r=o},R1=function(e){for(var t=this._pt,n,s;t;)s=t._next,t.p===e&&!t.op||t.op===e?nc(this,t,"_pt"):t.dep||(n=1),t=s;return!n},I1=function(e,t,n,s){s.mSet(e,t,s.m.call(s.tween,n,s.mt),s)},Yg=function(e){for(var t=e._pt,n,s,r,o;t;){for(n=t._next,s=r;s&&s.pr>t.pr;)s=s._next;(t._prev=s?s._prev:o)?t._prev._next=t:r=t,(t._next=s)?s._prev=t:o=t,t=n}e._pt=r},Dn=(function(){function i(t,n,s,r,o,a,l,c,u){this.t=n,this.s=r,this.c=o,this.p=s,this.r=a||qg,this.d=l||this,this.set=c||Fd,this.pr=u||0,this._next=t,t&&(t._prev=this)}var e=i.prototype;return e.modifier=function(n,s,r){this.mSet=this.mSet||this.set,this.set=I1,this.m=n,this.mt=r,this.tween=s},i})();In(wd+"parent,duration,ease,delay,overwrite,runBackwards,startAt,yoyo,immediateRender,repeat,repeatDelay,data,paused,reversed,lazy,callbackScope,stringFilter,id,yoyoEase,stagger,inherit,repeatRefresh,keyframes,autoRevert,scrollTrigger",function(i){return Ed[i]=1});Yn.TweenMax=Yn.TweenLite=Gt;Yn.TimelineLite=Yn.TimelineMax=mn;Pt=new mn({sortChildren:!1,defaults:go,autoRemoveChildren:!0,id:"root",smoothChildTiming:!0});Qn.stringFilter=Og;var dr=[],Sl={},D1=[],rm=0,P1=0,cu=function(e){return(Sl[e]||D1).map(function(t){return t()})},bf=function(){var e=Date.now(),t=[];e-rm>2&&(cu("matchMediaInit"),dr.forEach(function(n){var s=n.queries,r=n.conditions,o,a,l,c;for(a in s)o=yi.matchMedia(s[a]).matches,o&&(l=1),o!==r[a]&&(r[a]=o,c=1);c&&(n.revert(),l&&t.push(n))}),cu("matchMediaRevert"),t.forEach(function(n){return n.onMatch(n,function(s){return n.add(null,s)})}),rm=e,cu("matchMedia"))},Kg=(function(){function i(t,n){this.selector=n&&Sf(n),this.data=[],this._r=[],this.isReverted=!1,this.id=P1++,t&&this.add(t)}var e=i.prototype;return e.add=function(n,s,r){Ut(n)&&(r=s,s=n,n=Ut);var o=this,a=function(){var c=Tt,u=o.selector,f;return c&&c!==o&&c.data.push(o),r&&(o.selector=Sf(r)),Tt=o,f=s.apply(o,arguments),Ut(f)&&o._r.push(f),Tt=c,o.selector=u,o.isReverted=!1,f};return o.last=a,n===Ut?a(o,function(l){return o.add(null,l)}):n?o[n]=a:a},e.ignore=function(n){var s=Tt;Tt=null,n(this),Tt=s},e.getTweens=function(){var n=[];return this.data.forEach(function(s){return s instanceof i?n.push.apply(n,s.getTweens()):s instanceof Gt&&!(s.parent&&s.parent.data==="nested")&&n.push(s)}),n},e.clear=function(){this._r.length=this.data.length=0},e.kill=function(n,s){var r=this;if(n?(function(){for(var a=r.getTweens(),l=r.data.length,c;l--;)c=r.data[l],c.data==="isFlip"&&(c.revert(),c.getChildren(!0,!0,!1).forEach(function(u){return a.splice(a.indexOf(u),1)}));for(a.map(function(u){return{g:u._dur||u._delay||u._sat&&!u._sat.vars.immediateRender?u.globalTime(0):-1/0,t:u}}).sort(function(u,f){return f.g-u.g||-1/0}).forEach(function(u){return u.t.revert(n)}),l=r.data.length;l--;)c=r.data[l],c instanceof mn?c.data!=="nested"&&(c.scrollTrigger&&c.scrollTrigger.revert(),c.kill()):!(c instanceof Gt)&&c.revert&&c.revert(n);r._r.forEach(function(u){return u(n,r)}),r.isReverted=!0})():this.data.forEach(function(a){return a.kill&&a.kill()}),this.clear(),s)for(var o=dr.length;o--;)dr[o].id===this.id&&dr.splice(o,1)},e.revert=function(n){this.kill(n||{})},i})(),F1=(function(){function i(t){this.contexts=[],this.scope=t,Tt&&Tt.data.push(this)}var e=i.prototype;return e.add=function(n,s,r){Oi(n)||(n={matches:n});var o=new Kg(0,r||this.scope),a=o.conditions={},l,c,u;Tt&&!o.selector&&(o.selector=Tt.selector),this.contexts.push(o),s=o.add("onMatch",s),o.queries=n;for(c in n)c==="all"?u=1:(l=yi.matchMedia(n[c]),l&&(dr.indexOf(o)<0&&dr.push(o),(a[c]=l.matches)&&(u=1),l.addListener?l.addListener(bf):l.addEventListener("change",bf)));return u&&s(o,function(f){return o.add(null,f)}),this},e.revert=function(n){this.kill(n||{})},e.kill=function(n){this.contexts.forEach(function(s){return s.kill(n,!0)})},i})(),Nl={registerPlugin:function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];t.forEach(function(s){return Lg(s)})},timeline:function(e){return new mn(e)},getTweensOf:function(e,t){return Pt.getTweensOf(e,t)},getProperty:function(e,t,n,s){jt(e)&&(e=oi(e)[0]);var r=cr(e||{}).get,o=n?vg:Sg;return n==="native"&&(n=""),e&&(t?o((kn[t]&&kn[t].get||r)(e,t,n,s)):function(a,l,c){return o((kn[a]&&kn[a].get||r)(e,a,l,c))})},quickSetter:function(e,t,n){if(e=oi(e),e.length>1){var s=e.map(function(u){return Fn.quickSetter(u,t,n)}),r=s.length;return function(u){for(var f=r;f--;)s[f](u)}}e=e[0]||{};var o=kn[t],a=cr(e),l=a.harness&&(a.harness.aliases||{})[t]||t,c=o?function(u){var f=new o;Xr._pt=0,f.init(e,n?u+n:u,Xr,0,[e]),f.render(1,f),Xr._pt&&Bd(1,Xr)}:a.set(e,l);return o?c:function(u){return c(e,l,n?u+n:u,a,1)}},quickTo:function(e,t,n){var s,r=Fn.to(e,Kn((s={},s[t]="+=0.1",s.paused=!0,s.stagger=0,s),n||{})),o=function(l,c,u){return r.resetTo(t,l,c,u)};return o.tween=r,o},isTweening:function(e){return Pt.getTweensOf(e,!0).length>0},defaults:function(e){return e&&e.ease&&(e.ease=fr(e.ease,go.ease)),em(go,e||{})},config:function(e){return em(Qn,e||{})},registerEffect:function(e){var t=e.name,n=e.effect,s=e.plugins,r=e.defaults,o=e.extendTimeline;(s||"").split(",").forEach(function(a){return a&&!kn[a]&&!Yn[a]&&da(t+" effect requires "+a+" plugin.")}),ru[t]=function(a,l,c){return n(oi(a),Kn(l||{},r),c)},o&&(mn.prototype[t]=function(a,l,c){return this.add(ru[t](a,Oi(l)?l:(c=l)&&{},this),c)})},registerEase:function(e,t){tt[e]=fr(t)},parseEase:function(e,t){return arguments.length?fr(e,t):tt},getById:function(e){return Pt.getById(e)},exportRoot:function(e,t){e===void 0&&(e={});var n=new mn(e),s,r;for(n.smoothChildTiming=Rn(e.smoothChildTiming),Pt.remove(n),n._dp=0,n._time=n._tTime=Pt._time,s=Pt._first;s;)r=s._next,(t||!(!s._dur&&s instanceof Gt&&s.vars.onComplete===s._targets[0]))&&Ti(n,s,s._start-s._delay),s=r;return Ti(Pt,n,0),n},context:function(e,t){return e?new Kg(e,t):Tt},matchMedia:function(e){return new F1(e)},matchMediaRefresh:function(){return dr.forEach(function(e){var t=e.conditions,n,s;for(s in t)t[s]&&(t[s]=!1,n=1);n&&e.revert()})||bf()},addEventListener:function(e,t){var n=Sl[e]||(Sl[e]=[]);~n.indexOf(t)||n.push(t)},removeEventListener:function(e,t){var n=Sl[e],s=n&&n.indexOf(t);s>=0&&n.splice(s,1)},utils:{wrap:f1,wrapYoyo:d1,distribute:wg,random:Ig,snap:Rg,normalize:u1,getUnit:sn,clamp:o1,splitColor:Bg,toArray:oi,selector:Sf,mapRange:Pg,pipe:l1,unitize:c1,interpolate:h1,shuffle:Eg},install:mg,effects:ru,ticker:Hn,updateRoot:mn.updateRoot,plugins:kn,globalTimeline:Pt,core:{PropTween:Dn,globals:gg,Tween:Gt,Timeline:mn,Animation:ga,getCache:cr,_removeLinkedListItem:nc,reverting:function(){return Jt},context:function(e){return e&&Tt&&(Tt.data.push(e),e._ctx=Tt),Tt},suppressOverwrites:function(e){return yd=e}}};In("to,from,fromTo,delayedCall,set,killTweensOf",function(i){return Nl[i]=Gt[i]});Hn.add(mn.updateRoot);Xr=Nl.to({},{duration:0});var L1=function(e,t){for(var n=e._pt;n&&n.p!==t&&n.op!==t&&n.fp!==t;)n=n._next;return n},B1=function(e,t){var n=e._targets,s,r,o;for(s in t)for(r=n.length;r--;)o=e._ptLookup[r][s],o&&(o=o.d)&&(o._pt&&(o=L1(o,s)),o&&o.modifier&&o.modifier(t[s],e,n[r],s))},uu=function(e,t){return{name:e,headless:1,rawVars:1,init:function(s,r,o){o._onInit=function(a){var l,c;if(jt(r)&&(l={},In(r,function(u){return l[u]=1}),r=l),t){l={};for(c in r)l[c]=t(r[c]);r=l}B1(a,r)}}}},Fn=Nl.registerPlugin({name:"attr",init:function(e,t,n,s,r){var o,a,l;this.tween=n;for(o in t)l=e.getAttribute(o)||"",a=this.add(e,"setAttribute",(l||0)+"",t[o],s,r,0,0,o),a.op=o,a.b=l,this._props.push(o)},render:function(e,t){for(var n=t._pt;n;)Jt?n.set(n.t,n.p,n.b,n):n.r(e,n.d),n=n._next}},{name:"endArray",headless:1,init:function(e,t){for(var n=t.length;n--;)this.add(e,n,e[n]||0,t[n],0,0,0,0,0,1)}},uu("roundProps",vf),uu("modifiers"),uu("snap",Rg))||Nl;Gt.version=mn.version=Fn.version="3.14.2";pg=1;Md()&&So();tt.Power0;tt.Power1;tt.Power2;tt.Power3;tt.Power4;tt.Linear;tt.Quad;tt.Cubic;tt.Quart;tt.Quint;tt.Strong;tt.Elastic;tt.Back;tt.SteppedEase;tt.Bounce;tt.Sine;tt.Expo;tt.Circ;var om,Es,io,Ud,or,am,Od,U1=function(){return typeof window<"u"},cs={},tr=180/Math.PI,so=Math.PI/180,Nr=Math.atan2,lm=1e8,Nd=/([A-Z])/g,O1=/(left|right|width|margin|padding|x)/i,N1=/[\s,\(]\S/,Ii={autoAlpha:"opacity,visibility",scale:"scaleX,scaleY",alpha:"opacity"},Mf=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},z1=function(e,t){return t.set(t.t,t.p,e===1?t.e:Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},k1=function(e,t){return t.set(t.t,t.p,e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},H1=function(e,t){return t.set(t.t,t.p,e===1?t.e:e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},V1=function(e,t){var n=t.s+t.c*e;t.set(t.t,t.p,~~(n+(n<0?-.5:.5))+t.u,t)},jg=function(e,t){return t.set(t.t,t.p,e?t.e:t.b,t)},$g=function(e,t){return t.set(t.t,t.p,e!==1?t.b:t.e,t)},G1=function(e,t,n){return e.style[t]=n},W1=function(e,t,n){return e.style.setProperty(t,n)},X1=function(e,t,n){return e._gsap[t]=n},q1=function(e,t,n){return e._gsap.scaleX=e._gsap.scaleY=n},Q1=function(e,t,n,s,r){var o=e._gsap;o.scaleX=o.scaleY=n,o.renderTransform(r,o)},Y1=function(e,t,n,s,r){var o=e._gsap;o[t]=n,o.renderTransform(r,o)},Ft="transform",Pn=Ft+"Origin",K1=function i(e,t){var n=this,s=this.target,r=s.style,o=s._gsap;if(e in cs&&r){if(this.tfm=this.tfm||{},e!=="transform")e=Ii[e]||e,~e.indexOf(",")?e.split(",").forEach(function(a){return n.tfm[a]=Zi(s,a)}):this.tfm[e]=o.x?o[e]:Zi(s,e),e===Pn&&(this.tfm.zOrigin=o.zOrigin);else return Ii.transform.split(",").forEach(function(a){return i.call(n,a,t)});if(this.props.indexOf(Ft)>=0)return;o.svg&&(this.svgo=s.getAttribute("data-svg-origin"),this.props.push(Pn,t,"")),e=Ft}(r||t)&&this.props.push(e,t,r[e])},Zg=function(e){e.translate&&(e.removeProperty("translate"),e.removeProperty("scale"),e.removeProperty("rotate"))},j1=function(){var e=this.props,t=this.target,n=t.style,s=t._gsap,r,o;for(r=0;r<e.length;r+=3)e[r+1]?e[r+1]===2?t[e[r]](e[r+2]):t[e[r]]=e[r+2]:e[r+2]?n[e[r]]=e[r+2]:n.removeProperty(e[r].substr(0,2)==="--"?e[r]:e[r].replace(Nd,"-$1").toLowerCase());if(this.tfm){for(o in this.tfm)s[o]=this.tfm[o];s.svg&&(s.renderTransform(),t.setAttribute("data-svg-origin",this.svgo||"")),r=Od(),(!r||!r.isStart)&&!n[Ft]&&(Zg(n),s.zOrigin&&n[Pn]&&(n[Pn]+=" "+s.zOrigin+"px",s.zOrigin=0,s.renderTransform()),s.uncache=1)}},Jg=function(e,t){var n={target:e,props:[],revert:j1,save:K1};return e._gsap||Fn.core.getCache(e),t&&e.style&&e.nodeType&&t.split(",").forEach(function(s){return n.save(s)}),n},ex,Cf=function(e,t){var n=Es.createElementNS?Es.createElementNS((t||"http://www.w3.org/1999/xhtml").replace(/^https/,"http"),e):Es.createElement(e);return n&&n.style?n:Es.createElement(e)},Xn=function i(e,t,n){var s=getComputedStyle(e);return s[t]||s.getPropertyValue(t.replace(Nd,"-$1").toLowerCase())||s.getPropertyValue(t)||!n&&i(e,vo(t)||t,1)||""},cm="O,Moz,ms,Ms,Webkit".split(","),vo=function(e,t,n){var s=t||or,r=s.style,o=5;if(e in r&&!n)return e;for(e=e.charAt(0).toUpperCase()+e.substr(1);o--&&!(cm[o]+e in r););return o<0?null:(o===3?"ms":o>=0?cm[o]:"")+e},Tf=function(){U1()&&window.document&&(om=window,Es=om.document,io=Es.documentElement,or=Cf("div")||{style:{}},Cf("div"),Ft=vo(Ft),Pn=Ft+"Origin",or.style.cssText="border-width:0;line-height:0;position:absolute;padding:0",ex=!!vo("perspective"),Od=Fn.core.reverting,Ud=1)},um=function(e){var t=e.ownerSVGElement,n=Cf("svg",t&&t.getAttribute("xmlns")||"http://www.w3.org/2000/svg"),s=e.cloneNode(!0),r;s.style.display="block",n.appendChild(s),io.appendChild(n);try{r=s.getBBox()}catch{}return n.removeChild(s),io.removeChild(n),r},fm=function(e,t){for(var n=t.length;n--;)if(e.hasAttribute(t[n]))return e.getAttribute(t[n])},tx=function(e){var t,n;try{t=e.getBBox()}catch{t=um(e),n=1}return t&&(t.width||t.height)||n||(t=um(e)),t&&!t.width&&!t.x&&!t.y?{x:+fm(e,["x","cx","x1"])||0,y:+fm(e,["y","cy","y1"])||0,width:0,height:0}:t},nx=function(e){return!!(e.getCTM&&(!e.parentNode||e.ownerSVGElement)&&tx(e))},Ns=function(e,t){if(t){var n=e.style,s;t in cs&&t!==Pn&&(t=Ft),n.removeProperty?(s=t.substr(0,2),(s==="ms"||t.substr(0,6)==="webkit")&&(t="-"+t),n.removeProperty(s==="--"?t:t.replace(Nd,"-$1").toLowerCase())):n.removeAttribute(t)}},ws=function(e,t,n,s,r,o){var a=new Dn(e._pt,t,n,0,1,o?$g:jg);return e._pt=a,a.b=s,a.e=r,e._props.push(n),a},dm={deg:1,rad:1,turn:1},$1={grid:1,flex:1},zs=function i(e,t,n,s){var r=parseFloat(n)||0,o=(n+"").trim().substr((r+"").length)||"px",a=or.style,l=O1.test(t),c=e.tagName.toLowerCase()==="svg",u=(c?"client":"offset")+(l?"Width":"Height"),f=100,d=s==="px",h=s==="%",x,p,g,m;if(s===o||!r||dm[s]||dm[o])return r;if(o!=="px"&&!d&&(r=i(e,t,n,"px")),m=e.getCTM&&nx(e),(h||o==="%")&&(cs[t]||~t.indexOf("adius")))return x=m?e.getBBox()[l?"width":"height"]:e[u],Ot(h?r/x*f:r/100*x);if(a[l?"width":"height"]=f+(d?o:s),p=s!=="rem"&&~t.indexOf("adius")||s==="em"&&e.appendChild&&!c?e:e.parentNode,m&&(p=(e.ownerSVGElement||{}).parentNode),(!p||p===Es||!p.appendChild)&&(p=Es.body),g=p._gsap,g&&h&&g.width&&l&&g.time===Hn.time&&!g.uncache)return Ot(r/g.width*f);if(h&&(t==="height"||t==="width")){var _=e.style[t];e.style[t]=f+s,x=e[u],_?e.style[t]=_:Ns(e,t)}else(h||o==="%")&&!$1[Xn(p,"display")]&&(a.position=Xn(e,"position")),p===e&&(a.position="static"),p.appendChild(or),x=or[u],p.removeChild(or),a.position="absolute";return l&&h&&(g=cr(p),g.time=Hn.time,g.width=p[u]),Ot(d?x*r/f:x&&r?f/x*r:0)},Zi=function(e,t,n,s){var r;return Ud||Tf(),t in Ii&&t!=="transform"&&(t=Ii[t],~t.indexOf(",")&&(t=t.split(",")[0])),cs[t]&&t!=="transform"?(r=_a(e,s),r=t!=="transformOrigin"?r[t]:r.svg?r.origin:kl(Xn(e,Pn))+" "+r.zOrigin+"px"):(r=e.style[t],(!r||r==="auto"||s||~(r+"").indexOf("calc("))&&(r=zl[t]&&zl[t](e,t,n)||Xn(e,t)||_g(e,t)||(t==="opacity"?1:0))),n&&!~(r+"").trim().indexOf(" ")?zs(e,t,r,n)+n:r},Z1=function(e,t,n,s){if(!n||n==="none"){var r=vo(t,e,1),o=r&&Xn(e,r,1);o&&o!==n?(t=r,n=o):t==="borderColor"&&(n=Xn(e,"borderTopColor"))}var a=new Dn(this._pt,e.style,t,0,1,Qg),l=0,c=0,u,f,d,h,x,p,g,m,_,A,S,v;if(a.b=n,a.e=s,n+="",s+="",s.substring(0,6)==="var(--"&&(s=Xn(e,s.substring(4,s.indexOf(")")))),s==="auto"&&(p=e.style[t],e.style[t]=s,s=Xn(e,t)||s,p?e.style[t]=p:Ns(e,t)),u=[n,s],Og(u),n=u[0],s=u[1],d=n.match(Wr)||[],v=s.match(Wr)||[],v.length){for(;f=Wr.exec(s);)g=f[0],_=s.substring(l,f.index),x?x=(x+1)%5:(_.substr(-5)==="rgba("||_.substr(-5)==="hsla(")&&(x=1),g!==(p=d[c++]||"")&&(h=parseFloat(p)||0,S=p.substr((h+"").length),g.charAt(1)==="="&&(g=no(h,g)+S),m=parseFloat(g),A=g.substr((m+"").length),l=Wr.lastIndex-A.length,A||(A=A||Qn.units[t]||S,l===s.length&&(s+=A,a.e+=A)),S!==A&&(h=zs(e,t,p,A)||0),a._pt={_next:a._pt,p:_||c===1?_:",",s:h,c:m-h,m:x&&x<4||t==="zIndex"?Math.round:0});a.c=l<s.length?s.substring(l,s.length):""}else a.r=t==="display"&&s==="none"?$g:jg;return hg.test(s)&&(a.e=0),this._pt=a,a},hm={top:"0%",bottom:"100%",left:"0%",right:"100%",center:"50%"},J1=function(e){var t=e.split(" "),n=t[0],s=t[1]||"50%";return(n==="top"||n==="bottom"||s==="left"||s==="right")&&(e=n,n=s,s=e),t[0]=hm[n]||n,t[1]=hm[s]||s,t.join(" ")},ew=function(e,t){if(t.tween&&t.tween._time===t.tween._dur){var n=t.t,s=n.style,r=t.u,o=n._gsap,a,l,c;if(r==="all"||r===!0)s.cssText="",l=1;else for(r=r.split(","),c=r.length;--c>-1;)a=r[c],cs[a]&&(l=1,a=a==="transformOrigin"?Pn:Ft),Ns(n,a);l&&(Ns(n,Ft),o&&(o.svg&&n.removeAttribute("transform"),s.scale=s.rotate=s.translate="none",_a(n,1),o.uncache=1,Zg(s)))}},zl={clearProps:function(e,t,n,s,r){if(r.data!=="isFromStart"){var o=e._pt=new Dn(e._pt,t,n,0,0,ew);return o.u=s,o.pr=-10,o.tween=r,e._props.push(n),1}}},xa=[1,0,0,1,0,0],ix={},sx=function(e){return e==="matrix(1, 0, 0, 1, 0, 0)"||e==="none"||!e},pm=function(e){var t=Xn(e,Ft);return sx(t)?xa:t.substr(7).match(dg).map(Ot)},zd=function(e,t){var n=e._gsap||cr(e),s=e.style,r=pm(e),o,a,l,c;return n.svg&&e.getAttribute("transform")?(l=e.transform.baseVal.consolidate().matrix,r=[l.a,l.b,l.c,l.d,l.e,l.f],r.join(",")==="1,0,0,1,0,0"?xa:r):(r===xa&&!e.offsetParent&&e!==io&&!n.svg&&(l=s.display,s.display="block",o=e.parentNode,(!o||!e.offsetParent&&!e.getBoundingClientRect().width)&&(c=1,a=e.nextElementSibling,io.appendChild(e)),r=pm(e),l?s.display=l:Ns(e,"display"),c&&(a?o.insertBefore(e,a):o?o.appendChild(e):io.removeChild(e))),t&&r.length>6?[r[0],r[1],r[4],r[5],r[12],r[13]]:r)},Ef=function(e,t,n,s,r,o){var a=e._gsap,l=r||zd(e,!0),c=a.xOrigin||0,u=a.yOrigin||0,f=a.xOffset||0,d=a.yOffset||0,h=l[0],x=l[1],p=l[2],g=l[3],m=l[4],_=l[5],A=t.split(" "),S=parseFloat(A[0])||0,v=parseFloat(A[1])||0,y,M,E,b;n?l!==xa&&(M=h*g-x*p)&&(E=S*(g/M)+v*(-p/M)+(p*_-g*m)/M,b=S*(-x/M)+v*(h/M)-(h*_-x*m)/M,S=E,v=b):(y=tx(e),S=y.x+(~A[0].indexOf("%")?S/100*y.width:S),v=y.y+(~(A[1]||A[0]).indexOf("%")?v/100*y.height:v)),s||s!==!1&&a.smooth?(m=S-c,_=v-u,a.xOffset=f+(m*h+_*p)-m,a.yOffset=d+(m*x+_*g)-_):a.xOffset=a.yOffset=0,a.xOrigin=S,a.yOrigin=v,a.smooth=!!s,a.origin=t,a.originIsAbsolute=!!n,e.style[Pn]="0px 0px",o&&(ws(o,a,"xOrigin",c,S),ws(o,a,"yOrigin",u,v),ws(o,a,"xOffset",f,a.xOffset),ws(o,a,"yOffset",d,a.yOffset)),e.setAttribute("data-svg-origin",S+" "+v)},_a=function(e,t){var n=e._gsap||new Hg(e);if("x"in n&&!t&&!n.uncache)return n;var s=e.style,r=n.scaleX<0,o="px",a="deg",l=getComputedStyle(e),c=Xn(e,Pn)||"0",u,f,d,h,x,p,g,m,_,A,S,v,y,M,E,b,C,I,F,U,O,k,z,V,H,$,oe,Se,we,Le,fe,re;return u=f=d=p=g=m=_=A=S=0,h=x=1,n.svg=!!(e.getCTM&&nx(e)),l.translate&&((l.translate!=="none"||l.scale!=="none"||l.rotate!=="none")&&(s[Ft]=(l.translate!=="none"?"translate3d("+(l.translate+" 0 0").split(" ").slice(0,3).join(", ")+") ":"")+(l.rotate!=="none"?"rotate("+l.rotate+") ":"")+(l.scale!=="none"?"scale("+l.scale.split(" ").join(",")+") ":"")+(l[Ft]!=="none"?l[Ft]:"")),s.scale=s.rotate=s.translate="none"),M=zd(e,n.svg),n.svg&&(n.uncache?(H=e.getBBox(),c=n.xOrigin-H.x+"px "+(n.yOrigin-H.y)+"px",V=""):V=!t&&e.getAttribute("data-svg-origin"),Ef(e,V||c,!!V||n.originIsAbsolute,n.smooth!==!1,M)),v=n.xOrigin||0,y=n.yOrigin||0,M!==xa&&(I=M[0],F=M[1],U=M[2],O=M[3],u=k=M[4],f=z=M[5],M.length===6?(h=Math.sqrt(I*I+F*F),x=Math.sqrt(O*O+U*U),p=I||F?Nr(F,I)*tr:0,_=U||O?Nr(U,O)*tr+p:0,_&&(x*=Math.abs(Math.cos(_*so))),n.svg&&(u-=v-(v*I+y*U),f-=y-(v*F+y*O))):(re=M[6],Le=M[7],oe=M[8],Se=M[9],we=M[10],fe=M[11],u=M[12],f=M[13],d=M[14],E=Nr(re,we),g=E*tr,E&&(b=Math.cos(-E),C=Math.sin(-E),V=k*b+oe*C,H=z*b+Se*C,$=re*b+we*C,oe=k*-C+oe*b,Se=z*-C+Se*b,we=re*-C+we*b,fe=Le*-C+fe*b,k=V,z=H,re=$),E=Nr(-U,we),m=E*tr,E&&(b=Math.cos(-E),C=Math.sin(-E),V=I*b-oe*C,H=F*b-Se*C,$=U*b-we*C,fe=O*C+fe*b,I=V,F=H,U=$),E=Nr(F,I),p=E*tr,E&&(b=Math.cos(E),C=Math.sin(E),V=I*b+F*C,H=k*b+z*C,F=F*b-I*C,z=z*b-k*C,I=V,k=H),g&&Math.abs(g)+Math.abs(p)>359.9&&(g=p=0,m=180-m),h=Ot(Math.sqrt(I*I+F*F+U*U)),x=Ot(Math.sqrt(z*z+re*re)),E=Nr(k,z),_=Math.abs(E)>2e-4?E*tr:0,S=fe?1/(fe<0?-fe:fe):0),n.svg&&(V=e.getAttribute("transform"),n.forceCSS=e.setAttribute("transform","")||!sx(Xn(e,Ft)),V&&e.setAttribute("transform",V))),Math.abs(_)>90&&Math.abs(_)<270&&(r?(h*=-1,_+=p<=0?180:-180,p+=p<=0?180:-180):(x*=-1,_+=_<=0?180:-180)),t=t||n.uncache,n.x=u-((n.xPercent=u&&(!t&&n.xPercent||(Math.round(e.offsetWidth/2)===Math.round(-u)?-50:0)))?e.offsetWidth*n.xPercent/100:0)+o,n.y=f-((n.yPercent=f&&(!t&&n.yPercent||(Math.round(e.offsetHeight/2)===Math.round(-f)?-50:0)))?e.offsetHeight*n.yPercent/100:0)+o,n.z=d+o,n.scaleX=Ot(h),n.scaleY=Ot(x),n.rotation=Ot(p)+a,n.rotationX=Ot(g)+a,n.rotationY=Ot(m)+a,n.skewX=_+a,n.skewY=A+a,n.transformPerspective=S+o,(n.zOrigin=parseFloat(c.split(" ")[2])||!t&&n.zOrigin||0)&&(s[Pn]=kl(c)),n.xOffset=n.yOffset=0,n.force3D=Qn.force3D,n.renderTransform=n.svg?nw:ex?rx:tw,n.uncache=0,n},kl=function(e){return(e=e.split(" "))[0]+" "+e[1]},fu=function(e,t,n){var s=sn(t);return Ot(parseFloat(t)+parseFloat(zs(e,"x",n+"px",s)))+s},tw=function(e,t){t.z="0px",t.rotationY=t.rotationX="0deg",t.force3D=0,rx(e,t)},Zs="0deg",Fo="0px",Js=") ",rx=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.z,c=n.rotation,u=n.rotationY,f=n.rotationX,d=n.skewX,h=n.skewY,x=n.scaleX,p=n.scaleY,g=n.transformPerspective,m=n.force3D,_=n.target,A=n.zOrigin,S="",v=m==="auto"&&e&&e!==1||m===!0;if(A&&(f!==Zs||u!==Zs)){var y=parseFloat(u)*so,M=Math.sin(y),E=Math.cos(y),b;y=parseFloat(f)*so,b=Math.cos(y),o=fu(_,o,M*b*-A),a=fu(_,a,-Math.sin(y)*-A),l=fu(_,l,E*b*-A+A)}g!==Fo&&(S+="perspective("+g+Js),(s||r)&&(S+="translate("+s+"%, "+r+"%) "),(v||o!==Fo||a!==Fo||l!==Fo)&&(S+=l!==Fo||v?"translate3d("+o+", "+a+", "+l+") ":"translate("+o+", "+a+Js),c!==Zs&&(S+="rotate("+c+Js),u!==Zs&&(S+="rotateY("+u+Js),f!==Zs&&(S+="rotateX("+f+Js),(d!==Zs||h!==Zs)&&(S+="skew("+d+", "+h+Js),(x!==1||p!==1)&&(S+="scale("+x+", "+p+Js),_.style[Ft]=S||"translate(0, 0)"},nw=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.rotation,c=n.skewX,u=n.skewY,f=n.scaleX,d=n.scaleY,h=n.target,x=n.xOrigin,p=n.yOrigin,g=n.xOffset,m=n.yOffset,_=n.forceCSS,A=parseFloat(o),S=parseFloat(a),v,y,M,E,b;l=parseFloat(l),c=parseFloat(c),u=parseFloat(u),u&&(u=parseFloat(u),c+=u,l+=u),l||c?(l*=so,c*=so,v=Math.cos(l)*f,y=Math.sin(l)*f,M=Math.sin(l-c)*-d,E=Math.cos(l-c)*d,c&&(u*=so,b=Math.tan(c-u),b=Math.sqrt(1+b*b),M*=b,E*=b,u&&(b=Math.tan(u),b=Math.sqrt(1+b*b),v*=b,y*=b)),v=Ot(v),y=Ot(y),M=Ot(M),E=Ot(E)):(v=f,E=d,y=M=0),(A&&!~(o+"").indexOf("px")||S&&!~(a+"").indexOf("px"))&&(A=zs(h,"x",o,"px"),S=zs(h,"y",a,"px")),(x||p||g||m)&&(A=Ot(A+x-(x*v+p*M)+g),S=Ot(S+p-(x*y+p*E)+m)),(s||r)&&(b=h.getBBox(),A=Ot(A+s/100*b.width),S=Ot(S+r/100*b.height)),b="matrix("+v+","+y+","+M+","+E+","+A+","+S+")",h.setAttribute("transform",b),_&&(h.style[Ft]=b)},iw=function(e,t,n,s,r){var o=360,a=jt(r),l=parseFloat(r)*(a&&~r.indexOf("rad")?tr:1),c=l-s,u=s+c+"deg",f,d;return a&&(f=r.split("_")[1],f==="short"&&(c%=o,c!==c%(o/2)&&(c+=c<0?o:-o)),f==="cw"&&c<0?c=(c+o*lm)%o-~~(c/o)*o:f==="ccw"&&c>0&&(c=(c-o*lm)%o-~~(c/o)*o)),e._pt=d=new Dn(e._pt,t,n,s,c,z1),d.e=u,d.u="deg",e._props.push(n),d},mm=function(e,t){for(var n in t)e[n]=t[n];return e},sw=function(e,t,n){var s=mm({},n._gsap),r="perspective,force3D,transformOrigin,svgOrigin",o=n.style,a,l,c,u,f,d,h,x;s.svg?(c=n.getAttribute("transform"),n.setAttribute("transform",""),o[Ft]=t,a=_a(n,1),Ns(n,Ft),n.setAttribute("transform",c)):(c=getComputedStyle(n)[Ft],o[Ft]=t,a=_a(n,1),o[Ft]=c);for(l in cs)c=s[l],u=a[l],c!==u&&r.indexOf(l)<0&&(h=sn(c),x=sn(u),f=h!==x?zs(n,l,c,x):parseFloat(c),d=parseFloat(u),e._pt=new Dn(e._pt,a,l,f,d-f,Mf),e._pt.u=x||0,e._props.push(l));mm(a,s)};In("padding,margin,Width,Radius",function(i,e){var t="Top",n="Right",s="Bottom",r="Left",o=(e<3?[t,n,s,r]:[t+r,t+n,s+n,s+r]).map(function(a){return e<2?i+a:"border"+a+i});zl[e>1?"border"+i:i]=function(a,l,c,u,f){var d,h;if(arguments.length<4)return d=o.map(function(x){return Zi(a,x,c)}),h=d.join(" "),h.split(d[0]).length===5?d[0]:h;d=(u+"").split(" "),h={},o.forEach(function(x,p){return h[x]=d[p]=d[p]||d[(p-1)/2|0]}),a.init(l,h,f)}});var ox={name:"css",register:Tf,targetTest:function(e){return e.style&&e.nodeType},init:function(e,t,n,s,r){var o=this._props,a=e.style,l=n.vars.startAt,c,u,f,d,h,x,p,g,m,_,A,S,v,y,M,E,b;Ud||Tf(),this.styles=this.styles||Jg(e),E=this.styles.props,this.tween=n;for(p in t)if(p!=="autoRound"&&(u=t[p],!(kn[p]&&Vg(p,t,n,s,e,r)))){if(h=typeof u,x=zl[p],h==="function"&&(u=u.call(n,s,e,r),h=typeof u),h==="string"&&~u.indexOf("random(")&&(u=pa(u)),x)x(this,e,p,u,n)&&(M=1);else if(p.substr(0,2)==="--")c=(getComputedStyle(e).getPropertyValue(p)+"").trim(),u+="",Fs.lastIndex=0,Fs.test(c)||(g=sn(c),m=sn(u),m?g!==m&&(c=zs(e,p,c,m)+m):g&&(u+=g)),this.add(a,"setProperty",c,u,s,r,0,0,p),o.push(p),E.push(p,0,a[p]);else if(h!=="undefined"){if(l&&p in l?(c=typeof l[p]=="function"?l[p].call(n,s,e,r):l[p],jt(c)&&~c.indexOf("random(")&&(c=pa(c)),sn(c+"")||c==="auto"||(c+=Qn.units[p]||sn(Zi(e,p))||""),(c+"").charAt(1)==="="&&(c=Zi(e,p))):c=Zi(e,p),d=parseFloat(c),_=h==="string"&&u.charAt(1)==="="&&u.substr(0,2),_&&(u=u.substr(2)),f=parseFloat(u),p in Ii&&(p==="autoAlpha"&&(d===1&&Zi(e,"visibility")==="hidden"&&f&&(d=0),E.push("visibility",0,a.visibility),ws(this,a,"visibility",d?"inherit":"hidden",f?"inherit":"hidden",!f)),p!=="scale"&&p!=="transform"&&(p=Ii[p],~p.indexOf(",")&&(p=p.split(",")[0]))),A=p in cs,A){if(this.styles.save(p),b=u,h==="string"&&u.substring(0,6)==="var(--"){if(u=Xn(e,u.substring(4,u.indexOf(")"))),u.substring(0,5)==="calc("){var C=e.style.perspective;e.style.perspective=u,u=Xn(e,"perspective"),C?e.style.perspective=C:Ns(e,"perspective")}f=parseFloat(u)}if(S||(v=e._gsap,v.renderTransform&&!t.parseTransform||_a(e,t.parseTransform),y=t.smoothOrigin!==!1&&v.smooth,S=this._pt=new Dn(this._pt,a,Ft,0,1,v.renderTransform,v,0,-1),S.dep=1),p==="scale")this._pt=new Dn(this._pt,v,"scaleY",v.scaleY,(_?no(v.scaleY,_+f):f)-v.scaleY||0,Mf),this._pt.u=0,o.push("scaleY",p),p+="X";else if(p==="transformOrigin"){E.push(Pn,0,a[Pn]),u=J1(u),v.svg?Ef(e,u,0,y,0,this):(m=parseFloat(u.split(" ")[2])||0,m!==v.zOrigin&&ws(this,v,"zOrigin",v.zOrigin,m),ws(this,a,p,kl(c),kl(u)));continue}else if(p==="svgOrigin"){Ef(e,u,1,y,0,this);continue}else if(p in ix){iw(this,v,p,d,_?no(d,_+u):u);continue}else if(p==="smoothOrigin"){ws(this,v,"smooth",v.smooth,u);continue}else if(p==="force3D"){v[p]=u;continue}else if(p==="transform"){sw(this,u,e);continue}}else p in a||(p=vo(p)||p);if(A||(f||f===0)&&(d||d===0)&&!N1.test(u)&&p in a)g=(c+"").substr((d+"").length),f||(f=0),m=sn(u)||(p in Qn.units?Qn.units[p]:g),g!==m&&(d=zs(e,p,c,m)),this._pt=new Dn(this._pt,A?v:a,p,d,(_?no(d,_+f):f)-d,!A&&(m==="px"||p==="zIndex")&&t.autoRound!==!1?V1:Mf),this._pt.u=m||0,A&&b!==u?(this._pt.b=c,this._pt.e=b,this._pt.r=H1):g!==m&&m!=="%"&&(this._pt.b=c,this._pt.r=k1);else if(p in a)Z1.call(this,e,p,c,_?_+u:u);else if(p in e)this.add(e,p,c||e[p],_?_+u:u,s,r);else if(p!=="parseTransform"){Td(p,u);continue}A||(p in a?E.push(p,0,a[p]):typeof e[p]=="function"?E.push(p,2,e[p]()):E.push(p,1,c||e[p])),o.push(p)}}M&&Yg(this)},render:function(e,t){if(t.tween._time||!Od())for(var n=t._pt;n;)n.r(e,n.d),n=n._next;else t.styles.revert()},get:Zi,aliases:Ii,getSetter:function(e,t,n){var s=Ii[t];return s&&s.indexOf(",")<0&&(t=s),t in cs&&t!==Pn&&(e._gsap.x||Zi(e,"x"))?n&&am===n?t==="scale"?q1:X1:(am=n||{})&&(t==="scale"?Q1:Y1):e.style&&!bd(e.style[t])?G1:~t.indexOf("-")?W1:Ld(e,t)},core:{_removeProperty:Ns,_getMatrix:zd}};Fn.utils.checkPrefix=vo;Fn.core.getStyleSaver=Jg;(function(i,e,t,n){var s=In(i+","+e+","+t,function(r){cs[r]=1});In(e,function(r){Qn.units[r]="deg",ix[r]=1}),Ii[s[13]]=i+","+e,In(n,function(r){var o=r.split(":");Ii[o[1]]=s[o[0]]})})("x,y,z,scale,scaleX,scaleY,xPercent,yPercent","rotation,rotationX,rotationY,skewX,skewY","transform,transformOrigin,svgOrigin,force3D,smoothOrigin,transformPerspective","0:translateX,1:translateY,2:translateZ,8:rotate,8:rotationZ,8:rotateZ,9:rotateX,10:rotateY");In("x,y,z,top,right,bottom,left,width,height,fontSize,padding,margin,perspective",function(i){Qn.units[i]="px"});Fn.registerPlugin(ox);var Vr=Fn.registerPlugin(ox)||Fn;Vr.core.Tween;const rw=(i,e)=>{const t=i.__vccOpts||i;for(const[n,s]of e)t[n]=s;return t},ow={key:0,class:"loading-overlay"},aw={key:1,class:"error-overlay"},lw={class:"error-msg"},cw={class:"controls-ui"},uw={class:"search-panel"},fw=["onClick"],dw=["src"],hw={key:1,class:"camera-tag-overlay"},pw={class:"camera-title-mini"},mw={class:"camera-tag-text"},gw={key:2},xw=["src"],_w={key:0,class:"ref-info"},Aw={class:"info-tag"},Sw={class:"info-tag"},vw={class:"info-tag"},yw={__name:"GaussianViewer",setup(i){const e=yn(null),t=yn(!1),n=yn(!1),s=yn(!1),r=yn(!1),o=yn([]),a=yn(""),l=yn(""),c=yn({}),u=yn({x:0,y:0,z:0}),f=yn({x:0,y:0,z:0}),d=yn(""),h=y0(()=>{if(!a.value.trim())return o.value.filter(re=>re.tag);const fe=a.value.trim().toLowerCase();return o.value.filter(re=>re.tag&&re.tag.toLowerCase().includes(fe))}),x=()=>{h.value.length>0?E(h.value[0]):alert("场景中没有找到符合该描述的视角哦~")};let p,g;const m=yn({x:0,y:0}),_=()=>{if(!p||!p.camera)return;const fe=new _i().setFromQuaternion(p.camera.quaternion,"YXZ");u.value={x:(fe.x*180/Math.PI).toFixed(1),y:(fe.y*180/Math.PI).toFixed(1),z:(fe.z*180/Math.PI).toFixed(1)}},A={FLY_IN:0,DIFFUSION:1,COLORING:2,FINISHED:3},S={isLoaded:!1,lastFrameTime:0,phase:A.FLY_IN,flyDuration:1.5,diffusionDuration:1,colorDuration:4},v={uTime:{value:0},uCenter:{value:new B(0,0,0)},uGeoRadius:{value:0},uColorRadius:{value:0},uMaxRadius:{value:50},uParticleProgress:{value:0}},y=fe=>{if(!p)return;const re=fe.getSplatCount();fe.updateMatrixWorld();let X=1/0,ee=1/0,pe=1/0,be=-1/0,xe=-1/0,Ce=-1/0;const P=new B,L=Math.max(1,Math.floor(re/1e3));for(let le=0;le<re;le+=L)fe.getSplatCenter(le,P),P.applyMatrix4(fe.matrixWorld),P.x<X&&(X=P.x),P.x>be&&(be=P.x),P.y<ee&&(ee=P.y),P.y>xe&&(xe=P.y),P.z<pe&&(pe=P.z),P.z>Ce&&(Ce=P.z);const q=(X+be)/2,w=(ee+xe)/2,te=(pe+Ce)/2,ie=Math.max(be-X,xe-ee,Ce-pe);v.uCenter.value.set(q,w,te),v.uMaxRadius.value=ie*.7;let ue=6e4;re<4e4?ue=re:re>1e6&&(ue=4e5);const Z=Math.ceil(re/ue);let de=ie/200*window.devicePixelRatio;de<.5&&(de=.5);const ne=ie*1;console.log(`[Adaptive] MaxDim: ${ie.toFixed(2)}, Particles: ~${Math.floor(re/Z)}, Size: ${de.toFixed(2)}`);const ge=new An,R=[],T=[],G=[];for(let le=0;le<re;le+=Z){fe.getSplatCenter(le,P),P.applyMatrix4(fe.matrixWorld),T.push(P.x,P.y,P.z);const j=ne+Math.random()*(ie*.5),De=Math.random()*Math.PI*2,_e=Math.acos(2*Math.random()-1),Ue=q+j*Math.sin(_e)*Math.cos(De),N=w+j*Math.sin(_e)*Math.sin(De),J=te+j*Math.cos(_e);R.push(Ue,N,J),G.push(Math.random())}ge.setAttribute("position",new on(R,3)),ge.setAttribute("aTarget",new on(T,3)),ge.setAttribute("aRandom",new on(G,1));const se=new _n({uniforms:{uProgress:v.uParticleProgress,uSize:{value:de},uColor:{value:new nt(.6,.6,.6)}},vertexShader:`
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
    `,transparent:!0,opacity:1,depthTest:!0,depthWrite:!1});g=new pv(ge,se),g.frustumCulled=!1,p.threeScene.add(g)},M=fe=>{if(!fe||!fe.material)return;const re=fe.material;re.uniforms=re.uniforms||{},re.uniforms.uGeoRadius=v.uGeoRadius,re.uniforms.uColorRadius=v.uColorRadius,re.uniforms.uMaxRadius=v.uMaxRadius,re.uniforms.uCenter=v.uCenter,re.vertexShader=`varying vec3 vWorldPosition;
`+re.vertexShader;const X=re.vertexShader.lastIndexOf("}");if(X!==-1){const be=`vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
`;re.vertexShader=re.vertexShader.substring(0,X)+be+"}"}const ee=`
    uniform float uGeoRadius;
    uniform float uColorRadius;
    uniform float uMaxRadius;
    uniform vec3 uCenter;
    varying vec3 vWorldPosition;
  `;re.fragmentShader=ee+re.fragmentShader;const pe=re.fragmentShader.lastIndexOf("}");if(pe!==-1){const be=re.fragmentShader.substring(0,pe),xe=`
      float distFromCenter = distance(vWorldPosition, uCenter);
      
      if (distFromCenter > uGeoRadius) {
          discard;
      }
      if (distFromCenter > uColorRadius) {
          if (gl_FragColor.a < 0.8) discard; 
          gl_FragColor.a = 1.0; 
          gl_FragColor.rgb = vec3(0.6, 0.6, 0.6);
      } 
    `;re.fragmentShader=be+xe+"}"}re.needsUpdate=!0},E=fe=>{if(!p||!p.camera)return;const re=p.camera,X=p.getSplatMesh();l.value=fe.image_url;const ee=new qe().fromArray(fe.matrix),pe=new qe;X?(X.updateMatrixWorld(),pe.copy(X.matrixWorld).multiply(ee)):pe.copy(ee);const be=new B,xe=new bt,Ce=new B;pe.decompose(be,xe,Ce);const P=fe.fl_y||c.value.fl_y,L=fe.h||c.value.h;if(P&&L){const Z=2*Math.atan(L/2/P)*(180/Math.PI);Vr.to(re,{fov:Z,duration:1.5,ease:"power3.inOut",onUpdate:()=>re.updateProjectionMatrix()})}re.near>.001&&(re.near=.001,re.updateProjectionMatrix());const q=new B(0,0,-1).applyQuaternion(xe),w=be.clone().add(q.multiplyScalar(5));n.value=!1,p.controls&&(p.controls.enabled=!1);const te=re.position.clone(),ie=re.quaternion.clone(),ue={t:0};Vr.killTweensOf(re.position),Vr.killTweensOf(re.quaternion),Vr.killTweensOf(ue),Vr.to(ue,{t:1,duration:1.5,ease:"power3.inOut",onUpdate:()=>{re.position.lerpVectors(te,be,ue.t),re.quaternion.slerpQuaternions(ie,xe,ue.t)},onComplete:()=>{const Z=new _i().setFromQuaternion(re.quaternion,"YXZ");f.value={x:(Z.x*180/Math.PI).toFixed(1),y:(Z.y*180/Math.PI).toFixed(1),z:(Z.z*180/Math.PI).toFixed(1)},m.value={x:0,y:0},_(),p.controls&&(p.controls.target.copy(w),p.controls.update(),p.controls.enabled=!0)}})},b=()=>{const fe=/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);return{rootElement:e.value,cameraUp:[0,1,0],initialCameraPosition:[0,0,5],initialCameraLookAt:[0,0,0],useBuiltInControls:!1,gpuAcceleratedSort:!1,webXRMode:r.value?ys.VR:ys.None,sharedMemoryForWorkers:!1,antialiased:!fe}},C=async(fe="./models/scene_auto_sync_raw.ply")=>{if(!s.value){s.value=!0;try{p&&(p.renderer.setAnimationLoop(null),p.dispose&&await p.dispose(),p=null),e.value&&(e.value.innerHTML=""),S.isLoaded=!1,S.phase=A.FLY_IN,v.uParticleProgress.value=0,v.uGeoRadius.value=0,v.uColorRadius.value=0;const re=b();p=new Gr(re),window.viewer=p,await p.addSplatScene(fe,{showLoadingUI:!0,progressiveLoad:!1,rotation:[0,0,0,1]}),s.value=!1,window.BrainDanceChannel&&window.BrainDanceChannel.postMessage(JSON.stringify({status:"success",msg:"模型加载完成"}));let X="",ee="",pe=!1,be="./models/webgl_poses_with_tags.json";if(fe.includes("/proxy/")){pe=!0;const Ce=fe.split("/proxy/");X=Ce[0]+"/proxy/";const P=decodeURIComponent(Ce[1]);ee=P.substring(0,P.lastIndexOf("/")),be=X+encodeURIComponent(ee+"/webgl_poses_with_tags.json")}else fe.startsWith("http")&&!fe.includes("127.0.0.1")&&(ee=fe.substring(0,fe.lastIndexOf("/")),be=ee+"/webgl_poses_with_tags.json");fetch(be).then(Ce=>Ce.json()).then(Ce=>{Ce.frames?(c.value={w:Ce.w,h:Ce.h,fl_x:Ce.fl_x,fl_y:Ce.fl_y},o.value=Ce.frames.map(P=>{let L=P.image_url;return pe?(L.startsWith("/models/")&&(L=L.replace("/models/","/")),L=X+encodeURIComponent(ee+L)):ee?(L.startsWith("/models/")&&(L=L.replace("/models/","/")),L=ee+L):L.startsWith("/")&&(L="."+L),{id:P.id,matrix:P.matrix,image_url:L,tag:P.tag}})):o.value=Ce}).catch(Ce=>console.error("加载位姿失败:",Ce));const xe=p.getSplatMesh();xe.visible=!1,setTimeout(()=>{xe&&(y(xe),M(xe),F(),S.lastFrameTime=Date.now(),S.startTime=Date.now(),S.isLoaded=!0)},200),p.renderer.setAnimationLoop(()=>{if(p.update(),p.render(),!S.isLoaded||S.phase===A.FINISHED)return;const Ce=Date.now(),P=(Ce-S.lastFrameTime)/1e3||.016;if(S.lastFrameTime=Ce,S.phase===A.FLY_IN){const L=1/S.flyDuration;let q=v.uParticleProgress.value+P*L;if(q>=1.2){q=1.2;const w=p.getSplatMesh();w&&(w.visible=!0),S.phase=A.DIFFUSION,S.diffuseTime=0}v.uParticleProgress.value=q}else if(S.phase===A.DIFFUSION){S.diffuseTime+=P;const L=Math.min(S.diffuseTime/S.diffusionDuration,1),q=v.uMaxRadius.value;v.uGeoRadius.value=L*(q*1.5),g&&g.material&&(g.material.opacity=1-L),L>=1&&(g&&(g.visible=!1),v.uGeoRadius.value=99999,S.phase=A.COLORING,S.colorStartTime=Ce)}else if(S.phase===A.COLORING){const L=(Ce-S.colorStartTime)/1e3,q=v.uMaxRadius.value,w=L/S.colorDuration;v.uColorRadius.value=w*(q*1.5),w>=1&&(S.phase=A.FINISHED,v.uColorRadius.value=99999)}}),I()}catch(re){console.error("error:",re),s.value=!1,d.value=re&&(re.message||String(re))||"模型加载失败，请检查模型 URL 是否正确可访问"}}},I=()=>{p&&(p.controls&&(p.controls.dispose(),p.controls=null),console.log("Controls explicitly disabled for debugging"))},F=()=>{if(t.value)return;const fe=v.uCenter.value,X=v.uMaxRadius.value/.7*2;p.controls&&(p.controls.target.copy(fe),p.controls.update()),p.camera.position.set(fe.x,fe.y,fe.z+X),p.camera.lookAt(fe)},U=async()=>{if(!r.value){alert("需HTTPS");return}t.value?(p.xr&&p.xr.exitVR(),t.value=!1):(p.xr&&p.xr.enterVR(),t.value=!0)},O=()=>{n.value=!n.value},k=()=>{const fe=window.location.hostname==="localhost"||window.location.hostname==="127.0.0.1",re=window.location.protocol==="https:";r.value=fe||re},z=yn(!1),V={x:0,y:0},H=fe=>{z.value=!0,V.x=fe.clientX,V.y=fe.clientY},$=fe=>{if(!z.value||!p||!p.camera)return;const re=fe.clientX-V.x,pe=(fe.clientY-V.y)*.2,be=.01;p.camera.rotateX(pe*Math.PI/180),p.camera.translateX(-re*be),p.camera.updateProjectionMatrix(),_(),V.x=fe.clientX,V.y=fe.clientY},oe=()=>{z.value=!1},Se=fe=>{fe.touches.length>0&&(z.value=!0,V.x=fe.touches[0].clientX,V.y=fe.touches[0].clientY)},we=fe=>{if(!z.value||!p||!p.camera||fe.touches.length===0)return;const re=fe.touches[0].clientX-V.x,pe=(fe.touches[0].clientY-V.y)*.2,be=.01;m.value.x+=pe,p.camera.rotateX(pe*Math.PI/180),p.camera.translateX(-re*be),p.camera.updateProjectionMatrix(),_(),V.x=fe.touches[0].clientX,V.y=fe.touches[0].clientY},Le=()=>{z.value=!1};return $m(()=>{window.loadModelFromFlutter=fe=>{console.log("准备加载模型: ",fe),C(fe)},e.value&&(k(),window.BrainDanceChannel?window.BrainDanceChannel.postMessage(JSON.stringify({status:"ready"})):C(),window.addEventListener("mousedown",H),window.addEventListener("mousemove",$),window.addEventListener("mouseup",oe))}),Zm(async()=>{delete window.loadModelFromFlutter,window.removeEventListener("mousemove",$),window.removeEventListener("mouseup",oe),p&&(p.renderer.setAnimationLoop(null),await p.dispose())}),(fe,re)=>(Cn(),Nn("div",{class:"app-container",onMousedown:H,onMousemove:$,onMouseup:oe,onMouseleave:oe,onTouchstart:Se,onTouchmove:Ar(we,["prevent"]),onTouchend:Le,onTouchcancel:Le},[Ht("div",{ref_key:"containerRef",ref:e,class:"viewer-container"},null,512),s.value?(Cn(),Nn("div",ow,"正在加载模型...")):fs("",!0),d.value?(Cn(),Nn("div",aw,[re[7]||(re[7]=Ht("div",{class:"error-icon"},"⚠️",-1)),re[8]||(re[8]=Ht("div",{class:"error-title"},"模型加载失败",-1)),Ht("div",lw,hi(d.value),1),Ht("button",{class:"error-retry",onClick:re[0]||(re[0]=X=>d.value="")},"关闭")])):fs("",!0),Ht("div",cw,[r.value?(Cn(),Nn("button",{key:0,onClick:U,class:Yr({active:t.value})},hi(t.value?"退出 VR":"进入 VR"),3)):fs("",!0),Ht("button",{onClick:O,class:Yr({active:n.value})},hi(n.value?"停止旋转":"自动旋转"),3)]),Ht("div",uw,[o_(Ht("input",{type:"text","onUpdate:modelValue":re[1]||(re[1]=X=>a.value=X),onKeyup:NA(x,["enter"]),placeholder:"搜索想要的视角 (如: 正面特写...)",class:"search-input"},null,544),[[LA,a.value]]),Ht("button",{onClick:x,class:"search-btn"},"🔍 搜索视角")]),h.value.length>0?(Cn(),Nn("div",{key:2,class:"camera-track",onMousedown:re[2]||(re[2]=Ar(()=>{},["stop"])),onTouchstart:re[3]||(re[3]=Ar(()=>{},["stop"])),onTouchmove:re[4]||(re[4]=Ar(()=>{},["stop"])),onTouchend:re[5]||(re[5]=Ar(()=>{},["stop"]))},[(Cn(!0),Nn(bi,null,C_(h.value,(X,ee)=>(Cn(),Nn("div",{key:X.id,class:Yr(["camera-btn",{active:l.value===X.image_url}]),onClick:Ar(pe=>E(X),["stop"])},[X.image_url?(Cn(),Nn("img",{key:0,src:X.image_url,class:"btn-thumb"},null,8,dw)):fs("",!0),X.tag?(Cn(),Nn("div",hw,[Ht("div",pw,"镜 "+hi(X.id.split(".")[0].replace("frame_","")),1),Ht("div",mw,hi(X.tag),1)])):X.image_url?fs("",!0):(Cn(),Nn("span",gw,"镜头 "+hi(ee+1),1))],10,fw))),128))],32)):fs("",!0),l.value?(Cn(),Nn("div",{key:3,class:"reference-overlay",onClick:re[6]||(re[6]=X=>l.value="")},[re[9]||(re[9]=Ht("div",{class:"ref-title"},"参考原图",-1)),Ht("img",{src:l.value,class:"ref-img"},null,8,xw),c.value.fl_y?(Cn(),Nn("div",_w,[Ht("span",Aw,"焦距: "+hi(c.value.fl_y.toFixed(1))+" px",1),Ht("span",Sw,"FOV: "+hi((2*Math.atan(c.value.h/(2*c.value.fl_y))*(180/Math.PI)).toFixed(1))+"°",1),Ht("span",vw,"分辨率: "+hi(c.value.w)+"x"+hi(c.value.h),1)])):fs("",!0),re[10]||(re[10]=Ht("div",{class:"ref-hint"},"点击关闭对比",-1))])):fs("",!0)],32))}},bw=rw(yw,[["__scopeId","data-v-4f55b623"]]),Mw={__name:"App",setup(i){return(e,t)=>(Cn(),Nn("main",null,[Pi(bw)]))}};HA(Mw).mount("#app");
