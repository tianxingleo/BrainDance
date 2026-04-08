(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))n(s);new MutationObserver(s=>{for(const r of s)if(r.type==="childList")for(const o of r.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&n(o)}).observe(document,{childList:!0,subtree:!0});function t(s){const r={};return s.integrity&&(r.integrity=s.integrity),s.referrerPolicy&&(r.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?r.credentials="include":s.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function n(s){if(s.ep)return;s.ep=!0;const r=t(s);fetch(s.href,r)}})();function Rd(i){const e=Object.create(null);for(const t of i.split(","))e[t]=1;return t=>t in e}const Ht={},Ro=[],ns=()=>{},b0=()=>!1,Oc=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&(i.charCodeAt(2)>122||i.charCodeAt(2)<97),Id=i=>i.startsWith("onUpdate:"),Qn=Object.assign,Dd=(i,e)=>{const t=i.indexOf(e);t>-1&&i.splice(t,1)},A_=Object.prototype.hasOwnProperty,Lt=(i,e)=>A_.call(i,e),ct=Array.isArray,Io=i=>ul(i)==="[object Map]",Nc=i=>ul(i)==="[object Set]",Yh=i=>ul(i)==="[object Date]",xt=i=>typeof i=="function",gn=i=>typeof i=="string",ss=i=>typeof i=="symbol",Wt=i=>i!==null&&typeof i=="object",M0=i=>(Wt(i)||xt(i))&&xt(i.then)&&xt(i.catch),C0=Object.prototype.toString,ul=i=>C0.call(i),S_=i=>ul(i).slice(8,-1),T0=i=>ul(i)==="[object Object]",Pd=i=>gn(i)&&i!=="NaN"&&i[0]!=="-"&&""+parseInt(i,10)===i,Ca=Rd(",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"),zc=i=>{const e=Object.create(null);return(t=>e[t]||(e[t]=i(t)))},y_=/-\w/g,ar=zc(i=>i.replace(y_,e=>e.slice(1).toUpperCase())),b_=/\B([A-Z])/g,hr=zc(i=>i.replace(b_,"-$1").toLowerCase()),E0=zc(i=>i.charAt(0).toUpperCase()+i.slice(1)),au=zc(i=>i?`on${E0(i)}`:""),nr=(i,e)=>!Object.is(i,e),sc=(i,...e)=>{for(let t=0;t<i.length;t++)i[t](...e)},w0=(i,e,t,n=!1)=>{Object.defineProperty(i,e,{configurable:!0,enumerable:!1,writable:n,value:t})},Fd=i=>{const e=parseFloat(i);return isNaN(e)?i:e};let Qh;const kc=()=>Qh||(Qh=typeof globalThis<"u"?globalThis:typeof self<"u"?self:typeof window<"u"?window:typeof global<"u"?global:{});function Ld(i){if(ct(i)){const e={};for(let t=0;t<i.length;t++){const n=i[t],s=gn(n)?E_(n):Ld(n);if(s)for(const r in s)e[r]=s[r]}return e}else if(gn(i)||Wt(i))return i}const M_=/;(?![^(]*\))/g,C_=/:([^]+)/,T_=/\/\*[^]*?\*\//g;function E_(i){const e={};return i.replace(T_,"").split(M_).forEach(t=>{if(t){const n=t.split(C_);n.length>1&&(e[n[0].trim()]=n[1].trim())}}),e}function $i(i){let e="";if(gn(i))e=i;else if(ct(i))for(let t=0;t<i.length;t++){const n=$i(i[t]);n&&(e+=n+" ")}else if(Wt(i))for(const t in i)i[t]&&(e+=t+" ");return e.trim()}const w_="itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly",R_=Rd(w_);function R0(i){return!!i||i===""}function I_(i,e){if(i.length!==e.length)return!1;let t=!0;for(let n=0;t&&n<i.length;n++)t=Hc(i[n],e[n]);return t}function Hc(i,e){if(i===e)return!0;let t=Yh(i),n=Yh(e);if(t||n)return t&&n?i.getTime()===e.getTime():!1;if(t=ss(i),n=ss(e),t||n)return i===e;if(t=ct(i),n=ct(e),t||n)return t&&n?I_(i,e):!1;if(t=Wt(i),n=Wt(e),t||n){if(!t||!n)return!1;const s=Object.keys(i).length,r=Object.keys(e).length;if(s!==r)return!1;for(const o in i){const a=i.hasOwnProperty(o),l=e.hasOwnProperty(o);if(a&&!l||!a&&l||!Hc(i[o],e[o]))return!1}}return String(i)===String(e)}function I0(i,e){return i.findIndex(t=>Hc(t,e))}const D0=i=>!!(i&&i.__v_isRef===!0),Mn=i=>gn(i)?i:i==null?"":ct(i)||Wt(i)&&(i.toString===C0||!xt(i.toString))?D0(i)?Mn(i.value):JSON.stringify(i,P0,2):String(i),P0=(i,e)=>D0(e)?P0(i,e.value):Io(e)?{[`Map(${e.size})`]:[...e.entries()].reduce((t,[n,s],r)=>(t[lu(n,r)+" =>"]=s,t),{})}:Nc(e)?{[`Set(${e.size})`]:[...e.values()].map(t=>lu(t))}:ss(e)?lu(e):Wt(e)&&!ct(e)&&!T0(e)?String(e):e,lu=(i,e="")=>{var t;return ss(i)?`Symbol(${(t=i.description)!=null?t:e})`:i};let $n;class D_{constructor(e=!1){this.detached=e,this._active=!0,this._on=0,this.effects=[],this.cleanups=[],this._isPaused=!1,this.parent=$n,!e&&$n&&(this.index=($n.scopes||($n.scopes=[])).push(this)-1)}get active(){return this._active}pause(){if(this._active){this._isPaused=!0;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].pause();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].pause()}}resume(){if(this._active&&this._isPaused){this._isPaused=!1;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].resume();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].resume()}}run(e){if(this._active){const t=$n;try{return $n=this,e()}finally{$n=t}}}on(){++this._on===1&&(this.prevScope=$n,$n=this)}off(){this._on>0&&--this._on===0&&($n=this.prevScope,this.prevScope=void 0)}stop(e){if(this._active){this._active=!1;let t,n;for(t=0,n=this.effects.length;t<n;t++)this.effects[t].stop();for(this.effects.length=0,t=0,n=this.cleanups.length;t<n;t++)this.cleanups[t]();if(this.cleanups.length=0,this.scopes){for(t=0,n=this.scopes.length;t<n;t++)this.scopes[t].stop(!0);this.scopes.length=0}if(!this.detached&&this.parent&&!e){const s=this.parent.scopes.pop();s&&s!==this&&(this.parent.scopes[this.index]=s,s.index=this.index)}this.parent=void 0}}}function P_(){return $n}let Gt;const cu=new WeakSet;class F0{constructor(e){this.fn=e,this.deps=void 0,this.depsTail=void 0,this.flags=5,this.next=void 0,this.cleanup=void 0,this.scheduler=void 0,$n&&$n.active&&$n.effects.push(this)}pause(){this.flags|=64}resume(){this.flags&64&&(this.flags&=-65,cu.has(this)&&(cu.delete(this),this.trigger()))}notify(){this.flags&2&&!(this.flags&32)||this.flags&8||B0(this)}run(){if(!(this.flags&1))return this.fn();this.flags|=2,Kh(this),U0(this);const e=Gt,t=Vi;Gt=this,Vi=!0;try{return this.fn()}finally{O0(this),Gt=e,Vi=t,this.flags&=-3}}stop(){if(this.flags&1){for(let e=this.deps;e;e=e.nextDep)Od(e);this.deps=this.depsTail=void 0,Kh(this),this.onStop&&this.onStop(),this.flags&=-2}}trigger(){this.flags&64?cu.add(this):this.scheduler?this.scheduler():this.runIfDirty()}runIfDirty(){pf(this)&&this.run()}get dirty(){return pf(this)}}let L0=0,Ta,Ea;function B0(i,e=!1){if(i.flags|=8,e){i.next=Ea,Ea=i;return}i.next=Ta,Ta=i}function Bd(){L0++}function Ud(){if(--L0>0)return;if(Ea){let e=Ea;for(Ea=void 0;e;){const t=e.next;e.next=void 0,e.flags&=-9,e=t}}let i;for(;Ta;){let e=Ta;for(Ta=void 0;e;){const t=e.next;if(e.next=void 0,e.flags&=-9,e.flags&1)try{e.trigger()}catch(n){i||(i=n)}e=t}}if(i)throw i}function U0(i){for(let e=i.deps;e;e=e.nextDep)e.version=-1,e.prevActiveLink=e.dep.activeLink,e.dep.activeLink=e}function O0(i){let e,t=i.depsTail,n=t;for(;n;){const s=n.prevDep;n.version===-1?(n===t&&(t=s),Od(n),F_(n)):e=n,n.dep.activeLink=n.prevActiveLink,n.prevActiveLink=void 0,n=s}i.deps=e,i.depsTail=t}function pf(i){for(let e=i.deps;e;e=e.nextDep)if(e.dep.version!==e.version||e.dep.computed&&(N0(e.dep.computed)||e.dep.version!==e.version))return!0;return!!i._dirty}function N0(i){if(i.flags&4&&!(i.flags&16)||(i.flags&=-17,i.globalVersion===Ga)||(i.globalVersion=Ga,!i.isSSR&&i.flags&128&&(!i.deps&&!i._dirty||!pf(i))))return;i.flags|=2;const e=i.dep,t=Gt,n=Vi;Gt=i,Vi=!0;try{U0(i);const s=i.fn(i._value);(e.version===0||nr(s,i._value))&&(i.flags|=128,i._value=s,e.version++)}catch(s){throw e.version++,s}finally{Gt=t,Vi=n,O0(i),i.flags&=-3}}function Od(i,e=!1){const{dep:t,prevSub:n,nextSub:s}=i;if(n&&(n.nextSub=s,i.prevSub=void 0),s&&(s.prevSub=n,i.nextSub=void 0),t.subs===i&&(t.subs=n,!n&&t.computed)){t.computed.flags&=-5;for(let r=t.computed.deps;r;r=r.nextDep)Od(r,!0)}!e&&!--t.sc&&t.map&&t.map.delete(t.key)}function F_(i){const{prevDep:e,nextDep:t}=i;e&&(e.nextDep=t,i.prevDep=void 0),t&&(t.prevDep=e,i.nextDep=void 0)}let Vi=!0;const z0=[];function Fs(){z0.push(Vi),Vi=!1}function Ls(){const i=z0.pop();Vi=i===void 0?!0:i}function Kh(i){const{cleanup:e}=i;if(i.cleanup=void 0,e){const t=Gt;Gt=void 0;try{e()}finally{Gt=t}}}let Ga=0;class L_{constructor(e,t){this.sub=e,this.dep=t,this.version=t.version,this.nextDep=this.prevDep=this.nextSub=this.prevSub=this.prevActiveLink=void 0}}class Nd{constructor(e){this.computed=e,this.version=0,this.activeLink=void 0,this.subs=void 0,this.map=void 0,this.key=void 0,this.sc=0,this.__v_skip=!0}track(e){if(!Gt||!Vi||Gt===this.computed)return;let t=this.activeLink;if(t===void 0||t.sub!==Gt)t=this.activeLink=new L_(Gt,this),Gt.deps?(t.prevDep=Gt.depsTail,Gt.depsTail.nextDep=t,Gt.depsTail=t):Gt.deps=Gt.depsTail=t,k0(t);else if(t.version===-1&&(t.version=this.version,t.nextDep)){const n=t.nextDep;n.prevDep=t.prevDep,t.prevDep&&(t.prevDep.nextDep=n),t.prevDep=Gt.depsTail,t.nextDep=void 0,Gt.depsTail.nextDep=t,Gt.depsTail=t,Gt.deps===t&&(Gt.deps=n)}return t}trigger(e){this.version++,Ga++,this.notify(e)}notify(e){Bd();try{for(let t=this.subs;t;t=t.prevSub)t.sub.notify()&&t.sub.dep.notify()}finally{Ud()}}}function k0(i){if(i.dep.sc++,i.sub.flags&4){const e=i.dep.computed;if(e&&!i.dep.subs){e.flags|=20;for(let n=e.deps;n;n=n.nextDep)k0(n)}const t=i.dep.subs;t!==i&&(i.prevSub=t,t&&(t.nextSub=i)),i.dep.subs=i}}const mf=new WeakMap,Gr=Symbol(""),gf=Symbol(""),Wa=Symbol("");function Dn(i,e,t){if(Vi&&Gt){let n=mf.get(i);n||mf.set(i,n=new Map);let s=n.get(t);s||(n.set(t,s=new Nd),s.map=n,s.key=t),s.track()}}function Es(i,e,t,n,s,r){const o=mf.get(i);if(!o){Ga++;return}const a=l=>{l&&l.trigger()};if(Bd(),e==="clear")o.forEach(a);else{const l=ct(i),c=l&&Pd(t);if(l&&t==="length"){const u=Number(n);o.forEach((f,d)=>{(d==="length"||d===Wa||!ss(d)&&d>=u)&&a(f)})}else switch((t!==void 0||o.has(void 0))&&a(o.get(t)),c&&a(o.get(Wa)),e){case"add":l?c&&a(o.get("length")):(a(o.get(Gr)),Io(i)&&a(o.get(gf)));break;case"delete":l||(a(o.get(Gr)),Io(i)&&a(o.get(gf)));break;case"set":Io(i)&&a(o.get(Gr));break}}Ud()}function no(i){const e=Ft(i);return e===i?e:(Dn(e,"iterate",Wa),Fi(i)?e:e.map(Gi))}function Vc(i){return Dn(i=Ft(i),"iterate",Wa),i}function qs(i,e){return Bs(i)?Wr(i)?Vo(Gi(e)):Vo(e):Gi(e)}const B_={__proto__:null,[Symbol.iterator](){return uu(this,Symbol.iterator,i=>qs(this,i))},concat(...i){return no(this).concat(...i.map(e=>ct(e)?no(e):e))},entries(){return uu(this,"entries",i=>(i[1]=qs(this,i[1]),i))},every(i,e){return hs(this,"every",i,e,void 0,arguments)},filter(i,e){return hs(this,"filter",i,e,t=>t.map(n=>qs(this,n)),arguments)},find(i,e){return hs(this,"find",i,e,t=>qs(this,t),arguments)},findIndex(i,e){return hs(this,"findIndex",i,e,void 0,arguments)},findLast(i,e){return hs(this,"findLast",i,e,t=>qs(this,t),arguments)},findLastIndex(i,e){return hs(this,"findLastIndex",i,e,void 0,arguments)},forEach(i,e){return hs(this,"forEach",i,e,void 0,arguments)},includes(...i){return fu(this,"includes",i)},indexOf(...i){return fu(this,"indexOf",i)},join(i){return no(this).join(i)},lastIndexOf(...i){return fu(this,"lastIndexOf",i)},map(i,e){return hs(this,"map",i,e,void 0,arguments)},pop(){return fa(this,"pop")},push(...i){return fa(this,"push",i)},reduce(i,...e){return jh(this,"reduce",i,e)},reduceRight(i,...e){return jh(this,"reduceRight",i,e)},shift(){return fa(this,"shift")},some(i,e){return hs(this,"some",i,e,void 0,arguments)},splice(...i){return fa(this,"splice",i)},toReversed(){return no(this).toReversed()},toSorted(i){return no(this).toSorted(i)},toSpliced(...i){return no(this).toSpliced(...i)},unshift(...i){return fa(this,"unshift",i)},values(){return uu(this,"values",i=>qs(this,i))}};function uu(i,e,t){const n=Vc(i),s=n[e]();return n!==i&&!Fi(i)&&(s._next=s.next,s.next=()=>{const r=s._next();return r.done||(r.value=t(r.value)),r}),s}const U_=Array.prototype;function hs(i,e,t,n,s,r){const o=Vc(i),a=o!==i&&!Fi(i),l=o[e];if(l!==U_[e]){const f=l.apply(i,r);return a?Gi(f):f}let c=t;o!==i&&(a?c=function(f,d){return t.call(this,qs(i,f),d,i)}:t.length>2&&(c=function(f,d){return t.call(this,f,d,i)}));const u=l.call(o,c,n);return a&&s?s(u):u}function jh(i,e,t,n){const s=Vc(i);let r=t;return s!==i&&(Fi(i)?t.length>3&&(r=function(o,a,l){return t.call(this,o,a,l,i)}):r=function(o,a,l){return t.call(this,o,qs(i,a),l,i)}),s[e](r,...n)}function fu(i,e,t){const n=Ft(i);Dn(n,"iterate",Wa);const s=n[e](...t);return(s===-1||s===!1)&&Vd(t[0])?(t[0]=Ft(t[0]),n[e](...t)):s}function fa(i,e,t=[]){Fs(),Bd();const n=Ft(i)[e].apply(i,t);return Ud(),Ls(),n}const O_=Rd("__proto__,__v_isRef,__isVue"),H0=new Set(Object.getOwnPropertyNames(Symbol).filter(i=>i!=="arguments"&&i!=="caller").map(i=>Symbol[i]).filter(ss));function N_(i){ss(i)||(i=String(i));const e=Ft(this);return Dn(e,"has",i),e.hasOwnProperty(i)}class V0{constructor(e=!1,t=!1){this._isReadonly=e,this._isShallow=t}get(e,t,n){if(t==="__v_skip")return e.__v_skip;const s=this._isReadonly,r=this._isShallow;if(t==="__v_isReactive")return!s;if(t==="__v_isReadonly")return s;if(t==="__v_isShallow")return r;if(t==="__v_raw")return n===(s?r?Q_:q0:r?X0:W0).get(e)||Object.getPrototypeOf(e)===Object.getPrototypeOf(n)?e:void 0;const o=ct(e);if(!s){let l;if(o&&(l=B_[t]))return l;if(t==="hasOwnProperty")return N_}const a=Reflect.get(e,t,Fn(e)?e:n);if((ss(t)?H0.has(t):O_(t))||(s||Dn(e,"get",t),r))return a;if(Fn(a)){const l=o&&Pd(t)?a:a.value;return s&&Wt(l)?_f(l):l}return Wt(a)?s?_f(a):kd(a):a}}class G0 extends V0{constructor(e=!1){super(!1,e)}set(e,t,n,s){let r=e[t];const o=ct(e)&&Pd(t);if(!this._isShallow){const c=Bs(r);if(!Fi(n)&&!Bs(n)&&(r=Ft(r),n=Ft(n)),!o&&Fn(r)&&!Fn(n))return c||(r.value=n),!0}const a=o?Number(t)<e.length:Lt(e,t),l=Reflect.set(e,t,n,Fn(e)?e:s);return e===Ft(s)&&(a?nr(n,r)&&Es(e,"set",t,n):Es(e,"add",t,n)),l}deleteProperty(e,t){const n=Lt(e,t);e[t];const s=Reflect.deleteProperty(e,t);return s&&n&&Es(e,"delete",t,void 0),s}has(e,t){const n=Reflect.has(e,t);return(!ss(t)||!H0.has(t))&&Dn(e,"has",t),n}ownKeys(e){return Dn(e,"iterate",ct(e)?"length":Gr),Reflect.ownKeys(e)}}class z_ extends V0{constructor(e=!1){super(!0,e)}set(e,t){return!0}deleteProperty(e,t){return!0}}const k_=new G0,H_=new z_,V_=new G0(!0);const xf=i=>i,yl=i=>Reflect.getPrototypeOf(i);function G_(i,e,t){return function(...n){const s=this.__v_raw,r=Ft(s),o=Io(r),a=i==="entries"||i===Symbol.iterator&&o,l=i==="keys"&&o,c=s[i](...n),u=t?xf:e?Vo:Gi;return!e&&Dn(r,"iterate",l?gf:Gr),{next(){const{value:f,done:d}=c.next();return d?{value:f,done:d}:{value:a?[u(f[0]),u(f[1])]:u(f),done:d}},[Symbol.iterator](){return this}}}}function bl(i){return function(...e){return i==="delete"?!1:i==="clear"?void 0:this}}function W_(i,e){const t={get(s){const r=this.__v_raw,o=Ft(r),a=Ft(s);i||(nr(s,a)&&Dn(o,"get",s),Dn(o,"get",a));const{has:l}=yl(o),c=e?xf:i?Vo:Gi;if(l.call(o,s))return c(r.get(s));if(l.call(o,a))return c(r.get(a));r!==o&&r.get(s)},get size(){const s=this.__v_raw;return!i&&Dn(Ft(s),"iterate",Gr),s.size},has(s){const r=this.__v_raw,o=Ft(r),a=Ft(s);return i||(nr(s,a)&&Dn(o,"has",s),Dn(o,"has",a)),s===a?r.has(s):r.has(s)||r.has(a)},forEach(s,r){const o=this,a=o.__v_raw,l=Ft(a),c=e?xf:i?Vo:Gi;return!i&&Dn(l,"iterate",Gr),a.forEach((u,f)=>s.call(r,c(u),c(f),o))}};return Qn(t,i?{add:bl("add"),set:bl("set"),delete:bl("delete"),clear:bl("clear")}:{add(s){!e&&!Fi(s)&&!Bs(s)&&(s=Ft(s));const r=Ft(this);return yl(r).has.call(r,s)||(r.add(s),Es(r,"add",s,s)),this},set(s,r){!e&&!Fi(r)&&!Bs(r)&&(r=Ft(r));const o=Ft(this),{has:a,get:l}=yl(o);let c=a.call(o,s);c||(s=Ft(s),c=a.call(o,s));const u=l.call(o,s);return o.set(s,r),c?nr(r,u)&&Es(o,"set",s,r):Es(o,"add",s,r),this},delete(s){const r=Ft(this),{has:o,get:a}=yl(r);let l=o.call(r,s);l||(s=Ft(s),l=o.call(r,s)),a&&a.call(r,s);const c=r.delete(s);return l&&Es(r,"delete",s,void 0),c},clear(){const s=Ft(this),r=s.size!==0,o=s.clear();return r&&Es(s,"clear",void 0,void 0),o}}),["keys","values","entries",Symbol.iterator].forEach(s=>{t[s]=G_(s,i,e)}),t}function zd(i,e){const t=W_(i,e);return(n,s,r)=>s==="__v_isReactive"?!i:s==="__v_isReadonly"?i:s==="__v_raw"?n:Reflect.get(Lt(t,s)&&s in n?t:n,s,r)}const X_={get:zd(!1,!1)},q_={get:zd(!1,!0)},Y_={get:zd(!0,!1)};const W0=new WeakMap,X0=new WeakMap,q0=new WeakMap,Q_=new WeakMap;function K_(i){switch(i){case"Object":case"Array":return 1;case"Map":case"Set":case"WeakMap":case"WeakSet":return 2;default:return 0}}function j_(i){return i.__v_skip||!Object.isExtensible(i)?0:K_(S_(i))}function kd(i){return Bs(i)?i:Hd(i,!1,k_,X_,W0)}function $_(i){return Hd(i,!1,V_,q_,X0)}function _f(i){return Hd(i,!0,H_,Y_,q0)}function Hd(i,e,t,n,s){if(!Wt(i)||i.__v_raw&&!(e&&i.__v_isReactive))return i;const r=j_(i);if(r===0)return i;const o=s.get(i);if(o)return o;const a=new Proxy(i,r===2?n:t);return s.set(i,a),a}function Wr(i){return Bs(i)?Wr(i.__v_raw):!!(i&&i.__v_isReactive)}function Bs(i){return!!(i&&i.__v_isReadonly)}function Fi(i){return!!(i&&i.__v_isShallow)}function Vd(i){return i?!!i.__v_raw:!1}function Ft(i){const e=i&&i.__v_raw;return e?Ft(e):i}function Z_(i){return!Lt(i,"__v_skip")&&Object.isExtensible(i)&&w0(i,"__v_skip",!0),i}const Gi=i=>Wt(i)?kd(i):i,Vo=i=>Wt(i)?_f(i):i;function Fn(i){return i?i.__v_isRef===!0:!1}function yt(i){return J_(i,!1)}function J_(i,e){return Fn(i)?i:new ev(i,e)}class ev{constructor(e,t){this.dep=new Nd,this.__v_isRef=!0,this.__v_isShallow=!1,this._rawValue=t?e:Ft(e),this._value=t?e:Gi(e),this.__v_isShallow=t}get value(){return this.dep.track(),this._value}set value(e){const t=this._rawValue,n=this.__v_isShallow||Fi(e)||Bs(e);e=n?e:Ft(e),nr(e,t)&&(this._rawValue=e,this._value=n?e:Gi(e),this.dep.trigger())}}function vf(i){return Fn(i)?i.value:i}const tv={get:(i,e,t)=>e==="__v_raw"?i:vf(Reflect.get(i,e,t)),set:(i,e,t,n)=>{const s=i[e];return Fn(s)&&!Fn(t)?(s.value=t,!0):Reflect.set(i,e,t,n)}};function Y0(i){return Wr(i)?i:new Proxy(i,tv)}class nv{constructor(e,t,n){this.fn=e,this.setter=t,this._value=void 0,this.dep=new Nd(this),this.__v_isRef=!0,this.deps=void 0,this.depsTail=void 0,this.flags=16,this.globalVersion=Ga-1,this.next=void 0,this.effect=this,this.__v_isReadonly=!t,this.isSSR=n}notify(){if(this.flags|=16,!(this.flags&8)&&Gt!==this)return B0(this,!0),!0}get value(){const e=this.dep.track();return N0(this),e&&(e.version=this.dep.version),this._value}set value(e){this.setter&&this.setter(e)}}function iv(i,e,t=!1){let n,s;return xt(i)?n=i:(n=i.get,s=i.set),new nv(n,s,t)}const Ml={},gc=new WeakMap;let Fr;function sv(i,e=!1,t=Fr){if(t){let n=gc.get(t);n||gc.set(t,n=[]),n.push(i)}}function rv(i,e,t=Ht){const{immediate:n,deep:s,once:r,scheduler:o,augmentJob:a,call:l}=t,c=v=>s?v:Fi(v)||s===!1||s===0?ws(v,1):ws(v);let u,f,d,h,x=!1,p=!1;if(Fn(i)?(f=()=>i.value,x=Fi(i)):Wr(i)?(f=()=>c(i),x=!0):ct(i)?(p=!0,x=i.some(v=>Wr(v)||Fi(v)),f=()=>i.map(v=>{if(Fn(v))return v.value;if(Wr(v))return c(v);if(xt(v))return l?l(v,2):v()})):xt(i)?e?f=l?()=>l(i,2):i:f=()=>{if(d){Fs();try{d()}finally{Ls()}}const v=Fr;Fr=u;try{return l?l(i,3,[h]):i(h)}finally{Fr=v}}:f=ns,e&&s){const v=f,S=s===!0?1/0:s;f=()=>ws(v(),S)}const g=P_(),m=()=>{u.stop(),g&&g.active&&Dd(g.effects,u)};if(r&&e){const v=e;e=(...S)=>{v(...S),m()}}let _=p?new Array(i.length).fill(Ml):Ml;const A=v=>{if(!(!(u.flags&1)||!u.dirty&&!v))if(e){const S=u.run();if(s||x||(p?S.some((y,M)=>nr(y,_[M])):nr(S,_))){d&&d();const y=Fr;Fr=u;try{const M=[S,_===Ml?void 0:p&&_[0]===Ml?[]:_,h];_=S,l?l(e,3,M):e(...M)}finally{Fr=y}}}else u.run()};return a&&a(A),u=new F0(f),u.scheduler=o?()=>o(A,!1):A,h=v=>sv(v,!1,u),d=u.onStop=()=>{const v=gc.get(u);if(v){if(l)l(v,4);else for(const S of v)S();gc.delete(u)}},e?n?A(!0):_=u.run():o?o(A.bind(null,!0),!0):u.run(),m.pause=u.pause.bind(u),m.resume=u.resume.bind(u),m.stop=m,m}function ws(i,e=1/0,t){if(e<=0||!Wt(i)||i.__v_skip||(t=t||new Map,(t.get(i)||0)>=e))return i;if(t.set(i,e),e--,Fn(i))ws(i.value,e,t);else if(ct(i))for(let n=0;n<i.length;n++)ws(i[n],e,t);else if(Nc(i)||Io(i))i.forEach(n=>{ws(n,e,t)});else if(T0(i)){for(const n in i)ws(i[n],e,t);for(const n of Object.getOwnPropertySymbols(i))Object.prototype.propertyIsEnumerable.call(i,n)&&ws(i[n],e,t)}return i}function fl(i,e,t,n){try{return n?i(...n):i()}catch(s){Gc(s,e,t)}}function rs(i,e,t,n){if(xt(i)){const s=fl(i,e,t,n);return s&&M0(s)&&s.catch(r=>{Gc(r,e,t)}),s}if(ct(i)){const s=[];for(let r=0;r<i.length;r++)s.push(rs(i[r],e,t,n));return s}}function Gc(i,e,t,n=!0){const s=e?e.vnode:null,{errorHandler:r,throwUnhandledErrorInProduction:o}=e&&e.appContext.config||Ht;if(e){let a=e.parent;const l=e.proxy,c=`https://vuejs.org/error-reference/#runtime-${t}`;for(;a;){const u=a.ec;if(u){for(let f=0;f<u.length;f++)if(u[f](i,l,c)===!1)return}a=a.parent}if(r){Fs(),fl(r,null,10,[i,l,c]),Ls();return}}ov(i,t,s,n,o)}function ov(i,e,t,n=!0,s=!1){if(s)throw i;console.error(i)}const Hn=[];let qi=-1;const Do=[];let Ys=null,bo=0;const Q0=Promise.resolve();let xc=null;function K0(i){const e=xc||Q0;return i?e.then(this?i.bind(this):i):e}function av(i){let e=qi+1,t=Hn.length;for(;e<t;){const n=e+t>>>1,s=Hn[n],r=Xa(s);r<i||r===i&&s.flags&2?e=n+1:t=n}return e}function Gd(i){if(!(i.flags&1)){const e=Xa(i),t=Hn[Hn.length-1];!t||!(i.flags&2)&&e>=Xa(t)?Hn.push(i):Hn.splice(av(e),0,i),i.flags|=1,j0()}}function j0(){xc||(xc=Q0.then(Z0))}function lv(i){ct(i)?Do.push(...i):Ys&&i.id===-1?Ys.splice(bo+1,0,i):i.flags&1||(Do.push(i),i.flags|=1),j0()}function $h(i,e,t=qi+1){for(;t<Hn.length;t++){const n=Hn[t];if(n&&n.flags&2){if(i&&n.id!==i.uid)continue;Hn.splice(t,1),t--,n.flags&4&&(n.flags&=-2),n(),n.flags&4||(n.flags&=-2)}}}function $0(i){if(Do.length){const e=[...new Set(Do)].sort((t,n)=>Xa(t)-Xa(n));if(Do.length=0,Ys){Ys.push(...e);return}for(Ys=e,bo=0;bo<Ys.length;bo++){const t=Ys[bo];t.flags&4&&(t.flags&=-2),t.flags&8||t(),t.flags&=-2}Ys=null,bo=0}}const Xa=i=>i.id==null?i.flags&2?-1:1/0:i.id;function Z0(i){try{for(qi=0;qi<Hn.length;qi++){const e=Hn[qi];e&&!(e.flags&8)&&(e.flags&4&&(e.flags&=-2),fl(e,e.i,e.i?15:14),e.flags&4||(e.flags&=-2))}}finally{for(;qi<Hn.length;qi++){const e=Hn[qi];e&&(e.flags&=-2)}qi=-1,Hn.length=0,$0(),xc=null,(Hn.length||Do.length)&&Z0()}}let wi=null,J0=null;function _c(i){const e=wi;return wi=i,J0=i&&i.type.__scopeId||null,e}function cv(i,e=wi,t){if(!e||i._n)return i;const n=(...s)=>{n._d&&lp(-1);const r=_c(e);let o;try{o=i(...s)}finally{_c(r),n._d&&lp(1)}return o};return n._n=!0,n._c=!0,n._d=!0,n}function Sr(i,e){if(wi===null)return i;const t=Yc(wi),n=i.dirs||(i.dirs=[]);for(let s=0;s<e.length;s++){let[r,o,a,l=Ht]=e[s];r&&(xt(r)&&(r={mounted:r,updated:r}),r.deep&&ws(o),n.push({dir:r,instance:t,value:o,oldValue:void 0,arg:a,modifiers:l}))}return i}function yr(i,e,t,n){const s=i.dirs,r=e&&e.dirs;for(let o=0;o<s.length;o++){const a=s[o];r&&(a.oldValue=r[o].value);let l=a.dir[n];l&&(Fs(),rs(l,t,8,[i.el,a,i,e]),Ls())}}const uv=Symbol("_vte"),fv=i=>i.__isTeleport,dv=Symbol("_leaveCb");function Wd(i,e){i.shapeFlag&6&&i.component?(i.transition=e,Wd(i.component.subTree,e)):i.shapeFlag&128?(i.ssContent.transition=e.clone(i.ssContent),i.ssFallback.transition=e.clone(i.ssFallback)):i.transition=e}function eg(i){i.ids=[i.ids[0]+i.ids[2]+++"-",0,0]}const vc=new WeakMap;function wa(i,e,t,n,s=!1){if(ct(i)){i.forEach((x,p)=>wa(x,e&&(ct(e)?e[p]:e),t,n,s));return}if(Ra(n)&&!s){n.shapeFlag&512&&n.type.__asyncResolved&&n.component.subTree.component&&wa(i,e,t,n.component.subTree);return}const r=n.shapeFlag&4?Yc(n.component):n.el,o=s?null:r,{i:a,r:l}=i,c=e&&e.r,u=a.refs===Ht?a.refs={}:a.refs,f=a.setupState,d=Ft(f),h=f===Ht?b0:x=>Lt(d,x);if(c!=null&&c!==l){if(Zh(e),gn(c))u[c]=null,h(c)&&(f[c]=null);else if(Fn(c)){c.value=null;const x=e;x.k&&(u[x.k]=null)}}if(xt(l))fl(l,a,12,[o,u]);else{const x=gn(l),p=Fn(l);if(x||p){const g=()=>{if(i.f){const m=x?h(l)?f[l]:u[l]:l.value;if(s)ct(m)&&Dd(m,r);else if(ct(m))m.includes(r)||m.push(r);else if(x)u[l]=[r],h(l)&&(f[l]=u[l]);else{const _=[r];l.value=_,i.k&&(u[i.k]=_)}}else x?(u[l]=o,h(l)&&(f[l]=o)):p&&(l.value=o,i.k&&(u[i.k]=o))};if(o){const m=()=>{g(),vc.delete(i)};m.id=-1,vc.set(i,m),ci(m,t)}else Zh(i),g()}}}function Zh(i){const e=vc.get(i);e&&(e.flags|=8,vc.delete(i))}kc().requestIdleCallback;kc().cancelIdleCallback;const Ra=i=>!!i.type.__asyncLoader,tg=i=>i.type.__isKeepAlive;function hv(i,e){ng(i,"a",e)}function pv(i,e){ng(i,"da",e)}function ng(i,e,t=Gn){const n=i.__wdc||(i.__wdc=()=>{let s=t;for(;s;){if(s.isDeactivated)return;s=s.parent}return i()});if(Wc(e,n,t),t){let s=t.parent;for(;s&&s.parent;)tg(s.parent.vnode)&&mv(n,e,t,s),s=s.parent}}function mv(i,e,t,n){const s=Wc(e,i,n,!0);ig(()=>{Dd(n[e],s)},t)}function Wc(i,e,t=Gn,n=!1){if(t){const s=t[i]||(t[i]=[]),r=e.__weh||(e.__weh=(...o)=>{Fs();const a=dl(t),l=rs(e,t,i,o);return a(),Ls(),l});return n?s.unshift(r):s.push(r),r}}const Ns=i=>(e,t=Gn)=>{(!Ya||i==="sp")&&Wc(i,(...n)=>e(...n),t)},gv=Ns("bm"),Xd=Ns("m"),xv=Ns("bu"),_v=Ns("u"),qd=Ns("bum"),ig=Ns("um"),vv=Ns("sp"),Av=Ns("rtg"),Sv=Ns("rtc");function yv(i,e=Gn){Wc("ec",i,e)}const bv=Symbol.for("v-ndc");function Jh(i,e,t,n){let s;const r=t,o=ct(i);if(o||gn(i)){const a=o&&Wr(i);let l=!1,c=!1;a&&(l=!Fi(i),c=Bs(i),i=Vc(i)),s=new Array(i.length);for(let u=0,f=i.length;u<f;u++)s[u]=e(l?c?Vo(Gi(i[u])):Gi(i[u]):i[u],u,void 0,r)}else if(typeof i=="number"){s=new Array(i);for(let a=0;a<i;a++)s[a]=e(a+1,a,void 0,r)}else if(Wt(i))if(i[Symbol.iterator])s=Array.from(i,(a,l)=>e(a,l,void 0,r));else{const a=Object.keys(i);s=new Array(a.length);for(let l=0,c=a.length;l<c;l++){const u=a[l];s[l]=e(i[u],u,l,r)}}else s=[];return s}const Af=i=>i?Cg(i)?Yc(i):Af(i.parent):null,Ia=Qn(Object.create(null),{$:i=>i,$el:i=>i.vnode.el,$data:i=>i.data,$props:i=>i.props,$attrs:i=>i.attrs,$slots:i=>i.slots,$refs:i=>i.refs,$parent:i=>Af(i.parent),$root:i=>Af(i.root),$host:i=>i.ce,$emit:i=>i.emit,$options:i=>rg(i),$forceUpdate:i=>i.f||(i.f=()=>{Gd(i.update)}),$nextTick:i=>i.n||(i.n=K0.bind(i.proxy)),$watch:i=>Bv.bind(i)}),du=(i,e)=>i!==Ht&&!i.__isScriptSetup&&Lt(i,e),Mv={get({_:i},e){if(e==="__v_skip")return!0;const{ctx:t,setupState:n,data:s,props:r,accessCache:o,type:a,appContext:l}=i;if(e[0]!=="$"){const d=o[e];if(d!==void 0)switch(d){case 1:return n[e];case 2:return s[e];case 4:return t[e];case 3:return r[e]}else{if(du(n,e))return o[e]=1,n[e];if(s!==Ht&&Lt(s,e))return o[e]=2,s[e];if(Lt(r,e))return o[e]=3,r[e];if(t!==Ht&&Lt(t,e))return o[e]=4,t[e];Sf&&(o[e]=0)}}const c=Ia[e];let u,f;if(c)return e==="$attrs"&&Dn(i.attrs,"get",""),c(i);if((u=a.__cssModules)&&(u=u[e]))return u;if(t!==Ht&&Lt(t,e))return o[e]=4,t[e];if(f=l.config.globalProperties,Lt(f,e))return f[e]},set({_:i},e,t){const{data:n,setupState:s,ctx:r}=i;return du(s,e)?(s[e]=t,!0):n!==Ht&&Lt(n,e)?(n[e]=t,!0):Lt(i.props,e)||e[0]==="$"&&e.slice(1)in i?!1:(r[e]=t,!0)},has({_:{data:i,setupState:e,accessCache:t,ctx:n,appContext:s,props:r,type:o}},a){let l;return!!(t[a]||i!==Ht&&a[0]!=="$"&&Lt(i,a)||du(e,a)||Lt(r,a)||Lt(n,a)||Lt(Ia,a)||Lt(s.config.globalProperties,a)||(l=o.__cssModules)&&l[a])},defineProperty(i,e,t){return t.get!=null?i._.accessCache[e]=0:Lt(t,"value")&&this.set(i,e,t.value,null),Reflect.defineProperty(i,e,t)}};function ep(i){return ct(i)?i.reduce((e,t)=>(e[t]=null,e),{}):i}let Sf=!0;function Cv(i){const e=rg(i),t=i.proxy,n=i.ctx;Sf=!1,e.beforeCreate&&tp(e.beforeCreate,i,"bc");const{data:s,computed:r,methods:o,watch:a,provide:l,inject:c,created:u,beforeMount:f,mounted:d,beforeUpdate:h,updated:x,activated:p,deactivated:g,beforeDestroy:m,beforeUnmount:_,destroyed:A,unmounted:v,render:S,renderTracked:y,renderTriggered:M,errorCaptured:E,serverPrefetch:b,expose:C,inheritAttrs:D,components:F,directives:O,filters:z}=e;if(c&&Tv(c,n,null),o)for(const q in o){const G=o[q];xt(G)&&(n[q]=G.bind(t))}if(s){const q=s.call(t,t);Wt(q)&&(i.data=kd(q))}if(Sf=!0,r)for(const q in r){const G=r[q],$=xt(G)?G.bind(t,t):xt(G.get)?G.get.bind(t,t):ns,fe=!xt(G)&&xt(G.set)?G.set.bind(t):ns,Y=Mi({get:$,set:fe});Object.defineProperty(n,q,{enumerable:!0,configurable:!0,get:()=>Y.value,set:we=>Y.value=we})}if(a)for(const q in a)sg(a[q],n,t,q);if(l){const q=xt(l)?l.call(t):l;Reflect.ownKeys(q).forEach(G=>{Pv(G,q[G])})}u&&tp(u,i,"c");function H(q,G){ct(G)?G.forEach($=>q($.bind(t))):G&&q(G.bind(t))}if(H(gv,f),H(Xd,d),H(xv,h),H(_v,x),H(hv,p),H(pv,g),H(yv,E),H(Sv,y),H(Av,M),H(qd,_),H(ig,v),H(vv,b),ct(C))if(C.length){const q=i.exposed||(i.exposed={});C.forEach(G=>{Object.defineProperty(q,G,{get:()=>t[G],set:$=>t[G]=$,enumerable:!0})})}else i.exposed||(i.exposed={});S&&i.render===ns&&(i.render=S),D!=null&&(i.inheritAttrs=D),F&&(i.components=F),O&&(i.directives=O),b&&eg(i)}function Tv(i,e,t=ns){ct(i)&&(i=yf(i));for(const n in i){const s=i[n];let r;Wt(s)?"default"in s?r=rc(s.from||n,s.default,!0):r=rc(s.from||n):r=rc(s),Fn(r)?Object.defineProperty(e,n,{enumerable:!0,configurable:!0,get:()=>r.value,set:o=>r.value=o}):e[n]=r}}function tp(i,e,t){rs(ct(i)?i.map(n=>n.bind(e.proxy)):i.bind(e.proxy),e,t)}function sg(i,e,t,n){let s=n.includes(".")?lg(t,n):()=>t[n];if(gn(i)){const r=e[i];xt(r)&&Da(s,r)}else if(xt(i))Da(s,i.bind(t));else if(Wt(i))if(ct(i))i.forEach(r=>sg(r,e,t,n));else{const r=xt(i.handler)?i.handler.bind(t):e[i.handler];xt(r)&&Da(s,r,i)}}function rg(i){const e=i.type,{mixins:t,extends:n}=e,{mixins:s,optionsCache:r,config:{optionMergeStrategies:o}}=i.appContext,a=r.get(e);let l;return a?l=a:!s.length&&!t&&!n?l=e:(l={},s.length&&s.forEach(c=>Ac(l,c,o,!0)),Ac(l,e,o)),Wt(e)&&r.set(e,l),l}function Ac(i,e,t,n=!1){const{mixins:s,extends:r}=e;r&&Ac(i,r,t,!0),s&&s.forEach(o=>Ac(i,o,t,!0));for(const o in e)if(!(n&&o==="expose")){const a=Ev[o]||t&&t[o];i[o]=a?a(i[o],e[o]):e[o]}return i}const Ev={data:np,props:ip,emits:ip,methods:Sa,computed:Sa,beforeCreate:On,created:On,beforeMount:On,mounted:On,beforeUpdate:On,updated:On,beforeDestroy:On,beforeUnmount:On,destroyed:On,unmounted:On,activated:On,deactivated:On,errorCaptured:On,serverPrefetch:On,components:Sa,directives:Sa,watch:Rv,provide:np,inject:wv};function np(i,e){return e?i?function(){return Qn(xt(i)?i.call(this,this):i,xt(e)?e.call(this,this):e)}:e:i}function wv(i,e){return Sa(yf(i),yf(e))}function yf(i){if(ct(i)){const e={};for(let t=0;t<i.length;t++)e[i[t]]=i[t];return e}return i}function On(i,e){return i?[...new Set([].concat(i,e))]:e}function Sa(i,e){return i?Qn(Object.create(null),i,e):e}function ip(i,e){return i?ct(i)&&ct(e)?[...new Set([...i,...e])]:Qn(Object.create(null),ep(i),ep(e??{})):e}function Rv(i,e){if(!i)return e;if(!e)return i;const t=Qn(Object.create(null),i);for(const n in e)t[n]=On(i[n],e[n]);return t}function og(){return{app:null,config:{isNativeTag:b0,performance:!1,globalProperties:{},optionMergeStrategies:{},errorHandler:void 0,warnHandler:void 0,compilerOptions:{}},mixins:[],components:{},directives:{},provides:Object.create(null),optionsCache:new WeakMap,propsCache:new WeakMap,emitsCache:new WeakMap}}let Iv=0;function Dv(i,e){return function(n,s=null){xt(n)||(n=Qn({},n)),s!=null&&!Wt(s)&&(s=null);const r=og(),o=new WeakSet,a=[];let l=!1;const c=r.app={_uid:Iv++,_component:n,_props:s,_container:null,_context:r,_instance:null,version:pA,get config(){return r.config},set config(u){},use(u,...f){return o.has(u)||(u&&xt(u.install)?(o.add(u),u.install(c,...f)):xt(u)&&(o.add(u),u(c,...f))),c},mixin(u){return r.mixins.includes(u)||r.mixins.push(u),c},component(u,f){return f?(r.components[u]=f,c):r.components[u]},directive(u,f){return f?(r.directives[u]=f,c):r.directives[u]},mount(u,f,d){if(!l){const h=c._ceVNode||is(n,s);return h.appContext=r,d===!0?d="svg":d===!1&&(d=void 0),i(h,u,d),l=!0,c._container=u,u.__vue_app__=c,Yc(h.component)}},onUnmount(u){a.push(u)},unmount(){l&&(rs(a,c._instance,16),i(null,c._container),delete c._container.__vue_app__)},provide(u,f){return r.provides[u]=f,c},runWithContext(u){const f=Po;Po=c;try{return u()}finally{Po=f}}};return c}}let Po=null;function Pv(i,e){if(Gn){let t=Gn.provides;const n=Gn.parent&&Gn.parent.provides;n===t&&(t=Gn.provides=Object.create(n)),t[i]=e}}function rc(i,e,t=!1){const n=lA();if(n||Po){let s=Po?Po._context.provides:n?n.parent==null||n.ce?n.vnode.appContext&&n.vnode.appContext.provides:n.parent.provides:void 0;if(s&&i in s)return s[i];if(arguments.length>1)return t&&xt(e)?e.call(n&&n.proxy):e}}const Fv=Symbol.for("v-scx"),Lv=()=>rc(Fv);function Da(i,e,t){return ag(i,e,t)}function ag(i,e,t=Ht){const{immediate:n,deep:s,flush:r,once:o}=t,a=Qn({},t),l=e&&n||!e&&r!=="post";let c;if(Ya){if(r==="sync"){const h=Lv();c=h.__watcherHandles||(h.__watcherHandles=[])}else if(!l){const h=()=>{};return h.stop=ns,h.resume=ns,h.pause=ns,h}}const u=Gn;a.call=(h,x,p)=>rs(h,u,x,p);let f=!1;r==="post"?a.scheduler=h=>{ci(h,u&&u.suspense)}:r!=="sync"&&(f=!0,a.scheduler=(h,x)=>{x?h():Gd(h)}),a.augmentJob=h=>{e&&(h.flags|=4),f&&(h.flags|=2,u&&(h.id=u.uid,h.i=u))};const d=rv(i,e,a);return Ya&&(c?c.push(d):l&&d()),d}function Bv(i,e,t){const n=this.proxy,s=gn(i)?i.includes(".")?lg(n,i):()=>n[i]:i.bind(n,n);let r;xt(e)?r=e:(r=e.handler,t=e);const o=dl(this),a=ag(s,r.bind(n),t);return o(),a}function lg(i,e){const t=e.split(".");return()=>{let n=i;for(let s=0;s<t.length&&n;s++)n=n[t[s]];return n}}const Uv=(i,e)=>e==="modelValue"||e==="model-value"?i.modelModifiers:i[`${e}Modifiers`]||i[`${ar(e)}Modifiers`]||i[`${hr(e)}Modifiers`];function Ov(i,e,...t){if(i.isUnmounted)return;const n=i.vnode.props||Ht;let s=t;const r=e.startsWith("update:"),o=r&&Uv(n,e.slice(7));o&&(o.trim&&(s=t.map(u=>gn(u)?u.trim():u)),o.number&&(s=t.map(Fd)));let a,l=n[a=au(e)]||n[a=au(ar(e))];!l&&r&&(l=n[a=au(hr(e))]),l&&rs(l,i,6,s);const c=n[a+"Once"];if(c){if(!i.emitted)i.emitted={};else if(i.emitted[a])return;i.emitted[a]=!0,rs(c,i,6,s)}}const Nv=new WeakMap;function cg(i,e,t=!1){const n=t?Nv:e.emitsCache,s=n.get(i);if(s!==void 0)return s;const r=i.emits;let o={},a=!1;if(!xt(i)){const l=c=>{const u=cg(c,e,!0);u&&(a=!0,Qn(o,u))};!t&&e.mixins.length&&e.mixins.forEach(l),i.extends&&l(i.extends),i.mixins&&i.mixins.forEach(l)}return!r&&!a?(Wt(i)&&n.set(i,null),null):(ct(r)?r.forEach(l=>o[l]=null):Qn(o,r),Wt(i)&&n.set(i,o),o)}function Xc(i,e){return!i||!Oc(e)?!1:(e=e.slice(2).replace(/Once$/,""),Lt(i,e[0].toLowerCase()+e.slice(1))||Lt(i,hr(e))||Lt(i,e))}function sp(i){const{type:e,vnode:t,proxy:n,withProxy:s,propsOptions:[r],slots:o,attrs:a,emit:l,render:c,renderCache:u,props:f,data:d,setupState:h,ctx:x,inheritAttrs:p}=i,g=_c(i);let m,_;try{if(t.shapeFlag&4){const v=s||n,S=v;m=Qi(c.call(S,v,u,f,h,d,x)),_=a}else{const v=e;m=Qi(v.length>1?v(f,{attrs:a,slots:o,emit:l}):v(f,null)),_=e.props?a:zv(a)}}catch(v){Pa.length=0,Gc(v,i,1),m=is(lr)}let A=m;if(_&&p!==!1){const v=Object.keys(_),{shapeFlag:S}=A;v.length&&S&7&&(r&&v.some(Id)&&(_=kv(_,r)),A=Go(A,_,!1,!0))}return t.dirs&&(A=Go(A,null,!1,!0),A.dirs=A.dirs?A.dirs.concat(t.dirs):t.dirs),t.transition&&Wd(A,t.transition),m=A,_c(g),m}const zv=i=>{let e;for(const t in i)(t==="class"||t==="style"||Oc(t))&&((e||(e={}))[t]=i[t]);return e},kv=(i,e)=>{const t={};for(const n in i)(!Id(n)||!(n.slice(9)in e))&&(t[n]=i[n]);return t};function Hv(i,e,t){const{props:n,children:s,component:r}=i,{props:o,children:a,patchFlag:l}=e,c=r.emitsOptions;if(e.dirs||e.transition)return!0;if(t&&l>=0){if(l&1024)return!0;if(l&16)return n?rp(n,o,c):!!o;if(l&8){const u=e.dynamicProps;for(let f=0;f<u.length;f++){const d=u[f];if(o[d]!==n[d]&&!Xc(c,d))return!0}}}else return(s||a)&&(!a||!a.$stable)?!0:n===o?!1:n?o?rp(n,o,c):!0:!!o;return!1}function rp(i,e,t){const n=Object.keys(e);if(n.length!==Object.keys(i).length)return!0;for(let s=0;s<n.length;s++){const r=n[s];if(e[r]!==i[r]&&!Xc(t,r))return!0}return!1}function Vv({vnode:i,parent:e},t){for(;e;){const n=e.subTree;if(n.suspense&&n.suspense.activeBranch===i&&(n.el=i.el),n===i)(i=e.vnode).el=t,e=e.parent;else break}}const ug={},fg=()=>Object.create(ug),dg=i=>Object.getPrototypeOf(i)===ug;function Gv(i,e,t,n=!1){const s={},r=fg();i.propsDefaults=Object.create(null),hg(i,e,s,r);for(const o in i.propsOptions[0])o in s||(s[o]=void 0);t?i.props=n?s:$_(s):i.type.props?i.props=s:i.props=r,i.attrs=r}function Wv(i,e,t,n){const{props:s,attrs:r,vnode:{patchFlag:o}}=i,a=Ft(s),[l]=i.propsOptions;let c=!1;if((n||o>0)&&!(o&16)){if(o&8){const u=i.vnode.dynamicProps;for(let f=0;f<u.length;f++){let d=u[f];if(Xc(i.emitsOptions,d))continue;const h=e[d];if(l)if(Lt(r,d))h!==r[d]&&(r[d]=h,c=!0);else{const x=ar(d);s[x]=bf(l,a,x,h,i,!1)}else h!==r[d]&&(r[d]=h,c=!0)}}}else{hg(i,e,s,r)&&(c=!0);let u;for(const f in a)(!e||!Lt(e,f)&&((u=hr(f))===f||!Lt(e,u)))&&(l?t&&(t[f]!==void 0||t[u]!==void 0)&&(s[f]=bf(l,a,f,void 0,i,!0)):delete s[f]);if(r!==a)for(const f in r)(!e||!Lt(e,f))&&(delete r[f],c=!0)}c&&Es(i.attrs,"set","")}function hg(i,e,t,n){const[s,r]=i.propsOptions;let o=!1,a;if(e)for(let l in e){if(Ca(l))continue;const c=e[l];let u;s&&Lt(s,u=ar(l))?!r||!r.includes(u)?t[u]=c:(a||(a={}))[u]=c:Xc(i.emitsOptions,l)||(!(l in n)||c!==n[l])&&(n[l]=c,o=!0)}if(r){const l=Ft(t),c=a||Ht;for(let u=0;u<r.length;u++){const f=r[u];t[f]=bf(s,l,f,c[f],i,!Lt(c,f))}}return o}function bf(i,e,t,n,s,r){const o=i[t];if(o!=null){const a=Lt(o,"default");if(a&&n===void 0){const l=o.default;if(o.type!==Function&&!o.skipFactory&&xt(l)){const{propsDefaults:c}=s;if(t in c)n=c[t];else{const u=dl(s);n=c[t]=l.call(null,e),u()}}else n=l;s.ce&&s.ce._setProp(t,n)}o[0]&&(r&&!a?n=!1:o[1]&&(n===""||n===hr(t))&&(n=!0))}return n}const Xv=new WeakMap;function pg(i,e,t=!1){const n=t?Xv:e.propsCache,s=n.get(i);if(s)return s;const r=i.props,o={},a=[];let l=!1;if(!xt(i)){const u=f=>{l=!0;const[d,h]=pg(f,e,!0);Qn(o,d),h&&a.push(...h)};!t&&e.mixins.length&&e.mixins.forEach(u),i.extends&&u(i.extends),i.mixins&&i.mixins.forEach(u)}if(!r&&!l)return Wt(i)&&n.set(i,Ro),Ro;if(ct(r))for(let u=0;u<r.length;u++){const f=ar(r[u]);op(f)&&(o[f]=Ht)}else if(r)for(const u in r){const f=ar(u);if(op(f)){const d=r[u],h=o[f]=ct(d)||xt(d)?{type:d}:Qn({},d),x=h.type;let p=!1,g=!0;if(ct(x))for(let m=0;m<x.length;++m){const _=x[m],A=xt(_)&&_.name;if(A==="Boolean"){p=!0;break}else A==="String"&&(g=!1)}else p=xt(x)&&x.name==="Boolean";h[0]=p,h[1]=g,(p||Lt(h,"default"))&&a.push(f)}}const c=[o,a];return Wt(i)&&n.set(i,c),c}function op(i){return i[0]!=="$"&&!Ca(i)}const Yd=i=>i==="_"||i==="_ctx"||i==="$stable",Qd=i=>ct(i)?i.map(Qi):[Qi(i)],qv=(i,e,t)=>{if(e._n)return e;const n=cv((...s)=>Qd(e(...s)),t);return n._c=!1,n},mg=(i,e,t)=>{const n=i._ctx;for(const s in i){if(Yd(s))continue;const r=i[s];if(xt(r))e[s]=qv(s,r,n);else if(r!=null){const o=Qd(r);e[s]=()=>o}}},gg=(i,e)=>{const t=Qd(e);i.slots.default=()=>t},xg=(i,e,t)=>{for(const n in e)(t||!Yd(n))&&(i[n]=e[n])},Yv=(i,e,t)=>{const n=i.slots=fg();if(i.vnode.shapeFlag&32){const s=e._;s?(xg(n,e,t),t&&w0(n,"_",s,!0)):mg(e,n)}else e&&gg(i,e)},Qv=(i,e,t)=>{const{vnode:n,slots:s}=i;let r=!0,o=Ht;if(n.shapeFlag&32){const a=e._;a?t&&a===1?r=!1:xg(s,e,t):(r=!e.$stable,mg(e,s)),o=e}else e&&(gg(i,e),o={default:1});if(r)for(const a in s)!Yd(a)&&o[a]==null&&delete s[a]},ci=Jv;function Kv(i){return jv(i)}function jv(i,e){const t=kc();t.__VUE__=!0;const{insert:n,remove:s,patchProp:r,createElement:o,createText:a,createComment:l,setText:c,setElementText:u,parentNode:f,nextSibling:d,setScopeId:h=ns,insertStaticContent:x}=i,p=(U,N,K,R=null,te=null,oe=null,pe=void 0,ie=null,me=!!N.dynamicChildren)=>{if(U===N)return;U&&!da(U,N)&&(R=ue(U),we(U,te,oe,!0),U=null),N.patchFlag===-2&&(me=!1,N.dynamicChildren=null);const{type:se,ref:ve,shapeFlag:I}=N;switch(se){case qc:g(U,N,K,R);break;case lr:m(U,N,K,R);break;case pu:U==null&&_(N,K,R,pe);break;case zi:F(U,N,K,R,te,oe,pe,ie,me);break;default:I&1?S(U,N,K,R,te,oe,pe,ie,me):I&6?O(U,N,K,R,te,oe,pe,ie,me):(I&64||I&128)&&se.process(U,N,K,R,te,oe,pe,ie,me,Ee)}ve!=null&&te?wa(ve,U&&U.ref,oe,N||U,!N):ve==null&&U&&U.ref!=null&&wa(U.ref,null,oe,U,!0)},g=(U,N,K,R)=>{if(U==null)n(N.el=a(N.children),K,R);else{const te=N.el=U.el;N.children!==U.children&&c(te,N.children)}},m=(U,N,K,R)=>{U==null?n(N.el=l(N.children||""),K,R):N.el=U.el},_=(U,N,K,R)=>{[U.el,U.anchor]=x(U.children,N,K,R,U.el,U.anchor)},A=({el:U,anchor:N},K,R)=>{let te;for(;U&&U!==N;)te=d(U),n(U,K,R),U=te;n(N,K,R)},v=({el:U,anchor:N})=>{let K;for(;U&&U!==N;)K=d(U),s(U),U=K;s(N)},S=(U,N,K,R,te,oe,pe,ie,me)=>{if(N.type==="svg"?pe="svg":N.type==="math"&&(pe="mathml"),U==null)y(N,K,R,te,oe,pe,ie,me);else{const se=U.el&&U.el._isVueCE?U.el:null;try{se&&se._beginPatch(),b(U,N,te,oe,pe,ie,me)}finally{se&&se._endPatch()}}},y=(U,N,K,R,te,oe,pe,ie)=>{let me,se;const{props:ve,shapeFlag:I,transition:T,dirs:X}=U;if(me=U.el=o(U.type,oe,ve&&ve.is,ve),I&8?u(me,U.children):I&16&&E(U.children,me,null,R,te,hu(U,oe),pe,ie),X&&yr(U,null,R,"created"),M(me,U,U.scopeId,pe,R),ve){for(const de in ve)de!=="value"&&!Ca(de)&&r(me,de,null,ve[de],oe,R);"value"in ve&&r(me,"value",null,ve.value,oe),(se=ve.onVnodeBeforeMount)&&Xi(se,R,U)}X&&yr(U,null,R,"beforeMount");const re=$v(te,T);re&&T.beforeEnter(me),n(me,N,K),((se=ve&&ve.onVnodeMounted)||re||X)&&ci(()=>{se&&Xi(se,R,U),re&&T.enter(me),X&&yr(U,null,R,"mounted")},te)},M=(U,N,K,R,te)=>{if(K&&h(U,K),R)for(let oe=0;oe<R.length;oe++)h(U,R[oe]);if(te){let oe=te.subTree;if(N===oe||Ag(oe.type)&&(oe.ssContent===N||oe.ssFallback===N)){const pe=te.vnode;M(U,pe,pe.scopeId,pe.slotScopeIds,te.parent)}}},E=(U,N,K,R,te,oe,pe,ie,me=0)=>{for(let se=me;se<U.length;se++){const ve=U[se]=ie?Qs(U[se]):Qi(U[se]);p(null,ve,N,K,R,te,oe,pe,ie)}},b=(U,N,K,R,te,oe,pe)=>{const ie=N.el=U.el;let{patchFlag:me,dynamicChildren:se,dirs:ve}=N;me|=U.patchFlag&16;const I=U.props||Ht,T=N.props||Ht;let X;if(K&&br(K,!1),(X=T.onVnodeBeforeUpdate)&&Xi(X,K,N,U),ve&&yr(N,U,K,"beforeUpdate"),K&&br(K,!0),(I.innerHTML&&T.innerHTML==null||I.textContent&&T.textContent==null)&&u(ie,""),se?C(U.dynamicChildren,se,ie,K,R,hu(N,te),oe):pe||G(U,N,ie,null,K,R,hu(N,te),oe,!1),me>0){if(me&16)D(ie,I,T,K,te);else if(me&2&&I.class!==T.class&&r(ie,"class",null,T.class,te),me&4&&r(ie,"style",I.style,T.style,te),me&8){const re=N.dynamicProps;for(let de=0;de<re.length;de++){const ee=re[de],Ue=I[ee],ye=T[ee];(ye!==Ue||ee==="value")&&r(ie,ee,Ue,ye,te,K)}}me&1&&U.children!==N.children&&u(ie,N.children)}else!pe&&se==null&&D(ie,I,T,K,te);((X=T.onVnodeUpdated)||ve)&&ci(()=>{X&&Xi(X,K,N,U),ve&&yr(N,U,K,"updated")},R)},C=(U,N,K,R,te,oe,pe)=>{for(let ie=0;ie<N.length;ie++){const me=U[ie],se=N[ie],ve=me.el&&(me.type===zi||!da(me,se)||me.shapeFlag&198)?f(me.el):K;p(me,se,ve,null,R,te,oe,pe,!0)}},D=(U,N,K,R,te)=>{if(N!==K){if(N!==Ht)for(const oe in N)!Ca(oe)&&!(oe in K)&&r(U,oe,N[oe],null,te,R);for(const oe in K){if(Ca(oe))continue;const pe=K[oe],ie=N[oe];pe!==ie&&oe!=="value"&&r(U,oe,ie,pe,te,R)}"value"in K&&r(U,"value",N.value,K.value,te)}},F=(U,N,K,R,te,oe,pe,ie,me)=>{const se=N.el=U?U.el:a(""),ve=N.anchor=U?U.anchor:a("");let{patchFlag:I,dynamicChildren:T,slotScopeIds:X}=N;X&&(ie=ie?ie.concat(X):X),U==null?(n(se,K,R),n(ve,K,R),E(N.children||[],K,ve,te,oe,pe,ie,me)):I>0&&I&64&&T&&U.dynamicChildren?(C(U.dynamicChildren,T,K,te,oe,pe,ie),(N.key!=null||te&&N===te.subTree)&&_g(U,N,!0)):G(U,N,K,ve,te,oe,pe,ie,me)},O=(U,N,K,R,te,oe,pe,ie,me)=>{N.slotScopeIds=ie,U==null?N.shapeFlag&512?te.ctx.activate(N,K,R,pe,me):z(N,K,R,te,oe,pe,me):V(U,N,me)},z=(U,N,K,R,te,oe,pe)=>{const ie=U.component=aA(U,R,te);if(tg(U)&&(ie.ctx.renderer=Ee),cA(ie,!1,pe),ie.asyncDep){if(te&&te.registerDep(ie,H,pe),!U.el){const me=ie.subTree=is(lr);m(null,me,N,K),U.placeholder=me.el}}else H(ie,U,N,K,te,oe,pe)},V=(U,N,K)=>{const R=N.component=U.component;if(Hv(U,N,K))if(R.asyncDep&&!R.asyncResolved){q(R,N,K);return}else R.next=N,R.update();else N.el=U.el,R.vnode=N},H=(U,N,K,R,te,oe,pe)=>{const ie=()=>{if(U.isMounted){let{next:I,bu:T,u:X,parent:re,vnode:de}=U;{const k=vg(U);if(k){I&&(I.el=de.el,q(U,I,pe)),k.asyncDep.then(()=>{U.isUnmounted||ie()});return}}let ee=I,Ue;br(U,!1),I?(I.el=de.el,q(U,I,pe)):I=de,T&&sc(T),(Ue=I.props&&I.props.onVnodeBeforeUpdate)&&Xi(Ue,re,I,de),br(U,!0);const ye=sp(U),Xe=U.subTree;U.subTree=ye,p(Xe,ye,f(Xe.el),ue(Xe),U,te,oe),I.el=ye.el,ee===null&&Vv(U,ye.el),X&&ci(X,te),(Ue=I.props&&I.props.onVnodeUpdated)&&ci(()=>Xi(Ue,re,I,de),te)}else{let I;const{el:T,props:X}=N,{bm:re,m:de,parent:ee,root:Ue,type:ye}=U,Xe=Ra(N);br(U,!1),re&&sc(re),!Xe&&(I=X&&X.onVnodeBeforeMount)&&Xi(I,ee,N),br(U,!0);{Ue.ce&&Ue.ce._def.shadowRoot!==!1&&Ue.ce._injectChildStyle(ye);const k=U.subTree=sp(U);p(null,k,K,R,U,te,oe),N.el=k.el}if(de&&ci(de,te),!Xe&&(I=X&&X.onVnodeMounted)){const k=N;ci(()=>Xi(I,ee,k),te)}(N.shapeFlag&256||ee&&Ra(ee.vnode)&&ee.vnode.shapeFlag&256)&&U.a&&ci(U.a,te),U.isMounted=!0,N=K=R=null}};U.scope.on();const me=U.effect=new F0(ie);U.scope.off();const se=U.update=me.run.bind(me),ve=U.job=me.runIfDirty.bind(me);ve.i=U,ve.id=U.uid,me.scheduler=()=>Gd(ve),br(U,!0),se()},q=(U,N,K)=>{N.component=U;const R=U.vnode.props;U.vnode=N,U.next=null,Wv(U,N.props,R,K),Qv(U,N.children,K),Fs(),$h(U),Ls()},G=(U,N,K,R,te,oe,pe,ie,me=!1)=>{const se=U&&U.children,ve=U?U.shapeFlag:0,I=N.children,{patchFlag:T,shapeFlag:X}=N;if(T>0){if(T&128){fe(se,I,K,R,te,oe,pe,ie,me);return}else if(T&256){$(se,I,K,R,te,oe,pe,ie,me);return}}X&8?(ve&16&&ne(se,te,oe),I!==se&&u(K,I)):ve&16?X&16?fe(se,I,K,R,te,oe,pe,ie,me):ne(se,te,oe,!0):(ve&8&&u(K,""),X&16&&E(I,K,R,te,oe,pe,ie,me))},$=(U,N,K,R,te,oe,pe,ie,me)=>{U=U||Ro,N=N||Ro;const se=U.length,ve=N.length,I=Math.min(se,ve);let T;for(T=0;T<I;T++){const X=N[T]=me?Qs(N[T]):Qi(N[T]);p(U[T],X,K,null,te,oe,pe,ie,me)}se>ve?ne(U,te,oe,!0,!1,I):E(N,K,R,te,oe,pe,ie,me,I)},fe=(U,N,K,R,te,oe,pe,ie,me)=>{let se=0;const ve=N.length;let I=U.length-1,T=ve-1;for(;se<=I&&se<=T;){const X=U[se],re=N[se]=me?Qs(N[se]):Qi(N[se]);if(da(X,re))p(X,re,K,null,te,oe,pe,ie,me);else break;se++}for(;se<=I&&se<=T;){const X=U[I],re=N[T]=me?Qs(N[T]):Qi(N[T]);if(da(X,re))p(X,re,K,null,te,oe,pe,ie,me);else break;I--,T--}if(se>I){if(se<=T){const X=T+1,re=X<ve?N[X].el:R;for(;se<=T;)p(null,N[se]=me?Qs(N[se]):Qi(N[se]),K,re,te,oe,pe,ie,me),se++}}else if(se>T)for(;se<=I;)we(U[se],te,oe,!0),se++;else{const X=se,re=se,de=new Map;for(se=re;se<=T;se++){const Re=N[se]=me?Qs(N[se]):Qi(N[se]);Re.key!=null&&de.set(Re.key,se)}let ee,Ue=0;const ye=T-re+1;let Xe=!1,k=0;const Z=new Array(ye);for(se=0;se<ye;se++)Z[se]=0;for(se=X;se<=I;se++){const Re=U[se];if(Ue>=ye){we(Re,te,oe,!0);continue}let Be;if(Re.key!=null)Be=de.get(Re.key);else for(ee=re;ee<=T;ee++)if(Z[ee-re]===0&&da(Re,N[ee])){Be=ee;break}Be===void 0?we(Re,te,oe,!0):(Z[Be-re]=se+1,Be>=k?k=Be:Xe=!0,p(Re,N[Be],K,null,te,oe,pe,ie,me),Ue++)}const xe=Xe?Zv(Z):Ro;for(ee=xe.length-1,se=ye-1;se>=0;se--){const Re=re+se,Be=N[Re],Fe=N[Re+1],je=Re+1<ve?Fe.el||Fe.placeholder:R;Z[se]===0?p(null,Be,K,je,te,oe,pe,ie,me):Xe&&(ee<0||se!==xe[ee]?Y(Be,K,je,2):ee--)}}},Y=(U,N,K,R,te=null)=>{const{el:oe,type:pe,transition:ie,children:me,shapeFlag:se}=U;if(se&6){Y(U.component.subTree,N,K,R);return}if(se&128){U.suspense.move(N,K,R);return}if(se&64){pe.move(U,N,K,Ee);return}if(pe===zi){n(oe,N,K);for(let I=0;I<me.length;I++)Y(me[I],N,K,R);n(U.anchor,N,K);return}if(pe===pu){A(U,N,K);return}if(R!==2&&se&1&&ie)if(R===0)ie.beforeEnter(oe),n(oe,N,K),ci(()=>ie.enter(oe),te);else{const{leave:I,delayLeave:T,afterLeave:X}=ie,re=()=>{U.ctx.isUnmounted?s(oe):n(oe,N,K)},de=()=>{oe._isLeaving&&oe[dv](!0),I(oe,()=>{re(),X&&X()})};T?T(oe,re,de):de()}else n(oe,N,K)},we=(U,N,K,R=!1,te=!1)=>{const{type:oe,props:pe,ref:ie,children:me,dynamicChildren:se,shapeFlag:ve,patchFlag:I,dirs:T,cacheIndex:X}=U;if(I===-2&&(te=!1),ie!=null&&(Fs(),wa(ie,null,K,U,!0),Ls()),X!=null&&(N.renderCache[X]=void 0),ve&256){N.ctx.deactivate(U);return}const re=ve&1&&T,de=!Ra(U);let ee;if(de&&(ee=pe&&pe.onVnodeBeforeUnmount)&&Xi(ee,N,U),ve&6)We(U.component,K,R);else{if(ve&128){U.suspense.unmount(K,R);return}re&&yr(U,null,N,"beforeUnmount"),ve&64?U.type.remove(U,N,K,Ee,R):se&&!se.hasOnce&&(oe!==zi||I>0&&I&64)?ne(se,N,K,!1,!0):(oe===zi&&I&384||!te&&ve&16)&&ne(me,N,K),R&&ze(U)}(de&&(ee=pe&&pe.onVnodeUnmounted)||re)&&ci(()=>{ee&&Xi(ee,N,U),re&&yr(U,null,N,"unmounted")},K)},ze=U=>{const{type:N,el:K,anchor:R,transition:te}=U;if(N===zi){ke(K,R);return}if(N===pu){v(U);return}const oe=()=>{s(K),te&&!te.persisted&&te.afterLeave&&te.afterLeave()};if(U.shapeFlag&1&&te&&!te.persisted){const{leave:pe,delayLeave:ie}=te,me=()=>pe(K,oe);ie?ie(U.el,oe,me):me()}else oe()},ke=(U,N)=>{let K;for(;U!==N;)K=d(U),s(U),U=K;s(N)},We=(U,N,K)=>{const{bum:R,scope:te,job:oe,subTree:pe,um:ie,m:me,a:se}=U;ap(me),ap(se),R&&sc(R),te.stop(),oe&&(oe.flags|=8,we(pe,U,N,K)),ie&&ci(ie,N),ci(()=>{U.isUnmounted=!0},N)},ne=(U,N,K,R=!1,te=!1,oe=0)=>{for(let pe=oe;pe<U.length;pe++)we(U[pe],N,K,R,te)},ue=U=>{if(U.shapeFlag&6)return ue(U.component.subTree);if(U.shapeFlag&128)return U.suspense.next();const N=d(U.anchor||U.el),K=N&&N[uv];return K?d(K):N};let Se=!1;const he=(U,N,K)=>{U==null?N._vnode&&we(N._vnode,null,null,!0):p(N._vnode||null,U,N,null,null,null,K),N._vnode=U,Se||(Se=!0,$h(),$0(),Se=!1)},Ee={p,um:we,m:Y,r:ze,mt:z,mc:E,pc:G,pbc:C,n:ue,o:i};return{render:he,hydrate:void 0,createApp:Dv(he)}}function hu({type:i,props:e},t){return t==="svg"&&i==="foreignObject"||t==="mathml"&&i==="annotation-xml"&&e&&e.encoding&&e.encoding.includes("html")?void 0:t}function br({effect:i,job:e},t){t?(i.flags|=32,e.flags|=4):(i.flags&=-33,e.flags&=-5)}function $v(i,e){return(!i||i&&!i.pendingBranch)&&e&&!e.persisted}function _g(i,e,t=!1){const n=i.children,s=e.children;if(ct(n)&&ct(s))for(let r=0;r<n.length;r++){const o=n[r];let a=s[r];a.shapeFlag&1&&!a.dynamicChildren&&((a.patchFlag<=0||a.patchFlag===32)&&(a=s[r]=Qs(s[r]),a.el=o.el),!t&&a.patchFlag!==-2&&_g(o,a)),a.type===qc&&a.patchFlag!==-1&&(a.el=o.el),a.type===lr&&!a.el&&(a.el=o.el)}}function Zv(i){const e=i.slice(),t=[0];let n,s,r,o,a;const l=i.length;for(n=0;n<l;n++){const c=i[n];if(c!==0){if(s=t[t.length-1],i[s]<c){e[n]=s,t.push(n);continue}for(r=0,o=t.length-1;r<o;)a=r+o>>1,i[t[a]]<c?r=a+1:o=a;c<i[t[r]]&&(r>0&&(e[n]=t[r-1]),t[r]=n)}}for(r=t.length,o=t[r-1];r-- >0;)t[r]=o,o=e[o];return t}function vg(i){const e=i.subTree.component;if(e)return e.asyncDep&&!e.asyncResolved?e:vg(e)}function ap(i){if(i)for(let e=0;e<i.length;e++)i[e].flags|=8}const Ag=i=>i.__isSuspense;function Jv(i,e){e&&e.pendingBranch?ct(i)?e.effects.push(...i):e.effects.push(i):lv(i)}const zi=Symbol.for("v-fgt"),qc=Symbol.for("v-txt"),lr=Symbol.for("v-cmt"),pu=Symbol.for("v-stc"),Pa=[];let hi=null;function jt(i=!1){Pa.push(hi=i?null:[])}function eA(){Pa.pop(),hi=Pa[Pa.length-1]||null}let qa=1;function lp(i,e=!1){qa+=i,i<0&&hi&&e&&(hi.hasOnce=!0)}function Sg(i){return i.dynamicChildren=qa>0?hi||Ro:null,eA(),qa>0&&hi&&hi.push(i),i}function on(i,e,t,n,s,r){return Sg(De(i,e,t,n,s,r,!0))}function yg(i,e,t,n,s){return Sg(is(i,e,t,n,s,!0))}function bg(i){return i?i.__v_isVNode===!0:!1}function da(i,e){return i.type===e.type&&i.key===e.key}const Mg=({key:i})=>i??null,oc=({ref:i,ref_key:e,ref_for:t})=>(typeof i=="number"&&(i=""+i),i!=null?gn(i)||Fn(i)||xt(i)?{i:wi,r:i,k:e,f:!!t}:i:null);function De(i,e=null,t=null,n=0,s=null,r=i===zi?0:1,o=!1,a=!1){const l={__v_isVNode:!0,__v_skip:!0,type:i,props:e,key:e&&Mg(e),ref:e&&oc(e),scopeId:J0,slotScopeIds:null,children:t,component:null,suspense:null,ssContent:null,ssFallback:null,dirs:null,transition:null,el:null,anchor:null,target:null,targetStart:null,targetAnchor:null,staticCount:0,shapeFlag:r,patchFlag:n,dynamicProps:s,dynamicChildren:null,appContext:null,ctx:wi};return a?(Kd(l,t),r&128&&i.normalize(l)):t&&(l.shapeFlag|=gn(t)?8:16),qa>0&&!o&&hi&&(l.patchFlag>0||r&6)&&l.patchFlag!==32&&hi.push(l),l}const is=tA;function tA(i,e=null,t=null,n=0,s=null,r=!1){if((!i||i===bv)&&(i=lr),bg(i)){const a=Go(i,e,!0);return t&&Kd(a,t),qa>0&&!r&&hi&&(a.shapeFlag&6?hi[hi.indexOf(i)]=a:hi.push(a)),a.patchFlag=-2,a}if(hA(i)&&(i=i.__vccOpts),e){e=nA(e);let{class:a,style:l}=e;a&&!gn(a)&&(e.class=$i(a)),Wt(l)&&(Vd(l)&&!ct(l)&&(l=Qn({},l)),e.style=Ld(l))}const o=gn(i)?1:Ag(i)?128:fv(i)?64:Wt(i)?4:xt(i)?2:0;return De(i,e,t,n,s,o,r,!0)}function nA(i){return i?Vd(i)||dg(i)?Qn({},i):i:null}function Go(i,e,t=!1,n=!1){const{props:s,ref:r,patchFlag:o,children:a,transition:l}=i,c=e?sA(s||{},e):s,u={__v_isVNode:!0,__v_skip:!0,type:i.type,props:c,key:c&&Mg(c),ref:e&&e.ref?t&&r?ct(r)?r.concat(oc(e)):[r,oc(e)]:oc(e):r,scopeId:i.scopeId,slotScopeIds:i.slotScopeIds,children:a,target:i.target,targetStart:i.targetStart,targetAnchor:i.targetAnchor,staticCount:i.staticCount,shapeFlag:i.shapeFlag,patchFlag:e&&i.type!==zi?o===-1?16:o|16:o,dynamicProps:i.dynamicProps,dynamicChildren:i.dynamicChildren,appContext:i.appContext,dirs:i.dirs,transition:l,component:i.component,suspense:i.suspense,ssContent:i.ssContent&&Go(i.ssContent),ssFallback:i.ssFallback&&Go(i.ssFallback),placeholder:i.placeholder,el:i.el,anchor:i.anchor,ctx:i.ctx,ce:i.ce};return l&&n&&Wd(u,l.clone(u)),u}function iA(i=" ",e=0){return is(qc,null,i,e)}function kn(i="",e=!1){return e?(jt(),yg(lr,null,i)):is(lr,null,i)}function Qi(i){return i==null||typeof i=="boolean"?is(lr):ct(i)?is(zi,null,i.slice()):bg(i)?Qs(i):is(qc,null,String(i))}function Qs(i){return i.el===null&&i.patchFlag!==-1||i.memo?i:Go(i)}function Kd(i,e){let t=0;const{shapeFlag:n}=i;if(e==null)e=null;else if(ct(e))t=16;else if(typeof e=="object")if(n&65){const s=e.default;s&&(s._c&&(s._d=!1),Kd(i,s()),s._c&&(s._d=!0));return}else{t=32;const s=e._;!s&&!dg(e)?e._ctx=wi:s===3&&wi&&(wi.slots._===1?e._=1:(e._=2,i.patchFlag|=1024))}else xt(e)?(e={default:e,_ctx:wi},t=32):(e=String(e),n&64?(t=16,e=[iA(e)]):t=8);i.children=e,i.shapeFlag|=t}function sA(...i){const e={};for(let t=0;t<i.length;t++){const n=i[t];for(const s in n)if(s==="class")e.class!==n.class&&(e.class=$i([e.class,n.class]));else if(s==="style")e.style=Ld([e.style,n.style]);else if(Oc(s)){const r=e[s],o=n[s];o&&r!==o&&!(ct(r)&&r.includes(o))&&(e[s]=r?[].concat(r,o):o)}else s!==""&&(e[s]=n[s])}return e}function Xi(i,e,t,n=null){rs(i,e,7,[t,n])}const rA=og();let oA=0;function aA(i,e,t){const n=i.type,s=(e?e.appContext:i.appContext)||rA,r={uid:oA++,vnode:i,type:n,parent:e,appContext:s,root:null,next:null,subTree:null,effect:null,update:null,job:null,scope:new D_(!0),render:null,proxy:null,exposed:null,exposeProxy:null,withProxy:null,provides:e?e.provides:Object.create(s.provides),ids:e?e.ids:["",0,0],accessCache:null,renderCache:[],components:null,directives:null,propsOptions:pg(n,s),emitsOptions:cg(n,s),emit:null,emitted:null,propsDefaults:Ht,inheritAttrs:n.inheritAttrs,ctx:Ht,data:Ht,props:Ht,attrs:Ht,slots:Ht,refs:Ht,setupState:Ht,setupContext:null,suspense:t,suspenseId:t?t.pendingId:0,asyncDep:null,asyncResolved:!1,isMounted:!1,isUnmounted:!1,isDeactivated:!1,bc:null,c:null,bm:null,m:null,bu:null,u:null,um:null,bum:null,da:null,a:null,rtg:null,rtc:null,ec:null,sp:null};return r.ctx={_:r},r.root=e?e.root:r,r.emit=Ov.bind(null,r),i.ce&&i.ce(r),r}let Gn=null;const lA=()=>Gn||wi;let Sc,Mf;{const i=kc(),e=(t,n)=>{let s;return(s=i[t])||(s=i[t]=[]),s.push(n),r=>{s.length>1?s.forEach(o=>o(r)):s[0](r)}};Sc=e("__VUE_INSTANCE_SETTERS__",t=>Gn=t),Mf=e("__VUE_SSR_SETTERS__",t=>Ya=t)}const dl=i=>{const e=Gn;return Sc(i),i.scope.on(),()=>{i.scope.off(),Sc(e)}},cp=()=>{Gn&&Gn.scope.off(),Sc(null)};function Cg(i){return i.vnode.shapeFlag&4}let Ya=!1;function cA(i,e=!1,t=!1){e&&Mf(e);const{props:n,children:s}=i.vnode,r=Cg(i);Gv(i,n,r,e),Yv(i,s,t||e);const o=r?uA(i,e):void 0;return e&&Mf(!1),o}function uA(i,e){const t=i.type;i.accessCache=Object.create(null),i.proxy=new Proxy(i.ctx,Mv);const{setup:n}=t;if(n){Fs();const s=i.setupContext=n.length>1?dA(i):null,r=dl(i),o=fl(n,i,0,[i.props,s]),a=M0(o);if(Ls(),r(),(a||i.sp)&&!Ra(i)&&eg(i),a){if(o.then(cp,cp),e)return o.then(l=>{up(i,l)}).catch(l=>{Gc(l,i,0)});i.asyncDep=o}else up(i,o)}else Tg(i)}function up(i,e,t){xt(e)?i.type.__ssrInlineRender?i.ssrRender=e:i.render=e:Wt(e)&&(i.setupState=Y0(e)),Tg(i)}function Tg(i,e,t){const n=i.type;i.render||(i.render=n.render||ns);{const s=dl(i);Fs();try{Cv(i)}finally{Ls(),s()}}}const fA={get(i,e){return Dn(i,"get",""),i[e]}};function dA(i){const e=t=>{i.exposed=t||{}};return{attrs:new Proxy(i.attrs,fA),slots:i.slots,emit:i.emit,expose:e}}function Yc(i){return i.exposed?i.exposeProxy||(i.exposeProxy=new Proxy(Y0(Z_(i.exposed)),{get(e,t){if(t in e)return e[t];if(t in Ia)return Ia[t](i)},has(e,t){return t in e||t in Ia}})):i.proxy}function hA(i){return xt(i)&&"__vccOpts"in i}const Mi=(i,e)=>iv(i,e,Ya),pA="3.5.25";let Cf;const fp=typeof window<"u"&&window.trustedTypes;if(fp)try{Cf=fp.createPolicy("vue",{createHTML:i=>i})}catch{}const Eg=Cf?i=>Cf.createHTML(i):i=>i,mA="http://www.w3.org/2000/svg",gA="http://www.w3.org/1998/Math/MathML",bs=typeof document<"u"?document:null,dp=bs&&bs.createElement("template"),xA={insert:(i,e,t)=>{e.insertBefore(i,t||null)},remove:i=>{const e=i.parentNode;e&&e.removeChild(i)},createElement:(i,e,t,n)=>{const s=e==="svg"?bs.createElementNS(mA,i):e==="mathml"?bs.createElementNS(gA,i):t?bs.createElement(i,{is:t}):bs.createElement(i);return i==="select"&&n&&n.multiple!=null&&s.setAttribute("multiple",n.multiple),s},createText:i=>bs.createTextNode(i),createComment:i=>bs.createComment(i),setText:(i,e)=>{i.nodeValue=e},setElementText:(i,e)=>{i.textContent=e},parentNode:i=>i.parentNode,nextSibling:i=>i.nextSibling,querySelector:i=>bs.querySelector(i),setScopeId(i,e){i.setAttribute(e,"")},insertStaticContent(i,e,t,n,s,r){const o=t?t.previousSibling:e.lastChild;if(s&&(s===r||s.nextSibling))for(;e.insertBefore(s.cloneNode(!0),t),!(s===r||!(s=s.nextSibling)););else{dp.innerHTML=Eg(n==="svg"?`<svg>${i}</svg>`:n==="mathml"?`<math>${i}</math>`:i);const a=dp.content;if(n==="svg"||n==="mathml"){const l=a.firstChild;for(;l.firstChild;)a.appendChild(l.firstChild);a.removeChild(l)}e.insertBefore(a,t)}return[o?o.nextSibling:e.firstChild,t?t.previousSibling:e.lastChild]}},_A=Symbol("_vtc");function vA(i,e,t){const n=i[_A];n&&(e=(e?[e,...n]:[...n]).join(" ")),e==null?i.removeAttribute("class"):t?i.setAttribute("class",e):i.className=e}const hp=Symbol("_vod"),AA=Symbol("_vsh"),SA=Symbol(""),yA=/(?:^|;)\s*display\s*:/;function bA(i,e,t){const n=i.style,s=gn(t);let r=!1;if(t&&!s){if(e)if(gn(e))for(const o of e.split(";")){const a=o.slice(0,o.indexOf(":")).trim();t[a]==null&&ac(n,a,"")}else for(const o in e)t[o]==null&&ac(n,o,"");for(const o in t)o==="display"&&(r=!0),ac(n,o,t[o])}else if(s){if(e!==t){const o=n[SA];o&&(t+=";"+o),n.cssText=t,r=yA.test(t)}}else e&&i.removeAttribute("style");hp in i&&(i[hp]=r?n.display:"",i[AA]&&(n.display="none"))}const pp=/\s*!important$/;function ac(i,e,t){if(ct(t))t.forEach(n=>ac(i,e,n));else if(t==null&&(t=""),e.startsWith("--"))i.setProperty(e,t);else{const n=MA(i,e);pp.test(t)?i.setProperty(hr(n),t.replace(pp,""),"important"):i[n]=t}}const mp=["Webkit","Moz","ms"],mu={};function MA(i,e){const t=mu[e];if(t)return t;let n=ar(e);if(n!=="filter"&&n in i)return mu[e]=n;n=E0(n);for(let s=0;s<mp.length;s++){const r=mp[s]+n;if(r in i)return mu[e]=r}return e}const gp="http://www.w3.org/1999/xlink";function xp(i,e,t,n,s,r=R_(e)){n&&e.startsWith("xlink:")?t==null?i.removeAttributeNS(gp,e.slice(6,e.length)):i.setAttributeNS(gp,e,t):t==null||r&&!R0(t)?i.removeAttribute(e):i.setAttribute(e,r?"":ss(t)?String(t):t)}function _p(i,e,t,n,s){if(e==="innerHTML"||e==="textContent"){t!=null&&(i[e]=e==="innerHTML"?Eg(t):t);return}const r=i.tagName;if(e==="value"&&r!=="PROGRESS"&&!r.includes("-")){const a=r==="OPTION"?i.getAttribute("value")||"":i.value,l=t==null?i.type==="checkbox"?"on":"":String(t);(a!==l||!("_value"in i))&&(i.value=l),t==null&&i.removeAttribute(e),i._value=t;return}let o=!1;if(t===""||t==null){const a=typeof i[e];a==="boolean"?t=R0(t):t==null&&a==="string"?(t="",o=!0):a==="number"&&(t=0,o=!0)}try{i[e]=t}catch{}o&&i.removeAttribute(s||e)}function Or(i,e,t,n){i.addEventListener(e,t,n)}function CA(i,e,t,n){i.removeEventListener(e,t,n)}const vp=Symbol("_vei");function TA(i,e,t,n,s=null){const r=i[vp]||(i[vp]={}),o=r[e];if(n&&o)o.value=n;else{const[a,l]=EA(e);if(n){const c=r[e]=IA(n,s);Or(i,a,c,l)}else o&&(CA(i,a,o,l),r[e]=void 0)}}const Ap=/(?:Once|Passive|Capture)$/;function EA(i){let e;if(Ap.test(i)){e={};let n;for(;n=i.match(Ap);)i=i.slice(0,i.length-n[0].length),e[n[0].toLowerCase()]=!0}return[i[2]===":"?i.slice(3):hr(i.slice(2)),e]}let gu=0;const wA=Promise.resolve(),RA=()=>gu||(wA.then(()=>gu=0),gu=Date.now());function IA(i,e){const t=n=>{if(!n._vts)n._vts=Date.now();else if(n._vts<=t.attached)return;rs(DA(n,t.value),e,5,[n])};return t.value=i,t.attached=RA(),t}function DA(i,e){if(ct(e)){const t=i.stopImmediatePropagation;return i.stopImmediatePropagation=()=>{t.call(i),i._stopped=!0},e.map(n=>s=>!s._stopped&&n&&n(s))}else return e}const Sp=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&i.charCodeAt(2)>96&&i.charCodeAt(2)<123,PA=(i,e,t,n,s,r)=>{const o=s==="svg";e==="class"?vA(i,n,o):e==="style"?bA(i,t,n):Oc(e)?Id(e)||TA(i,e,t,n,r):(e[0]==="."?(e=e.slice(1),!0):e[0]==="^"?(e=e.slice(1),!1):FA(i,e,n,o))?(_p(i,e,n),!i.tagName.includes("-")&&(e==="value"||e==="checked"||e==="selected")&&xp(i,e,n,o,r,e!=="value")):i._isVueCE&&(/[A-Z]/.test(e)||!gn(n))?_p(i,ar(e),n,r,e):(e==="true-value"?i._trueValue=n:e==="false-value"&&(i._falseValue=n),xp(i,e,n,o))};function FA(i,e,t,n){if(n)return!!(e==="innerHTML"||e==="textContent"||e in i&&Sp(e)&&xt(t));if(e==="spellcheck"||e==="draggable"||e==="translate"||e==="autocorrect"||e==="sandbox"&&i.tagName==="IFRAME"||e==="form"||e==="list"&&i.tagName==="INPUT"||e==="type"&&i.tagName==="TEXTAREA")return!1;if(e==="width"||e==="height"){const s=i.tagName;if(s==="IMG"||s==="VIDEO"||s==="CANVAS"||s==="SOURCE")return!1}return Sp(e)&&gn(t)?!1:e in i}const yc=i=>{const e=i.props["onUpdate:modelValue"]||!1;return ct(e)?t=>sc(e,t):e};function LA(i){i.target.composing=!0}function yp(i){const e=i.target;e.composing&&(e.composing=!1,e.dispatchEvent(new Event("input")))}const Fo=Symbol("_assign");function bp(i,e,t){return e&&(i=i.trim()),t&&(i=Fd(i)),i}const ha={created(i,{modifiers:{lazy:e,trim:t,number:n}},s){i[Fo]=yc(s);const r=n||s.props&&s.props.type==="number";Or(i,e?"change":"input",o=>{o.target.composing||i[Fo](bp(i.value,t,r))}),(t||r)&&Or(i,"change",()=>{i.value=bp(i.value,t,r)}),e||(Or(i,"compositionstart",LA),Or(i,"compositionend",yp),Or(i,"change",yp))},mounted(i,{value:e}){i.value=e??""},beforeUpdate(i,{value:e,oldValue:t,modifiers:{lazy:n,trim:s,number:r}},o){if(i[Fo]=yc(o),i.composing)return;const a=(r||i.type==="number")&&!/^0\d/.test(i.value)?Fd(i.value):i.value,l=e??"";a!==l&&(document.activeElement===i&&i.type!=="range"&&(n&&e===t||s&&i.value.trim()===l)||(i.value=l))}},Mp={deep:!0,created(i,e,t){i[Fo]=yc(t),Or(i,"change",()=>{const n=i._modelValue,s=BA(i),r=i.checked,o=i[Fo];if(ct(n)){const a=I0(n,s),l=a!==-1;if(r&&!l)o(n.concat(s));else if(!r&&l){const c=[...n];c.splice(a,1),o(c)}}else if(Nc(n)){const a=new Set(n);r?a.add(s):a.delete(s),o(a)}else o(wg(i,r))})},mounted:Cp,beforeUpdate(i,e,t){i[Fo]=yc(t),Cp(i,e,t)}};function Cp(i,{value:e,oldValue:t},n){i._modelValue=e;let s;if(ct(e))s=I0(e,n.props.value)>-1;else if(Nc(e))s=e.has(n.props.value);else{if(e===t)return;s=Hc(e,wg(i,!0))}i.checked!==s&&(i.checked=s)}function BA(i){return"_value"in i?i._value:i.value}function wg(i,e){const t=e?"_trueValue":"_falseValue";return t in i?i[t]:e}const UA=["ctrl","shift","alt","meta"],OA={stop:i=>i.stopPropagation(),prevent:i=>i.preventDefault(),self:i=>i.target!==i.currentTarget,ctrl:i=>!i.ctrlKey,shift:i=>!i.shiftKey,alt:i=>!i.altKey,meta:i=>!i.metaKey,left:i=>"button"in i&&i.button!==0,middle:i=>"button"in i&&i.button!==1,right:i=>"button"in i&&i.button!==2,exact:(i,e)=>UA.some(t=>i[`${t}Key`]&&!e.includes(t))},Rt=(i,e)=>{const t=i._withMods||(i._withMods={}),n=e.join(".");return t[n]||(t[n]=((s,...r)=>{for(let o=0;o<e.length;o++){const a=OA[e[o]];if(a&&a(s,e))return}return i(s,...r)}))},NA={esc:"escape",space:" ",up:"arrow-up",left:"arrow-left",right:"arrow-right",down:"arrow-down",delete:"backspace"},zA=(i,e)=>{const t=i._withKeys||(i._withKeys={}),n=e.join(".");return t[n]||(t[n]=(s=>{if(!("key"in s))return;const r=hr(s.key);if(e.some(o=>o===r||NA[o]===r))return i(s)}))},kA=Qn({patchProp:PA},xA);let Tp;function HA(){return Tp||(Tp=Kv(kA))}const VA=((...i)=>{const e=HA().createApp(...i),{mount:t}=e;return e.mount=n=>{const s=WA(n);if(!s)return;const r=e._component;!xt(r)&&!r.render&&!r.template&&(r.template=s.innerHTML),s.nodeType===1&&(s.textContent="");const o=t(s,!1,GA(s));return s instanceof Element&&(s.removeAttribute("v-cloak"),s.setAttribute("data-v-app","")),o},e});function GA(i){if(i instanceof SVGElement)return"svg";if(typeof MathMLElement=="function"&&i instanceof MathMLElement)return"mathml"}function WA(i){return gn(i)?document.querySelector(i):i}const jd="181",io={ROTATE:0,DOLLY:1,PAN:2},so={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},XA=0,Ep=1,qA=2,Rg=1,YA=2,Ss=3,os=0,Jn=1,Ei=2,Ds=0,ir=1,wp=2,Rp=3,Ip=4,Ig=5,Nr=100,QA=101,KA=102,jA=103,$A=104,ZA=200,JA=201,eS=202,tS=203,Qa=204,Ka=205,nS=206,iS=207,sS=208,rS=209,oS=210,aS=211,lS=212,cS=213,uS=214,Tf=0,Ef=1,wf=2,Wo=3,Rf=4,If=5,Df=6,Pf=7,Dg=0,fS=1,dS=2,sr=0,hS=1,pS=2,mS=3,gS=4,xS=5,_S=6,vS=7,Pg=300,Xo=301,qo=302,Ff=303,Lf=304,Qc=306,Bf=1e3,Is=1001,Uf=1002,xi=1003,AS=1004,Cl=1005,Ri=1006,xu=1007,kr=1008,as=1009,Fg=1010,Lg=1011,ja=1012,$d=1013,Ii=1014,Hi=1015,jr=1016,Zd=1017,Jd=1018,$a=1020,Bg=35902,Ug=35899,Og=1021,Ng=1022,Xn=1023,Yo=1026,Za=1027,zg=1028,Kc=1029,eh=1030,th=1031,Lo=1033,lc=33776,cc=33777,uc=33778,fc=33779,Of=35840,Nf=35841,zf=35842,kf=35843,Hf=36196,Vf=37492,Gf=37496,Wf=37808,Xf=37809,qf=37810,Yf=37811,Qf=37812,Kf=37813,jf=37814,$f=37815,Zf=37816,Jf=37817,ed=37818,td=37819,nd=37820,id=37821,sd=36492,rd=36494,od=36495,ad=36283,ld=36284,cd=36285,ud=36286,SS=3200,yS=3201,bS=0,MS=1,js="",Ci="srgb",Qo="srgb-linear",bc="linear",Ot="srgb",ro=7680,Dp=519,CS=512,TS=513,ES=514,kg=515,wS=516,RS=517,IS=518,DS=519,Pp=35044,PS=35048,Fp="300 es",Zi=2e3,Mc=2001;function Hg(i){for(let e=i.length-1;e>=0;--e)if(i[e]>=65535)return!0;return!1}function Cc(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function FS(){const i=Cc("canvas");return i.style.display="block",i}const Lp={};function Bp(...i){const e="THREE."+i.shift();console.log(e,...i)}function ft(...i){const e="THREE."+i.shift();console.warn(e,...i)}function fn(...i){const e="THREE."+i.shift();console.error(e,...i)}function Ja(...i){const e=i.join(" ");e in Lp||(Lp[e]=!0,ft(...i))}function LS(i,e,t){return new Promise(function(n,s){function r(){switch(i.clientWaitSync(e,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:s();break;case i.TIMEOUT_EXPIRED:setTimeout(r,t);break;default:n()}}setTimeout(r,t)})}class $r{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[e]===void 0&&(n[e]=[]),n[e].indexOf(t)===-1&&n[e].push(t)}hasEventListener(e,t){const n=this._listeners;return n===void 0?!1:n[e]!==void 0&&n[e].indexOf(t)!==-1}removeEventListener(e,t){const n=this._listeners;if(n===void 0)return;const s=n[e];if(s!==void 0){const r=s.indexOf(t);r!==-1&&s.splice(r,1)}}dispatchEvent(e){const t=this._listeners;if(t===void 0)return;const n=t[e.type];if(n!==void 0){e.target=this;const s=n.slice(0);for(let r=0,o=s.length;r<o;r++)s[r].call(this,e);e.target=null}}}const Rn=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"];let Up=1234567;const Fa=Math.PI/180,el=180/Math.PI;function sa(){const i=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(Rn[i&255]+Rn[i>>8&255]+Rn[i>>16&255]+Rn[i>>24&255]+"-"+Rn[e&255]+Rn[e>>8&255]+"-"+Rn[e>>16&15|64]+Rn[e>>24&255]+"-"+Rn[t&63|128]+Rn[t>>8&255]+"-"+Rn[t>>16&255]+Rn[t>>24&255]+Rn[n&255]+Rn[n>>8&255]+Rn[n>>16&255]+Rn[n>>24&255]).toLowerCase()}function gt(i,e,t){return Math.max(e,Math.min(t,i))}function nh(i,e){return(i%e+e)%e}function BS(i,e,t,n,s){return n+(i-e)*(s-n)/(t-e)}function US(i,e,t){return i!==e?(t-i)/(e-i):0}function La(i,e,t){return(1-t)*i+t*e}function OS(i,e,t,n){return La(i,e,1-Math.exp(-t*n))}function NS(i,e=1){return e-Math.abs(nh(i,e*2)-e)}function zS(i,e,t){return i<=e?0:i>=t?1:(i=(i-e)/(t-e),i*i*(3-2*i))}function kS(i,e,t){return i<=e?0:i>=t?1:(i=(i-e)/(t-e),i*i*i*(i*(i*6-15)+10))}function HS(i,e){return i+Math.floor(Math.random()*(e-i+1))}function VS(i,e){return i+Math.random()*(e-i)}function GS(i){return i*(.5-Math.random())}function WS(i){i!==void 0&&(Up=i);let e=Up+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=e+Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}function XS(i){return i*Fa}function qS(i){return i*el}function YS(i){return(i&i-1)===0&&i!==0}function QS(i){return Math.pow(2,Math.ceil(Math.log(i)/Math.LN2))}function KS(i){return Math.pow(2,Math.floor(Math.log(i)/Math.LN2))}function jS(i,e,t,n,s){const r=Math.cos,o=Math.sin,a=r(t/2),l=o(t/2),c=r((e+n)/2),u=o((e+n)/2),f=r((e-n)/2),d=o((e-n)/2),h=r((n-e)/2),x=o((n-e)/2);switch(s){case"XYX":i.set(a*u,l*f,l*d,a*c);break;case"YZY":i.set(l*d,a*u,l*f,a*c);break;case"ZXZ":i.set(l*f,l*d,a*u,a*c);break;case"XZX":i.set(a*u,l*x,l*h,a*c);break;case"YXY":i.set(l*h,a*u,l*x,a*c);break;case"ZYZ":i.set(l*x,l*h,a*u,a*c);break;default:ft("MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+s)}}function Mo(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return i/4294967295;case Uint16Array:return i/65535;case Uint8Array:return i/255;case Int32Array:return Math.max(i/2147483647,-1);case Int16Array:return Math.max(i/32767,-1);case Int8Array:return Math.max(i/127,-1);default:throw new Error("Invalid component type.")}}function Nn(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return Math.round(i*4294967295);case Uint16Array:return Math.round(i*65535);case Uint8Array:return Math.round(i*255);case Int32Array:return Math.round(i*2147483647);case Int16Array:return Math.round(i*32767);case Int8Array:return Math.round(i*127);default:throw new Error("Invalid component type.")}}const Sn={DEG2RAD:Fa,RAD2DEG:el,generateUUID:sa,clamp:gt,euclideanModulo:nh,mapLinear:BS,inverseLerp:US,lerp:La,damp:OS,pingpong:NS,smoothstep:zS,smootherstep:kS,randInt:HS,randFloat:VS,randFloatSpread:GS,seededRandom:WS,degToRad:XS,radToDeg:qS,isPowerOfTwo:YS,ceilPowerOfTwo:QS,floorPowerOfTwo:KS,setQuaternionFromProperEuler:jS,normalize:Nn,denormalize:Mo};class Ke{constructor(e=0,t=0){Ke.prototype.isVector2=!0,this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const t=this.x,n=this.y,s=e.elements;return this.x=s[0]*t+s[3]*n+s[6],this.y=s[1]*t+s[4]*n+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=gt(this.x,e.x,t.x),this.y=gt(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=gt(this.x,e,t),this.y=gt(this.y,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(gt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(gt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y;return t*t+n*n}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){const n=Math.cos(t),s=Math.sin(t),r=this.x-e.x,o=this.y-e.y;return this.x=r*n-o*s+e.x,this.y=r*s+o*n+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class Vt{constructor(e=0,t=0,n=0,s=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=n,this._w=s}static slerpFlat(e,t,n,s,r,o,a){let l=n[s+0],c=n[s+1],u=n[s+2],f=n[s+3],d=r[o+0],h=r[o+1],x=r[o+2],p=r[o+3];if(a<=0){e[t+0]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f;return}if(a>=1){e[t+0]=d,e[t+1]=h,e[t+2]=x,e[t+3]=p;return}if(f!==p||l!==d||c!==h||u!==x){let g=l*d+c*h+u*x+f*p;g<0&&(d=-d,h=-h,x=-x,p=-p,g=-g);let m=1-a;if(g<.9995){const _=Math.acos(g),A=Math.sin(_);m=Math.sin(m*_)/A,a=Math.sin(a*_)/A,l=l*m+d*a,c=c*m+h*a,u=u*m+x*a,f=f*m+p*a}else{l=l*m+d*a,c=c*m+h*a,u=u*m+x*a,f=f*m+p*a;const _=1/Math.sqrt(l*l+c*c+u*u+f*f);l*=_,c*=_,u*=_,f*=_}}e[t]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f}static multiplyQuaternionsFlat(e,t,n,s,r,o){const a=n[s],l=n[s+1],c=n[s+2],u=n[s+3],f=r[o],d=r[o+1],h=r[o+2],x=r[o+3];return e[t]=a*x+u*f+l*h-c*d,e[t+1]=l*x+u*d+c*f-a*h,e[t+2]=c*x+u*h+a*d-l*f,e[t+3]=u*x-a*f-l*d-c*h,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,n,s){return this._x=e,this._y=t,this._z=n,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){const n=e._x,s=e._y,r=e._z,o=e._order,a=Math.cos,l=Math.sin,c=a(n/2),u=a(s/2),f=a(r/2),d=l(n/2),h=l(s/2),x=l(r/2);switch(o){case"XYZ":this._x=d*u*f+c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f-d*h*x;break;case"YXZ":this._x=d*u*f+c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f+d*h*x;break;case"ZXY":this._x=d*u*f-c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f-d*h*x;break;case"ZYX":this._x=d*u*f-c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f+d*h*x;break;case"YZX":this._x=d*u*f+c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f-d*h*x;break;case"XZY":this._x=d*u*f-c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f+d*h*x;break;default:ft("Quaternion: .setFromEuler() encountered an unknown order: "+o)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){const n=t/2,s=Math.sin(n);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(e){const t=e.elements,n=t[0],s=t[4],r=t[8],o=t[1],a=t[5],l=t[9],c=t[2],u=t[6],f=t[10],d=n+a+f;if(d>0){const h=.5/Math.sqrt(d+1);this._w=.25/h,this._x=(u-l)*h,this._y=(r-c)*h,this._z=(o-s)*h}else if(n>a&&n>f){const h=2*Math.sqrt(1+n-a-f);this._w=(u-l)/h,this._x=.25*h,this._y=(s+o)/h,this._z=(r+c)/h}else if(a>f){const h=2*Math.sqrt(1+a-n-f);this._w=(r-c)/h,this._x=(s+o)/h,this._y=.25*h,this._z=(l+u)/h}else{const h=2*Math.sqrt(1+f-n-a);this._w=(o-s)/h,this._x=(r+c)/h,this._y=(l+u)/h,this._z=.25*h}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let n=e.dot(t)+1;return n<1e-8?(n=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=n):(this._x=0,this._y=-e.z,this._z=e.y,this._w=n)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=n),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(gt(this.dot(e),-1,1)))}rotateTowards(e,t){const n=this.angleTo(e);if(n===0)return this;const s=Math.min(1,t/n);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){const n=e._x,s=e._y,r=e._z,o=e._w,a=t._x,l=t._y,c=t._z,u=t._w;return this._x=n*u+o*a+s*c-r*l,this._y=s*u+o*l+r*a-n*c,this._z=r*u+o*c+n*l-s*a,this._w=o*u-n*a-s*l-r*c,this._onChangeCallback(),this}slerp(e,t){if(t<=0)return this;if(t>=1)return this.copy(e);let n=e._x,s=e._y,r=e._z,o=e._w,a=this.dot(e);a<0&&(n=-n,s=-s,r=-r,o=-o,a=-a);let l=1-t;if(a<.9995){const c=Math.acos(a),u=Math.sin(c);l=Math.sin(l*c)/u,t=Math.sin(t*c)/u,this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this._onChangeCallback()}else this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this.normalize();return this}slerpQuaternions(e,t,n){return this.copy(e).slerp(t,n)}random(){const e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),n=Math.random(),s=Math.sqrt(1-n),r=Math.sqrt(n);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(t),r*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class B{constructor(e=0,t=0,n=0){B.prototype.isVector3=!0,this.x=e,this.y=t,this.z=n}set(e,t,n){return n===void 0&&(n=this.z),this.x=e,this.y=t,this.z=n,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(Op.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(Op.setFromAxisAngle(e,t))}applyMatrix3(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[3]*n+r[6]*s,this.y=r[1]*t+r[4]*n+r[7]*s,this.z=r[2]*t+r[5]*n+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=e.elements,o=1/(r[3]*t+r[7]*n+r[11]*s+r[15]);return this.x=(r[0]*t+r[4]*n+r[8]*s+r[12])*o,this.y=(r[1]*t+r[5]*n+r[9]*s+r[13])*o,this.z=(r[2]*t+r[6]*n+r[10]*s+r[14])*o,this}applyQuaternion(e){const t=this.x,n=this.y,s=this.z,r=e.x,o=e.y,a=e.z,l=e.w,c=2*(o*s-a*n),u=2*(a*t-r*s),f=2*(r*n-o*t);return this.x=t+l*c+o*f-a*u,this.y=n+l*u+a*c-r*f,this.z=s+l*f+r*u-o*c,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[4]*n+r[8]*s,this.y=r[1]*t+r[5]*n+r[9]*s,this.z=r[2]*t+r[6]*n+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=gt(this.x,e.x,t.x),this.y=gt(this.y,e.y,t.y),this.z=gt(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=gt(this.x,e,t),this.y=gt(this.y,e,t),this.z=gt(this.z,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(gt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){const n=e.x,s=e.y,r=e.z,o=t.x,a=t.y,l=t.z;return this.x=s*l-r*a,this.y=r*o-n*l,this.z=n*a-s*o,this}projectOnVector(e){const t=e.lengthSq();if(t===0)return this.set(0,0,0);const n=e.dot(this)/t;return this.copy(e).multiplyScalar(n)}projectOnPlane(e){return _u.copy(this).projectOnVector(e),this.sub(_u)}reflect(e){return this.sub(_u.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(gt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y,s=this.z-e.z;return t*t+n*n+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,n){const s=Math.sin(t)*e;return this.x=s*Math.sin(n),this.y=Math.cos(t)*e,this.z=s*Math.cos(n),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,n){return this.x=e*Math.sin(t),this.y=n,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){const t=this.setFromMatrixColumn(e,0).length(),n=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=n,this.z=s,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,t=Math.random()*2-1,n=Math.sqrt(1-t*t);return this.x=n*Math.cos(e),this.y=t,this.z=n*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const _u=new B,Op=new Vt;class lt{constructor(e,t,n,s,r,o,a,l,c){lt.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c)}set(e,t,n,s,r,o,a,l,c){const u=this.elements;return u[0]=e,u[1]=s,u[2]=a,u[3]=t,u[4]=r,u[5]=l,u[6]=n,u[7]=o,u[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],this}extractBasis(e,t,n){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[3],l=n[6],c=n[1],u=n[4],f=n[7],d=n[2],h=n[5],x=n[8],p=s[0],g=s[3],m=s[6],_=s[1],A=s[4],v=s[7],S=s[2],y=s[5],M=s[8];return r[0]=o*p+a*_+l*S,r[3]=o*g+a*A+l*y,r[6]=o*m+a*v+l*M,r[1]=c*p+u*_+f*S,r[4]=c*g+u*A+f*y,r[7]=c*m+u*v+f*M,r[2]=d*p+h*_+x*S,r[5]=d*g+h*A+x*y,r[8]=d*m+h*v+x*M,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8];return t*o*u-t*a*c-n*r*u+n*a*l+s*r*c-s*o*l}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=u*o-a*c,d=a*l-u*r,h=c*r-o*l,x=t*f+n*d+s*h;if(x===0)return this.set(0,0,0,0,0,0,0,0,0);const p=1/x;return e[0]=f*p,e[1]=(s*c-u*n)*p,e[2]=(a*n-s*o)*p,e[3]=d*p,e[4]=(u*t-s*l)*p,e[5]=(s*r-a*t)*p,e[6]=h*p,e[7]=(n*l-c*t)*p,e[8]=(o*t-n*r)*p,this}transpose(){let e;const t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,n,s,r,o,a){const l=Math.cos(r),c=Math.sin(r);return this.set(n*l,n*c,-n*(l*o+c*a)+o+e,-s*c,s*l,-s*(-c*o+l*a)+a+t,0,0,1),this}scale(e,t){return this.premultiply(vu.makeScale(e,t)),this}rotate(e){return this.premultiply(vu.makeRotation(-e)),this}translate(e,t){return this.premultiply(vu.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,n,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<9;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<9;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e}clone(){return new this.constructor().fromArray(this.elements)}}const vu=new lt,Np=new lt().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),zp=new lt().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function $S(){const i={enabled:!0,workingColorSpace:Qo,spaces:{},convert:function(s,r,o){return this.enabled===!1||r===o||!r||!o||(this.spaces[r].transfer===Ot&&(s.r=Ps(s.r),s.g=Ps(s.g),s.b=Ps(s.b)),this.spaces[r].primaries!==this.spaces[o].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[o].fromXYZ)),this.spaces[o].transfer===Ot&&(s.r=Bo(s.r),s.g=Bo(s.g),s.b=Bo(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===js?bc:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,o){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[o].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return Ja("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),i.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return Ja("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),i.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],n=[.3127,.329];return i.define({[Qo]:{primaries:e,whitePoint:n,transfer:bc,toXYZ:Np,fromXYZ:zp,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:Ci},outputColorSpaceConfig:{drawingBufferColorSpace:Ci}},[Ci]:{primaries:e,whitePoint:n,transfer:Ot,toXYZ:Np,fromXYZ:zp,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:Ci}}}),i}const Et=$S();function Ps(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function Bo(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}let oo;class ZS{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let n;if(e instanceof HTMLCanvasElement)n=e;else{oo===void 0&&(oo=Cc("canvas")),oo.width=e.width,oo.height=e.height;const s=oo.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),n=oo}return n.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const t=Cc("canvas");t.width=e.width,t.height=e.height;const n=t.getContext("2d");n.drawImage(e,0,0,e.width,e.height);const s=n.getImageData(0,0,e.width,e.height),r=s.data;for(let o=0;o<r.length;o++)r[o]=Ps(r[o]/255)*255;return n.putImageData(s,0,0),t}else if(e.data){const t=e.data.slice(0);for(let n=0;n<t.length;n++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[n]=Math.floor(Ps(t[n]/255)*255):t[n]=Ps(t[n]);return{data:t,width:e.width,height:e.height}}else return ft("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let JS=0;class ih{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:JS++}),this.uuid=sa(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):t instanceof VideoFrame?e.set(t.displayHeight,t.displayWidth,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const n={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let o=0,a=s.length;o<a;o++)s[o].isDataTexture?r.push(Au(s[o].image)):r.push(Au(s[o]))}else r=Au(s);n.url=r}return t||(e.images[this.uuid]=n),n}}function Au(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?ZS.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(ft("Texture: Unable to serialize Texture."),{})}let ey=0;const Su=new B;class qn extends $r{constructor(e=qn.DEFAULT_IMAGE,t=qn.DEFAULT_MAPPING,n=Is,s=Is,r=Ri,o=kr,a=Xn,l=as,c=qn.DEFAULT_ANISOTROPY,u=js){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:ey++}),this.uuid=sa(),this.name="",this.source=new ih(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=n,this.wrapT=s,this.magFilter=r,this.minFilter=o,this.anisotropy=c,this.format=a,this.internalFormat=null,this.type=l,this.offset=new Ke(0,0),this.repeat=new Ke(1,1),this.center=new Ke(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new lt,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=u,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0}get width(){return this.source.getSize(Su).x}get height(){return this.source.getSize(Su).y}get depth(){return this.source.getSize(Su).z}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const t in e){const n=e[t];if(n===void 0){ft(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){ft(`Texture.setValues(): property '${t}' does not exist.`);continue}s&&n&&s.isVector2&&n.isVector2||s&&n&&s.isVector3&&n.isVector3||s&&n&&s.isMatrix3&&n.isMatrix3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const n={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),t||(e.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==Pg)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case Bf:e.x=e.x-Math.floor(e.x);break;case Is:e.x=e.x<0?0:1;break;case Uf:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case Bf:e.y=e.y-Math.floor(e.y);break;case Is:e.y=e.y<0?0:1;break;case Uf:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}qn.DEFAULT_IMAGE=null;qn.DEFAULT_MAPPING=Pg;qn.DEFAULT_ANISOTROPY=1;class Jt{constructor(e=0,t=0,n=0,s=1){Jt.prototype.isVector4=!0,this.x=e,this.y=t,this.z=n,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,n,s){return this.x=e,this.y=t,this.z=n,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=this.w,o=e.elements;return this.x=o[0]*t+o[4]*n+o[8]*s+o[12]*r,this.y=o[1]*t+o[5]*n+o[9]*s+o[13]*r,this.z=o[2]*t+o[6]*n+o[10]*s+o[14]*r,this.w=o[3]*t+o[7]*n+o[11]*s+o[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,n,s,r;const l=e.elements,c=l[0],u=l[4],f=l[8],d=l[1],h=l[5],x=l[9],p=l[2],g=l[6],m=l[10];if(Math.abs(u-d)<.01&&Math.abs(f-p)<.01&&Math.abs(x-g)<.01){if(Math.abs(u+d)<.1&&Math.abs(f+p)<.1&&Math.abs(x+g)<.1&&Math.abs(c+h+m-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;const A=(c+1)/2,v=(h+1)/2,S=(m+1)/2,y=(u+d)/4,M=(f+p)/4,E=(x+g)/4;return A>v&&A>S?A<.01?(n=0,s=.707106781,r=.707106781):(n=Math.sqrt(A),s=y/n,r=M/n):v>S?v<.01?(n=.707106781,s=0,r=.707106781):(s=Math.sqrt(v),n=y/s,r=E/s):S<.01?(n=.707106781,s=.707106781,r=0):(r=Math.sqrt(S),n=M/r,s=E/r),this.set(n,s,r,t),this}let _=Math.sqrt((g-x)*(g-x)+(f-p)*(f-p)+(d-u)*(d-u));return Math.abs(_)<.001&&(_=1),this.x=(g-x)/_,this.y=(f-p)/_,this.z=(d-u)/_,this.w=Math.acos((c+h+m-1)/2),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=gt(this.x,e.x,t.x),this.y=gt(this.y,e.y,t.y),this.z=gt(this.z,e.z,t.z),this.w=gt(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=gt(this.x,e,t),this.y=gt(this.y,e,t),this.z=gt(this.z,e,t),this.w=gt(this.w,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(gt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this.w=e.w+(t.w-e.w)*n,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class ty extends $r{constructor(e=1,t=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:Ri,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},n),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=n.depth,this.scissor=new Jt(0,0,e,t),this.scissorTest=!1,this.viewport=new Jt(0,0,e,t);const s={width:e,height:t,depth:n.depth},r=new qn(s);this.textures=[];const o=n.count;for(let a=0;a<o;a++)this.textures[a]=r.clone(),this.textures[a].isRenderTargetTexture=!0,this.textures[a].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview}_setTextureOptions(e={}){const t={minFilter:Ri,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let n=0;n<this.textures.length;n++)this.textures[n].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,n=1){if(this.width!==e||this.height!==t||this.depth!==n){this.width=e,this.height=t,this.depth=n;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=t,this.textures[s].image.depth=n,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,n=e.textures.length;t<n;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;const s=Object.assign({},e.textures[t].image);this.textures[t].source=new ih(s)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class cr extends ty{constructor(e=1,t=1,n={}){super(e,t,n),this.isWebGLRenderTarget=!0}}class Vg extends qn{constructor(e=null,t=1,n=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=xi,this.minFilter=xi,this.wrapR=Is,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class ny extends qn{constructor(e=null,t=1,n=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=xi,this.minFilter=xi,this.wrapR=Is,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class Ji{constructor(e=new B(1/0,1/0,1/0),t=new B(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t+=3)this.expandByPoint(Ui.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,n=e.count;t<n;t++)this.expandByPoint(Ui.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){const n=Ui.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(n),this.max.copy(e).add(n),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);const n=e.geometry;if(n!==void 0){const r=n.getAttribute("position");if(t===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let o=0,a=r.count;o<a;o++)e.isMesh===!0?e.getVertexPosition(o,Ui):Ui.fromBufferAttribute(r,o),Ui.applyMatrix4(e.matrixWorld),this.expandByPoint(Ui);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),Tl.copy(e.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),Tl.copy(n.boundingBox)),Tl.applyMatrix4(e.matrixWorld),this.union(Tl)}const s=e.children;for(let r=0,o=s.length;r<o;r++)this.expandByObject(s[r],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,Ui),Ui.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,n;return e.normal.x>0?(t=e.normal.x*this.min.x,n=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,n=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,n+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,n+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,n+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,n+=e.normal.z*this.min.z),t<=-e.constant&&n>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(pa),El.subVectors(this.max,pa),ao.subVectors(e.a,pa),lo.subVectors(e.b,pa),co.subVectors(e.c,pa),zs.subVectors(lo,ao),ks.subVectors(co,lo),Mr.subVectors(ao,co);let t=[0,-zs.z,zs.y,0,-ks.z,ks.y,0,-Mr.z,Mr.y,zs.z,0,-zs.x,ks.z,0,-ks.x,Mr.z,0,-Mr.x,-zs.y,zs.x,0,-ks.y,ks.x,0,-Mr.y,Mr.x,0];return!yu(t,ao,lo,co,El)||(t=[1,0,0,0,1,0,0,0,1],!yu(t,ao,lo,co,El))?!1:(wl.crossVectors(zs,ks),t=[wl.x,wl.y,wl.z],yu(t,ao,lo,co,El))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,Ui).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(Ui).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(ps[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),ps[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),ps[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),ps[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),ps[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),ps[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),ps[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),ps[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(ps),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const ps=[new B,new B,new B,new B,new B,new B,new B,new B],Ui=new B,Tl=new Ji,ao=new B,lo=new B,co=new B,zs=new B,ks=new B,Mr=new B,pa=new B,El=new B,wl=new B,Cr=new B;function yu(i,e,t,n,s){for(let r=0,o=i.length-3;r<=o;r+=3){Cr.fromArray(i,r);const a=s.x*Math.abs(Cr.x)+s.y*Math.abs(Cr.y)+s.z*Math.abs(Cr.z),l=e.dot(Cr),c=t.dot(Cr),u=n.dot(Cr);if(Math.max(-Math.max(l,c,u),Math.min(l,c,u))>a)return!1}return!0}const iy=new Ji,ma=new B,bu=new B;class jc{constructor(e=new B,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){const n=this.center;t!==void 0?n.copy(t):iy.setFromPoints(e).getCenter(n);let s=0;for(let r=0,o=e.length;r<o;r++)s=Math.max(s,n.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){const n=this.center.distanceToSquared(e);return t.copy(e),n>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;ma.subVectors(e,this.center);const t=ma.lengthSq();if(t>this.radius*this.radius){const n=Math.sqrt(t),s=(n-this.radius)*.5;this.center.addScaledVector(ma,s/n),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(bu.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(ma.copy(e.center).add(bu)),this.expandByPoint(ma.copy(e.center).sub(bu))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}const ms=new B,Mu=new B,Rl=new B,Hs=new B,Cu=new B,Il=new B,Tu=new B;let sh=class{constructor(e=new B,t=new B(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,ms)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);const n=t.dot(this.direction);return n<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const t=ms.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(ms.copy(this.origin).addScaledVector(this.direction,t),ms.distanceToSquared(e))}distanceSqToSegment(e,t,n,s){Mu.copy(e).add(t).multiplyScalar(.5),Rl.copy(t).sub(e).normalize(),Hs.copy(this.origin).sub(Mu);const r=e.distanceTo(t)*.5,o=-this.direction.dot(Rl),a=Hs.dot(this.direction),l=-Hs.dot(Rl),c=Hs.lengthSq(),u=Math.abs(1-o*o);let f,d,h,x;if(u>0)if(f=o*l-a,d=o*a-l,x=r*u,f>=0)if(d>=-x)if(d<=x){const p=1/u;f*=p,d*=p,h=f*(f+o*d+2*a)+d*(o*f+d+2*l)+c}else d=r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;else d=-r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;else d<=-x?(f=Math.max(0,-(-o*r+a)),d=f>0?-r:Math.min(Math.max(-r,-l),r),h=-f*f+d*(d+2*l)+c):d<=x?(f=0,d=Math.min(Math.max(-r,-l),r),h=d*(d+2*l)+c):(f=Math.max(0,-(o*r+a)),d=f>0?r:Math.min(Math.max(-r,-l),r),h=-f*f+d*(d+2*l)+c);else d=o>0?-r:r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;return n&&n.copy(this.origin).addScaledVector(this.direction,f),s&&s.copy(Mu).addScaledVector(Rl,d),h}intersectSphere(e,t){ms.subVectors(e.center,this.origin);const n=ms.dot(this.direction),s=ms.dot(ms)-n*n,r=e.radius*e.radius;if(s>r)return null;const o=Math.sqrt(r-s),a=n-o,l=n+o;return l<0?null:a<0?this.at(l,t):this.at(a,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(e.normal)+e.constant)/t;return n>=0?n:null}intersectPlane(e,t){const n=this.distanceToPlane(e);return n===null?null:this.at(n,t)}intersectsPlane(e){const t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let n,s,r,o,a,l;const c=1/this.direction.x,u=1/this.direction.y,f=1/this.direction.z,d=this.origin;return c>=0?(n=(e.min.x-d.x)*c,s=(e.max.x-d.x)*c):(n=(e.max.x-d.x)*c,s=(e.min.x-d.x)*c),u>=0?(r=(e.min.y-d.y)*u,o=(e.max.y-d.y)*u):(r=(e.max.y-d.y)*u,o=(e.min.y-d.y)*u),n>o||r>s||((r>n||isNaN(n))&&(n=r),(o<s||isNaN(s))&&(s=o),f>=0?(a=(e.min.z-d.z)*f,l=(e.max.z-d.z)*f):(a=(e.max.z-d.z)*f,l=(e.min.z-d.z)*f),n>l||a>s)||((a>n||n!==n)&&(n=a),(l<s||s!==s)&&(s=l),s<0)?null:this.at(n>=0?n:s,t)}intersectsBox(e){return this.intersectBox(e,ms)!==null}intersectTriangle(e,t,n,s,r){Cu.subVectors(t,e),Il.subVectors(n,e),Tu.crossVectors(Cu,Il);let o=this.direction.dot(Tu),a;if(o>0){if(s)return null;a=1}else if(o<0)a=-1,o=-o;else return null;Hs.subVectors(this.origin,e);const l=a*this.direction.dot(Il.crossVectors(Hs,Il));if(l<0)return null;const c=a*this.direction.dot(Cu.cross(Hs));if(c<0||l+c>o)return null;const u=-a*Hs.dot(Tu);return u<0?null:this.at(u/o,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}};class st{constructor(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g){st.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g)}set(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g){const m=this.elements;return m[0]=e,m[4]=t,m[8]=n,m[12]=s,m[1]=r,m[5]=o,m[9]=a,m[13]=l,m[2]=c,m[6]=u,m[10]=f,m[14]=d,m[3]=h,m[7]=x,m[11]=p,m[15]=g,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new st().fromArray(this.elements)}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],this}copyPosition(e){const t=this.elements,n=e.elements;return t[12]=n[12],t[13]=n[13],t[14]=n[14],this}setFromMatrix3(e){const t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,n){return e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this}makeBasis(e,t,n){return this.set(e.x,t.x,n.x,0,e.y,t.y,n.y,0,e.z,t.z,n.z,0,0,0,0,1),this}extractRotation(e){const t=this.elements,n=e.elements,s=1/uo.setFromMatrixColumn(e,0).length(),r=1/uo.setFromMatrixColumn(e,1).length(),o=1/uo.setFromMatrixColumn(e,2).length();return t[0]=n[0]*s,t[1]=n[1]*s,t[2]=n[2]*s,t[3]=0,t[4]=n[4]*r,t[5]=n[5]*r,t[6]=n[6]*r,t[7]=0,t[8]=n[8]*o,t[9]=n[9]*o,t[10]=n[10]*o,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){const t=this.elements,n=e.x,s=e.y,r=e.z,o=Math.cos(n),a=Math.sin(n),l=Math.cos(s),c=Math.sin(s),u=Math.cos(r),f=Math.sin(r);if(e.order==="XYZ"){const d=o*u,h=o*f,x=a*u,p=a*f;t[0]=l*u,t[4]=-l*f,t[8]=c,t[1]=h+x*c,t[5]=d-p*c,t[9]=-a*l,t[2]=p-d*c,t[6]=x+h*c,t[10]=o*l}else if(e.order==="YXZ"){const d=l*u,h=l*f,x=c*u,p=c*f;t[0]=d+p*a,t[4]=x*a-h,t[8]=o*c,t[1]=o*f,t[5]=o*u,t[9]=-a,t[2]=h*a-x,t[6]=p+d*a,t[10]=o*l}else if(e.order==="ZXY"){const d=l*u,h=l*f,x=c*u,p=c*f;t[0]=d-p*a,t[4]=-o*f,t[8]=x+h*a,t[1]=h+x*a,t[5]=o*u,t[9]=p-d*a,t[2]=-o*c,t[6]=a,t[10]=o*l}else if(e.order==="ZYX"){const d=o*u,h=o*f,x=a*u,p=a*f;t[0]=l*u,t[4]=x*c-h,t[8]=d*c+p,t[1]=l*f,t[5]=p*c+d,t[9]=h*c-x,t[2]=-c,t[6]=a*l,t[10]=o*l}else if(e.order==="YZX"){const d=o*l,h=o*c,x=a*l,p=a*c;t[0]=l*u,t[4]=p-d*f,t[8]=x*f+h,t[1]=f,t[5]=o*u,t[9]=-a*u,t[2]=-c*u,t[6]=h*f+x,t[10]=d-p*f}else if(e.order==="XZY"){const d=o*l,h=o*c,x=a*l,p=a*c;t[0]=l*u,t[4]=-f,t[8]=c*u,t[1]=d*f+p,t[5]=o*u,t[9]=h*f-x,t[2]=x*f-h,t[6]=a*u,t[10]=p*f+d}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(sy,e,ry)}lookAt(e,t,n){const s=this.elements;return ai.subVectors(e,t),ai.lengthSq()===0&&(ai.z=1),ai.normalize(),Vs.crossVectors(n,ai),Vs.lengthSq()===0&&(Math.abs(n.z)===1?ai.x+=1e-4:ai.z+=1e-4,ai.normalize(),Vs.crossVectors(n,ai)),Vs.normalize(),Dl.crossVectors(ai,Vs),s[0]=Vs.x,s[4]=Dl.x,s[8]=ai.x,s[1]=Vs.y,s[5]=Dl.y,s[9]=ai.y,s[2]=Vs.z,s[6]=Dl.z,s[10]=ai.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[4],l=n[8],c=n[12],u=n[1],f=n[5],d=n[9],h=n[13],x=n[2],p=n[6],g=n[10],m=n[14],_=n[3],A=n[7],v=n[11],S=n[15],y=s[0],M=s[4],E=s[8],b=s[12],C=s[1],D=s[5],F=s[9],O=s[13],z=s[2],V=s[6],H=s[10],q=s[14],G=s[3],$=s[7],fe=s[11],Y=s[15];return r[0]=o*y+a*C+l*z+c*G,r[4]=o*M+a*D+l*V+c*$,r[8]=o*E+a*F+l*H+c*fe,r[12]=o*b+a*O+l*q+c*Y,r[1]=u*y+f*C+d*z+h*G,r[5]=u*M+f*D+d*V+h*$,r[9]=u*E+f*F+d*H+h*fe,r[13]=u*b+f*O+d*q+h*Y,r[2]=x*y+p*C+g*z+m*G,r[6]=x*M+p*D+g*V+m*$,r[10]=x*E+p*F+g*H+m*fe,r[14]=x*b+p*O+g*q+m*Y,r[3]=_*y+A*C+v*z+S*G,r[7]=_*M+A*D+v*V+S*$,r[11]=_*E+A*F+v*H+S*fe,r[15]=_*b+A*O+v*q+S*Y,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[4],s=e[8],r=e[12],o=e[1],a=e[5],l=e[9],c=e[13],u=e[2],f=e[6],d=e[10],h=e[14],x=e[3],p=e[7],g=e[11],m=e[15];return x*(+r*l*f-s*c*f-r*a*d+n*c*d+s*a*h-n*l*h)+p*(+t*l*h-t*c*d+r*o*d-s*o*h+s*c*u-r*l*u)+g*(+t*c*f-t*a*h-r*o*f+n*o*h+r*a*u-n*c*u)+m*(-s*a*u-t*l*f+t*a*d+s*o*f-n*o*d+n*l*u)}transpose(){const e=this.elements;let t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,n){const s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=t,s[14]=n),this}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=e[9],d=e[10],h=e[11],x=e[12],p=e[13],g=e[14],m=e[15],_=f*g*c-p*d*c+p*l*h-a*g*h-f*l*m+a*d*m,A=x*d*c-u*g*c-x*l*h+o*g*h+u*l*m-o*d*m,v=u*p*c-x*f*c+x*a*h-o*p*h-u*a*m+o*f*m,S=x*f*l-u*p*l-x*a*d+o*p*d+u*a*g-o*f*g,y=t*_+n*A+s*v+r*S;if(y===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const M=1/y;return e[0]=_*M,e[1]=(p*d*r-f*g*r-p*s*h+n*g*h+f*s*m-n*d*m)*M,e[2]=(a*g*r-p*l*r+p*s*c-n*g*c-a*s*m+n*l*m)*M,e[3]=(f*l*r-a*d*r-f*s*c+n*d*c+a*s*h-n*l*h)*M,e[4]=A*M,e[5]=(u*g*r-x*d*r+x*s*h-t*g*h-u*s*m+t*d*m)*M,e[6]=(x*l*r-o*g*r-x*s*c+t*g*c+o*s*m-t*l*m)*M,e[7]=(o*d*r-u*l*r+u*s*c-t*d*c-o*s*h+t*l*h)*M,e[8]=v*M,e[9]=(x*f*r-u*p*r-x*n*h+t*p*h+u*n*m-t*f*m)*M,e[10]=(o*p*r-x*a*r+x*n*c-t*p*c-o*n*m+t*a*m)*M,e[11]=(u*a*r-o*f*r-u*n*c+t*f*c+o*n*h-t*a*h)*M,e[12]=S*M,e[13]=(u*p*s-x*f*s+x*n*d-t*p*d-u*n*g+t*f*g)*M,e[14]=(x*a*s-o*p*s-x*n*l+t*p*l+o*n*g-t*a*g)*M,e[15]=(o*f*s-u*a*s+u*n*l-t*f*l-o*n*d+t*a*d)*M,this}scale(e){const t=this.elements,n=e.x,s=e.y,r=e.z;return t[0]*=n,t[4]*=s,t[8]*=r,t[1]*=n,t[5]*=s,t[9]*=r,t[2]*=n,t[6]*=s,t[10]*=r,t[3]*=n,t[7]*=s,t[11]*=r,this}getMaxScaleOnAxis(){const e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],n=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,n,s))}makeTranslation(e,t,n){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,n,0,0,0,1),this}makeRotationX(e){const t=Math.cos(e),n=Math.sin(e);return this.set(1,0,0,0,0,t,-n,0,0,n,t,0,0,0,0,1),this}makeRotationY(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,0,n,0,0,1,0,0,-n,0,t,0,0,0,0,1),this}makeRotationZ(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,0,n,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){const n=Math.cos(t),s=Math.sin(t),r=1-n,o=e.x,a=e.y,l=e.z,c=r*o,u=r*a;return this.set(c*o+n,c*a-s*l,c*l+s*a,0,c*a+s*l,u*a+n,u*l-s*o,0,c*l-s*a,u*l+s*o,r*l*l+n,0,0,0,0,1),this}makeScale(e,t,n){return this.set(e,0,0,0,0,t,0,0,0,0,n,0,0,0,0,1),this}makeShear(e,t,n,s,r,o){return this.set(1,n,r,0,e,1,o,0,t,s,1,0,0,0,0,1),this}compose(e,t,n){const s=this.elements,r=t._x,o=t._y,a=t._z,l=t._w,c=r+r,u=o+o,f=a+a,d=r*c,h=r*u,x=r*f,p=o*u,g=o*f,m=a*f,_=l*c,A=l*u,v=l*f,S=n.x,y=n.y,M=n.z;return s[0]=(1-(p+m))*S,s[1]=(h+v)*S,s[2]=(x-A)*S,s[3]=0,s[4]=(h-v)*y,s[5]=(1-(d+m))*y,s[6]=(g+_)*y,s[7]=0,s[8]=(x+A)*M,s[9]=(g-_)*M,s[10]=(1-(d+p))*M,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,t,n){const s=this.elements;let r=uo.set(s[0],s[1],s[2]).length();const o=uo.set(s[4],s[5],s[6]).length(),a=uo.set(s[8],s[9],s[10]).length();this.determinant()<0&&(r=-r),e.x=s[12],e.y=s[13],e.z=s[14],Oi.copy(this);const c=1/r,u=1/o,f=1/a;return Oi.elements[0]*=c,Oi.elements[1]*=c,Oi.elements[2]*=c,Oi.elements[4]*=u,Oi.elements[5]*=u,Oi.elements[6]*=u,Oi.elements[8]*=f,Oi.elements[9]*=f,Oi.elements[10]*=f,t.setFromRotationMatrix(Oi),n.x=r,n.y=o,n.z=a,this}makePerspective(e,t,n,s,r,o,a=Zi,l=!1){const c=this.elements,u=2*r/(t-e),f=2*r/(n-s),d=(t+e)/(t-e),h=(n+s)/(n-s);let x,p;if(l)x=r/(o-r),p=o*r/(o-r);else if(a===Zi)x=-(o+r)/(o-r),p=-2*o*r/(o-r);else if(a===Mc)x=-o/(o-r),p=-o*r/(o-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=d,c[12]=0,c[1]=0,c[5]=f,c[9]=h,c[13]=0,c[2]=0,c[6]=0,c[10]=x,c[14]=p,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(e,t,n,s,r,o,a=Zi,l=!1){const c=this.elements,u=2/(t-e),f=2/(n-s),d=-(t+e)/(t-e),h=-(n+s)/(n-s);let x,p;if(l)x=1/(o-r),p=o/(o-r);else if(a===Zi)x=-2/(o-r),p=-(o+r)/(o-r);else if(a===Mc)x=-1/(o-r),p=-r/(o-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=0,c[12]=d,c[1]=0,c[5]=f,c[9]=0,c[13]=h,c[2]=0,c[6]=0,c[10]=x,c[14]=p,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<16;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<16;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e[t+9]=n[9],e[t+10]=n[10],e[t+11]=n[11],e[t+12]=n[12],e[t+13]=n[13],e[t+14]=n[14],e[t+15]=n[15],e}}const uo=new B,Oi=new st,sy=new B(0,0,0),ry=new B(1,1,1),Vs=new B,Dl=new B,ai=new B,kp=new st,Hp=new Vt;class Wi{constructor(e=0,t=0,n=0,s=Wi.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=n,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,n,s=this._order){return this._x=e,this._y=t,this._z=n,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,n=!0){const s=e.elements,r=s[0],o=s[4],a=s[8],l=s[1],c=s[5],u=s[9],f=s[2],d=s[6],h=s[10];switch(t){case"XYZ":this._y=Math.asin(gt(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-u,h),this._z=Math.atan2(-o,r)):(this._x=Math.atan2(d,c),this._z=0);break;case"YXZ":this._x=Math.asin(-gt(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(a,h),this._z=Math.atan2(l,c)):(this._y=Math.atan2(-f,r),this._z=0);break;case"ZXY":this._x=Math.asin(gt(d,-1,1)),Math.abs(d)<.9999999?(this._y=Math.atan2(-f,h),this._z=Math.atan2(-o,c)):(this._y=0,this._z=Math.atan2(l,r));break;case"ZYX":this._y=Math.asin(-gt(f,-1,1)),Math.abs(f)<.9999999?(this._x=Math.atan2(d,h),this._z=Math.atan2(l,r)):(this._x=0,this._z=Math.atan2(-o,c));break;case"YZX":this._z=Math.asin(gt(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(-u,c),this._y=Math.atan2(-f,r)):(this._x=0,this._y=Math.atan2(a,h));break;case"XZY":this._z=Math.asin(-gt(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(d,c),this._y=Math.atan2(a,r)):(this._x=Math.atan2(-u,h),this._y=0);break;default:ft("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,n===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,n){return kp.makeRotationFromQuaternion(e),this.setFromRotationMatrix(kp,t,n)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return Hp.setFromEuler(this),this.setFromQuaternion(Hp,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}Wi.DEFAULT_ORDER="XYZ";class Gg{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let oy=0;const Vp=new B,fo=new Vt,gs=new st,Pl=new B,ga=new B,ay=new B,ly=new Vt,Gp=new B(1,0,0),Wp=new B(0,1,0),Xp=new B(0,0,1),qp={type:"added"},cy={type:"removed"},ho={type:"childadded",child:null},Eu={type:"childremoved",child:null};class mn extends $r{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:oy++}),this.uuid=sa(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=mn.DEFAULT_UP.clone();const e=new B,t=new Wi,n=new Vt,s=new B(1,1,1);function r(){n.setFromEuler(t,!1)}function o(){t.setFromQuaternion(n,void 0,!1)}t._onChange(r),n._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new st},normalMatrix:{value:new lt}}),this.matrix=new st,this.matrixWorld=new st,this.matrixAutoUpdate=mn.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=mn.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new Gg,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return fo.setFromAxisAngle(e,t),this.quaternion.multiply(fo),this}rotateOnWorldAxis(e,t){return fo.setFromAxisAngle(e,t),this.quaternion.premultiply(fo),this}rotateX(e){return this.rotateOnAxis(Gp,e)}rotateY(e){return this.rotateOnAxis(Wp,e)}rotateZ(e){return this.rotateOnAxis(Xp,e)}translateOnAxis(e,t){return Vp.copy(e).applyQuaternion(this.quaternion),this.position.add(Vp.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(Gp,e)}translateY(e){return this.translateOnAxis(Wp,e)}translateZ(e){return this.translateOnAxis(Xp,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(gs.copy(this.matrixWorld).invert())}lookAt(e,t,n){e.isVector3?Pl.copy(e):Pl.set(e,t,n);const s=this.parent;this.updateWorldMatrix(!0,!1),ga.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?gs.lookAt(ga,Pl,this.up):gs.lookAt(Pl,ga,this.up),this.quaternion.setFromRotationMatrix(gs),s&&(gs.extractRotation(s.matrixWorld),fo.setFromRotationMatrix(gs),this.quaternion.premultiply(fo.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(fn("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(qp),ho.child=e,this.dispatchEvent(ho),ho.child=null):fn("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(cy),Eu.child=e,this.dispatchEvent(Eu),Eu.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),gs.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),gs.multiply(e.parent.matrixWorld)),e.applyMatrix4(gs),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(qp),ho.child=e,this.dispatchEvent(ho),ho.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let n=0,s=this.children.length;n<s;n++){const o=this.children[n].getObjectByProperty(e,t);if(o!==void 0)return o}}getObjectsByProperty(e,t,n=[]){this[e]===t&&n.push(this);const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].getObjectsByProperty(e,t,n);return n}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(ga,e,ay),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(ga,ly,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverseVisible(e)}traverseAncestors(e){const t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].updateMatrixWorld(e)}updateWorldMatrix(e,t){const n=this.parent;if(e===!0&&n!==null&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),t===!0){const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].updateWorldMatrix(!1,!0)}}toJSON(e){const t=e===void 0||typeof e=="string",n={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(a=>({...a,boundingBox:a.boundingBox?a.boundingBox.toJSON():void 0,boundingSphere:a.boundingSphere?a.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(a=>({...a})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(a,l){return a[l.uuid]===void 0&&(a[l.uuid]=l.toJSON(e)),l.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);const a=this.geometry.parameters;if(a!==void 0&&a.shapes!==void 0){const l=a.shapes;if(Array.isArray(l))for(let c=0,u=l.length;c<u;c++){const f=l[c];r(e.shapes,f)}else r(e.shapes,l)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const a=[];for(let l=0,c=this.material.length;l<c;l++)a.push(r(e.materials,this.material[l]));s.material=a}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let a=0;a<this.children.length;a++)s.children.push(this.children[a].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let a=0;a<this.animations.length;a++){const l=this.animations[a];s.animations.push(r(e.animations,l))}}if(t){const a=o(e.geometries),l=o(e.materials),c=o(e.textures),u=o(e.images),f=o(e.shapes),d=o(e.skeletons),h=o(e.animations),x=o(e.nodes);a.length>0&&(n.geometries=a),l.length>0&&(n.materials=l),c.length>0&&(n.textures=c),u.length>0&&(n.images=u),f.length>0&&(n.shapes=f),d.length>0&&(n.skeletons=d),h.length>0&&(n.animations=h),x.length>0&&(n.nodes=x)}return n.object=s,n;function o(a){const l=[];for(const c in a){const u=a[c];delete u.metadata,l.push(u)}return l}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let n=0;n<e.children.length;n++){const s=e.children[n];this.add(s.clone())}return this}}mn.DEFAULT_UP=new B(0,1,0);mn.DEFAULT_MATRIX_AUTO_UPDATE=!0;mn.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const Ni=new B,xs=new B,wu=new B,_s=new B,po=new B,mo=new B,Yp=new B,Ru=new B,Iu=new B,Du=new B,Pu=new Jt,Fu=new Jt,Lu=new Jt;class ki{constructor(e=new B,t=new B,n=new B){this.a=e,this.b=t,this.c=n}static getNormal(e,t,n,s){s.subVectors(n,t),Ni.subVectors(e,t),s.cross(Ni);const r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,t,n,s,r){Ni.subVectors(s,t),xs.subVectors(n,t),wu.subVectors(e,t);const o=Ni.dot(Ni),a=Ni.dot(xs),l=Ni.dot(wu),c=xs.dot(xs),u=xs.dot(wu),f=o*c-a*a;if(f===0)return r.set(0,0,0),null;const d=1/f,h=(c*l-a*u)*d,x=(o*u-a*l)*d;return r.set(1-h-x,x,h)}static containsPoint(e,t,n,s){return this.getBarycoord(e,t,n,s,_s)===null?!1:_s.x>=0&&_s.y>=0&&_s.x+_s.y<=1}static getInterpolation(e,t,n,s,r,o,a,l){return this.getBarycoord(e,t,n,s,_s)===null?(l.x=0,l.y=0,"z"in l&&(l.z=0),"w"in l&&(l.w=0),null):(l.setScalar(0),l.addScaledVector(r,_s.x),l.addScaledVector(o,_s.y),l.addScaledVector(a,_s.z),l)}static getInterpolatedAttribute(e,t,n,s,r,o){return Pu.setScalar(0),Fu.setScalar(0),Lu.setScalar(0),Pu.fromBufferAttribute(e,t),Fu.fromBufferAttribute(e,n),Lu.fromBufferAttribute(e,s),o.setScalar(0),o.addScaledVector(Pu,r.x),o.addScaledVector(Fu,r.y),o.addScaledVector(Lu,r.z),o}static isFrontFacing(e,t,n,s){return Ni.subVectors(n,t),xs.subVectors(e,t),Ni.cross(xs).dot(s)<0}set(e,t,n){return this.a.copy(e),this.b.copy(t),this.c.copy(n),this}setFromPointsAndIndices(e,t,n,s){return this.a.copy(e[t]),this.b.copy(e[n]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,t,n,s){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,n),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return Ni.subVectors(this.c,this.b),xs.subVectors(this.a,this.b),Ni.cross(xs).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return ki.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return ki.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,n,s,r){return ki.getInterpolation(e,this.a,this.b,this.c,t,n,s,r)}containsPoint(e){return ki.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return ki.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){const n=this.a,s=this.b,r=this.c;let o,a;po.subVectors(s,n),mo.subVectors(r,n),Ru.subVectors(e,n);const l=po.dot(Ru),c=mo.dot(Ru);if(l<=0&&c<=0)return t.copy(n);Iu.subVectors(e,s);const u=po.dot(Iu),f=mo.dot(Iu);if(u>=0&&f<=u)return t.copy(s);const d=l*f-u*c;if(d<=0&&l>=0&&u<=0)return o=l/(l-u),t.copy(n).addScaledVector(po,o);Du.subVectors(e,r);const h=po.dot(Du),x=mo.dot(Du);if(x>=0&&h<=x)return t.copy(r);const p=h*c-l*x;if(p<=0&&c>=0&&x<=0)return a=c/(c-x),t.copy(n).addScaledVector(mo,a);const g=u*x-h*f;if(g<=0&&f-u>=0&&h-x>=0)return Yp.subVectors(r,s),a=(f-u)/(f-u+(h-x)),t.copy(s).addScaledVector(Yp,a);const m=1/(g+p+d);return o=p*m,a=d*m,t.copy(n).addScaledVector(po,o).addScaledVector(mo,a)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}const Wg={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Gs={h:0,s:0,l:0},Fl={h:0,s:0,l:0};function Bu(i,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?i+(e-i)*6*t:t<1/2?e:t<2/3?i+(e-i)*6*(2/3-t):i}class bt{constructor(e,t,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,n)}set(e,t,n){if(t===void 0&&n===void 0){const s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,t,n);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=Ci){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,Et.colorSpaceToWorking(this,t),this}setRGB(e,t,n,s=Et.workingColorSpace){return this.r=e,this.g=t,this.b=n,Et.colorSpaceToWorking(this,s),this}setHSL(e,t,n,s=Et.workingColorSpace){if(e=nh(e,1),t=gt(t,0,1),n=gt(n,0,1),t===0)this.r=this.g=this.b=n;else{const r=n<=.5?n*(1+t):n+t-n*t,o=2*n-r;this.r=Bu(o,r,e+1/3),this.g=Bu(o,r,e),this.b=Bu(o,r,e-1/3)}return Et.colorSpaceToWorking(this,s),this}setStyle(e,t=Ci){function n(r){r!==void 0&&parseFloat(r)<1&&ft("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r;const o=s[1],a=s[2];switch(o){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,t);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,t);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,t);break;default:ft("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){const r=s[1],o=r.length;if(o===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,t);if(o===6)return this.setHex(parseInt(r,16),t);ft("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=Ci){const n=Wg[e.toLowerCase()];return n!==void 0?this.setHex(n,t):ft("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=Ps(e.r),this.g=Ps(e.g),this.b=Ps(e.b),this}copyLinearToSRGB(e){return this.r=Bo(e.r),this.g=Bo(e.g),this.b=Bo(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Ci){return Et.workingToColorSpace(In.copy(this),e),Math.round(gt(In.r*255,0,255))*65536+Math.round(gt(In.g*255,0,255))*256+Math.round(gt(In.b*255,0,255))}getHexString(e=Ci){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=Et.workingColorSpace){Et.workingToColorSpace(In.copy(this),t);const n=In.r,s=In.g,r=In.b,o=Math.max(n,s,r),a=Math.min(n,s,r);let l,c;const u=(a+o)/2;if(a===o)l=0,c=0;else{const f=o-a;switch(c=u<=.5?f/(o+a):f/(2-o-a),o){case n:l=(s-r)/f+(s<r?6:0);break;case s:l=(r-n)/f+2;break;case r:l=(n-s)/f+4;break}l/=6}return e.h=l,e.s=c,e.l=u,e}getRGB(e,t=Et.workingColorSpace){return Et.workingToColorSpace(In.copy(this),t),e.r=In.r,e.g=In.g,e.b=In.b,e}getStyle(e=Ci){Et.workingToColorSpace(In.copy(this),e);const t=In.r,n=In.g,s=In.b;return e!==Ci?`color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(n*255)},${Math.round(s*255)})`}offsetHSL(e,t,n){return this.getHSL(Gs),this.setHSL(Gs.h+e,Gs.s+t,Gs.l+n)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,n){return this.r=e.r+(t.r-e.r)*n,this.g=e.g+(t.g-e.g)*n,this.b=e.b+(t.b-e.b)*n,this}lerpHSL(e,t){this.getHSL(Gs),e.getHSL(Fl);const n=La(Gs.h,Fl.h,t),s=La(Gs.s,Fl.s,t),r=La(Gs.l,Fl.l,t);return this.setHSL(n,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const t=this.r,n=this.g,s=this.b,r=e.elements;return this.r=r[0]*t+r[3]*n+r[6]*s,this.g=r[1]*t+r[4]*n+r[7]*s,this.b=r[2]*t+r[5]*n+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const In=new bt;bt.NAMES=Wg;let uy=0;class hl extends $r{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:uy++}),this.uuid=sa(),this.name="",this.type="Material",this.blending=ir,this.side=os,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=Qa,this.blendDst=Ka,this.blendEquation=Nr,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new bt(0,0,0),this.blendAlpha=0,this.depthFunc=Wo,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Dp,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=ro,this.stencilZFail=ro,this.stencilZPass=ro,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const t in e){const n=e[t];if(n===void 0){ft(`Material: parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){ft(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(n):s&&s.isVector3&&n&&n.isVector3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});const n={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(e).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(e).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(e).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(e).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(e).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==ir&&(n.blending=this.blending),this.side!==os&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==Qa&&(n.blendSrc=this.blendSrc),this.blendDst!==Ka&&(n.blendDst=this.blendDst),this.blendEquation!==Nr&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==Wo&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==Dp&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==ro&&(n.stencilFail=this.stencilFail),this.stencilZFail!==ro&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==ro&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function s(r){const o=[];for(const a in r){const l=r[a];delete l.metadata,o.push(l)}return o}if(t){const r=s(e.textures),o=s(e.images);r.length>0&&(n.textures=r),o.length>0&&(n.images=o)}return n}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const t=e.clippingPlanes;let n=null;if(t!==null){const s=t.length;n=new Array(s);for(let r=0;r!==s;++r)n[r]=t[r].clone()}return this.clippingPlanes=n,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}class Kr extends hl{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new bt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Wi,this.combine=Dg,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const Rs=fy();function fy(){const i=new ArrayBuffer(4),e=new Float32Array(i),t=new Uint32Array(i),n=new Uint32Array(512),s=new Uint32Array(512);for(let l=0;l<256;++l){const c=l-127;c<-27?(n[l]=0,n[l|256]=32768,s[l]=24,s[l|256]=24):c<-14?(n[l]=1024>>-c-14,n[l|256]=1024>>-c-14|32768,s[l]=-c-1,s[l|256]=-c-1):c<=15?(n[l]=c+15<<10,n[l|256]=c+15<<10|32768,s[l]=13,s[l|256]=13):c<128?(n[l]=31744,n[l|256]=64512,s[l]=24,s[l|256]=24):(n[l]=31744,n[l|256]=64512,s[l]=13,s[l|256]=13)}const r=new Uint32Array(2048),o=new Uint32Array(64),a=new Uint32Array(64);for(let l=1;l<1024;++l){let c=l<<13,u=0;for(;(c&8388608)===0;)c<<=1,u-=8388608;c&=-8388609,u+=947912704,r[l]=c|u}for(let l=1024;l<2048;++l)r[l]=939524096+(l-1024<<13);for(let l=1;l<31;++l)o[l]=l<<23;o[31]=1199570944,o[32]=2147483648;for(let l=33;l<63;++l)o[l]=2147483648+(l-32<<23);o[63]=3347054592;for(let l=1;l<64;++l)l!==32&&(a[l]=1024);return{floatView:e,uint32View:t,baseTable:n,shiftTable:s,mantissaTable:r,exponentTable:o,offsetTable:a}}function dy(i){Math.abs(i)>65504&&ft("DataUtils.toHalfFloat(): Value out of range."),i=gt(i,-65504,65504),Rs.floatView[0]=i;const e=Rs.uint32View[0],t=e>>23&511;return Rs.baseTable[t]+((e&8388607)>>Rs.shiftTable[t])}function hy(i){const e=i>>10;return Rs.uint32View[0]=Rs.mantissaTable[Rs.offsetTable[e]+(i&1023)]+Rs.exponentTable[e],Rs.floatView[0]}class tl{static toHalfFloat(e){return dy(e)}static fromHalfFloat(e){return hy(e)}}const dn=new B,Ll=new Ke;let py=0;class Li{constructor(e,t,n=!1){if(Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:py++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=n,this.usage=Pp,this.updateRanges=[],this.gpuType=Hi,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,n){e*=this.itemSize,n*=t.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=t.array[n+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,n=this.count;t<n;t++)Ll.fromBufferAttribute(this,t),Ll.applyMatrix3(e),this.setXY(t,Ll.x,Ll.y);else if(this.itemSize===3)for(let t=0,n=this.count;t<n;t++)dn.fromBufferAttribute(this,t),dn.applyMatrix3(e),this.setXYZ(t,dn.x,dn.y,dn.z);return this}applyMatrix4(e){for(let t=0,n=this.count;t<n;t++)dn.fromBufferAttribute(this,t),dn.applyMatrix4(e),this.setXYZ(t,dn.x,dn.y,dn.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)dn.fromBufferAttribute(this,t),dn.applyNormalMatrix(e),this.setXYZ(t,dn.x,dn.y,dn.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)dn.fromBufferAttribute(this,t),dn.transformDirection(e),this.setXYZ(t,dn.x,dn.y,dn.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let n=this.array[e*this.itemSize+t];return this.normalized&&(n=Mo(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=Nn(n,this.array)),this.array[e*this.itemSize+t]=n,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=Mo(t,this.array)),t}setX(e,t){return this.normalized&&(t=Nn(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=Mo(t,this.array)),t}setY(e,t){return this.normalized&&(t=Nn(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=Mo(t,this.array)),t}setZ(e,t){return this.normalized&&(t=Nn(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=Mo(t,this.array)),t}setW(e,t){return this.normalized&&(t=Nn(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,n){return e*=this.itemSize,this.normalized&&(t=Nn(t,this.array),n=Nn(n,this.array)),this.array[e+0]=t,this.array[e+1]=n,this}setXYZ(e,t,n,s){return e*=this.itemSize,this.normalized&&(t=Nn(t,this.array),n=Nn(n,this.array),s=Nn(s,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this}setXYZW(e,t,n,s,r){return e*=this.itemSize,this.normalized&&(t=Nn(t,this.array),n=Nn(n,this.array),s=Nn(s,this.array),r=Nn(r,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==Pp&&(e.usage=this.usage),e}}class Xg extends Li{constructor(e,t,n){super(new Uint16Array(e),t,n)}}class qg extends Li{constructor(e,t,n){super(new Uint32Array(e),t,n)}}class Ln extends Li{constructor(e,t,n){super(new Float32Array(e),t,n)}}let my=0;const yi=new st,Uu=new mn,go=new B,li=new Ji,xa=new Ji,An=new B;class Kn extends $r{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:my++}),this.uuid=sa(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(Hg(e)?qg:Xg)(e,1):this.index=e,this}setIndirect(e){return this.indirect=e,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,n=0){this.groups.push({start:e,count:t,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){const t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const r=new lt().getNormalMatrix(e);n.applyNormalMatrix(r),n.needsUpdate=!0}const s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return yi.makeRotationFromQuaternion(e),this.applyMatrix4(yi),this}rotateX(e){return yi.makeRotationX(e),this.applyMatrix4(yi),this}rotateY(e){return yi.makeRotationY(e),this.applyMatrix4(yi),this}rotateZ(e){return yi.makeRotationZ(e),this.applyMatrix4(yi),this}translate(e,t,n){return yi.makeTranslation(e,t,n),this.applyMatrix4(yi),this}scale(e,t,n){return yi.makeScale(e,t,n),this.applyMatrix4(yi),this}lookAt(e){return Uu.lookAt(e),Uu.updateMatrix(),this.applyMatrix4(Uu.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(go).negate(),this.translate(go.x,go.y,go.z),this}setFromPoints(e){const t=this.getAttribute("position");if(t===void 0){const n=[];for(let s=0,r=e.length;s<r;s++){const o=e[s];n.push(o.x,o.y,o.z||0)}this.setAttribute("position",new Ln(n,3))}else{const n=Math.min(e.length,t.count);for(let s=0;s<n;s++){const r=e[s];t.setXYZ(s,r.x,r.y,r.z||0)}e.length>t.count&&ft("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new Ji);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){fn("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new B(-1/0,-1/0,-1/0),new B(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let n=0,s=t.length;n<s;n++){const r=t[n];li.setFromBufferAttribute(r),this.morphTargetsRelative?(An.addVectors(this.boundingBox.min,li.min),this.boundingBox.expandByPoint(An),An.addVectors(this.boundingBox.max,li.max),this.boundingBox.expandByPoint(An)):(this.boundingBox.expandByPoint(li.min),this.boundingBox.expandByPoint(li.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&fn('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new jc);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){fn("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new B,1/0);return}if(e){const n=this.boundingSphere.center;if(li.setFromBufferAttribute(e),t)for(let r=0,o=t.length;r<o;r++){const a=t[r];xa.setFromBufferAttribute(a),this.morphTargetsRelative?(An.addVectors(li.min,xa.min),li.expandByPoint(An),An.addVectors(li.max,xa.max),li.expandByPoint(An)):(li.expandByPoint(xa.min),li.expandByPoint(xa.max))}li.getCenter(n);let s=0;for(let r=0,o=e.count;r<o;r++)An.fromBufferAttribute(e,r),s=Math.max(s,n.distanceToSquared(An));if(t)for(let r=0,o=t.length;r<o;r++){const a=t[r],l=this.morphTargetsRelative;for(let c=0,u=a.count;c<u;c++)An.fromBufferAttribute(a,c),l&&(go.fromBufferAttribute(e,c),An.add(go)),s=Math.max(s,n.distanceToSquared(An))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&fn('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){fn("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=t.position,s=t.normal,r=t.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new Li(new Float32Array(4*n.count),4));const o=this.getAttribute("tangent"),a=[],l=[];for(let E=0;E<n.count;E++)a[E]=new B,l[E]=new B;const c=new B,u=new B,f=new B,d=new Ke,h=new Ke,x=new Ke,p=new B,g=new B;function m(E,b,C){c.fromBufferAttribute(n,E),u.fromBufferAttribute(n,b),f.fromBufferAttribute(n,C),d.fromBufferAttribute(r,E),h.fromBufferAttribute(r,b),x.fromBufferAttribute(r,C),u.sub(c),f.sub(c),h.sub(d),x.sub(d);const D=1/(h.x*x.y-x.x*h.y);isFinite(D)&&(p.copy(u).multiplyScalar(x.y).addScaledVector(f,-h.y).multiplyScalar(D),g.copy(f).multiplyScalar(h.x).addScaledVector(u,-x.x).multiplyScalar(D),a[E].add(p),a[b].add(p),a[C].add(p),l[E].add(g),l[b].add(g),l[C].add(g))}let _=this.groups;_.length===0&&(_=[{start:0,count:e.count}]);for(let E=0,b=_.length;E<b;++E){const C=_[E],D=C.start,F=C.count;for(let O=D,z=D+F;O<z;O+=3)m(e.getX(O+0),e.getX(O+1),e.getX(O+2))}const A=new B,v=new B,S=new B,y=new B;function M(E){S.fromBufferAttribute(s,E),y.copy(S);const b=a[E];A.copy(b),A.sub(S.multiplyScalar(S.dot(b))).normalize(),v.crossVectors(y,b);const D=v.dot(l[E])<0?-1:1;o.setXYZW(E,A.x,A.y,A.z,D)}for(let E=0,b=_.length;E<b;++E){const C=_[E],D=C.start,F=C.count;for(let O=D,z=D+F;O<z;O+=3)M(e.getX(O+0)),M(e.getX(O+1)),M(e.getX(O+2))}}computeVertexNormals(){const e=this.index,t=this.getAttribute("position");if(t!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new Li(new Float32Array(t.count*3),3),this.setAttribute("normal",n);else for(let d=0,h=n.count;d<h;d++)n.setXYZ(d,0,0,0);const s=new B,r=new B,o=new B,a=new B,l=new B,c=new B,u=new B,f=new B;if(e)for(let d=0,h=e.count;d<h;d+=3){const x=e.getX(d+0),p=e.getX(d+1),g=e.getX(d+2);s.fromBufferAttribute(t,x),r.fromBufferAttribute(t,p),o.fromBufferAttribute(t,g),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),a.fromBufferAttribute(n,x),l.fromBufferAttribute(n,p),c.fromBufferAttribute(n,g),a.add(u),l.add(u),c.add(u),n.setXYZ(x,a.x,a.y,a.z),n.setXYZ(p,l.x,l.y,l.z),n.setXYZ(g,c.x,c.y,c.z)}else for(let d=0,h=t.count;d<h;d+=3)s.fromBufferAttribute(t,d+0),r.fromBufferAttribute(t,d+1),o.fromBufferAttribute(t,d+2),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),n.setXYZ(d+0,u.x,u.y,u.z),n.setXYZ(d+1,u.x,u.y,u.z),n.setXYZ(d+2,u.x,u.y,u.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let t=0,n=e.count;t<n;t++)An.fromBufferAttribute(e,t),An.normalize(),e.setXYZ(t,An.x,An.y,An.z)}toNonIndexed(){function e(a,l){const c=a.array,u=a.itemSize,f=a.normalized,d=new c.constructor(l.length*u);let h=0,x=0;for(let p=0,g=l.length;p<g;p++){a.isInterleavedBufferAttribute?h=l[p]*a.data.stride+a.offset:h=l[p]*u;for(let m=0;m<u;m++)d[x++]=c[h++]}return new Li(d,u,f)}if(this.index===null)return ft("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const t=new Kn,n=this.index.array,s=this.attributes;for(const a in s){const l=s[a],c=e(l,n);t.setAttribute(a,c)}const r=this.morphAttributes;for(const a in r){const l=[],c=r[a];for(let u=0,f=c.length;u<f;u++){const d=c[u],h=e(d,n);l.push(h)}t.morphAttributes[a]=l}t.morphTargetsRelative=this.morphTargetsRelative;const o=this.groups;for(let a=0,l=o.length;a<l;a++){const c=o[a];t.addGroup(c.start,c.count,c.materialIndex)}return t}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const l=this.parameters;for(const c in l)l[c]!==void 0&&(e[c]=l[c]);return e}e.data={attributes:{}};const t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});const n=this.attributes;for(const l in n){const c=n[l];e.data.attributes[l]=c.toJSON(e.data)}const s={};let r=!1;for(const l in this.morphAttributes){const c=this.morphAttributes[l],u=[];for(let f=0,d=c.length;f<d;f++){const h=c[f];u.push(h.toJSON(e.data))}u.length>0&&(s[l]=u,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);const o=this.groups;o.length>0&&(e.data.groups=JSON.parse(JSON.stringify(o)));const a=this.boundingSphere;return a!==null&&(e.data.boundingSphere=a.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const t={};this.name=e.name;const n=e.index;n!==null&&this.setIndex(n.clone());const s=e.attributes;for(const c in s){const u=s[c];this.setAttribute(c,u.clone(t))}const r=e.morphAttributes;for(const c in r){const u=[],f=r[c];for(let d=0,h=f.length;d<h;d++)u.push(f[d].clone(t));this.morphAttributes[c]=u}this.morphTargetsRelative=e.morphTargetsRelative;const o=e.groups;for(let c=0,u=o.length;c<u;c++){const f=o[c];this.addGroup(f.start,f.count,f.materialIndex)}const a=e.boundingBox;a!==null&&(this.boundingBox=a.clone());const l=e.boundingSphere;return l!==null&&(this.boundingSphere=l.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const Qp=new st,Tr=new sh,Bl=new jc,Kp=new B,Ul=new B,Ol=new B,Nl=new B,Ou=new B,zl=new B,jp=new B,kl=new B;class hn extends mn{constructor(e=new Kn,t=new Kr){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}getVertexPosition(e,t){const n=this.geometry,s=n.attributes.position,r=n.morphAttributes.position,o=n.morphTargetsRelative;t.fromBufferAttribute(s,e);const a=this.morphTargetInfluences;if(r&&a){zl.set(0,0,0);for(let l=0,c=r.length;l<c;l++){const u=a[l],f=r[l];u!==0&&(Ou.fromBufferAttribute(f,e),o?zl.addScaledVector(Ou,u):zl.addScaledVector(Ou.sub(t),u))}t.add(zl)}return t}raycast(e,t){const n=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),Bl.copy(n.boundingSphere),Bl.applyMatrix4(r),Tr.copy(e.ray).recast(e.near),!(Bl.containsPoint(Tr.origin)===!1&&(Tr.intersectSphere(Bl,Kp)===null||Tr.origin.distanceToSquared(Kp)>(e.far-e.near)**2))&&(Qp.copy(r).invert(),Tr.copy(e.ray).applyMatrix4(Qp),!(n.boundingBox!==null&&Tr.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(e,t,Tr)))}_computeIntersections(e,t,n){let s;const r=this.geometry,o=this.material,a=r.index,l=r.attributes.position,c=r.attributes.uv,u=r.attributes.uv1,f=r.attributes.normal,d=r.groups,h=r.drawRange;if(a!==null)if(Array.isArray(o))for(let x=0,p=d.length;x<p;x++){const g=d[x],m=o[g.materialIndex],_=Math.max(g.start,h.start),A=Math.min(a.count,Math.min(g.start+g.count,h.start+h.count));for(let v=_,S=A;v<S;v+=3){const y=a.getX(v),M=a.getX(v+1),E=a.getX(v+2);s=Hl(this,m,e,n,c,u,f,y,M,E),s&&(s.faceIndex=Math.floor(v/3),s.face.materialIndex=g.materialIndex,t.push(s))}}else{const x=Math.max(0,h.start),p=Math.min(a.count,h.start+h.count);for(let g=x,m=p;g<m;g+=3){const _=a.getX(g),A=a.getX(g+1),v=a.getX(g+2);s=Hl(this,o,e,n,c,u,f,_,A,v),s&&(s.faceIndex=Math.floor(g/3),t.push(s))}}else if(l!==void 0)if(Array.isArray(o))for(let x=0,p=d.length;x<p;x++){const g=d[x],m=o[g.materialIndex],_=Math.max(g.start,h.start),A=Math.min(l.count,Math.min(g.start+g.count,h.start+h.count));for(let v=_,S=A;v<S;v+=3){const y=v,M=v+1,E=v+2;s=Hl(this,m,e,n,c,u,f,y,M,E),s&&(s.faceIndex=Math.floor(v/3),s.face.materialIndex=g.materialIndex,t.push(s))}}else{const x=Math.max(0,h.start),p=Math.min(l.count,h.start+h.count);for(let g=x,m=p;g<m;g+=3){const _=g,A=g+1,v=g+2;s=Hl(this,o,e,n,c,u,f,_,A,v),s&&(s.faceIndex=Math.floor(g/3),t.push(s))}}}}function gy(i,e,t,n,s,r,o,a){let l;if(e.side===Jn?l=n.intersectTriangle(o,r,s,!0,a):l=n.intersectTriangle(s,r,o,e.side===os,a),l===null)return null;kl.copy(a),kl.applyMatrix4(i.matrixWorld);const c=t.ray.origin.distanceTo(kl);return c<t.near||c>t.far?null:{distance:c,point:kl.clone(),object:i}}function Hl(i,e,t,n,s,r,o,a,l,c){i.getVertexPosition(a,Ul),i.getVertexPosition(l,Ol),i.getVertexPosition(c,Nl);const u=gy(i,e,t,n,Ul,Ol,Nl,jp);if(u){const f=new B;ki.getBarycoord(jp,Ul,Ol,Nl,f),s&&(u.uv=ki.getInterpolatedAttribute(s,a,l,c,f,new Ke)),r&&(u.uv1=ki.getInterpolatedAttribute(r,a,l,c,f,new Ke)),o&&(u.normal=ki.getInterpolatedAttribute(o,a,l,c,f,new B),u.normal.dot(n.direction)>0&&u.normal.multiplyScalar(-1));const d={a,b:l,c,normal:new B,materialIndex:0};ki.getNormal(Ul,Ol,Nl,d.normal),u.face=d,u.barycoord=f}return u}class ra extends Kn{constructor(e=1,t=1,n=1,s=1,r=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:n,widthSegments:s,heightSegments:r,depthSegments:o};const a=this;s=Math.floor(s),r=Math.floor(r),o=Math.floor(o);const l=[],c=[],u=[],f=[];let d=0,h=0;x("z","y","x",-1,-1,n,t,e,o,r,0),x("z","y","x",1,-1,n,t,-e,o,r,1),x("x","z","y",1,1,e,n,t,s,o,2),x("x","z","y",1,-1,e,n,-t,s,o,3),x("x","y","z",1,-1,e,t,n,s,r,4),x("x","y","z",-1,-1,e,t,-n,s,r,5),this.setIndex(l),this.setAttribute("position",new Ln(c,3)),this.setAttribute("normal",new Ln(u,3)),this.setAttribute("uv",new Ln(f,2));function x(p,g,m,_,A,v,S,y,M,E,b){const C=v/M,D=S/E,F=v/2,O=S/2,z=y/2,V=M+1,H=E+1;let q=0,G=0;const $=new B;for(let fe=0;fe<H;fe++){const Y=fe*D-O;for(let we=0;we<V;we++){const ze=we*C-F;$[p]=ze*_,$[g]=Y*A,$[m]=z,c.push($.x,$.y,$.z),$[p]=0,$[g]=0,$[m]=y>0?1:-1,u.push($.x,$.y,$.z),f.push(we/M),f.push(1-fe/E),q+=1}}for(let fe=0;fe<E;fe++)for(let Y=0;Y<M;Y++){const we=d+Y+V*fe,ze=d+Y+V*(fe+1),ke=d+(Y+1)+V*(fe+1),We=d+(Y+1)+V*fe;l.push(we,ze,We),l.push(ze,ke,We),G+=6}a.addGroup(h,G,b),h+=G,d+=q}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new ra(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}function Ko(i){const e={};for(const t in i){e[t]={};for(const n in i[t]){const s=i[t][n];s&&(s.isColor||s.isMatrix3||s.isMatrix4||s.isVector2||s.isVector3||s.isVector4||s.isTexture||s.isQuaternion)?s.isRenderTargetTexture?(ft("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][n]=null):e[t][n]=s.clone():Array.isArray(s)?e[t][n]=s.slice():e[t][n]=s}}return e}function zn(i){const e={};for(let t=0;t<i.length;t++){const n=Ko(i[t]);for(const s in n)e[s]=n[s]}return e}function xy(i){const e=[];for(let t=0;t<i.length;t++)e.push(i[t].clone());return e}function Yg(i){const e=i.getRenderTarget();return e===null?i.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:Et.workingColorSpace}const _y={clone:Ko,merge:zn};var vy=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,Ay=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class Yn extends hl{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=vy,this.fragmentShader=Ay,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=Ko(e.uniforms),this.uniformsGroups=xy(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this}toJSON(e){const t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(const s in this.uniforms){const o=this.uniforms[s].value;o&&o.isTexture?t.uniforms[s]={type:"t",value:o.toJSON(e).uuid}:o&&o.isColor?t.uniforms[s]={type:"c",value:o.getHex()}:o&&o.isVector2?t.uniforms[s]={type:"v2",value:o.toArray()}:o&&o.isVector3?t.uniforms[s]={type:"v3",value:o.toArray()}:o&&o.isVector4?t.uniforms[s]={type:"v4",value:o.toArray()}:o&&o.isMatrix3?t.uniforms[s]={type:"m3",value:o.toArray()}:o&&o.isMatrix4?t.uniforms[s]={type:"m4",value:o.toArray()}:t.uniforms[s]={value:o}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;const n={};for(const s in this.extensions)this.extensions[s]===!0&&(n[s]=!0);return Object.keys(n).length>0&&(t.extensions=n),t}}class Qg extends mn{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new st,this.projectionMatrix=new st,this.projectionMatrixInverse=new st,this.coordinateSystem=Zi,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const Ws=new B,$p=new Ke,Zp=new Ke;class Ti extends Qg{constructor(e=50,t=1,n=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=n,this.far=s,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const t=.5*this.getFilmHeight()/e;this.fov=el*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(Fa*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return el*2*Math.atan(Math.tan(Fa*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,n){Ws.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(Ws.x,Ws.y).multiplyScalar(-e/Ws.z),Ws.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(Ws.x,Ws.y).multiplyScalar(-e/Ws.z)}getViewSize(e,t){return this.getViewBounds(e,$p,Zp),t.subVectors(Zp,$p)}setViewOffset(e,t,n,s,r,o){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let t=e*Math.tan(Fa*.5*this.fov)/this.zoom,n=2*t,s=this.aspect*n,r=-.5*s;const o=this.view;if(this.view!==null&&this.view.enabled){const l=o.fullWidth,c=o.fullHeight;r+=o.offsetX*s/l,t-=o.offsetY*n/c,s*=o.width/l,n*=o.height/c}const a=this.filmOffset;a!==0&&(r+=e*a/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,t,t-n,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}}const xo=-90,_o=1;class Sy extends mn{constructor(e,t,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const s=new Ti(xo,_o,e,t);s.layers=this.layers,this.add(s);const r=new Ti(xo,_o,e,t);r.layers=this.layers,this.add(r);const o=new Ti(xo,_o,e,t);o.layers=this.layers,this.add(o);const a=new Ti(xo,_o,e,t);a.layers=this.layers,this.add(a);const l=new Ti(xo,_o,e,t);l.layers=this.layers,this.add(l);const c=new Ti(xo,_o,e,t);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){const e=this.coordinateSystem,t=this.children.concat(),[n,s,r,o,a,l]=t;for(const c of t)this.remove(c);if(e===Zi)n.up.set(0,1,0),n.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),o.up.set(0,0,1),o.lookAt(0,-1,0),a.up.set(0,1,0),a.lookAt(0,0,1),l.up.set(0,1,0),l.lookAt(0,0,-1);else if(e===Mc)n.up.set(0,-1,0),n.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),o.up.set(0,0,-1),o.lookAt(0,-1,0),a.up.set(0,-1,0),a.lookAt(0,0,1),l.up.set(0,-1,0),l.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const c of t)this.add(c),c.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[r,o,a,l,c,u]=this.children,f=e.getRenderTarget(),d=e.getActiveCubeFace(),h=e.getActiveMipmapLevel(),x=e.xr.enabled;e.xr.enabled=!1;const p=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,e.setRenderTarget(n,0,s),e.render(t,r),e.setRenderTarget(n,1,s),e.render(t,o),e.setRenderTarget(n,2,s),e.render(t,a),e.setRenderTarget(n,3,s),e.render(t,l),e.setRenderTarget(n,4,s),e.render(t,c),n.texture.generateMipmaps=p,e.setRenderTarget(n,5,s),e.render(t,u),e.setRenderTarget(f,d,h),e.xr.enabled=x,n.texture.needsPMREMUpdate=!0}}class Kg extends qn{constructor(e=[],t=Xo,n,s,r,o,a,l,c,u){super(e,t,n,s,r,o,a,l,c,u),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class yy extends cr{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;const n={width:e,height:e,depth:1},s=[n,n,n,n,n,n];this.texture=new Kg(s),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

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
			`},s=new ra(5,5,5),r=new Yn({name:"CubemapFromEquirect",uniforms:Ko(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:Jn,blending:Ds});r.uniforms.tEquirect.value=t;const o=new hn(s,r),a=t.minFilter;return t.minFilter===kr&&(t.minFilter=Ri),new Sy(1,10,this).update(e,o),t.minFilter=a,o.geometry.dispose(),o.material.dispose(),this}clear(e,t=!0,n=!0,s=!0){const r=e.getRenderTarget();for(let o=0;o<6;o++)e.setRenderTarget(this,o),e.clear(t,n,s);e.setRenderTarget(r)}}class Vl extends mn{constructor(){super(),this.isGroup=!0,this.type="Group"}}const by={type:"move"};class Nu{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Vl,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Vl,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new B,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new B),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Vl,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new B,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new B),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const t=this._hand;if(t)for(const n of e.hand.values())this._getHandJoint(t,n)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,n){let s=null,r=null,o=null;const a=this._targetRay,l=this._grip,c=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(c&&e.hand){o=!0;for(const p of e.hand.values()){const g=t.getJointPose(p,n),m=this._getHandJoint(c,p);g!==null&&(m.matrix.fromArray(g.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,m.jointRadius=g.radius),m.visible=g!==null}const u=c.joints["index-finger-tip"],f=c.joints["thumb-tip"],d=u.position.distanceTo(f.position),h=.02,x=.005;c.inputState.pinching&&d>h+x?(c.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!c.inputState.pinching&&d<=h-x&&(c.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else l!==null&&e.gripSpace&&(r=t.getPose(e.gripSpace,n),r!==null&&(l.matrix.fromArray(r.transform.matrix),l.matrix.decompose(l.position,l.rotation,l.scale),l.matrixWorldNeedsUpdate=!0,r.linearVelocity?(l.hasLinearVelocity=!0,l.linearVelocity.copy(r.linearVelocity)):l.hasLinearVelocity=!1,r.angularVelocity?(l.hasAngularVelocity=!0,l.angularVelocity.copy(r.angularVelocity)):l.hasAngularVelocity=!1));a!==null&&(s=t.getPose(e.targetRaySpace,n),s===null&&r!==null&&(s=r),s!==null&&(a.matrix.fromArray(s.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),a.matrixWorldNeedsUpdate=!0,s.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(s.linearVelocity)):a.hasLinearVelocity=!1,s.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(s.angularVelocity)):a.hasAngularVelocity=!1,this.dispatchEvent(by)))}return a!==null&&(a.visible=s!==null),l!==null&&(l.visible=r!==null),c!==null&&(c.visible=o!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){const n=new Vl;n.matrixAutoUpdate=!1,n.visible=!1,e.joints[t.jointName]=n,e.add(n)}return e.joints[t.jointName]}}class My extends mn{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Wi,this.environmentIntensity=1,this.environmentRotation=new Wi,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}}class ys extends qn{constructor(e=null,t=1,n=1,s,r,o,a,l,c=xi,u=xi,f,d){super(null,o,a,l,c,u,s,r,f,d),this.isDataTexture=!0,this.image={data:e,width:t,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class Cy extends Li{constructor(e,t,n,s=1){super(e,t,n),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=s}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}toJSON(){const e=super.toJSON();return e.meshPerAttribute=this.meshPerAttribute,e.isInstancedBufferAttribute=!0,e}}const zu=new B,Ty=new B,Ey=new lt;class Ks{constructor(e=new B(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,n,s){return this.normal.set(e,t,n),this.constant=s,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,n){const s=zu.subVectors(n,t).cross(Ty.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t){const n=e.delta(zu),s=this.normal.dot(n);if(s===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;const r=-(e.start.dot(this.normal)+this.constant)/s;return r<0||r>1?null:t.copy(e.start).addScaledVector(n,r)}intersectsLine(e){const t=this.distanceToPoint(e.start),n=this.distanceToPoint(e.end);return t<0&&n>0||n<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){const n=t||Ey.getNormalMatrix(e),s=this.coplanarPoint(zu).applyMatrix4(e),r=this.normal.applyMatrix3(n).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const Er=new jc,wy=new Ke(.5,.5),Gl=new B;class jg{constructor(e=new Ks,t=new Ks,n=new Ks,s=new Ks,r=new Ks,o=new Ks){this.planes=[e,t,n,s,r,o]}set(e,t,n,s,r,o){const a=this.planes;return a[0].copy(e),a[1].copy(t),a[2].copy(n),a[3].copy(s),a[4].copy(r),a[5].copy(o),this}copy(e){const t=this.planes;for(let n=0;n<6;n++)t[n].copy(e.planes[n]);return this}setFromProjectionMatrix(e,t=Zi,n=!1){const s=this.planes,r=e.elements,o=r[0],a=r[1],l=r[2],c=r[3],u=r[4],f=r[5],d=r[6],h=r[7],x=r[8],p=r[9],g=r[10],m=r[11],_=r[12],A=r[13],v=r[14],S=r[15];if(s[0].setComponents(c-o,h-u,m-x,S-_).normalize(),s[1].setComponents(c+o,h+u,m+x,S+_).normalize(),s[2].setComponents(c+a,h+f,m+p,S+A).normalize(),s[3].setComponents(c-a,h-f,m-p,S-A).normalize(),n)s[4].setComponents(l,d,g,v).normalize(),s[5].setComponents(c-l,h-d,m-g,S-v).normalize();else if(s[4].setComponents(c-l,h-d,m-g,S-v).normalize(),t===Zi)s[5].setComponents(c+l,h+d,m+g,S+v).normalize();else if(t===Mc)s[5].setComponents(l,d,g,v).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),Er.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),Er.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(Er)}intersectsSprite(e){Er.center.set(0,0,0);const t=wy.distanceTo(e.center);return Er.radius=.7071067811865476+t,Er.applyMatrix4(e.matrixWorld),this.intersectsSphere(Er)}intersectsSphere(e){const t=this.planes,n=e.center,s=-e.radius;for(let r=0;r<6;r++)if(t[r].distanceToPoint(n)<s)return!1;return!0}intersectsBox(e){const t=this.planes;for(let n=0;n<6;n++){const s=t[n];if(Gl.x=s.normal.x>0?e.max.x:e.min.x,Gl.y=s.normal.y>0?e.max.y:e.min.y,Gl.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(Gl)<0)return!1}return!0}containsPoint(e){const t=this.planes;for(let n=0;n<6;n++)if(t[n].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class Ry extends hl{constructor(e){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new bt(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.size=e.size,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}}const Jp=new st,fd=new sh,Wl=new jc,Xl=new B;class Iy extends mn{constructor(e=new Kn,t=new Ry){super(),this.isPoints=!0,this.type="Points",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}raycast(e,t){const n=this.geometry,s=this.matrixWorld,r=e.params.Points.threshold,o=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),Wl.copy(n.boundingSphere),Wl.applyMatrix4(s),Wl.radius+=r,e.ray.intersectsSphere(Wl)===!1)return;Jp.copy(s).invert(),fd.copy(e.ray).applyMatrix4(Jp);const a=r/((this.scale.x+this.scale.y+this.scale.z)/3),l=a*a,c=n.index,f=n.attributes.position;if(c!==null){const d=Math.max(0,o.start),h=Math.min(c.count,o.start+o.count);for(let x=d,p=h;x<p;x++){const g=c.getX(x);Xl.fromBufferAttribute(f,g),em(Xl,g,l,s,e,t,this)}}else{const d=Math.max(0,o.start),h=Math.min(f.count,o.start+o.count);for(let x=d,p=h;x<p;x++)Xl.fromBufferAttribute(f,x),em(Xl,x,l,s,e,t,this)}}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}}function em(i,e,t,n,s,r,o){const a=fd.distanceSqToPoint(i);if(a<t){const l=new B;fd.closestPointToPoint(i,l),l.applyMatrix4(n);const c=s.ray.origin.distanceTo(l);if(c<s.near||c>s.far)return;r.push({distance:c,distanceToRay:Math.sqrt(a),point:l,index:e,face:null,faceIndex:null,barycoord:null,object:o})}}class rh extends qn{constructor(e,t,n=Ii,s,r,o,a=xi,l=xi,c,u=Yo,f=1){if(u!==Yo&&u!==Za)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const d={width:e,height:t,depth:f};super(d,s,r,o,a,l,u,n,c),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new ih(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}}class $g extends qn{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class nl extends Kn{constructor(e=1,t=1,n=1,s=32,r=1,o=!1,a=0,l=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:n,radialSegments:s,heightSegments:r,openEnded:o,thetaStart:a,thetaLength:l};const c=this;s=Math.floor(s),r=Math.floor(r);const u=[],f=[],d=[],h=[];let x=0;const p=[],g=n/2;let m=0;_(),o===!1&&(e>0&&A(!0),t>0&&A(!1)),this.setIndex(u),this.setAttribute("position",new Ln(f,3)),this.setAttribute("normal",new Ln(d,3)),this.setAttribute("uv",new Ln(h,2));function _(){const v=new B,S=new B;let y=0;const M=(t-e)/n;for(let E=0;E<=r;E++){const b=[],C=E/r,D=C*(t-e)+e;for(let F=0;F<=s;F++){const O=F/s,z=O*l+a,V=Math.sin(z),H=Math.cos(z);S.x=D*V,S.y=-C*n+g,S.z=D*H,f.push(S.x,S.y,S.z),v.set(V,M,H).normalize(),d.push(v.x,v.y,v.z),h.push(O,1-C),b.push(x++)}p.push(b)}for(let E=0;E<s;E++)for(let b=0;b<r;b++){const C=p[b][E],D=p[b+1][E],F=p[b+1][E+1],O=p[b][E+1];(e>0||b!==0)&&(u.push(C,D,O),y+=3),(t>0||b!==r-1)&&(u.push(D,F,O),y+=3)}c.addGroup(m,y,0),m+=y}function A(v){const S=x,y=new Ke,M=new B;let E=0;const b=v===!0?e:t,C=v===!0?1:-1;for(let F=1;F<=s;F++)f.push(0,g*C,0),d.push(0,C,0),h.push(.5,.5),x++;const D=x;for(let F=0;F<=s;F++){const z=F/s*l+a,V=Math.cos(z),H=Math.sin(z);M.x=b*H,M.y=g*C,M.z=b*V,f.push(M.x,M.y,M.z),d.push(0,C,0),y.x=V*.5+.5,y.y=H*.5*C+.5,h.push(y.x,y.y),x++}for(let F=0;F<s;F++){const O=S+F,z=D+F;v===!0?u.push(z,z+1,O):u.push(z+1,z,O),E+=3}c.addGroup(m,E,v===!0?1:2),m+=E}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new nl(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class oh extends nl{constructor(e=1,t=1,n=32,s=1,r=!1,o=0,a=Math.PI*2){super(0,e,t,n,s,r,o,a),this.type="ConeGeometry",this.parameters={radius:e,height:t,radialSegments:n,heightSegments:s,openEnded:r,thetaStart:o,thetaLength:a}}static fromJSON(e){return new oh(e.radius,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class Dy{constructor(){this.type="Curve",this.arcLengthDivisions=200,this.needsUpdate=!1,this.cacheArcLengths=null}getPoint(){ft("Curve: .getPoint() not implemented.")}getPointAt(e,t){const n=this.getUtoTmapping(e);return this.getPoint(n,t)}getPoints(e=5){const t=[];for(let n=0;n<=e;n++)t.push(this.getPoint(n/e));return t}getSpacedPoints(e=5){const t=[];for(let n=0;n<=e;n++)t.push(this.getPointAt(n/e));return t}getLength(){const e=this.getLengths();return e[e.length-1]}getLengths(e=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===e+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;const t=[];let n,s=this.getPoint(0),r=0;t.push(0);for(let o=1;o<=e;o++)n=this.getPoint(o/e),r+=n.distanceTo(s),t.push(r),s=n;return this.cacheArcLengths=t,t}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(e,t=null){const n=this.getLengths();let s=0;const r=n.length;let o;t?o=t:o=e*n[r-1];let a=0,l=r-1,c;for(;a<=l;)if(s=Math.floor(a+(l-a)/2),c=n[s]-o,c<0)a=s+1;else if(c>0)l=s-1;else{l=s;break}if(s=l,n[s]===o)return s/(r-1);const u=n[s],d=n[s+1]-u,h=(o-u)/d;return(s+h)/(r-1)}getTangent(e,t){let s=e-1e-4,r=e+1e-4;s<0&&(s=0),r>1&&(r=1);const o=this.getPoint(s),a=this.getPoint(r),l=t||(o.isVector2?new Ke:new B);return l.copy(a).sub(o).normalize(),l}getTangentAt(e,t){const n=this.getUtoTmapping(e);return this.getTangent(n,t)}computeFrenetFrames(e,t=!1){const n=new B,s=[],r=[],o=[],a=new B,l=new st;for(let h=0;h<=e;h++){const x=h/e;s[h]=this.getTangentAt(x,new B)}r[0]=new B,o[0]=new B;let c=Number.MAX_VALUE;const u=Math.abs(s[0].x),f=Math.abs(s[0].y),d=Math.abs(s[0].z);u<=c&&(c=u,n.set(1,0,0)),f<=c&&(c=f,n.set(0,1,0)),d<=c&&n.set(0,0,1),a.crossVectors(s[0],n).normalize(),r[0].crossVectors(s[0],a),o[0].crossVectors(s[0],r[0]);for(let h=1;h<=e;h++){if(r[h]=r[h-1].clone(),o[h]=o[h-1].clone(),a.crossVectors(s[h-1],s[h]),a.length()>Number.EPSILON){a.normalize();const x=Math.acos(gt(s[h-1].dot(s[h]),-1,1));r[h].applyMatrix4(l.makeRotationAxis(a,x))}o[h].crossVectors(s[h],r[h])}if(t===!0){let h=Math.acos(gt(r[0].dot(r[e]),-1,1));h/=e,s[0].dot(a.crossVectors(r[0],r[e]))>0&&(h=-h);for(let x=1;x<=e;x++)r[x].applyMatrix4(l.makeRotationAxis(s[x],h*x)),o[x].crossVectors(s[x],r[x])}return{tangents:s,normals:r,binormals:o}}clone(){return new this.constructor().copy(this)}copy(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}toJSON(){const e={metadata:{version:4.7,type:"Curve",generator:"Curve.toJSON"}};return e.arcLengthDivisions=this.arcLengthDivisions,e.type=this.type,e}fromJSON(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}}function ah(){let i=0,e=0,t=0,n=0;function s(r,o,a,l){i=r,e=a,t=-3*r+3*o-2*a-l,n=2*r-2*o+a+l}return{initCatmullRom:function(r,o,a,l,c){s(o,a,c*(a-r),c*(l-o))},initNonuniformCatmullRom:function(r,o,a,l,c,u,f){let d=(o-r)/c-(a-r)/(c+u)+(a-o)/u,h=(a-o)/u-(l-o)/(u+f)+(l-a)/f;d*=u,h*=u,s(o,a,d,h)},calc:function(r){const o=r*r,a=o*r;return i+e*r+t*o+n*a}}}const ql=new B,ku=new ah,Hu=new ah,Vu=new ah;class tm extends Dy{constructor(e=[],t=!1,n="centripetal",s=.5){super(),this.isCatmullRomCurve3=!0,this.type="CatmullRomCurve3",this.points=e,this.closed=t,this.curveType=n,this.tension=s}getPoint(e,t=new B){const n=t,s=this.points,r=s.length,o=(r-(this.closed?0:1))*e;let a=Math.floor(o),l=o-a;this.closed?a+=a>0?0:(Math.floor(Math.abs(a)/r)+1)*r:l===0&&a===r-1&&(a=r-2,l=1);let c,u;this.closed||a>0?c=s[(a-1)%r]:(ql.subVectors(s[0],s[1]).add(s[0]),c=ql);const f=s[a%r],d=s[(a+1)%r];if(this.closed||a+2<r?u=s[(a+2)%r]:(ql.subVectors(s[r-1],s[r-2]).add(s[r-1]),u=ql),this.curveType==="centripetal"||this.curveType==="chordal"){const h=this.curveType==="chordal"?.5:.25;let x=Math.pow(c.distanceToSquared(f),h),p=Math.pow(f.distanceToSquared(d),h),g=Math.pow(d.distanceToSquared(u),h);p<1e-4&&(p=1),x<1e-4&&(x=p),g<1e-4&&(g=p),ku.initNonuniformCatmullRom(c.x,f.x,d.x,u.x,x,p,g),Hu.initNonuniformCatmullRom(c.y,f.y,d.y,u.y,x,p,g),Vu.initNonuniformCatmullRom(c.z,f.z,d.z,u.z,x,p,g)}else this.curveType==="catmullrom"&&(ku.initCatmullRom(c.x,f.x,d.x,u.x,this.tension),Hu.initCatmullRom(c.y,f.y,d.y,u.y,this.tension),Vu.initCatmullRom(c.z,f.z,d.z,u.z,this.tension));return n.set(ku.calc(l),Hu.calc(l),Vu.calc(l)),n}copy(e){super.copy(e),this.points=[];for(let t=0,n=e.points.length;t<n;t++){const s=e.points[t];this.points.push(s.clone())}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}toJSON(){const e=super.toJSON();e.points=[];for(let t=0,n=this.points.length;t<n;t++){const s=this.points[t];e.points.push(s.toArray())}return e.closed=this.closed,e.curveType=this.curveType,e.tension=this.tension,e}fromJSON(e){super.fromJSON(e),this.points=[];for(let t=0,n=e.points.length;t<n;t++){const s=e.points[t];this.points.push(new B().fromArray(s))}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}}class jo extends Kn{constructor(e=1,t=1,n=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:n,heightSegments:s};const r=e/2,o=t/2,a=Math.floor(n),l=Math.floor(s),c=a+1,u=l+1,f=e/a,d=t/l,h=[],x=[],p=[],g=[];for(let m=0;m<u;m++){const _=m*d-o;for(let A=0;A<c;A++){const v=A*f-r;x.push(v,-_,0),p.push(0,0,1),g.push(A/a),g.push(1-m/l)}}for(let m=0;m<l;m++)for(let _=0;_<a;_++){const A=_+c*m,v=_+c*(m+1),S=_+1+c*(m+1),y=_+1+c*m;h.push(A,v,y),h.push(v,S,y)}this.setIndex(h),this.setAttribute("position",new Ln(x,3)),this.setAttribute("normal",new Ln(p,3)),this.setAttribute("uv",new Ln(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new jo(e.width,e.height,e.widthSegments,e.heightSegments)}}class Tc extends Kn{constructor(e=1,t=32,n=16,s=0,r=Math.PI*2,o=0,a=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:e,widthSegments:t,heightSegments:n,phiStart:s,phiLength:r,thetaStart:o,thetaLength:a},t=Math.max(3,Math.floor(t)),n=Math.max(2,Math.floor(n));const l=Math.min(o+a,Math.PI);let c=0;const u=[],f=new B,d=new B,h=[],x=[],p=[],g=[];for(let m=0;m<=n;m++){const _=[],A=m/n;let v=0;m===0&&o===0?v=.5/t:m===n&&l===Math.PI&&(v=-.5/t);for(let S=0;S<=t;S++){const y=S/t;f.x=-e*Math.cos(s+y*r)*Math.sin(o+A*a),f.y=e*Math.cos(o+A*a),f.z=e*Math.sin(s+y*r)*Math.sin(o+A*a),x.push(f.x,f.y,f.z),d.copy(f).normalize(),p.push(d.x,d.y,d.z),g.push(y+v,1-A),_.push(c++)}u.push(_)}for(let m=0;m<n;m++)for(let _=0;_<t;_++){const A=u[m][_+1],v=u[m][_],S=u[m+1][_],y=u[m+1][_+1];(m!==0||o>0)&&h.push(A,v,y),(m!==n-1||l<Math.PI)&&h.push(v,S,y)}this.setIndex(h),this.setAttribute("position",new Ln(x,3)),this.setAttribute("normal",new Ln(p,3)),this.setAttribute("uv",new Ln(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Tc(e.radius,e.widthSegments,e.heightSegments,e.phiStart,e.phiLength,e.thetaStart,e.thetaLength)}}class Py extends hl{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=SS,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class Fy extends hl{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}class lh extends Qg{constructor(e=-1,t=1,n=1,s=-1,r=.1,o=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=n,this.bottom=s,this.near=r,this.far=o,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,n,s,r,o){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,s=(this.top+this.bottom)/2;let r=n-e,o=n+e,a=s+t,l=s-t;if(this.view!==null&&this.view.enabled){const c=(this.right-this.left)/this.view.fullWidth/this.zoom,u=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=c*this.view.offsetX,o=r+c*this.view.width,a-=u*this.view.offsetY,l=a-u*this.view.height}this.projectionMatrix.makeOrthographic(r,o,a,l,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}}class Ly extends Kn{constructor(){super(),this.isInstancedBufferGeometry=!0,this.type="InstancedBufferGeometry",this.instanceCount=1/0}copy(e){return super.copy(e),this.instanceCount=e.instanceCount,this}toJSON(){const e=super.toJSON();return e.instanceCount=this.instanceCount,e.isInstancedBufferGeometry=!0,e}}class By extends Ti{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}class nm{constructor(e=1,t=0,n=0){this.radius=e,this.phi=t,this.theta=n}set(e,t,n){return this.radius=e,this.phi=t,this.theta=n,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=gt(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,n){return this.radius=Math.sqrt(e*e+t*t+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,n),this.phi=Math.acos(gt(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}function im(i,e,t,n){const s=Uy(n);switch(t){case Og:return i*e;case zg:return i*e/s.components*s.byteLength;case Kc:return i*e/s.components*s.byteLength;case eh:return i*e*2/s.components*s.byteLength;case th:return i*e*2/s.components*s.byteLength;case Ng:return i*e*3/s.components*s.byteLength;case Xn:return i*e*4/s.components*s.byteLength;case Lo:return i*e*4/s.components*s.byteLength;case lc:case cc:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case uc:case fc:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Nf:case kf:return Math.max(i,16)*Math.max(e,8)/4;case Of:case zf:return Math.max(i,8)*Math.max(e,8)/2;case Hf:case Vf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case Gf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Wf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Xf:return Math.floor((i+4)/5)*Math.floor((e+3)/4)*16;case qf:return Math.floor((i+4)/5)*Math.floor((e+4)/5)*16;case Yf:return Math.floor((i+5)/6)*Math.floor((e+4)/5)*16;case Qf:return Math.floor((i+5)/6)*Math.floor((e+5)/6)*16;case Kf:return Math.floor((i+7)/8)*Math.floor((e+4)/5)*16;case jf:return Math.floor((i+7)/8)*Math.floor((e+5)/6)*16;case $f:return Math.floor((i+7)/8)*Math.floor((e+7)/8)*16;case Zf:return Math.floor((i+9)/10)*Math.floor((e+4)/5)*16;case Jf:return Math.floor((i+9)/10)*Math.floor((e+5)/6)*16;case ed:return Math.floor((i+9)/10)*Math.floor((e+7)/8)*16;case td:return Math.floor((i+9)/10)*Math.floor((e+9)/10)*16;case nd:return Math.floor((i+11)/12)*Math.floor((e+9)/10)*16;case id:return Math.floor((i+11)/12)*Math.floor((e+11)/12)*16;case sd:case rd:case od:return Math.ceil(i/4)*Math.ceil(e/4)*16;case ad:case ld:return Math.ceil(i/4)*Math.ceil(e/4)*8;case cd:case ud:return Math.ceil(i/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function Uy(i){switch(i){case as:case Fg:return{byteLength:1,components:1};case ja:case Lg:case jr:return{byteLength:2,components:1};case Zd:case Jd:return{byteLength:2,components:4};case Ii:case $d:case Hi:return{byteLength:4,components:1};case Bg:case Ug:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${i}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:jd}}));typeof window<"u"&&(window.__THREE__?ft("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=jd);function Zg(){let i=null,e=!1,t=null,n=null;function s(r,o){t(r,o),n=i.requestAnimationFrame(s)}return{start:function(){e!==!0&&t!==null&&(n=i.requestAnimationFrame(s),e=!0)},stop:function(){i.cancelAnimationFrame(n),e=!1},setAnimationLoop:function(r){t=r},setContext:function(r){i=r}}}function Oy(i){const e=new WeakMap;function t(a,l){const c=a.array,u=a.usage,f=c.byteLength,d=i.createBuffer();i.bindBuffer(l,d),i.bufferData(l,c,u),a.onUploadCallback();let h;if(c instanceof Float32Array)h=i.FLOAT;else if(typeof Float16Array<"u"&&c instanceof Float16Array)h=i.HALF_FLOAT;else if(c instanceof Uint16Array)a.isFloat16BufferAttribute?h=i.HALF_FLOAT:h=i.UNSIGNED_SHORT;else if(c instanceof Int16Array)h=i.SHORT;else if(c instanceof Uint32Array)h=i.UNSIGNED_INT;else if(c instanceof Int32Array)h=i.INT;else if(c instanceof Int8Array)h=i.BYTE;else if(c instanceof Uint8Array)h=i.UNSIGNED_BYTE;else if(c instanceof Uint8ClampedArray)h=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+c);return{buffer:d,type:h,bytesPerElement:c.BYTES_PER_ELEMENT,version:a.version,size:f}}function n(a,l,c){const u=l.array,f=l.updateRanges;if(i.bindBuffer(c,a),f.length===0)i.bufferSubData(c,0,u);else{f.sort((h,x)=>h.start-x.start);let d=0;for(let h=1;h<f.length;h++){const x=f[d],p=f[h];p.start<=x.start+x.count+1?x.count=Math.max(x.count,p.start+p.count-x.start):(++d,f[d]=p)}f.length=d+1;for(let h=0,x=f.length;h<x;h++){const p=f[h];i.bufferSubData(c,p.start*u.BYTES_PER_ELEMENT,u,p.start,p.count)}l.clearUpdateRanges()}l.onUploadCallback()}function s(a){return a.isInterleavedBufferAttribute&&(a=a.data),e.get(a)}function r(a){a.isInterleavedBufferAttribute&&(a=a.data);const l=e.get(a);l&&(i.deleteBuffer(l.buffer),e.delete(a))}function o(a,l){if(a.isInterleavedBufferAttribute&&(a=a.data),a.isGLBufferAttribute){const u=e.get(a);(!u||u.version<a.version)&&e.set(a,{buffer:a.buffer,type:a.type,bytesPerElement:a.elementSize,version:a.version});return}const c=e.get(a);if(c===void 0)e.set(a,t(a,l));else if(c.version<a.version){if(c.size!==a.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(c.buffer,a,l),c.version=a.version}}return{get:s,remove:r,update:o}}var Ny=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,zy=`#ifdef USE_ALPHAHASH
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
#endif`,ky=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,Hy=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Vy=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,Gy=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,Wy=`#ifdef USE_AOMAP
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
#endif`,Xy=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,qy=`#ifdef USE_BATCHING
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
#endif`,Yy=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,Qy=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Ky=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,jy=`float G_BlinnPhong_Implicit( ) {
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
} // validated`,$y=`#ifdef USE_IRIDESCENCE
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
#endif`,Zy=`#ifdef USE_BUMPMAP
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
#endif`,Jy=`#if NUM_CLIPPING_PLANES > 0
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
#endif`,eb=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,tb=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,nb=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,ib=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,sb=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,rb=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,ob=`#if defined( USE_COLOR_ALPHA )
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
#endif`,ab=`#define PI 3.141592653589793
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
} // validated`,lb=`#ifdef ENVMAP_TYPE_CUBE_UV
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
#endif`,cb=`vec3 transformedNormal = objectNormal;
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
#endif`,ub=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,fb=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,db=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,hb=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,pb="gl_FragColor = linearToOutputTexel( gl_FragColor );",mb=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,gb=`#ifdef USE_ENVMAP
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
#endif`,xb=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,_b=`#ifdef USE_ENVMAP
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
#endif`,vb=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,Ab=`#ifdef USE_ENVMAP
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
#endif`,Sb=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,yb=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,bb=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,Mb=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,Cb=`#ifdef USE_GRADIENTMAP
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
}`,Tb=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,Eb=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,wb=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,Rb=`uniform bool receiveShadow;
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
#endif`,Ib=`#ifdef USE_ENVMAP
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
#endif`,Db=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,Pb=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,Fb=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,Lb=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,Bb=`PhysicalMaterial material;
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
#endif`,Ub=`uniform sampler2D dfgLUT;
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
}`,Ob=`
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
#endif`,Nb=`#if defined( RE_IndirectDiffuse )
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
#endif`,zb=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,kb=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,Hb=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Vb=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Gb=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,Wb=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Xb=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,qb=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
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
#endif`,Yb=`#if defined( USE_POINTS_UV )
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
#endif`,Qb=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Kb=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,jb=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,$b=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Zb=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Jb=`#ifdef USE_MORPHTARGETS
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
#endif`,eM=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,tM=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
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
vec3 nonPerturbedNormal = normal;`,nM=`#ifdef USE_NORMALMAP_OBJECTSPACE
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
#endif`,iM=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,sM=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,rM=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,oM=`#ifdef USE_NORMALMAP
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
#endif`,aM=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,lM=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,cM=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,uM=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,fM=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,dM=`vec3 packNormalToRGB( const in vec3 normal ) {
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
}`,hM=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,pM=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,mM=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,gM=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,xM=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,_M=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,vM=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,AM=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,SM=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
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
#endif`,yM=`float getShadowMask() {
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
}`,bM=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,MM=`#ifdef USE_SKINNING
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
#endif`,CM=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,TM=`#ifdef USE_SKINNING
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
#endif`,EM=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,wM=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,RM=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,IM=`#ifndef saturate
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
vec3 CustomToneMapping( vec3 color ) { return color; }`,DM=`#ifdef USE_TRANSMISSION
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
#endif`,PM=`#ifdef USE_TRANSMISSION
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
#endif`,FM=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,LM=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,BM=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,UM=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const OM=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,NM=`uniform sampler2D t2D;
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
}`,zM=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,kM=`#ifdef ENVMAP_TYPE_CUBE
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
}`,HM=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,VM=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,GM=`#include <common>
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
}`,WM=`#if DEPTH_PACKING == 3200
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
}`,XM=`#define DISTANCE
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
}`,qM=`#define DISTANCE
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
}`,YM=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,QM=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,KM=`uniform float scale;
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
}`,jM=`uniform vec3 diffuse;
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
}`,$M=`#include <common>
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
}`,ZM=`uniform vec3 diffuse;
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
}`,JM=`#define LAMBERT
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
}`,eC=`#define LAMBERT
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
}`,tC=`#define MATCAP
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
}`,nC=`#define MATCAP
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
}`,iC=`#define NORMAL
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
}`,sC=`#define NORMAL
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
}`,rC=`#define PHONG
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
}`,oC=`#define PHONG
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
}`,aC=`#define STANDARD
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
}`,lC=`#define STANDARD
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
}`,cC=`#define TOON
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
}`,uC=`#define TOON
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
}`,fC=`uniform float size;
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
}`,dC=`uniform vec3 diffuse;
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
}`,hC=`#include <common>
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
}`,pC=`uniform vec3 color;
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
}`,mC=`uniform float rotation;
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
}`,gC=`uniform vec3 diffuse;
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
}`,_t={alphahash_fragment:Ny,alphahash_pars_fragment:zy,alphamap_fragment:ky,alphamap_pars_fragment:Hy,alphatest_fragment:Vy,alphatest_pars_fragment:Gy,aomap_fragment:Wy,aomap_pars_fragment:Xy,batching_pars_vertex:qy,batching_vertex:Yy,begin_vertex:Qy,beginnormal_vertex:Ky,bsdfs:jy,iridescence_fragment:$y,bumpmap_pars_fragment:Zy,clipping_planes_fragment:Jy,clipping_planes_pars_fragment:eb,clipping_planes_pars_vertex:tb,clipping_planes_vertex:nb,color_fragment:ib,color_pars_fragment:sb,color_pars_vertex:rb,color_vertex:ob,common:ab,cube_uv_reflection_fragment:lb,defaultnormal_vertex:cb,displacementmap_pars_vertex:ub,displacementmap_vertex:fb,emissivemap_fragment:db,emissivemap_pars_fragment:hb,colorspace_fragment:pb,colorspace_pars_fragment:mb,envmap_fragment:gb,envmap_common_pars_fragment:xb,envmap_pars_fragment:_b,envmap_pars_vertex:vb,envmap_physical_pars_fragment:Ib,envmap_vertex:Ab,fog_vertex:Sb,fog_pars_vertex:yb,fog_fragment:bb,fog_pars_fragment:Mb,gradientmap_pars_fragment:Cb,lightmap_pars_fragment:Tb,lights_lambert_fragment:Eb,lights_lambert_pars_fragment:wb,lights_pars_begin:Rb,lights_toon_fragment:Db,lights_toon_pars_fragment:Pb,lights_phong_fragment:Fb,lights_phong_pars_fragment:Lb,lights_physical_fragment:Bb,lights_physical_pars_fragment:Ub,lights_fragment_begin:Ob,lights_fragment_maps:Nb,lights_fragment_end:zb,logdepthbuf_fragment:kb,logdepthbuf_pars_fragment:Hb,logdepthbuf_pars_vertex:Vb,logdepthbuf_vertex:Gb,map_fragment:Wb,map_pars_fragment:Xb,map_particle_fragment:qb,map_particle_pars_fragment:Yb,metalnessmap_fragment:Qb,metalnessmap_pars_fragment:Kb,morphinstance_vertex:jb,morphcolor_vertex:$b,morphnormal_vertex:Zb,morphtarget_pars_vertex:Jb,morphtarget_vertex:eM,normal_fragment_begin:tM,normal_fragment_maps:nM,normal_pars_fragment:iM,normal_pars_vertex:sM,normal_vertex:rM,normalmap_pars_fragment:oM,clearcoat_normal_fragment_begin:aM,clearcoat_normal_fragment_maps:lM,clearcoat_pars_fragment:cM,iridescence_pars_fragment:uM,opaque_fragment:fM,packing:dM,premultiplied_alpha_fragment:hM,project_vertex:pM,dithering_fragment:mM,dithering_pars_fragment:gM,roughnessmap_fragment:xM,roughnessmap_pars_fragment:_M,shadowmap_pars_fragment:vM,shadowmap_pars_vertex:AM,shadowmap_vertex:SM,shadowmask_pars_fragment:yM,skinbase_vertex:bM,skinning_pars_vertex:MM,skinning_vertex:CM,skinnormal_vertex:TM,specularmap_fragment:EM,specularmap_pars_fragment:wM,tonemapping_fragment:RM,tonemapping_pars_fragment:IM,transmission_fragment:DM,transmission_pars_fragment:PM,uv_pars_fragment:FM,uv_pars_vertex:LM,uv_vertex:BM,worldpos_vertex:UM,background_vert:OM,background_frag:NM,backgroundCube_vert:zM,backgroundCube_frag:kM,cube_vert:HM,cube_frag:VM,depth_vert:GM,depth_frag:WM,distanceRGBA_vert:XM,distanceRGBA_frag:qM,equirect_vert:YM,equirect_frag:QM,linedashed_vert:KM,linedashed_frag:jM,meshbasic_vert:$M,meshbasic_frag:ZM,meshlambert_vert:JM,meshlambert_frag:eC,meshmatcap_vert:tC,meshmatcap_frag:nC,meshnormal_vert:iC,meshnormal_frag:sC,meshphong_vert:rC,meshphong_frag:oC,meshphysical_vert:aC,meshphysical_frag:lC,meshtoon_vert:cC,meshtoon_frag:uC,points_vert:fC,points_frag:dC,shadow_vert:hC,shadow_frag:pC,sprite_vert:mC,sprite_frag:gC},Ne={common:{diffuse:{value:new bt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new lt},alphaMap:{value:null},alphaMapTransform:{value:new lt},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new lt}},envmap:{envMap:{value:null},envMapRotation:{value:new lt},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new lt}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new lt}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new lt},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new lt},normalScale:{value:new Ke(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new lt},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new lt}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new lt}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new lt}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new bt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new bt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new lt},alphaTest:{value:0},uvTransform:{value:new lt}},sprite:{diffuse:{value:new bt(16777215)},opacity:{value:1},center:{value:new Ke(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new lt},alphaMap:{value:null},alphaMapTransform:{value:new lt},alphaTest:{value:0}}},Ki={basic:{uniforms:zn([Ne.common,Ne.specularmap,Ne.envmap,Ne.aomap,Ne.lightmap,Ne.fog]),vertexShader:_t.meshbasic_vert,fragmentShader:_t.meshbasic_frag},lambert:{uniforms:zn([Ne.common,Ne.specularmap,Ne.envmap,Ne.aomap,Ne.lightmap,Ne.emissivemap,Ne.bumpmap,Ne.normalmap,Ne.displacementmap,Ne.fog,Ne.lights,{emissive:{value:new bt(0)}}]),vertexShader:_t.meshlambert_vert,fragmentShader:_t.meshlambert_frag},phong:{uniforms:zn([Ne.common,Ne.specularmap,Ne.envmap,Ne.aomap,Ne.lightmap,Ne.emissivemap,Ne.bumpmap,Ne.normalmap,Ne.displacementmap,Ne.fog,Ne.lights,{emissive:{value:new bt(0)},specular:{value:new bt(1118481)},shininess:{value:30}}]),vertexShader:_t.meshphong_vert,fragmentShader:_t.meshphong_frag},standard:{uniforms:zn([Ne.common,Ne.envmap,Ne.aomap,Ne.lightmap,Ne.emissivemap,Ne.bumpmap,Ne.normalmap,Ne.displacementmap,Ne.roughnessmap,Ne.metalnessmap,Ne.fog,Ne.lights,{emissive:{value:new bt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:_t.meshphysical_vert,fragmentShader:_t.meshphysical_frag},toon:{uniforms:zn([Ne.common,Ne.aomap,Ne.lightmap,Ne.emissivemap,Ne.bumpmap,Ne.normalmap,Ne.displacementmap,Ne.gradientmap,Ne.fog,Ne.lights,{emissive:{value:new bt(0)}}]),vertexShader:_t.meshtoon_vert,fragmentShader:_t.meshtoon_frag},matcap:{uniforms:zn([Ne.common,Ne.bumpmap,Ne.normalmap,Ne.displacementmap,Ne.fog,{matcap:{value:null}}]),vertexShader:_t.meshmatcap_vert,fragmentShader:_t.meshmatcap_frag},points:{uniforms:zn([Ne.points,Ne.fog]),vertexShader:_t.points_vert,fragmentShader:_t.points_frag},dashed:{uniforms:zn([Ne.common,Ne.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:_t.linedashed_vert,fragmentShader:_t.linedashed_frag},depth:{uniforms:zn([Ne.common,Ne.displacementmap]),vertexShader:_t.depth_vert,fragmentShader:_t.depth_frag},normal:{uniforms:zn([Ne.common,Ne.bumpmap,Ne.normalmap,Ne.displacementmap,{opacity:{value:1}}]),vertexShader:_t.meshnormal_vert,fragmentShader:_t.meshnormal_frag},sprite:{uniforms:zn([Ne.sprite,Ne.fog]),vertexShader:_t.sprite_vert,fragmentShader:_t.sprite_frag},background:{uniforms:{uvTransform:{value:new lt},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:_t.background_vert,fragmentShader:_t.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new lt}},vertexShader:_t.backgroundCube_vert,fragmentShader:_t.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:_t.cube_vert,fragmentShader:_t.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:_t.equirect_vert,fragmentShader:_t.equirect_frag},distanceRGBA:{uniforms:zn([Ne.common,Ne.displacementmap,{referencePosition:{value:new B},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:_t.distanceRGBA_vert,fragmentShader:_t.distanceRGBA_frag},shadow:{uniforms:zn([Ne.lights,Ne.fog,{color:{value:new bt(0)},opacity:{value:1}}]),vertexShader:_t.shadow_vert,fragmentShader:_t.shadow_frag}};Ki.physical={uniforms:zn([Ki.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new lt},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new lt},clearcoatNormalScale:{value:new Ke(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new lt},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new lt},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new lt},sheen:{value:0},sheenColor:{value:new bt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new lt},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new lt},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new lt},transmissionSamplerSize:{value:new Ke},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new lt},attenuationDistance:{value:0},attenuationColor:{value:new bt(0)},specularColor:{value:new bt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new lt},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new lt},anisotropyVector:{value:new Ke},anisotropyMap:{value:null},anisotropyMapTransform:{value:new lt}}]),vertexShader:_t.meshphysical_vert,fragmentShader:_t.meshphysical_frag};const Yl={r:0,b:0,g:0},wr=new Wi,xC=new st;function _C(i,e,t,n,s,r,o){const a=new bt(0);let l=r===!0?0:1,c,u,f=null,d=0,h=null;function x(A){let v=A.isScene===!0?A.background:null;return v&&v.isTexture&&(v=(A.backgroundBlurriness>0?t:e).get(v)),v}function p(A){let v=!1;const S=x(A);S===null?m(a,l):S&&S.isColor&&(m(S,1),v=!0);const y=i.xr.getEnvironmentBlendMode();y==="additive"?n.buffers.color.setClear(0,0,0,1,o):y==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,o),(i.autoClear||v)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function g(A,v){const S=x(v);S&&(S.isCubeTexture||S.mapping===Qc)?(u===void 0&&(u=new hn(new ra(1,1,1),new Yn({name:"BackgroundCubeMaterial",uniforms:Ko(Ki.backgroundCube.uniforms),vertexShader:Ki.backgroundCube.vertexShader,fragmentShader:Ki.backgroundCube.fragmentShader,side:Jn,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),u.geometry.deleteAttribute("normal"),u.geometry.deleteAttribute("uv"),u.onBeforeRender=function(y,M,E){this.matrixWorld.copyPosition(E.matrixWorld)},Object.defineProperty(u.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),s.update(u)),wr.copy(v.backgroundRotation),wr.x*=-1,wr.y*=-1,wr.z*=-1,S.isCubeTexture&&S.isRenderTargetTexture===!1&&(wr.y*=-1,wr.z*=-1),u.material.uniforms.envMap.value=S,u.material.uniforms.flipEnvMap.value=S.isCubeTexture&&S.isRenderTargetTexture===!1?-1:1,u.material.uniforms.backgroundBlurriness.value=v.backgroundBlurriness,u.material.uniforms.backgroundIntensity.value=v.backgroundIntensity,u.material.uniforms.backgroundRotation.value.setFromMatrix4(xC.makeRotationFromEuler(wr)),u.material.toneMapped=Et.getTransfer(S.colorSpace)!==Ot,(f!==S||d!==S.version||h!==i.toneMapping)&&(u.material.needsUpdate=!0,f=S,d=S.version,h=i.toneMapping),u.layers.enableAll(),A.unshift(u,u.geometry,u.material,0,0,null)):S&&S.isTexture&&(c===void 0&&(c=new hn(new jo(2,2),new Yn({name:"BackgroundMaterial",uniforms:Ko(Ki.background.uniforms),vertexShader:Ki.background.vertexShader,fragmentShader:Ki.background.fragmentShader,side:os,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),s.update(c)),c.material.uniforms.t2D.value=S,c.material.uniforms.backgroundIntensity.value=v.backgroundIntensity,c.material.toneMapped=Et.getTransfer(S.colorSpace)!==Ot,S.matrixAutoUpdate===!0&&S.updateMatrix(),c.material.uniforms.uvTransform.value.copy(S.matrix),(f!==S||d!==S.version||h!==i.toneMapping)&&(c.material.needsUpdate=!0,f=S,d=S.version,h=i.toneMapping),c.layers.enableAll(),A.unshift(c,c.geometry,c.material,0,0,null))}function m(A,v){A.getRGB(Yl,Yg(i)),n.buffers.color.setClear(Yl.r,Yl.g,Yl.b,v,o)}function _(){u!==void 0&&(u.geometry.dispose(),u.material.dispose(),u=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return a},setClearColor:function(A,v=1){a.set(A),l=v,m(a,l)},getClearAlpha:function(){return l},setClearAlpha:function(A){l=A,m(a,l)},render:p,addToRenderList:g,dispose:_}}function vC(i,e){const t=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},s=d(null);let r=s,o=!1;function a(C,D,F,O,z){let V=!1;const H=f(O,F,D);r!==H&&(r=H,c(r.object)),V=h(C,O,F,z),V&&x(C,O,F,z),z!==null&&e.update(z,i.ELEMENT_ARRAY_BUFFER),(V||o)&&(o=!1,v(C,D,F,O),z!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,e.get(z).buffer))}function l(){return i.createVertexArray()}function c(C){return i.bindVertexArray(C)}function u(C){return i.deleteVertexArray(C)}function f(C,D,F){const O=F.wireframe===!0;let z=n[C.id];z===void 0&&(z={},n[C.id]=z);let V=z[D.id];V===void 0&&(V={},z[D.id]=V);let H=V[O];return H===void 0&&(H=d(l()),V[O]=H),H}function d(C){const D=[],F=[],O=[];for(let z=0;z<t;z++)D[z]=0,F[z]=0,O[z]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:D,enabledAttributes:F,attributeDivisors:O,object:C,attributes:{},index:null}}function h(C,D,F,O){const z=r.attributes,V=D.attributes;let H=0;const q=F.getAttributes();for(const G in q)if(q[G].location>=0){const fe=z[G];let Y=V[G];if(Y===void 0&&(G==="instanceMatrix"&&C.instanceMatrix&&(Y=C.instanceMatrix),G==="instanceColor"&&C.instanceColor&&(Y=C.instanceColor)),fe===void 0||fe.attribute!==Y||Y&&fe.data!==Y.data)return!0;H++}return r.attributesNum!==H||r.index!==O}function x(C,D,F,O){const z={},V=D.attributes;let H=0;const q=F.getAttributes();for(const G in q)if(q[G].location>=0){let fe=V[G];fe===void 0&&(G==="instanceMatrix"&&C.instanceMatrix&&(fe=C.instanceMatrix),G==="instanceColor"&&C.instanceColor&&(fe=C.instanceColor));const Y={};Y.attribute=fe,fe&&fe.data&&(Y.data=fe.data),z[G]=Y,H++}r.attributes=z,r.attributesNum=H,r.index=O}function p(){const C=r.newAttributes;for(let D=0,F=C.length;D<F;D++)C[D]=0}function g(C){m(C,0)}function m(C,D){const F=r.newAttributes,O=r.enabledAttributes,z=r.attributeDivisors;F[C]=1,O[C]===0&&(i.enableVertexAttribArray(C),O[C]=1),z[C]!==D&&(i.vertexAttribDivisor(C,D),z[C]=D)}function _(){const C=r.newAttributes,D=r.enabledAttributes;for(let F=0,O=D.length;F<O;F++)D[F]!==C[F]&&(i.disableVertexAttribArray(F),D[F]=0)}function A(C,D,F,O,z,V,H){H===!0?i.vertexAttribIPointer(C,D,F,z,V):i.vertexAttribPointer(C,D,F,O,z,V)}function v(C,D,F,O){p();const z=O.attributes,V=F.getAttributes(),H=D.defaultAttributeValues;for(const q in V){const G=V[q];if(G.location>=0){let $=z[q];if($===void 0&&(q==="instanceMatrix"&&C.instanceMatrix&&($=C.instanceMatrix),q==="instanceColor"&&C.instanceColor&&($=C.instanceColor)),$!==void 0){const fe=$.normalized,Y=$.itemSize,we=e.get($);if(we===void 0)continue;const ze=we.buffer,ke=we.type,We=we.bytesPerElement,ne=ke===i.INT||ke===i.UNSIGNED_INT||$.gpuType===$d;if($.isInterleavedBufferAttribute){const ue=$.data,Se=ue.stride,he=$.offset;if(ue.isInstancedInterleavedBuffer){for(let Ee=0;Ee<G.locationSize;Ee++)m(G.location+Ee,ue.meshPerAttribute);C.isInstancedMesh!==!0&&O._maxInstanceCount===void 0&&(O._maxInstanceCount=ue.meshPerAttribute*ue.count)}else for(let Ee=0;Ee<G.locationSize;Ee++)g(G.location+Ee);i.bindBuffer(i.ARRAY_BUFFER,ze);for(let Ee=0;Ee<G.locationSize;Ee++)A(G.location+Ee,Y/G.locationSize,ke,fe,Se*We,(he+Y/G.locationSize*Ee)*We,ne)}else{if($.isInstancedBufferAttribute){for(let ue=0;ue<G.locationSize;ue++)m(G.location+ue,$.meshPerAttribute);C.isInstancedMesh!==!0&&O._maxInstanceCount===void 0&&(O._maxInstanceCount=$.meshPerAttribute*$.count)}else for(let ue=0;ue<G.locationSize;ue++)g(G.location+ue);i.bindBuffer(i.ARRAY_BUFFER,ze);for(let ue=0;ue<G.locationSize;ue++)A(G.location+ue,Y/G.locationSize,ke,fe,Y*We,Y/G.locationSize*ue*We,ne)}}else if(H!==void 0){const fe=H[q];if(fe!==void 0)switch(fe.length){case 2:i.vertexAttrib2fv(G.location,fe);break;case 3:i.vertexAttrib3fv(G.location,fe);break;case 4:i.vertexAttrib4fv(G.location,fe);break;default:i.vertexAttrib1fv(G.location,fe)}}}}_()}function S(){E();for(const C in n){const D=n[C];for(const F in D){const O=D[F];for(const z in O)u(O[z].object),delete O[z];delete D[F]}delete n[C]}}function y(C){if(n[C.id]===void 0)return;const D=n[C.id];for(const F in D){const O=D[F];for(const z in O)u(O[z].object),delete O[z];delete D[F]}delete n[C.id]}function M(C){for(const D in n){const F=n[D];if(F[C.id]===void 0)continue;const O=F[C.id];for(const z in O)u(O[z].object),delete O[z];delete F[C.id]}}function E(){b(),o=!0,r!==s&&(r=s,c(r.object))}function b(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:a,reset:E,resetDefaultState:b,dispose:S,releaseStatesOfGeometry:y,releaseStatesOfProgram:M,initAttributes:p,enableAttribute:g,disableUnusedAttributes:_}}function AC(i,e,t){let n;function s(c){n=c}function r(c,u){i.drawArrays(n,c,u),t.update(u,n,1)}function o(c,u,f){f!==0&&(i.drawArraysInstanced(n,c,u,f),t.update(u,n,f))}function a(c,u,f){if(f===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,c,0,u,0,f);let h=0;for(let x=0;x<f;x++)h+=u[x];t.update(h,n,1)}function l(c,u,f,d){if(f===0)return;const h=e.get("WEBGL_multi_draw");if(h===null)for(let x=0;x<c.length;x++)o(c[x],u[x],d[x]);else{h.multiDrawArraysInstancedWEBGL(n,c,0,u,0,d,0,f);let x=0;for(let p=0;p<f;p++)x+=u[p]*d[p];t.update(x,n,1)}}this.setMode=s,this.render=r,this.renderInstances=o,this.renderMultiDraw=a,this.renderMultiDrawInstances=l}function SC(i,e,t,n){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){const M=e.get("EXT_texture_filter_anisotropic");s=i.getParameter(M.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function o(M){return!(M!==Xn&&n.convert(M)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function a(M){const E=M===jr&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(M!==as&&n.convert(M)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE)&&M!==Hi&&!E)}function l(M){if(M==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";M="mediump"}return M==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let c=t.precision!==void 0?t.precision:"highp";const u=l(c);u!==c&&(ft("WebGLRenderer:",c,"not supported, using",u,"instead."),c=u);const f=t.logarithmicDepthBuffer===!0,d=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),h=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),x=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),p=i.getParameter(i.MAX_TEXTURE_SIZE),g=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),m=i.getParameter(i.MAX_VERTEX_ATTRIBS),_=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),A=i.getParameter(i.MAX_VARYING_VECTORS),v=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),S=x>0,y=i.getParameter(i.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:l,textureFormatReadable:o,textureTypeReadable:a,precision:c,logarithmicDepthBuffer:f,reversedDepthBuffer:d,maxTextures:h,maxVertexTextures:x,maxTextureSize:p,maxCubemapSize:g,maxAttributes:m,maxVertexUniforms:_,maxVaryings:A,maxFragmentUniforms:v,vertexTextures:S,maxSamples:y}}function yC(i){const e=this;let t=null,n=0,s=!1,r=!1;const o=new Ks,a=new lt,l={value:null,needsUpdate:!1};this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(f,d){const h=f.length!==0||d||n!==0||s;return s=d,n=f.length,h},this.beginShadows=function(){r=!0,u(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(f,d){t=u(f,d,0)},this.setState=function(f,d,h){const x=f.clippingPlanes,p=f.clipIntersection,g=f.clipShadows,m=i.get(f);if(!s||x===null||x.length===0||r&&!g)r?u(null):c();else{const _=r?0:n,A=_*4;let v=m.clippingState||null;l.value=v,v=u(x,d,A,h);for(let S=0;S!==A;++S)v[S]=t[S];m.clippingState=v,this.numIntersection=p?this.numPlanes:0,this.numPlanes+=_}};function c(){l.value!==t&&(l.value=t,l.needsUpdate=n>0),e.numPlanes=n,e.numIntersection=0}function u(f,d,h,x){const p=f!==null?f.length:0;let g=null;if(p!==0){if(g=l.value,x!==!0||g===null){const m=h+p*4,_=d.matrixWorldInverse;a.getNormalMatrix(_),(g===null||g.length<m)&&(g=new Float32Array(m));for(let A=0,v=h;A!==p;++A,v+=4)o.copy(f[A]).applyMatrix4(_,a),o.normal.toArray(g,v),g[v+3]=o.constant}l.value=g,l.needsUpdate=!0}return e.numPlanes=p,e.numIntersection=0,g}}function bC(i){let e=new WeakMap;function t(o,a){return a===Ff?o.mapping=Xo:a===Lf&&(o.mapping=qo),o}function n(o){if(o&&o.isTexture){const a=o.mapping;if(a===Ff||a===Lf)if(e.has(o)){const l=e.get(o).texture;return t(l,o.mapping)}else{const l=o.image;if(l&&l.height>0){const c=new yy(l.height);return c.fromEquirectangularTexture(i,o),e.set(o,c),o.addEventListener("dispose",s),t(c.texture,o.mapping)}else return null}}return o}function s(o){const a=o.target;a.removeEventListener("dispose",s);const l=e.get(a);l!==void 0&&(e.delete(a),l.dispose())}function r(){e=new WeakMap}return{get:n,dispose:r}}const $s=4,sm=[.125,.215,.35,.446,.526,.582],zr=20,MC=256,_a=new lh,rm=new bt;let Gu=null,Wu=0,Xu=0,qu=!1;const CC=new B;class om{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,n=.1,s=100,r={}){const{size:o=256,position:a=CC}=r;Gu=this._renderer.getRenderTarget(),Wu=this._renderer.getActiveCubeFace(),Xu=this._renderer.getActiveMipmapLevel(),qu=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(o);const l=this._allocateTargets();return l.depthBuffer=!0,this._sceneToCubeUV(e,n,s,l,a),t>0&&this._blur(l,0,0,t),this._applyPMREM(l),this._cleanup(l),l}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=cm(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=lm(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(Gu,Wu,Xu),this._renderer.xr.enabled=qu,e.scissorTest=!1,vo(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===Xo||e.mapping===qo?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),Gu=this._renderer.getRenderTarget(),Wu=this._renderer.getActiveCubeFace(),Xu=this._renderer.getActiveMipmapLevel(),qu=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const n=t||this._allocateTargets();return this._textureToCubeUV(e,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,n={magFilter:Ri,minFilter:Ri,generateMipmaps:!1,type:jr,format:Xn,colorSpace:Qo,depthBuffer:!1},s=am(e,t,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=am(e,t,n);const{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=TC(r)),this._blurMaterial=wC(r,e,t),this._ggxMaterial=EC(r,e,t)}return s}_compileMaterial(e){const t=new hn(new Kn,e);this._renderer.compile(t,_a)}_sceneToCubeUV(e,t,n,s,r){const l=new Ti(90,1,t,n),c=[1,-1,1,1,1,1],u=[1,1,1,-1,-1,-1],f=this._renderer,d=f.autoClear,h=f.toneMapping;f.getClearColor(rm),f.toneMapping=sr,f.autoClear=!1,f.state.buffers.depth.getReversed()&&(f.setRenderTarget(s),f.clearDepth(),f.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new hn(new ra,new Kr({name:"PMREM.Background",side:Jn,depthWrite:!1,depthTest:!1})));const p=this._backgroundBox,g=p.material;let m=!1;const _=e.background;_?_.isColor&&(g.color.copy(_),e.background=null,m=!0):(g.color.copy(rm),m=!0);for(let A=0;A<6;A++){const v=A%3;v===0?(l.up.set(0,c[A],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x+u[A],r.y,r.z)):v===1?(l.up.set(0,0,c[A]),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y+u[A],r.z)):(l.up.set(0,c[A],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y,r.z+u[A]));const S=this._cubeSize;vo(s,v*S,A>2?S:0,S,S),f.setRenderTarget(s),m&&f.render(p,l),f.render(e,l)}f.toneMapping=h,f.autoClear=d,e.background=_}_textureToCubeUV(e,t){const n=this._renderer,s=e.mapping===Xo||e.mapping===qo;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=cm()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=lm());const r=s?this._cubemapMaterial:this._equirectMaterial,o=this._lodMeshes[0];o.material=r;const a=r.uniforms;a.envMap.value=e;const l=this._cubeSize;vo(t,0,0,3*l,2*l),n.setRenderTarget(t),n.render(o,_a)}_applyPMREM(e){const t=this._renderer,n=t.autoClear;t.autoClear=!1;const s=this._lodMeshes.length;for(let r=1;r<s;r++)this._applyGGXFilter(e,r-1,r);t.autoClear=n}_applyGGXFilter(e,t,n){const s=this._renderer,r=this._pingPongRenderTarget,o=this._ggxMaterial,a=this._lodMeshes[n];a.material=o;const l=o.uniforms,c=n/(this._lodMeshes.length-1),u=t/(this._lodMeshes.length-1),f=Math.sqrt(c*c-u*u),d=.05+c*.95,h=f*d,{_lodMax:x}=this,p=this._sizeLods[n],g=3*p*(n>x-$s?n-x+$s:0),m=4*(this._cubeSize-p);l.envMap.value=e.texture,l.roughness.value=h,l.mipInt.value=x-t,vo(r,g,m,3*p,2*p),s.setRenderTarget(r),s.render(a,_a),l.envMap.value=r.texture,l.roughness.value=0,l.mipInt.value=x-n,vo(e,g,m,3*p,2*p),s.setRenderTarget(e),s.render(a,_a)}_blur(e,t,n,s,r){const o=this._pingPongRenderTarget;this._halfBlur(e,o,t,n,s,"latitudinal",r),this._halfBlur(o,e,n,n,s,"longitudinal",r)}_halfBlur(e,t,n,s,r,o,a){const l=this._renderer,c=this._blurMaterial;o!=="latitudinal"&&o!=="longitudinal"&&fn("blur direction must be either latitudinal or longitudinal!");const u=3,f=this._lodMeshes[s];f.material=c;const d=c.uniforms,h=this._sizeLods[n]-1,x=isFinite(r)?Math.PI/(2*h):2*Math.PI/(2*zr-1),p=r/x,g=isFinite(r)?1+Math.floor(u*p):zr;g>zr&&ft(`sigmaRadians, ${r}, is too large and will clip, as it requested ${g} samples when the maximum is set to ${zr}`);const m=[];let _=0;for(let M=0;M<zr;++M){const E=M/p,b=Math.exp(-E*E/2);m.push(b),M===0?_+=b:M<g&&(_+=2*b)}for(let M=0;M<m.length;M++)m[M]=m[M]/_;d.envMap.value=e.texture,d.samples.value=g,d.weights.value=m,d.latitudinal.value=o==="latitudinal",a&&(d.poleAxis.value=a);const{_lodMax:A}=this;d.dTheta.value=x,d.mipInt.value=A-n;const v=this._sizeLods[s],S=3*v*(s>A-$s?s-A+$s:0),y=4*(this._cubeSize-v);vo(t,S,y,3*v,2*v),l.setRenderTarget(t),l.render(f,_a)}}function TC(i){const e=[],t=[],n=[];let s=i;const r=i-$s+1+sm.length;for(let o=0;o<r;o++){const a=Math.pow(2,s);e.push(a);let l=1/a;o>i-$s?l=sm[o-i+$s-1]:o===0&&(l=0),t.push(l);const c=1/(a-2),u=-c,f=1+c,d=[u,u,f,u,f,f,u,u,f,f,u,f],h=6,x=6,p=3,g=2,m=1,_=new Float32Array(p*x*h),A=new Float32Array(g*x*h),v=new Float32Array(m*x*h);for(let y=0;y<h;y++){const M=y%3*2/3-1,E=y>2?0:-1,b=[M,E,0,M+2/3,E,0,M+2/3,E+1,0,M,E,0,M+2/3,E+1,0,M,E+1,0];_.set(b,p*x*y),A.set(d,g*x*y);const C=[y,y,y,y,y,y];v.set(C,m*x*y)}const S=new Kn;S.setAttribute("position",new Li(_,p)),S.setAttribute("uv",new Li(A,g)),S.setAttribute("faceIndex",new Li(v,m)),n.push(new hn(S,null)),s>$s&&s--}return{lodMeshes:n,sizeLods:e,sigmas:t}}function am(i,e,t){const n=new cr(i,e,t);return n.texture.mapping=Qc,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function vo(i,e,t,n,s){i.viewport.set(e,t,n,s),i.scissor.set(e,t,n,s)}function EC(i,e,t){return new Yn({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:MC,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:$c(),fragmentShader:`

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
		`,blending:Ds,depthTest:!1,depthWrite:!1})}function wC(i,e,t){const n=new Float32Array(zr),s=new B(0,1,0);return new Yn({name:"SphericalGaussianBlur",defines:{n:zr,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:$c(),fragmentShader:`

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
		`,blending:Ds,depthTest:!1,depthWrite:!1})}function lm(){return new Yn({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:$c(),fragmentShader:`

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
		`,blending:Ds,depthTest:!1,depthWrite:!1})}function cm(){return new Yn({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:$c(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:Ds,depthTest:!1,depthWrite:!1})}function $c(){return`

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
	`}function RC(i){let e=new WeakMap,t=null;function n(a){if(a&&a.isTexture){const l=a.mapping,c=l===Ff||l===Lf,u=l===Xo||l===qo;if(c||u){let f=e.get(a);const d=f!==void 0?f.texture.pmremVersion:0;if(a.isRenderTargetTexture&&a.pmremVersion!==d)return t===null&&(t=new om(i)),f=c?t.fromEquirectangular(a,f):t.fromCubemap(a,f),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),f.texture;if(f!==void 0)return f.texture;{const h=a.image;return c&&h&&h.height>0||u&&h&&s(h)?(t===null&&(t=new om(i)),f=c?t.fromEquirectangular(a):t.fromCubemap(a),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),a.addEventListener("dispose",r),f.texture):null}}}return a}function s(a){let l=0;const c=6;for(let u=0;u<c;u++)a[u]!==void 0&&l++;return l===c}function r(a){const l=a.target;l.removeEventListener("dispose",r);const c=e.get(l);c!==void 0&&(e.delete(l),c.dispose())}function o(){e=new WeakMap,t!==null&&(t.dispose(),t=null)}return{get:n,dispose:o}}function IC(i){const e={};function t(n){if(e[n]!==void 0)return e[n];const s=i.getExtension(n);return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(n){const s=t(n);return s===null&&Ja("WebGLRenderer: "+n+" extension not supported."),s}}}function DC(i,e,t,n){const s={},r=new WeakMap;function o(f){const d=f.target;d.index!==null&&e.remove(d.index);for(const x in d.attributes)e.remove(d.attributes[x]);d.removeEventListener("dispose",o),delete s[d.id];const h=r.get(d);h&&(e.remove(h),r.delete(d)),n.releaseStatesOfGeometry(d),d.isInstancedBufferGeometry===!0&&delete d._maxInstanceCount,t.memory.geometries--}function a(f,d){return s[d.id]===!0||(d.addEventListener("dispose",o),s[d.id]=!0,t.memory.geometries++),d}function l(f){const d=f.attributes;for(const h in d)e.update(d[h],i.ARRAY_BUFFER)}function c(f){const d=[],h=f.index,x=f.attributes.position;let p=0;if(h!==null){const _=h.array;p=h.version;for(let A=0,v=_.length;A<v;A+=3){const S=_[A+0],y=_[A+1],M=_[A+2];d.push(S,y,y,M,M,S)}}else if(x!==void 0){const _=x.array;p=x.version;for(let A=0,v=_.length/3-1;A<v;A+=3){const S=A+0,y=A+1,M=A+2;d.push(S,y,y,M,M,S)}}else return;const g=new(Hg(d)?qg:Xg)(d,1);g.version=p;const m=r.get(f);m&&e.remove(m),r.set(f,g)}function u(f){const d=r.get(f);if(d){const h=f.index;h!==null&&d.version<h.version&&c(f)}else c(f);return r.get(f)}return{get:a,update:l,getWireframeAttribute:u}}function PC(i,e,t){let n;function s(d){n=d}let r,o;function a(d){r=d.type,o=d.bytesPerElement}function l(d,h){i.drawElements(n,h,r,d*o),t.update(h,n,1)}function c(d,h,x){x!==0&&(i.drawElementsInstanced(n,h,r,d*o,x),t.update(h,n,x))}function u(d,h,x){if(x===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,h,0,r,d,0,x);let g=0;for(let m=0;m<x;m++)g+=h[m];t.update(g,n,1)}function f(d,h,x,p){if(x===0)return;const g=e.get("WEBGL_multi_draw");if(g===null)for(let m=0;m<d.length;m++)c(d[m]/o,h[m],p[m]);else{g.multiDrawElementsInstancedWEBGL(n,h,0,r,d,0,p,0,x);let m=0;for(let _=0;_<x;_++)m+=h[_]*p[_];t.update(m,n,1)}}this.setMode=s,this.setIndex=a,this.render=l,this.renderInstances=c,this.renderMultiDraw=u,this.renderMultiDrawInstances=f}function FC(i){const e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function n(r,o,a){switch(t.calls++,o){case i.TRIANGLES:t.triangles+=a*(r/3);break;case i.LINES:t.lines+=a*(r/2);break;case i.LINE_STRIP:t.lines+=a*(r-1);break;case i.LINE_LOOP:t.lines+=a*r;break;case i.POINTS:t.points+=a*r;break;default:fn("WebGLInfo: Unknown draw mode:",o);break}}function s(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:s,update:n}}function LC(i,e,t){const n=new WeakMap,s=new Jt;function r(o,a,l){const c=o.morphTargetInfluences,u=a.morphAttributes.position||a.morphAttributes.normal||a.morphAttributes.color,f=u!==void 0?u.length:0;let d=n.get(a);if(d===void 0||d.count!==f){let C=function(){E.dispose(),n.delete(a),a.removeEventListener("dispose",C)};var h=C;d!==void 0&&d.texture.dispose();const x=a.morphAttributes.position!==void 0,p=a.morphAttributes.normal!==void 0,g=a.morphAttributes.color!==void 0,m=a.morphAttributes.position||[],_=a.morphAttributes.normal||[],A=a.morphAttributes.color||[];let v=0;x===!0&&(v=1),p===!0&&(v=2),g===!0&&(v=3);let S=a.attributes.position.count*v,y=1;S>e.maxTextureSize&&(y=Math.ceil(S/e.maxTextureSize),S=e.maxTextureSize);const M=new Float32Array(S*y*4*f),E=new Vg(M,S,y,f);E.type=Hi,E.needsUpdate=!0;const b=v*4;for(let D=0;D<f;D++){const F=m[D],O=_[D],z=A[D],V=S*y*4*D;for(let H=0;H<F.count;H++){const q=H*b;x===!0&&(s.fromBufferAttribute(F,H),M[V+q+0]=s.x,M[V+q+1]=s.y,M[V+q+2]=s.z,M[V+q+3]=0),p===!0&&(s.fromBufferAttribute(O,H),M[V+q+4]=s.x,M[V+q+5]=s.y,M[V+q+6]=s.z,M[V+q+7]=0),g===!0&&(s.fromBufferAttribute(z,H),M[V+q+8]=s.x,M[V+q+9]=s.y,M[V+q+10]=s.z,M[V+q+11]=z.itemSize===4?s.w:1)}}d={count:f,texture:E,size:new Ke(S,y)},n.set(a,d),a.addEventListener("dispose",C)}if(o.isInstancedMesh===!0&&o.morphTexture!==null)l.getUniforms().setValue(i,"morphTexture",o.morphTexture,t);else{let x=0;for(let g=0;g<c.length;g++)x+=c[g];const p=a.morphTargetsRelative?1:1-x;l.getUniforms().setValue(i,"morphTargetBaseInfluence",p),l.getUniforms().setValue(i,"morphTargetInfluences",c)}l.getUniforms().setValue(i,"morphTargetsTexture",d.texture,t),l.getUniforms().setValue(i,"morphTargetsTextureSize",d.size)}return{update:r}}function BC(i,e,t,n){let s=new WeakMap;function r(l){const c=n.render.frame,u=l.geometry,f=e.get(l,u);if(s.get(f)!==c&&(e.update(f),s.set(f,c)),l.isInstancedMesh&&(l.hasEventListener("dispose",a)===!1&&l.addEventListener("dispose",a),s.get(l)!==c&&(t.update(l.instanceMatrix,i.ARRAY_BUFFER),l.instanceColor!==null&&t.update(l.instanceColor,i.ARRAY_BUFFER),s.set(l,c))),l.isSkinnedMesh){const d=l.skeleton;s.get(d)!==c&&(d.update(),s.set(d,c))}return f}function o(){s=new WeakMap}function a(l){const c=l.target;c.removeEventListener("dispose",a),t.remove(c.instanceMatrix),c.instanceColor!==null&&t.remove(c.instanceColor)}return{update:r,dispose:o}}const Jg=new qn,um=new rh(1,1),ex=new Vg,tx=new ny,nx=new Kg,fm=[],dm=[],hm=new Float32Array(16),pm=new Float32Array(9),mm=new Float32Array(4);function oa(i,e,t){const n=i[0];if(n<=0||n>0)return i;const s=e*t;let r=fm[s];if(r===void 0&&(r=new Float32Array(s),fm[s]=r),e!==0){n.toArray(r,0);for(let o=1,a=0;o!==e;++o)a+=t,i[o].toArray(r,a)}return r}function _n(i,e){if(i.length!==e.length)return!1;for(let t=0,n=i.length;t<n;t++)if(i[t]!==e[t])return!1;return!0}function vn(i,e){for(let t=0,n=e.length;t<n;t++)i[t]=e[t]}function Zc(i,e){let t=dm[e];t===void 0&&(t=new Int32Array(e),dm[e]=t);for(let n=0;n!==e;++n)t[n]=i.allocateTextureUnit();return t}function UC(i,e){const t=this.cache;t[0]!==e&&(i.uniform1f(this.addr,e),t[0]=e)}function OC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(_n(t,e))return;i.uniform2fv(this.addr,e),vn(t,e)}}function NC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(i.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(_n(t,e))return;i.uniform3fv(this.addr,e),vn(t,e)}}function zC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(_n(t,e))return;i.uniform4fv(this.addr,e),vn(t,e)}}function kC(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(_n(t,e))return;i.uniformMatrix2fv(this.addr,!1,e),vn(t,e)}else{if(_n(t,n))return;mm.set(n),i.uniformMatrix2fv(this.addr,!1,mm),vn(t,n)}}function HC(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(_n(t,e))return;i.uniformMatrix3fv(this.addr,!1,e),vn(t,e)}else{if(_n(t,n))return;pm.set(n),i.uniformMatrix3fv(this.addr,!1,pm),vn(t,n)}}function VC(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(_n(t,e))return;i.uniformMatrix4fv(this.addr,!1,e),vn(t,e)}else{if(_n(t,n))return;hm.set(n),i.uniformMatrix4fv(this.addr,!1,hm),vn(t,n)}}function GC(i,e){const t=this.cache;t[0]!==e&&(i.uniform1i(this.addr,e),t[0]=e)}function WC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(_n(t,e))return;i.uniform2iv(this.addr,e),vn(t,e)}}function XC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(_n(t,e))return;i.uniform3iv(this.addr,e),vn(t,e)}}function qC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(_n(t,e))return;i.uniform4iv(this.addr,e),vn(t,e)}}function YC(i,e){const t=this.cache;t[0]!==e&&(i.uniform1ui(this.addr,e),t[0]=e)}function QC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(_n(t,e))return;i.uniform2uiv(this.addr,e),vn(t,e)}}function KC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(_n(t,e))return;i.uniform3uiv(this.addr,e),vn(t,e)}}function jC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(_n(t,e))return;i.uniform4uiv(this.addr,e),vn(t,e)}}function $C(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s);let r;this.type===i.SAMPLER_2D_SHADOW?(um.compareFunction=kg,r=um):r=Jg,t.setTexture2D(e||r,s)}function ZC(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture3D(e||tx,s)}function JC(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTextureCube(e||nx,s)}function eT(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture2DArray(e||ex,s)}function tT(i){switch(i){case 5126:return UC;case 35664:return OC;case 35665:return NC;case 35666:return zC;case 35674:return kC;case 35675:return HC;case 35676:return VC;case 5124:case 35670:return GC;case 35667:case 35671:return WC;case 35668:case 35672:return XC;case 35669:case 35673:return qC;case 5125:return YC;case 36294:return QC;case 36295:return KC;case 36296:return jC;case 35678:case 36198:case 36298:case 36306:case 35682:return $C;case 35679:case 36299:case 36307:return ZC;case 35680:case 36300:case 36308:case 36293:return JC;case 36289:case 36303:case 36311:case 36292:return eT}}function nT(i,e){i.uniform1fv(this.addr,e)}function iT(i,e){const t=oa(e,this.size,2);i.uniform2fv(this.addr,t)}function sT(i,e){const t=oa(e,this.size,3);i.uniform3fv(this.addr,t)}function rT(i,e){const t=oa(e,this.size,4);i.uniform4fv(this.addr,t)}function oT(i,e){const t=oa(e,this.size,4);i.uniformMatrix2fv(this.addr,!1,t)}function aT(i,e){const t=oa(e,this.size,9);i.uniformMatrix3fv(this.addr,!1,t)}function lT(i,e){const t=oa(e,this.size,16);i.uniformMatrix4fv(this.addr,!1,t)}function cT(i,e){i.uniform1iv(this.addr,e)}function uT(i,e){i.uniform2iv(this.addr,e)}function fT(i,e){i.uniform3iv(this.addr,e)}function dT(i,e){i.uniform4iv(this.addr,e)}function hT(i,e){i.uniform1uiv(this.addr,e)}function pT(i,e){i.uniform2uiv(this.addr,e)}function mT(i,e){i.uniform3uiv(this.addr,e)}function gT(i,e){i.uniform4uiv(this.addr,e)}function xT(i,e,t){const n=this.cache,s=e.length,r=Zc(t,s);_n(n,r)||(i.uniform1iv(this.addr,r),vn(n,r));for(let o=0;o!==s;++o)t.setTexture2D(e[o]||Jg,r[o])}function _T(i,e,t){const n=this.cache,s=e.length,r=Zc(t,s);_n(n,r)||(i.uniform1iv(this.addr,r),vn(n,r));for(let o=0;o!==s;++o)t.setTexture3D(e[o]||tx,r[o])}function vT(i,e,t){const n=this.cache,s=e.length,r=Zc(t,s);_n(n,r)||(i.uniform1iv(this.addr,r),vn(n,r));for(let o=0;o!==s;++o)t.setTextureCube(e[o]||nx,r[o])}function AT(i,e,t){const n=this.cache,s=e.length,r=Zc(t,s);_n(n,r)||(i.uniform1iv(this.addr,r),vn(n,r));for(let o=0;o!==s;++o)t.setTexture2DArray(e[o]||ex,r[o])}function ST(i){switch(i){case 5126:return nT;case 35664:return iT;case 35665:return sT;case 35666:return rT;case 35674:return oT;case 35675:return aT;case 35676:return lT;case 5124:case 35670:return cT;case 35667:case 35671:return uT;case 35668:case 35672:return fT;case 35669:case 35673:return dT;case 5125:return hT;case 36294:return pT;case 36295:return mT;case 36296:return gT;case 35678:case 36198:case 36298:case 36306:case 35682:return xT;case 35679:case 36299:case 36307:return _T;case 35680:case 36300:case 36308:case 36293:return vT;case 36289:case 36303:case 36311:case 36292:return AT}}class yT{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.setValue=tT(t.type)}}class bT{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=ST(t.type)}}class MT{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,n){const s=this.seq;for(let r=0,o=s.length;r!==o;++r){const a=s[r];a.setValue(e,t[a.id],n)}}}const Yu=/(\w+)(\])?(\[|\.)?/g;function gm(i,e){i.seq.push(e),i.map[e.id]=e}function CT(i,e,t){const n=i.name,s=n.length;for(Yu.lastIndex=0;;){const r=Yu.exec(n),o=Yu.lastIndex;let a=r[1];const l=r[2]==="]",c=r[3];if(l&&(a=a|0),c===void 0||c==="["&&o+2===s){gm(t,c===void 0?new yT(a,i,e):new bT(a,i,e));break}else{let f=t.map[a];f===void 0&&(f=new MT(a),gm(t,f)),t=f}}}class dc{constructor(e,t){this.seq=[],this.map={};const n=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let s=0;s<n;++s){const r=e.getActiveUniform(t,s),o=e.getUniformLocation(t,r.name);CT(r,o,this)}}setValue(e,t,n,s){const r=this.map[t];r!==void 0&&r.setValue(e,n,s)}setOptional(e,t,n){const s=t[n];s!==void 0&&this.setValue(e,n,s)}static upload(e,t,n,s){for(let r=0,o=t.length;r!==o;++r){const a=t[r],l=n[a.id];l.needsUpdate!==!1&&a.setValue(e,l.value,s)}}static seqWithValue(e,t){const n=[];for(let s=0,r=e.length;s!==r;++s){const o=e[s];o.id in t&&n.push(o)}return n}}function xm(i,e,t){const n=i.createShader(e);return i.shaderSource(n,t),i.compileShader(n),n}const TT=37297;let ET=0;function wT(i,e){const t=i.split(`
`),n=[],s=Math.max(e-6,0),r=Math.min(e+6,t.length);for(let o=s;o<r;o++){const a=o+1;n.push(`${a===e?">":" "} ${a}: ${t[o]}`)}return n.join(`
`)}const _m=new lt;function RT(i){Et._getMatrix(_m,Et.workingColorSpace,i);const e=`mat3( ${_m.elements.map(t=>t.toFixed(4))} )`;switch(Et.getTransfer(i)){case bc:return[e,"LinearTransferOETF"];case Ot:return[e,"sRGBTransferOETF"];default:return ft("WebGLProgram: Unsupported color space: ",i),[e,"LinearTransferOETF"]}}function vm(i,e,t){const n=i.getShaderParameter(e,i.COMPILE_STATUS),r=(i.getShaderInfoLog(e)||"").trim();if(n&&r==="")return"";const o=/ERROR: 0:(\d+)/.exec(r);if(o){const a=parseInt(o[1]);return t.toUpperCase()+`

`+r+`

`+wT(i.getShaderSource(e),a)}else return r}function IT(i,e){const t=RT(e);return[`vec4 ${i}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}function DT(i,e){let t;switch(e){case hS:t="Linear";break;case pS:t="Reinhard";break;case mS:t="Cineon";break;case gS:t="ACESFilmic";break;case _S:t="AgX";break;case vS:t="Neutral";break;case xS:t="Custom";break;default:ft("WebGLProgram: Unsupported toneMapping:",e),t="Linear"}return"vec3 "+i+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}const Ql=new B;function PT(){Et.getLuminanceCoefficients(Ql);const i=Ql.x.toFixed(4),e=Ql.y.toFixed(4),t=Ql.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function FT(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(ya).join(`
`)}function LT(i){const e=[];for(const t in i){const n=i[t];n!==!1&&e.push("#define "+t+" "+n)}return e.join(`
`)}function BT(i,e){const t={},n=i.getProgramParameter(e,i.ACTIVE_ATTRIBUTES);for(let s=0;s<n;s++){const r=i.getActiveAttrib(e,s),o=r.name;let a=1;r.type===i.FLOAT_MAT2&&(a=2),r.type===i.FLOAT_MAT3&&(a=3),r.type===i.FLOAT_MAT4&&(a=4),t[o]={type:r.type,location:i.getAttribLocation(e,o),locationSize:a}}return t}function ya(i){return i!==""}function Am(i,e){const t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return i.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function Sm(i,e){return i.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const UT=/^[ \t]*#include +<([\w\d./]+)>/gm;function dd(i){return i.replace(UT,NT)}const OT=new Map;function NT(i,e){let t=_t[e];if(t===void 0){const n=OT.get(e);if(n!==void 0)t=_t[n],ft('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,n);else throw new Error("Can not resolve #include <"+e+">")}return dd(t)}const zT=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function ym(i){return i.replace(zT,kT)}function kT(i,e,t,n){let s="";for(let r=parseInt(e);r<parseInt(t);r++)s+=n.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function bm(i){let e=`precision ${i.precision} float;
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
#define LOW_PRECISION`),e}function HT(i){let e="SHADOWMAP_TYPE_BASIC";return i.shadowMapType===Rg?e="SHADOWMAP_TYPE_PCF":i.shadowMapType===YA?e="SHADOWMAP_TYPE_PCF_SOFT":i.shadowMapType===Ss&&(e="SHADOWMAP_TYPE_VSM"),e}function VT(i){let e="ENVMAP_TYPE_CUBE";if(i.envMap)switch(i.envMapMode){case Xo:case qo:e="ENVMAP_TYPE_CUBE";break;case Qc:e="ENVMAP_TYPE_CUBE_UV";break}return e}function GT(i){let e="ENVMAP_MODE_REFLECTION";if(i.envMap)switch(i.envMapMode){case qo:e="ENVMAP_MODE_REFRACTION";break}return e}function WT(i){let e="ENVMAP_BLENDING_NONE";if(i.envMap)switch(i.combine){case Dg:e="ENVMAP_BLENDING_MULTIPLY";break;case fS:e="ENVMAP_BLENDING_MIX";break;case dS:e="ENVMAP_BLENDING_ADD";break}return e}function XT(i){const e=i.envMapCubeUVHeight;if(e===null)return null;const t=Math.log2(e)-2,n=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:n,maxMip:t}}function qT(i,e,t,n){const s=i.getContext(),r=t.defines;let o=t.vertexShader,a=t.fragmentShader;const l=HT(t),c=VT(t),u=GT(t),f=WT(t),d=XT(t),h=FT(t),x=LT(r),p=s.createProgram();let g,m,_=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(g=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x].filter(ya).join(`
`),g.length>0&&(g+=`
`),m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x].filter(ya).join(`
`),m.length>0&&(m+=`
`)):(g=[bm(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+u:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(ya).join(`
`),m=[bm(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+c:"",t.envMap?"#define "+u:"",t.envMap?"#define "+f:"",d?"#define CUBEUV_TEXEL_WIDTH "+d.texelWidth:"",d?"#define CUBEUV_TEXEL_HEIGHT "+d.texelHeight:"",d?"#define CUBEUV_MAX_MIP "+d.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor||t.batchingColor?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==sr?"#define TONE_MAPPING":"",t.toneMapping!==sr?_t.tonemapping_pars_fragment:"",t.toneMapping!==sr?DT("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",_t.colorspace_pars_fragment,IT("linearToOutputTexel",t.outputColorSpace),PT(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(ya).join(`
`)),o=dd(o),o=Am(o,t),o=Sm(o,t),a=dd(a),a=Am(a,t),a=Sm(a,t),o=ym(o),a=ym(a),t.isRawShaderMaterial!==!0&&(_=`#version 300 es
`,g=[h,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+g,m=["#define varying in",t.glslVersion===Fp?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===Fp?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+m);const A=_+g+o,v=_+m+a,S=xm(s,s.VERTEX_SHADER,A),y=xm(s,s.FRAGMENT_SHADER,v);s.attachShader(p,S),s.attachShader(p,y),t.index0AttributeName!==void 0?s.bindAttribLocation(p,0,t.index0AttributeName):t.morphTargets===!0&&s.bindAttribLocation(p,0,"position"),s.linkProgram(p);function M(D){if(i.debug.checkShaderErrors){const F=s.getProgramInfoLog(p)||"",O=s.getShaderInfoLog(S)||"",z=s.getShaderInfoLog(y)||"",V=F.trim(),H=O.trim(),q=z.trim();let G=!0,$=!0;if(s.getProgramParameter(p,s.LINK_STATUS)===!1)if(G=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(s,p,S,y);else{const fe=vm(s,S,"vertex"),Y=vm(s,y,"fragment");fn("THREE.WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(p,s.VALIDATE_STATUS)+`

Material Name: `+D.name+`
Material Type: `+D.type+`

Program Info Log: `+V+`
`+fe+`
`+Y)}else V!==""?ft("WebGLProgram: Program Info Log:",V):(H===""||q==="")&&($=!1);$&&(D.diagnostics={runnable:G,programLog:V,vertexShader:{log:H,prefix:g},fragmentShader:{log:q,prefix:m}})}s.deleteShader(S),s.deleteShader(y),E=new dc(s,p),b=BT(s,p)}let E;this.getUniforms=function(){return E===void 0&&M(this),E};let b;this.getAttributes=function(){return b===void 0&&M(this),b};let C=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return C===!1&&(C=s.getProgramParameter(p,TT)),C},this.destroy=function(){n.releaseStatesOfProgram(this),s.deleteProgram(p),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=ET++,this.cacheKey=e,this.usedTimes=1,this.program=p,this.vertexShader=S,this.fragmentShader=y,this}let YT=0;class QT{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const t=e.vertexShader,n=e.fragmentShader,s=this._getShaderStage(t),r=this._getShaderStage(n),o=this._getShaderCacheForMaterial(e);return o.has(s)===!1&&(o.add(s),s.usedTimes++),o.has(r)===!1&&(o.add(r),r.usedTimes++),this}remove(e){const t=this.materialCache.get(e);for(const n of t)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const t=this.materialCache;let n=t.get(e);return n===void 0&&(n=new Set,t.set(e,n)),n}_getShaderStage(e){const t=this.shaderCache;let n=t.get(e);return n===void 0&&(n=new KT(e),t.set(e,n)),n}}class KT{constructor(e){this.id=YT++,this.code=e,this.usedTimes=0}}function jT(i,e,t,n,s,r,o){const a=new Gg,l=new QT,c=new Set,u=[],f=s.logarithmicDepthBuffer,d=s.vertexTextures;let h=s.precision;const x={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function p(b){return c.add(b),b===0?"uv":`uv${b}`}function g(b,C,D,F,O){const z=F.fog,V=O.geometry,H=b.isMeshStandardMaterial?F.environment:null,q=(b.isMeshStandardMaterial?t:e).get(b.envMap||H),G=q&&q.mapping===Qc?q.image.height:null,$=x[b.type];b.precision!==null&&(h=s.getMaxPrecision(b.precision),h!==b.precision&&ft("WebGLProgram.getParameters:",b.precision,"not supported, using",h,"instead."));const fe=V.morphAttributes.position||V.morphAttributes.normal||V.morphAttributes.color,Y=fe!==void 0?fe.length:0;let we=0;V.morphAttributes.position!==void 0&&(we=1),V.morphAttributes.normal!==void 0&&(we=2),V.morphAttributes.color!==void 0&&(we=3);let ze,ke,We,ne;if($){const ot=Ki[$];ze=ot.vertexShader,ke=ot.fragmentShader}else ze=b.vertexShader,ke=b.fragmentShader,l.update(b),We=l.getVertexShaderID(b),ne=l.getFragmentShaderID(b);const ue=i.getRenderTarget(),Se=i.state.buffers.depth.getReversed(),he=O.isInstancedMesh===!0,Ee=O.isBatchedMesh===!0,Ze=!!b.map,U=!!b.matcap,N=!!q,K=!!b.aoMap,R=!!b.lightMap,te=!!b.bumpMap,oe=!!b.normalMap,pe=!!b.displacementMap,ie=!!b.emissiveMap,me=!!b.metalnessMap,se=!!b.roughnessMap,ve=b.anisotropy>0,I=b.clearcoat>0,T=b.dispersion>0,X=b.iridescence>0,re=b.sheen>0,de=b.transmission>0,ee=ve&&!!b.anisotropyMap,Ue=I&&!!b.clearcoatMap,ye=I&&!!b.clearcoatNormalMap,Xe=I&&!!b.clearcoatRoughnessMap,k=X&&!!b.iridescenceMap,Z=X&&!!b.iridescenceThicknessMap,xe=re&&!!b.sheenColorMap,Re=re&&!!b.sheenRoughnessMap,Be=!!b.specularMap,Fe=!!b.specularColorMap,je=!!b.specularIntensityMap,W=de&&!!b.transmissionMap,Le=de&&!!b.thicknessMap,Me=!!b.gradientMap,be=!!b.alphaMap,Ae=b.alphaTest>0,ge=!!b.alphaHash,qe=!!b.extensions;let Je=sr;b.toneMapped&&(ue===null||ue.isXRRenderTarget===!0)&&(Je=i.toneMapping);const rt={shaderID:$,shaderType:b.type,shaderName:b.name,vertexShader:ze,fragmentShader:ke,defines:b.defines,customVertexShaderID:We,customFragmentShaderID:ne,isRawShaderMaterial:b.isRawShaderMaterial===!0,glslVersion:b.glslVersion,precision:h,batching:Ee,batchingColor:Ee&&O._colorsTexture!==null,instancing:he,instancingColor:he&&O.instanceColor!==null,instancingMorph:he&&O.morphTexture!==null,supportsVertexTextures:d,outputColorSpace:ue===null?i.outputColorSpace:ue.isXRRenderTarget===!0?ue.texture.colorSpace:Qo,alphaToCoverage:!!b.alphaToCoverage,map:Ze,matcap:U,envMap:N,envMapMode:N&&q.mapping,envMapCubeUVHeight:G,aoMap:K,lightMap:R,bumpMap:te,normalMap:oe,displacementMap:d&&pe,emissiveMap:ie,normalMapObjectSpace:oe&&b.normalMapType===MS,normalMapTangentSpace:oe&&b.normalMapType===bS,metalnessMap:me,roughnessMap:se,anisotropy:ve,anisotropyMap:ee,clearcoat:I,clearcoatMap:Ue,clearcoatNormalMap:ye,clearcoatRoughnessMap:Xe,dispersion:T,iridescence:X,iridescenceMap:k,iridescenceThicknessMap:Z,sheen:re,sheenColorMap:xe,sheenRoughnessMap:Re,specularMap:Be,specularColorMap:Fe,specularIntensityMap:je,transmission:de,transmissionMap:W,thicknessMap:Le,gradientMap:Me,opaque:b.transparent===!1&&b.blending===ir&&b.alphaToCoverage===!1,alphaMap:be,alphaTest:Ae,alphaHash:ge,combine:b.combine,mapUv:Ze&&p(b.map.channel),aoMapUv:K&&p(b.aoMap.channel),lightMapUv:R&&p(b.lightMap.channel),bumpMapUv:te&&p(b.bumpMap.channel),normalMapUv:oe&&p(b.normalMap.channel),displacementMapUv:pe&&p(b.displacementMap.channel),emissiveMapUv:ie&&p(b.emissiveMap.channel),metalnessMapUv:me&&p(b.metalnessMap.channel),roughnessMapUv:se&&p(b.roughnessMap.channel),anisotropyMapUv:ee&&p(b.anisotropyMap.channel),clearcoatMapUv:Ue&&p(b.clearcoatMap.channel),clearcoatNormalMapUv:ye&&p(b.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Xe&&p(b.clearcoatRoughnessMap.channel),iridescenceMapUv:k&&p(b.iridescenceMap.channel),iridescenceThicknessMapUv:Z&&p(b.iridescenceThicknessMap.channel),sheenColorMapUv:xe&&p(b.sheenColorMap.channel),sheenRoughnessMapUv:Re&&p(b.sheenRoughnessMap.channel),specularMapUv:Be&&p(b.specularMap.channel),specularColorMapUv:Fe&&p(b.specularColorMap.channel),specularIntensityMapUv:je&&p(b.specularIntensityMap.channel),transmissionMapUv:W&&p(b.transmissionMap.channel),thicknessMapUv:Le&&p(b.thicknessMap.channel),alphaMapUv:be&&p(b.alphaMap.channel),vertexTangents:!!V.attributes.tangent&&(oe||ve),vertexColors:b.vertexColors,vertexAlphas:b.vertexColors===!0&&!!V.attributes.color&&V.attributes.color.itemSize===4,pointsUvs:O.isPoints===!0&&!!V.attributes.uv&&(Ze||be),fog:!!z,useFog:b.fog===!0,fogExp2:!!z&&z.isFogExp2,flatShading:b.flatShading===!0&&b.wireframe===!1,sizeAttenuation:b.sizeAttenuation===!0,logarithmicDepthBuffer:f,reversedDepthBuffer:Se,skinning:O.isSkinnedMesh===!0,morphTargets:V.morphAttributes.position!==void 0,morphNormals:V.morphAttributes.normal!==void 0,morphColors:V.morphAttributes.color!==void 0,morphTargetsCount:Y,morphTextureStride:we,numDirLights:C.directional.length,numPointLights:C.point.length,numSpotLights:C.spot.length,numSpotLightMaps:C.spotLightMap.length,numRectAreaLights:C.rectArea.length,numHemiLights:C.hemi.length,numDirLightShadows:C.directionalShadowMap.length,numPointLightShadows:C.pointShadowMap.length,numSpotLightShadows:C.spotShadowMap.length,numSpotLightShadowsWithMaps:C.numSpotLightShadowsWithMaps,numLightProbes:C.numLightProbes,numClippingPlanes:o.numPlanes,numClipIntersection:o.numIntersection,dithering:b.dithering,shadowMapEnabled:i.shadowMap.enabled&&D.length>0,shadowMapType:i.shadowMap.type,toneMapping:Je,decodeVideoTexture:Ze&&b.map.isVideoTexture===!0&&Et.getTransfer(b.map.colorSpace)===Ot,decodeVideoTextureEmissive:ie&&b.emissiveMap.isVideoTexture===!0&&Et.getTransfer(b.emissiveMap.colorSpace)===Ot,premultipliedAlpha:b.premultipliedAlpha,doubleSided:b.side===Ei,flipSided:b.side===Jn,useDepthPacking:b.depthPacking>=0,depthPacking:b.depthPacking||0,index0AttributeName:b.index0AttributeName,extensionClipCullDistance:qe&&b.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(qe&&b.extensions.multiDraw===!0||Ee)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:b.customProgramCacheKey()};return rt.vertexUv1s=c.has(1),rt.vertexUv2s=c.has(2),rt.vertexUv3s=c.has(3),c.clear(),rt}function m(b){const C=[];if(b.shaderID?C.push(b.shaderID):(C.push(b.customVertexShaderID),C.push(b.customFragmentShaderID)),b.defines!==void 0)for(const D in b.defines)C.push(D),C.push(b.defines[D]);return b.isRawShaderMaterial===!1&&(_(C,b),A(C,b),C.push(i.outputColorSpace)),C.push(b.customProgramCacheKey),C.join()}function _(b,C){b.push(C.precision),b.push(C.outputColorSpace),b.push(C.envMapMode),b.push(C.envMapCubeUVHeight),b.push(C.mapUv),b.push(C.alphaMapUv),b.push(C.lightMapUv),b.push(C.aoMapUv),b.push(C.bumpMapUv),b.push(C.normalMapUv),b.push(C.displacementMapUv),b.push(C.emissiveMapUv),b.push(C.metalnessMapUv),b.push(C.roughnessMapUv),b.push(C.anisotropyMapUv),b.push(C.clearcoatMapUv),b.push(C.clearcoatNormalMapUv),b.push(C.clearcoatRoughnessMapUv),b.push(C.iridescenceMapUv),b.push(C.iridescenceThicknessMapUv),b.push(C.sheenColorMapUv),b.push(C.sheenRoughnessMapUv),b.push(C.specularMapUv),b.push(C.specularColorMapUv),b.push(C.specularIntensityMapUv),b.push(C.transmissionMapUv),b.push(C.thicknessMapUv),b.push(C.combine),b.push(C.fogExp2),b.push(C.sizeAttenuation),b.push(C.morphTargetsCount),b.push(C.morphAttributeCount),b.push(C.numDirLights),b.push(C.numPointLights),b.push(C.numSpotLights),b.push(C.numSpotLightMaps),b.push(C.numHemiLights),b.push(C.numRectAreaLights),b.push(C.numDirLightShadows),b.push(C.numPointLightShadows),b.push(C.numSpotLightShadows),b.push(C.numSpotLightShadowsWithMaps),b.push(C.numLightProbes),b.push(C.shadowMapType),b.push(C.toneMapping),b.push(C.numClippingPlanes),b.push(C.numClipIntersection),b.push(C.depthPacking)}function A(b,C){a.disableAll(),C.supportsVertexTextures&&a.enable(0),C.instancing&&a.enable(1),C.instancingColor&&a.enable(2),C.instancingMorph&&a.enable(3),C.matcap&&a.enable(4),C.envMap&&a.enable(5),C.normalMapObjectSpace&&a.enable(6),C.normalMapTangentSpace&&a.enable(7),C.clearcoat&&a.enable(8),C.iridescence&&a.enable(9),C.alphaTest&&a.enable(10),C.vertexColors&&a.enable(11),C.vertexAlphas&&a.enable(12),C.vertexUv1s&&a.enable(13),C.vertexUv2s&&a.enable(14),C.vertexUv3s&&a.enable(15),C.vertexTangents&&a.enable(16),C.anisotropy&&a.enable(17),C.alphaHash&&a.enable(18),C.batching&&a.enable(19),C.dispersion&&a.enable(20),C.batchingColor&&a.enable(21),C.gradientMap&&a.enable(22),b.push(a.mask),a.disableAll(),C.fog&&a.enable(0),C.useFog&&a.enable(1),C.flatShading&&a.enable(2),C.logarithmicDepthBuffer&&a.enable(3),C.reversedDepthBuffer&&a.enable(4),C.skinning&&a.enable(5),C.morphTargets&&a.enable(6),C.morphNormals&&a.enable(7),C.morphColors&&a.enable(8),C.premultipliedAlpha&&a.enable(9),C.shadowMapEnabled&&a.enable(10),C.doubleSided&&a.enable(11),C.flipSided&&a.enable(12),C.useDepthPacking&&a.enable(13),C.dithering&&a.enable(14),C.transmission&&a.enable(15),C.sheen&&a.enable(16),C.opaque&&a.enable(17),C.pointsUvs&&a.enable(18),C.decodeVideoTexture&&a.enable(19),C.decodeVideoTextureEmissive&&a.enable(20),C.alphaToCoverage&&a.enable(21),b.push(a.mask)}function v(b){const C=x[b.type];let D;if(C){const F=Ki[C];D=_y.clone(F.uniforms)}else D=b.uniforms;return D}function S(b,C){let D;for(let F=0,O=u.length;F<O;F++){const z=u[F];if(z.cacheKey===C){D=z,++D.usedTimes;break}}return D===void 0&&(D=new qT(i,C,b,r),u.push(D)),D}function y(b){if(--b.usedTimes===0){const C=u.indexOf(b);u[C]=u[u.length-1],u.pop(),b.destroy()}}function M(b){l.remove(b)}function E(){l.dispose()}return{getParameters:g,getProgramCacheKey:m,getUniforms:v,acquireProgram:S,releaseProgram:y,releaseShaderCache:M,programs:u,dispose:E}}function $T(){let i=new WeakMap;function e(o){return i.has(o)}function t(o){let a=i.get(o);return a===void 0&&(a={},i.set(o,a)),a}function n(o){i.delete(o)}function s(o,a,l){i.get(o)[a]=l}function r(){i=new WeakMap}return{has:e,get:t,remove:n,update:s,dispose:r}}function ZT(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.material.id!==e.material.id?i.material.id-e.material.id:i.z!==e.z?i.z-e.z:i.id-e.id}function Mm(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.z!==e.z?e.z-i.z:i.id-e.id}function Cm(){const i=[];let e=0;const t=[],n=[],s=[];function r(){e=0,t.length=0,n.length=0,s.length=0}function o(f,d,h,x,p,g){let m=i[e];return m===void 0?(m={id:f.id,object:f,geometry:d,material:h,groupOrder:x,renderOrder:f.renderOrder,z:p,group:g},i[e]=m):(m.id=f.id,m.object=f,m.geometry=d,m.material=h,m.groupOrder=x,m.renderOrder=f.renderOrder,m.z=p,m.group=g),e++,m}function a(f,d,h,x,p,g){const m=o(f,d,h,x,p,g);h.transmission>0?n.push(m):h.transparent===!0?s.push(m):t.push(m)}function l(f,d,h,x,p,g){const m=o(f,d,h,x,p,g);h.transmission>0?n.unshift(m):h.transparent===!0?s.unshift(m):t.unshift(m)}function c(f,d){t.length>1&&t.sort(f||ZT),n.length>1&&n.sort(d||Mm),s.length>1&&s.sort(d||Mm)}function u(){for(let f=e,d=i.length;f<d;f++){const h=i[f];if(h.id===null)break;h.id=null,h.object=null,h.geometry=null,h.material=null,h.group=null}}return{opaque:t,transmissive:n,transparent:s,init:r,push:a,unshift:l,finish:u,sort:c}}function JT(){let i=new WeakMap;function e(n,s){const r=i.get(n);let o;return r===void 0?(o=new Cm,i.set(n,[o])):s>=r.length?(o=new Cm,r.push(o)):o=r[s],o}function t(){i=new WeakMap}return{get:e,dispose:t}}function e1(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new B,color:new bt};break;case"SpotLight":t={position:new B,direction:new B,color:new bt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new B,color:new bt,distance:0,decay:0};break;case"HemisphereLight":t={direction:new B,skyColor:new bt,groundColor:new bt};break;case"RectAreaLight":t={color:new bt,position:new B,halfWidth:new B,halfHeight:new B};break}return i[e.id]=t,t}}}function t1(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ke};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ke};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ke,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[e.id]=t,t}}}let n1=0;function i1(i,e){return(e.castShadow?2:0)-(i.castShadow?2:0)+(e.map?1:0)-(i.map?1:0)}function s1(i){const e=new e1,t=t1(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)n.probe.push(new B);const s=new B,r=new st,o=new st;function a(c){let u=0,f=0,d=0;for(let b=0;b<9;b++)n.probe[b].set(0,0,0);let h=0,x=0,p=0,g=0,m=0,_=0,A=0,v=0,S=0,y=0,M=0;c.sort(i1);for(let b=0,C=c.length;b<C;b++){const D=c[b],F=D.color,O=D.intensity,z=D.distance,V=D.shadow&&D.shadow.map?D.shadow.map.texture:null;if(D.isAmbientLight)u+=F.r*O,f+=F.g*O,d+=F.b*O;else if(D.isLightProbe){for(let H=0;H<9;H++)n.probe[H].addScaledVector(D.sh.coefficients[H],O);M++}else if(D.isDirectionalLight){const H=e.get(D);if(H.color.copy(D.color).multiplyScalar(D.intensity),D.castShadow){const q=D.shadow,G=t.get(D);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,n.directionalShadow[h]=G,n.directionalShadowMap[h]=V,n.directionalShadowMatrix[h]=D.shadow.matrix,_++}n.directional[h]=H,h++}else if(D.isSpotLight){const H=e.get(D);H.position.setFromMatrixPosition(D.matrixWorld),H.color.copy(F).multiplyScalar(O),H.distance=z,H.coneCos=Math.cos(D.angle),H.penumbraCos=Math.cos(D.angle*(1-D.penumbra)),H.decay=D.decay,n.spot[p]=H;const q=D.shadow;if(D.map&&(n.spotLightMap[S]=D.map,S++,q.updateMatrices(D),D.castShadow&&y++),n.spotLightMatrix[p]=q.matrix,D.castShadow){const G=t.get(D);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,n.spotShadow[p]=G,n.spotShadowMap[p]=V,v++}p++}else if(D.isRectAreaLight){const H=e.get(D);H.color.copy(F).multiplyScalar(O),H.halfWidth.set(D.width*.5,0,0),H.halfHeight.set(0,D.height*.5,0),n.rectArea[g]=H,g++}else if(D.isPointLight){const H=e.get(D);if(H.color.copy(D.color).multiplyScalar(D.intensity),H.distance=D.distance,H.decay=D.decay,D.castShadow){const q=D.shadow,G=t.get(D);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,G.shadowCameraNear=q.camera.near,G.shadowCameraFar=q.camera.far,n.pointShadow[x]=G,n.pointShadowMap[x]=V,n.pointShadowMatrix[x]=D.shadow.matrix,A++}n.point[x]=H,x++}else if(D.isHemisphereLight){const H=e.get(D);H.skyColor.copy(D.color).multiplyScalar(O),H.groundColor.copy(D.groundColor).multiplyScalar(O),n.hemi[m]=H,m++}}g>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=Ne.LTC_FLOAT_1,n.rectAreaLTC2=Ne.LTC_FLOAT_2):(n.rectAreaLTC1=Ne.LTC_HALF_1,n.rectAreaLTC2=Ne.LTC_HALF_2)),n.ambient[0]=u,n.ambient[1]=f,n.ambient[2]=d;const E=n.hash;(E.directionalLength!==h||E.pointLength!==x||E.spotLength!==p||E.rectAreaLength!==g||E.hemiLength!==m||E.numDirectionalShadows!==_||E.numPointShadows!==A||E.numSpotShadows!==v||E.numSpotMaps!==S||E.numLightProbes!==M)&&(n.directional.length=h,n.spot.length=p,n.rectArea.length=g,n.point.length=x,n.hemi.length=m,n.directionalShadow.length=_,n.directionalShadowMap.length=_,n.pointShadow.length=A,n.pointShadowMap.length=A,n.spotShadow.length=v,n.spotShadowMap.length=v,n.directionalShadowMatrix.length=_,n.pointShadowMatrix.length=A,n.spotLightMatrix.length=v+S-y,n.spotLightMap.length=S,n.numSpotLightShadowsWithMaps=y,n.numLightProbes=M,E.directionalLength=h,E.pointLength=x,E.spotLength=p,E.rectAreaLength=g,E.hemiLength=m,E.numDirectionalShadows=_,E.numPointShadows=A,E.numSpotShadows=v,E.numSpotMaps=S,E.numLightProbes=M,n.version=n1++)}function l(c,u){let f=0,d=0,h=0,x=0,p=0;const g=u.matrixWorldInverse;for(let m=0,_=c.length;m<_;m++){const A=c[m];if(A.isDirectionalLight){const v=n.directional[f];v.direction.setFromMatrixPosition(A.matrixWorld),s.setFromMatrixPosition(A.target.matrixWorld),v.direction.sub(s),v.direction.transformDirection(g),f++}else if(A.isSpotLight){const v=n.spot[h];v.position.setFromMatrixPosition(A.matrixWorld),v.position.applyMatrix4(g),v.direction.setFromMatrixPosition(A.matrixWorld),s.setFromMatrixPosition(A.target.matrixWorld),v.direction.sub(s),v.direction.transformDirection(g),h++}else if(A.isRectAreaLight){const v=n.rectArea[x];v.position.setFromMatrixPosition(A.matrixWorld),v.position.applyMatrix4(g),o.identity(),r.copy(A.matrixWorld),r.premultiply(g),o.extractRotation(r),v.halfWidth.set(A.width*.5,0,0),v.halfHeight.set(0,A.height*.5,0),v.halfWidth.applyMatrix4(o),v.halfHeight.applyMatrix4(o),x++}else if(A.isPointLight){const v=n.point[d];v.position.setFromMatrixPosition(A.matrixWorld),v.position.applyMatrix4(g),d++}else if(A.isHemisphereLight){const v=n.hemi[p];v.direction.setFromMatrixPosition(A.matrixWorld),v.direction.transformDirection(g),p++}}}return{setup:a,setupView:l,state:n}}function Tm(i){const e=new s1(i),t=[],n=[];function s(u){c.camera=u,t.length=0,n.length=0}function r(u){t.push(u)}function o(u){n.push(u)}function a(){e.setup(t)}function l(u){e.setupView(t,u)}const c={lightsArray:t,shadowsArray:n,camera:null,lights:e,transmissionRenderTarget:{}};return{init:s,state:c,setupLights:a,setupLightsView:l,pushLight:r,pushShadow:o}}function r1(i){let e=new WeakMap;function t(s,r=0){const o=e.get(s);let a;return o===void 0?(a=new Tm(i),e.set(s,[a])):r>=o.length?(a=new Tm(i),o.push(a)):a=o[r],a}function n(){e=new WeakMap}return{get:t,dispose:n}}const o1=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,a1=`uniform sampler2D shadow_pass;
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
}`;function l1(i,e,t){let n=new jg;const s=new Ke,r=new Ke,o=new Jt,a=new Py({depthPacking:yS}),l=new Fy,c={},u=t.maxTextureSize,f={[os]:Jn,[Jn]:os,[Ei]:Ei},d=new Yn({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Ke},radius:{value:4}},vertexShader:o1,fragmentShader:a1}),h=d.clone();h.defines.HORIZONTAL_PASS=1;const x=new Kn;x.setAttribute("position",new Li(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const p=new hn(x,d),g=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Rg;let m=this.type;this.render=function(y,M,E){if(g.enabled===!1||g.autoUpdate===!1&&g.needsUpdate===!1||y.length===0)return;const b=i.getRenderTarget(),C=i.getActiveCubeFace(),D=i.getActiveMipmapLevel(),F=i.state;F.setBlending(Ds),F.buffers.depth.getReversed()===!0?F.buffers.color.setClear(0,0,0,0):F.buffers.color.setClear(1,1,1,1),F.buffers.depth.setTest(!0),F.setScissorTest(!1);const O=m!==Ss&&this.type===Ss,z=m===Ss&&this.type!==Ss;for(let V=0,H=y.length;V<H;V++){const q=y[V],G=q.shadow;if(G===void 0){ft("WebGLShadowMap:",q,"has no shadow.");continue}if(G.autoUpdate===!1&&G.needsUpdate===!1)continue;s.copy(G.mapSize);const $=G.getFrameExtents();if(s.multiply($),r.copy(G.mapSize),(s.x>u||s.y>u)&&(s.x>u&&(r.x=Math.floor(u/$.x),s.x=r.x*$.x,G.mapSize.x=r.x),s.y>u&&(r.y=Math.floor(u/$.y),s.y=r.y*$.y,G.mapSize.y=r.y)),G.map===null||O===!0||z===!0){const Y=this.type!==Ss?{minFilter:xi,magFilter:xi}:{};G.map!==null&&G.map.dispose(),G.map=new cr(s.x,s.y,Y),G.map.texture.name=q.name+".shadowMap",G.camera.updateProjectionMatrix()}i.setRenderTarget(G.map),i.clear();const fe=G.getViewportCount();for(let Y=0;Y<fe;Y++){const we=G.getViewport(Y);o.set(r.x*we.x,r.y*we.y,r.x*we.z,r.y*we.w),F.viewport(o),G.updateMatrices(q,Y),n=G.getFrustum(),v(M,E,G.camera,q,this.type)}G.isPointLightShadow!==!0&&this.type===Ss&&_(G,E),G.needsUpdate=!1}m=this.type,g.needsUpdate=!1,i.setRenderTarget(b,C,D)};function _(y,M){const E=e.update(p);d.defines.VSM_SAMPLES!==y.blurSamples&&(d.defines.VSM_SAMPLES=y.blurSamples,h.defines.VSM_SAMPLES=y.blurSamples,d.needsUpdate=!0,h.needsUpdate=!0),y.mapPass===null&&(y.mapPass=new cr(s.x,s.y)),d.uniforms.shadow_pass.value=y.map.texture,d.uniforms.resolution.value=y.mapSize,d.uniforms.radius.value=y.radius,i.setRenderTarget(y.mapPass),i.clear(),i.renderBufferDirect(M,null,E,d,p,null),h.uniforms.shadow_pass.value=y.mapPass.texture,h.uniforms.resolution.value=y.mapSize,h.uniforms.radius.value=y.radius,i.setRenderTarget(y.map),i.clear(),i.renderBufferDirect(M,null,E,h,p,null)}function A(y,M,E,b){let C=null;const D=E.isPointLight===!0?y.customDistanceMaterial:y.customDepthMaterial;if(D!==void 0)C=D;else if(C=E.isPointLight===!0?l:a,i.localClippingEnabled&&M.clipShadows===!0&&Array.isArray(M.clippingPlanes)&&M.clippingPlanes.length!==0||M.displacementMap&&M.displacementScale!==0||M.alphaMap&&M.alphaTest>0||M.map&&M.alphaTest>0||M.alphaToCoverage===!0){const F=C.uuid,O=M.uuid;let z=c[F];z===void 0&&(z={},c[F]=z);let V=z[O];V===void 0&&(V=C.clone(),z[O]=V,M.addEventListener("dispose",S)),C=V}if(C.visible=M.visible,C.wireframe=M.wireframe,b===Ss?C.side=M.shadowSide!==null?M.shadowSide:M.side:C.side=M.shadowSide!==null?M.shadowSide:f[M.side],C.alphaMap=M.alphaMap,C.alphaTest=M.alphaToCoverage===!0?.5:M.alphaTest,C.map=M.map,C.clipShadows=M.clipShadows,C.clippingPlanes=M.clippingPlanes,C.clipIntersection=M.clipIntersection,C.displacementMap=M.displacementMap,C.displacementScale=M.displacementScale,C.displacementBias=M.displacementBias,C.wireframeLinewidth=M.wireframeLinewidth,C.linewidth=M.linewidth,E.isPointLight===!0&&C.isMeshDistanceMaterial===!0){const F=i.properties.get(C);F.light=E}return C}function v(y,M,E,b,C){if(y.visible===!1)return;if(y.layers.test(M.layers)&&(y.isMesh||y.isLine||y.isPoints)&&(y.castShadow||y.receiveShadow&&C===Ss)&&(!y.frustumCulled||n.intersectsObject(y))){y.modelViewMatrix.multiplyMatrices(E.matrixWorldInverse,y.matrixWorld);const O=e.update(y),z=y.material;if(Array.isArray(z)){const V=O.groups;for(let H=0,q=V.length;H<q;H++){const G=V[H],$=z[G.materialIndex];if($&&$.visible){const fe=A(y,$,b,C);y.onBeforeShadow(i,y,M,E,O,fe,G),i.renderBufferDirect(E,null,O,fe,y,G),y.onAfterShadow(i,y,M,E,O,fe,G)}}}else if(z.visible){const V=A(y,z,b,C);y.onBeforeShadow(i,y,M,E,O,V,null),i.renderBufferDirect(E,null,O,V,y,null),y.onAfterShadow(i,y,M,E,O,V,null)}}const F=y.children;for(let O=0,z=F.length;O<z;O++)v(F[O],M,E,b,C)}function S(y){y.target.removeEventListener("dispose",S);for(const E in c){const b=c[E],C=y.target.uuid;C in b&&(b[C].dispose(),delete b[C])}}}const c1={[Tf]:Ef,[wf]:Df,[Rf]:Pf,[Wo]:If,[Ef]:Tf,[Df]:wf,[Pf]:Rf,[If]:Wo};function u1(i,e){function t(){let W=!1;const Le=new Jt;let Me=null;const be=new Jt(0,0,0,0);return{setMask:function(Ae){Me!==Ae&&!W&&(i.colorMask(Ae,Ae,Ae,Ae),Me=Ae)},setLocked:function(Ae){W=Ae},setClear:function(Ae,ge,qe,Je,rt){rt===!0&&(Ae*=Je,ge*=Je,qe*=Je),Le.set(Ae,ge,qe,Je),be.equals(Le)===!1&&(i.clearColor(Ae,ge,qe,Je),be.copy(Le))},reset:function(){W=!1,Me=null,be.set(-1,0,0,0)}}}function n(){let W=!1,Le=!1,Me=null,be=null,Ae=null;return{setReversed:function(ge){if(Le!==ge){const qe=e.get("EXT_clip_control");ge?qe.clipControlEXT(qe.LOWER_LEFT_EXT,qe.ZERO_TO_ONE_EXT):qe.clipControlEXT(qe.LOWER_LEFT_EXT,qe.NEGATIVE_ONE_TO_ONE_EXT),Le=ge;const Je=Ae;Ae=null,this.setClear(Je)}},getReversed:function(){return Le},setTest:function(ge){ge?ue(i.DEPTH_TEST):Se(i.DEPTH_TEST)},setMask:function(ge){Me!==ge&&!W&&(i.depthMask(ge),Me=ge)},setFunc:function(ge){if(Le&&(ge=c1[ge]),be!==ge){switch(ge){case Tf:i.depthFunc(i.NEVER);break;case Ef:i.depthFunc(i.ALWAYS);break;case wf:i.depthFunc(i.LESS);break;case Wo:i.depthFunc(i.LEQUAL);break;case Rf:i.depthFunc(i.EQUAL);break;case If:i.depthFunc(i.GEQUAL);break;case Df:i.depthFunc(i.GREATER);break;case Pf:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}be=ge}},setLocked:function(ge){W=ge},setClear:function(ge){Ae!==ge&&(Le&&(ge=1-ge),i.clearDepth(ge),Ae=ge)},reset:function(){W=!1,Me=null,be=null,Ae=null,Le=!1}}}function s(){let W=!1,Le=null,Me=null,be=null,Ae=null,ge=null,qe=null,Je=null,rt=null;return{setTest:function(ot){W||(ot?ue(i.STENCIL_TEST):Se(i.STENCIL_TEST))},setMask:function(ot){Le!==ot&&!W&&(i.stencilMask(ot),Le=ot)},setFunc:function(ot,Si,ri){(Me!==ot||be!==Si||Ae!==ri)&&(i.stencilFunc(ot,Si,ri),Me=ot,be=Si,Ae=ri)},setOp:function(ot,Si,ri){(ge!==ot||qe!==Si||Je!==ri)&&(i.stencilOp(ot,Si,ri),ge=ot,qe=Si,Je=ri)},setLocked:function(ot){W=ot},setClear:function(ot){rt!==ot&&(i.clearStencil(ot),rt=ot)},reset:function(){W=!1,Le=null,Me=null,be=null,Ae=null,ge=null,qe=null,Je=null,rt=null}}}const r=new t,o=new n,a=new s,l=new WeakMap,c=new WeakMap;let u={},f={},d=new WeakMap,h=[],x=null,p=!1,g=null,m=null,_=null,A=null,v=null,S=null,y=null,M=new bt(0,0,0),E=0,b=!1,C=null,D=null,F=null,O=null,z=null;const V=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let H=!1,q=0;const G=i.getParameter(i.VERSION);G.indexOf("WebGL")!==-1?(q=parseFloat(/^WebGL (\d)/.exec(G)[1]),H=q>=1):G.indexOf("OpenGL ES")!==-1&&(q=parseFloat(/^OpenGL ES (\d)/.exec(G)[1]),H=q>=2);let $=null,fe={};const Y=i.getParameter(i.SCISSOR_BOX),we=i.getParameter(i.VIEWPORT),ze=new Jt().fromArray(Y),ke=new Jt().fromArray(we);function We(W,Le,Me,be){const Ae=new Uint8Array(4),ge=i.createTexture();i.bindTexture(W,ge),i.texParameteri(W,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(W,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let qe=0;qe<Me;qe++)W===i.TEXTURE_3D||W===i.TEXTURE_2D_ARRAY?i.texImage3D(Le,0,i.RGBA,1,1,be,0,i.RGBA,i.UNSIGNED_BYTE,Ae):i.texImage2D(Le+qe,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,Ae);return ge}const ne={};ne[i.TEXTURE_2D]=We(i.TEXTURE_2D,i.TEXTURE_2D,1),ne[i.TEXTURE_CUBE_MAP]=We(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),ne[i.TEXTURE_2D_ARRAY]=We(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),ne[i.TEXTURE_3D]=We(i.TEXTURE_3D,i.TEXTURE_3D,1,1),r.setClear(0,0,0,1),o.setClear(1),a.setClear(0),ue(i.DEPTH_TEST),o.setFunc(Wo),te(!1),oe(Ep),ue(i.CULL_FACE),K(Ds);function ue(W){u[W]!==!0&&(i.enable(W),u[W]=!0)}function Se(W){u[W]!==!1&&(i.disable(W),u[W]=!1)}function he(W,Le){return f[W]!==Le?(i.bindFramebuffer(W,Le),f[W]=Le,W===i.DRAW_FRAMEBUFFER&&(f[i.FRAMEBUFFER]=Le),W===i.FRAMEBUFFER&&(f[i.DRAW_FRAMEBUFFER]=Le),!0):!1}function Ee(W,Le){let Me=h,be=!1;if(W){Me=d.get(Le),Me===void 0&&(Me=[],d.set(Le,Me));const Ae=W.textures;if(Me.length!==Ae.length||Me[0]!==i.COLOR_ATTACHMENT0){for(let ge=0,qe=Ae.length;ge<qe;ge++)Me[ge]=i.COLOR_ATTACHMENT0+ge;Me.length=Ae.length,be=!0}}else Me[0]!==i.BACK&&(Me[0]=i.BACK,be=!0);be&&i.drawBuffers(Me)}function Ze(W){return x!==W?(i.useProgram(W),x=W,!0):!1}const U={[Nr]:i.FUNC_ADD,[QA]:i.FUNC_SUBTRACT,[KA]:i.FUNC_REVERSE_SUBTRACT};U[jA]=i.MIN,U[$A]=i.MAX;const N={[ZA]:i.ZERO,[JA]:i.ONE,[eS]:i.SRC_COLOR,[Qa]:i.SRC_ALPHA,[oS]:i.SRC_ALPHA_SATURATE,[sS]:i.DST_COLOR,[nS]:i.DST_ALPHA,[tS]:i.ONE_MINUS_SRC_COLOR,[Ka]:i.ONE_MINUS_SRC_ALPHA,[rS]:i.ONE_MINUS_DST_COLOR,[iS]:i.ONE_MINUS_DST_ALPHA,[aS]:i.CONSTANT_COLOR,[lS]:i.ONE_MINUS_CONSTANT_COLOR,[cS]:i.CONSTANT_ALPHA,[uS]:i.ONE_MINUS_CONSTANT_ALPHA};function K(W,Le,Me,be,Ae,ge,qe,Je,rt,ot){if(W===Ds){p===!0&&(Se(i.BLEND),p=!1);return}if(p===!1&&(ue(i.BLEND),p=!0),W!==Ig){if(W!==g||ot!==b){if((m!==Nr||v!==Nr)&&(i.blendEquation(i.FUNC_ADD),m=Nr,v=Nr),ot)switch(W){case ir:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case wp:i.blendFunc(i.ONE,i.ONE);break;case Rp:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case Ip:i.blendFuncSeparate(i.DST_COLOR,i.ONE_MINUS_SRC_ALPHA,i.ZERO,i.ONE);break;default:fn("WebGLState: Invalid blending: ",W);break}else switch(W){case ir:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case wp:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE,i.ONE,i.ONE);break;case Rp:fn("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case Ip:fn("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:fn("WebGLState: Invalid blending: ",W);break}_=null,A=null,S=null,y=null,M.set(0,0,0),E=0,g=W,b=ot}return}Ae=Ae||Le,ge=ge||Me,qe=qe||be,(Le!==m||Ae!==v)&&(i.blendEquationSeparate(U[Le],U[Ae]),m=Le,v=Ae),(Me!==_||be!==A||ge!==S||qe!==y)&&(i.blendFuncSeparate(N[Me],N[be],N[ge],N[qe]),_=Me,A=be,S=ge,y=qe),(Je.equals(M)===!1||rt!==E)&&(i.blendColor(Je.r,Je.g,Je.b,rt),M.copy(Je),E=rt),g=W,b=!1}function R(W,Le){W.side===Ei?Se(i.CULL_FACE):ue(i.CULL_FACE);let Me=W.side===Jn;Le&&(Me=!Me),te(Me),W.blending===ir&&W.transparent===!1?K(Ds):K(W.blending,W.blendEquation,W.blendSrc,W.blendDst,W.blendEquationAlpha,W.blendSrcAlpha,W.blendDstAlpha,W.blendColor,W.blendAlpha,W.premultipliedAlpha),o.setFunc(W.depthFunc),o.setTest(W.depthTest),o.setMask(W.depthWrite),r.setMask(W.colorWrite);const be=W.stencilWrite;a.setTest(be),be&&(a.setMask(W.stencilWriteMask),a.setFunc(W.stencilFunc,W.stencilRef,W.stencilFuncMask),a.setOp(W.stencilFail,W.stencilZFail,W.stencilZPass)),ie(W.polygonOffset,W.polygonOffsetFactor,W.polygonOffsetUnits),W.alphaToCoverage===!0?ue(i.SAMPLE_ALPHA_TO_COVERAGE):Se(i.SAMPLE_ALPHA_TO_COVERAGE)}function te(W){C!==W&&(W?i.frontFace(i.CW):i.frontFace(i.CCW),C=W)}function oe(W){W!==XA?(ue(i.CULL_FACE),W!==D&&(W===Ep?i.cullFace(i.BACK):W===qA?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):Se(i.CULL_FACE),D=W}function pe(W){W!==F&&(H&&i.lineWidth(W),F=W)}function ie(W,Le,Me){W?(ue(i.POLYGON_OFFSET_FILL),(O!==Le||z!==Me)&&(i.polygonOffset(Le,Me),O=Le,z=Me)):Se(i.POLYGON_OFFSET_FILL)}function me(W){W?ue(i.SCISSOR_TEST):Se(i.SCISSOR_TEST)}function se(W){W===void 0&&(W=i.TEXTURE0+V-1),$!==W&&(i.activeTexture(W),$=W)}function ve(W,Le,Me){Me===void 0&&($===null?Me=i.TEXTURE0+V-1:Me=$);let be=fe[Me];be===void 0&&(be={type:void 0,texture:void 0},fe[Me]=be),(be.type!==W||be.texture!==Le)&&($!==Me&&(i.activeTexture(Me),$=Me),i.bindTexture(W,Le||ne[W]),be.type=W,be.texture=Le)}function I(){const W=fe[$];W!==void 0&&W.type!==void 0&&(i.bindTexture(W.type,null),W.type=void 0,W.texture=void 0)}function T(){try{i.compressedTexImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function X(){try{i.compressedTexImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function re(){try{i.texSubImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function de(){try{i.texSubImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function ee(){try{i.compressedTexSubImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function Ue(){try{i.compressedTexSubImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function ye(){try{i.texStorage2D(...arguments)}catch(W){W("WebGLState:",W)}}function Xe(){try{i.texStorage3D(...arguments)}catch(W){W("WebGLState:",W)}}function k(){try{i.texImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function Z(){try{i.texImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function xe(W){ze.equals(W)===!1&&(i.scissor(W.x,W.y,W.z,W.w),ze.copy(W))}function Re(W){ke.equals(W)===!1&&(i.viewport(W.x,W.y,W.z,W.w),ke.copy(W))}function Be(W,Le){let Me=c.get(Le);Me===void 0&&(Me=new WeakMap,c.set(Le,Me));let be=Me.get(W);be===void 0&&(be=i.getUniformBlockIndex(Le,W.name),Me.set(W,be))}function Fe(W,Le){const be=c.get(Le).get(W);l.get(Le)!==be&&(i.uniformBlockBinding(Le,be,W.__bindingPointIndex),l.set(Le,be))}function je(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),o.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),u={},$=null,fe={},f={},d=new WeakMap,h=[],x=null,p=!1,g=null,m=null,_=null,A=null,v=null,S=null,y=null,M=new bt(0,0,0),E=0,b=!1,C=null,D=null,F=null,O=null,z=null,ze.set(0,0,i.canvas.width,i.canvas.height),ke.set(0,0,i.canvas.width,i.canvas.height),r.reset(),o.reset(),a.reset()}return{buffers:{color:r,depth:o,stencil:a},enable:ue,disable:Se,bindFramebuffer:he,drawBuffers:Ee,useProgram:Ze,setBlending:K,setMaterial:R,setFlipSided:te,setCullFace:oe,setLineWidth:pe,setPolygonOffset:ie,setScissorTest:me,activeTexture:se,bindTexture:ve,unbindTexture:I,compressedTexImage2D:T,compressedTexImage3D:X,texImage2D:k,texImage3D:Z,updateUBOMapping:Be,uniformBlockBinding:Fe,texStorage2D:ye,texStorage3D:Xe,texSubImage2D:re,texSubImage3D:de,compressedTexSubImage2D:ee,compressedTexSubImage3D:Ue,scissor:xe,viewport:Re,reset:je}}function f1(i,e,t,n,s,r,o){const a=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new Ke,u=new WeakMap;let f;const d=new WeakMap;let h=!1;try{h=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function x(I,T){return h?new OffscreenCanvas(I,T):Cc("canvas")}function p(I,T,X){let re=1;const de=ve(I);if((de.width>X||de.height>X)&&(re=X/Math.max(de.width,de.height)),re<1)if(typeof HTMLImageElement<"u"&&I instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&I instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&I instanceof ImageBitmap||typeof VideoFrame<"u"&&I instanceof VideoFrame){const ee=Math.floor(re*de.width),Ue=Math.floor(re*de.height);f===void 0&&(f=x(ee,Ue));const ye=T?x(ee,Ue):f;return ye.width=ee,ye.height=Ue,ye.getContext("2d").drawImage(I,0,0,ee,Ue),ft("WebGLRenderer: Texture has been resized from ("+de.width+"x"+de.height+") to ("+ee+"x"+Ue+")."),ye}else return"data"in I&&ft("WebGLRenderer: Image in DataTexture is too big ("+de.width+"x"+de.height+")."),I;return I}function g(I){return I.generateMipmaps}function m(I){i.generateMipmap(I)}function _(I){return I.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:I.isWebGL3DRenderTarget?i.TEXTURE_3D:I.isWebGLArrayRenderTarget||I.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function A(I,T,X,re,de=!1){if(I!==null){if(i[I]!==void 0)return i[I];ft("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+I+"'")}let ee=T;if(T===i.RED&&(X===i.FLOAT&&(ee=i.R32F),X===i.HALF_FLOAT&&(ee=i.R16F),X===i.UNSIGNED_BYTE&&(ee=i.R8)),T===i.RED_INTEGER&&(X===i.UNSIGNED_BYTE&&(ee=i.R8UI),X===i.UNSIGNED_SHORT&&(ee=i.R16UI),X===i.UNSIGNED_INT&&(ee=i.R32UI),X===i.BYTE&&(ee=i.R8I),X===i.SHORT&&(ee=i.R16I),X===i.INT&&(ee=i.R32I)),T===i.RG&&(X===i.FLOAT&&(ee=i.RG32F),X===i.HALF_FLOAT&&(ee=i.RG16F),X===i.UNSIGNED_BYTE&&(ee=i.RG8)),T===i.RG_INTEGER&&(X===i.UNSIGNED_BYTE&&(ee=i.RG8UI),X===i.UNSIGNED_SHORT&&(ee=i.RG16UI),X===i.UNSIGNED_INT&&(ee=i.RG32UI),X===i.BYTE&&(ee=i.RG8I),X===i.SHORT&&(ee=i.RG16I),X===i.INT&&(ee=i.RG32I)),T===i.RGB_INTEGER&&(X===i.UNSIGNED_BYTE&&(ee=i.RGB8UI),X===i.UNSIGNED_SHORT&&(ee=i.RGB16UI),X===i.UNSIGNED_INT&&(ee=i.RGB32UI),X===i.BYTE&&(ee=i.RGB8I),X===i.SHORT&&(ee=i.RGB16I),X===i.INT&&(ee=i.RGB32I)),T===i.RGBA_INTEGER&&(X===i.UNSIGNED_BYTE&&(ee=i.RGBA8UI),X===i.UNSIGNED_SHORT&&(ee=i.RGBA16UI),X===i.UNSIGNED_INT&&(ee=i.RGBA32UI),X===i.BYTE&&(ee=i.RGBA8I),X===i.SHORT&&(ee=i.RGBA16I),X===i.INT&&(ee=i.RGBA32I)),T===i.RGB&&(X===i.UNSIGNED_INT_5_9_9_9_REV&&(ee=i.RGB9_E5),X===i.UNSIGNED_INT_10F_11F_11F_REV&&(ee=i.R11F_G11F_B10F)),T===i.RGBA){const Ue=de?bc:Et.getTransfer(re);X===i.FLOAT&&(ee=i.RGBA32F),X===i.HALF_FLOAT&&(ee=i.RGBA16F),X===i.UNSIGNED_BYTE&&(ee=Ue===Ot?i.SRGB8_ALPHA8:i.RGBA8),X===i.UNSIGNED_SHORT_4_4_4_4&&(ee=i.RGBA4),X===i.UNSIGNED_SHORT_5_5_5_1&&(ee=i.RGB5_A1)}return(ee===i.R16F||ee===i.R32F||ee===i.RG16F||ee===i.RG32F||ee===i.RGBA16F||ee===i.RGBA32F)&&e.get("EXT_color_buffer_float"),ee}function v(I,T){let X;return I?T===null||T===Ii||T===$a?X=i.DEPTH24_STENCIL8:T===Hi?X=i.DEPTH32F_STENCIL8:T===ja&&(X=i.DEPTH24_STENCIL8,ft("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):T===null||T===Ii||T===$a?X=i.DEPTH_COMPONENT24:T===Hi?X=i.DEPTH_COMPONENT32F:T===ja&&(X=i.DEPTH_COMPONENT16),X}function S(I,T){return g(I)===!0||I.isFramebufferTexture&&I.minFilter!==xi&&I.minFilter!==Ri?Math.log2(Math.max(T.width,T.height))+1:I.mipmaps!==void 0&&I.mipmaps.length>0?I.mipmaps.length:I.isCompressedTexture&&Array.isArray(I.image)?T.mipmaps.length:1}function y(I){const T=I.target;T.removeEventListener("dispose",y),E(T),T.isVideoTexture&&u.delete(T)}function M(I){const T=I.target;T.removeEventListener("dispose",M),C(T)}function E(I){const T=n.get(I);if(T.__webglInit===void 0)return;const X=I.source,re=d.get(X);if(re){const de=re[T.__cacheKey];de.usedTimes--,de.usedTimes===0&&b(I),Object.keys(re).length===0&&d.delete(X)}n.remove(I)}function b(I){const T=n.get(I);i.deleteTexture(T.__webglTexture);const X=I.source,re=d.get(X);delete re[T.__cacheKey],o.memory.textures--}function C(I){const T=n.get(I);if(I.depthTexture&&(I.depthTexture.dispose(),n.remove(I.depthTexture)),I.isWebGLCubeRenderTarget)for(let re=0;re<6;re++){if(Array.isArray(T.__webglFramebuffer[re]))for(let de=0;de<T.__webglFramebuffer[re].length;de++)i.deleteFramebuffer(T.__webglFramebuffer[re][de]);else i.deleteFramebuffer(T.__webglFramebuffer[re]);T.__webglDepthbuffer&&i.deleteRenderbuffer(T.__webglDepthbuffer[re])}else{if(Array.isArray(T.__webglFramebuffer))for(let re=0;re<T.__webglFramebuffer.length;re++)i.deleteFramebuffer(T.__webglFramebuffer[re]);else i.deleteFramebuffer(T.__webglFramebuffer);if(T.__webglDepthbuffer&&i.deleteRenderbuffer(T.__webglDepthbuffer),T.__webglMultisampledFramebuffer&&i.deleteFramebuffer(T.__webglMultisampledFramebuffer),T.__webglColorRenderbuffer)for(let re=0;re<T.__webglColorRenderbuffer.length;re++)T.__webglColorRenderbuffer[re]&&i.deleteRenderbuffer(T.__webglColorRenderbuffer[re]);T.__webglDepthRenderbuffer&&i.deleteRenderbuffer(T.__webglDepthRenderbuffer)}const X=I.textures;for(let re=0,de=X.length;re<de;re++){const ee=n.get(X[re]);ee.__webglTexture&&(i.deleteTexture(ee.__webglTexture),o.memory.textures--),n.remove(X[re])}n.remove(I)}let D=0;function F(){D=0}function O(){const I=D;return I>=s.maxTextures&&ft("WebGLTextures: Trying to use "+I+" texture units while this GPU supports only "+s.maxTextures),D+=1,I}function z(I){const T=[];return T.push(I.wrapS),T.push(I.wrapT),T.push(I.wrapR||0),T.push(I.magFilter),T.push(I.minFilter),T.push(I.anisotropy),T.push(I.internalFormat),T.push(I.format),T.push(I.type),T.push(I.generateMipmaps),T.push(I.premultiplyAlpha),T.push(I.flipY),T.push(I.unpackAlignment),T.push(I.colorSpace),T.join()}function V(I,T){const X=n.get(I);if(I.isVideoTexture&&me(I),I.isRenderTargetTexture===!1&&I.isExternalTexture!==!0&&I.version>0&&X.__version!==I.version){const re=I.image;if(re===null)ft("WebGLRenderer: Texture marked for update but no image data found.");else if(re.complete===!1)ft("WebGLRenderer: Texture marked for update but image is incomplete");else{ne(X,I,T);return}}else I.isExternalTexture&&(X.__webglTexture=I.sourceTexture?I.sourceTexture:null);t.bindTexture(i.TEXTURE_2D,X.__webglTexture,i.TEXTURE0+T)}function H(I,T){const X=n.get(I);if(I.isRenderTargetTexture===!1&&I.version>0&&X.__version!==I.version){ne(X,I,T);return}else I.isExternalTexture&&(X.__webglTexture=I.sourceTexture?I.sourceTexture:null);t.bindTexture(i.TEXTURE_2D_ARRAY,X.__webglTexture,i.TEXTURE0+T)}function q(I,T){const X=n.get(I);if(I.isRenderTargetTexture===!1&&I.version>0&&X.__version!==I.version){ne(X,I,T);return}t.bindTexture(i.TEXTURE_3D,X.__webglTexture,i.TEXTURE0+T)}function G(I,T){const X=n.get(I);if(I.version>0&&X.__version!==I.version){ue(X,I,T);return}t.bindTexture(i.TEXTURE_CUBE_MAP,X.__webglTexture,i.TEXTURE0+T)}const $={[Bf]:i.REPEAT,[Is]:i.CLAMP_TO_EDGE,[Uf]:i.MIRRORED_REPEAT},fe={[xi]:i.NEAREST,[AS]:i.NEAREST_MIPMAP_NEAREST,[Cl]:i.NEAREST_MIPMAP_LINEAR,[Ri]:i.LINEAR,[xu]:i.LINEAR_MIPMAP_NEAREST,[kr]:i.LINEAR_MIPMAP_LINEAR},Y={[CS]:i.NEVER,[DS]:i.ALWAYS,[TS]:i.LESS,[kg]:i.LEQUAL,[ES]:i.EQUAL,[IS]:i.GEQUAL,[wS]:i.GREATER,[RS]:i.NOTEQUAL};function we(I,T){if(T.type===Hi&&e.has("OES_texture_float_linear")===!1&&(T.magFilter===Ri||T.magFilter===xu||T.magFilter===Cl||T.magFilter===kr||T.minFilter===Ri||T.minFilter===xu||T.minFilter===Cl||T.minFilter===kr)&&ft("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(I,i.TEXTURE_WRAP_S,$[T.wrapS]),i.texParameteri(I,i.TEXTURE_WRAP_T,$[T.wrapT]),(I===i.TEXTURE_3D||I===i.TEXTURE_2D_ARRAY)&&i.texParameteri(I,i.TEXTURE_WRAP_R,$[T.wrapR]),i.texParameteri(I,i.TEXTURE_MAG_FILTER,fe[T.magFilter]),i.texParameteri(I,i.TEXTURE_MIN_FILTER,fe[T.minFilter]),T.compareFunction&&(i.texParameteri(I,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(I,i.TEXTURE_COMPARE_FUNC,Y[T.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(T.magFilter===xi||T.minFilter!==Cl&&T.minFilter!==kr||T.type===Hi&&e.has("OES_texture_float_linear")===!1)return;if(T.anisotropy>1||n.get(T).__currentAnisotropy){const X=e.get("EXT_texture_filter_anisotropic");i.texParameterf(I,X.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(T.anisotropy,s.getMaxAnisotropy())),n.get(T).__currentAnisotropy=T.anisotropy}}}function ze(I,T){let X=!1;I.__webglInit===void 0&&(I.__webglInit=!0,T.addEventListener("dispose",y));const re=T.source;let de=d.get(re);de===void 0&&(de={},d.set(re,de));const ee=z(T);if(ee!==I.__cacheKey){de[ee]===void 0&&(de[ee]={texture:i.createTexture(),usedTimes:0},o.memory.textures++,X=!0),de[ee].usedTimes++;const Ue=de[I.__cacheKey];Ue!==void 0&&(de[I.__cacheKey].usedTimes--,Ue.usedTimes===0&&b(T)),I.__cacheKey=ee,I.__webglTexture=de[ee].texture}return X}function ke(I,T,X){return Math.floor(Math.floor(I/X)/T)}function We(I,T,X,re){const ee=I.updateRanges;if(ee.length===0)t.texSubImage2D(i.TEXTURE_2D,0,0,0,T.width,T.height,X,re,T.data);else{ee.sort((Z,xe)=>Z.start-xe.start);let Ue=0;for(let Z=1;Z<ee.length;Z++){const xe=ee[Ue],Re=ee[Z],Be=xe.start+xe.count,Fe=ke(Re.start,T.width,4),je=ke(xe.start,T.width,4);Re.start<=Be+1&&Fe===je&&ke(Re.start+Re.count-1,T.width,4)===Fe?xe.count=Math.max(xe.count,Re.start+Re.count-xe.start):(++Ue,ee[Ue]=Re)}ee.length=Ue+1;const ye=i.getParameter(i.UNPACK_ROW_LENGTH),Xe=i.getParameter(i.UNPACK_SKIP_PIXELS),k=i.getParameter(i.UNPACK_SKIP_ROWS);i.pixelStorei(i.UNPACK_ROW_LENGTH,T.width);for(let Z=0,xe=ee.length;Z<xe;Z++){const Re=ee[Z],Be=Math.floor(Re.start/4),Fe=Math.ceil(Re.count/4),je=Be%T.width,W=Math.floor(Be/T.width),Le=Fe,Me=1;i.pixelStorei(i.UNPACK_SKIP_PIXELS,je),i.pixelStorei(i.UNPACK_SKIP_ROWS,W),t.texSubImage2D(i.TEXTURE_2D,0,je,W,Le,Me,X,re,T.data)}I.clearUpdateRanges(),i.pixelStorei(i.UNPACK_ROW_LENGTH,ye),i.pixelStorei(i.UNPACK_SKIP_PIXELS,Xe),i.pixelStorei(i.UNPACK_SKIP_ROWS,k)}}function ne(I,T,X){let re=i.TEXTURE_2D;(T.isDataArrayTexture||T.isCompressedArrayTexture)&&(re=i.TEXTURE_2D_ARRAY),T.isData3DTexture&&(re=i.TEXTURE_3D);const de=ze(I,T),ee=T.source;t.bindTexture(re,I.__webglTexture,i.TEXTURE0+X);const Ue=n.get(ee);if(ee.version!==Ue.__version||de===!0){t.activeTexture(i.TEXTURE0+X);const ye=Et.getPrimaries(Et.workingColorSpace),Xe=T.colorSpace===js?null:Et.getPrimaries(T.colorSpace),k=T.colorSpace===js||ye===Xe?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,T.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,T.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,k);let Z=p(T.image,!1,s.maxTextureSize);Z=se(T,Z);const xe=r.convert(T.format,T.colorSpace),Re=r.convert(T.type);let Be=A(T.internalFormat,xe,Re,T.colorSpace,T.isVideoTexture);we(re,T);let Fe;const je=T.mipmaps,W=T.isVideoTexture!==!0,Le=Ue.__version===void 0||de===!0,Me=ee.dataReady,be=S(T,Z);if(T.isDepthTexture)Be=v(T.format===Za,T.type),Le&&(W?t.texStorage2D(i.TEXTURE_2D,1,Be,Z.width,Z.height):t.texImage2D(i.TEXTURE_2D,0,Be,Z.width,Z.height,0,xe,Re,null));else if(T.isDataTexture)if(je.length>0){W&&Le&&t.texStorage2D(i.TEXTURE_2D,be,Be,je[0].width,je[0].height);for(let Ae=0,ge=je.length;Ae<ge;Ae++)Fe=je[Ae],W?Me&&t.texSubImage2D(i.TEXTURE_2D,Ae,0,0,Fe.width,Fe.height,xe,Re,Fe.data):t.texImage2D(i.TEXTURE_2D,Ae,Be,Fe.width,Fe.height,0,xe,Re,Fe.data);T.generateMipmaps=!1}else W?(Le&&t.texStorage2D(i.TEXTURE_2D,be,Be,Z.width,Z.height),Me&&We(T,Z,xe,Re)):t.texImage2D(i.TEXTURE_2D,0,Be,Z.width,Z.height,0,xe,Re,Z.data);else if(T.isCompressedTexture)if(T.isCompressedArrayTexture){W&&Le&&t.texStorage3D(i.TEXTURE_2D_ARRAY,be,Be,je[0].width,je[0].height,Z.depth);for(let Ae=0,ge=je.length;Ae<ge;Ae++)if(Fe=je[Ae],T.format!==Xn)if(xe!==null)if(W){if(Me)if(T.layerUpdates.size>0){const qe=im(Fe.width,Fe.height,T.format,T.type);for(const Je of T.layerUpdates){const rt=Fe.data.subarray(Je*qe/Fe.data.BYTES_PER_ELEMENT,(Je+1)*qe/Fe.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,Ae,0,0,Je,Fe.width,Fe.height,1,xe,rt)}T.clearLayerUpdates()}else t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,Ae,0,0,0,Fe.width,Fe.height,Z.depth,xe,Fe.data)}else t.compressedTexImage3D(i.TEXTURE_2D_ARRAY,Ae,Be,Fe.width,Fe.height,Z.depth,0,Fe.data,0,0);else ft("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else W?Me&&t.texSubImage3D(i.TEXTURE_2D_ARRAY,Ae,0,0,0,Fe.width,Fe.height,Z.depth,xe,Re,Fe.data):t.texImage3D(i.TEXTURE_2D_ARRAY,Ae,Be,Fe.width,Fe.height,Z.depth,0,xe,Re,Fe.data)}else{W&&Le&&t.texStorage2D(i.TEXTURE_2D,be,Be,je[0].width,je[0].height);for(let Ae=0,ge=je.length;Ae<ge;Ae++)Fe=je[Ae],T.format!==Xn?xe!==null?W?Me&&t.compressedTexSubImage2D(i.TEXTURE_2D,Ae,0,0,Fe.width,Fe.height,xe,Fe.data):t.compressedTexImage2D(i.TEXTURE_2D,Ae,Be,Fe.width,Fe.height,0,Fe.data):ft("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):W?Me&&t.texSubImage2D(i.TEXTURE_2D,Ae,0,0,Fe.width,Fe.height,xe,Re,Fe.data):t.texImage2D(i.TEXTURE_2D,Ae,Be,Fe.width,Fe.height,0,xe,Re,Fe.data)}else if(T.isDataArrayTexture)if(W){if(Le&&t.texStorage3D(i.TEXTURE_2D_ARRAY,be,Be,Z.width,Z.height,Z.depth),Me)if(T.layerUpdates.size>0){const Ae=im(Z.width,Z.height,T.format,T.type);for(const ge of T.layerUpdates){const qe=Z.data.subarray(ge*Ae/Z.data.BYTES_PER_ELEMENT,(ge+1)*Ae/Z.data.BYTES_PER_ELEMENT);t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,ge,Z.width,Z.height,1,xe,Re,qe)}T.clearLayerUpdates()}else t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,Z.width,Z.height,Z.depth,xe,Re,Z.data)}else t.texImage3D(i.TEXTURE_2D_ARRAY,0,Be,Z.width,Z.height,Z.depth,0,xe,Re,Z.data);else if(T.isData3DTexture)W?(Le&&t.texStorage3D(i.TEXTURE_3D,be,Be,Z.width,Z.height,Z.depth),Me&&t.texSubImage3D(i.TEXTURE_3D,0,0,0,0,Z.width,Z.height,Z.depth,xe,Re,Z.data)):t.texImage3D(i.TEXTURE_3D,0,Be,Z.width,Z.height,Z.depth,0,xe,Re,Z.data);else if(T.isFramebufferTexture){if(Le)if(W)t.texStorage2D(i.TEXTURE_2D,be,Be,Z.width,Z.height);else{let Ae=Z.width,ge=Z.height;for(let qe=0;qe<be;qe++)t.texImage2D(i.TEXTURE_2D,qe,Be,Ae,ge,0,xe,Re,null),Ae>>=1,ge>>=1}}else if(je.length>0){if(W&&Le){const Ae=ve(je[0]);t.texStorage2D(i.TEXTURE_2D,be,Be,Ae.width,Ae.height)}for(let Ae=0,ge=je.length;Ae<ge;Ae++)Fe=je[Ae],W?Me&&t.texSubImage2D(i.TEXTURE_2D,Ae,0,0,xe,Re,Fe):t.texImage2D(i.TEXTURE_2D,Ae,Be,xe,Re,Fe);T.generateMipmaps=!1}else if(W){if(Le){const Ae=ve(Z);t.texStorage2D(i.TEXTURE_2D,be,Be,Ae.width,Ae.height)}Me&&t.texSubImage2D(i.TEXTURE_2D,0,0,0,xe,Re,Z)}else t.texImage2D(i.TEXTURE_2D,0,Be,xe,Re,Z);g(T)&&m(re),Ue.__version=ee.version,T.onUpdate&&T.onUpdate(T)}I.__version=T.version}function ue(I,T,X){if(T.image.length!==6)return;const re=ze(I,T),de=T.source;t.bindTexture(i.TEXTURE_CUBE_MAP,I.__webglTexture,i.TEXTURE0+X);const ee=n.get(de);if(de.version!==ee.__version||re===!0){t.activeTexture(i.TEXTURE0+X);const Ue=Et.getPrimaries(Et.workingColorSpace),ye=T.colorSpace===js?null:Et.getPrimaries(T.colorSpace),Xe=T.colorSpace===js||Ue===ye?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,T.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,T.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,Xe);const k=T.isCompressedTexture||T.image[0].isCompressedTexture,Z=T.image[0]&&T.image[0].isDataTexture,xe=[];for(let ge=0;ge<6;ge++)!k&&!Z?xe[ge]=p(T.image[ge],!0,s.maxCubemapSize):xe[ge]=Z?T.image[ge].image:T.image[ge],xe[ge]=se(T,xe[ge]);const Re=xe[0],Be=r.convert(T.format,T.colorSpace),Fe=r.convert(T.type),je=A(T.internalFormat,Be,Fe,T.colorSpace),W=T.isVideoTexture!==!0,Le=ee.__version===void 0||re===!0,Me=de.dataReady;let be=S(T,Re);we(i.TEXTURE_CUBE_MAP,T);let Ae;if(k){W&&Le&&t.texStorage2D(i.TEXTURE_CUBE_MAP,be,je,Re.width,Re.height);for(let ge=0;ge<6;ge++){Ae=xe[ge].mipmaps;for(let qe=0;qe<Ae.length;qe++){const Je=Ae[qe];T.format!==Xn?Be!==null?W?Me&&t.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,qe,0,0,Je.width,Je.height,Be,Je.data):t.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,qe,je,Je.width,Je.height,0,Je.data):ft("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):W?Me&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,qe,0,0,Je.width,Je.height,Be,Fe,Je.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,qe,je,Je.width,Je.height,0,Be,Fe,Je.data)}}}else{if(Ae=T.mipmaps,W&&Le){Ae.length>0&&be++;const ge=ve(xe[0]);t.texStorage2D(i.TEXTURE_CUBE_MAP,be,je,ge.width,ge.height)}for(let ge=0;ge<6;ge++)if(Z){W?Me&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,0,0,xe[ge].width,xe[ge].height,Be,Fe,xe[ge].data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,je,xe[ge].width,xe[ge].height,0,Be,Fe,xe[ge].data);for(let qe=0;qe<Ae.length;qe++){const rt=Ae[qe].image[ge].image;W?Me&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,qe+1,0,0,rt.width,rt.height,Be,Fe,rt.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,qe+1,je,rt.width,rt.height,0,Be,Fe,rt.data)}}else{W?Me&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,0,0,Be,Fe,xe[ge]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,je,Be,Fe,xe[ge]);for(let qe=0;qe<Ae.length;qe++){const Je=Ae[qe];W?Me&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,qe+1,0,0,Be,Fe,Je.image[ge]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,qe+1,je,Be,Fe,Je.image[ge])}}}g(T)&&m(i.TEXTURE_CUBE_MAP),ee.__version=de.version,T.onUpdate&&T.onUpdate(T)}I.__version=T.version}function Se(I,T,X,re,de,ee){const Ue=r.convert(X.format,X.colorSpace),ye=r.convert(X.type),Xe=A(X.internalFormat,Ue,ye,X.colorSpace),k=n.get(T),Z=n.get(X);if(Z.__renderTarget=T,!k.__hasExternalTextures){const xe=Math.max(1,T.width>>ee),Re=Math.max(1,T.height>>ee);de===i.TEXTURE_3D||de===i.TEXTURE_2D_ARRAY?t.texImage3D(de,ee,Xe,xe,Re,T.depth,0,Ue,ye,null):t.texImage2D(de,ee,Xe,xe,Re,0,Ue,ye,null)}t.bindFramebuffer(i.FRAMEBUFFER,I),ie(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,re,de,Z.__webglTexture,0,pe(T)):(de===i.TEXTURE_2D||de>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&de<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,re,de,Z.__webglTexture,ee),t.bindFramebuffer(i.FRAMEBUFFER,null)}function he(I,T,X){if(i.bindRenderbuffer(i.RENDERBUFFER,I),T.depthBuffer){const re=T.depthTexture,de=re&&re.isDepthTexture?re.type:null,ee=v(T.stencilBuffer,de),Ue=T.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,ye=pe(T);ie(T)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,ye,ee,T.width,T.height):X?i.renderbufferStorageMultisample(i.RENDERBUFFER,ye,ee,T.width,T.height):i.renderbufferStorage(i.RENDERBUFFER,ee,T.width,T.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,Ue,i.RENDERBUFFER,I)}else{const re=T.textures;for(let de=0;de<re.length;de++){const ee=re[de],Ue=r.convert(ee.format,ee.colorSpace),ye=r.convert(ee.type),Xe=A(ee.internalFormat,Ue,ye,ee.colorSpace),k=pe(T);X&&ie(T)===!1?i.renderbufferStorageMultisample(i.RENDERBUFFER,k,Xe,T.width,T.height):ie(T)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,k,Xe,T.width,T.height):i.renderbufferStorage(i.RENDERBUFFER,Xe,T.width,T.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function Ee(I,T){if(T&&T.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(t.bindFramebuffer(i.FRAMEBUFFER,I),!(T.depthTexture&&T.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const re=n.get(T.depthTexture);re.__renderTarget=T,(!re.__webglTexture||T.depthTexture.image.width!==T.width||T.depthTexture.image.height!==T.height)&&(T.depthTexture.image.width=T.width,T.depthTexture.image.height=T.height,T.depthTexture.needsUpdate=!0),V(T.depthTexture,0);const de=re.__webglTexture,ee=pe(T);if(T.depthTexture.format===Yo)ie(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,de,0,ee):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,de,0);else if(T.depthTexture.format===Za)ie(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,de,0,ee):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,de,0);else throw new Error("Unknown depthTexture format")}function Ze(I){const T=n.get(I),X=I.isWebGLCubeRenderTarget===!0;if(T.__boundDepthTexture!==I.depthTexture){const re=I.depthTexture;if(T.__depthDisposeCallback&&T.__depthDisposeCallback(),re){const de=()=>{delete T.__boundDepthTexture,delete T.__depthDisposeCallback,re.removeEventListener("dispose",de)};re.addEventListener("dispose",de),T.__depthDisposeCallback=de}T.__boundDepthTexture=re}if(I.depthTexture&&!T.__autoAllocateDepthBuffer){if(X)throw new Error("target.depthTexture not supported in Cube render targets");const re=I.texture.mipmaps;re&&re.length>0?Ee(T.__webglFramebuffer[0],I):Ee(T.__webglFramebuffer,I)}else if(X){T.__webglDepthbuffer=[];for(let re=0;re<6;re++)if(t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer[re]),T.__webglDepthbuffer[re]===void 0)T.__webglDepthbuffer[re]=i.createRenderbuffer(),he(T.__webglDepthbuffer[re],I,!1);else{const de=I.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,ee=T.__webglDepthbuffer[re];i.bindRenderbuffer(i.RENDERBUFFER,ee),i.framebufferRenderbuffer(i.FRAMEBUFFER,de,i.RENDERBUFFER,ee)}}else{const re=I.texture.mipmaps;if(re&&re.length>0?t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer[0]):t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer),T.__webglDepthbuffer===void 0)T.__webglDepthbuffer=i.createRenderbuffer(),he(T.__webglDepthbuffer,I,!1);else{const de=I.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,ee=T.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,ee),i.framebufferRenderbuffer(i.FRAMEBUFFER,de,i.RENDERBUFFER,ee)}}t.bindFramebuffer(i.FRAMEBUFFER,null)}function U(I,T,X){const re=n.get(I);T!==void 0&&Se(re.__webglFramebuffer,I,I.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),X!==void 0&&Ze(I)}function N(I){const T=I.texture,X=n.get(I),re=n.get(T);I.addEventListener("dispose",M);const de=I.textures,ee=I.isWebGLCubeRenderTarget===!0,Ue=de.length>1;if(Ue||(re.__webglTexture===void 0&&(re.__webglTexture=i.createTexture()),re.__version=T.version,o.memory.textures++),ee){X.__webglFramebuffer=[];for(let ye=0;ye<6;ye++)if(T.mipmaps&&T.mipmaps.length>0){X.__webglFramebuffer[ye]=[];for(let Xe=0;Xe<T.mipmaps.length;Xe++)X.__webglFramebuffer[ye][Xe]=i.createFramebuffer()}else X.__webglFramebuffer[ye]=i.createFramebuffer()}else{if(T.mipmaps&&T.mipmaps.length>0){X.__webglFramebuffer=[];for(let ye=0;ye<T.mipmaps.length;ye++)X.__webglFramebuffer[ye]=i.createFramebuffer()}else X.__webglFramebuffer=i.createFramebuffer();if(Ue)for(let ye=0,Xe=de.length;ye<Xe;ye++){const k=n.get(de[ye]);k.__webglTexture===void 0&&(k.__webglTexture=i.createTexture(),o.memory.textures++)}if(I.samples>0&&ie(I)===!1){X.__webglMultisampledFramebuffer=i.createFramebuffer(),X.__webglColorRenderbuffer=[],t.bindFramebuffer(i.FRAMEBUFFER,X.__webglMultisampledFramebuffer);for(let ye=0;ye<de.length;ye++){const Xe=de[ye];X.__webglColorRenderbuffer[ye]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,X.__webglColorRenderbuffer[ye]);const k=r.convert(Xe.format,Xe.colorSpace),Z=r.convert(Xe.type),xe=A(Xe.internalFormat,k,Z,Xe.colorSpace,I.isXRRenderTarget===!0),Re=pe(I);i.renderbufferStorageMultisample(i.RENDERBUFFER,Re,xe,I.width,I.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+ye,i.RENDERBUFFER,X.__webglColorRenderbuffer[ye])}i.bindRenderbuffer(i.RENDERBUFFER,null),I.depthBuffer&&(X.__webglDepthRenderbuffer=i.createRenderbuffer(),he(X.__webglDepthRenderbuffer,I,!0)),t.bindFramebuffer(i.FRAMEBUFFER,null)}}if(ee){t.bindTexture(i.TEXTURE_CUBE_MAP,re.__webglTexture),we(i.TEXTURE_CUBE_MAP,T);for(let ye=0;ye<6;ye++)if(T.mipmaps&&T.mipmaps.length>0)for(let Xe=0;Xe<T.mipmaps.length;Xe++)Se(X.__webglFramebuffer[ye][Xe],I,T,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+ye,Xe);else Se(X.__webglFramebuffer[ye],I,T,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+ye,0);g(T)&&m(i.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(Ue){for(let ye=0,Xe=de.length;ye<Xe;ye++){const k=de[ye],Z=n.get(k);let xe=i.TEXTURE_2D;(I.isWebGL3DRenderTarget||I.isWebGLArrayRenderTarget)&&(xe=I.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(xe,Z.__webglTexture),we(xe,k),Se(X.__webglFramebuffer,I,k,i.COLOR_ATTACHMENT0+ye,xe,0),g(k)&&m(xe)}t.unbindTexture()}else{let ye=i.TEXTURE_2D;if((I.isWebGL3DRenderTarget||I.isWebGLArrayRenderTarget)&&(ye=I.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(ye,re.__webglTexture),we(ye,T),T.mipmaps&&T.mipmaps.length>0)for(let Xe=0;Xe<T.mipmaps.length;Xe++)Se(X.__webglFramebuffer[Xe],I,T,i.COLOR_ATTACHMENT0,ye,Xe);else Se(X.__webglFramebuffer,I,T,i.COLOR_ATTACHMENT0,ye,0);g(T)&&m(ye),t.unbindTexture()}I.depthBuffer&&Ze(I)}function K(I){const T=I.textures;for(let X=0,re=T.length;X<re;X++){const de=T[X];if(g(de)){const ee=_(I),Ue=n.get(de).__webglTexture;t.bindTexture(ee,Ue),m(ee),t.unbindTexture()}}}const R=[],te=[];function oe(I){if(I.samples>0){if(ie(I)===!1){const T=I.textures,X=I.width,re=I.height;let de=i.COLOR_BUFFER_BIT;const ee=I.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Ue=n.get(I),ye=T.length>1;if(ye)for(let k=0;k<T.length;k++)t.bindFramebuffer(i.FRAMEBUFFER,Ue.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.RENDERBUFFER,null),t.bindFramebuffer(i.FRAMEBUFFER,Ue.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.TEXTURE_2D,null,0);t.bindFramebuffer(i.READ_FRAMEBUFFER,Ue.__webglMultisampledFramebuffer);const Xe=I.texture.mipmaps;Xe&&Xe.length>0?t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Ue.__webglFramebuffer[0]):t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Ue.__webglFramebuffer);for(let k=0;k<T.length;k++){if(I.resolveDepthBuffer&&(I.depthBuffer&&(de|=i.DEPTH_BUFFER_BIT),I.stencilBuffer&&I.resolveStencilBuffer&&(de|=i.STENCIL_BUFFER_BIT)),ye){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,Ue.__webglColorRenderbuffer[k]);const Z=n.get(T[k]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,Z,0)}i.blitFramebuffer(0,0,X,re,0,0,X,re,de,i.NEAREST),l===!0&&(R.length=0,te.length=0,R.push(i.COLOR_ATTACHMENT0+k),I.depthBuffer&&I.resolveDepthBuffer===!1&&(R.push(ee),te.push(ee),i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,te)),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,R))}if(t.bindFramebuffer(i.READ_FRAMEBUFFER,null),t.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),ye)for(let k=0;k<T.length;k++){t.bindFramebuffer(i.FRAMEBUFFER,Ue.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.RENDERBUFFER,Ue.__webglColorRenderbuffer[k]);const Z=n.get(T[k]).__webglTexture;t.bindFramebuffer(i.FRAMEBUFFER,Ue.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.TEXTURE_2D,Z,0)}t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Ue.__webglMultisampledFramebuffer)}else if(I.depthBuffer&&I.resolveDepthBuffer===!1&&l){const T=I.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[T])}}}function pe(I){return Math.min(s.maxSamples,I.samples)}function ie(I){const T=n.get(I);return I.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&T.__useRenderToTexture!==!1}function me(I){const T=o.render.frame;u.get(I)!==T&&(u.set(I,T),I.update())}function se(I,T){const X=I.colorSpace,re=I.format,de=I.type;return I.isCompressedTexture===!0||I.isVideoTexture===!0||X!==Qo&&X!==js&&(Et.getTransfer(X)===Ot?(re!==Xn||de!==as)&&ft("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):fn("WebGLTextures: Unsupported texture color space:",X)),T}function ve(I){return typeof HTMLImageElement<"u"&&I instanceof HTMLImageElement?(c.width=I.naturalWidth||I.width,c.height=I.naturalHeight||I.height):typeof VideoFrame<"u"&&I instanceof VideoFrame?(c.width=I.displayWidth,c.height=I.displayHeight):(c.width=I.width,c.height=I.height),c}this.allocateTextureUnit=O,this.resetTextureUnits=F,this.setTexture2D=V,this.setTexture2DArray=H,this.setTexture3D=q,this.setTextureCube=G,this.rebindTextures=U,this.setupRenderTarget=N,this.updateRenderTargetMipmap=K,this.updateMultisampleRenderTarget=oe,this.setupDepthRenderbuffer=Ze,this.setupFrameBufferTexture=Se,this.useMultisampledRTT=ie}function ix(i,e){function t(n,s=js){let r;const o=Et.getTransfer(s);if(n===as)return i.UNSIGNED_BYTE;if(n===Zd)return i.UNSIGNED_SHORT_4_4_4_4;if(n===Jd)return i.UNSIGNED_SHORT_5_5_5_1;if(n===Bg)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===Ug)return i.UNSIGNED_INT_10F_11F_11F_REV;if(n===Fg)return i.BYTE;if(n===Lg)return i.SHORT;if(n===ja)return i.UNSIGNED_SHORT;if(n===$d)return i.INT;if(n===Ii)return i.UNSIGNED_INT;if(n===Hi)return i.FLOAT;if(n===jr)return i.HALF_FLOAT;if(n===Og)return i.ALPHA;if(n===Ng)return i.RGB;if(n===Xn)return i.RGBA;if(n===Yo)return i.DEPTH_COMPONENT;if(n===Za)return i.DEPTH_STENCIL;if(n===zg)return i.RED;if(n===Kc)return i.RED_INTEGER;if(n===eh)return i.RG;if(n===th)return i.RG_INTEGER;if(n===Lo)return i.RGBA_INTEGER;if(n===lc||n===cc||n===uc||n===fc)if(o===Ot)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(n===lc)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===cc)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===uc)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===fc)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(n===lc)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===cc)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===uc)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===fc)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===Of||n===Nf||n===zf||n===kf)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(n===Of)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===Nf)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===zf)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===kf)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===Hf||n===Vf||n===Gf)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(n===Hf||n===Vf)return o===Ot?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(n===Gf)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(n===Wf||n===Xf||n===qf||n===Yf||n===Qf||n===Kf||n===jf||n===$f||n===Zf||n===Jf||n===ed||n===td||n===nd||n===id)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(n===Wf)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===Xf)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===qf)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===Yf)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===Qf)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===Kf)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===jf)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===$f)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===Zf)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===Jf)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===ed)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===td)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===nd)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===id)return o===Ot?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===sd||n===rd||n===od)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(n===sd)return o===Ot?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===rd)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===od)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===ad||n===ld||n===cd||n===ud)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(n===ad)return r.COMPRESSED_RED_RGTC1_EXT;if(n===ld)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===cd)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===ud)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===$a?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:t}}const d1=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,h1=`
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

}`;class p1{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){const n=new $g(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=n}}getMesh(e){if(this.texture!==null&&this.mesh===null){const t=e.cameras[0].viewport,n=new Yn({vertexShader:d1,fragmentShader:h1,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new hn(new jo(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class m1 extends $r{constructor(e,t){super();const n=this;let s=null,r=1,o=null,a="local-floor",l=1,c=null,u=null,f=null,d=null,h=null,x=null;const p=typeof XRWebGLBinding<"u",g=new p1,m={},_=t.getContextAttributes();let A=null,v=null;const S=[],y=[],M=new Ke;let E=null;const b=new Ti;b.viewport=new Jt;const C=new Ti;C.viewport=new Jt;const D=[b,C],F=new By;let O=null,z=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(ne){let ue=S[ne];return ue===void 0&&(ue=new Nu,S[ne]=ue),ue.getTargetRaySpace()},this.getControllerGrip=function(ne){let ue=S[ne];return ue===void 0&&(ue=new Nu,S[ne]=ue),ue.getGripSpace()},this.getHand=function(ne){let ue=S[ne];return ue===void 0&&(ue=new Nu,S[ne]=ue),ue.getHandSpace()};function V(ne){const ue=y.indexOf(ne.inputSource);if(ue===-1)return;const Se=S[ue];Se!==void 0&&(Se.update(ne.inputSource,ne.frame,c||o),Se.dispatchEvent({type:ne.type,data:ne.inputSource}))}function H(){s.removeEventListener("select",V),s.removeEventListener("selectstart",V),s.removeEventListener("selectend",V),s.removeEventListener("squeeze",V),s.removeEventListener("squeezestart",V),s.removeEventListener("squeezeend",V),s.removeEventListener("end",H),s.removeEventListener("inputsourceschange",q);for(let ne=0;ne<S.length;ne++){const ue=y[ne];ue!==null&&(y[ne]=null,S[ne].disconnect(ue))}O=null,z=null,g.reset();for(const ne in m)delete m[ne];e.setRenderTarget(A),h=null,d=null,f=null,s=null,v=null,We.stop(),n.isPresenting=!1,e.setPixelRatio(E),e.setSize(M.width,M.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(ne){r=ne,n.isPresenting===!0&&ft("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(ne){a=ne,n.isPresenting===!0&&ft("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return c||o},this.setReferenceSpace=function(ne){c=ne},this.getBaseLayer=function(){return d!==null?d:h},this.getBinding=function(){return f===null&&p&&(f=new XRWebGLBinding(s,t)),f},this.getFrame=function(){return x},this.getSession=function(){return s},this.setSession=async function(ne){if(s=ne,s!==null){if(A=e.getRenderTarget(),s.addEventListener("select",V),s.addEventListener("selectstart",V),s.addEventListener("selectend",V),s.addEventListener("squeeze",V),s.addEventListener("squeezestart",V),s.addEventListener("squeezeend",V),s.addEventListener("end",H),s.addEventListener("inputsourceschange",q),_.xrCompatible!==!0&&await t.makeXRCompatible(),E=e.getPixelRatio(),e.getSize(M),p&&"createProjectionLayer"in XRWebGLBinding.prototype){let Se=null,he=null,Ee=null;_.depth&&(Ee=_.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,Se=_.stencil?Za:Yo,he=_.stencil?$a:Ii);const Ze={colorFormat:t.RGBA8,depthFormat:Ee,scaleFactor:r};f=this.getBinding(),d=f.createProjectionLayer(Ze),s.updateRenderState({layers:[d]}),e.setPixelRatio(1),e.setSize(d.textureWidth,d.textureHeight,!1),v=new cr(d.textureWidth,d.textureHeight,{format:Xn,type:as,depthTexture:new rh(d.textureWidth,d.textureHeight,he,void 0,void 0,void 0,void 0,void 0,void 0,Se),stencilBuffer:_.stencil,colorSpace:e.outputColorSpace,samples:_.antialias?4:0,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1})}else{const Se={antialias:_.antialias,alpha:!0,depth:_.depth,stencil:_.stencil,framebufferScaleFactor:r};h=new XRWebGLLayer(s,t,Se),s.updateRenderState({baseLayer:h}),e.setPixelRatio(1),e.setSize(h.framebufferWidth,h.framebufferHeight,!1),v=new cr(h.framebufferWidth,h.framebufferHeight,{format:Xn,type:as,colorSpace:e.outputColorSpace,stencilBuffer:_.stencil,resolveDepthBuffer:h.ignoreDepthValues===!1,resolveStencilBuffer:h.ignoreDepthValues===!1})}v.isXRRenderTarget=!0,this.setFoveation(l),c=null,o=await s.requestReferenceSpace(a),We.setContext(s),We.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return g.getDepthTexture()};function q(ne){for(let ue=0;ue<ne.removed.length;ue++){const Se=ne.removed[ue],he=y.indexOf(Se);he>=0&&(y[he]=null,S[he].disconnect(Se))}for(let ue=0;ue<ne.added.length;ue++){const Se=ne.added[ue];let he=y.indexOf(Se);if(he===-1){for(let Ze=0;Ze<S.length;Ze++)if(Ze>=y.length){y.push(Se),he=Ze;break}else if(y[Ze]===null){y[Ze]=Se,he=Ze;break}if(he===-1)break}const Ee=S[he];Ee&&Ee.connect(Se)}}const G=new B,$=new B;function fe(ne,ue,Se){G.setFromMatrixPosition(ue.matrixWorld),$.setFromMatrixPosition(Se.matrixWorld);const he=G.distanceTo($),Ee=ue.projectionMatrix.elements,Ze=Se.projectionMatrix.elements,U=Ee[14]/(Ee[10]-1),N=Ee[14]/(Ee[10]+1),K=(Ee[9]+1)/Ee[5],R=(Ee[9]-1)/Ee[5],te=(Ee[8]-1)/Ee[0],oe=(Ze[8]+1)/Ze[0],pe=U*te,ie=U*oe,me=he/(-te+oe),se=me*-te;if(ue.matrixWorld.decompose(ne.position,ne.quaternion,ne.scale),ne.translateX(se),ne.translateZ(me),ne.matrixWorld.compose(ne.position,ne.quaternion,ne.scale),ne.matrixWorldInverse.copy(ne.matrixWorld).invert(),Ee[10]===-1)ne.projectionMatrix.copy(ue.projectionMatrix),ne.projectionMatrixInverse.copy(ue.projectionMatrixInverse);else{const ve=U+me,I=N+me,T=pe-se,X=ie+(he-se),re=K*N/I*ve,de=R*N/I*ve;ne.projectionMatrix.makePerspective(T,X,re,de,ve,I),ne.projectionMatrixInverse.copy(ne.projectionMatrix).invert()}}function Y(ne,ue){ue===null?ne.matrixWorld.copy(ne.matrix):ne.matrixWorld.multiplyMatrices(ue.matrixWorld,ne.matrix),ne.matrixWorldInverse.copy(ne.matrixWorld).invert()}this.updateCamera=function(ne){if(s===null)return;let ue=ne.near,Se=ne.far;g.texture!==null&&(g.depthNear>0&&(ue=g.depthNear),g.depthFar>0&&(Se=g.depthFar)),F.near=C.near=b.near=ue,F.far=C.far=b.far=Se,(O!==F.near||z!==F.far)&&(s.updateRenderState({depthNear:F.near,depthFar:F.far}),O=F.near,z=F.far),F.layers.mask=ne.layers.mask|6,b.layers.mask=F.layers.mask&3,C.layers.mask=F.layers.mask&5;const he=ne.parent,Ee=F.cameras;Y(F,he);for(let Ze=0;Ze<Ee.length;Ze++)Y(Ee[Ze],he);Ee.length===2?fe(F,b,C):F.projectionMatrix.copy(b.projectionMatrix),we(ne,F,he)};function we(ne,ue,Se){Se===null?ne.matrix.copy(ue.matrixWorld):(ne.matrix.copy(Se.matrixWorld),ne.matrix.invert(),ne.matrix.multiply(ue.matrixWorld)),ne.matrix.decompose(ne.position,ne.quaternion,ne.scale),ne.updateMatrixWorld(!0),ne.projectionMatrix.copy(ue.projectionMatrix),ne.projectionMatrixInverse.copy(ue.projectionMatrixInverse),ne.isPerspectiveCamera&&(ne.fov=el*2*Math.atan(1/ne.projectionMatrix.elements[5]),ne.zoom=1)}this.getCamera=function(){return F},this.getFoveation=function(){if(!(d===null&&h===null))return l},this.setFoveation=function(ne){l=ne,d!==null&&(d.fixedFoveation=ne),h!==null&&h.fixedFoveation!==void 0&&(h.fixedFoveation=ne)},this.hasDepthSensing=function(){return g.texture!==null},this.getDepthSensingMesh=function(){return g.getMesh(F)},this.getCameraTexture=function(ne){return m[ne]};let ze=null;function ke(ne,ue){if(u=ue.getViewerPose(c||o),x=ue,u!==null){const Se=u.views;h!==null&&(e.setRenderTargetFramebuffer(v,h.framebuffer),e.setRenderTarget(v));let he=!1;Se.length!==F.cameras.length&&(F.cameras.length=0,he=!0);for(let N=0;N<Se.length;N++){const K=Se[N];let R=null;if(h!==null)R=h.getViewport(K);else{const oe=f.getViewSubImage(d,K);R=oe.viewport,N===0&&(e.setRenderTargetTextures(v,oe.colorTexture,oe.depthStencilTexture),e.setRenderTarget(v))}let te=D[N];te===void 0&&(te=new Ti,te.layers.enable(N),te.viewport=new Jt,D[N]=te),te.matrix.fromArray(K.transform.matrix),te.matrix.decompose(te.position,te.quaternion,te.scale),te.projectionMatrix.fromArray(K.projectionMatrix),te.projectionMatrixInverse.copy(te.projectionMatrix).invert(),te.viewport.set(R.x,R.y,R.width,R.height),N===0&&(F.matrix.copy(te.matrix),F.matrix.decompose(F.position,F.quaternion,F.scale)),he===!0&&F.cameras.push(te)}const Ee=s.enabledFeatures;if(Ee&&Ee.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&p){f=n.getBinding();const N=f.getDepthInformation(Se[0]);N&&N.isValid&&N.texture&&g.init(N,s.renderState)}if(Ee&&Ee.includes("camera-access")&&p){e.state.unbindTexture(),f=n.getBinding();for(let N=0;N<Se.length;N++){const K=Se[N].camera;if(K){let R=m[K];R||(R=new $g,m[K]=R);const te=f.getCameraImage(K);R.sourceTexture=te}}}}for(let Se=0;Se<S.length;Se++){const he=y[Se],Ee=S[Se];he!==null&&Ee!==void 0&&Ee.update(he,ue,c||o)}ze&&ze(ne,ue),ue.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:ue}),x=null}const We=new Zg;We.setAnimationLoop(ke),this.setAnimationLoop=function(ne){ze=ne},this.dispose=function(){}}}const Rr=new Wi,g1=new st;function x1(i,e){function t(g,m){g.matrixAutoUpdate===!0&&g.updateMatrix(),m.value.copy(g.matrix)}function n(g,m){m.color.getRGB(g.fogColor.value,Yg(i)),m.isFog?(g.fogNear.value=m.near,g.fogFar.value=m.far):m.isFogExp2&&(g.fogDensity.value=m.density)}function s(g,m,_,A,v){m.isMeshBasicMaterial||m.isMeshLambertMaterial?r(g,m):m.isMeshToonMaterial?(r(g,m),f(g,m)):m.isMeshPhongMaterial?(r(g,m),u(g,m)):m.isMeshStandardMaterial?(r(g,m),d(g,m),m.isMeshPhysicalMaterial&&h(g,m,v)):m.isMeshMatcapMaterial?(r(g,m),x(g,m)):m.isMeshDepthMaterial?r(g,m):m.isMeshDistanceMaterial?(r(g,m),p(g,m)):m.isMeshNormalMaterial?r(g,m):m.isLineBasicMaterial?(o(g,m),m.isLineDashedMaterial&&a(g,m)):m.isPointsMaterial?l(g,m,_,A):m.isSpriteMaterial?c(g,m):m.isShadowMaterial?(g.color.value.copy(m.color),g.opacity.value=m.opacity):m.isShaderMaterial&&(m.uniformsNeedUpdate=!1)}function r(g,m){g.opacity.value=m.opacity,m.color&&g.diffuse.value.copy(m.color),m.emissive&&g.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity),m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.bumpMap&&(g.bumpMap.value=m.bumpMap,t(m.bumpMap,g.bumpMapTransform),g.bumpScale.value=m.bumpScale,m.side===Jn&&(g.bumpScale.value*=-1)),m.normalMap&&(g.normalMap.value=m.normalMap,t(m.normalMap,g.normalMapTransform),g.normalScale.value.copy(m.normalScale),m.side===Jn&&g.normalScale.value.negate()),m.displacementMap&&(g.displacementMap.value=m.displacementMap,t(m.displacementMap,g.displacementMapTransform),g.displacementScale.value=m.displacementScale,g.displacementBias.value=m.displacementBias),m.emissiveMap&&(g.emissiveMap.value=m.emissiveMap,t(m.emissiveMap,g.emissiveMapTransform)),m.specularMap&&(g.specularMap.value=m.specularMap,t(m.specularMap,g.specularMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest);const _=e.get(m),A=_.envMap,v=_.envMapRotation;A&&(g.envMap.value=A,Rr.copy(v),Rr.x*=-1,Rr.y*=-1,Rr.z*=-1,A.isCubeTexture&&A.isRenderTargetTexture===!1&&(Rr.y*=-1,Rr.z*=-1),g.envMapRotation.value.setFromMatrix4(g1.makeRotationFromEuler(Rr)),g.flipEnvMap.value=A.isCubeTexture&&A.isRenderTargetTexture===!1?-1:1,g.reflectivity.value=m.reflectivity,g.ior.value=m.ior,g.refractionRatio.value=m.refractionRatio),m.lightMap&&(g.lightMap.value=m.lightMap,g.lightMapIntensity.value=m.lightMapIntensity,t(m.lightMap,g.lightMapTransform)),m.aoMap&&(g.aoMap.value=m.aoMap,g.aoMapIntensity.value=m.aoMapIntensity,t(m.aoMap,g.aoMapTransform))}function o(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform))}function a(g,m){g.dashSize.value=m.dashSize,g.totalSize.value=m.dashSize+m.gapSize,g.scale.value=m.scale}function l(g,m,_,A){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.size.value=m.size*_,g.scale.value=A*.5,m.map&&(g.map.value=m.map,t(m.map,g.uvTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function c(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.rotation.value=m.rotation,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function u(g,m){g.specular.value.copy(m.specular),g.shininess.value=Math.max(m.shininess,1e-4)}function f(g,m){m.gradientMap&&(g.gradientMap.value=m.gradientMap)}function d(g,m){g.metalness.value=m.metalness,m.metalnessMap&&(g.metalnessMap.value=m.metalnessMap,t(m.metalnessMap,g.metalnessMapTransform)),g.roughness.value=m.roughness,m.roughnessMap&&(g.roughnessMap.value=m.roughnessMap,t(m.roughnessMap,g.roughnessMapTransform)),m.envMap&&(g.envMapIntensity.value=m.envMapIntensity)}function h(g,m,_){g.ior.value=m.ior,m.sheen>0&&(g.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen),g.sheenRoughness.value=m.sheenRoughness,m.sheenColorMap&&(g.sheenColorMap.value=m.sheenColorMap,t(m.sheenColorMap,g.sheenColorMapTransform)),m.sheenRoughnessMap&&(g.sheenRoughnessMap.value=m.sheenRoughnessMap,t(m.sheenRoughnessMap,g.sheenRoughnessMapTransform))),m.clearcoat>0&&(g.clearcoat.value=m.clearcoat,g.clearcoatRoughness.value=m.clearcoatRoughness,m.clearcoatMap&&(g.clearcoatMap.value=m.clearcoatMap,t(m.clearcoatMap,g.clearcoatMapTransform)),m.clearcoatRoughnessMap&&(g.clearcoatRoughnessMap.value=m.clearcoatRoughnessMap,t(m.clearcoatRoughnessMap,g.clearcoatRoughnessMapTransform)),m.clearcoatNormalMap&&(g.clearcoatNormalMap.value=m.clearcoatNormalMap,t(m.clearcoatNormalMap,g.clearcoatNormalMapTransform),g.clearcoatNormalScale.value.copy(m.clearcoatNormalScale),m.side===Jn&&g.clearcoatNormalScale.value.negate())),m.dispersion>0&&(g.dispersion.value=m.dispersion),m.iridescence>0&&(g.iridescence.value=m.iridescence,g.iridescenceIOR.value=m.iridescenceIOR,g.iridescenceThicknessMinimum.value=m.iridescenceThicknessRange[0],g.iridescenceThicknessMaximum.value=m.iridescenceThicknessRange[1],m.iridescenceMap&&(g.iridescenceMap.value=m.iridescenceMap,t(m.iridescenceMap,g.iridescenceMapTransform)),m.iridescenceThicknessMap&&(g.iridescenceThicknessMap.value=m.iridescenceThicknessMap,t(m.iridescenceThicknessMap,g.iridescenceThicknessMapTransform))),m.transmission>0&&(g.transmission.value=m.transmission,g.transmissionSamplerMap.value=_.texture,g.transmissionSamplerSize.value.set(_.width,_.height),m.transmissionMap&&(g.transmissionMap.value=m.transmissionMap,t(m.transmissionMap,g.transmissionMapTransform)),g.thickness.value=m.thickness,m.thicknessMap&&(g.thicknessMap.value=m.thicknessMap,t(m.thicknessMap,g.thicknessMapTransform)),g.attenuationDistance.value=m.attenuationDistance,g.attenuationColor.value.copy(m.attenuationColor)),m.anisotropy>0&&(g.anisotropyVector.value.set(m.anisotropy*Math.cos(m.anisotropyRotation),m.anisotropy*Math.sin(m.anisotropyRotation)),m.anisotropyMap&&(g.anisotropyMap.value=m.anisotropyMap,t(m.anisotropyMap,g.anisotropyMapTransform))),g.specularIntensity.value=m.specularIntensity,g.specularColor.value.copy(m.specularColor),m.specularColorMap&&(g.specularColorMap.value=m.specularColorMap,t(m.specularColorMap,g.specularColorMapTransform)),m.specularIntensityMap&&(g.specularIntensityMap.value=m.specularIntensityMap,t(m.specularIntensityMap,g.specularIntensityMapTransform))}function x(g,m){m.matcap&&(g.matcap.value=m.matcap)}function p(g,m){const _=e.get(m).light;g.referencePosition.value.setFromMatrixPosition(_.matrixWorld),g.nearDistance.value=_.shadow.camera.near,g.farDistance.value=_.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:s}}function _1(i,e,t,n){let s={},r={},o=[];const a=i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);function l(_,A){const v=A.program;n.uniformBlockBinding(_,v)}function c(_,A){let v=s[_.id];v===void 0&&(x(_),v=u(_),s[_.id]=v,_.addEventListener("dispose",g));const S=A.program;n.updateUBOMapping(_,S);const y=e.render.frame;r[_.id]!==y&&(d(_),r[_.id]=y)}function u(_){const A=f();_.__bindingPointIndex=A;const v=i.createBuffer(),S=_.__size,y=_.usage;return i.bindBuffer(i.UNIFORM_BUFFER,v),i.bufferData(i.UNIFORM_BUFFER,S,y),i.bindBuffer(i.UNIFORM_BUFFER,null),i.bindBufferBase(i.UNIFORM_BUFFER,A,v),v}function f(){for(let _=0;_<a;_++)if(o.indexOf(_)===-1)return o.push(_),_;return fn("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function d(_){const A=s[_.id],v=_.uniforms,S=_.__cache;i.bindBuffer(i.UNIFORM_BUFFER,A);for(let y=0,M=v.length;y<M;y++){const E=Array.isArray(v[y])?v[y]:[v[y]];for(let b=0,C=E.length;b<C;b++){const D=E[b];if(h(D,y,b,S)===!0){const F=D.__offset,O=Array.isArray(D.value)?D.value:[D.value];let z=0;for(let V=0;V<O.length;V++){const H=O[V],q=p(H);typeof H=="number"||typeof H=="boolean"?(D.__data[0]=H,i.bufferSubData(i.UNIFORM_BUFFER,F+z,D.__data)):H.isMatrix3?(D.__data[0]=H.elements[0],D.__data[1]=H.elements[1],D.__data[2]=H.elements[2],D.__data[3]=0,D.__data[4]=H.elements[3],D.__data[5]=H.elements[4],D.__data[6]=H.elements[5],D.__data[7]=0,D.__data[8]=H.elements[6],D.__data[9]=H.elements[7],D.__data[10]=H.elements[8],D.__data[11]=0):(H.toArray(D.__data,z),z+=q.storage/Float32Array.BYTES_PER_ELEMENT)}i.bufferSubData(i.UNIFORM_BUFFER,F,D.__data)}}}i.bindBuffer(i.UNIFORM_BUFFER,null)}function h(_,A,v,S){const y=_.value,M=A+"_"+v;if(S[M]===void 0)return typeof y=="number"||typeof y=="boolean"?S[M]=y:S[M]=y.clone(),!0;{const E=S[M];if(typeof y=="number"||typeof y=="boolean"){if(E!==y)return S[M]=y,!0}else if(E.equals(y)===!1)return E.copy(y),!0}return!1}function x(_){const A=_.uniforms;let v=0;const S=16;for(let M=0,E=A.length;M<E;M++){const b=Array.isArray(A[M])?A[M]:[A[M]];for(let C=0,D=b.length;C<D;C++){const F=b[C],O=Array.isArray(F.value)?F.value:[F.value];for(let z=0,V=O.length;z<V;z++){const H=O[z],q=p(H),G=v%S,$=G%q.boundary,fe=G+$;v+=$,fe!==0&&S-fe<q.storage&&(v+=S-fe),F.__data=new Float32Array(q.storage/Float32Array.BYTES_PER_ELEMENT),F.__offset=v,v+=q.storage}}}const y=v%S;return y>0&&(v+=S-y),_.__size=v,_.__cache={},this}function p(_){const A={boundary:0,storage:0};return typeof _=="number"||typeof _=="boolean"?(A.boundary=4,A.storage=4):_.isVector2?(A.boundary=8,A.storage=8):_.isVector3||_.isColor?(A.boundary=16,A.storage=12):_.isVector4?(A.boundary=16,A.storage=16):_.isMatrix3?(A.boundary=48,A.storage=48):_.isMatrix4?(A.boundary=64,A.storage=64):_.isTexture?ft("WebGLRenderer: Texture samplers can not be part of an uniforms group."):ft("WebGLRenderer: Unsupported uniform value type.",_),A}function g(_){const A=_.target;A.removeEventListener("dispose",g);const v=o.indexOf(A.__bindingPointIndex);o.splice(v,1),i.deleteBuffer(s[A.id]),delete s[A.id],delete r[A.id]}function m(){for(const _ in s)i.deleteBuffer(s[_]);o=[],s={},r={}}return{bind:l,update:c,dispose:m}}const v1=new Uint16Array([11481,15204,11534,15171,11808,15015,12385,14843,12894,14716,13396,14600,13693,14483,13976,14366,14237,14171,14405,13961,14511,13770,14605,13598,14687,13444,14760,13305,14822,13066,14876,12857,14923,12675,14963,12517,14997,12379,15025,12230,15049,12023,15070,11843,15086,11687,15100,11551,15111,11433,15120,11330,15127,11217,15132,11060,15135,10922,15138,10801,15139,10695,15139,10600,13012,14923,13020,14917,13064,14886,13176,14800,13349,14666,13513,14526,13724,14398,13960,14230,14200,14020,14383,13827,14488,13651,14583,13491,14667,13348,14740,13132,14803,12908,14856,12713,14901,12542,14938,12394,14968,12241,14992,12017,15010,11822,15024,11654,15034,11507,15041,11380,15044,11269,15044,11081,15042,10913,15037,10764,15031,10635,15023,10520,15014,10419,15003,10330,13657,14676,13658,14673,13670,14660,13698,14622,13750,14547,13834,14442,13956,14317,14112,14093,14291,13889,14407,13704,14499,13538,14586,13389,14664,13201,14733,12966,14792,12758,14842,12577,14882,12418,14915,12272,14940,12033,14959,11826,14972,11646,14980,11490,14983,11355,14983,11212,14979,11008,14971,10830,14961,10675,14950,10540,14936,10420,14923,10315,14909,10204,14894,10041,14089,14460,14090,14459,14096,14452,14112,14431,14141,14388,14186,14305,14252,14130,14341,13941,14399,13756,14467,13585,14539,13430,14610,13272,14677,13026,14737,12808,14790,12617,14833,12449,14869,12303,14896,12065,14916,11845,14929,11655,14937,11490,14939,11347,14936,11184,14930,10970,14921,10783,14912,10621,14900,10480,14885,10356,14867,10247,14848,10062,14827,9894,14805,9745,14400,14208,14400,14206,14402,14198,14406,14174,14415,14122,14427,14035,14444,13913,14469,13767,14504,13613,14548,13463,14598,13324,14651,13082,14704,12858,14752,12658,14795,12483,14831,12330,14860,12106,14881,11875,14895,11675,14903,11501,14905,11351,14903,11178,14900,10953,14892,10757,14880,10589,14865,10442,14847,10313,14827,10162,14805,9965,14782,9792,14757,9642,14731,9507,14562,13883,14562,13883,14563,13877,14566,13862,14570,13830,14576,13773,14584,13689,14595,13582,14613,13461,14637,13336,14668,13120,14704,12897,14741,12695,14776,12516,14808,12358,14835,12150,14856,11910,14870,11701,14878,11519,14882,11361,14884,11187,14880,10951,14871,10748,14858,10572,14842,10418,14823,10286,14801,10099,14777,9897,14751,9722,14725,9567,14696,9430,14666,9309,14702,13604,14702,13604,14702,13600,14703,13591,14705,13570,14707,13533,14709,13477,14712,13400,14718,13305,14727,13106,14743,12907,14762,12716,14784,12539,14807,12380,14827,12190,14844,11943,14855,11727,14863,11539,14870,11376,14871,11204,14868,10960,14858,10748,14845,10565,14829,10406,14809,10269,14786,10058,14761,9852,14734,9671,14705,9512,14674,9374,14641,9253,14608,9076,14821,13366,14821,13365,14821,13364,14821,13358,14821,13344,14821,13320,14819,13252,14817,13145,14815,13011,14814,12858,14817,12698,14823,12539,14832,12389,14841,12214,14850,11968,14856,11750,14861,11558,14866,11390,14867,11226,14862,10972,14853,10754,14840,10565,14823,10401,14803,10259,14780,10032,14754,9820,14725,9635,14694,9473,14661,9333,14627,9203,14593,8988,14557,8798,14923,13014,14922,13014,14922,13012,14922,13004,14920,12987,14919,12957,14915,12907,14909,12834,14902,12738,14894,12623,14888,12498,14883,12370,14880,12203,14878,11970,14875,11759,14873,11569,14874,11401,14872,11243,14865,10986,14855,10762,14842,10568,14825,10401,14804,10255,14781,10017,14754,9799,14725,9611,14692,9445,14658,9301,14623,9139,14587,8920,14548,8729,14509,8562,15008,12672,15008,12672,15008,12671,15007,12667,15005,12656,15001,12637,14997,12605,14989,12556,14978,12490,14966,12407,14953,12313,14940,12136,14927,11934,14914,11742,14903,11563,14896,11401,14889,11247,14879,10992,14866,10767,14851,10570,14833,10400,14812,10252,14789,10007,14761,9784,14731,9592,14698,9424,14663,9279,14627,9088,14588,8868,14548,8676,14508,8508,14467,8360,15080,12386,15080,12386,15079,12385,15078,12383,15076,12378,15072,12367,15066,12347,15057,12315,15045,12253,15030,12138,15012,11998,14993,11845,14972,11685,14951,11530,14935,11383,14920,11228,14904,10981,14887,10762,14870,10567,14850,10397,14827,10248,14803,9997,14774,9771,14743,9578,14710,9407,14674,9259,14637,9048,14596,8826,14555,8632,14514,8464,14471,8317,14427,8182,15139,12008,15139,12008,15138,12008,15137,12007,15135,12003,15130,11990,15124,11969,15115,11929,15102,11872,15086,11794,15064,11693,15041,11581,15013,11459,14987,11336,14966,11170,14944,10944,14921,10738,14898,10552,14875,10387,14850,10239,14824,9983,14794,9758,14762,9563,14728,9392,14692,9244,14653,9014,14611,8791,14569,8597,14526,8427,14481,8281,14436,8110,14391,7885,15188,11617,15188,11617,15187,11617,15186,11618,15183,11617,15179,11612,15173,11601,15163,11581,15150,11546,15133,11495,15110,11427,15083,11346,15051,11246,15024,11057,14996,10868,14967,10687,14938,10517,14911,10362,14882,10206,14853,9956,14821,9737,14787,9543,14752,9375,14715,9228,14675,8980,14632,8760,14589,8565,14544,8395,14498,8248,14451,8049,14404,7824,14357,7630,15228,11298,15228,11298,15227,11299,15226,11301,15223,11303,15219,11302,15213,11299,15204,11290,15191,11271,15174,11217,15150,11129,15119,11015,15087,10886,15057,10744,15024,10599,14990,10455,14957,10318,14924,10143,14891,9911,14856,9701,14820,9516,14782,9352,14744,9200,14703,8946,14659,8725,14615,8533,14568,8366,14521,8220,14472,7992,14423,7770,14374,7578,14315,7408,15260,10819,15260,10819,15259,10822,15258,10826,15256,10832,15251,10836,15246,10841,15237,10838,15225,10821,15207,10788,15183,10734,15151,10660,15120,10571,15087,10469,15049,10359,15012,10249,14974,10041,14937,9837,14900,9647,14860,9475,14820,9320,14779,9147,14736,8902,14691,8688,14646,8499,14598,8335,14549,8189,14499,7940,14448,7720,14397,7529,14347,7363,14256,7218,15285,10410,15285,10411,15285,10413,15284,10418,15282,10425,15278,10434,15272,10442,15264,10449,15252,10445,15235,10433,15210,10403,15179,10358,15149,10301,15113,10218,15073,10059,15033,9894,14991,9726,14951,9565,14909,9413,14865,9273,14822,9073,14777,8845,14730,8641,14682,8459,14633,8300,14583,8129,14531,7883,14479,7670,14426,7482,14373,7321,14305,7176,14201,6939,15305,9939,15305,9940,15305,9945,15304,9955,15302,9967,15298,9989,15293,10010,15286,10033,15274,10044,15258,10045,15233,10022,15205,9975,15174,9903,15136,9808,15095,9697,15053,9578,15009,9451,14965,9327,14918,9198,14871,8973,14825,8766,14775,8579,14725,8408,14675,8259,14622,8058,14569,7821,14515,7615,14460,7435,14405,7276,14350,7108,14256,6866,14149,6653,15321,9444,15321,9445,15321,9448,15320,9458,15317,9470,15314,9490,15310,9515,15302,9540,15292,9562,15276,9579,15251,9577,15226,9559,15195,9519,15156,9463,15116,9389,15071,9304,15025,9208,14978,9023,14927,8838,14878,8661,14827,8496,14774,8344,14722,8206,14667,7973,14612,7749,14556,7555,14499,7382,14443,7229,14385,7025,14322,6791,14210,6588,14100,6409,15333,8920,15333,8921,15332,8927,15332,8943,15329,8965,15326,9002,15322,9048,15316,9106,15307,9162,15291,9204,15267,9221,15244,9221,15212,9196,15175,9134,15133,9043,15088,8930,15040,8801,14990,8665,14938,8526,14886,8391,14830,8261,14775,8087,14719,7866,14661,7664,14603,7482,14544,7322,14485,7178,14426,6936,14367,6713,14281,6517,14166,6348,14054,6198,15341,8360,15341,8361,15341,8366,15341,8379,15339,8399,15336,8431,15332,8473,15326,8527,15318,8585,15302,8632,15281,8670,15258,8690,15227,8690,15191,8664,15149,8612,15104,8543,15055,8456,15001,8360,14948,8259,14892,8122,14834,7923,14776,7734,14716,7558,14656,7397,14595,7250,14534,7070,14472,6835,14410,6628,14350,6443,14243,6283,14125,6135,14010,5889,15348,7715,15348,7717,15348,7725,15347,7745,15345,7780,15343,7836,15339,7905,15334,8e3,15326,8103,15310,8193,15293,8239,15270,8270,15240,8287,15204,8283,15163,8260,15118,8223,15067,8143,15014,8014,14958,7873,14899,7723,14839,7573,14778,7430,14715,7293,14652,7164,14588,6931,14524,6720,14460,6531,14396,6362,14330,6210,14207,6015,14086,5781,13969,5576,15352,7114,15352,7116,15352,7128,15352,7159,15350,7195,15348,7237,15345,7299,15340,7374,15332,7457,15317,7544,15301,7633,15280,7703,15251,7754,15216,7775,15176,7767,15131,7733,15079,7670,15026,7588,14967,7492,14906,7387,14844,7278,14779,7171,14714,6965,14648,6770,14581,6587,14515,6420,14448,6269,14382,6123,14299,5881,14172,5665,14049,5477,13929,5310,15355,6329,15355,6330,15355,6339,15355,6362,15353,6410,15351,6472,15349,6572,15344,6688,15337,6835,15323,6985,15309,7142,15287,7220,15260,7277,15226,7310,15188,7326,15142,7318,15090,7285,15036,7239,14976,7177,14914,7045,14849,6892,14782,6736,14714,6581,14645,6433,14576,6293,14506,6164,14438,5946,14369,5733,14270,5540,14140,5369,14014,5216,13892,5043,15357,5483,15357,5484,15357,5496,15357,5528,15356,5597,15354,5692,15351,5835,15347,6011,15339,6195,15328,6317,15314,6446,15293,6566,15268,6668,15235,6746,15197,6796,15152,6811,15101,6790,15046,6748,14985,6673,14921,6583,14854,6479,14785,6371,14714,6259,14643,6149,14571,5946,14499,5750,14428,5567,14358,5401,14242,5250,14109,5111,13980,4870,13856,4657,15359,4555,15359,4557,15358,4573,15358,4633,15357,4715,15355,4841,15353,5061,15349,5216,15342,5391,15331,5577,15318,5770,15299,5967,15274,6150,15243,6223,15206,6280,15161,6310,15111,6317,15055,6300,14994,6262,14928,6208,14860,6141,14788,5994,14715,5838,14641,5684,14566,5529,14492,5384,14418,5247,14346,5121,14216,4892,14079,4682,13948,4496,13822,4330,15359,3498,15359,3501,15359,3520,15359,3598,15358,3719,15356,3860,15355,4137,15351,4305,15344,4563,15334,4809,15321,5116,15303,5273,15280,5418,15250,5547,15214,5653,15170,5722,15120,5761,15064,5763,15002,5733,14935,5673,14865,5597,14792,5504,14716,5400,14640,5294,14563,5185,14486,5041,14410,4841,14335,4655,14191,4482,14051,4325,13918,4183,13790,4012,15360,2282,15360,2285,15360,2306,15360,2401,15359,2547,15357,2748,15355,3103,15352,3349,15345,3675,15336,4020,15324,4272,15307,4496,15285,4716,15255,4908,15220,5086,15178,5170,15128,5214,15072,5234,15010,5231,14943,5206,14871,5166,14796,5102,14718,4971,14639,4833,14559,4687,14480,4541,14402,4401,14315,4268,14167,4142,14025,3958,13888,3747,13759,3556,15360,923,15360,925,15360,946,15360,1052,15359,1214,15357,1494,15356,1892,15352,2274,15346,2663,15338,3099,15326,3393,15309,3679,15288,3980,15260,4183,15226,4325,15185,4437,15136,4517,15080,4570,15018,4591,14950,4581,14877,4545,14800,4485,14720,4411,14638,4325,14556,4231,14475,4136,14395,3988,14297,3803,14145,3628,13999,3465,13861,3314,13729,3177,15360,263,15360,264,15360,272,15360,325,15359,407,15358,548,15356,780,15352,1144,15347,1580,15339,2099,15328,2425,15312,2795,15292,3133,15264,3329,15232,3517,15191,3689,15143,3819,15088,3923,15025,3978,14956,3999,14882,3979,14804,3931,14722,3855,14639,3756,14554,3645,14470,3529,14388,3409,14279,3289,14124,3173,13975,3055,13834,2848,13701,2658,15360,49,15360,49,15360,52,15360,75,15359,111,15358,201,15356,283,15353,519,15348,726,15340,1045,15329,1415,15314,1795,15295,2173,15269,2410,15237,2649,15197,2866,15150,3054,15095,3140,15032,3196,14963,3228,14888,3236,14808,3224,14725,3191,14639,3146,14553,3088,14466,2976,14382,2836,14262,2692,14103,2549,13952,2409,13808,2278,13674,2154,15360,4,15360,4,15360,4,15360,13,15359,33,15358,59,15357,112,15353,199,15348,302,15341,456,15331,628,15316,827,15297,1082,15272,1332,15241,1601,15202,1851,15156,2069,15101,2172,15039,2256,14970,2314,14894,2348,14813,2358,14728,2344,14640,2311,14551,2263,14463,2203,14376,2133,14247,2059,14084,1915,13930,1761,13784,1609,13648,1464,15360,0,15360,0,15360,0,15360,3,15359,18,15358,26,15357,53,15354,80,15348,97,15341,165,15332,238,15318,326,15299,427,15275,529,15245,654,15207,771,15161,885,15108,994,15046,1089,14976,1170,14900,1229,14817,1266,14731,1284,14641,1282,14550,1260,14460,1223,14370,1174,14232,1116,14066,1050,13909,981,13761,910,13623,839]);let vs=null;function A1(){return vs===null&&(vs=new ys(v1,32,32,eh,jr),vs.minFilter=Ri,vs.magFilter=Ri,vs.wrapS=Is,vs.wrapT=Is,vs.generateMipmaps=!1,vs.needsUpdate=!0),vs}class S1{constructor(e={}){const{canvas:t=FS(),context:n=null,depth:s=!0,stencil:r=!1,alpha:o=!1,antialias:a=!1,premultipliedAlpha:l=!0,preserveDrawingBuffer:c=!1,powerPreference:u="default",failIfMajorPerformanceCaveat:f=!1,reversedDepthBuffer:d=!1}=e;this.isWebGLRenderer=!0;let h;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");h=n.getContextAttributes().alpha}else h=o;const x=new Set([Lo,th,Kc]),p=new Set([as,Ii,ja,$a,Zd,Jd]),g=new Uint32Array(4),m=new Int32Array(4);let _=null,A=null;const v=[],S=[];this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=sr,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const y=this;let M=!1;this._outputColorSpace=Ci;let E=0,b=0,C=null,D=-1,F=null;const O=new Jt,z=new Jt;let V=null;const H=new bt(0);let q=0,G=t.width,$=t.height,fe=1,Y=null,we=null;const ze=new Jt(0,0,G,$),ke=new Jt(0,0,G,$);let We=!1;const ne=new jg;let ue=!1,Se=!1;const he=new st,Ee=new B,Ze=new Jt,U={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let N=!1;function K(){return C===null?fe:1}let R=n;function te(P,Q){return t.getContext(P,Q)}try{const P={alpha:!0,depth:s,stencil:r,antialias:a,premultipliedAlpha:l,preserveDrawingBuffer:c,powerPreference:u,failIfMajorPerformanceCaveat:f};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${jd}`),t.addEventListener("webglcontextlost",Ae,!1),t.addEventListener("webglcontextrestored",ge,!1),t.addEventListener("webglcontextcreationerror",qe,!1),R===null){const Q="webgl2";if(R=te(Q,P),R===null)throw te(Q)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(P){throw P("WebGLRenderer: "+P.message),P}let oe,pe,ie,me,se,ve,I,T,X,re,de,ee,Ue,ye,Xe,k,Z,xe,Re,Be,Fe,je,W,Le;function Me(){oe=new IC(R),oe.init(),je=new ix(R,oe),pe=new SC(R,oe,e,je),ie=new u1(R,oe),pe.reversedDepthBuffer&&d&&ie.buffers.depth.setReversed(!0),me=new FC(R),se=new $T,ve=new f1(R,oe,ie,se,pe,je,me),I=new bC(y),T=new RC(y),X=new Oy(R),W=new vC(R,X),re=new DC(R,X,me,W),de=new BC(R,re,X,me),Re=new LC(R,pe,ve),k=new yC(se),ee=new jT(y,I,T,oe,pe,W,k),Ue=new x1(y,se),ye=new JT,Xe=new r1(oe),xe=new _C(y,I,T,ie,de,h,l),Z=new l1(y,de,pe),Le=new _1(R,me,pe,ie),Be=new AC(R,oe,me),Fe=new PC(R,oe,me),me.programs=ee.programs,y.capabilities=pe,y.extensions=oe,y.properties=se,y.renderLists=ye,y.shadowMap=Z,y.state=ie,y.info=me}Me();const be=new m1(y,R);this.xr=be,this.getContext=function(){return R},this.getContextAttributes=function(){return R.getContextAttributes()},this.forceContextLoss=function(){const P=oe.get("WEBGL_lose_context");P&&P.loseContext()},this.forceContextRestore=function(){const P=oe.get("WEBGL_lose_context");P&&P.restoreContext()},this.getPixelRatio=function(){return fe},this.setPixelRatio=function(P){P!==void 0&&(fe=P,this.setSize(G,$,!1))},this.getSize=function(P){return P.set(G,$)},this.setSize=function(P,Q,ae=!0){if(be.isPresenting){ft("WebGLRenderer: Can't change size while VR device is presenting.");return}G=P,$=Q,t.width=Math.floor(P*fe),t.height=Math.floor(Q*fe),ae===!0&&(t.style.width=P+"px",t.style.height=Q+"px"),this.setViewport(0,0,P,Q)},this.getDrawingBufferSize=function(P){return P.set(G*fe,$*fe).floor()},this.setDrawingBufferSize=function(P,Q,ae){G=P,$=Q,fe=ae,t.width=Math.floor(P*ae),t.height=Math.floor(Q*ae),this.setViewport(0,0,P,Q)},this.getCurrentViewport=function(P){return P.copy(O)},this.getViewport=function(P){return P.copy(ze)},this.setViewport=function(P,Q,ae,ce){P.isVector4?ze.set(P.x,P.y,P.z,P.w):ze.set(P,Q,ae,ce),ie.viewport(O.copy(ze).multiplyScalar(fe).round())},this.getScissor=function(P){return P.copy(ke)},this.setScissor=function(P,Q,ae,ce){P.isVector4?ke.set(P.x,P.y,P.z,P.w):ke.set(P,Q,ae,ce),ie.scissor(z.copy(ke).multiplyScalar(fe).round())},this.getScissorTest=function(){return We},this.setScissorTest=function(P){ie.setScissorTest(We=P)},this.setOpaqueSort=function(P){Y=P},this.setTransparentSort=function(P){we=P},this.getClearColor=function(P){return P.copy(xe.getClearColor())},this.setClearColor=function(){xe.setClearColor(...arguments)},this.getClearAlpha=function(){return xe.getClearAlpha()},this.setClearAlpha=function(){xe.setClearAlpha(...arguments)},this.clear=function(P=!0,Q=!0,ae=!0){let ce=0;if(P){let j=!1;if(C!==null){const Ce=C.texture.format;j=x.has(Ce)}if(j){const Ce=C.texture.type,Ge=p.has(Ce),Ye=xe.getClearColor(),He=xe.getClearAlpha(),it=Ye.r,at=Ye.g,$e=Ye.b;Ge?(g[0]=it,g[1]=at,g[2]=$e,g[3]=He,R.clearBufferuiv(R.COLOR,0,g)):(m[0]=it,m[1]=at,m[2]=$e,m[3]=He,R.clearBufferiv(R.COLOR,0,m))}else ce|=R.COLOR_BUFFER_BIT}Q&&(ce|=R.DEPTH_BUFFER_BIT),ae&&(ce|=R.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),R.clear(ce)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){t.removeEventListener("webglcontextlost",Ae,!1),t.removeEventListener("webglcontextrestored",ge,!1),t.removeEventListener("webglcontextcreationerror",qe,!1),xe.dispose(),ye.dispose(),Xe.dispose(),se.dispose(),I.dispose(),T.dispose(),de.dispose(),W.dispose(),Le.dispose(),ee.dispose(),be.dispose(),be.removeEventListener("sessionstart",Jr),be.removeEventListener("sessionend",gl),cs.stop()};function Ae(P){P.preventDefault(),Bp("WebGLRenderer: Context Lost."),M=!0}function ge(){Bp("WebGLRenderer: Context Restored."),M=!1;const P=me.autoReset,Q=Z.enabled,ae=Z.autoUpdate,ce=Z.needsUpdate,j=Z.type;Me(),me.autoReset=P,Z.enabled=Q,Z.autoUpdate=ae,Z.needsUpdate=ce,Z.type=j}function qe(P){fn("WebGLRenderer: A WebGL context could not be created. Reason: ",P.statusMessage)}function Je(P){const Q=P.target;Q.removeEventListener("dispose",Je),rt(Q)}function rt(P){ot(P),se.remove(P)}function ot(P){const Q=se.get(P).programs;Q!==void 0&&(Q.forEach(function(ae){ee.releaseProgram(ae)}),P.isShaderMaterial&&ee.releaseShaderCache(P))}this.renderBufferDirect=function(P,Q,ae,ce,j,Ce){Q===null&&(Q=U);const Ge=j.isMesh&&j.matrixWorld.determinant()<0,Ye=_l(P,Q,ae,ce,j);ie.setMaterial(ce,Ge);let He=ae.index,it=1;if(ce.wireframe===!0){if(He=re.getWireframeAttribute(ae),He===void 0)return;it=2}const at=ae.drawRange,$e=ae.attributes.position;let vt=at.start*it,Mt=(at.start+at.count)*it;Ce!==null&&(vt=Math.max(vt,Ce.start*it),Mt=Math.min(Mt,(Ce.start+Ce.count)*it)),He!==null?(vt=Math.max(vt,0),Mt=Math.min(Mt,He.count)):$e!=null&&(vt=Math.max(vt,0),Mt=Math.min(Mt,$e.count));const Xt=Mt-vt;if(Xt<0||Xt===1/0)return;W.setup(j,ce,Ye,ae,He);let Qt,Dt=Be;if(He!==null&&(Qt=X.get(He),Dt=Fe,Dt.setIndex(Qt)),j.isMesh)ce.wireframe===!0?(ie.setLineWidth(ce.wireframeLinewidth*K()),Dt.setMode(R.LINES)):Dt.setMode(R.TRIANGLES);else if(j.isLine){let tt=ce.linewidth;tt===void 0&&(tt=1),ie.setLineWidth(tt*K()),j.isLineSegments?Dt.setMode(R.LINES):j.isLineLoop?Dt.setMode(R.LINE_LOOP):Dt.setMode(R.LINE_STRIP)}else j.isPoints?Dt.setMode(R.POINTS):j.isSprite&&Dt.setMode(R.TRIANGLES);if(j.isBatchedMesh)if(j._multiDrawInstances!==null)Ja("WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."),Dt.renderMultiDrawInstances(j._multiDrawStarts,j._multiDrawCounts,j._multiDrawCount,j._multiDrawInstances);else if(oe.get("WEBGL_multi_draw"))Dt.renderMultiDraw(j._multiDrawStarts,j._multiDrawCounts,j._multiDrawCount);else{const tt=j._multiDrawStarts,qt=j._multiDrawCounts,St=j._multiDrawCount,Un=He?X.get(He).bytesPerElement:1,fs=se.get(ce).currentProgram.getUniforms();for(let En=0;En<St;En++)fs.setValue(R,"_gl_DrawID",En),Dt.render(tt[En]/Un,qt[En])}else if(j.isInstancedMesh)Dt.renderInstances(vt,Xt,j.count);else if(ae.isInstancedBufferGeometry){const tt=ae._maxInstanceCount!==void 0?ae._maxInstanceCount:1/0,qt=Math.min(ae.instanceCount,tt);Dt.renderInstances(vt,Xt,qt)}else Dt.render(vt,Xt)};function Si(P,Q,ae){P.transparent===!0&&P.side===Ei&&P.forceSinglePass===!1?(P.side=Jn,P.needsUpdate=!0,eo(P,Q,ae),P.side=os,P.needsUpdate=!0,eo(P,Q,ae),P.side=Ei):eo(P,Q,ae)}this.compile=function(P,Q,ae=null){ae===null&&(ae=P),A=Xe.get(ae),A.init(Q),S.push(A),ae.traverseVisible(function(j){j.isLight&&j.layers.test(Q.layers)&&(A.pushLight(j),j.castShadow&&A.pushShadow(j))}),P!==ae&&P.traverseVisible(function(j){j.isLight&&j.layers.test(Q.layers)&&(A.pushLight(j),j.castShadow&&A.pushShadow(j))}),A.setupLights();const ce=new Set;return P.traverse(function(j){if(!(j.isMesh||j.isPoints||j.isLine||j.isSprite))return;const Ce=j.material;if(Ce)if(Array.isArray(Ce))for(let Ge=0;Ge<Ce.length;Ge++){const Ye=Ce[Ge];Si(Ye,ae,j),ce.add(Ye)}else Si(Ce,ae,j),ce.add(Ce)}),A=S.pop(),ce},this.compileAsync=function(P,Q,ae=null){const ce=this.compile(P,Q,ae);return new Promise(j=>{function Ce(){if(ce.forEach(function(Ge){se.get(Ge).currentProgram.isReady()&&ce.delete(Ge)}),ce.size===0){j(P);return}setTimeout(Ce,10)}oe.get("KHR_parallel_shader_compile")!==null?Ce():setTimeout(Ce,10)})};let ri=null;function mr(P){ri&&ri(P)}function Jr(){cs.stop()}function gl(){cs.start()}const cs=new Zg;cs.setAnimationLoop(mr),typeof self<"u"&&cs.setContext(self),this.setAnimationLoop=function(P){ri=P,be.setAnimationLoop(P),P===null?cs.stop():cs.start()},be.addEventListener("sessionstart",Jr),be.addEventListener("sessionend",gl),this.render=function(P,Q){if(Q!==void 0&&Q.isCamera!==!0){fn("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(M===!0)return;if(P.matrixWorldAutoUpdate===!0&&P.updateMatrixWorld(),Q.parent===null&&Q.matrixWorldAutoUpdate===!0&&Q.updateMatrixWorld(),be.enabled===!0&&be.isPresenting===!0&&(be.cameraAutoUpdate===!0&&be.updateCamera(Q),Q=be.getCamera()),P.isScene===!0&&P.onBeforeRender(y,P,Q,C),A=Xe.get(P,S.length),A.init(Q),S.push(A),he.multiplyMatrices(Q.projectionMatrix,Q.matrixWorldInverse),ne.setFromProjectionMatrix(he,Zi,Q.reversedDepth),Se=this.localClippingEnabled,ue=k.init(this.clippingPlanes,Se),_=ye.get(P,v.length),_.init(),v.push(_),be.enabled===!0&&be.isPresenting===!0){const Ce=y.xr.getDepthSensingMesh();Ce!==null&&gr(Ce,Q,-1/0,y.sortObjects)}gr(P,Q,0,y.sortObjects),_.finish(),y.sortObjects===!0&&_.sort(Y,we),N=be.enabled===!1||be.isPresenting===!1||be.hasDepthSensing()===!1,N&&xe.addToRenderList(_,P),this.info.render.frame++,ue===!0&&k.beginShadows();const ae=A.state.shadowsArray;Z.render(ae,P,Q),ue===!0&&k.endShadows(),this.info.autoReset===!0&&this.info.reset();const ce=_.opaque,j=_.transmissive;if(A.setupLights(),Q.isArrayCamera){const Ce=Q.cameras;if(j.length>0)for(let Ge=0,Ye=Ce.length;Ge<Ye;Ge++){const He=Ce[Ge];us(ce,j,P,He)}N&&xe.render(P);for(let Ge=0,Ye=Ce.length;Ge<Ye;Ge++){const He=Ce[Ge];xl(_,P,He,He.viewport)}}else j.length>0&&us(ce,j,P,Q),N&&xe.render(P),xl(_,P,Q);C!==null&&b===0&&(ve.updateMultisampleRenderTarget(C),ve.updateRenderTargetMipmap(C)),P.isScene===!0&&P.onAfterRender(y,P,Q),W.resetDefaultState(),D=-1,F=null,S.pop(),S.length>0?(A=S[S.length-1],ue===!0&&k.setGlobalState(y.clippingPlanes,A.state.camera)):A=null,v.pop(),v.length>0?_=v[v.length-1]:_=null};function gr(P,Q,ae,ce){if(P.visible===!1)return;if(P.layers.test(Q.layers)){if(P.isGroup)ae=P.renderOrder;else if(P.isLOD)P.autoUpdate===!0&&P.update(Q);else if(P.isLight)A.pushLight(P),P.castShadow&&A.pushShadow(P);else if(P.isSprite){if(!P.frustumCulled||ne.intersectsSprite(P)){ce&&Ze.setFromMatrixPosition(P.matrixWorld).applyMatrix4(he);const Ge=de.update(P),Ye=P.material;Ye.visible&&_.push(P,Ge,Ye,ae,Ze.z,null)}}else if((P.isMesh||P.isLine||P.isPoints)&&(!P.frustumCulled||ne.intersectsObject(P))){const Ge=de.update(P),Ye=P.material;if(ce&&(P.boundingSphere!==void 0?(P.boundingSphere===null&&P.computeBoundingSphere(),Ze.copy(P.boundingSphere.center)):(Ge.boundingSphere===null&&Ge.computeBoundingSphere(),Ze.copy(Ge.boundingSphere.center)),Ze.applyMatrix4(P.matrixWorld).applyMatrix4(he)),Array.isArray(Ye)){const He=Ge.groups;for(let it=0,at=He.length;it<at;it++){const $e=He[it],vt=Ye[$e.materialIndex];vt&&vt.visible&&_.push(P,Ge,vt,ae,Ze.z,$e)}}else Ye.visible&&_.push(P,Ge,Ye,ae,Ze.z,null)}}const Ce=P.children;for(let Ge=0,Ye=Ce.length;Ge<Ye;Ge++)gr(Ce[Ge],Q,ae,ce)}function xl(P,Q,ae,ce){const{opaque:j,transmissive:Ce,transparent:Ge}=P;A.setupLightsView(ae),ue===!0&&k.setGlobalState(y.clippingPlanes,ae),ce&&ie.viewport(O.copy(ce)),j.length>0&&xr(j,Q,ae),Ce.length>0&&xr(Ce,Q,ae),Ge.length>0&&xr(Ge,Q,ae),ie.buffers.depth.setTest(!0),ie.buffers.depth.setMask(!0),ie.buffers.color.setMask(!0),ie.setPolygonOffset(!1)}function us(P,Q,ae,ce){if((ae.isScene===!0?ae.overrideMaterial:null)!==null)return;A.state.transmissionRenderTarget[ce.id]===void 0&&(A.state.transmissionRenderTarget[ce.id]=new cr(1,1,{generateMipmaps:!0,type:oe.has("EXT_color_buffer_half_float")||oe.has("EXT_color_buffer_float")?jr:as,minFilter:kr,samples:4,stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:Et.workingColorSpace}));const Ce=A.state.transmissionRenderTarget[ce.id],Ge=ce.viewport||O;Ce.setSize(Ge.z*y.transmissionResolutionScale,Ge.w*y.transmissionResolutionScale);const Ye=y.getRenderTarget(),He=y.getActiveCubeFace(),it=y.getActiveMipmapLevel();y.setRenderTarget(Ce),y.getClearColor(H),q=y.getClearAlpha(),q<1&&y.setClearColor(16777215,.5),y.clear(),N&&xe.render(ae);const at=y.toneMapping;y.toneMapping=sr;const $e=ce.viewport;if(ce.viewport!==void 0&&(ce.viewport=void 0),A.setupLightsView(ce),ue===!0&&k.setGlobalState(y.clippingPlanes,ce),xr(P,ae,ce),ve.updateMultisampleRenderTarget(Ce),ve.updateRenderTargetMipmap(Ce),oe.has("WEBGL_multisampled_render_to_texture")===!1){let vt=!1;for(let Mt=0,Xt=Q.length;Mt<Xt;Mt++){const Qt=Q[Mt],{object:Dt,geometry:tt,material:qt,group:St}=Qt;if(qt.side===Ei&&Dt.layers.test(ce.layers)){const Un=qt.side;qt.side=Jn,qt.needsUpdate=!0,aa(Dt,ae,ce,tt,qt,St),qt.side=Un,qt.needsUpdate=!0,vt=!0}}vt===!0&&(ve.updateMultisampleRenderTarget(Ce),ve.updateRenderTargetMipmap(Ce))}y.setRenderTarget(Ye,He,it),y.setClearColor(H,q),$e!==void 0&&(ce.viewport=$e),y.toneMapping=at}function xr(P,Q,ae){const ce=Q.isScene===!0?Q.overrideMaterial:null;for(let j=0,Ce=P.length;j<Ce;j++){const Ge=P[j],{object:Ye,geometry:He,group:it}=Ge;let at=Ge.material;at.allowOverride===!0&&ce!==null&&(at=ce),Ye.layers.test(ae.layers)&&aa(Ye,Q,ae,He,at,it)}}function aa(P,Q,ae,ce,j,Ce){P.onBeforeRender(y,Q,ae,ce,j,Ce),P.modelViewMatrix.multiplyMatrices(ae.matrixWorldInverse,P.matrixWorld),P.normalMatrix.getNormalMatrix(P.modelViewMatrix),j.onBeforeRender(y,Q,ae,ce,P,Ce),j.transparent===!0&&j.side===Ei&&j.forceSinglePass===!1?(j.side=Jn,j.needsUpdate=!0,y.renderBufferDirect(ae,Q,ce,j,P,Ce),j.side=os,j.needsUpdate=!0,y.renderBufferDirect(ae,Q,ce,j,P,Ce),j.side=Ei):y.renderBufferDirect(ae,Q,ce,j,P,Ce),P.onAfterRender(y,Q,ae,ce,j,Ce)}function eo(P,Q,ae){Q.isScene!==!0&&(Q=U);const ce=se.get(P),j=A.state.lights,Ce=A.state.shadowsArray,Ge=j.state.version,Ye=ee.getParameters(P,j.state,Ce,Q,ae),He=ee.getProgramCacheKey(Ye);let it=ce.programs;ce.environment=P.isMeshStandardMaterial?Q.environment:null,ce.fog=Q.fog,ce.envMap=(P.isMeshStandardMaterial?T:I).get(P.envMap||ce.environment),ce.envMapRotation=ce.environment!==null&&P.envMap===null?Q.environmentRotation:P.envMapRotation,it===void 0&&(P.addEventListener("dispose",Je),it=new Map,ce.programs=it);let at=it.get(He);if(at!==void 0){if(ce.currentProgram===at&&ce.lightsStateVersion===Ge)return la(P,Ye),at}else Ye.uniforms=ee.getUniforms(P),P.onBeforeCompile(Ye,y),at=ee.acquireProgram(Ye,He),it.set(He,at),ce.uniforms=Ye.uniforms;const $e=ce.uniforms;return(!P.isShaderMaterial&&!P.isRawShaderMaterial||P.clipping===!0)&&($e.clippingPlanes=k.uniform),la(P,Ye),ce.needsLights=ua(P),ce.lightsStateVersion=Ge,ce.needsLights&&($e.ambientLightColor.value=j.state.ambient,$e.lightProbe.value=j.state.probe,$e.directionalLights.value=j.state.directional,$e.directionalLightShadows.value=j.state.directionalShadow,$e.spotLights.value=j.state.spot,$e.spotLightShadows.value=j.state.spotShadow,$e.rectAreaLights.value=j.state.rectArea,$e.ltc_1.value=j.state.rectAreaLTC1,$e.ltc_2.value=j.state.rectAreaLTC2,$e.pointLights.value=j.state.point,$e.pointLightShadows.value=j.state.pointShadow,$e.hemisphereLights.value=j.state.hemi,$e.directionalShadowMap.value=j.state.directionalShadowMap,$e.directionalShadowMatrix.value=j.state.directionalShadowMatrix,$e.spotShadowMap.value=j.state.spotShadowMap,$e.spotLightMatrix.value=j.state.spotLightMatrix,$e.spotLightMap.value=j.state.spotLightMap,$e.pointShadowMap.value=j.state.pointShadowMap,$e.pointShadowMatrix.value=j.state.pointShadowMatrix),ce.currentProgram=at,ce.uniformsList=null,at}function to(P){if(P.uniformsList===null){const Q=P.currentProgram.getUniforms();P.uniformsList=dc.seqWithValue(Q.seq,P.uniforms)}return P.uniformsList}function la(P,Q){const ae=se.get(P);ae.outputColorSpace=Q.outputColorSpace,ae.batching=Q.batching,ae.batchingColor=Q.batchingColor,ae.instancing=Q.instancing,ae.instancingColor=Q.instancingColor,ae.instancingMorph=Q.instancingMorph,ae.skinning=Q.skinning,ae.morphTargets=Q.morphTargets,ae.morphNormals=Q.morphNormals,ae.morphColors=Q.morphColors,ae.morphTargetsCount=Q.morphTargetsCount,ae.numClippingPlanes=Q.numClippingPlanes,ae.numIntersection=Q.numClipIntersection,ae.vertexAlphas=Q.vertexAlphas,ae.vertexTangents=Q.vertexTangents,ae.toneMapping=Q.toneMapping}function _l(P,Q,ae,ce,j){Q.isScene!==!0&&(Q=U),ve.resetTextureUnits();const Ce=Q.fog,Ge=ce.isMeshStandardMaterial?Q.environment:null,Ye=C===null?y.outputColorSpace:C.isXRRenderTarget===!0?C.texture.colorSpace:Qo,He=(ce.isMeshStandardMaterial?T:I).get(ce.envMap||Ge),it=ce.vertexColors===!0&&!!ae.attributes.color&&ae.attributes.color.itemSize===4,at=!!ae.attributes.tangent&&(!!ce.normalMap||ce.anisotropy>0),$e=!!ae.morphAttributes.position,vt=!!ae.morphAttributes.normal,Mt=!!ae.morphAttributes.color;let Xt=sr;ce.toneMapped&&(C===null||C.isXRRenderTarget===!0)&&(Xt=y.toneMapping);const Qt=ae.morphAttributes.position||ae.morphAttributes.normal||ae.morphAttributes.color,Dt=Qt!==void 0?Qt.length:0,tt=se.get(ce),qt=A.state.lights;if(ue===!0&&(Se===!0||P!==F)){const wt=P===F&&ce.id===D;k.setState(ce,P,wt)}let St=!1;ce.version===tt.__version?(tt.needsLights&&tt.lightsStateVersion!==qt.state.version||tt.outputColorSpace!==Ye||j.isBatchedMesh&&tt.batching===!1||!j.isBatchedMesh&&tt.batching===!0||j.isBatchedMesh&&tt.batchingColor===!0&&j.colorTexture===null||j.isBatchedMesh&&tt.batchingColor===!1&&j.colorTexture!==null||j.isInstancedMesh&&tt.instancing===!1||!j.isInstancedMesh&&tt.instancing===!0||j.isSkinnedMesh&&tt.skinning===!1||!j.isSkinnedMesh&&tt.skinning===!0||j.isInstancedMesh&&tt.instancingColor===!0&&j.instanceColor===null||j.isInstancedMesh&&tt.instancingColor===!1&&j.instanceColor!==null||j.isInstancedMesh&&tt.instancingMorph===!0&&j.morphTexture===null||j.isInstancedMesh&&tt.instancingMorph===!1&&j.morphTexture!==null||tt.envMap!==He||ce.fog===!0&&tt.fog!==Ce||tt.numClippingPlanes!==void 0&&(tt.numClippingPlanes!==k.numPlanes||tt.numIntersection!==k.numIntersection)||tt.vertexAlphas!==it||tt.vertexTangents!==at||tt.morphTargets!==$e||tt.morphNormals!==vt||tt.morphColors!==Mt||tt.toneMapping!==Xt||tt.morphTargetsCount!==Dt)&&(St=!0):(St=!0,tt.__version=ce.version);let Un=tt.currentProgram;St===!0&&(Un=eo(ce,Q,j));let fs=!1,En=!1,_r=!1;const dt=Un.getUniforms(),ht=tt.uniforms;if(ie.useProgram(Un.program)&&(fs=!0,En=!0,_r=!0),ce.id!==D&&(D=ce.id,En=!0),fs||F!==P){ie.buffers.depth.getReversed()&&P.reversedDepth!==!0&&(P._reversedDepth=!0,P.updateProjectionMatrix()),dt.setValue(R,"projectionMatrix",P.projectionMatrix),dt.setValue(R,"viewMatrix",P.matrixWorldInverse);const ln=dt.map.cameraPosition;ln!==void 0&&ln.setValue(R,Ee.setFromMatrixPosition(P.matrixWorld)),pe.logarithmicDepthBuffer&&dt.setValue(R,"logDepthBufFC",2/(Math.log(P.far+1)/Math.LN2)),(ce.isMeshPhongMaterial||ce.isMeshToonMaterial||ce.isMeshLambertMaterial||ce.isMeshBasicMaterial||ce.isMeshStandardMaterial||ce.isShaderMaterial)&&dt.setValue(R,"isOrthographic",P.isOrthographicCamera===!0),F!==P&&(F=P,En=!0,_r=!0)}if(j.isSkinnedMesh){dt.setOptional(R,j,"bindMatrix"),dt.setOptional(R,j,"bindMatrixInverse");const wt=j.skeleton;wt&&(wt.boneTexture===null&&wt.computeBoneTexture(),dt.setValue(R,"boneTexture",wt.boneTexture,ve))}j.isBatchedMesh&&(dt.setOptional(R,j,"batchingTexture"),dt.setValue(R,"batchingTexture",j._matricesTexture,ve),dt.setOptional(R,j,"batchingIdTexture"),dt.setValue(R,"batchingIdTexture",j._indirectTexture,ve),dt.setOptional(R,j,"batchingColorTexture"),j._colorsTexture!==null&&dt.setValue(R,"batchingColorTexture",j._colorsTexture,ve));const mt=ae.morphAttributes;if((mt.position!==void 0||mt.normal!==void 0||mt.color!==void 0)&&Re.update(j,ae,Un),(En||tt.receiveShadow!==j.receiveShadow)&&(tt.receiveShadow=j.receiveShadow,dt.setValue(R,"receiveShadow",j.receiveShadow)),ce.isMeshGouraudMaterial&&ce.envMap!==null&&(ht.envMap.value=He,ht.flipEnvMap.value=He.isCubeTexture&&He.isRenderTargetTexture===!1?-1:1),ce.isMeshStandardMaterial&&ce.envMap===null&&Q.environment!==null&&(ht.envMapIntensity.value=Q.environmentIntensity),ht.dfgLUT!==void 0&&(ht.dfgLUT.value=A1()),En&&(dt.setValue(R,"toneMappingExposure",y.toneMappingExposure),tt.needsLights&&ca(ht,_r),Ce&&ce.fog===!0&&Ue.refreshFogUniforms(ht,Ce),Ue.refreshMaterialUniforms(ht,ce,fe,$,A.state.transmissionRenderTarget[P.id]),dc.upload(R,to(tt),ht,ve)),ce.isShaderMaterial&&ce.uniformsNeedUpdate===!0&&(dc.upload(R,to(tt),ht,ve),ce.uniformsNeedUpdate=!1),ce.isSpriteMaterial&&dt.setValue(R,"center",j.center),dt.setValue(R,"modelViewMatrix",j.modelViewMatrix),dt.setValue(R,"normalMatrix",j.normalMatrix),dt.setValue(R,"modelMatrix",j.matrixWorld),ce.isShaderMaterial||ce.isRawShaderMaterial){const wt=ce.uniformsGroups;for(let ln=0,vr=wt.length;ln<vr;ln++){const Bi=wt[ln];Le.update(Bi,Un),Le.bind(Bi,Un)}}return Un}function ca(P,Q){P.ambientLightColor.needsUpdate=Q,P.lightProbe.needsUpdate=Q,P.directionalLights.needsUpdate=Q,P.directionalLightShadows.needsUpdate=Q,P.pointLights.needsUpdate=Q,P.pointLightShadows.needsUpdate=Q,P.spotLights.needsUpdate=Q,P.spotLightShadows.needsUpdate=Q,P.rectAreaLights.needsUpdate=Q,P.hemisphereLights.needsUpdate=Q}function ua(P){return P.isMeshLambertMaterial||P.isMeshToonMaterial||P.isMeshPhongMaterial||P.isMeshStandardMaterial||P.isShadowMaterial||P.isShaderMaterial&&P.lights===!0}this.getActiveCubeFace=function(){return E},this.getActiveMipmapLevel=function(){return b},this.getRenderTarget=function(){return C},this.setRenderTargetTextures=function(P,Q,ae){const ce=se.get(P);ce.__autoAllocateDepthBuffer=P.resolveDepthBuffer===!1,ce.__autoAllocateDepthBuffer===!1&&(ce.__useRenderToTexture=!1),se.get(P.texture).__webglTexture=Q,se.get(P.depthTexture).__webglTexture=ce.__autoAllocateDepthBuffer?void 0:ae,ce.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(P,Q){const ae=se.get(P);ae.__webglFramebuffer=Q,ae.__useDefaultFramebuffer=Q===void 0};const vl=R.createFramebuffer();this.setRenderTarget=function(P,Q=0,ae=0){C=P,E=Q,b=ae;let ce=!0,j=null,Ce=!1,Ge=!1;if(P){const He=se.get(P);if(He.__useDefaultFramebuffer!==void 0)ie.bindFramebuffer(R.FRAMEBUFFER,null),ce=!1;else if(He.__webglFramebuffer===void 0)ve.setupRenderTarget(P);else if(He.__hasExternalTextures)ve.rebindTextures(P,se.get(P.texture).__webglTexture,se.get(P.depthTexture).__webglTexture);else if(P.depthBuffer){const $e=P.depthTexture;if(He.__boundDepthTexture!==$e){if($e!==null&&se.has($e)&&(P.width!==$e.image.width||P.height!==$e.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");ve.setupDepthRenderbuffer(P)}}const it=P.texture;(it.isData3DTexture||it.isDataArrayTexture||it.isCompressedArrayTexture)&&(Ge=!0);const at=se.get(P).__webglFramebuffer;P.isWebGLCubeRenderTarget?(Array.isArray(at[Q])?j=at[Q][ae]:j=at[Q],Ce=!0):P.samples>0&&ve.useMultisampledRTT(P)===!1?j=se.get(P).__webglMultisampledFramebuffer:Array.isArray(at)?j=at[ae]:j=at,O.copy(P.viewport),z.copy(P.scissor),V=P.scissorTest}else O.copy(ze).multiplyScalar(fe).floor(),z.copy(ke).multiplyScalar(fe).floor(),V=We;if(ae!==0&&(j=vl),ie.bindFramebuffer(R.FRAMEBUFFER,j)&&ce&&ie.drawBuffers(P,j),ie.viewport(O),ie.scissor(z),ie.setScissorTest(V),Ce){const He=se.get(P.texture);R.framebufferTexture2D(R.FRAMEBUFFER,R.COLOR_ATTACHMENT0,R.TEXTURE_CUBE_MAP_POSITIVE_X+Q,He.__webglTexture,ae)}else if(Ge){const He=Q;for(let it=0;it<P.textures.length;it++){const at=se.get(P.textures[it]);R.framebufferTextureLayer(R.FRAMEBUFFER,R.COLOR_ATTACHMENT0+it,at.__webglTexture,ae,He)}}else if(P!==null&&ae!==0){const He=se.get(P.texture);R.framebufferTexture2D(R.FRAMEBUFFER,R.COLOR_ATTACHMENT0,R.TEXTURE_2D,He.__webglTexture,ae)}D=-1},this.readRenderTargetPixels=function(P,Q,ae,ce,j,Ce,Ge,Ye=0){if(!(P&&P.isWebGLRenderTarget)){fn("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let He=se.get(P).__webglFramebuffer;if(P.isWebGLCubeRenderTarget&&Ge!==void 0&&(He=He[Ge]),He){ie.bindFramebuffer(R.FRAMEBUFFER,He);try{const it=P.textures[Ye],at=it.format,$e=it.type;if(!pe.textureFormatReadable(at)){fn("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!pe.textureTypeReadable($e)){fn("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}Q>=0&&Q<=P.width-ce&&ae>=0&&ae<=P.height-j&&(P.textures.length>1&&R.readBuffer(R.COLOR_ATTACHMENT0+Ye),R.readPixels(Q,ae,ce,j,je.convert(at),je.convert($e),Ce))}finally{const it=C!==null?se.get(C).__webglFramebuffer:null;ie.bindFramebuffer(R.FRAMEBUFFER,it)}}},this.readRenderTargetPixelsAsync=async function(P,Q,ae,ce,j,Ce,Ge,Ye=0){if(!(P&&P.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let He=se.get(P).__webglFramebuffer;if(P.isWebGLCubeRenderTarget&&Ge!==void 0&&(He=He[Ge]),He)if(Q>=0&&Q<=P.width-ce&&ae>=0&&ae<=P.height-j){ie.bindFramebuffer(R.FRAMEBUFFER,He);const it=P.textures[Ye],at=it.format,$e=it.type;if(!pe.textureFormatReadable(at))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!pe.textureTypeReadable($e))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const vt=R.createBuffer();R.bindBuffer(R.PIXEL_PACK_BUFFER,vt),R.bufferData(R.PIXEL_PACK_BUFFER,Ce.byteLength,R.STREAM_READ),P.textures.length>1&&R.readBuffer(R.COLOR_ATTACHMENT0+Ye),R.readPixels(Q,ae,ce,j,je.convert(at),je.convert($e),0);const Mt=C!==null?se.get(C).__webglFramebuffer:null;ie.bindFramebuffer(R.FRAMEBUFFER,Mt);const Xt=R.fenceSync(R.SYNC_GPU_COMMANDS_COMPLETE,0);return R.flush(),await LS(R,Xt,4),R.bindBuffer(R.PIXEL_PACK_BUFFER,vt),R.getBufferSubData(R.PIXEL_PACK_BUFFER,0,Ce),R.deleteBuffer(vt),R.deleteSync(Xt),Ce}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(P,Q=null,ae=0){const ce=Math.pow(2,-ae),j=Math.floor(P.image.width*ce),Ce=Math.floor(P.image.height*ce),Ge=Q!==null?Q.x:0,Ye=Q!==null?Q.y:0;ve.setTexture2D(P,0),R.copyTexSubImage2D(R.TEXTURE_2D,ae,0,0,Ge,Ye,j,Ce),ie.unbindTexture()};const iu=R.createFramebuffer(),su=R.createFramebuffer();this.copyTextureToTexture=function(P,Q,ae=null,ce=null,j=0,Ce=null){Ce===null&&(j!==0?(Ja("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."),Ce=j,j=0):Ce=0);let Ge,Ye,He,it,at,$e,vt,Mt,Xt;const Qt=P.isCompressedTexture?P.mipmaps[Ce]:P.image;if(ae!==null)Ge=ae.max.x-ae.min.x,Ye=ae.max.y-ae.min.y,He=ae.isBox3?ae.max.z-ae.min.z:1,it=ae.min.x,at=ae.min.y,$e=ae.isBox3?ae.min.z:0;else{const mt=Math.pow(2,-j);Ge=Math.floor(Qt.width*mt),Ye=Math.floor(Qt.height*mt),P.isDataArrayTexture?He=Qt.depth:P.isData3DTexture?He=Math.floor(Qt.depth*mt):He=1,it=0,at=0,$e=0}ce!==null?(vt=ce.x,Mt=ce.y,Xt=ce.z):(vt=0,Mt=0,Xt=0);const Dt=je.convert(Q.format),tt=je.convert(Q.type);let qt;Q.isData3DTexture?(ve.setTexture3D(Q,0),qt=R.TEXTURE_3D):Q.isDataArrayTexture||Q.isCompressedArrayTexture?(ve.setTexture2DArray(Q,0),qt=R.TEXTURE_2D_ARRAY):(ve.setTexture2D(Q,0),qt=R.TEXTURE_2D),R.pixelStorei(R.UNPACK_FLIP_Y_WEBGL,Q.flipY),R.pixelStorei(R.UNPACK_PREMULTIPLY_ALPHA_WEBGL,Q.premultiplyAlpha),R.pixelStorei(R.UNPACK_ALIGNMENT,Q.unpackAlignment);const St=R.getParameter(R.UNPACK_ROW_LENGTH),Un=R.getParameter(R.UNPACK_IMAGE_HEIGHT),fs=R.getParameter(R.UNPACK_SKIP_PIXELS),En=R.getParameter(R.UNPACK_SKIP_ROWS),_r=R.getParameter(R.UNPACK_SKIP_IMAGES);R.pixelStorei(R.UNPACK_ROW_LENGTH,Qt.width),R.pixelStorei(R.UNPACK_IMAGE_HEIGHT,Qt.height),R.pixelStorei(R.UNPACK_SKIP_PIXELS,it),R.pixelStorei(R.UNPACK_SKIP_ROWS,at),R.pixelStorei(R.UNPACK_SKIP_IMAGES,$e);const dt=P.isDataArrayTexture||P.isData3DTexture,ht=Q.isDataArrayTexture||Q.isData3DTexture;if(P.isDepthTexture){const mt=se.get(P),wt=se.get(Q),ln=se.get(mt.__renderTarget),vr=se.get(wt.__renderTarget);ie.bindFramebuffer(R.READ_FRAMEBUFFER,ln.__webglFramebuffer),ie.bindFramebuffer(R.DRAW_FRAMEBUFFER,vr.__webglFramebuffer);for(let Bi=0;Bi<He;Bi++)dt&&(R.framebufferTextureLayer(R.READ_FRAMEBUFFER,R.COLOR_ATTACHMENT0,se.get(P).__webglTexture,j,$e+Bi),R.framebufferTextureLayer(R.DRAW_FRAMEBUFFER,R.COLOR_ATTACHMENT0,se.get(Q).__webglTexture,Ce,Xt+Bi)),R.blitFramebuffer(it,at,Ge,Ye,vt,Mt,Ge,Ye,R.DEPTH_BUFFER_BIT,R.NEAREST);ie.bindFramebuffer(R.READ_FRAMEBUFFER,null),ie.bindFramebuffer(R.DRAW_FRAMEBUFFER,null)}else if(j!==0||P.isRenderTargetTexture||se.has(P)){const mt=se.get(P),wt=se.get(Q);ie.bindFramebuffer(R.READ_FRAMEBUFFER,iu),ie.bindFramebuffer(R.DRAW_FRAMEBUFFER,su);for(let ln=0;ln<He;ln++)dt?R.framebufferTextureLayer(R.READ_FRAMEBUFFER,R.COLOR_ATTACHMENT0,mt.__webglTexture,j,$e+ln):R.framebufferTexture2D(R.READ_FRAMEBUFFER,R.COLOR_ATTACHMENT0,R.TEXTURE_2D,mt.__webglTexture,j),ht?R.framebufferTextureLayer(R.DRAW_FRAMEBUFFER,R.COLOR_ATTACHMENT0,wt.__webglTexture,Ce,Xt+ln):R.framebufferTexture2D(R.DRAW_FRAMEBUFFER,R.COLOR_ATTACHMENT0,R.TEXTURE_2D,wt.__webglTexture,Ce),j!==0?R.blitFramebuffer(it,at,Ge,Ye,vt,Mt,Ge,Ye,R.COLOR_BUFFER_BIT,R.NEAREST):ht?R.copyTexSubImage3D(qt,Ce,vt,Mt,Xt+ln,it,at,Ge,Ye):R.copyTexSubImage2D(qt,Ce,vt,Mt,it,at,Ge,Ye);ie.bindFramebuffer(R.READ_FRAMEBUFFER,null),ie.bindFramebuffer(R.DRAW_FRAMEBUFFER,null)}else ht?P.isDataTexture||P.isData3DTexture?R.texSubImage3D(qt,Ce,vt,Mt,Xt,Ge,Ye,He,Dt,tt,Qt.data):Q.isCompressedArrayTexture?R.compressedTexSubImage3D(qt,Ce,vt,Mt,Xt,Ge,Ye,He,Dt,Qt.data):R.texSubImage3D(qt,Ce,vt,Mt,Xt,Ge,Ye,He,Dt,tt,Qt):P.isDataTexture?R.texSubImage2D(R.TEXTURE_2D,Ce,vt,Mt,Ge,Ye,Dt,tt,Qt.data):P.isCompressedTexture?R.compressedTexSubImage2D(R.TEXTURE_2D,Ce,vt,Mt,Qt.width,Qt.height,Dt,Qt.data):R.texSubImage2D(R.TEXTURE_2D,Ce,vt,Mt,Ge,Ye,Dt,tt,Qt);R.pixelStorei(R.UNPACK_ROW_LENGTH,St),R.pixelStorei(R.UNPACK_IMAGE_HEIGHT,Un),R.pixelStorei(R.UNPACK_SKIP_PIXELS,fs),R.pixelStorei(R.UNPACK_SKIP_ROWS,En),R.pixelStorei(R.UNPACK_SKIP_IMAGES,_r),Ce===0&&Q.generateMipmaps&&R.generateMipmap(qt),ie.unbindTexture()},this.initRenderTarget=function(P){se.get(P).__webglFramebuffer===void 0&&ve.setupRenderTarget(P)},this.initTexture=function(P){P.isCubeTexture?ve.setTextureCube(P,0):P.isData3DTexture?ve.setTexture3D(P,0):P.isDataArrayTexture||P.isCompressedArrayTexture?ve.setTexture2DArray(P,0):ve.setTexture2D(P,0),ie.unbindTexture()},this.resetState=function(){E=0,b=0,C=null,ie.reset(),W.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Zi}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const t=this.getContext();t.drawingBufferColorSpace=Et._getDrawingBufferColorSpace(e),t.unpackColorSpace=Et._getUnpackColorSpace()}}class Zs{static idGen=0;constructor(e,t){let n,s;this.promise=new Promise((c,u)=>{n=c,s=u});const r=n.bind(this),o=s.bind(this),a=(...c)=>{r(...c)},l=c=>{o(c)};e(a.bind(this),l.bind(this)),this.abortHandler=t,this.id=Zs.idGen++}then(e){return new Zs((t,n)=>{this.promise=this.promise.then((...s)=>{const r=e(...s);r instanceof Promise||r instanceof Zs?r.then((...o)=>{t(...o)}):t(r)}).catch(s=>{n(s)})},this.abortHandler)}catch(e){return new Zs(t=>{this.promise=this.promise.then((...n)=>{t(...n)}).catch(e)},this.abortHandler)}abort(e){this.abortHandler&&this.abortHandler(e)}}class sx extends Error{constructor(e){super(e)}}(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){i[0]=t;const n=e[0];let s=n>>16&32768,r=n>>12&2047;const o=n>>23&255;return o<103?s:o>142?(s|=31744,s|=(o==255?0:1)&&n&8388607,s):o<113?(r|=2048,s|=(r>>114-o)+(r>>113-o&1),s):(s|=o-112<<10|r>>1,s+=r&1,s)}})();const Qu=(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){return i[0]=t,e[0]}})(),y1=function(i,e){return i[e]+(i[e+1]<<8)+(i[e+2]<<16)+(i[e+3]<<24)},Jc=function(i,e,t=!0,n){const s=new AbortController,r=s.signal;let o=!1;const a=u=>{s.abort(u),o=!0};let l=!1;const c=(u,f,d,h)=>{e&&!l&&(e(u,f,d,h),u===100&&(l=!0))};return new Zs((u,f)=>{const d={signal:r};n&&(d.headers=n),fetch(i,d).then(async h=>{if(!h.ok){const A=await h.text();f(new Error(`Fetch failed: ${h.status} ${h.statusText} ${A}`));return}const x=h.body.getReader();let p=0,g=h.headers.get("Content-Length"),m=g?parseInt(g):void 0;const _=[];for(;!o;)try{const{value:A,done:v}=await x.read();if(v){if(c(100,"100%",A,m),t){const M=new Blob(_).arrayBuffer();u(M)}else u();break}p+=A.length;let S,y;m!==void 0&&(S=p/m*100,y=`${S.toFixed(2)}%`),t&&_.push(A),c(S,y,A,m)}catch(A){f(A);return}}).catch(h=>{f(new sx(h))})},a)},$t=function(i,e,t){return Math.max(Math.min(i,t),e)},Ao=function(){return performance.now()/1e3},Co=i=>{if(i.geometry&&(i.geometry.dispose(),i.geometry=null),i.material&&(i.material.dispose(),i.material=null),i.children)for(let e of i.children)Co(e)},pi=(i,e)=>new Promise(t=>{window.setTimeout(()=>{t(i?i():void 0)},e?1:50)}),Uo=(i=0)=>{let e=0;if(i===1)e=9;else if(i===2)e=24;else if(i===3)e=45;else if(i>3)throw new Error("getSphericalHarmonicsComponentCountForDegree() -> Invalid spherical harmonics degree");return e},ch=()=>{let i,e;return{promise:new Promise((n,s)=>{i=n,e=s}),resolve:i,reject:e}},Ku=i=>{let e,t;return i||(i=()=>{}),{promise:new Zs((s,r)=>{e=s,t=r},i),resolve:e,reject:t}};class b1{constructor(e,t,n){this.major=e,this.minor=t,this.patch=n}toString(){return`${this.major}_${this.minor}_${this.patch}`}}function uh(){const i=navigator.userAgent;return i.indexOf("iPhone")>0||i.indexOf("iPad")>0}function rx(){if(uh()){const i=navigator.userAgent.match(/OS (\d+)_(\d+)_?(\d+)?/);return new b1(parseInt(i[1]||0,10),parseInt(i[2]||0,10),parseInt(i[3]||0,10))}else return null}const M1=14;class Pe{static OFFSET={X:0,Y:1,Z:2,SCALE0:3,SCALE1:4,SCALE2:5,ROTATION0:6,ROTATION1:7,ROTATION2:8,ROTATION3:9,FDC0:10,FDC1:11,FDC2:12,OPACITY:13,FRC0:14,FRC1:15,FRC2:16,FRC3:17,FRC4:18,FRC5:19,FRC6:20,FRC7:21,FRC8:22,FRC9:23,FRC10:24,FRC11:25,FRC12:26,FRC13:27,FRC14:28,FRC15:29,FRC16:30,FRC17:31,FRC18:32,FRC19:33,FRC20:34,FRC21:35,FRC22:36,FRC23:37};constructor(e=0){this.sphericalHarmonicsDegree=e,this.sphericalHarmonicsCount=Uo(this.sphericalHarmonicsDegree),this.componentCount=this.sphericalHarmonicsCount+M1,this.defaultSphericalHarmonics=new Array(this.sphericalHarmonicsCount).fill(0),this.splats=[],this.splatCount=0}static createSplat(e=0){const t=[0,0,0,1,1,1,1,0,0,0,0,0,0,0];let n=Uo(e);for(let s=0;s<n;s++)t.push(0);return t}addSplat(e){this.splats.push(e),this.splatCount++}getSplat(e){return this.splats[e]}addDefaultSplat(){const e=Pe.createSplat(this.sphericalHarmonicsDegree);return this.addSplat(e),e}addSplatFromComonents(e,t,n,s,r,o,a,l,c,u,f,d,h,x,...p){const g=[e,t,n,s,r,o,a,l,c,u,f,d,h,x,...this.defaultSphericalHarmonics];for(let m=0;m<p.length&&m<this.sphericalHarmonicsCount;m++)g[m]=p[m];return this.addSplat(g),g}addSplatFromArray(e,t){const n=e.splats[t],s=Pe.createSplat(this.sphericalHarmonicsDegree);for(let r=0;r<this.componentCount&&r<n.length;r++)s[r]=n[r];this.addSplat(s)}}class Nt{static DefaultSplatSortDistanceMapPrecision=16;static MemoryPageSize=65536;static BytesPerFloat=4;static BytesPerInt=4;static MaxScenes=32;static ProgressiveLoadSectionSize=262144;static ProgressiveLoadSectionDelayDuration=15;static SphericalHarmonics8BitCompressionRange=3}const C1=Nt.SphericalHarmonics8BitCompressionRange,Xs=C1/2,xn=tl.toHalfFloat.bind(tl),fh=tl.fromHalfFloat.bind(tl),Kt=(i,e,t=!1,n,s)=>{if(e===0)return i;if(e===1||e===2&&!t)return tl.fromHalfFloat(i);if(e===2)return dh(i,n,s)},Ba=(i,e,t)=>{i=$t(i,e,t);const n=t-e;return $t(Math.floor((i-e)/n*255),0,255)},dh=(i,e,t)=>{const n=t-e;return i/255*n+e},ox=(i,e,t)=>Ba(fh(i,e,t)),T1=(i,e,t)=>xn(dh(i,e,t)),Pt=(i,e,t,n=!1)=>t===0?i.getFloat32(e*4,!0):t===1||t===2&&!n?i.getUint16(e*2,!0):i.getUint8(e,!0),E1=(function(){const i=e=>e;return function(e,t,n,s=!1){if(t===n)return e;let r=i;return t===2&&s?n===1?r=T1:n==0&&(r=dh):t===2||t===1?n===0?r=fh:n==2&&(s?r=ox:r=i):t===0&&(n===1?r=xn:n==2&&(s?r=Ba:r=xn)),r(e)}})(),So=(i,e,t,n,s=0)=>{const r=new Uint8Array(i,e),o=new Uint8Array(t,n);for(let a=0;a<s;a++)o[a]=r[a]};class J{static CurrentMajorVersion=0;static CurrentMinorVersion=1;static CenterComponentCount=3;static ScaleComponentCount=3;static RotationComponentCount=4;static ColorComponentCount=4;static CovarianceComponentCount=6;static SplatScaleOffsetFloat=3;static SplatRotationOffsetFloat=6;static CompressionLevels={0:{BytesPerCenter:12,BytesPerScale:12,BytesPerRotation:16,BytesPerColor:4,ScaleOffsetBytes:12,RotationffsetBytes:24,ColorOffsetBytes:40,SphericalHarmonicsOffsetBytes:44,ScaleRange:1,BytesPerSphericalHarmonicsComponent:4,SphericalHarmonicsOffsetFloat:11,SphericalHarmonicsDegrees:{0:{BytesPerSplat:44},1:{BytesPerSplat:80},2:{BytesPerSplat:140}}},1:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:2,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:42},2:{BytesPerSplat:72}}},2:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:1,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:33},2:{BytesPerSplat:48}}}};static CovarianceSizeFloats=6;static HeaderSizeBytes=4096;static SectionHeaderSizeBytes=1024;static BucketStorageSizeBytes=12;static BucketStorageSizeFloats=3;static BucketBlockSize=5;static BucketSize=256;constructor(e,t=!0){this.constructFromBuffer(e,t)}getSplatCount(){return this.splatCount}getMaxSplatCount(){return this.maxSplatCount}getMinSphericalHarmonicsDegree(){let e=0;for(let t=0;t<this.sections.length;t++){const n=this.sections[t];(t===0||n.sphericalHarmonicsDegree<e)&&(e=n.sphericalHarmonicsDegree)}return e}getBucketIndex(e,t){let n;const s=e.fullBucketCount*e.bucketSize;if(t<s)n=Math.floor(t/e.bucketSize);else{let r=s;n=e.fullBucketCount;let o=0;for(;r<e.splatCount;){let a=e.partiallyFilledBucketLengths[o];if(t>=r&&t<r+a)break;r+=a,n++,o++}}return n}getSplatCenter(e,t,n){const s=this.globalSplatIndexToSectionMap[e],r=this.sections[s],o=e-r.splatCountOffset,a=r.bytesPerSplat*o,l=new DataView(this.bufferData,r.dataBase+a),c=Pt(l,0,this.compressionLevel),u=Pt(l,1,this.compressionLevel),f=Pt(l,2,this.compressionLevel);if(this.compressionLevel>=1){const h=this.getBucketIndex(r,o)*J.BucketStorageSizeFloats,x=r.compressionScaleFactor,p=r.compressionScaleRange;t.x=(c-p)*x+r.bucketArray[h],t.y=(u-p)*x+r.bucketArray[h+1],t.z=(f-p)*x+r.bucketArray[h+2]}else t.x=c,t.y=u,t.z=f;n&&t.applyMatrix4(n)}getSplatScaleAndRotation=(function(){const e=new st,t=new st,n=new st,s=new B,r=new B,o=new Vt;return function(a,l,c,u,f){const d=this.globalSplatIndexToSectionMap[a],h=this.sections[d],x=a-h.splatCountOffset,p=h.bytesPerSplat*x+J.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,g=new DataView(this.bufferData,h.dataBase+p);r.set(Kt(Pt(g,0,this.compressionLevel),this.compressionLevel),Kt(Pt(g,1,this.compressionLevel),this.compressionLevel),Kt(Pt(g,2,this.compressionLevel),this.compressionLevel)),f&&(f.x!==void 0&&(r.x=f.x),f.y!==void 0&&(r.y=f.y),f.z!==void 0&&(r.z=f.z)),o.set(Kt(Pt(g,4,this.compressionLevel),this.compressionLevel),Kt(Pt(g,5,this.compressionLevel),this.compressionLevel),Kt(Pt(g,6,this.compressionLevel),this.compressionLevel),Kt(Pt(g,3,this.compressionLevel),this.compressionLevel)),u?(e.makeScale(r.x,r.y,r.z),t.makeRotationFromQuaternion(o),n.copy(e).multiply(t).multiply(u),n.decompose(s,c,l)):(l.copy(r),c.copy(o))}})();getSplatColor(e,t){const n=this.globalSplatIndexToSectionMap[e],s=this.sections[n],r=e-s.splatCountOffset,o=s.bytesPerSplat*r+J.CompressionLevels[this.compressionLevel].ColorOffsetBytes,a=new Uint8Array(this.bufferData,s.dataBase+o,4);t.set(a[0],a[1],a[2],a[3])}fillSplatCenterArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);const a=new B;for(let l=n;l<=s;l++){const c=this.globalSplatIndexToSectionMap[l],u=this.sections[c],f=l-u.splatCountOffset,d=(l-n+r)*J.CenterComponentCount,h=u.bytesPerSplat*f,x=new DataView(this.bufferData,u.dataBase+h),p=Pt(x,0,this.compressionLevel),g=Pt(x,1,this.compressionLevel),m=Pt(x,2,this.compressionLevel);if(this.compressionLevel>=1){const A=this.getBucketIndex(u,f)*J.BucketStorageSizeFloats,v=u.compressionScaleFactor,S=u.compressionScaleRange;a.x=(p-S)*v+u.bucketArray[A],a.y=(g-S)*v+u.bucketArray[A+1],a.z=(m-S)*v+u.bucketArray[A+2]}else a.x=p,a.y=g,a.z=m;t&&a.applyMatrix4(t),e[d]=a.x,e[d+1]=a.y,e[d+2]=a.z}}fillSplatScaleRotationArray=(function(){const e=new st,t=new st,n=new st,s=new B,r=new Vt,o=new B,a=l=>{const c=l.w<0?-1:1;l.x*=c,l.y*=c,l.z*=c,l.w*=c};return function(l,c,u,f,d,h,x,p){const g=this.splatCount;f=f||0,d=d||g-1,h===void 0&&(h=f);const m=(_,A)=>E1(_,A,x);for(let _=f;_<=d;_++){const A=this.globalSplatIndexToSectionMap[_],v=this.sections[A],S=_-v.splatCountOffset,y=v.bytesPerSplat*S+J.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,M=(_-f+h)*J.ScaleComponentCount,E=(_-f+h)*J.RotationComponentCount,b=new DataView(this.bufferData,v.dataBase+y),C=p&&p.x!==void 0?p.x:Pt(b,0,this.compressionLevel),D=p&&p.y!==void 0?p.y:Pt(b,1,this.compressionLevel),F=p&&p.z!==void 0?p.z:Pt(b,2,this.compressionLevel),O=Pt(b,3,this.compressionLevel),z=Pt(b,4,this.compressionLevel),V=Pt(b,5,this.compressionLevel),H=Pt(b,6,this.compressionLevel);s.set(Kt(C,this.compressionLevel),Kt(D,this.compressionLevel),Kt(F,this.compressionLevel)),r.set(Kt(z,this.compressionLevel),Kt(V,this.compressionLevel),Kt(H,this.compressionLevel),Kt(O,this.compressionLevel)).normalize(),u&&(o.set(0,0,0),e.makeScale(s.x,s.y,s.z),t.makeRotationFromQuaternion(r),n.identity().premultiply(e).premultiply(t),n.premultiply(u),n.decompose(o,r,s),r.normalize()),a(r),l&&(l[M]=m(s.x,0),l[M+1]=m(s.y,0),l[M+2]=m(s.z,0)),c&&(c[E]=m(r.x,0),c[E+1]=m(r.y,0),c[E+2]=m(r.z,0),c[E+3]=m(r.w,0))}}})();static computeCovariance=(function(){const e=new st,t=new lt,n=new lt,s=new lt,r=new lt,o=new lt,a=new lt;return function(l,c,u,f,d=0,h){e.makeScale(l.x,l.y,l.z),t.setFromMatrix4(e),e.makeRotationFromQuaternion(c),n.setFromMatrix4(e),s.copy(n).multiply(t),r.copy(s).transpose().premultiply(s),u&&(o.setFromMatrix4(u),a.copy(o).transpose(),r.multiply(a),r.premultiply(o)),h>=1?(f[d]=xn(r.elements[0]),f[d+1]=xn(r.elements[3]),f[d+2]=xn(r.elements[6]),f[d+3]=xn(r.elements[4]),f[d+4]=xn(r.elements[7]),f[d+5]=xn(r.elements[8])):(f[d]=r.elements[0],f[d+1]=r.elements[3],f[d+2]=r.elements[6],f[d+3]=r.elements[4],f[d+4]=r.elements[7],f[d+5]=r.elements[8])}})();fillSplatCovarianceArray(e,t,n,s,r,o){const a=this.splatCount,l=new B,c=new Vt;n=n||0,s=s||a-1,r===void 0&&(r=n);for(let u=n;u<=s;u++){const f=this.globalSplatIndexToSectionMap[u],d=this.sections[f],h=u-d.splatCountOffset,x=(u-n+r)*J.CovarianceComponentCount,p=d.bytesPerSplat*h+J.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,g=new DataView(this.bufferData,d.dataBase+p);l.set(Kt(Pt(g,0,this.compressionLevel),this.compressionLevel),Kt(Pt(g,1,this.compressionLevel),this.compressionLevel),Kt(Pt(g,2,this.compressionLevel),this.compressionLevel)),c.set(Kt(Pt(g,4,this.compressionLevel),this.compressionLevel),Kt(Pt(g,5,this.compressionLevel),this.compressionLevel),Kt(Pt(g,6,this.compressionLevel),this.compressionLevel),Kt(Pt(g,3,this.compressionLevel),this.compressionLevel)),J.computeCovariance(l,c,t,e,x,o)}}fillSplatColorArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);for(let a=n;a<=s;a++){const l=this.globalSplatIndexToSectionMap[a],c=this.sections[l],u=a-c.splatCountOffset,f=(a-n+r)*J.ColorComponentCount,d=c.bytesPerSplat*u+J.CompressionLevels[this.compressionLevel].ColorOffsetBytes,h=new Uint8Array(this.bufferData,c.dataBase+d);let x=h[3];x=x>=t?x:0,e[f]=h[0],e[f+1]=h[1],e[f+2]=h[2],e[f+3]=x}}fillSphericalHarmonicsArray=(function(){for(let z=0;z<15;z++)new B;const e=new lt,t=new st,n=new B,s=new B,r=new Vt,o=[],a=[],l=[],c=[],u=[],f=[],d=[],h=[],x=[],p=[],g=[],m=[],_=[],A=[],v=[],S=[],y=[],M=[],E=z=>z,b=(z,V,H,q)=>{z[0]=V,z[1]=H,z[2]=q},C=(z,V,H,q,G)=>{z[0]=Pt(V,q,G,!0),z[1]=Pt(V,q+H,G,!0),z[2]=Pt(V,q+H+H,G,!0)},D=(z,V)=>{V[0]=z[0],V[1]=z[1],V[2]=z[2]},F=(z,V,H,q)=>{V[H]=q(z[0]),V[H+1]=q(z[1]),V[H+2]=q(z[2])},O=(z,V,H,q,G)=>(V[0]=Kt(z[0],H,!0,q,G),V[1]=Kt(z[1],H,!0,q,G),V[2]=Kt(z[2],H,!0,q,G),V);return function(z,V,H,q,G,$,fe){const Y=this.splatCount;q=q||0,G=G||Y-1,$===void 0&&($=q),H&&V>=1&&(t.copy(H),t.decompose(n,r,s),r.normalize(),t.makeRotationFromQuaternion(r),e.setFromMatrix4(t),b(o,e.elements[4],-e.elements[7],e.elements[1]),b(a,-e.elements[5],e.elements[8],-e.elements[2]),b(l,e.elements[3],-e.elements[6],e.elements[0]));const we=ke=>ox(ke,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff),ze=ke=>Ba(ke,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff);for(let ke=q;ke<=G;ke++){const We=this.globalSplatIndexToSectionMap[ke],ne=this.sections[We];V=Math.min(V,ne.sphericalHarmonicsDegree);const ue=Uo(V),Se=ke-ne.splatCountOffset,he=ne.bytesPerSplat*Se+J.CompressionLevels[this.compressionLevel].SphericalHarmonicsOffsetBytes,Ee=new DataView(this.bufferData,ne.dataBase+he),Ze=(ke-q+$)*ue;let U=H?0:this.compressionLevel,N=E;U!==fe&&(U===1?fe===0?N=fh:fe==2&&(N=we):U===0&&(fe===1?N=xn:fe==2&&(N=ze)));const K=this.minSphericalHarmonicsCoeff,R=this.maxSphericalHarmonicsCoeff;V>=1&&(C(x,Ee,3,0,this.compressionLevel),C(p,Ee,3,1,this.compressionLevel),C(g,Ee,3,2,this.compressionLevel),H?(O(x,x,this.compressionLevel,K,R),O(p,p,this.compressionLevel,K,R),O(g,g,this.compressionLevel,K,R),J.rotateSphericalHarmonics3(x,p,g,o,a,l,A,v,S)):(D(x,A),D(p,v),D(g,S)),F(A,z,Ze,N),F(v,z,Ze+3,N),F(S,z,Ze+6,N),V>=2&&(C(x,Ee,5,9,this.compressionLevel),C(p,Ee,5,10,this.compressionLevel),C(g,Ee,5,11,this.compressionLevel),C(m,Ee,5,12,this.compressionLevel),C(_,Ee,5,13,this.compressionLevel),H?(O(x,x,this.compressionLevel,K,R),O(p,p,this.compressionLevel,K,R),O(g,g,this.compressionLevel,K,R),O(m,m,this.compressionLevel,K,R),O(_,_,this.compressionLevel,K,R),J.rotateSphericalHarmonics5(x,p,g,m,_,o,a,l,c,u,f,d,h,A,v,S,y,M)):(D(x,A),D(p,v),D(g,S),D(m,y),D(_,M)),F(A,z,Ze+9,N),F(v,z,Ze+12,N),F(S,z,Ze+15,N),F(y,z,Ze+18,N),F(M,z,Ze+21,N)))}}})();static dot3=(e,t,n,s,r)=>{r[0]=r[1]=r[2]=0;const o=s[0],a=s[1],l=s[2];J.addInto3(e[0]*o,e[1]*o,e[2]*o,r),J.addInto3(t[0]*a,t[1]*a,t[2]*a,r),J.addInto3(n[0]*l,n[1]*l,n[2]*l,r)};static addInto3=(e,t,n,s)=>{s[0]=s[0]+e,s[1]=s[1]+t,s[2]=s[2]+n};static dot5=(e,t,n,s,r,o,a)=>{a[0]=a[1]=a[2]=0;const l=o[0],c=o[1],u=o[2],f=o[3],d=o[4];J.addInto3(e[0]*l,e[1]*l,e[2]*l,a),J.addInto3(t[0]*c,t[1]*c,t[2]*c,a),J.addInto3(n[0]*u,n[1]*u,n[2]*u,a),J.addInto3(s[0]*f,s[1]*f,s[2]*f,a),J.addInto3(r[0]*d,r[1]*d,r[2]*d,a)};static rotateSphericalHarmonics3=(e,t,n,s,r,o,a,l,c)=>{J.dot3(e,t,n,s,a),J.dot3(e,t,n,r,l),J.dot3(e,t,n,o,c)};static rotateSphericalHarmonics5=(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g,m,_)=>{const A=Math.sqrt(.25),v=Math.sqrt(3/4),S=Math.sqrt(1/3),y=Math.sqrt(4/3),M=Math.sqrt(1/12);c[0]=A*(l[2]*o[0]+l[0]*o[2]+(o[2]*l[0]+o[0]*l[2])),c[1]=l[1]*o[0]+o[1]*l[0],c[2]=v*(l[1]*o[1]+o[1]*l[1]),c[3]=l[1]*o[2]+o[1]*l[2],c[4]=A*(l[2]*o[2]-l[0]*o[0]+(o[2]*l[2]-o[0]*l[0])),J.dot5(e,t,n,s,r,c,x),u[0]=A*(a[2]*o[0]+a[0]*o[2]+(o[2]*a[0]+o[0]*a[2])),u[1]=a[1]*o[0]+o[1]*a[0],u[2]=v*(a[1]*o[1]+o[1]*a[1]),u[3]=a[1]*o[2]+o[1]*a[2],u[4]=A*(a[2]*o[2]-a[0]*o[0]+(o[2]*a[2]-o[0]*a[0])),J.dot5(e,t,n,s,r,u,p),f[0]=S*(a[2]*a[0]+a[0]*a[2])+-M*(l[2]*l[0]+l[0]*l[2]+(o[2]*o[0]+o[0]*o[2])),f[1]=y*a[1]*a[0]+-S*(l[1]*l[0]+o[1]*o[0]),f[2]=a[1]*a[1]+-A*(l[1]*l[1]+o[1]*o[1]),f[3]=y*a[1]*a[2]+-S*(l[1]*l[2]+o[1]*o[2]),f[4]=S*(a[2]*a[2]-a[0]*a[0])+-M*(l[2]*l[2]-l[0]*l[0]+(o[2]*o[2]-o[0]*o[0])),J.dot5(e,t,n,s,r,f,g),d[0]=A*(a[2]*l[0]+a[0]*l[2]+(l[2]*a[0]+l[0]*a[2])),d[1]=a[1]*l[0]+l[1]*a[0],d[2]=v*(a[1]*l[1]+l[1]*a[1]),d[3]=a[1]*l[2]+l[1]*a[2],d[4]=A*(a[2]*l[2]-a[0]*l[0]+(l[2]*a[2]-l[0]*a[0])),J.dot5(e,t,n,s,r,d,m),h[0]=A*(l[2]*l[0]+l[0]*l[2]-(o[2]*o[0]+o[0]*o[2])),h[1]=l[1]*l[0]-o[1]*o[0],h[2]=v*(l[1]*l[1]-o[1]*o[1]),h[3]=l[1]*l[2]-o[1]*o[2],h[4]=A*(l[2]*l[2]-l[0]*l[0]-(o[2]*o[2]-o[0]*o[0])),J.dot5(e,t,n,s,r,h,_)};static parseHeader(e){const t=new Uint8Array(e,0,J.HeaderSizeBytes),n=new Uint16Array(e,0,J.HeaderSizeBytes/2),s=new Uint32Array(e,0,J.HeaderSizeBytes/4),r=new Float32Array(e,0,J.HeaderSizeBytes/4),o=t[0],a=t[1],l=s[1],c=s[2],u=s[3],f=s[4],d=n[10],h=new B(r[6],r[7],r[8]),x=r[9]||-Xs,p=r[10]||Xs;return{versionMajor:o,versionMinor:a,maxSectionCount:l,sectionCount:c,maxSplatCount:u,splatCount:f,compressionLevel:d,sceneCenter:h,minSphericalHarmonicsCoeff:x,maxSphericalHarmonicsCoeff:p}}static writeHeaderCountsToBuffer(e,t,n){const s=new Uint32Array(n,0,J.HeaderSizeBytes/4);s[2]=e,s[4]=t}static writeHeaderToBuffer(e,t){const n=new Uint8Array(t,0,J.HeaderSizeBytes),s=new Uint16Array(t,0,J.HeaderSizeBytes/2),r=new Uint32Array(t,0,J.HeaderSizeBytes/4),o=new Float32Array(t,0,J.HeaderSizeBytes/4);n[0]=e.versionMajor,n[1]=e.versionMinor,n[2]=0,n[3]=0,r[1]=e.maxSectionCount,r[2]=e.sectionCount,r[3]=e.maxSplatCount,r[4]=e.splatCount,s[10]=e.compressionLevel,o[6]=e.sceneCenter.x,o[7]=e.sceneCenter.y,o[8]=e.sceneCenter.z,o[9]=e.minSphericalHarmonicsCoeff||-Xs,o[10]=e.maxSphericalHarmonicsCoeff||Xs}static parseSectionHeaders(e,t,n=0,s){const r=e.compressionLevel,o=e.maxSectionCount,a=new Uint16Array(t,n,o*J.SectionHeaderSizeBytes/2),l=new Uint32Array(t,n,o*J.SectionHeaderSizeBytes/4),c=new Float32Array(t,n,o*J.SectionHeaderSizeBytes/4),u=[];let f=0,d=f/2,h=f/4,x=J.HeaderSizeBytes+e.maxSectionCount*J.SectionHeaderSizeBytes,p=0;for(let g=0;g<o;g++){const m=l[h+1],_=l[h+2],A=l[h+3],v=c[h+4],S=v/2,y=a[d+10],M=l[h+6]||J.CompressionLevels[r].ScaleRange,E=l[h+8],b=l[h+9],C=b*4,D=y*A+C,F=a[d+20],{bytesPerSplat:O}=J.calculateComponentStorage(r,F),z=O*m,V=z+D,H={bytesPerSplat:O,splatCountOffset:p,splatCount:s?m:0,maxSplatCount:m,bucketSize:_,bucketCount:A,bucketBlockSize:v,halfBucketBlockSize:S,bucketStorageSizeBytes:y,bucketsStorageSizeBytes:D,splatDataStorageSizeBytes:z,storageSizeBytes:V,compressionScaleRange:M,compressionScaleFactor:S/M,base:x,bucketsBase:x+C,dataBase:x+D,fullBucketCount:E,partiallyFilledBucketCount:b,sphericalHarmonicsDegree:F};u[g]=H,x+=V,f+=J.SectionHeaderSizeBytes,d=f/2,h=f/4,p+=m}return u}static writeSectionHeaderToBuffer(e,t,n,s=0){const r=new Uint16Array(n,s,J.SectionHeaderSizeBytes/2),o=new Uint32Array(n,s,J.SectionHeaderSizeBytes/4),a=new Float32Array(n,s,J.SectionHeaderSizeBytes/4);o[0]=e.splatCount,o[1]=e.maxSplatCount,o[2]=t>=1?e.bucketSize:0,o[3]=t>=1?e.bucketCount:0,a[4]=t>=1?e.bucketBlockSize:0,r[10]=t>=1?J.BucketStorageSizeBytes:0,o[6]=t>=1?e.compressionScaleRange:0,o[7]=e.storageSizeBytes,o[8]=t>=1?e.fullBucketCount:0,o[9]=t>=1?e.partiallyFilledBucketCount:0,r[20]=e.sphericalHarmonicsDegree}static writeSectionHeaderSplatCountToBuffer(e,t,n=0){const s=new Uint32Array(t,n,J.SectionHeaderSizeBytes/4);s[0]=e}constructFromBuffer(e,t){this.bufferData=e,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSectionMap=[];const n=J.parseHeader(this.bufferData);this.versionMajor=n.versionMajor,this.versionMinor=n.versionMinor,this.maxSectionCount=n.maxSectionCount,this.sectionCount=t?n.maxSectionCount:0,this.maxSplatCount=n.maxSplatCount,this.splatCount=t?n.maxSplatCount:0,this.compressionLevel=n.compressionLevel,this.sceneCenter=new B().copy(n.sceneCenter),this.minSphericalHarmonicsCoeff=n.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff=n.maxSphericalHarmonicsCoeff,this.sections=J.parseSectionHeaders(n,this.bufferData,J.HeaderSizeBytes,t),this.linkBufferArrays(),this.buildMaps()}static calculateComponentStorage(e,t){const n=J.CompressionLevels[e].BytesPerCenter,s=J.CompressionLevels[e].BytesPerScale,r=J.CompressionLevels[e].BytesPerRotation,o=J.CompressionLevels[e].BytesPerColor,a=Uo(t),l=J.CompressionLevels[e].BytesPerSphericalHarmonicsComponent*a,c=n+s+r+o+l;return{bytesPerCenter:n,bytesPerScale:s,bytesPerRotation:r,bytesPerColor:o,sphericalHarmonicsComponentsPerSplat:a,sphericalHarmonicsBytesPerSplat:l,bytesPerSplat:c}}linkBufferArrays(){for(let e=0;e<this.maxSectionCount;e++){const t=this.sections[e];t.bucketArray=new Float32Array(this.bufferData,t.bucketsBase,t.bucketCount*J.BucketStorageSizeFloats),t.partiallyFilledBucketCount>0&&(t.partiallyFilledBucketLengths=new Uint32Array(this.bufferData,t.base,t.partiallyFilledBucketCount))}}buildMaps(){let e=0;for(let t=0;t<this.maxSectionCount;t++){const n=this.sections[t];for(let s=0;s<n.maxSplatCount;s++){const r=e+s;this.globalSplatIndexToLocalSplatIndexMap[r]=s,this.globalSplatIndexToSectionMap[r]=t}e+=n.maxSplatCount}}updateLoadedCounts(e,t){J.writeHeaderCountsToBuffer(e,t,this.bufferData),this.sectionCount=e,this.splatCount=t}updateSectionLoadedCounts(e,t){const n=J.HeaderSizeBytes+J.SectionHeaderSizeBytes*e;J.writeSectionHeaderSplatCountToBuffer(t,this.bufferData,n),this.sections[e].splatCount=t}static writeSplatDataToSectionBuffer=(function(){const e=new ArrayBuffer(12),t=new ArrayBuffer(12),n=new ArrayBuffer(16),s=new ArrayBuffer(4),r=new ArrayBuffer(256),o=new Vt,a=new B,l=new B,{X:c,Y:u,Z:f,SCALE0:d,SCALE1:h,SCALE2:x,ROTATION0:p,ROTATION1:g,ROTATION2:m,ROTATION3:_,FDC0:A,FDC1:v,FDC2:S,OPACITY:y,FRC0:M,FRC9:E}=Pe.OFFSET,b=(C,D,F)=>{const O=F*2+1;return C=Math.round(C*D)+F,$t(C,0,O)};return function(C,D,F,O,z,V,H,q,G=-Xs,$=Xs){const fe=Uo(z),Y=J.CompressionLevels[O].BytesPerCenter,we=J.CompressionLevels[O].BytesPerScale,ze=J.CompressionLevels[O].BytesPerRotation,ke=J.CompressionLevels[O].BytesPerColor,We=F,ne=We+Y,ue=ne+we,Se=ue+ze,he=Se+ke;if(C[p]!==void 0?(o.set(C[p],C[g],C[m],C[_]),o.normalize()):o.set(1,0,0,0),C[d]!==void 0?a.set(C[d]||0,C[h]||0,C[x]||0):a.set(0,0,0),O===0){const Ze=new Float32Array(D,We,J.CenterComponentCount),U=new Float32Array(D,ue,J.RotationComponentCount),N=new Float32Array(D,ne,J.ScaleComponentCount);if(U.set([o.x,o.y,o.z,o.w]),N.set([a.x,a.y,a.z]),Ze.set([C[c],C[u],C[f]]),z>0){const K=new Float32Array(D,he,fe);if(z>=1){for(let R=0;R<9;R++)K[R]=C[M+R]||0;if(z>=2)for(let R=0;R<15;R++)K[R+9]=C[E+R]||0}}}else{const Ze=new Uint16Array(e,0,J.CenterComponentCount),U=new Uint16Array(n,0,J.RotationComponentCount),N=new Uint16Array(t,0,J.ScaleComponentCount);if(U.set([xn(o.x),xn(o.y),xn(o.z),xn(o.w)]),N.set([xn(a.x),xn(a.y),xn(a.z)]),l.set(C[c],C[u],C[f]).sub(V),l.x=b(l.x,H,q),l.y=b(l.y,H,q),l.z=b(l.z,H,q),Ze.set([l.x,l.y,l.z]),z>0){const K=O===1?Uint16Array:Uint8Array,R=O===1?2:1,te=new K(r,0,fe);if(z>=1){for(let pe=0;pe<9;pe++){const ie=C[M+pe]||0;te[pe]=O===1?xn(ie):Ba(ie,G,$)}const oe=9*R;if(So(te.buffer,0,D,he,oe),z>=2){for(let pe=0;pe<15;pe++){const ie=C[E+pe]||0;te[pe+9]=O===1?xn(ie):Ba(ie,G,$)}So(te.buffer,oe,D,he+oe,15*R)}}}So(Ze.buffer,0,D,We,6),So(N.buffer,0,D,ne,6),So(U.buffer,0,D,ue,8)}const Ee=new Uint8ClampedArray(s,0,4);Ee.set([C[A]||0,C[v]||0,C[S]||0]),Ee[3]=C[y]||0,So(Ee.buffer,0,D,Se,4)}})();static generateFromUncompressedSplatArrays(e,t,n,s,r,o,a=[]){let l=0;for(let S=0;S<e.length;S++){const y=e[S];l=Math.max(y.sphericalHarmonicsDegree,l)}let c,u;for(let S=0;S<e.length;S++){const y=e[S];for(let M=0;M<y.splats.length;M++){const E=y.splats[M];for(let b=Pe.OFFSET.FRC0;b<Pe.OFFSET.FRC23&&b<E.length;b++)(!c||E[b]<c)&&(c=E[b]),(!u||E[b]>u)&&(u=E[b])}}c=c||-Xs,u=u||Xs;const{bytesPerSplat:f}=J.calculateComponentStorage(n,l),d=J.CompressionLevels[n].ScaleRange,h=[],x=[];let p=0;for(let S=0;S<e.length;S++){const y=e[S],M=new Pe(l);for(let We=0;We<y.splatCount;We++){const ne=y.splats[We];(ne[Pe.OFFSET.OPACITY]||0)>=t&&M.addSplat(ne)}const E=a[S]||{},b=(E.blockSizeFactor||1)*(r||J.BucketBlockSize),C=Math.ceil((E.bucketSizeFactor||1)*(o||J.BucketSize)),D=J.computeBucketsForUncompressedSplatArray(M,b,C),F=D.fullBuckets.length,O=D.partiallyFullBuckets.map(We=>We.splats.length),z=O.length,V=[...D.fullBuckets,...D.partiallyFullBuckets],H=M.splats.length*f,q=z*4,G=n>=1?V.length*J.BucketStorageSizeBytes+q:0,$=H+G,fe=new ArrayBuffer($),Y=d/(b*.5),we=new B;let ze=0;for(let We=0;We<V.length;We++){const ne=V[We];we.fromArray(ne.center);for(let ue=0;ue<ne.splats.length;ue++){let Se=ne.splats[ue];const he=M.splats[Se],Ee=G+ze*f;J.writeSplatDataToSectionBuffer(he,fe,Ee,n,l,we,Y,d,c,u),ze++}}if(p+=ze,n>=1){const We=new Uint32Array(fe,0,O.length*4);for(let ue=0;ue<O.length;ue++)We[ue]=O[ue];const ne=new Float32Array(fe,q,V.length*J.BucketStorageSizeFloats);for(let ue=0;ue<V.length;ue++){const Se=V[ue],he=ue*3;ne[he]=Se.center[0],ne[he+1]=Se.center[1],ne[he+2]=Se.center[2]}}h.push(fe);const ke=new ArrayBuffer(J.SectionHeaderSizeBytes);J.writeSectionHeaderToBuffer({maxSplatCount:ze,splatCount:ze,bucketSize:C,bucketCount:V.length,bucketBlockSize:b,compressionScaleRange:d,storageSizeBytes:$,fullBucketCount:F,partiallyFilledBucketCount:z,sphericalHarmonicsDegree:l},n,ke,0),x.push(ke)}let g=0;for(let S of h)g+=S.byteLength;const m=J.HeaderSizeBytes+J.SectionHeaderSizeBytes*h.length+g,_=new ArrayBuffer(m);J.writeHeaderToBuffer({versionMajor:0,versionMinor:1,maxSectionCount:h.length,sectionCount:h.length,maxSplatCount:p,splatCount:p,compressionLevel:n,sceneCenter:s,minSphericalHarmonicsCoeff:c,maxSphericalHarmonicsCoeff:u},_);let A=J.HeaderSizeBytes;for(let S of x)new Uint8Array(_,A,J.SectionHeaderSizeBytes).set(new Uint8Array(S)),A+=J.SectionHeaderSizeBytes;for(let S of h)new Uint8Array(_,A,S.byteLength).set(new Uint8Array(S)),A+=S.byteLength;return new J(_)}static computeBucketsForUncompressedSplatArray(e,t,n){let s=e.splatCount;const r=t/2,o=new B,a=new B;for(let p=0;p<s;p++){const g=e.splats[p],m=[g[Pe.OFFSET.X],g[Pe.OFFSET.Y],g[Pe.OFFSET.Z]];(p===0||m[0]<o.x)&&(o.x=m[0]),(p===0||m[0]>a.x)&&(a.x=m[0]),(p===0||m[1]<o.y)&&(o.y=m[1]),(p===0||m[1]>a.y)&&(a.y=m[1]),(p===0||m[2]<o.z)&&(o.z=m[2]),(p===0||m[2]>a.z)&&(a.z=m[2])}const l=new B().copy(a).sub(o),c=Math.ceil(l.y/t),u=Math.ceil(l.z/t),f=new B,d=[],h={};for(let p=0;p<s;p++){const g=e.splats[p],m=[g[Pe.OFFSET.X],g[Pe.OFFSET.Y],g[Pe.OFFSET.Z]],_=Math.floor((m[0]-o.x)/t),A=Math.floor((m[1]-o.y)/t),v=Math.floor((m[2]-o.z)/t);f.x=_*t+o.x+r,f.y=A*t+o.y+r,f.z=v*t+o.z+r;const S=_*(c*u)+A*u+v;let y=h[S];y||(h[S]=y={splats:[],center:f.toArray()}),y.splats.push(p),y.splats.length>=n&&(d.push(y),h[S]=null)}const x=[];for(let p in h)if(h.hasOwnProperty(p)){const g=h[p];g&&x.push(g)}return{fullBuckets:d,partiallyFullBuckets:x}}static preallocateUncompressed(e,t){const n=J.CompressionLevels[0].SphericalHarmonicsDegrees[t],s=J.HeaderSizeBytes+J.SectionHeaderSizeBytes,r=s+n.BytesPerSplat*e,o=new ArrayBuffer(r);return J.writeHeaderToBuffer({versionMajor:J.CurrentMajorVersion,versionMinor:J.CurrentMinorVersion,maxSectionCount:1,sectionCount:1,maxSplatCount:e,splatCount:e,compressionLevel:0,sceneCenter:new B},o),J.writeSectionHeaderToBuffer({maxSplatCount:e,splatCount:e,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:t},0,o,J.HeaderSizeBytes),{splatBuffer:new J(o,!0),splatBufferDataOffsetBytes:s}}}const Em=new Uint8Array([112,108,121,10]),wm=new Uint8Array([10,101,110,100,95,104,101,97,100,101,114,10]),ju="end_header",$u=new Map([["char",Int8Array],["uchar",Uint8Array],["short",Int16Array],["ushort",Uint16Array],["int",Int32Array],["uint",Uint32Array],["float",Float32Array],["double",Float64Array]]),es=(i,e)=>{const t=(1<<e)-1;return(i&t)/t},Rm=(i,e)=>{i.x=es(e>>>21,11),i.y=es(e>>>11,10),i.z=es(e,11)},w1=(i,e)=>{i.x=es(e>>>24,8),i.y=es(e>>>16,8),i.z=es(e>>>8,8),i.w=es(e,8)},R1=(i,e)=>{const t=1/(Math.sqrt(2)*.5),n=(es(e>>>20,10)-.5)*t,s=(es(e>>>10,10)-.5)*t,r=(es(e,10)-.5)*t,o=Math.sqrt(1-(n*n+s*s+r*r));switch(e>>>30){case 0:i.set(o,n,s,r);break;case 1:i.set(n,o,s,r);break;case 2:i.set(n,s,o,r);break;case 3:i.set(n,s,r,o);break}},As=(i,e,t)=>i*(1-t)+e*t,en=(i,e)=>i.properties.find(t=>t.name===e&&t.storage)?.storage;class Tt{static decodeHeaderText(e){let t,n,s,r;const o=e.split(`
`).filter(f=>!f.startsWith("comment "));let a=0,l=!1;for(let f=1;f<o.length;++f){const d=o[f].split(" ");switch(d[0]){case"format":if(d[1]!=="binary_little_endian")throw new Error("Unsupported ply format");break;case"element":t={name:d[1],count:parseInt(d[2],10),properties:[],storageSizeBytes:0},t.name==="chunk"?n=t:t.name==="vertex"?s=t:t.name==="sh"&&(r=t);break;case"property":{if(!$u.has(d[1]))throw new Error(`Unrecognized property data type '${d[1]}' in ply header`);const h=$u.get(d[1]),x=h.BYTES_PER_ELEMENT*t.count;t.name==="vertex"&&(a+=h.BYTES_PER_ELEMENT),t.properties.push({type:d[1],name:d[2],storage:null,byteSize:h.BYTES_PER_ELEMENT,storageSizeByes:x}),t.storageSizeBytes+=x;break}case ju:l=!0;break;default:throw new Error(`Unrecognized header value '${d[0]}' in ply header`)}if(l)break}let c=0,u=0;return r&&(u=r.properties.length,r.properties.length>=45?c=3:r.properties.length>=24?c=2:r.properties.length>=9&&(c=1)),{chunkElement:n,vertexElement:s,shElement:r,bytesPerSplat:a,headerSizeBytes:e.indexOf(ju)+ju.length+1,sphericalHarmonicsDegree:c,sphericalHarmonicsPerSplat:u}}static decodeHeader(e){const t=(h,x)=>{const p=h.length-x.length;let g,m;for(g=0;g<=p;++g){for(m=0;m<x.length&&h[g+m]===x[m];++m);if(m===x.length)return g}return-1},n=(h,x)=>{if(h.length<x.length)return!1;for(let p=0;p<x.length;++p)if(h[p]!==x[p])return!1;return!0};let s=new Uint8Array(e),r;if(s.length>=Em.length&&!n(s,Em))throw new Error("Invalid PLY header");if(r=t(s,wm),r===-1)throw new Error("End of PLY header not found");const o=new TextDecoder("ascii").decode(s.slice(0,r)),{chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f,bytesPerSplat:d}=Tt.decodeHeaderText(o);return{headerSizeBytes:r+wm.length,bytesPerSplat:d,chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f}}static readElementData(e,t,n,s,r,o=null){let a=t instanceof DataView?t:new DataView(t);s=s||0,r=r||e.count-1;for(let l=s;l<=r;++l)for(let c=0;c<e.properties.length;++c){const u=e.properties[c],f=$u.get(u.type),d=f.BYTES_PER_ELEMENT*e.count;if((!u.storage||u.storage.byteLength<d)&&(!o||o(u.name))&&(u.storage=new f(e.count)),u.storage)switch(u.type){case"char":u.storage[l]=a.getInt8(n);break;case"uchar":u.storage[l]=a.getUint8(n);break;case"short":u.storage[l]=a.getInt16(n,!0);break;case"ushort":u.storage[l]=a.getUint16(n,!0);break;case"int":u.storage[l]=a.getInt32(n,!0);break;case"uint":u.storage[l]=a.getUint32(n,!0);break;case"float":u.storage[l]=a.getFloat32(n,!0);break;case"double":u.storage[l]=a.getFloat64(n,!0);break}n+=u.byteSize}return n}static readPly(e,t=null){const n=Tt.decodeHeader(e);let s=Tt.readElementData(n.chunkElement,e,n.headerSizeBytes,null,null,t);return s=Tt.readElementData(n.vertexElement,e,s,null,null,t),Tt.readElementData(n.shElement,e,s,null,null,t),{chunkElement:n.chunkElement,vertexElement:n.vertexElement,shElement:n.shElement,sphericalHarmonicsDegree:n.sphericalHarmonicsDegree,sphericalHarmonicsPerSplat:n.sphericalHarmonicsPerSplat}}static getElementStorageArrays(e,t,n){const s={};if(t){const r=en(e,"min_r"),o=en(e,"min_g"),a=en(e,"min_b"),l=en(e,"max_r"),c=en(e,"max_g"),u=en(e,"max_b"),f=en(e,"min_x"),d=en(e,"min_y"),h=en(e,"min_z"),x=en(e,"max_x"),p=en(e,"max_y"),g=en(e,"max_z"),m=en(e,"min_scale_x"),_=en(e,"min_scale_y"),A=en(e,"min_scale_z"),v=en(e,"max_scale_x"),S=en(e,"max_scale_y"),y=en(e,"max_scale_z"),M=en(t,"packed_position"),E=en(t,"packed_rotation"),b=en(t,"packed_scale"),C=en(t,"packed_color");s.colorExtremes={minR:r,maxR:l,minG:o,maxG:c,minB:a,maxB:u},s.positionExtremes={minX:f,maxX:x,minY:d,maxY:p,minZ:h,maxZ:g},s.scaleExtremes={minScaleX:m,maxScaleX:v,minScaleY:_,maxScaleY:S,minScaleZ:A,maxScaleZ:y},s.position=M,s.rotation=E,s.scale=b,s.color=C}if(n){const r={};for(let o=0;o<45;o++){const a=`f_rest_${o}`,l=en(n,a);if(l)r[a]=l;else break}s.sh=r}return s}static decompressBaseSplat=(function(){const e=new B,t=new Vt,n=new B,s=new Jt,r=Pe.OFFSET;return function(o,a,l,c,u,f,d,h,x,p){p=p||Pe.createSplat();const g=Math.floor((a+o)/256);return Rm(e,l[o]),R1(t,d[o]),Rm(n,u[o]),w1(s,x[o]),p[r.X]=As(c.minX[g],c.maxX[g],e.x),p[r.Y]=As(c.minY[g],c.maxY[g],e.y),p[r.Z]=As(c.minZ[g],c.maxZ[g],e.z),p[r.ROTATION0]=t.x,p[r.ROTATION1]=t.y,p[r.ROTATION2]=t.z,p[r.ROTATION3]=t.w,p[r.SCALE0]=Math.exp(As(f.minScaleX[g],f.maxScaleX[g],n.x)),p[r.SCALE1]=Math.exp(As(f.minScaleY[g],f.maxScaleY[g],n.y)),p[r.SCALE2]=Math.exp(As(f.minScaleZ[g],f.maxScaleZ[g],n.z)),h.minR&&h.maxR?p[r.FDC0]=$t(Math.round(As(h.minR[g],h.maxR[g],s.x)*255),0,255):p[r.FDC0]=$t(Math.floor(s.x*255),0,255),h.minG&&h.maxG?p[r.FDC1]=$t(Math.round(As(h.minG[g],h.maxG[g],s.y)*255),0,255):p[r.FDC1]=$t(Math.floor(s.y*255),0,255),h.minB&&h.maxB?p[r.FDC2]=$t(Math.round(As(h.minB[g],h.maxB[g],s.z)*255),0,255):p[r.FDC2]=$t(Math.floor(s.z*255),0,255),p[r.OPACITY]=$t(Math.floor(s.w*255),0,255),p}})();static decompressSphericalHarmonics=(function(){const e=[0,3,8,15],t=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(n,s,r,o,a){a=a||Pe.createSplat();let l=e[r],c=e[o];for(let u=0;u<3;++u)for(let f=0;f<15;++f){const d=t[u*15+f];f<l&&f<c&&(a[Pe.OFFSET.FRC0+d]=s[u*c+f][n]*(8/255)-4)}return a}})();static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l,c=null){Tt.readElementData(t,o,0,n,s,c);const u=J.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,{positionExtremes:f,scaleExtremes:d,colorExtremes:h,position:x,rotation:p,scale:g,color:m}=Tt.getElementStorageArrays(e,t),_=Pe.createSplat();for(let A=n;A<=s;++A){Tt.decompressBaseSplat(A,r,x,f,g,d,p,h,m,_);const v=A*u+l;J.writeSplatDataToSectionBuffer(_,a,v,0,0)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a,l=null){Tt.readElementData(t,o,0,n,s,l);const{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:d,rotation:h,scale:x,color:p}=Tt.getElementStorageArrays(e,t);for(let g=n;g<=s;++g){const m=Pe.createSplat();Tt.decompressBaseSplat(g,r,d,c,x,u,h,f,p,m),a.addSplat(m)}}static parseSphericalHarmonicsToUncompressedSplatArraySection(e,t,n,s,r,o,a,l,c,u=null){Tt.readElementData(t,r,o,n,s,u);const{sh:f}=Tt.getElementStorageArrays(e,void 0,t),d=Object.values(f);for(let h=n;h<=s;++h)Tt.decompressSphericalHarmonics(h,d,a,l,c.splats[h])}static parseToUncompressedSplatArray(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=Tt.readPly(e);t=Math.min(t,o);const a=new Pe(t),{positionExtremes:l,scaleExtremes:c,colorExtremes:u,position:f,rotation:d,scale:h,color:x}=Tt.getElementStorageArrays(n,s);let p;if(t>0){const{sh:g}=Tt.getElementStorageArrays(n,void 0,r);p=Object.values(g)}for(let g=0;g<s.count;++g){a.addDefaultSplat();const m=a.getSplat(a.splatCount-1);Tt.decompressBaseSplat(g,0,f,l,h,c,d,u,x,m),t>0&&Tt.decompressSphericalHarmonics(g,p,t,o,m)}return a}static parseToUncompressedSplatBuffer(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=Tt.readPly(e);t=Math.min(t,o);const{splatBuffer:a,splatBufferDataOffsetBytes:l}=J.preallocateUncompressed(s.count,t),{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:d,rotation:h,scale:x,color:p}=Tt.getElementStorageArrays(n,s);let g;if(t>0){const{sh:A}=Tt.getElementStorageArrays(n,void 0,r);g=Object.values(A)}const m=J.CompressionLevels[0].SphericalHarmonicsDegrees[t].BytesPerSplat,_=Pe.createSplat(t);for(let A=0;A<s.count;++A){Tt.decompressBaseSplat(A,0,d,c,x,u,h,f,p,_),t>0&&Tt.decompressSphericalHarmonics(A,g,t,o,_);const v=A*m+l;J.writeSplatDataToSectionBuffer(_,a.bufferData,v,0,t)}return a}}const Vn={INRIAV1:0,INRIAV2:1,PlayCanvasCompressed:2},[ax,hh,ph,mh,gh,xh,_h]=[0,1,2,3,4,5,6],Im={double:ax,int:hh,uint:ph,float:mh,short:gh,ushort:xh,uchar:_h},I1={[ax]:8,[hh]:4,[ph]:4,[mh]:4,[gh]:2,[xh]:2,[_h]:1};class It{static HeaderEndToken="end_header";static decodeSectionHeader(e,t,n=0){const s=[];let r=!1,o=-1,a=0,l=!1,c=null;const u=[],f=[],d=[],h={};for(let m=n;m<e.length;m++){const _=e[m].trim();if(_.startsWith("element"))if(r){o--;break}else{r=!0,n=m,o=m;const A=_.split(" ");let v=0;for(let S of A){const y=S.trim();y.length>0&&(v++,v===2?c=y:v===3&&(a=parseInt(y)))}}else if(_.startsWith("property")){const A=_.match(/(\w+)\s+(\w+)\s+(\w+)/);if(A){const v=A[2],S=A[3];d.push(S);const y=t[S];h[S]=v;const M=Im[v];y!==void 0&&(u.push(y),f[y]=M)}}if(_===It.HeaderEndToken){l=!0;break}r&&(s.push(_),o++)}const x=[];let p=0;for(let m of d){const _=h[m];if(h.hasOwnProperty(m)){const A=t[m];A!==void 0&&(x[A]=p)}p+=I1[Im[_]]}const g=It.decodeSphericalHarmonicsFromSectionHeader(d,t);return{headerLines:s,headerStartLine:n,headerEndLine:o,fieldTypes:f,fieldIds:u,fieldOffsets:x,bytesPerVertex:p,vertexCount:a,dataSizeBytes:p*a,endOfHeader:l,sectionName:c,sphericalHarmonicsDegree:g.degree,sphericalHarmonicsCoefficientsPerChannel:g.coefficientsPerChannel,sphericalHarmonicsDegree1Fields:g.degree1Fields,sphericalHarmonicsDegree2Fields:g.degree2Fields}}static decodeSphericalHarmonicsFromSectionHeader(e,t){let n=0,s=0;for(let l of e)l.startsWith("f_rest")&&n++;s=n/3;let r=0;s>=3&&(r=1),s>=8&&(r=2);let o=[],a=[];for(let l=0;l<3;l++){if(r>=1)for(let c=0;c<3;c++)o.push(t["f_rest_"+(c+s*l)]);if(r>=2)for(let c=0;c<5;c++)a.push(t["f_rest_"+(c+s*l+3)])}return{degree:r,coefficientsPerChannel:s,degree1Fields:o,degree2Fields:a}}static getHeaderSectionNames(e){const t=[];for(let n of e)if(n.startsWith("element")){const s=n.split(" ");let r=0;for(let o of s){const a=o.trim();a.length>0&&(r++,r===2&&t.push(a))}}return t}static checkTextForEndHeader(e){return!!e.includes(It.HeaderEndToken)}static checkBufferForEndHeader(e,t,n,s){const r=new Uint8Array(e,Math.max(0,t-n),n),o=s.decode(r);return It.checkTextForEndHeader(o)}static extractHeaderFromBufferToText(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,It.checkBufferForEndHeader(e,n,r*2,t))break}return s}static readHeaderFromBuffer(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,It.checkBufferForEndHeader(e,n,r*2,t))break}return s}static convertHeaderTextToLines(e){const t=e.split(`
`),n=[];for(let s=0;s<t.length;s++){const r=t[s].trim();if(n.push(r),r===It.HeaderEndToken)break}return n}static determineHeaderFormatFromHeaderText(e){const t=It.convertHeaderTextToLines(e);let n=Vn.INRIAV1;for(let s=0;s<t.length;s++){const r=t[s].trim();if(r.startsWith("element chunk")||r.match(/[A-Za-z]*packed_[A-Za-z]*/))n=Vn.PlayCanvasCompressed;else if(r.startsWith("element codebook_centers"))n=Vn.INRIAV2;else if(r===It.HeaderEndToken)break}return n}static determineHeaderFormatFromPlyBuffer(e){const t=It.extractHeaderFromBufferToText(e);return It.determineHeaderFormatFromHeaderText(t)}static readVertex(e,t,n,s,r,o,a=!0){const l=n*t.bytesPerVertex+s,c=t.fieldOffsets,u=t.fieldTypes;for(let f of r){const d=u[f];d===mh?o[f]=e.getFloat32(l+c[f],!0):d===gh?o[f]=e.getInt16(l+c[f],!0):d===xh?o[f]=e.getUint16(l+c[f],!0):d===hh?o[f]=e.getInt32(l+c[f],!0):d===ph?o[f]=e.getUint32(l+c[f],!0):d===_h&&(a?o[f]=e.getUint8(l+c[f])/255:o[f]=e.getUint8(l+c[f]))}}}const lx=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0"],D1=lx.map((i,e)=>e),[Dm,P1,F1,L1,B1,U1,O1,N1,z1,k1,Pm,H1,V1,Fm,Lm,G1,W1,X1]=D1;class Cn{static decodeHeaderLines(e){let t=0;e.forEach(u=>{u.includes("f_rest_")&&t++});let n=0;t>=45?n=45:t>=24?n=24:t>=9&&(n=9);let r=Array.from(Array(Math.max(n-1,0))).map((u,f)=>`f_rest_${f+1}`);const o=[...lx,...r],a=o.map((u,f)=>f),l=a.reduce((u,f)=>(u[o[f]]=f,u),{}),c=It.decodeSectionHeader(e,l,0);return c.splatCount=c.vertexCount,c.bytesPerSplat=c.bytesPerVertex,c.fieldsToReadIndexes=a,c}static decodeHeaderText(e){const t=It.convertHeaderTextToLines(e),n=Cn.decodeHeaderLines(t);return n.headerText=e,n.headerSizeBytes=e.indexOf(It.HeaderEndToken)+It.HeaderEndToken.length+1,n}static decodeHeaderFromBuffer(e){const t=It.readHeaderFromBuffer(e);return Cn.decodeHeaderText(t)}static findSplatData(e,t){return new DataView(e,t.headerSizeBytes)}static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l=0){l=Math.min(l,e.sphericalHarmonicsDegree);const c=J.CompressionLevels[0].SphericalHarmonicsDegrees[l].BytesPerSplat;for(let u=t;u<=n;u++){const f=Cn.parseToUncompressedSplat(s,u,e,r,l),d=u*c+a;J.writeSplatDataToSectionBuffer(f,o,d,0,l)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a=0){a=Math.min(a,e.sphericalHarmonicsDegree);for(let l=t;l<=n;l++){const c=Cn.parseToUncompressedSplat(s,l,e,r,a);o.addSplat(c)}}static decodeSectionSplatData(e,t,n,s,r=!0){if(s=Math.min(s,n.sphericalHarmonicsDegree),r){const o=new Pe(s);for(let a=0;a<t;a++){const l=Cn.parseToUncompressedSplat(e,a,n,0,s);o.addSplat(l)}return o}else{const{splatBuffer:o,splatBufferDataOffsetBytes:a}=J.preallocateUncompressed(t,s);return Cn.parseToUncompressedSplatBufferSection(n,0,t-1,e,0,o.bufferData,a,s),o}}static parseToUncompressedSplat=(function(){let e=[];const t=new Vt,n=Pe.OFFSET.X,s=Pe.OFFSET.Y,r=Pe.OFFSET.Z,o=Pe.OFFSET.SCALE0,a=Pe.OFFSET.SCALE1,l=Pe.OFFSET.SCALE2,c=Pe.OFFSET.ROTATION0,u=Pe.OFFSET.ROTATION1,f=Pe.OFFSET.ROTATION2,d=Pe.OFFSET.ROTATION3,h=Pe.OFFSET.FDC0,x=Pe.OFFSET.FDC1,p=Pe.OFFSET.FDC2,g=Pe.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=Pe.OFFSET.FRC0+_;return function(_,A,v,S=0,y=0){y=Math.min(y,v.sphericalHarmonicsDegree),Cn.readSplat(_,v,A,S,e);const M=Pe.createSplat(y);if(e[Dm]!==void 0?(M[o]=Math.exp(e[Dm]),M[a]=Math.exp(e[P1]),M[l]=Math.exp(e[F1])):(M[o]=.01,M[a]=.01,M[l]=.01),e[Pm]!==void 0){const E=.28209479177387814;M[h]=(.5+E*e[Pm])*255,M[x]=(.5+E*e[H1])*255,M[p]=(.5+E*e[V1])*255}else e[Lm]!==void 0?(M[h]=e[Lm]*255,M[x]=e[G1]*255,M[p]=e[W1]*255):(M[h]=0,M[x]=0,M[p]=0);if(e[Fm]!==void 0&&(M[g]=1/(1+Math.exp(-e[Fm]))*255),M[h]=$t(Math.floor(M[h]),0,255),M[x]=$t(Math.floor(M[x]),0,255),M[p]=$t(Math.floor(M[p]),0,255),M[g]=$t(Math.floor(M[g]),0,255),y>=1&&e[X1]!==void 0){for(let E=0;E<9;E++)M[m[E]]=e[v.sphericalHarmonicsDegree1Fields[E]];if(y>=2)for(let E=0;E<15;E++)M[m[9+E]]=e[v.sphericalHarmonicsDegree2Fields[E]]}return t.set(e[L1],e[B1],e[U1],e[O1]),t.normalize(),M[c]=t.x,M[u]=t.y,M[f]=t.z,M[d]=t.w,M[n]=e[N1],M[s]=e[z1],M[r]=e[k1],M}})();static readSplat(e,t,n,s,r){return It.readVertex(e,t,n,s,t.fieldsToReadIndexes,r,!0)}static parseToUncompressedSplatArray(e,t=0){const{header:n,splatCount:s,splatData:r}=Bm(e);return Cn.decodeSectionSplatData(r,s,n,t,!0)}static parseToUncompressedSplatBuffer(e,t=0){const{header:n,splatCount:s,splatData:r}=Bm(e);return Cn.decodeSectionSplatData(r,s,n,t,!1)}}function Bm(i){const e=Cn.decodeHeaderFromBuffer(i),t=e.splatCount,n=Cn.findSplatData(i,e);return{header:e,splatCount:t,splatData:n}}const cx=["features_dc","features_rest_0","features_rest_1","features_rest_2","features_rest_3","features_rest_4","features_rest_5","features_rest_6","features_rest_7","features_rest_8","features_rest_9","features_rest_10","features_rest_11","features_rest_12","features_rest_13","features_rest_14","opacity","scaling","rotation_re","rotation_im"],Kl=cx.map((i,e)=>e),[jl,q1,Y1,Um,$l,Q1,Zu]=[0,1,4,16,17,18,19],ux=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0","f_rest_1","f_rest_2","f_rest_3","f_rest_4","f_rest_5","f_rest_6","f_rest_7","f_rest_8","f_rest_9","f_rest_10","f_rest_11","f_rest_12","f_rest_13","f_rest_14","f_rest_15","f_rest_16","f_rest_17","f_rest_18","f_rest_19","f_rest_20","f_rest_21","f_rest_22","f_rest_23","f_rest_24","f_rest_25","f_rest_26","f_rest_27","f_rest_28","f_rest_29","f_rest_30","f_rest_31","f_rest_32","f_rest_33","f_rest_34","f_rest_35","f_rest_36","f_rest_37","f_rest_38","f_rest_39","f_rest_40","f_rest_41","f_rest_42","f_rest_43","f_rest_44","f_rest_45"],hd=ux.map((i,e)=>e),[Om,K1,j1,$1,Z1,J1,eE,tE,nE,iE,pd,fx,dx,Nm]=hd,zm=pd,sE=fx,rE=dx,Zl=i=>{const e=(31744&i)>>10,t=1023&i;return(i>>15?-1:1)*(e?e===31?t?NaN:1/0:Math.pow(2,e-15)*(1+t/1024):t/1024*6103515625e-14)};class ui{static decodeSectionHeadersFromHeaderLines(e){const t=hd.reduce((u,f)=>(u[ux[f]]=f,u),{}),n=Kl.reduce((u,f)=>(u[cx[f]]=f,u),{}),s=It.getHeaderSectionNames(e);let r;for(let u=0;u<s.length;u++)s[u]==="codebook_centers"&&(r=u);let o=0,a=!1;const l=[];let c=0;for(;!a;){let u;c===r?u=It.decodeSectionHeader(e,n,o):u=It.decodeSectionHeader(e,t,o),a=u.endOfHeader,o=u.headerEndLine+1,a||(u.splatCount=u.vertexCount,u.bytesPerSplat=u.bytesPerVertex),l.push(u),c++}return l}static decodeSectionHeadersFromHeaderText(e){const t=It.convertHeaderTextToLines(e);return ui.decodeSectionHeadersFromHeaderLines(t)}static getSplatCountFromSectionHeaders(e){let t=0;for(let n of e)n.sectionName!=="codebook_centers"&&(t+=n.vertexCount);return t}static decodeHeaderFromHeaderText(e){const t=e.indexOf(It.HeaderEndToken)+It.HeaderEndToken.length+1,n=ui.decodeSectionHeadersFromHeaderText(e),s=ui.getSplatCountFromSectionHeaders(n);return{headerSizeBytes:t,sectionHeaders:n,splatCount:s}}static decodeHeaderFromBuffer(e){const t=It.readHeaderFromBuffer(e);return ui.decodeHeaderFromHeaderText(t)}static findVertexData(e,t,n){let s=t.headerSizeBytes;for(let r=0;r<n&&r<t.sectionHeaders.length;r++){const o=t.sectionHeaders[r];s+=o.dataSizeBytes}return new DataView(e,s,t.sectionHeaders[n].dataSizeBytes)}static decodeCodeBook(e,t){const n=[],s=[];for(let r=0;r<t.vertexCount;r++){It.readVertex(e,t,r,0,Kl,n);for(let o of Kl){const a=Kl[o];let l=s[a];l||(s[a]=l=[]),l.push(n[o])}}for(let r=0;r<s.length;r++){const o=s[r],a=.28209479177387814;for(let l=0;l<o.length;l++){const c=Zl(o[l]);r===Um?o[l]=Math.round(1/(1+Math.exp(-c))*255):r===jl?o[l]=Math.round((.5+a*c)*255):r===$l?o[l]=Math.exp(c):o[l]=c}}return s}static decodeSectionSplatData(e,t,n,s,r){r=Math.min(r,n.sphericalHarmonicsDegree);const o=new Pe(r);for(let a=0;a<t;a++){const l=ui.parseToUncompressedSplat(e,a,n,s,0,r);o.addSplat(l)}return o}static parseToUncompressedSplat=(function(){let e=[];const t=new Vt,n=Pe.OFFSET.X,s=Pe.OFFSET.Y,r=Pe.OFFSET.Z,o=Pe.OFFSET.SCALE0,a=Pe.OFFSET.SCALE1,l=Pe.OFFSET.SCALE2,c=Pe.OFFSET.ROTATION0,u=Pe.OFFSET.ROTATION1,f=Pe.OFFSET.ROTATION2,d=Pe.OFFSET.ROTATION3,h=Pe.OFFSET.FDC0,x=Pe.OFFSET.FDC1,p=Pe.OFFSET.FDC2,g=Pe.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=Pe.OFFSET.FRC0+_;return function(_,A,v,S,y=0,M=0){M=Math.min(M,v.sphericalHarmonicsDegree),ui.readSplat(_,v,A,y,e);const E=Pe.createSplat(M);if(e[Om]!==void 0?(E[o]=S[$l][e[Om]],E[a]=S[$l][e[K1]],E[l]=S[$l][e[j1]]):(E[o]=.01,E[a]=.01,E[l]=.01),e[pd]!==void 0?(E[h]=S[jl][e[pd]],E[x]=S[jl][e[fx]],E[p]=S[jl][e[dx]]):e[zm]!==void 0?(E[h]=e[zm]*255,E[x]=e[sE]*255,E[p]=e[rE]*255):(E[h]=0,E[x]=0,E[p]=0),e[Nm]!==void 0&&(E[g]=S[Um][e[Nm]]),E[h]=$t(Math.floor(E[h]),0,255),E[x]=$t(Math.floor(E[x]),0,255),E[p]=$t(Math.floor(E[p]),0,255),E[g]=$t(Math.floor(E[g]),0,255),M>=1&&v.sphericalHarmonicsDegree>=1){for(let O=0;O<9;O++){const z=S[q1+O%3];E[m[O]]=z[e[v.sphericalHarmonicsDegree1Fields[O]]]}if(M>=2&&v.sphericalHarmonicsDegree>=2)for(let O=0;O<15;O++){const z=S[Y1+O%5];E[m[9+O]]=z[e[v.sphericalHarmonicsDegree2Fields[O]]]}}const b=S[Q1][e[$1]],C=S[Zu][e[Z1]],D=S[Zu][e[J1]],F=S[Zu][e[eE]];return t.set(b,C,D,F),t.normalize(),E[c]=t.x,E[u]=t.y,E[f]=t.z,E[d]=t.w,E[n]=Zl(e[tE]),E[s]=Zl(e[nE]),E[r]=Zl(e[iE]),E}})();static readSplat(e,t,n,s,r){return It.readVertex(e,t,n,s,hd,r,!1)}static parseToUncompressedSplatArray(e,t=0){const n=[],s=ui.decodeHeaderFromBuffer(e,t);let r;for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName==="codebook_centers"){const c=ui.findVertexData(e,s,a);r=ui.decodeCodeBook(c,l)}}for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName!=="codebook_centers"){const c=l.vertexCount,u=ui.findVertexData(e,s,a),f=ui.decodeSectionSplatData(u,c,l,r,t);n.push(f)}}const o=new Pe(t);for(let a of n)for(let l of a.splats)o.addSplat(l);return o}}class km{static parseToUncompressedSplatArray(e,t=0){const n=It.determineHeaderFormatFromPlyBuffer(e);if(n===Vn.PlayCanvasCompressed)return Tt.parseToUncompressedSplatArray(e,t);if(n===Vn.INRIAV1)return Cn.parseToUncompressedSplatArray(e,t);if(n===Vn.INRIAV2)return ui.parseToUncompressedSplatArray(e,t)}static parseToUncompressedSplatBuffer(e,t=0){const n=It.determineHeaderFormatFromPlyBuffer(e);if(n===Vn.PlayCanvasCompressed)return Tt.parseToUncompressedSplatBuffer(e,t);if(n===Vn.INRIAV1)return Cn.parseToUncompressedSplatBuffer(e,t);if(n===Vn.INRIAV2)throw new Error("parseToUncompressedSplatBuffer() is not implemented for INRIA V2 PLY files")}}class vh{constructor(e,t,n,s){this.sectionCount=e,this.sectionFilters=t,this.groupingParameters=n,this.partitionGenerator=s}partitionUncompressedSplatArray(e){let t,n,s;if(this.partitionGenerator){const o=this.partitionGenerator(e);t=o.groupingParameters,n=o.sectionCount,s=o.sectionFilters}else t=this.groupingParameters,n=this.sectionCount,s=this.sectionFilters;const r=[];for(let o=0;o<n;o++){const a=new Pe(e.sphericalHarmonicsDegree),l=s[o];for(let c=0;c<e.splatCount;c++)l(c)&&a.addSplat(e.splats[c]);r.push(a)}return{splatArrays:r,parameters:t}}static getStandardPartitioner(e=0,t=new B,n=J.BucketBlockSize,s=J.BucketSize){const r=o=>{const a=Pe.OFFSET.X,l=Pe.OFFSET.Y,c=Pe.OFFSET.Z;e<=0&&(e=o.splatCount);const u=new B,f=.5,d=m=>{m.x=Math.floor(m.x/f)*f,m.y=Math.floor(m.y/f)*f,m.z=Math.floor(m.z/f)*f};o.splats.forEach(m=>{u.set(m[a],m[l],m[c]).sub(t),d(u),m.centerDist=u.lengthSq()}),o.splats.sort((m,_)=>{let A=m.centerDist,v=_.centerDist;return A>v?1:-1});const h=[],x=[];e=Math.min(o.splatCount,e);const p=Math.ceil(o.splatCount/e);let g=0;for(let m=0;m<p;m++){let _=g;h.push(A=>A>=_&&A<_+e),x.push({blocksSize:n,bucketSize:s}),g+=e}return{sectionCount:h.length,sectionFilters:h,groupingParameters:x}};return new vh(void 0,void 0,void 0,r)}}class pl{constructor(e,t,n,s,r,o,a){this.splatPartitioner=e,this.alphaRemovalThreshold=t,this.compressionLevel=n,this.sectionSize=s,this.sceneCenter=r?new B().copy(r):void 0,this.blockSize=o,this.bucketSize=a}generateFromUncompressedSplatArray(e){const t=this.splatPartitioner.partitionUncompressedSplatArray(e);return J.generateFromUncompressedSplatArrays(t.splatArrays,this.alphaRemovalThreshold,this.compressionLevel,this.sceneCenter,this.blockSize,this.bucketSize,t.parameters)}static getStandardGenerator(e=1,t=1,n=0,s=new B,r=J.BucketBlockSize,o=J.BucketSize){const a=vh.getStandardPartitioner(n,s,r,o);return new pl(a,e,t,n,s,r,o)}}const un={Downloading:0,Processing:1,Done:2};class Ec extends Error{constructor(e){super(e)}}const Yt={ProgressiveToSplatBuffer:0,ProgressiveToSplatArray:1,DownloadBeforeProcessing:2};function Hm(i,e){let t=0;for(let s of i)t+=s.sizeBytes;(!e||e.byteLength<t)&&(e=new ArrayBuffer(t));let n=0;for(let s of i)new Uint8Array(e,n,s.sizeBytes).set(s.data),n+=s.sizeBytes;return e}function Vm(i,e,t,n,s,r,o,a){return e?pl.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):J.generateFromUncompressedSplatArrays([i],t,0,new B)}class Ah{static loadFromURL(e,t,n,s,r,o,a=!0,l=0,c,u,f,d,h){let x;!n&&!a?x=Yt.DownloadBeforeProcessing:a?x=Yt.ProgressiveToSplatArray:x=Yt.ProgressiveToSplatBuffer;const p=Nt.ProgressiveLoadSectionSize,g=J.HeaderSizeBytes+J.SectionHeaderSizeBytes,m=1;let _,A,v,S,y,M=0,E=0,b=0,C=!1,D=!1,F=!1;const O=ch();let z=0,V=0,H=0,q=0,G="",$=null,fe=[],Y;const we=new TextDecoder,ze=(ke,We,ne)=>{const ue=ke>=100;if(ne&&(fe.push({data:ne,sizeBytes:ne.byteLength,startBytes:H,endBytes:H+ne.byteLength}),H+=ne.byteLength),x===Yt.DownloadBeforeProcessing)ue&&O.resolve(fe);else{if(C){if(_===Vn.PlayCanvasCompressed&&!D){const Se=$.headerSizeBytes+$.chunkElement.storageSizeBytes;y=Hm(fe,y),y.byteLength>=Se&&(Tt.readElementData($.chunkElement,y,$.headerSizeBytes),z=Se,V=Se,D=!0)}}else if(G+=we.decode(ne),It.checkTextForEndHeader(G)){if(_=It.determineHeaderFormatFromHeaderText(G),_===Vn.INRIAV1)$=Cn.decodeHeaderText(G),l=Math.min(l,$.sphericalHarmonicsDegree),M=$.splatCount,D=!0,q=$.headerSizeBytes+$.bytesPerSplat*M;else if(_===Vn.PlayCanvasCompressed){if($=Tt.decodeHeaderText(G),l=Math.min(l,$.sphericalHarmonicsDegree),x===Yt.ProgressiveToSplatBuffer&&l>0)throw new Ec("PlyLoader.loadFromURL() -> Selected PLY format has spherical harmonics data that cannot be progressively loaded.");M=$.vertexElement.count,q=$.headerSizeBytes+$.bytesPerSplat*M+$.chunkElement.storageSizeBytes}else{if(x===Yt.ProgressiveToSplatBuffer)throw new Ec("PlyLoader.loadFromURL() -> Selected PLY format cannot be progressively loaded.");x=Yt.DownloadBeforeProcessing;return}if(x===Yt.ProgressiveToSplatBuffer){const Se=J.CompressionLevels[0].SphericalHarmonicsDegrees[l],he=g+Se.BytesPerSplat*M;v=new ArrayBuffer(he),J.writeHeaderToBuffer({versionMajor:J.CurrentMajorVersion,versionMinor:J.CurrentMinorVersion,maxSectionCount:m,sectionCount:m,maxSplatCount:M,splatCount:0,compressionLevel:0,sceneCenter:new B},v)}else Y=new Pe(l);z=$.headerSizeBytes,V=$.headerSizeBytes,C=!0}if(C&&D&&fe.length>0&&(A=Hm(fe,A),H-z>p||H>=q&&!F||ue)){const he=F?$.sphericalHarmonicsPerSplat:$.bytesPerSplat,Ze=(F?H:Math.min(q,H))-V,U=Math.floor(Ze/he),N=U*he,K=H-V-N,R=V-fe[0].startBytes,te=new DataView(A,R,N);if(F)_===Vn.PlayCanvasCompressed&&x===Yt.ProgressiveToSplatArray&&(Tt.parseSphericalHarmonicsToUncompressedSplatArraySection($.chunkElement,$.shElement,b,b+U-1,te,0,l,$.sphericalHarmonicsDegree,Y),b+=U);else{if(x===Yt.ProgressiveToSplatBuffer){const oe=J.CompressionLevels[0].SphericalHarmonicsDegrees[l],pe=E*oe.BytesPerSplat+g;_===Vn.PlayCanvasCompressed?Tt.parseToUncompressedSplatBufferSection($.chunkElement,$.vertexElement,0,U-1,E,te,v,pe):Cn.parseToUncompressedSplatBufferSection($,0,U-1,te,0,v,pe,l)}else _===Vn.PlayCanvasCompressed?Tt.parseToUncompressedSplatArraySection($.chunkElement,$.vertexElement,0,U-1,E,te,Y):Cn.parseToUncompressedSplatArraySection($,0,U-1,te,0,Y,l);E+=U,x===Yt.ProgressiveToSplatBuffer&&(S||(J.writeSectionHeaderToBuffer({maxSplatCount:M,splatCount:E,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:l},0,v,J.HeaderSizeBytes),S=new J(v,!1)),S.updateLoadedCounts(1,E)),H>=q&&(F=!0)}if(K===0)fe=[];else{let oe=[],pe=0;for(let ie=fe.length-1;ie>=0;ie--){const me=fe[ie];if(pe+=me.sizeBytes,oe.unshift(me),pe>=K)break}fe=oe}z+=p,V+=N}s&&S&&s(S,ue),ue&&(x===Yt.ProgressiveToSplatBuffer?O.resolve(S):O.resolve(Y))}t&&t(ke,We,un.Downloading)};return t&&t(0,"0%",un.Downloading),Jc(e,ze,!1,c).then(()=>(t&&t(0,"0%",un.Processing),O.promise.then(ke=>{if(t&&t(100,"100%",un.Done),x===Yt.DownloadBeforeProcessing){const We=fe.map(ne=>ne.data);return new Blob(We).arrayBuffer().then(ne=>Ah.loadFromFileData(ne,r,o,a,l,u,f,d,h))}else return x===Yt.ProgressiveToSplatBuffer?ke:pi(()=>Vm(ke,a,r,o,u,f,d,h))})))}static loadFromFileData(e,t,n,s,r=0,o,a,l,c){return s?pi(()=>km.parseToUncompressedSplatArray(e,r)).then(u=>Vm(u,s,t,n,o,a,l,c)):pi(()=>km.parseToUncompressedSplatBuffer(e,r))}}const oE=i=>new ReadableStream({async start(e){e.enqueue(i),e.close()}});async function aE(i){try{const e=oE(i);if(!e)throw new Error("Failed to create stream from data");return await lE(e)}catch(e){throw console.error("Error decompressing gzipped data:",e),e}}async function lE(i){const e=i.pipeThrough(new DecompressionStream("gzip")),n=await new Response(e).arrayBuffer();return new Uint8Array(n)}const cE=1347635022,uE=1,fE=.15;function dE(i){const e=i>>15&1,t=i>>10&31,n=i&1023,s=e===1?-1:1;return t===0?s*Math.pow(2,-14)*n/1024:t===31?n!==0?NaN:s*(1/0):s*Math.pow(2,t-15)*(1+n/1024)}function hE(i){return(i-128)/128}function Hr(i){switch(i){case 0:return 0;case 1:return 3;case 2:return 8;case 3:return 15;default:return console.error(`[SPZ: ERROR] Unsupported SH degree: ${i}`),0}}const pE=(function(){let i=[];const e=new Vt,t=Pe.OFFSET.X,n=Pe.OFFSET.Y,s=Pe.OFFSET.Z,r=Pe.OFFSET.SCALE0,o=Pe.OFFSET.SCALE1,a=Pe.OFFSET.SCALE2,l=Pe.OFFSET.ROTATION0,c=Pe.OFFSET.ROTATION1,u=Pe.OFFSET.ROTATION2,f=Pe.OFFSET.ROTATION3,d=Pe.OFFSET.FDC0,h=Pe.OFFSET.FDC1,x=Pe.OFFSET.FDC2,p=Pe.OFFSET.OPACITY,g=[Hr(0),Hr(1),Hr(2),Hr(3)],m=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(_,A,v){v=Math.min(A,v);const S=Pe.createSplat(v);_.scale[0]!==void 0?(S[r]=_.scale[0],S[o]=_.scale[1],S[a]=_.scale[2]):(S[r]=.01,S[o]=.01,S[a]=.01),_.color[0]!==void 0?(S[d]=_.color[0],S[h]=_.color[1],S[x]=_.color[2]):i[RED]!==void 0?(S[d]=i[RED]*255,S[h]=i[GREEN]*255,S[x]=i[BLUE]*255):(S[d]=0,S[h]=0,S[x]=0),_.alpha!==void 0&&(S[p]=_.alpha),S[d]=$t(Math.floor(S[d]),0,255),S[h]=$t(Math.floor(S[h]),0,255),S[x]=$t(Math.floor(S[x]),0,255),S[p]=$t(Math.floor(S[p]),0,255);let y=g[v],M=g[A];for(let E=0;E<3;++E)for(let b=0;b<15;++b){const C=m[E*15+b];b<y&&b<M&&(S[Pe.OFFSET.FRC0+C]=_.sh[E*M+b])}return e.set(_.rotation[3],_.rotation[0],_.rotation[1],_.rotation[2]),e.normalize(),S[l]=e.x,S[c]=e.y,S[u]=e.z,S[f]=e.w,S[t]=_.position[0],S[n]=_.position[1],S[s]=_.position[2],S}})();function mE(i,e,t,n){return!(i.positions.length!==e*3*(n?2:3)||i.scales.length!==e*3||i.rotations.length!==e*3||i.alphas.length!==e||i.colors.length!==e*3||i.sh.length!==e*t*3)}function Gm(i,e,t,n,s){e=Math.min(e,i.shDegree);const r=i.numPoints,o=Hr(i.shDegree),a=i.positions.length===r*3*2;if(!mE(i,r,o,a))return null;const l={position:[],scale:[],rotation:[],alpha:void 0,color:[],sh:[]};let c;a&&(c=new Uint16Array(i.positions.buffer,i.positions.byteOffset,r*3));const u=1/(1<<i.fractionalBits),f=Hr(i.shDegree),d=.28209479177387814;for(let h=0;h<r;h++){if(a)for(let _=0;_<3;_++)l.position[_]=dE(c[h*3+_]);else for(let _=0;_<3;_++){const A=h*9+_*3;let v=i.positions[A];v|=i.positions[A+1]<<8,v|=i.positions[A+2]<<16,v|=v&8388608?4278190080:0,l.position[_]=v*u}for(let _=0;_<3;_++)l.scale[_]=Math.exp(i.scales[h*3+_]/16-10);const x=i.rotations.subarray(h*3,h*3+3),p=[x[0]/127.5-1,x[1]/127.5-1,x[2]/127.5-1];l.rotation[0]=p[0],l.rotation[1]=p[1],l.rotation[2]=p[2];const g=p[0]*p[0]+p[1]*p[1]+p[2]*p[2];l.rotation[3]=Math.sqrt(Math.max(0,1-g)),l.alpha=Math.floor(i.alphas[h]);for(let _=0;_<3;_++)l.color[_]=Math.floor(((i.colors[h*3+_]/255-.5)/fE*d+.5)*255);for(let _=0;_<3;_++)for(let A=0;A<f;A++)l.sh[_*f+A]=hE(i.sh[f*3*h+A*3+_]);const m=pE(l,i.shDegree,e);if(t){const _=J.CompressionLevels[0].SphericalHarmonicsDegrees[e].BytesPerSplat,A=h*_+s;J.writeSplatDataToSectionBuffer(m,n,A,0,e)}else n.addSplat(m)}}const gE=16,xE=1e7;function _E(i){const e=new DataView(i);let t=0;const n={magic:e.getUint32(t,!0),version:e.getUint32(t+4,!0),numPoints:e.getUint32(t+8,!0),shDegree:e.getUint8(t+12),fractionalBits:e.getUint8(t+13),flags:e.getUint8(t+14),reserved:e.getUint8(t+15)};if(t+=gE,n.magic!==cE)return console.error("[SPZ ERROR] deserializePackedGaussians: header not found"),null;if(n.version<1||n.version>2)return console.error(`[SPZ ERROR] deserializePackedGaussians: version not supported: ${n.version}`),null;if(n.numPoints>xE)return console.error(`[SPZ ERROR] deserializePackedGaussians: Too many points: ${n.numPoints}`),null;if(n.shDegree>3)return console.error(`[SPZ ERROR] deserializePackedGaussians: Unsupported SH degree: ${n.shDegree}`),null;const s=n.numPoints,r=Hr(n.shDegree),o=n.version===1,a={numPoints:s,shDegree:n.shDegree,fractionalBits:n.fractionalBits,antialiased:(n.flags&uE)!==0,positions:new Uint8Array(s*3*(o?2:3)),scales:new Uint8Array(s*3),rotations:new Uint8Array(s*3),alphas:new Uint8Array(s),colors:new Uint8Array(s*3),sh:new Uint8Array(s*r*3)};try{const l=new Uint8Array(i);let c=a.positions.length,u=t;if(a.positions.set(l.slice(u,u+c)),u+=c,a.alphas.set(l.slice(u,u+a.alphas.length)),u+=a.alphas.length,a.colors.set(l.slice(u,u+a.colors.length)),u+=a.colors.length,a.scales.set(l.slice(u,u+a.scales.length)),u+=a.scales.length,a.rotations.set(l.slice(u,u+a.rotations.length)),u+=a.rotations.length,a.sh.set(l.slice(u,u+a.sh.length)),u+a.sh.length!==i.byteLength)return console.error("[SPZ ERROR] deserializePackedGaussians: incorrect buffer size"),null}catch(l){return console.error("[SPZ ERROR] deserializePackedGaussians: read error",l),null}return a}async function vE(i){try{const e=await aE(i);return _E(e.buffer)}catch(e){return console.error("[SPZ ERROR] loadSpzPacked: decompression error",e),null}}class Sh{static loadFromURL(e,t,n,s,r=!0,o=0,a,l,c,u,f){return t&&t(0,"0%",un.Downloading),Jc(e,t,!0,a).then(d=>(t&&t(0,"0%",un.Processing),Sh.loadFromFileData(d,n,s,r,o,l,c,u,f)))}static async loadFromFileData(e,t,n,s,r=0,o,a,l,c){await pi();const u=await vE(e);r=Math.min(u.shDegree,r);const f=new Pe(r);if(s)return Gm(u,r,!1,f,0),pl.getStandardGenerator(t,n,o,a,l,c).generateFromUncompressedSplatArray(f);{const{splatBuffer:d,splatBufferDataOffsetBytes:h}=J.preallocateUncompressed(u.numPoints,r);return Gm(u,r,!0,d.bufferData,h),d}}}class Ut{static RowSizeBytes=32;static CenterSizeBytes=12;static ScaleSizeBytes=12;static RotationSizeBytes=4;static ColorSizeBytes=4;static parseToUncompressedSplatBufferSection(e,t,n,s,r,o){const a=J.CompressionLevels[0].BytesPerCenter,l=J.CompressionLevels[0].BytesPerScale,c=J.CompressionLevels[0].BytesPerRotation,u=J.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat;for(let f=e;f<=t;f++){const d=f*Ut.RowSizeBytes+s,h=new Float32Array(n,d,3),x=new Float32Array(n,d+Ut.CenterSizeBytes,3),p=new Uint8Array(n,d+Ut.CenterSizeBytes+Ut.ScaleSizeBytes,4),g=new Uint8Array(n,d+Ut.CenterSizeBytes+Ut.ScaleSizeBytes+Ut.RotationSizeBytes,4),m=new Vt((g[1]-128)/128,(g[2]-128)/128,(g[3]-128)/128,(g[0]-128)/128);m.normalize();const _=f*u+o,A=new Float32Array(r,_,3),v=new Float32Array(r,_+a,3),S=new Float32Array(r,_+a+l,4),y=new Uint8Array(r,_+a+l+c,4);A[0]=h[0],A[1]=h[1],A[2]=h[2],v[0]=x[0],v[1]=x[1],v[2]=x[2],S[0]=m.w,S[1]=m.x,S[2]=m.y,S[3]=m.z,y[0]=p[0],y[1]=p[1],y[2]=p[2],y[3]=p[3]}}static parseToUncompressedSplatArraySection(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*Ut.RowSizeBytes+s,l=new Float32Array(n,a,3),c=new Float32Array(n,a+Ut.CenterSizeBytes,3),u=new Uint8Array(n,a+Ut.CenterSizeBytes+Ut.ScaleSizeBytes,4),f=new Uint8Array(n,a+Ut.CenterSizeBytes+Ut.ScaleSizeBytes+Ut.RotationSizeBytes,4),d=new Vt((f[1]-128)/128,(f[2]-128)/128,(f[3]-128)/128,(f[0]-128)/128);d.normalize(),r.addSplatFromComonents(l[0],l[1],l[2],c[0],c[1],c[2],d.w,d.x,d.y,d.z,u[0],u[1],u[2],u[3])}}static parseStandardSplatToUncompressedSplatArray(e){const t=e.byteLength/Ut.RowSizeBytes,n=new Pe;for(let s=0;s<t;s++){const r=s*Ut.RowSizeBytes,o=new Float32Array(e,r,3),a=new Float32Array(e,r+Ut.CenterSizeBytes,3),l=new Uint8Array(e,r+Ut.CenterSizeBytes+Ut.ScaleSizeBytes,4),c=new Uint8Array(e,r+Ut.CenterSizeBytes+Ut.ScaleSizeBytes+Ut.ColorSizeBytes,4),u=new Vt((c[1]-128)/128,(c[2]-128)/128,(c[3]-128)/128,(c[0]-128)/128);u.normalize(),n.addSplatFromComonents(o[0],o[1],o[2],a[0],a[1],a[2],u.w,u.x,u.y,u.z,l[0],l[1],l[2],l[3])}return n}}function Wm(i,e,t,n,s,r,o,a){return e?pl.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):J.generateFromUncompressedSplatArrays([i],t,0,new B)}class yh{static loadFromURL(e,t,n,s,r,o,a=!0,l,c,u,f,d){let h=n?Yt.ProgressiveToSplatBuffer:Yt.ProgressiveToSplatArray;a&&(h=Yt.ProgressiveToSplatArray);const x=J.HeaderSizeBytes+J.SectionHeaderSizeBytes,p=Nt.ProgressiveLoadSectionSize,g=1;let m,_,A,v=0,S=0,y;const M=ch();let E=0,b=0,C=[];const D=(F,O,z,V)=>{const H=F>=100;if(z&&C.push(z),h===Yt.DownloadBeforeProcessing){H&&M.resolve(C);return}if(!V){if(n)throw new Ec("Cannon directly load .splat because no file size info is available.");h=Yt.DownloadBeforeProcessing;return}if(!m){v=V/Ut.RowSizeBytes,m=new ArrayBuffer(V);const q=J.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,G=x+q*v;h===Yt.ProgressiveToSplatBuffer?(_=new ArrayBuffer(G),J.writeHeaderToBuffer({versionMajor:J.CurrentMajorVersion,versionMinor:J.CurrentMinorVersion,maxSectionCount:g,sectionCount:g,maxSplatCount:v,splatCount:S,compressionLevel:0,sceneCenter:new B},_)):y=new Pe(0)}if(z){new Uint8Array(m,b,z.byteLength).set(new Uint8Array(z)),b+=z.byteLength;const q=b-E;if(q>p||H){const $=(H?q:p)/Ut.RowSizeBytes,fe=S+$;h===Yt.ProgressiveToSplatBuffer?Ut.parseToUncompressedSplatBufferSection(S,fe-1,m,0,_,x):Ut.parseToUncompressedSplatArraySection(S,fe-1,m,0,y),S=fe,h===Yt.ProgressiveToSplatBuffer&&(A||(J.writeSectionHeaderToBuffer({maxSplatCount:v,splatCount:S,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0},0,_,J.HeaderSizeBytes),A=new J(_,!1)),A.updateLoadedCounts(1,S),s&&s(A,H)),E+=p}}H&&(h===Yt.ProgressiveToSplatBuffer?M.resolve(A):M.resolve(y)),t&&t(F,O,un.Downloading)};return t&&t(0,"0%",un.Downloading),Jc(e,D,!1,l).then(()=>(t&&t(0,"0%",un.Processing),M.promise.then(F=>(t&&t(100,"100%",un.Done),h===Yt.DownloadBeforeProcessing?new Blob(C).arrayBuffer().then(O=>yh.loadFromFileData(O,r,o,a,c,u,f,d)):h===Yt.ProgressiveToSplatBuffer?F:pi(()=>Wm(F,a,r,o,c,u,f,d))))))}static loadFromFileData(e,t,n,s,r,o,a,l){return pi(()=>{const c=Ut.parseStandardSplatToUncompressedSplatArray(e);return Wm(c,s,t,n,r,o,a,l)})}}class Ua{static checkVersion(e){const t=J.CurrentMajorVersion,n=J.CurrentMinorVersion,s=J.parseHeader(e);if(s.versionMajor===t&&s.versionMinor>=n||s.versionMajor>t)return!0;throw new Error(`KSplat version not supported: v${s.versionMajor}.${s.versionMinor}. Minimum required: v${t}.${n}`)}static loadFromURL(e,t,n,s,r){let o,a,l,c,u=!1,f=!1,d,h=[],x=!1,p=!1,g=0,m=0,_=0,A=!1,v=!1,S=!1,y=[];const M=ch(),E=()=>{!u&&!f&&g>=J.HeaderSizeBytes&&(f=!0,new Blob(y).arrayBuffer().then(V=>{l=new ArrayBuffer(J.HeaderSizeBytes),new Uint8Array(l).set(new Uint8Array(V,0,J.HeaderSizeBytes)),Ua.checkVersion(l),f=!1,u=!0,c=J.parseHeader(l),window.setTimeout(()=>{D()},1)}))};let b=0;const C=()=>{b===0&&(b++,window.setTimeout(()=>{b--,F()},1))},D=()=>{const z=()=>{p=!0,new Blob(y).arrayBuffer().then(H=>{p=!1,x=!0,d=new ArrayBuffer(c.maxSectionCount*J.SectionHeaderSizeBytes),new Uint8Array(d).set(new Uint8Array(H,J.HeaderSizeBytes,c.maxSectionCount*J.SectionHeaderSizeBytes)),h=J.parseSectionHeaders(c,d,0,!1);let q=0;for(let $=0;$<c.maxSectionCount;$++)q+=h[$].storageSizeBytes;const G=J.HeaderSizeBytes+c.maxSectionCount*J.SectionHeaderSizeBytes+q;if(!o){o=new ArrayBuffer(G);let $=0;for(let fe=0;fe<y.length;fe++){const Y=y[fe];new Uint8Array(o,$,Y.byteLength).set(new Uint8Array(Y)),$+=Y.byteLength}}_=J.HeaderSizeBytes+J.SectionHeaderSizeBytes*c.maxSectionCount;for(let $=0;$<=h.length&&$<c.maxSectionCount;$++)_+=h[$].storageSizeBytes;C()})};!p&&!x&&u&&g>=J.HeaderSizeBytes+J.SectionHeaderSizeBytes*c.maxSectionCount&&z()},F=()=>{if(S)return;S=!0;const z=()=>{if(S=!1,x){if(v)return;if(A=g>=_,g-m>Nt.ProgressiveLoadSectionSize||A){m+=Nt.ProgressiveLoadSectionSize,v=m>=_,a||(a=new J(o,!1));const H=J.HeaderSizeBytes+J.SectionHeaderSizeBytes*c.maxSectionCount;let q=0,G=0,$=0;for(let we=0;we<c.maxSectionCount;we++){const ze=h[we],ke=q+ze.partiallyFilledBucketCount*4+ze.bucketStorageSizeBytes*ze.bucketCount,We=H+ke;if(m>=We){G++;const ne=m-We,he=J.CompressionLevels[c.compressionLevel].SphericalHarmonicsDegrees[ze.sphericalHarmonicsDegree].BytesPerSplat;let Ee=Math.floor(ne/he);Ee=Math.min(Ee,ze.maxSplatCount),$+=Ee,a.updateLoadedCounts(G,$),a.updateSectionLoadedCounts(we,Ee)}else break;q+=ze.storageSizeBytes}s(a,v);const fe=m/_*100,Y=fe.toFixed(2)+"%";t&&t(fe,Y,un.Downloading),v?M.resolve(a):F()}}};window.setTimeout(z,Nt.ProgressiveLoadSectionDelayDuration)};return Jc(e,(z,V,H)=>{H&&(y.push(H),o&&new Uint8Array(o,g,H.byteLength).set(new Uint8Array(H)),g+=H.byteLength),n?(E(),D(),F()):t&&t(z,V,un.Downloading)},!n,r).then(z=>(t&&t(0,"0%",un.Processing),(n?M.promise:Ua.loadFromFileData(z)).then(H=>(t&&t(100,"100%",un.Done),H))))}static loadFromFileData(e){return pi(()=>(Ua.checkVersion(e),new J(e)))}static downloadFile=(function(){let e;return function(t,n){const s=new Blob([t.bufferData],{type:"application/octet-stream"});e||(e=document.createElement("a"),document.body.appendChild(e)),e.download=n,e.href=URL.createObjectURL(s),e.click()}})()}const Zn={Splat:0,KSplat:1,Ply:2,Spz:3},Xm=i=>i.endsWith(".ply")?Zn.Ply:i.endsWith(".splat")?Zn.Splat:i.endsWith(".ksplat")?Zn.KSplat:i.endsWith(".spz")?Zn.Spz:null,qm={type:"change"},Ju={type:"start"},Ym={type:"end"},Jl=new sh,Qm=new Ks,AE=Math.cos(70*Sn.DEG2RAD);class ec extends $r{constructor(e,t){super(),this.object=e,this.domElement=t,this.domElement.style.touchAction="none",this.enabled=!0,this.target=new B,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"KeyA",UP:"KeyW",RIGHT:"KeyD",BOTTOM:"KeyS"},this.mouseButtons={LEFT:io.ROTATE,MIDDLE:io.DOLLY,RIGHT:io.PAN},this.touches={ONE:so.ROTATE,TWO:so.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this.getPolarAngle=function(){return a.phi},this.getAzimuthalAngle=function(){return a.theta},this.getDistance=function(){return this.object.position.distanceTo(this.target)},this.listenToKeyEvents=function(k){k.addEventListener("keydown",T),this._domElementKeyEvents=k},this.stopListenToKeyEvents=function(){this._domElementKeyEvents.removeEventListener("keydown",T),this._domElementKeyEvents=null},this.saveState=function(){n.target0.copy(n.target),n.position0.copy(n.object.position),n.zoom0=n.object.zoom},this.reset=function(){n.target.copy(n.target0),n.object.position.copy(n.position0),n.object.zoom=n.zoom0,this.clearDampedRotation(),this.clearDampedPan(),n.object.updateProjectionMatrix(),n.dispatchEvent(qm),n.update(),r=s.NONE},this.clearDampedRotation=function(){l.theta=0,l.phi=0},this.clearDampedPan=function(){u.set(0,0,0)},this.update=(function(){const k=new B,Z=new Vt().setFromUnitVectors(e.up,new B(0,1,0)),xe=Z.clone().invert(),Re=new B,Be=new Vt,Fe=new B,je=2*Math.PI;return function(){Z.setFromUnitVectors(e.up,new B(0,1,0)),xe.copy(Z).invert();const Le=n.object.position;k.copy(Le).sub(n.target),k.applyQuaternion(Z),a.setFromVector3(k),n.autoRotate&&r===s.NONE&&D(b()),n.enableDamping?(a.theta+=l.theta*n.dampingFactor,a.phi+=l.phi*n.dampingFactor):(a.theta+=l.theta,a.phi+=l.phi);let Me=n.minAzimuthAngle,be=n.maxAzimuthAngle;isFinite(Me)&&isFinite(be)&&(Me<-Math.PI?Me+=je:Me>Math.PI&&(Me-=je),be<-Math.PI?be+=je:be>Math.PI&&(be-=je),Me<=be?a.theta=Math.max(Me,Math.min(be,a.theta)):a.theta=a.theta>(Me+be)/2?Math.max(Me,a.theta):Math.min(be,a.theta)),a.phi=Math.max(n.minPolarAngle,Math.min(n.maxPolarAngle,a.phi)),a.makeSafe(),n.enableDamping===!0?n.target.addScaledVector(u,n.dampingFactor):n.target.add(u),n.zoomToCursor&&y||n.object.isOrthographicCamera?a.radius=$(a.radius):a.radius=$(a.radius*c),k.setFromSpherical(a),k.applyQuaternion(xe),Le.copy(n.target).add(k),n.object.lookAt(n.target),n.enableDamping===!0?(l.theta*=1-n.dampingFactor,l.phi*=1-n.dampingFactor,u.multiplyScalar(1-n.dampingFactor)):(l.set(0,0,0),u.set(0,0,0));let Ae=!1;if(n.zoomToCursor&&y){let ge=null;if(n.object.isPerspectiveCamera){const qe=k.length();ge=$(qe*c);const Je=qe-ge;n.object.position.addScaledVector(v,Je),n.object.updateMatrixWorld()}else if(n.object.isOrthographicCamera){const qe=new B(S.x,S.y,0);qe.unproject(n.object),n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),Ae=!0;const Je=new B(S.x,S.y,0);Je.unproject(n.object),n.object.position.sub(Je).add(qe),n.object.updateMatrixWorld(),ge=k.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),n.zoomToCursor=!1;ge!==null&&(this.screenSpacePanning?n.target.set(0,0,-1).transformDirection(n.object.matrix).multiplyScalar(ge).add(n.object.position):(Jl.origin.copy(n.object.position),Jl.direction.set(0,0,-1).transformDirection(n.object.matrix),Math.abs(n.object.up.dot(Jl.direction))<AE?e.lookAt(n.target):(Qm.setFromNormalAndCoplanarPoint(n.object.up,n.target),Jl.intersectPlane(Qm,n.target))))}else n.object.isOrthographicCamera&&(n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),Ae=!0);return c=1,y=!1,Ae||Re.distanceToSquared(n.object.position)>o||8*(1-Be.dot(n.object.quaternion))>o||Fe.distanceToSquared(n.target)>0?(n.dispatchEvent(qm),Re.copy(n.object.position),Be.copy(n.object.quaternion),Fe.copy(n.target),Ae=!1,!0):!1}})(),this.dispose=function(){n.domElement.removeEventListener("contextmenu",de),n.domElement.removeEventListener("pointerdown",pe),n.domElement.removeEventListener("pointercancel",me),n.domElement.removeEventListener("wheel",I),n.domElement.removeEventListener("pointermove",ie),n.domElement.removeEventListener("pointerup",me),n._domElementKeyEvents!==null&&(n._domElementKeyEvents.removeEventListener("keydown",T),n._domElementKeyEvents=null)};const n=this,s={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6};let r=s.NONE;const o=1e-6,a=new nm,l=new nm;let c=1;const u=new B,f=new Ke,d=new Ke,h=new Ke,x=new Ke,p=new Ke,g=new Ke,m=new Ke,_=new Ke,A=new Ke,v=new B,S=new Ke;let y=!1;const M=[],E={};function b(){return 2*Math.PI/60/60*n.autoRotateSpeed}function C(){return Math.pow(.95,n.zoomSpeed)}function D(k){l.theta-=k}function F(k){l.phi-=k}const O=(function(){const k=new B;return function(xe,Re){k.setFromMatrixColumn(Re,0),k.multiplyScalar(-xe),u.add(k)}})(),z=(function(){const k=new B;return function(xe,Re){n.screenSpacePanning===!0?k.setFromMatrixColumn(Re,1):(k.setFromMatrixColumn(Re,0),k.crossVectors(n.object.up,k)),k.multiplyScalar(xe),u.add(k)}})(),V=(function(){const k=new B;return function(xe,Re){const Be=n.domElement;if(n.object.isPerspectiveCamera){const Fe=n.object.position;k.copy(Fe).sub(n.target);let je=k.length();je*=Math.tan(n.object.fov/2*Math.PI/180),O(2*xe*je/Be.clientHeight,n.object.matrix),z(2*Re*je/Be.clientHeight,n.object.matrix)}else n.object.isOrthographicCamera?(O(xe*(n.object.right-n.object.left)/n.object.zoom/Be.clientWidth,n.object.matrix),z(Re*(n.object.top-n.object.bottom)/n.object.zoom/Be.clientHeight,n.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),n.enablePan=!1)}})();function H(k){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c/=k:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function q(k){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c*=k:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function G(k){if(!n.zoomToCursor)return;y=!0;const Z=n.domElement.getBoundingClientRect(),xe=k.clientX-Z.left,Re=k.clientY-Z.top,Be=Z.width,Fe=Z.height;S.x=xe/Be*2-1,S.y=-(Re/Fe)*2+1,v.set(S.x,S.y,1).unproject(e).sub(e.position).normalize()}function $(k){return Math.max(n.minDistance,Math.min(n.maxDistance,k))}function fe(k){f.set(k.clientX,k.clientY)}function Y(k){G(k),m.set(k.clientX,k.clientY)}function we(k){x.set(k.clientX,k.clientY)}function ze(k){d.set(k.clientX,k.clientY),h.subVectors(d,f).multiplyScalar(n.rotateSpeed);const Z=n.domElement;D(2*Math.PI*h.x/Z.clientHeight),F(2*Math.PI*h.y/Z.clientHeight),f.copy(d),n.update()}function ke(k){_.set(k.clientX,k.clientY),A.subVectors(_,m),A.y>0?H(C()):A.y<0&&q(C()),m.copy(_),n.update()}function We(k){p.set(k.clientX,k.clientY),g.subVectors(p,x).multiplyScalar(n.panSpeed),V(g.x,g.y),x.copy(p),n.update()}function ne(k){G(k),k.deltaY<0?q(C()):k.deltaY>0&&H(C()),n.update()}function ue(k){let Z=!1;switch(k.code){case n.keys.UP:k.ctrlKey||k.metaKey||k.shiftKey?F(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):V(0,n.keyPanSpeed),Z=!0;break;case n.keys.BOTTOM:k.ctrlKey||k.metaKey||k.shiftKey?F(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):V(0,-n.keyPanSpeed),Z=!0;break;case n.keys.LEFT:k.ctrlKey||k.metaKey||k.shiftKey?D(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):V(n.keyPanSpeed,0),Z=!0;break;case n.keys.RIGHT:k.ctrlKey||k.metaKey||k.shiftKey?D(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):V(-n.keyPanSpeed,0),Z=!0;break}Z&&(k.preventDefault(),n.update())}function Se(){if(M.length===1)f.set(M[0].pageX,M[0].pageY);else{const k=.5*(M[0].pageX+M[1].pageX),Z=.5*(M[0].pageY+M[1].pageY);f.set(k,Z)}}function he(){if(M.length===1)x.set(M[0].pageX,M[0].pageY);else{const k=.5*(M[0].pageX+M[1].pageX),Z=.5*(M[0].pageY+M[1].pageY);x.set(k,Z)}}function Ee(){const k=M[0].pageX-M[1].pageX,Z=M[0].pageY-M[1].pageY,xe=Math.sqrt(k*k+Z*Z);m.set(0,xe)}function Ze(){n.enableZoom&&Ee(),n.enablePan&&he()}function U(){n.enableZoom&&Ee(),n.enableRotate&&Se()}function N(k){if(M.length==1)d.set(k.pageX,k.pageY);else{const xe=Xe(k),Re=.5*(k.pageX+xe.x),Be=.5*(k.pageY+xe.y);d.set(Re,Be)}h.subVectors(d,f).multiplyScalar(n.rotateSpeed);const Z=n.domElement;D(2*Math.PI*h.x/Z.clientHeight),F(2*Math.PI*h.y/Z.clientHeight),f.copy(d)}function K(k){if(M.length===1)p.set(k.pageX,k.pageY);else{const Z=Xe(k),xe=.5*(k.pageX+Z.x),Re=.5*(k.pageY+Z.y);p.set(xe,Re)}g.subVectors(p,x).multiplyScalar(n.panSpeed),V(g.x,g.y),x.copy(p)}function R(k){const Z=Xe(k),xe=k.pageX-Z.x,Re=k.pageY-Z.y,Be=Math.sqrt(xe*xe+Re*Re);_.set(0,Be),A.set(0,Math.pow(_.y/m.y,n.zoomSpeed)),H(A.y),m.copy(_)}function te(k){n.enableZoom&&R(k),n.enablePan&&K(k)}function oe(k){n.enableZoom&&R(k),n.enableRotate&&N(k)}function pe(k){n.enabled!==!1&&(M.length===0&&(n.domElement.setPointerCapture(k.pointerId),n.domElement.addEventListener("pointermove",ie),n.domElement.addEventListener("pointerup",me)),ee(k),k.pointerType==="touch"?X(k):se(k))}function ie(k){n.enabled!==!1&&(k.pointerType==="touch"?re(k):ve(k))}function me(k){Ue(k),M.length===0&&(n.domElement.releasePointerCapture(k.pointerId),n.domElement.removeEventListener("pointermove",ie),n.domElement.removeEventListener("pointerup",me)),n.dispatchEvent(Ym),r=s.NONE}function se(k){let Z;switch(k.button){case 0:Z=n.mouseButtons.LEFT;break;case 1:Z=n.mouseButtons.MIDDLE;break;case 2:Z=n.mouseButtons.RIGHT;break;default:Z=-1}switch(Z){case io.DOLLY:if(n.enableZoom===!1)return;Y(k),r=s.DOLLY;break;case io.ROTATE:if(k.ctrlKey||k.metaKey||k.shiftKey){if(n.enablePan===!1)return;we(k),r=s.PAN}else{if(n.enableRotate===!1)return;fe(k),r=s.ROTATE}break;case io.PAN:if(k.ctrlKey||k.metaKey||k.shiftKey){if(n.enableRotate===!1)return;fe(k),r=s.ROTATE}else{if(n.enablePan===!1)return;we(k),r=s.PAN}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(Ju)}function ve(k){switch(r){case s.ROTATE:if(n.enableRotate===!1)return;ze(k);break;case s.DOLLY:if(n.enableZoom===!1)return;ke(k);break;case s.PAN:if(n.enablePan===!1)return;We(k);break}}function I(k){n.enabled===!1||n.enableZoom===!1||r!==s.NONE||(k.preventDefault(),n.dispatchEvent(Ju),ne(k),n.dispatchEvent(Ym))}function T(k){n.enabled===!1||n.enablePan===!1||ue(k)}function X(k){switch(ye(k),M.length){case 1:switch(n.touches.ONE){case so.ROTATE:if(n.enableRotate===!1)return;Se(),r=s.TOUCH_ROTATE;break;case so.PAN:if(n.enablePan===!1)return;he(),r=s.TOUCH_PAN;break;default:r=s.NONE}break;case 2:switch(n.touches.TWO){case so.DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;Ze(),r=s.TOUCH_DOLLY_PAN;break;case so.DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;U(),r=s.TOUCH_DOLLY_ROTATE;break;default:r=s.NONE}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(Ju)}function re(k){switch(ye(k),r){case s.TOUCH_ROTATE:if(n.enableRotate===!1)return;N(k),n.update();break;case s.TOUCH_PAN:if(n.enablePan===!1)return;K(k),n.update();break;case s.TOUCH_DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;te(k),n.update();break;case s.TOUCH_DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;oe(k),n.update();break;default:r=s.NONE}}function de(k){n.enabled!==!1&&k.preventDefault()}function ee(k){M.push(k)}function Ue(k){delete E[k.pointerId];for(let Z=0;Z<M.length;Z++)if(M[Z].pointerId==k.pointerId){M.splice(Z,1);return}}function ye(k){let Z=E[k.pointerId];Z===void 0&&(Z=new Ke,E[k.pointerId]=Z),Z.set(k.pageX,k.pageY)}function Xe(k){const Z=k.pointerId===M[0].pointerId?M[1]:M[0];return E[Z.pointerId]}n.domElement.addEventListener("contextmenu",de),n.domElement.addEventListener("pointerdown",pe),n.domElement.addEventListener("pointercancel",me),n.domElement.addEventListener("wheel",I,{passive:!1}),this.update()}}const SE=(i,e,t,n,s)=>{const r=performance.now();let o=i.style.display==="none"?0:parseFloat(i.style.opacity);isNaN(o)&&(o=1);const a=window.setInterval(()=>{const c=performance.now()-r;let u=Math.min(c/n,1);u>.999&&(u=1);let f;e?(f=(1-u)*o,f<1e-4&&(f=0)):f=(1-o)*u+o,f>0?(i.style.display=t,i.style.opacity=f):i.style.display="none",u>=1&&(s&&s(),window.clearInterval(a))},16);return a},yE=500;class bh{static elementIDGen=0;constructor(e,t){this.taskIDGen=0,this.elementID=bh.elementIDGen++,this.tasks=[],this.message=e||"Loading...",this.container=t||document.body,this.spinnerContainerOuter=document.createElement("div"),this.spinnerContainerOuter.className=`spinnerOuterContainer${this.elementID}`,this.spinnerContainerOuter.style.display="none",this.spinnerContainerPrimary=document.createElement("div"),this.spinnerContainerPrimary.className=`spinnerContainerPrimary${this.elementID}`,this.spinnerPrimary=document.createElement("div"),this.spinnerPrimary.classList.add(`spinner${this.elementID}`,`spinnerPrimary${this.elementID}`),this.messageContainerPrimary=document.createElement("div"),this.messageContainerPrimary.classList.add(`messageContainer${this.elementID}`,`messageContainerPrimary${this.elementID}`),this.messageContainerPrimary.innerHTML=this.message,this.spinnerContainerMin=document.createElement("div"),this.spinnerContainerMin.className=`spinnerContainerMin${this.elementID}`,this.spinnerMin=document.createElement("div"),this.spinnerMin.classList.add(`spinner${this.elementID}`,`spinnerMin${this.elementID}`),this.messageContainerMin=document.createElement("div"),this.messageContainerMin.classList.add(`messageContainer${this.elementID}`,`messageContainerMin${this.elementID}`),this.messageContainerMin.innerHTML=this.message,this.spinnerContainerPrimary.appendChild(this.spinnerPrimary),this.spinnerContainerPrimary.appendChild(this.messageContainerPrimary),this.spinnerContainerOuter.appendChild(this.spinnerContainerPrimary),this.spinnerContainerMin.appendChild(this.spinnerMin),this.spinnerContainerMin.appendChild(this.messageContainerMin),this.spinnerContainerOuter.appendChild(this.spinnerContainerMin);const n=document.createElement("style");n.innerHTML=`

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

        `,this.spinnerContainerOuter.appendChild(n),this.container.appendChild(this.spinnerContainerOuter),this.setMinimized(!1,!0),this.fadeTransitions=[]}addTask(e){const t={message:e,id:this.taskIDGen++};return this.tasks.push(t),this.update(),t.id}removeTask(e){let t=0;for(let n of this.tasks){if(n.id===e){this.tasks.splice(t,1);break}t++}this.update()}removeAllTasks(){this.tasks=[],this.update()}setMessageForTask(e,t){for(let n of this.tasks)if(n.id===e){n.message=t;break}this.update()}update(){this.tasks.length>0?(this.show(),this.setMessage(this.tasks[this.tasks.length-1].message)):this.hide()}show(){this.spinnerContainerOuter.style.display="block",this.visible=!0}hide(){this.spinnerContainerOuter.style.display="none",this.visible=!1}setContainer(e){this.container&&this.spinnerContainerOuter.parentElement===this.container&&this.container.removeChild(this.spinnerContainerOuter),e&&(this.container=e,this.container.appendChild(this.spinnerContainerOuter),this.spinnerContainerOuter.style.zIndex=this.container.style.zIndex+1)}setMinimized(e,t){const n=(s,r,o,a,l)=>{o?s.style.display=r?a:"none":this.fadeTransitions[l]=SE(s,!r,a,yE,()=>{this.fadeTransitions[l]=null})};n(this.spinnerContainerPrimary,!e,t,"block",0),n(this.spinnerContainerMin,e,t,"flex",1),this.minimized=e}setMessage(e){this.messageContainerPrimary.innerHTML=e,this.messageContainerMin.innerHTML=e}}class bE{constructor(e){this.idGen=0,this.tasks=[],this.container=e||document.body,this.progressBarContainerOuter=document.createElement("div"),this.progressBarContainerOuter.className="progressBarOuterContainer",this.progressBarContainerOuter.style.display="none",this.progressBarBox=document.createElement("div"),this.progressBarBox.className="progressBarBox",this.progressBarBackground=document.createElement("div"),this.progressBarBackground.className="progressBarBackground",this.progressBar=document.createElement("div"),this.progressBar.className="progressBar",this.progressBarBackground.appendChild(this.progressBar),this.progressBarBox.appendChild(this.progressBarBackground),this.progressBarContainerOuter.appendChild(this.progressBarBox);const t=document.createElement("style");t.innerHTML=`

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

        `,this.progressBarContainerOuter.appendChild(t),this.container.appendChild(this.progressBarContainerOuter)}show(){this.progressBarContainerOuter.style.display="block"}hide(){this.progressBarContainerOuter.style.display="none"}setProgress(e){this.progressBar.style.width=e+"%"}setContainer(e){this.container&&this.progressBarContainerOuter.parentElement===this.container&&this.container.removeChild(this.progressBarContainerOuter),e&&(this.container=e,this.container.appendChild(this.progressBarContainerOuter),this.progressBarContainerOuter.style.zIndex=this.container.style.zIndex+1)}}class ME{constructor(e){this.container=e||document.body,this.infoCells={};const t=[["Camera position","cameraPosition"],["Camera look-at","cameraLookAt"],["Camera up","cameraUp"],["Camera mode","orthographicCamera"],["Cursor position","cursorPosition"],["FPS","fps"],["Rendering:","renderSplatCount"],["Sort time","sortTime"],["Render window","renderWindow"],["Focal adjustment","focalAdjustment"],["Splat scale","splatScale"],["Point cloud mode","pointCloudMode"]];this.infoPanelContainer=document.createElement("div");const n=document.createElement("style");n.innerHTML=`

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

        `,this.infoPanelContainer.append(n),this.infoPanel=document.createElement("div"),this.infoPanel.className="infoPanel";const s=document.createElement("div");s.style.display="table";for(let r of t){const o=document.createElement("div");o.style.display="table-row",o.className="info-panel-row";const a=document.createElement("div");a.style.display="table-cell",a.innerHTML=`${r[0]}: `,a.classList.add("info-panel-cell","label-cell");const l=document.createElement("div");l.style.display="table-cell",l.style.width="10px",l.innerHTML=" ",l.className="info-panel-cell";const c=document.createElement("div");c.style.display="table-cell",c.innerHTML="",c.className="info-panel-cell",this.infoCells[r[1]]=c,o.appendChild(a),o.appendChild(l),o.appendChild(c),s.appendChild(o)}this.infoPanel.appendChild(s),this.infoPanelContainer.append(this.infoPanel),this.infoPanelContainer.style.display="none",this.container.appendChild(this.infoPanelContainer),this.visible=!1}update=function(e,t,n,s,r,o,a,l,c,u,f,d,h,x){const p=`${t.x.toFixed(5)}, ${t.y.toFixed(5)}, ${t.z.toFixed(5)}`;if(this.infoCells.cameraPosition.innerHTML!==p&&(this.infoCells.cameraPosition.innerHTML=p),n){const m=n,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cameraLookAt.innerHTML!==_&&(this.infoCells.cameraLookAt.innerHTML=_)}const g=`${s.x.toFixed(5)}, ${s.y.toFixed(5)}, ${s.z.toFixed(5)}`;if(this.infoCells.cameraUp.innerHTML!==g&&(this.infoCells.cameraUp.innerHTML=g),this.infoCells.orthographicCamera.innerHTML=r?"Orthographic":"Perspective",o){const m=o,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cursorPosition.innerHTML=_}else this.infoCells.cursorPosition.innerHTML="N/A";this.infoCells.fps.innerHTML=a,this.infoCells.renderWindow.innerHTML=`${e.x} x ${e.y}`,this.infoCells.renderSplatCount.innerHTML=`${c} splats out of ${l} (${u.toFixed(2)}%)`,this.infoCells.sortTime.innerHTML=`${f.toFixed(3)} ms`,this.infoCells.focalAdjustment.innerHTML=`${d.toFixed(3)}`,this.infoCells.splatScale.innerHTML=`${h.toFixed(3)}`,this.infoCells.pointCloudMode.innerHTML=`${x}`};setContainer(e){this.container&&this.infoPanelContainer.parentElement===this.container&&this.container.removeChild(this.infoPanelContainer),e&&(this.container=e,this.container.appendChild(this.infoPanelContainer),this.infoPanelContainer.style.zIndex=this.container.style.zIndex+1)}show(){this.infoPanelContainer.style.display="block",this.visible=!0}hide(){this.infoPanelContainer.style.display="none",this.visible=!1}}const Km=new B;class CE extends mn{constructor(e=new B(0,0,1),t=new B(0,0,0),n=1,s=.1,r=16776960,o=n*.2,a=o*.2){super(),this.type="ArrowHelper";const l=new nl(s,s,n,32);l.translate(0,n/2,0);const c=new nl(0,a,o,32);c.translate(0,n,0),this.position.copy(t),this.line=new hn(l,new Kr({color:r,toneMapped:!1})),this.line.matrixAutoUpdate=!1,this.add(this.line),this.cone=new hn(c,new Kr({color:r,toneMapped:!1})),this.cone.matrixAutoUpdate=!1,this.add(this.cone),this.setDirection(e)}setDirection(e){if(e.y>.99999)this.quaternion.set(0,0,0,1);else if(e.y<-.99999)this.quaternion.set(1,0,0,0);else{Km.set(e.z,0,-e.x).normalize();const t=Math.acos(e.y);this.quaternion.setFromAxisAngle(Km,t)}}setColor(e){this.line.material.color.set(e),this.cone.material.color.set(e)}copy(e){return super.copy(e,!1),this.line.copy(e.line),this.cone.copy(e.cone),this}dispose(){this.line.geometry.dispose(),this.line.material.dispose(),this.cone.geometry.dispose(),this.cone.material.dispose()}}class Oa{constructor(e){this.threeScene=e,this.splatRenderTarget=null,this.renderTargetCopyQuad=null,this.renderTargetCopyCamera=null,this.meshCursor=null,this.focusMarker=null,this.controlPlane=null,this.debugRoot=null,this.secondaryDebugRoot=null}updateSplatRenderTargetForRenderDimensions(e,t){this.destroySplatRendertarget(),this.splatRenderTarget=new cr(e,t,{format:Xn,stencilBuffer:!1,depthBuffer:!0}),this.splatRenderTarget.depthTexture=new rh(e,t),this.splatRenderTarget.depthTexture.format=Yo,this.splatRenderTarget.depthTexture.type=Ii}destroySplatRendertarget(){this.splatRenderTarget&&(this.splatRenderTarget=null)}setupRenderTargetCopyObjects(){const e={sourceColorTexture:{type:"t",value:null},sourceDepthTexture:{type:"t",value:null}},t=new Yn({vertexShader:`
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
            `,uniforms:e,depthWrite:!1,depthTest:!1,transparent:!0,blending:Ig,blendSrc:Qa,blendSrcAlpha:Qa,blendDst:Ka,blendDstAlpha:Ka});t.extensions.fragDepth=!0,this.renderTargetCopyQuad=new hn(new jo(2,2),t),this.renderTargetCopyCamera=new lh(-1,1,1,-1,0,1)}destroyRenderTargetCopyObjects(){this.renderTargetCopyQuad&&(Co(this.renderTargetCopyQuad),this.renderTargetCopyQuad=null)}setupMeshCursor(){if(!this.meshCursor){const e=new oh(.5,1.5,32),t=new Kr({color:16777215}),n=new hn(e,t);n.rotation.set(0,0,Math.PI),n.position.set(0,1,0);const s=new hn(e,t);s.position.set(0,-1,0);const r=new hn(e,t);r.rotation.set(0,0,Math.PI/2),r.position.set(1,0,0);const o=new hn(e,t);o.rotation.set(0,0,-Math.PI/2),o.position.set(-1,0,0),this.meshCursor=new mn,this.meshCursor.add(n),this.meshCursor.add(s),this.meshCursor.add(r),this.meshCursor.add(o),this.meshCursor.scale.set(.1,.1,.1),this.threeScene.add(this.meshCursor),this.meshCursor.visible=!1}}destroyMeshCursor(){this.meshCursor&&(Co(this.meshCursor),this.threeScene.remove(this.meshCursor),this.meshCursor=null)}setMeshCursorVisibility(e){this.meshCursor.visible=e}getMeschCursorVisibility(){return this.meshCursor.visible}setMeshCursorPosition(e){this.meshCursor.position.copy(e)}positionAndOrientMeshCursor(e,t){this.meshCursor.position.copy(e),this.meshCursor.up.copy(t.up),this.meshCursor.lookAt(t.position)}setupFocusMarker(){if(!this.focusMarker){const e=new Tc(.5,32,32),t=Oa.buildFocusMarkerMaterial();t.depthTest=!1,t.depthWrite=!1,t.transparent=!0,this.focusMarker=new hn(e,t)}}destroyFocusMarker(){this.focusMarker&&(Co(this.focusMarker),this.focusMarker=null)}updateFocusMarker=(function(){const e=new B,t=new st,n=new B;return function(s,r,o){t.copy(r.matrixWorld).invert(),e.copy(s).applyMatrix4(t),e.normalize().multiplyScalar(10),e.applyMatrix4(r.matrixWorld),n.copy(r.position).sub(s);const a=n.length();this.focusMarker.position.copy(s),this.focusMarker.scale.set(a,a,a),this.focusMarker.material.uniforms.realFocusPosition.value.copy(s),this.focusMarker.material.uniforms.viewport.value.copy(o),this.focusMarker.material.uniformsNeedUpdate=!0}})();setFocusMarkerVisibility(e){this.focusMarker.visible=e}setFocusMarkerOpacity(e){this.focusMarker.material.uniforms.opacity.value=e,this.focusMarker.material.uniformsNeedUpdate=!0}getFocusMarkerOpacity(){return this.focusMarker.material.uniforms.opacity.value}setupControlPlane(){if(!this.controlPlane){const e=new jo(1,1);e.rotateX(-Math.PI/2);const t=new Kr({color:16777215});t.transparent=!0,t.opacity=.6,t.depthTest=!1,t.depthWrite=!1,t.side=Ei;const n=new hn(e,t),s=new B(0,1,0);s.normalize();const r=new B(0,0,0),o=.5,a=.01,l=56576,c=new CE(s,r,o,a,l,.1,.03);this.controlPlane=new mn,this.controlPlane.add(n),this.controlPlane.add(c)}}destroyControlPlane(){this.controlPlane&&(Co(this.controlPlane),this.controlPlane=null)}setControlPlaneVisibility(e){this.controlPlane.visible=e}positionAndOrientControlPlane=(function(){const e=new Vt,t=new B(0,1,0);return function(n,s){e.setFromUnitVectors(t,s),this.controlPlane.position.copy(n),this.controlPlane.quaternion.copy(e)}})();addDebugMeshes(){this.debugRoot=this.createDebugMeshes(),this.secondaryDebugRoot=this.createSecondaryDebugMeshes(),this.threeScene.add(this.debugRoot),this.threeScene.add(this.secondaryDebugRoot)}destroyDebugMeshes(){for(let e of[this.debugRoot,this.secondaryDebugRoot])e&&(Co(e),this.threeScene.remove(e));this.debugRoot=null,this.secondaryDebugRoot=null}createDebugMeshes(e){const t=new Tc(1,32,32),n=new mn,s=(r,o)=>{let a=new hn(t,Oa.buildDebugMaterial(r));a.renderOrder=e,n.add(a),a.position.fromArray(o)};return s(16711680,[-50,0,0]),s(16711680,[50,0,0]),s(65280,[0,0,-50]),s(65280,[0,0,50]),s(16755200,[5,0,5]),n}createSecondaryDebugMeshes(e){const t=new ra(3,3,3),n=new mn;let s=12303291;const r=a=>{let l=new hn(t,Oa.buildDebugMaterial(s));l.renderOrder=e,n.add(l),l.position.fromArray(a)};let o=10;return r([-o,0,-o]),r([-o,0,o]),r([o,0,-o]),r([o,0,o]),n}static buildDebugMaterial(e){const t=`
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
        `,s={color:{type:"v3",value:new bt(e)}},r=new Yn({uniforms:s,vertexShader:t,fragmentShader:n,transparent:!1,depthTest:!0,depthWrite:!0,side:os});return r.extensions.fragDepth=!0,r}static buildFocusMarkerMaterial(e){const t=`
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
        `,s={color:{type:"v3",value:new bt(e)},realFocusPosition:{type:"v3",value:new B},viewport:{type:"v2",value:new Ke},opacity:{value:0}};return new Yn({uniforms:s,vertexShader:t,fragmentShader:n,transparent:!0,depthTest:!1,depthWrite:!1,side:os})}dispose(){this.destroyMeshCursor(),this.destroyFocusMarker(),this.destroyDebugMeshes(),this.destroyControlPlane(),this.destroyRenderTargetCopyObjects(),this.destroySplatRendertarget()}}const TE=new B(1,0,0),EE=new B(0,1,0),wE=new B(0,0,1);class ef{constructor(e=new B,t=new B){this.origin=new B,this.direction=new B,this.setParameters(e,t)}setParameters(e,t){this.origin.copy(e),this.direction.copy(t).normalize()}boxContainsPoint(e,t,n){return!(t.x<e.min.x-n||t.x>e.max.x+n||t.y<e.min.y-n||t.y>e.max.y+n||t.z<e.min.z-n||t.z>e.max.z+n)}intersectBox=(function(){const e=new B,t=[],n=[],s=[];return function(r,o){if(n[0]=this.origin.x,n[1]=this.origin.y,n[2]=this.origin.z,s[0]=this.direction.x,s[1]=this.direction.y,s[2]=this.direction.z,this.boxContainsPoint(r,this.origin,1e-4))return o&&(o.origin.copy(this.origin),o.normal.set(0,0,0),o.distance=-1),!0;for(let a=0;a<3;a++){if(s[a]==0)continue;const l=a==0?TE:a==1?EE:wE,c=s[a]<0?r.max:r.min;let u=-Math.sign(s[a]);t[0]=a==0?c.x:a==1?c.y:c.z;let f=t[0]-n[a];if(f*u<0){const d=(a+1)%3,h=(a+2)%3;if(t[2]=s[d]/s[a]*f+n[d],t[1]=s[h]/s[a]*f+n[h],e.set(t[a],t[h],t[d]),this.boxContainsPoint(r,e,1e-4))return o&&(o.origin.copy(e),o.normal.copy(l).multiplyScalar(u),o.distance=e.sub(this.origin).length()),!0}}return!1}})();intersectSphere=(function(){const e=new B;return function(t,n,s){e.copy(t).sub(this.origin);const r=e.dot(this.direction),o=r*r,l=e.dot(e)-o,c=n*n;if(l>c)return!1;const u=Math.sqrt(c-l),f=r-u,d=r+u;if(d<0)return!1;let h=f<0?d:f;return s&&(s.origin.copy(this.origin).addScaledVector(this.direction,h),s.normal.copy(s.origin).sub(t).normalize(),s.distance=h),!0}})()}class Mh{constructor(){this.origin=new B,this.normal=new B,this.distance=0,this.splatIndex=0}set(e,t,n,s){this.origin.copy(e),this.normal.copy(t),this.distance=n,this.splatIndex=s}clone(){const e=new Mh;return e.origin.copy(this.origin),e.normal.copy(this.normal),e.distance=this.distance,e.splatIndex=this.splatIndex,e}}const Cs={ThreeD:0,TwoD:1};class RE{constructor(e,t,n=!1){this.ray=new ef(e,t),this.raycastAgainstTrueSplatEllipsoid=n}setFromCameraAndScreenPosition=(function(){const e=new Ke;return function(t,n,s){if(e.x=n.x/s.x*2-1,e.y=(s.y-n.y)/s.y*2-1,t.isPerspectiveCamera)this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(e.x,e.y,.5).unproject(t).sub(this.ray.origin).normalize(),this.camera=t;else if(t.isOrthographicCamera)this.ray.origin.set(e.x,e.y,(t.near+t.far)/(t.near-t.far)).unproject(t),this.ray.direction.set(0,0,-1).transformDirection(t.matrixWorld),this.camera=t;else throw new Error("Raycaster::setFromCameraAndScreenPosition() -> Unsupported camera type")}})();intersectSplatMesh=(function(){const e=new st,t=new st,n=new st,s=new ef,r=new B;return function(o,a=[]){const l=o.getSplatTree();if(l){for(let c=0;c<l.subTrees.length;c++){const u=l.subTrees[c];t.copy(o.matrixWorld),o.dynamicMode&&(o.getSceneTransform(c,n),t.multiply(n)),e.copy(t).invert(),s.origin.copy(this.ray.origin).applyMatrix4(e),s.direction.copy(this.ray.origin).add(this.ray.direction),s.direction.applyMatrix4(e).sub(s.origin).normalize();const f=[];u.rootNode&&this.castRayAtSplatTreeNode(s,l,u.rootNode,f),f.forEach(d=>{d.origin.applyMatrix4(t),d.normal.applyMatrix4(t).normalize(),d.distance=r.copy(d.origin).sub(this.ray.origin).length()}),a.push(...f)}return a.sort((c,u)=>c.distance>u.distance?1:-1),a}}})();castRayAtSplatTreeNode=(function(){const e=new Jt,t=new B,n=new B,s=new Vt,r=new Mh,o=1e-7,a=new B(0,0,0),l=new st,c=new st,u=new st,f=new st,d=new st,h=new ef;return function(x,p,g,m=[]){if(x.intersectBox(g.boundingBox)){if(g.data&&g.data.indexes&&g.data.indexes.length>0)for(let _=0;_<g.data.indexes.length;_++){const A=g.data.indexes[_],v=p.splatMesh.getSceneIndexForSplat(A);if(p.splatMesh.getScene(v).visible&&(p.splatMesh.getSplatColor(A,e),p.splatMesh.getSplatCenter(A,t),p.splatMesh.getSplatScaleAndRotation(A,n,s),!(n.x<=o||n.y<=o||p.splatMesh.splatRenderMode===Cs.ThreeD&&n.z<=o)))if(this.raycastAgainstTrueSplatEllipsoid){c.makeScale(n.x,n.y,n.z),u.makeRotationFromQuaternion(s);const y=Math.log10(e.w)*2;if(l.makeScale(y,y,y),d.copy(l).multiply(u).multiply(c),f.copy(d).invert(),h.origin.copy(x.origin).sub(t).applyMatrix4(f),h.direction.copy(x.origin).add(x.direction).sub(t),h.direction.applyMatrix4(f).sub(h.origin).normalize(),h.intersectSphere(a,1,r)){const M=r.clone();M.splatIndex=A,M.origin.applyMatrix4(d).add(t),m.push(M)}}else{let y=n.x+n.y,M=2;if(p.splatMesh.splatRenderMode===Cs.ThreeD&&(y+=n.z,M=3),y=y/M,x.intersectSphere(t,y,r)){const E=r.clone();E.splatIndex=A,m.push(E)}}}if(g.children&&g.children.length>0)for(let _ of g.children)this.castRayAtSplatTreeNode(x,p,_,m);return m}}})()}class Oo{static buildVertexShaderBase(e=!1,t=!1,n=0,s=""){let r=`
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
            uniform float sceneOpacity[${Nt.MaxScenes}];
            uniform int sceneVisibility[${Nt.MaxScenes}];
        `),e&&(r+=`
            uniform highp mat4 transforms[${Nt.MaxScenes}];
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
        uniform float sphericalHarmonics8BitCompressionRangeMin[${Nt.MaxScenes}];
        uniform float sphericalHarmonics8BitCompressionRangeMax[${Nt.MaxScenes}];

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
        `}static getUniforms(e=!1,t=!1,n=0,s=1,r=!1){const o={sceneCenter:{type:"v3",value:new B},fadeInComplete:{type:"i",value:0},orthographicMode:{type:"i",value:0},visibleRegionFadeStartRadius:{type:"f",value:0},visibleRegionRadius:{type:"f",value:0},currentTime:{type:"f",value:0},firstRenderTime:{type:"f",value:0},centersColorsTexture:{type:"t",value:null},sphericalHarmonicsTexture:{type:"t",value:null},sphericalHarmonicsTextureR:{type:"t",value:null},sphericalHarmonicsTextureG:{type:"t",value:null},sphericalHarmonicsTextureB:{type:"t",value:null},sphericalHarmonics8BitCompressionRangeMin:{type:"f",value:[]},sphericalHarmonics8BitCompressionRangeMax:{type:"f",value:[]},focal:{type:"v2",value:new Ke},orthoZoom:{type:"f",value:1},inverseFocalAdjustment:{type:"f",value:1},viewport:{type:"v2",value:new Ke},basisViewport:{type:"v2",value:new Ke},debugColor:{type:"v3",value:new bt},centersColorsTextureSize:{type:"v2",value:new Ke(1024,1024)},sphericalHarmonicsDegree:{type:"i",value:n},sphericalHarmonicsTextureSize:{type:"v2",value:new Ke(1024,1024)},sphericalHarmonics8BitMode:{type:"i",value:0},sphericalHarmonicsMultiTextureMode:{type:"i",value:0},splatScale:{type:"f",value:s},pointCloudModeEnabled:{type:"i",value:r?1:0},sceneIndexesTexture:{type:"t",value:null},sceneIndexesTextureSize:{type:"v2",value:new Ke(1024,1024)},sceneCount:{type:"i",value:1}};for(let a=0;a<Nt.MaxScenes;a++)o.sphericalHarmonics8BitCompressionRangeMin.value.push(-3/2),o.sphericalHarmonics8BitCompressionRangeMax.value.push(Nt.SphericalHarmonics8BitCompressionRange/2);if(t){const a=[];for(let c=0;c<Nt.MaxScenes;c++)a.push(1);o.sceneOpacity={type:"f",value:a};const l=[];for(let c=0;c<Nt.MaxScenes;c++)l.push(1);o.sceneVisibility={type:"i",value:l}}if(e){const a=[];for(let l=0;l<Nt.MaxScenes;l++)a.push(new st);o.transforms={type:"mat4",value:a}}return o}}class wc{static build(e=!1,t=!1,n=!1,s=2048,r=1,o=!1,a=0,l=.3){let u=Oo.buildVertexShaderBase(e,t,a,`
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
        `);u+=wc.buildVertexShaderProjection(n,t,s,l);const f=wc.buildFragmentShader(),d=Oo.getUniforms(e,t,a,r,o);return d.covariancesTextureSize={type:"v2",value:new Ke(1024,1024)},d.covariancesTexture={type:"t",value:null},d.covariancesTextureHalfFloat={type:"t",value:null},d.covariancesAreHalfFloat={type:"i",value:0},new Yn({uniforms:d,vertexShader:u,fragmentShader:f,transparent:!0,alphaTest:1,blending:ir,depthTest:!0,depthWrite:!1,side:Ei})}static buildVertexShaderProjection(e,t,n,s){let r=`

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
        `,r+=Oo.getVertexShaderFadeIn(),r+="}",r}static buildFragmentShader(){let e=`
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
        `,e}}class Rc{static build(e=!1,t=!1,n=1,s=!1,r=0){let a=Oo.buildVertexShaderBase(e,t,r,`
            uniform vec2 scaleRotationsTextureSize;
            uniform highp sampler2D scaleRotationsTexture;
            varying mat3 vT;
            varying vec2 vQuadCenter;
            varying vec2 vFragCoord;
        `);a+=Rc.buildVertexShaderProjection();const l=Rc.buildFragmentShader(),c=Oo.getUniforms(e,t,r,n,s);return c.scaleRotationsTexture={type:"t",value:null},c.scaleRotationsTextureSize={type:"v2",value:new Ke(1024,1024)},new Yn({uniforms:c,vertexShader:a,fragmentShader:l,transparent:!0,alphaTest:1,blending:ir,depthTest:!0,depthWrite:!1,side:Ei})}static buildVertexShaderProjection(){let e=`

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
            `,e+=Oo.getVertexShaderFadeIn(),e+="}",e}static buildFragmentShader(){return`
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
        `}}class IE{static build(e){const t=new Kn;t.setIndex([0,1,2,0,2,3]);const n=new Float32Array(12),s=new Li(n,3);t.setAttribute("position",s),s.setXYZ(0,-1,-1,0),s.setXYZ(1,-1,1,0),s.setXYZ(2,1,1,0),s.setXYZ(3,1,-1,0),s.needsUpdate=!0;const r=new Ly().copy(t),o=new Uint32Array(e),a=new Cy(o,1,!1);return a.setUsage(PS),r.setAttribute("splatIndex",a),r.instanceCount=0,r}}class DE extends mn{constructor(e,t=new B,n=new Vt,s=new B(1,1,1),r=1,o=1,a=!0){super(),this.splatBuffer=e,this.position.copy(t),this.quaternion.copy(n),this.scale.copy(s),this.transform=new st,this.minimumAlpha=r,this.opacity=o,this.visible=a}copyTransformData(e){this.position.copy(e.position),this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.transform.copy(e.transform)}updateTransform(e){e?(this.matrixWorldAutoUpdate&&this.updateWorldMatrix(!0,!1),this.transform.copy(this.matrixWorld)):(this.matrixAutoUpdate&&this.updateMatrix(),this.transform.copy(this.matrix))}}class Ch{static idGen=0;constructor(e,t,n,s){this.min=new B().copy(e),this.max=new B().copy(t),this.boundingBox=new Ji(this.min,this.max),this.center=new B().copy(this.max).sub(this.min).multiplyScalar(.5).add(this.min),this.depth=n,this.children=[],this.data=null,this.id=s||Ch.idGen++}}class Na{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.sceneDimensions=new B,this.sceneMin=new B,this.sceneMax=new B,this.rootNode=null,this.nodesWithIndexes=[],this.splatMesh=null}static convertWorkerSubTreeNode(e){const t=new B().fromArray(e.min),n=new B().fromArray(e.max),s=new Ch(t,n,e.depth,e.id);if(e.data.indexes){s.data={indexes:[]};for(let r of e.data.indexes)s.data.indexes.push(r)}if(e.children)for(let r of e.children)s.children.push(Na.convertWorkerSubTreeNode(r));return s}static convertWorkerSubTree(e,t){const n=new Na(e.maxDepth,e.maxCentersPerNode);n.sceneMin=new B().fromArray(e.sceneMin),n.sceneMax=new B().fromArray(e.sceneMax),n.splatMesh=t,n.rootNode=Na.convertWorkerSubTreeNode(e.rootNode);const s=(r,o)=>{r.children.length===0&&o(r);for(let a of r.children)s(a,o)};return n.nodesWithIndexes=[],s(n.rootNode,r=>{r.data&&r.data.indexes&&r.data.indexes.length>0&&n.nodesWithIndexes.push(r)}),n}}function PE(i){let e=0;class t{constructor(l,c){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]]}containsPoint(l){return l[0]>=this.min[0]&&l[0]<=this.max[0]&&l[1]>=this.min[1]&&l[1]<=this.max[1]&&l[2]>=this.min[2]&&l[2]<=this.max[2]}}class n{constructor(l,c){this.maxDepth=l,this.maxCentersPerNode=c,this.sceneDimensions=[],this.sceneMin=[],this.sceneMax=[],this.rootNode=null,this.addedIndexes={},this.nodesWithIndexes=[],this.splatMesh=null,this.disposed=!1}}class s{constructor(l,c,u,f){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]],this.center=[(c[0]-l[0])*.5+l[0],(c[1]-l[1])*.5+l[1],(c[2]-l[2])*.5+l[2]],this.depth=u,this.children=[],this.data=null,this.id=f||e++}}processSplatTreeNode=function(a,l,c,u){const f=l.data.indexes.length;if(f<a.maxCentersPerNode||l.depth>a.maxDepth){const _=[];for(let A=0;A<l.data.indexes.length;A++)a.addedIndexes[l.data.indexes[A]]||(_.push(l.data.indexes[A]),a.addedIndexes[l.data.indexes[A]]=!0);l.data.indexes=_,l.data.indexes.sort((A,v)=>A>v?1:-1),a.nodesWithIndexes.push(l);return}const d=[l.max[0]-l.min[0],l.max[1]-l.min[1],l.max[2]-l.min[2]],h=[d[0]*.5,d[1]*.5,d[2]*.5],x=[l.min[0]+h[0],l.min[1]+h[1],l.min[2]+h[2]],p=[new t([x[0]-h[0],x[1],x[2]-h[2]],[x[0],x[1]+h[1],x[2]]),new t([x[0],x[1],x[2]-h[2]],[x[0]+h[0],x[1]+h[1],x[2]]),new t([x[0],x[1],x[2]],[x[0]+h[0],x[1]+h[1],x[2]+h[2]]),new t([x[0]-h[0],x[1],x[2]],[x[0],x[1]+h[1],x[2]+h[2]]),new t([x[0]-h[0],x[1]-h[1],x[2]-h[2]],[x[0],x[1],x[2]]),new t([x[0],x[1]-h[1],x[2]-h[2]],[x[0]+h[0],x[1],x[2]]),new t([x[0],x[1]-h[1],x[2]],[x[0]+h[0],x[1],x[2]+h[2]]),new t([x[0]-h[0],x[1]-h[1],x[2]],[x[0],x[1],x[2]+h[2]])],g=[];for(let _=0;_<p.length;_++)g[_]=[];const m=[0,0,0];for(let _=0;_<f;_++){const A=l.data.indexes[_],v=c[A];m[0]=u[v],m[1]=u[v+1],m[2]=u[v+2];for(let S=0;S<p.length;S++)p[S].containsPoint(m)&&g[S].push(A)}for(let _=0;_<p.length;_++){const A=new s(p[_].min,p[_].max,l.depth+1);A.data={indexes:g[_]},l.children.push(A)}l.data={};for(let _ of l.children)processSplatTreeNode(a,_,c,u)};const r=(a,l,c)=>{const u=[0,0,0],f=[0,0,0],d=[],h=Math.floor(a.length/4);for(let p=0;p<h;p++){const g=p*4,m=a[g],_=a[g+1],A=a[g+2],v=Math.round(a[g+3]);(p===0||m<u[0])&&(u[0]=m),(p===0||m>f[0])&&(f[0]=m),(p===0||_<u[1])&&(u[1]=_),(p===0||_>f[1])&&(f[1]=_),(p===0||A<u[2])&&(u[2]=A),(p===0||A>f[2])&&(f[2]=A),d.push(v)}const x=new n(l,c);return x.sceneMin=u,x.sceneMax=f,x.rootNode=new s(x.sceneMin,x.sceneMax,0),x.rootNode.data={indexes:d},x};function o(a,l,c){const u=[];for(let d of a){const h=Math.floor(d.length/4);for(let x=0;x<h;x++){const p=x*4,g=Math.round(d[p+3]);u[g]=p}}const f=[];for(let d of a){const h=r(d,l,c);f.push(h),processSplatTreeNode(h,h.rootNode,u,d)}i.postMessage({subTrees:f})}i.onmessage=a=>{a.data.process&&o(a.data.process.centers,a.data.process.maxDepth,a.data.process.maxCentersPerNode)}}function FE(i,e,t,n,s){i.postMessage({process:{centers:e,maxDepth:n,maxCentersPerNode:s}},t)}function LE(){return new Worker(URL.createObjectURL(new Blob(["(",PE.toString(),")(self)"],{type:"application/javascript"})))}class BE{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.subTrees=[],this.splatMesh=null}dispose(){this.diposeSplatTreeWorker(),this.disposed=!0}diposeSplatTreeWorker(){this.splatTreeWorker&&this.splatTreeWorker.terminate(),this.splatTreeWorker=null}processSplatMesh=function(e,t=()=>!0,n,s){this.splatTreeWorker||(this.splatTreeWorker=LE()),this.splatMesh=e,this.subTrees=[];const r=new B,o=(a,l)=>{const c=new Float32Array(l*4);let u=0;for(let f=0;f<l;f++){const d=f+a;if(t(d)){e.getSplatCenter(d,r);const h=u*4;c[h]=r.x,c[h+1]=r.y,c[h+2]=r.z,c[h+3]=d,u++}}return c};return new Promise(a=>{const l=()=>this.disposed?(this.diposeSplatTreeWorker(),a(),!0):!1;n&&n(!1),pi(()=>{if(l())return;const c=[];if(e.dynamicMode){let u=0;for(let f=0;f<e.scenes.length;f++){const h=e.getScene(f).splatBuffer.getSplatCount(),x=o(u,h);c.push(x),u+=h}}else{const u=o(0,e.getSplatCount());c.push(u)}this.splatTreeWorker.onmessage=u=>{l()||u.data.subTrees&&(s&&s(!1),pi(()=>{if(!l()){for(let f of u.data.subTrees){const d=Na.convertWorkerSubTree(f,e);this.subTrees.push(d)}this.diposeSplatTreeWorker(),s&&s(!0),pi(()=>{a()})}}))},pi(()=>{if(l())return;n&&n(!0);const u=c.map(f=>f.buffer);FE(this.splatTreeWorker,c,u,this.maxDepth,this.maxCentersPerNode)})})})};countLeaves(){let e=0;return this.visitLeaves(()=>{e++}),e}visitLeaves(e){const t=(n,s)=>{n.children.length===0&&s(n);for(let r of n.children)t(r,s)};for(let n of this.subTrees)t(n.rootNode,e)}}function UE(i){const e={};function t(n){if(e[n]!==void 0)return e[n];let s;switch(n){case"WEBGL_depth_texture":s=i.getExtension("WEBGL_depth_texture")||i.getExtension("MOZ_WEBGL_depth_texture")||i.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":s=i.getExtension("EXT_texture_filter_anisotropic")||i.getExtension("MOZ_EXT_texture_filter_anisotropic")||i.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":s=i.getExtension("WEBGL_compressed_texture_s3tc")||i.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":s=i.getExtension("WEBGL_compressed_texture_pvrtc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:s=i.getExtension(n)}return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(n){n.isWebGL2?(t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance")):(t("WEBGL_depth_texture"),t("OES_texture_float"),t("OES_texture_half_float"),t("OES_texture_half_float_linear"),t("OES_standard_derivatives"),t("OES_element_index_uint"),t("OES_vertex_array_object"),t("ANGLE_instanced_arrays")),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture")},get:function(n){const s=t(n);return s===null&&console.warn("THREE.WebGLRenderer: "+n+" extension not supported."),s}}}function OE(i,e,t){let n;function s(){if(n!==void 0)return n;if(e.has("EXT_texture_filter_anisotropic")===!0){const M=e.get("EXT_texture_filter_anisotropic");n=i.getParameter(M.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else n=0;return n}function r(M){if(M==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";M="mediump"}return M==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}const o=typeof WebGL2RenderingContext<"u"&&i.constructor.name==="WebGL2RenderingContext";let a=t.precision!==void 0?t.precision:"highp";const l=r(a);l!==a&&(console.warn("THREE.WebGLRenderer:",a,"not supported, using",l,"instead."),a=l);const c=o||e.has("WEBGL_draw_buffers"),u=t.logarithmicDepthBuffer===!0,f=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),d=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),h=i.getParameter(i.MAX_TEXTURE_SIZE),x=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),p=i.getParameter(i.MAX_VERTEX_ATTRIBS),g=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),m=i.getParameter(i.MAX_VARYING_VECTORS),_=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),A=d>0,v=o||e.has("OES_texture_float"),S=A&&v,y=o?i.getParameter(i.MAX_SAMPLES):0;return{isWebGL2:o,drawBuffers:c,getMaxAnisotropy:s,getMaxPrecision:r,precision:a,logarithmicDepthBuffer:u,maxTextures:f,maxVertexTextures:d,maxTextureSize:h,maxCubemapSize:x,maxAttributes:p,maxVertexUniforms:g,maxVaryings:m,maxFragmentUniforms:_,vertexTextures:A,floatFragmentTextures:v,floatVertexTextures:S,maxSamples:y}}const za={Default:0,Instant:2},No={None:0,Info:3},jm=new Kn,NE=new Kr,tc=6,zE=4,kE=4,HE=4,VE=6,GE=8,tf=4,nf=4,$m=1,WE=.012,XE=.003,Zm=1,Jm=16777216;class bn extends hn{constructor(e=Cs.ThreeD,t=!1,n=!1,s=!1,r=1,o=!0,a=!1,l=!1,c=1024,u=No.None,f=0,d=1,h=.3){super(jm,NE),this.renderer=void 0,this.splatRenderMode=e,this.dynamicMode=t,this.enableOptionalEffects=n,this.halfPrecisionCovariancesOnGPU=s,this.devicePixelRatio=r,this.enableDistancesComputationOnGPU=o,this.integerBasedDistancesComputation=a,this.antialiased=l,this.kernel2DSize=h,this.maxScreenSpaceSplatSize=c,this.logLevel=u,this.sphericalHarmonicsDegree=f,this.minSphericalHarmonicsDegree=0,this.sceneFadeInRateMultiplier=d,this.scenes=[],this.splatTree=null,this.baseSplatTree=null,this.splatDataTextures={},this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new Ji,this.calculatedSceneCenter=new B,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!1,this.lastRenderer=null,this.visible=!1}static buildScenes(e,t,n){const s=[];s.length=t.length;for(let r=0;r<t.length;r++){const o=t[r],a=n[r]||{};let l=a.position||[0,0,0],c=a.rotation||[0,0,0,1],u=a.scale||[1,1,1];const f=new B().fromArray(l),d=new Vt().fromArray(c),h=new B().fromArray(u),x=bn.createScene(o,f,d,h,a.splatAlphaRemovalThreshold||1,a.opacity,a.visible);e.add(x),s[r]=x}return s}static createScene(e,t,n,s,r,o=1,a=!0){return new DE(e,t,n,s,r,o,a)}static buildSplatIndexMaps(e){const t=[],n=[];let s=0;for(let r=0;r<e.length;r++){const a=e[r].getMaxSplatCount();for(let l=0;l<a;l++)t[s]=l,n[s]=r,s++}return{localSplatIndexMap:t,sceneIndexMap:n}}buildSplatTree=function(e=[],t,n){return new Promise(s=>{this.disposeSplatTree(),this.baseSplatTree=new BE(8,1e3);const r=performance.now(),o=new Jt;this.baseSplatTree.processSplatMesh(this,a=>{this.getSplatColor(a,o);const l=this.getSceneIndexForSplat(a),c=e[l]||1;return o.w>=c},t,n).then(()=>{const a=performance.now()-r;if(this.logLevel>=No.Info&&console.log("SplatTree build: "+a+" ms"),this.disposed)s();else{this.splatTree=this.baseSplatTree,this.baseSplatTree=null;let l=0,c=0,u=0;this.splatTree.visitLeaves(f=>{const d=f.data.indexes.length;d>0&&(c+=d,u++,l++)}),this.logLevel>=No.Info&&(console.log(`SplatTree leaves: ${this.splatTree.countLeaves()}`),console.log(`SplatTree leaves with splats:${l}`),c=c/u,console.log(`Avg splat count per node: ${c}`),console.log(`Total splat count: ${this.getSplatCount()}`)),s()}})})};build(e,t,n=!0,s=!1,r,o,a=!0){this.sceneOptions=t,this.finalBuild=s;const l=bn.getTotalMaxSplatCountForSplatBuffers(e),c=bn.buildScenes(this,e,t);if(n)for(let p=0;p<this.scenes.length&&p<c.length;p++){const g=c[p],m=this.getScene(p);g.copyTransformData(m)}this.scenes=c;let u=3;for(let p of e){const g=p.getMinSphericalHarmonicsDegree();g<u&&(u=g)}this.minSphericalHarmonicsDegree=Math.min(u,this.sphericalHarmonicsDegree);let f=!1;if(e.length!==this.lastBuildScenes.length)f=!0;else for(let p=0;p<e.length;p++)if(e[p]!==this.lastBuildScenes[p].splatBuffer){f=!0;break}let d=!0;if((this.scenes.length!==1||this.lastBuildSceneCount!==this.scenes.length||this.lastBuildMaxSplatCount!==l||f)&&(d=!1),!d){this.boundingBox=new Ji,a||(this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.firstRenderTime=-1),this.lastBuildScenes=[],this.lastBuildSplatCount=0,this.lastBuildMaxSplatCount=0,this.disposeMeshData(),this.geometry=IE.build(l),this.splatRenderMode===Cs.ThreeD?this.material=wc.build(this.dynamicMode,this.enableOptionalEffects,this.antialiased,this.maxScreenSpaceSplatSize,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree,this.kernel2DSize):this.material=Rc.build(this.dynamicMode,this.enableOptionalEffects,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree);const p=bn.buildSplatIndexMaps(e);this.globalSplatIndexToLocalSplatIndexMap=p.localSplatIndexMap,this.globalSplatIndexToSceneIndexMap=p.sceneIndexMap}const h=this.getSplatCount(!0);this.enableDistancesComputationOnGPU&&this.setupDistancesComputationTransformFeedback();const x=this.refreshGPUDataFromSplatBuffers(d);for(let p=0;p<this.scenes.length;p++)this.lastBuildScenes[p]=this.scenes[p];return this.lastBuildSplatCount=h,this.lastBuildMaxSplatCount=this.getMaxSplatCount(),this.lastBuildSceneCount=this.scenes.length,s&&this.scenes.length>0&&this.buildSplatTree(t.map(p=>p.splatAlphaRemovalThreshold||1),r,o).then(()=>{this.onSplatTreeReadyCallback&&this.onSplatTreeReadyCallback(this.splatTree),this.onSplatTreeReadyCallback=null}),this.visible=this.scenes.length>0,x}freeIntermediateSplatData(){const e=t=>{delete t.source.data,delete t.image,t.onUpdate=null};delete this.splatDataTextures.baseData.covariances,delete this.splatDataTextures.baseData.centers,delete this.splatDataTextures.baseData.colors,delete this.splatDataTextures.baseData.sphericalHarmonics,delete this.splatDataTextures.centerColors.data,delete this.splatDataTextures.covariances.data,this.splatDataTextures.sphericalHarmonics&&delete this.splatDataTextures.sphericalHarmonics.data,this.splatDataTextures.sceneIndexes&&delete this.splatDataTextures.sceneIndexes.data,this.splatDataTextures.centerColors.texture.needsUpdate=!0,this.splatDataTextures.centerColors.texture.onUpdate=()=>{e(this.splatDataTextures.centerColors.texture)},this.splatDataTextures.covariances.texture.needsUpdate=!0,this.splatDataTextures.covariances.texture.onUpdate=()=>{e(this.splatDataTextures.covariances.texture)},this.splatDataTextures.sphericalHarmonics&&(this.splatDataTextures.sphericalHarmonics.texture?(this.splatDataTextures.sphericalHarmonics.texture.needsUpdate=!0,this.splatDataTextures.sphericalHarmonics.texture.onUpdate=()=>{e(this.splatDataTextures.sphericalHarmonics.texture)}):this.splatDataTextures.sphericalHarmonics.textures.forEach(t=>{t.needsUpdate=!0,t.onUpdate=()=>{e(t)}})),this.splatDataTextures.sceneIndexes&&(this.splatDataTextures.sceneIndexes.texture.needsUpdate=!0,this.splatDataTextures.sceneIndexes.texture.onUpdate=()=>{e(this.splatDataTextures.sceneIndexes.texture)})}dispose(){this.disposeMeshData(),this.disposeTextures(),this.disposeSplatTree(),this.enableDistancesComputationOnGPU&&(this.computeDistancesOnGPUSyncTimeout&&(clearTimeout(this.computeDistancesOnGPUSyncTimeout),this.computeDistancesOnGPUSyncTimeout=null),this.disposeDistancesComputationGPUResources()),this.scenes=[],this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.renderer=null,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new Ji,this.calculatedSceneCenter=new B,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!0,this.lastRenderer=null,this.visible=!1}disposeMeshData(){this.geometry&&this.geometry!==jm&&(this.geometry.dispose(),this.geometry=null),this.material&&(this.material.dispose(),this.material=null)}disposeTextures(){for(let e in this.splatDataTextures)if(this.splatDataTextures.hasOwnProperty(e)){const t=this.splatDataTextures[e];t.texture&&(t.texture.dispose(),t.texture=null)}this.splatDataTextures=null}disposeSplatTree(){this.splatTree&&(this.splatTree.dispose(),this.splatTree=null),this.baseSplatTree&&(this.baseSplatTree.dispose(),this.baseSplatTree=null)}getSplatTree(){return this.splatTree}onSplatTreeReady(e){this.onSplatTreeReadyCallback=e}getDataForDistancesComputation(e,t){const n=this.integerBasedDistancesComputation?this.getIntegerCenters(e,t,!0):this.getFloatCenters(e,t,!0),s=this.getSceneIndexes(e,t);return{centers:n,sceneIndexes:s}}refreshGPUDataFromSplatBuffers(e){const t=this.getSplatCount(!0);this.refreshDataTexturesFromSplatBuffers(e);const n=e?this.lastBuildSplatCount:0,{centers:s,sceneIndexes:r}=this.getDataForDistancesComputation(n,t-1);return this.enableDistancesComputationOnGPU&&this.refreshGPUBuffersForDistancesComputation(s,r,e),{from:n,to:t-1,count:t-n,centers:s,sceneIndexes:r}}refreshGPUBuffersForDistancesComputation(e,t,n=!1){const s=n?this.lastBuildSplatCount:0;this.updateGPUCentersBufferForDistancesComputation(n,e,s),this.updateGPUTransformIndexesBufferForDistancesComputation(n,t,s)}refreshDataTexturesFromSplatBuffers(e){const t=this.getSplatCount(!0),n=this.lastBuildSplatCount,s=t-1;e?this.updateBaseDataFromSplatBuffers(n,s):(this.setupDataTextures(),this.updateBaseDataFromSplatBuffers()),this.updateDataTexturesFromBaseData(n,s),this.updateVisibleRegion(e)}setupDataTextures(){const e=this.getMaxSplatCount(),t=this.getSplatCount(!0);this.disposeTextures();const n=(M,E)=>{const b=new Ke(4096,1024);for(;b.x*b.y*M<e*E;)b.y*=2;return b},s=M=>M>=1?VE:kE,r=M=>{const E=s(M),b=n(E,6);return{elementsPerTexelStored:E,texSize:b}};let o=this.getTargetCovarianceCompressionLevel();const a=0,l=this.getTargetSphericalHarmonicsCompressionLevel();let c,u,f;if(this.splatRenderMode===Cs.ThreeD){const M=r(o);M.texSize.x*M.texSize.y>Jm&&o===0&&(o=1),c=new Float32Array(e*tc)}else u=new Float32Array(e*3),f=new Float32Array(e*4);const d=new Float32Array(e*3),h=new Uint8Array(e*4);let x=Float32Array;l===1?x=Uint16Array:l===2&&(x=Uint8Array);const p=Uo(this.minSphericalHarmonicsDegree),g=this.minSphericalHarmonicsDegree?new x(e*p):void 0,m=n(nf,4),_=new Uint32Array(m.x*m.y*nf);bn.updateCenterColorsPaddedData(0,t-1,d,h,_);const A=new ys(_,m.x,m.y,Lo,Ii);if(A.internalFormat="RGBA32UI",A.needsUpdate=!0,this.material.uniforms.centersColorsTexture.value=A,this.material.uniforms.centersColorsTextureSize.value.copy(m),this.material.uniformsNeedUpdate=!0,this.splatDataTextures={baseData:{covariances:c,scales:u,rotations:f,centers:d,colors:h,sphericalHarmonics:g},centerColors:{data:_,texture:A,size:m}},this.splatRenderMode===Cs.ThreeD){const M=r(o),E=M.elementsPerTexelStored,b=M.texSize;let C=o>=1?Uint32Array:Float32Array;const D=o>=1?GE:HE,F=new C(b.x*b.y*D);o===0?F.set(c):bn.updatePaddedCompressedCovariancesTextureData(c,F,0,0,c.length);let O;if(o>=1)O=new ys(F,b.x,b.y,Lo,Ii),O.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=O;else{O=new ys(F,b.x,b.y,Xn,Hi),this.material.uniforms.covariancesTexture.value=O;const z=new ys(new Uint32Array(32),2,2,Lo,Ii);z.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=z,z.needsUpdate=!0}O.needsUpdate=!0,this.material.uniforms.covariancesAreHalfFloat.value=o>=1?1:0,this.material.uniforms.covariancesTextureSize.value.copy(b),this.splatDataTextures.covariances={data:F,texture:O,size:b,compressionLevel:o,elementsPerTexelStored:E,elementsPerTexelAllocated:D}}else{const E=n(tf,6);let b=Float32Array,C=Hi;const D=new b(E.x*E.y*tf);bn.updateScaleRotationsPaddedData(0,t-1,u,f,D);const F=new ys(D,E.x,E.y,Xn,C);F.needsUpdate=!0,this.material.uniforms.scaleRotationsTexture.value=F,this.material.uniforms.scaleRotationsTextureSize.value.copy(E),this.splatDataTextures.scaleRotations={data:D,texture:F,size:E,compressionLevel:a}}if(g){const M=l===2?as:jr;let E=p;E%2!==0&&E++;const b=4,C=Xn;let D=n(b,E);if(D.x*D.y<=Jm){const F=D.x*D.y*b,O=new x(F);for(let V=0;V<t;V++){const H=p*V,q=E*V;for(let G=0;G<p;G++)O[q+G]=g[H+G]}const z=new ys(O,D.x,D.y,C,M);z.needsUpdate=!0,this.material.uniforms.sphericalHarmonicsTexture.value=z,this.splatDataTextures.sphericalHarmonics={componentCount:p,paddedComponentCount:E,data:O,textureCount:1,texture:z,size:D,compressionLevel:l,elementsPerTexel:b}}else{const F=p/3;E=F,E%2!==0&&E++,D=n(b,E);const O=D.x*D.y*b,z=[this.material.uniforms.sphericalHarmonicsTextureR,this.material.uniforms.sphericalHarmonicsTextureG,this.material.uniforms.sphericalHarmonicsTextureB],V=[],H=[];for(let q=0;q<3;q++){const G=new x(O);V.push(G);for(let fe=0;fe<t;fe++){const Y=p*fe,we=E*fe;if(F>=3){for(let ze=0;ze<3;ze++)G[we+ze]=g[Y+q*3+ze];if(F>=8)for(let ze=0;ze<5;ze++)G[we+3+ze]=g[Y+9+q*5+ze]}}const $=new ys(G,D.x,D.y,C,M);H.push($),$.needsUpdate=!0,z[q].value=$}this.material.uniforms.sphericalHarmonicsMultiTextureMode.value=1,this.splatDataTextures.sphericalHarmonics={componentCount:p,componentCountPerChannel:F,paddedComponentCount:E,data:V,textureCount:3,textures:H,size:D,compressionLevel:l,elementsPerTexel:b}}this.material.uniforms.sphericalHarmonicsTextureSize.value.copy(D),this.material.uniforms.sphericalHarmonics8BitMode.value=l===2?1:0;for(let F=0;F<this.scenes.length;F++){const O=this.scenes[F].splatBuffer;this.material.uniforms.sphericalHarmonics8BitCompressionRangeMin.value[F]=O.minSphericalHarmonicsCoeff,this.material.uniforms.sphericalHarmonics8BitCompressionRangeMax.value[F]=O.maxSphericalHarmonicsCoeff}this.material.uniformsNeedUpdate=!0}const v=n($m,4),S=new Uint32Array(v.x*v.y*$m);for(let M=0;M<t;M++)S[M]=this.globalSplatIndexToSceneIndexMap[M];const y=new ys(S,v.x,v.y,Kc,Ii);y.internalFormat="R32UI",y.needsUpdate=!0,this.material.uniforms.sceneIndexesTexture.value=y,this.material.uniforms.sceneIndexesTextureSize.value.copy(v),this.material.uniformsNeedUpdate=!0,this.splatDataTextures.sceneIndexes={data:S,texture:y,size:v},this.material.uniforms.sceneCount.value=this.scenes.length}updateBaseDataFromSplatBuffers(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0;this.fillSplatDataArrays(this.splatDataTextures.baseData.covariances,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,this.splatDataTextures.baseData.sphericalHarmonics,void 0,s,o,l,e,t,e)}updateDataTexturesFromBaseData(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0,c=this.splatDataTextures.centerColors,u=c.data,f=c.texture;bn.updateCenterColorsPaddedData(e,t,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,u);const d=this.renderer?this.renderer.properties.get(f):null;if(!d||!d.__webglTexture?f.needsUpdate=!0:this.updateDataTexture(u,c.texture,c.size,d,nf,zE,4,e,t),n){const _=n.texture,A=e*tc,v=t*tc;if(s===0)for(let y=A;y<=v;y++){const M=this.splatDataTextures.baseData.covariances[y];n.data[y]=M}else bn.updatePaddedCompressedCovariancesTextureData(this.splatDataTextures.baseData.covariances,n.data,e*n.elementsPerTexelAllocated,A,v);const S=this.renderer?this.renderer.properties.get(_):null;!S||!S.__webglTexture?_.needsUpdate=!0:s===0?this.updateDataTexture(n.data,n.texture,n.size,S,n.elementsPerTexelStored,tc,4,e,t):this.updateDataTexture(n.data,n.texture,n.size,S,n.elementsPerTexelAllocated,n.elementsPerTexelAllocated,2,e,t)}if(r){const _=r.data,A=r.texture,v=6,S=o===0?4:2;bn.updateScaleRotationsPaddedData(e,t,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,_);const y=this.renderer?this.renderer.properties.get(A):null;!y||!y.__webglTexture?A.needsUpdate=!0:this.updateDataTexture(_,r.texture,r.size,y,tf,v,S,e,t)}const h=this.splatDataTextures.baseData.sphericalHarmonics;if(h){let _=4;l===1?_=2:l===2&&(_=1);const A=(y,M,E,b,C)=>{const D=this.renderer?this.renderer.properties.get(y):null;!D||!D.__webglTexture?y.needsUpdate=!0:this.updateDataTexture(b,y,M,D,E,C,_,e,t)},v=a.componentCount,S=a.paddedComponentCount;if(a.textureCount===1){const y=a.data;for(let M=e;M<=t;M++){const E=v*M,b=S*M;for(let C=0;C<v;C++)y[b+C]=h[E+C]}A(a.texture,a.size,a.elementsPerTexel,y,S)}else{const y=a.componentCountPerChannel;for(let M=0;M<3;M++){const E=a.data[M];for(let b=e;b<=t;b++){const C=v*b,D=S*b;if(y>=3){for(let F=0;F<3;F++)E[D+F]=h[C+M*3+F];if(y>=8)for(let F=0;F<5;F++)E[D+3+F]=h[C+9+M*5+F]}}A(a.textures[M],a.size,a.elementsPerTexel,E,S)}}}const x=this.splatDataTextures.sceneIndexes,p=x.data;for(let _=this.lastBuildSplatCount;_<=t;_++)p[_]=this.globalSplatIndexToSceneIndexMap[_];const g=x.texture,m=this.renderer?this.renderer.properties.get(g):null;!m||!m.__webglTexture?g.needsUpdate=!0:this.updateDataTexture(p,x.texture,x.size,m,1,1,1,this.lastBuildSplatCount,t)}getTargetCovarianceCompressionLevel(){return this.halfPrecisionCovariancesOnGPU?1:0}getTargetSphericalHarmonicsCompressionLevel(){return Math.max(1,this.getMaximumSplatBufferCompressionLevel())}getMaximumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel>e)&&(e=s.compressionLevel)}return e}getMinimumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel<e)&&(e=s.compressionLevel)}return e}static computeTextureUpdateRegion(e,t,n,s,r){const o=r/s,a=e*o,l=Math.floor(a/n),c=l*n*s,u=t*o,f=Math.floor(u/n),d=f*n*s+n*s;return{dataStart:c,dataEnd:d,startRow:l,endRow:f}}updateDataTexture(e,t,n,s,r,o,a,l,c){const u=this.renderer.getContext(),f=bn.computeTextureUpdateRegion(l,c,n.x,r,o),d=f.dataEnd-f.dataStart,h=new e.constructor(e.buffer,f.dataStart*a,d),x=f.endRow-f.startRow+1,p=this.webGLUtils.convert(t.type),g=this.webGLUtils.convert(t.format,t.colorSpace),m=u.getParameter(u.TEXTURE_BINDING_2D);u.bindTexture(u.TEXTURE_2D,s.__webglTexture),u.texSubImage2D(u.TEXTURE_2D,0,0,f.startRow,n.x,x,g,p,h),u.bindTexture(u.TEXTURE_2D,m)}static updatePaddedCompressedCovariancesTextureData(e,t,n,s,r){let o=new DataView(t.buffer),a=n,l=0;for(let c=s;c<=r;c+=2)o.setUint16(a*2,e[c],!0),o.setUint16(a*2+2,e[c+1],!0),a+=2,l++,l>=3&&(a+=2,l=0)}static updateCenterColorsPaddedData(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*4,l=o*3,c=o*4;r[c]=y1(s,a),r[c+1]=Qu(n[l]),r[c+2]=Qu(n[l+1]),r[c+3]=Qu(n[l+2])}}static updateScaleRotationsPaddedData(e,t,n,s,r){for(let a=e;a<=t;a++){const l=a*3,c=a*4,u=a*6;r[u]=n[l],r[u+1]=n[l+1],r[u+2]=n[l+2],r[u+3]=s[c],r[u+4]=s[c+1],r[u+5]=s[c+2]}}updateVisibleRegion(e){const t=this.getSplatCount(!0),n=new B;if(!e){const r=new B;this.scenes.forEach(o=>{r.add(o.splatBuffer.sceneCenter)}),r.multiplyScalar(1/this.scenes.length),this.calculatedSceneCenter.copy(r),this.material.uniforms.sceneCenter.value.copy(this.calculatedSceneCenter),this.material.uniformsNeedUpdate=!0}const s=e?this.lastBuildSplatCount:0;for(let r=s;r<t;r++){this.getSplatCenter(r,n,!0);const o=n.sub(this.calculatedSceneCenter).length();o>this.maxSplatDistanceFromSceneCenter&&(this.maxSplatDistanceFromSceneCenter=o)}this.maxSplatDistanceFromSceneCenter-this.visibleRegionBufferRadius>Zm&&(this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter,this.visibleRegionRadius=Math.max(this.visibleRegionBufferRadius-Zm,0)),this.finalBuild&&(this.visibleRegionRadius=this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter),this.updateVisibleRegionFadeDistance()}updateVisibleRegionFadeDistance(e=za.Default){const t=WE*this.sceneFadeInRateMultiplier,n=XE*this.sceneFadeInRateMultiplier,s=this.finalBuild?t:n,r=e===za.Default?s:n;this.visibleRegionFadeStartRadius=(this.visibleRegionRadius-this.visibleRegionFadeStartRadius)*r+this.visibleRegionFadeStartRadius;const a=(this.visibleRegionBufferRadius>0?this.visibleRegionFadeStartRadius/this.visibleRegionBufferRadius:0)>.99,l=a||e===za.Instant?1:0;this.material.uniforms.visibleRegionFadeStartRadius.value=this.visibleRegionFadeStartRadius,this.material.uniforms.visibleRegionRadius.value=this.visibleRegionRadius,this.material.uniforms.firstRenderTime.value=this.firstRenderTime,this.material.uniforms.currentTime.value=performance.now(),this.material.uniforms.fadeInComplete.value=l,this.material.uniformsNeedUpdate=!0,this.visibleRegionChanging=!a}updateRenderIndexes(e,t){const n=this.geometry;n.attributes.splatIndex.set(e),n.attributes.splatIndex.needsUpdate=!0,t>0&&this.firstRenderTime===-1&&(this.firstRenderTime=performance.now()),n.instanceCount=t,n.setDrawRange(0,t)}updateTransforms(){for(let e=0;e<this.scenes.length;e++)this.getScene(e).updateTransform(this.dynamicMode)}updateUniforms=(function(){const e=new Ke;return function(t,n,s,r,o,a){if(this.getSplatCount()>0){if(e.set(t.x*this.devicePixelRatio,t.y*this.devicePixelRatio),this.material.uniforms.viewport.value.copy(e),this.material.uniforms.basisViewport.value.set(1/e.x,1/e.y),this.material.uniforms.focal.value.set(n,s),this.material.uniforms.orthographicMode.value=r?1:0,this.material.uniforms.orthoZoom.value=o,this.material.uniforms.inverseFocalAdjustment.value=a,this.dynamicMode)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.transforms.value[c].copy(this.getScene(c).transform);if(this.enableOptionalEffects)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.sceneOpacity.value[c]=$t(this.getScene(c).opacity,0,1),this.material.uniforms.sceneVisibility.value[c]=this.getScene(c).visible?1:0,this.material.uniformsNeedUpdate=!0;this.material.uniformsNeedUpdate=!0}}})();setSplatScale(e=1){this.splatScale=e,this.material.uniforms.splatScale.value=e,this.material.uniformsNeedUpdate=!0}getSplatScale(){return this.splatScale}setPointCloudModeEnabled(e){this.pointCloudModeEnabled=e,this.material.uniforms.pointCloudModeEnabled.value=e?1:0,this.material.uniformsNeedUpdate=!0}getPointCloudModeEnabled(){return this.pointCloudModeEnabled}getSplatDataTextures(){return this.splatDataTextures}getSplatCount(e=!1){return e?bn.getTotalSplatCountForScenes(this.scenes):this.lastBuildSplatCount}static getTotalSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getSplatCount());return t}static getTotalSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getSplatCount();return t}getMaxSplatCount(){return bn.getTotalMaxSplatCountForScenes(this.scenes)}static getTotalMaxSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getMaxSplatCount());return t}static getTotalMaxSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getMaxSplatCount();return t}disposeDistancesComputationGPUResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.vao&&(e.deleteVertexArray(this.distancesTransformFeedback.vao),this.distancesTransformFeedback.vao=null),this.distancesTransformFeedback.program&&(e.deleteProgram(this.distancesTransformFeedback.program),e.deleteShader(this.distancesTransformFeedback.vertexShader),e.deleteShader(this.distancesTransformFeedback.fragmentShader),this.distancesTransformFeedback.program=null,this.distancesTransformFeedback.vertexShader=null,this.distancesTransformFeedback.fragmentShader=null),this.disposeDistancesComputationGPUBufferResources(),this.distancesTransformFeedback.id&&(e.deleteTransformFeedback(this.distancesTransformFeedback.id),this.distancesTransformFeedback.id=null)}disposeDistancesComputationGPUBufferResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.centersBuffer&&(this.distancesTransformFeedback.centersBuffer=null,e.deleteBuffer(this.distancesTransformFeedback.centersBuffer)),this.distancesTransformFeedback.outDistancesBuffer&&(e.deleteBuffer(this.distancesTransformFeedback.outDistancesBuffer),this.distancesTransformFeedback.outDistancesBuffer=null)}setRenderer(e){if(e!==this.renderer){this.renderer=e;const t=this.renderer.getContext(),n=new UE(t),s=new OE(t,n,{});if(n.init(s),this.webGLUtils=new ix(t,n),this.enableDistancesComputationOnGPU&&this.getSplatCount()>0){this.setupDistancesComputationTransformFeedback();const{centers:r,sceneIndexes:o}=this.getDataForDistancesComputation(0,this.getSplatCount()-1);this.refreshGPUBuffersForDistancesComputation(r,o)}}}setupDistancesComputationTransformFeedback=(function(){let e;return function(){const t=this.getMaxSplatCount();if(!this.renderer)return;const n=this.lastRenderer!==this.renderer,s=e!==t;if(!n&&!s)return;n?this.disposeDistancesComputationGPUResources():s&&this.disposeDistancesComputationGPUBufferResources();const r=this.renderer.getContext(),o=(d,h,x)=>{const p=d.createShader(h);if(!p)return console.error("Fatal error: gl could not create a shader object."),null;if(d.shaderSource(p,x),d.compileShader(p),!d.getShaderParameter(p,d.COMPILE_STATUS)){let m="unknown";h===d.VERTEX_SHADER?m="vertex shader":h===d.FRAGMENT_SHADER&&(m="fragement shader");const _=d.getShaderInfoLog(p);return console.error("Failed to compile "+m+" with these errors:"+_),d.deleteShader(p),null}return p};let a;this.integerBasedDistancesComputation?(a=`#version 300 es
                in ivec4 center;
                flat out int distance;`,this.dynamicMode?a+=`
                        in uint sceneIndex;
                        uniform ivec4 transforms[${Nt.MaxScenes}];
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
                        uniform mat4 transforms[${Nt.MaxScenes}];
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
            `,c=r.getParameter(r.VERTEX_ARRAY_BINDING),u=r.getParameter(r.CURRENT_PROGRAM),f=u?r.getProgramParameter(u,r.DELETE_STATUS):!1;if(n&&(this.distancesTransformFeedback.vao=r.createVertexArray()),r.bindVertexArray(this.distancesTransformFeedback.vao),n){const d=r.createProgram(),h=o(r,r.VERTEX_SHADER,a),x=o(r,r.FRAGMENT_SHADER,l);if(!h||!x)throw new Error("Could not compile shaders for distances computation on GPU.");if(r.attachShader(d,h),r.attachShader(d,x),r.transformFeedbackVaryings(d,["distance"],r.SEPARATE_ATTRIBS),r.linkProgram(d),!r.getProgramParameter(d,r.LINK_STATUS)){const g=r.getProgramInfoLog(d);throw console.error("Fatal error: Failed to link program: "+g),r.deleteProgram(d),r.deleteShader(x),r.deleteShader(h),new Error("Could not link shaders for distances computation on GPU.")}this.distancesTransformFeedback.program=d,this.distancesTransformFeedback.vertexShader=h,this.distancesTransformFeedback.vertexShader=x}if(r.useProgram(this.distancesTransformFeedback.program),this.distancesTransformFeedback.centersLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"center"),this.dynamicMode){this.distancesTransformFeedback.sceneIndexesLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"sceneIndex");for(let d=0;d<this.scenes.length;d++)this.distancesTransformFeedback.transformsLocs[d]=r.getUniformLocation(this.distancesTransformFeedback.program,`transforms[${d}]`)}else this.distancesTransformFeedback.modelViewProjLoc=r.getUniformLocation(this.distancesTransformFeedback.program,"modelViewProj");(n||s)&&(this.distancesTransformFeedback.centersBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?r.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,r.INT,0,0):r.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,r.FLOAT,!1,0,0),this.dynamicMode&&(this.distancesTransformFeedback.sceneIndexesBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),r.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,r.UNSIGNED_INT,0,0))),(n||s)&&(this.distancesTransformFeedback.outDistancesBuffer=r.createBuffer()),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),r.bufferData(r.ARRAY_BUFFER,t*4,r.STATIC_READ),n&&(this.distancesTransformFeedback.id=r.createTransformFeedback()),r.bindTransformFeedback(r.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),r.bindBufferBase(r.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),u&&f!==!0&&r.useProgram(u),c&&r.bindVertexArray(c),this.lastRenderer=this.renderer,e=t}})();updateGPUCentersBufferForDistancesComputation(e,t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=this.integerBasedDistancesComputation?Uint32Array:Float32Array,a=16,l=n*a;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,l,t);else{const c=new o(this.getMaxSplatCount()*a);c.set(t),s.bufferData(s.ARRAY_BUFFER,c,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}updateGPUTransformIndexesBufferForDistancesComputation(e,t,n){if(!this.renderer||!this.dynamicMode)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=n*4;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,o,t);else{const a=new Uint32Array(this.getMaxSplatCount()*4);a.set(t),s.bufferData(s.ARRAY_BUFFER,a,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}getSceneIndexes(e,t){let n;const s=t-e+1;n=new Uint32Array(s);for(let r=e;r<=t;r++)n[r]=this.globalSplatIndexToSceneIndexMap[r];return n}fillTransformsArray=(function(){const e=[];return function(t){e.length!==t.length&&(e.length=t.length);for(let n=0;n<this.scenes.length;n++){const r=this.getScene(n).transform.elements;for(let o=0;o<16;o++)e[n*16+o]=r[o]}t.set(e)}})();computeDistancesOnGPU=(function(){const e=new st;return function(t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING),o=s.getParameter(s.CURRENT_PROGRAM),a=o?s.getProgramParameter(o,s.DELETE_STATUS):!1;if(s.bindVertexArray(this.distancesTransformFeedback.vao),s.useProgram(this.distancesTransformFeedback.program),s.enable(s.RASTERIZER_DISCARD),this.dynamicMode)for(let u=0;u<this.scenes.length;u++)if(e.copy(this.getScene(u).transform),e.premultiply(t),this.integerBasedDistancesComputation){const f=bn.getIntegerMatrixArray(e),d=[f[2],f[6],f[10],f[14]];s.uniform4i(this.distancesTransformFeedback.transformsLocs[u],d[0],d[1],d[2],d[3])}else s.uniformMatrix4fv(this.distancesTransformFeedback.transformsLocs[u],!1,e.elements);else if(this.integerBasedDistancesComputation){const u=bn.getIntegerMatrixArray(t),f=[u[2],u[6],u[10]];s.uniform3i(this.distancesTransformFeedback.modelViewProjLoc,f[0],f[1],f[2])}else{const u=[t.elements[2],t.elements[6],t.elements[10]];s.uniform3f(this.distancesTransformFeedback.modelViewProjLoc,u[0],u[1],u[2])}s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?s.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,s.INT,0,0):s.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,s.FLOAT,!1,0,0),this.dynamicMode&&(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),s.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,s.UNSIGNED_INT,0,0)),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),s.beginTransformFeedback(s.POINTS),s.drawArrays(s.POINTS,0,this.getSplatCount()),s.endTransformFeedback(),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,null),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,null),s.disable(s.RASTERIZER_DISCARD);const l=s.fenceSync(s.SYNC_GPU_COMMANDS_COMPLETE,0);s.flush();const c=new Promise(u=>{const f=()=>{if(this.disposed)u();else switch(s.clientWaitSync(l,0,0)){case s.TIMEOUT_EXPIRED:return this.computeDistancesOnGPUSyncTimeout=setTimeout(f),this.computeDistancesOnGPUSyncTimeout;case s.WAIT_FAILED:throw new Error("should never get here");default:this.computeDistancesOnGPUSyncTimeout=null,s.deleteSync(l);const p=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao),s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),s.getBufferSubData(s.ARRAY_BUFFER,0,n),s.bindBuffer(s.ARRAY_BUFFER,null),p&&s.bindVertexArray(p),u()}};this.computeDistancesOnGPUSyncTimeout=setTimeout(f)});return o&&a!==!0&&s.useProgram(o),r&&s.bindVertexArray(r),c}})();getLocalSplatParameters(e,t,n){n==null&&(n=!this.dynamicMode),t.splatBuffer=this.getSplatBufferForSplat(e),t.localIndex=this.getSplatLocalIndex(e),t.sceneTransform=n?this.getSceneTransformForSplat(e):null}fillSplatDataArrays(e,t,n,s,r,o,a,l=0,c=0,u=1,f,d,h=0,x){const p=new B;p.x=void 0,p.y=void 0,this.splatRenderMode===Cs.ThreeD?p.z=void 0:p.z=1;const g=new st;let m=0,_=this.scenes.length-1;x!=null&&x>=0&&x<=this.scenes.length&&(m=x,_=x);for(let A=m;A<=_;A++){a==null&&(a=!this.dynamicMode);const v=this.getScene(A),S=v.splatBuffer;let y;if(a&&(this.getSceneTransform(A,g),y=g),e&&S.fillSplatCovarianceArray(e,y,f,d,h,l),t||n){if(!t||!n)throw new Error('SplatMesh::fillSplatDataArrays() -> "scales" and "rotations" must both be valid.');S.fillSplatScaleRotationArray(t,n,y,f,d,h,c,p)}s&&S.fillSplatCenterArray(s,y,f,d,h),r&&S.fillSplatColorArray(r,v.minimumAlpha,f,d,h),o&&S.fillSphericalHarmonicsArray(o,this.minSphericalHarmonicsDegree,y,f,d,h,u),h+=S.getSplatCount()}}getIntegerCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e);let o,a=n?4:3;o=new Int32Array(s*a);for(let l=0;l<s;l++){for(let c=0;c<3;c++)o[l*a+c]=Math.round(r[l*3+c]*1e3);n&&(o[l*a+3]=1e3)}return o}getFloatCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);if(this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e),!n)return r;let o=new Float32Array(s*4);for(let a=0;a<s;a++){for(let l=0;l<3;l++)o[a*4+l]=r[a*3+l];o[a*4+3]=1}return o}getSplatCenter=(function(){const e={};return function(t,n,s){this.getLocalSplatParameters(t,e,s),e.splatBuffer.getSplatCenter(e.localIndex,n,e.sceneTransform)}})();getSplatScaleAndRotation=(function(){const e={},t=new B;return function(n,s,r,o){this.getLocalSplatParameters(n,e,o),t.x=void 0,t.y=void 0,t.z=void 0,this.splatRenderMode===Cs.TwoD&&(t.z=0),e.splatBuffer.getSplatScaleAndRotation(e.localIndex,s,r,e.sceneTransform,t)}})();getSplatColor=(function(){const e={};return function(t,n){this.getLocalSplatParameters(t,e),e.splatBuffer.getSplatColor(e.localIndex,n)}})();getSceneTransform(e,t){const n=this.getScene(e);n.updateTransform(this.dynamicMode),t.copy(n.transform)}getScene(e){if(e<0||e>=this.scenes.length)throw new Error("SplatMesh::getScene() -> Invalid scene index.");return this.scenes[e]}getSceneCount(){return this.scenes.length}getSplatBufferForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).splatBuffer}getSceneIndexForSplat(e){return this.globalSplatIndexToSceneIndexMap[e]}getSceneTransformForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).transform}getSplatLocalIndex(e){return this.globalSplatIndexToLocalSplatIndexMap[e]}static getIntegerMatrixArray(e){const t=e.elements,n=[];for(let s=0;s<16;s++)n[s]=Math.round(t[s]*1e3);return n}computeBoundingBox(e=!1,t){let n=this.getSplatCount();if(t!=null){if(t<0||t>=this.scenes.length)throw new Error("SplatMesh::computeBoundingBox() -> Invalid scene index.");n=this.scenes[t].splatBuffer.getSplatCount()}const s=new Float32Array(n*3);this.fillSplatDataArrays(null,null,null,s,null,null,e,void 0,void 0,void 0,void 0,t);const r=new B,o=new B;for(let a=0;a<n;a++){const l=a*3,c=s[l],u=s[l+1],f=s[l+2];(a===0||c<r.x)&&(r.x=c),(a===0||u<r.y)&&(r.y=u),(a===0||f<r.z)&&(r.z=f),(a===0||c>o.x)&&(o.x=c),(a===0||u>o.y)&&(o.y=u),(a===0||f>o.z)&&(o.z=f)}return new Ji(r,o)}}var qE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEbA2AAAGAQf39/f39/f39/f39/f39/fwBgAAF/AhIBA2VudgZtZW1vcnkCAwCAgAQDBAMAAQIHVAQRX193YXNtX2NhbGxfY3RvcnMAABhfX3dhc21fYXBwbHlfZGF0YV9yZWxvY3MAAAtzb3J0SW5kZXhlcwABE2Vtc2NyaXB0ZW5fdGxzX2luaXQAAgqWEAMDAAELihAEAXwDewN/A30gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEBA0AgAyABQQJ0IgVqIAIgACAFaigCAEECdGooAgAiBTYCACAFIAogBSAKSBshCiAFIA0gBSANShshDSABQQFqIgEgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiFWooAgAiFkECdGooAgAiFEcEQAJ/IAX9CQI4IAggFEEGdGoiDv0JAgwgDioCHP0gASAOKgIs/SACIA4qAjz9IAP95gEgBf0JAiggDv0JAgggDioCGP0gASAOKgIo/SACIA4qAjj9IAP95gEgBf0JAgggDv0JAgAgDioCEP0gASAOKgIg/SACIA4qAjD9IAP95gEgBf0JAhggDv0JAgQgDioCFP0gASAOKgIk/SACIA4qAjT9IAP95gH95AH95AH95AEiEf1f/QwAAAAAAECPQAAAAAAAQI9AIhL98gEiE/0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBP9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/REgDv0cAQJ/IBEgEf0NCAkKCwwNDg8AAAAAAAAAAP1fIBL98gEiEf0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9HAICfyAR/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAyESIBQhDwsgAyAVaiABIBZBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmogEf0bA2oiDjYCACAOIAogCiAOShshCiAOIA0gDSAOSBshDSACQQFqIgIgC0cNAAsMAwsCfyAFKgIIu/0UIAUqAhi7/SIB/QwAAAAAAECPQAAAAAAAQI9A/fIBIhH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIQ4CfyAR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyECAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQIgAv0RIA79HAEgBf0cAiESIAwhBQNAIAMgBUECdCICaiABIAAgAmooAgBBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmoiAjYCACACIAogAiAKSBshCiACIA0gAiANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEBA0AgAyABQQJ0IgVqAn8gAiAAIAVqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAFBAWoiASALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIRcgBSoCGCEYIAUqAgghGUH4////ByEKQYiAgIB4IQ0gDCEFA0ACfyAXIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCAZIAIqAgCUIBggAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIUaigCAEECdCIVaigCACIORwRAIAX9CQI4IAggDkEGdGoiD/0JAgwgDyoCHP0gASAPKgIs/SACIA8qAjz9IAP95gEgBf0JAiggD/0JAgggDyoCGP0gASAPKgIo/SACIA8qAjj9IAP95gEgBf0JAgggD/0JAgAgDyoCEP0gASAPKgIg/SACIA8qAjD9IAP95gEgBf0JAhggD/0JAgQgDyoCFP0gASAPKgIk/SACIA8qAjT9IAP95gH95AH95AH95AEhESAOIQ8LIAMgFGoCfyAR/R8DIAEgFUECdCIOQQxyaioCAJQgEf0fAiABIA5BCHJqKgIAlCAR/R8AIAEgDmoqAgCUIBH9HwEgASAOQQRyaioCAJSSkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSACQQFqIgIgC0cNAAsMAQtBiICAgHghDUH4////ByEKCyALIAxLBEAgCUEBa7MgDbIgCrKTlSEXIAwhDQNAAn8gFyADIA1BAnRqIgEoAgAgCmuylCIYi0MAAABPXQRAIBioDAELQYCAgIB4CyEOIAEgDjYCACAEIA5BAnRqIgEgASgCAEEBajYCACANQQFqIg0gC0cNAAsLIAlBAk8EQCAEKAIAIQ1BASEKA0AgBCAKQQJ0aiIBIAEoAgAgDWoiDTYCACAKQQFqIgogCUcNAAsLIAxBAEoEQCAMIQoDQCAGIApBAWsiAUECdCICaiAAIAJqKAIANgIAIApBAUshAiABIQogAg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCwsEAEEACw==",e0="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACEgEDZW52Bm1lbW9yeQIDAICABAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=",YE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQrrDwICAAvlDwQBfAN7B30DfyALIAprIQwCQAJAIA4EQCANBEBB+P///wchCkGIgICAeCENIAsgDE0NAyAMIQUDQCADIAVBAnQiAWogAiAAIAFqKAIAQQJ0aigCACIBNgIAIAEgCiABIApIGyEKIAEgDSABIA1KGyENIAVBAWoiBSALRw0ACwwDCyAPBEAgCyAMTQ0CQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIcaigCACIdQQJ0aigCACIbRwRAAn8gBf0JAjggCCAbQQZ0aiIO/QkCDCAOKgIc/SABIA4qAiz9IAIgDioCPP0gA/3mASAF/QkCKCAO/QkCCCAOKgIY/SABIA4qAij9IAIgDioCOP0gA/3mASAF/QkCCCAO/QkCACAOKgIQ/SABIA4qAiD9IAIgDioCMP0gA/3mASAF/QkCGCAO/QkCBCAOKgIU/SABIA4qAiT9IAIgDioCNP0gA/3mAf3kAf3kAf3kASIR/V/9DAAAAAAAQI9AAAAAAABAj0AiEv3yASIT/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOAn8gE/0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9ESAO/RwBAn8gESAR/Q0ICQoLDA0ODwABAgMAAQID/V8gEv3yASIR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAgJ/IBH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/RwDIRIgGyEPCyADIBxqIAEgHUEEdGr9AAAAIBL9tQEiEf0bACAR/RsBaiAR/RsCaiAR/RsDaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAgi7/RQgBSoCGLv9IgH9DAAAAAAAQI9AAAAAAABAj0D98gEiEf0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBH9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQL9ESAO/RwBIAX9HAIhEiAMIQUDQCADIAVBAnQiAmogASAAIAJqKAIAQQR0av0AAAAgEv21ASIR/RsAIBH9GwFqIBH9GwJqIgI2AgAgAiAKIAIgCkgbIQogAiANIAIgDUobIQ0gBUEBaiIFIAtHDQALDAILIA0EQEH4////ByEKQYiAgIB4IQ0gCyAMTQ0CIAwhBQNAIAMgBUECdCIBagJ/IAIgACABaigCAEECdGoqAgC7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgD0UEQCALIAxNDQEgBSoCKCEUIAUqAhghFSAFKgIIIRZB+P///wchCkGIgICAeCENIAwhBQNAAn8gFCABIAAgBUECdCIHaigCAEEEdGoiAioCCJQgFiACKgIAlCAVIAIqAgSUkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDiADIAdqIA42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gBUEBaiIFIAtHDQALDAILIAsgDE0NAEF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiG2ooAgBBAnQiHGooAgAiDkcEQCAFKgI4IhQgCCAOQQZ0aiIPKgI8lCAFKgIoIhUgDyoCOJQgBSoCCCIWIA8qAjCUIAUqAhgiFyAPKgI0lJKSkiEYIBQgDyoCLJQgFSAPKgIolCAWIA8qAiCUIBcgDyoCJJSSkpIhGSAUIA8qAhyUIBUgDyoCGJQgFiAPKgIQlCAXIA8qAhSUkpKSIRogFCAPKgIMlCAVIA8qAgiUIBYgDyoCAJQgFyAPKgIElJKSkiEUIA4hDwsgAyAbagJ/IBggASAcQQJ0aiIOKgIMlCAZIA4qAgiUIBQgDioCAJQgGiAOKgIElJKSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAJBAWoiAiALRw0ACwwBC0GIgICAeCENQfj///8HIQoLIAsgDEsEQCAJQQFrsyANsiAKspOVIRQgDCENA0ACfyAUIAMgDUECdGoiASgCACAKa7KUIhWLQwAAAE9dBEAgFagMAQtBgICAgHgLIQ4gASAONgIAIAQgDkECdGoiASABKAIAQQFqNgIAIA1BAWoiDSALRw0ACwsgCUECTwRAIAQoAgAhDUEBIQoDQCAEIApBAnRqIgEgASgCACANaiINNgIAIApBAWoiCiAJRw0ACwsgDEEASgRAIAwhCgNAIAYgCkEBayIBQQJ0IgJqIAAgAmooAgA2AgAgCkEBSyABIQoNAAsLIAsgDEoEQCALIQoDQCAGIAsgBCADIApBAWsiCkECdCIBaigCAEECdGoiAigCACIFa0ECdGogACABaigCADYCACACIAVBAWs2AgAgCiAMSg0ACwsL",QE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=";function KE(i){let e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g,m,_,A,v;function S(y,M,E,b,C,D,F){const O=performance.now();if(!n&&(new Uint32Array(t,a,C.byteLength/v.BytesPerInt).set(C),new Float32Array(t,u,F.byteLength/v.BytesPerFloat).set(F),b)){let G;s?G=new Int32Array(t,f,D.byteLength/v.BytesPerInt):G=new Float32Array(t,f,D.byteLength/v.BytesPerFloat),G.set(D)}g||(g=new Uint32Array(_)),new Float32Array(t,p,16).set(E),new Uint32Array(t,h,_).set(g),e.exports.sortIndexes(a,x,f,d,h,p,l,c,u,_,y,M,o,b,s,r);const z={sortDone:!0,splatSortCount:y,splatRenderCount:M,sortTime:0};if(!n){const H=new Uint32Array(t,l,M);(!m||m.length<M)&&(m=new Uint32Array(M)),m.set(H),z.sortedIndexes=m}const V=performance.now();z.sortTime=V-O,i.postMessage(z)}i.onmessage=y=>{if(y.data.centers)centers=y.data.centers,sceneIndexes=y.data.sceneIndexes,s?new Int32Array(t,x+y.data.range.from*v.BytesPerInt*4,y.data.range.count*4).set(new Int32Array(centers)):new Float32Array(t,x+y.data.range.from*v.BytesPerFloat*4,y.data.range.count*4).set(new Float32Array(centers)),r&&new Uint32Array(t,c+y.data.range.from*4,y.data.range.count).set(new Uint32Array(sceneIndexes)),A=y.data.range.from+y.data.range.count;else if(y.data.sort){const M=Math.min(y.data.sort.splatRenderCount||0,A),E=Math.min(y.data.sort.splatSortCount||0,A),b=y.data.sort.usePrecomputedDistances;let C,D,F;n||(C=y.data.sort.indexesToSort,F=y.data.sort.transforms,b&&(D=y.data.sort.precomputedDistances)),S(E,M,y.data.sort.modelViewProj,b,C,D,F)}else if(y.data.init){v=y.data.init.Constants,o=y.data.init.splatCount,n=y.data.init.useSharedMemory,s=y.data.init.integerBasedSort,r=y.data.init.dynamicMode,_=y.data.init.distanceMapRange,A=0;const M=s?v.BytesPerInt*4:v.BytesPerFloat*4,E=new Uint8Array(y.data.init.sorterWasmBytes),b=16*v.BytesPerFloat,C=o*v.BytesPerInt,D=o*M,F=b,O=s?o*v.BytesPerInt:o*v.BytesPerFloat,z=o*v.BytesPerInt,V=o*v.BytesPerInt,H=s?_*v.BytesPerInt*2:_*v.BytesPerFloat*2,q=r?o*v.BytesPerInt:0,G=r?v.MaxScenes*b:0,$=v.MemoryPageSize*32,fe=C+D+F+O+z+H+V+q+G+$,Y=Math.floor(fe/v.MemoryPageSize)+1,we={module:{},env:{memory:new WebAssembly.Memory({initial:Y,maximum:Y,shared:!0})}};WebAssembly.compile(E).then(ze=>WebAssembly.instantiate(ze,we)).then(ze=>{e=ze,a=0,x=a+C,p=x+D,f=p+F,d=f+O,h=d+z,l=h+H,c=l+V,u=c+q,t=we.env.memory.buffer,n?i.postMessage({sortSetupPhase1Complete:!0,indexesToSortBuffer:t,indexesToSortOffset:a,sortedIndexesBuffer:t,sortedIndexesOffset:l,precomputedDistancesBuffer:t,precomputedDistancesOffset:f,transformsBuffer:t,transformsOffset:u}):i.postMessage({sortSetupPhase1Complete:!0})})}}}function jE(i,e,t,n,s,r=Nt.DefaultSplatSortDistanceMapPrecision){const o=new Worker(URL.createObjectURL(new Blob(["(",KE.toString(),")(self)"],{type:"application/javascript"})));let a=qE;const l=uh()?rx():null;!t&&!e?(a=e0,l&&l.major<=16&&l.minor<4&&(a=QE)):t?e||l&&l.major<=16&&l.minor<4&&(a=YE):a=e0;const c=atob(a),u=new Uint8Array(c.length);for(let f=0;f<c.length;f++)u[f]=c.charCodeAt(f);return o.postMessage({init:{sorterWasmBytes:u.buffer,splatCount:i,useSharedMemory:e,integerBasedSort:n,dynamicMode:s,distanceMapRange:1<<r,Constants:{BytesPerFloat:Nt.BytesPerFloat,BytesPerInt:Nt.BytesPerInt,MemoryPageSize:Nt.MemoryPageSize,MaxScenes:Nt.MaxScenes}}}),o}const Lr={None:0,VR:1,AR:2};class $o{static createButton(e,t={}){const n=document.createElement("button");function s(){let c=null;async function u(h){h.addEventListener("end",f),await e.xr.setSession(h),n.textContent="EXIT VR",c=h}function f(){c.removeEventListener("end",f),n.textContent="ENTER VR",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="ENTER VR";const d={...t,optionalFeatures:["local-floor","bounded-floor","layers",...t.optionalFeatures||[]]};n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-vr",d).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",d).then(u).catch(h=>{console.warn(h)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",d).then(u).catch(h=>{console.warn(h)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="VR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="VR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="VRButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-vr").then(function(c){c?s():o(),c&&$o.xrSessionIsGranted&&n.click()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}static registerSessionGrantedListener(){if(typeof navigator<"u"&&"xr"in navigator){if(/WebXRViewer\//i.test(navigator.userAgent))return;navigator.xr.addEventListener("sessiongranted",()=>{$o.xrSessionIsGranted=!0})}}}$o.xrSessionIsGranted=!1;$o.registerSessionGrantedListener();class $E{static createButton(e,t={}){const n=document.createElement("button");function s(){if(t.domOverlay===void 0){const d=document.createElement("div");d.style.display="none",document.body.appendChild(d);const h=document.createElementNS("http://www.w3.org/2000/svg","svg");h.setAttribute("width",38),h.setAttribute("height",38),h.style.position="absolute",h.style.right="20px",h.style.top="20px",h.addEventListener("click",function(){c.end()}),d.appendChild(h);const x=document.createElementNS("http://www.w3.org/2000/svg","path");x.setAttribute("d","M 12,12 L 28,28 M 28,12 12,28"),x.setAttribute("stroke","#fff"),x.setAttribute("stroke-width",2),h.appendChild(x),t.optionalFeatures===void 0&&(t.optionalFeatures=[]),t.optionalFeatures.push("dom-overlay"),t.domOverlay={root:d}}let c=null;async function u(d){d.addEventListener("end",f),e.xr.setReferenceSpaceType("local"),await e.xr.setSession(d),n.textContent="STOP AR",t.domOverlay.root.style.display="",c=d}function f(){c.removeEventListener("end",f),n.textContent="START AR",t.domOverlay.root.style.display="none",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="START AR",n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-ar",t).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(d=>{console.warn(d)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(d=>{console.warn(d)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="AR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="AR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="ARButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-ar").then(function(c){c?s():o()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}}const sf={Always:0,Never:2},ZE=50,JE=.75,ew=15e5,tw=10,nw=2.5,iw=60;class To{constructor(e={}){if(e.cameraUp||(e.cameraUp=[0,1,0]),this.cameraUp=new B().fromArray(e.cameraUp),e.initialCameraPosition||(e.initialCameraPosition=[0,10,15]),this.initialCameraPosition=new B().fromArray(e.initialCameraPosition),e.initialCameraLookAt||(e.initialCameraLookAt=[0,0,0]),this.initialCameraLookAt=new B().fromArray(e.initialCameraLookAt),this.dropInMode=e.dropInMode||!1,(e.selfDrivenMode===void 0||e.selfDrivenMode===null)&&(e.selfDrivenMode=!0),this.selfDrivenMode=e.selfDrivenMode&&!this.dropInMode,this.selfDrivenUpdateFunc=this.selfDrivenUpdate.bind(this),e.useBuiltInControls===void 0&&(e.useBuiltInControls=!0),this.useBuiltInControls=e.useBuiltInControls,this.rootElement=e.rootElement,this.ignoreDevicePixelRatio=e.ignoreDevicePixelRatio||!1,this.devicePixelRatio=this.ignoreDevicePixelRatio?1:window.devicePixelRatio||1,this.halfPrecisionCovariancesOnGPU=e.halfPrecisionCovariancesOnGPU||!1,this.threeScene=e.threeScene,this.renderer=e.renderer,this.camera=e.camera,this.gpuAcceleratedSort=e.gpuAcceleratedSort||!1,(e.integerBasedSort===void 0||e.integerBasedSort===null)&&(e.integerBasedSort=!0),this.integerBasedSort=e.integerBasedSort,(e.sharedMemoryForWorkers===void 0||e.sharedMemoryForWorkers===null)&&(e.sharedMemoryForWorkers=!0),this.sharedMemoryForWorkers=e.sharedMemoryForWorkers,this.dynamicScene=!!e.dynamicScene,this.antialiased=e.antialiased||!1,this.kernel2DSize=e.kernel2DSize===void 0?.3:e.kernel2DSize,this.webXRMode=e.webXRMode||Lr.None,this.webXRMode!==Lr.None&&(this.gpuAcceleratedSort=!1),this.webXRActive=!1,this.webXRSessionInit=e.webXRSessionInit||{},this.renderMode=e.renderMode||sf.Always,this.sceneRevealMode=e.sceneRevealMode||za.Default,this.focalAdjustment=e.focalAdjustment||1,this.maxScreenSpaceSplatSize=e.maxScreenSpaceSplatSize||1024,this.logLevel=e.logLevel||No.None,this.sphericalHarmonicsDegree=e.sphericalHarmonicsDegree||0,this.enableOptionalEffects=e.enableOptionalEffects||!1,(e.enableSIMDInSort===void 0||e.enableSIMDInSort===null)&&(e.enableSIMDInSort=!0),this.enableSIMDInSort=e.enableSIMDInSort,(e.inMemoryCompressionLevel===void 0||e.inMemoryCompressionLevel===null)&&(e.inMemoryCompressionLevel=0),this.inMemoryCompressionLevel=e.inMemoryCompressionLevel,(e.optimizeSplatData===void 0||e.optimizeSplatData===null)&&(e.optimizeSplatData=!0),this.optimizeSplatData=e.optimizeSplatData,(e.freeIntermediateSplatData===void 0||e.freeIntermediateSplatData===null)&&(e.freeIntermediateSplatData=!1),this.freeIntermediateSplatData=e.freeIntermediateSplatData,uh()){const n=rx();n.major<17&&(this.enableSIMDInSort=!1),n.major<16&&(this.sharedMemoryForWorkers=!1)}(e.splatRenderMode===void 0||e.splatRenderMode===null)&&(e.splatRenderMode=Cs.ThreeD),this.splatRenderMode=e.splatRenderMode,this.sceneFadeInRateMultiplier=e.sceneFadeInRateMultiplier||1,this.splatSortDistanceMapPrecision=e.splatSortDistanceMapPrecision||Nt.DefaultSplatSortDistanceMapPrecision;const t=this.integerBasedSort?20:24;this.splatSortDistanceMapPrecision=$t(this.splatSortDistanceMapPrecision,10,t),this.onSplatMeshChangedCallback=null,this.createSplatMesh(),this.controls=null,this.perspectiveControls=null,this.orthographicControls=null,this.orthographicCamera=null,this.perspectiveCamera=null,this.showMeshCursor=!1,this.showControlPlane=!1,this.showInfo=!1,this.sceneHelper=null,this.sortWorker=null,this.sortRunning=!1,this.splatRenderCount=0,this.splatSortCount=0,this.lastSplatSortCount=0,this.sortWorkerIndexesToSort=null,this.sortWorkerSortedIndexes=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.preSortMessages=[],this.runAfterNextSort=[],this.selfDrivenModeRunning=!1,this.splatRenderReady=!1,this.raycaster=new RE,this.infoPanel=null,this.startInOrthographicMode=!1,this.currentFPS=0,this.lastSortTime=0,this.consecutiveRenderFrames=0,this.previousCameraTarget=new B,this.nextCameraTarget=new B,this.mousePosition=new Ke,this.mouseDownPosition=new Ke,this.mouseDownTime=null,this.resizeObserver=null,this.mouseMoveListener=null,this.mouseDownListener=null,this.mouseUpListener=null,this.keyDownListener=null,this.sortPromise=null,this.sortPromiseResolver=null,this.splatSceneDownloadPromises={},this.splatSceneDownloadAndBuildPromise=null,this.splatSceneRemovalPromise=null,this.loadingSpinner=new bh(null,this.rootElement||document.body),this.loadingSpinner.hide(),this.loadingProgressBar=new bE(this.rootElement||document.body),this.loadingProgressBar.hide(),this.infoPanel=new ME(this.rootElement||document.body),this.infoPanel.hide(),this.usingExternalCamera=!!(this.dropInMode||this.camera),this.usingExternalRenderer=!!(this.dropInMode||this.renderer),this.initialized=!1,this.disposing=!1,this.disposed=!1,this.disposePromise=null,this.dropInMode||this.init()}createSplatMesh(){this.splatMesh=new bn(this.splatRenderMode,this.dynamicScene,this.enableOptionalEffects,this.halfPrecisionCovariancesOnGPU,this.devicePixelRatio,this.gpuAcceleratedSort,this.integerBasedSort,this.antialiased,this.maxScreenSpaceSplatSize,this.logLevel,this.sphericalHarmonicsDegree,this.sceneFadeInRateMultiplier,this.kernel2DSize),this.splatMesh.frustumCulled=!1,this.onSplatMeshChangedCallback&&this.onSplatMeshChangedCallback()}init(){this.initialized||(this.rootElement||(this.usingExternalRenderer?this.rootElement=this.renderer.domElement||document.body:(this.rootElement=document.createElement("div"),this.rootElement.style.width="100%",this.rootElement.style.height="100%",this.rootElement.style.position="absolute",document.body.appendChild(this.rootElement))),this.setupCamera(),this.setupRenderer(),this.setupWebXR(this.webXRSessionInit),this.setupControls(),this.setupEventHandlers(),this.threeScene=this.threeScene||new My,this.sceneHelper=new Oa(this.threeScene),this.sceneHelper.setupMeshCursor(),this.sceneHelper.setupFocusMarker(),this.sceneHelper.setupControlPlane(),this.loadingProgressBar.setContainer(this.rootElement),this.loadingSpinner.setContainer(this.rootElement),this.infoPanel.setContainer(this.rootElement),this.initialized=!0)}setupCamera(){if(!this.usingExternalCamera){const e=new Ke;this.getRenderDimensions(e),this.perspectiveCamera=new Ti(ZE,e.x/e.y,.1,1e3),this.orthographicCamera=new lh(e.x/-2,e.x/2,e.y/2,e.y/-2,.1,1e3),this.camera=this.startInOrthographicMode?this.orthographicCamera:this.perspectiveCamera,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt)}}setupRenderer(){if(!this.usingExternalRenderer){const e=new Ke;this.getRenderDimensions(e),this.renderer=new S1({antialias:!1,precision:"highp"}),this.renderer.setPixelRatio(this.devicePixelRatio),this.renderer.autoClear=!0,this.renderer.setClearColor(new bt(0),0),this.renderer.setSize(e.x,e.y),this.resizeObserver=new ResizeObserver(()=>{this.getRenderDimensions(e),this.renderer.setSize(e.x,e.y),this.forceRenderNextFrame()}),this.resizeObserver.observe(this.rootElement),this.rootElement.appendChild(this.renderer.domElement)}}setupWebXR(e){this.webXRMode&&(this.webXRMode===Lr.VR?this.rootElement.appendChild($o.createButton(this.renderer,e)):this.webXRMode===Lr.AR&&this.rootElement.appendChild($E.createButton(this.renderer,e)),this.renderer.xr.addEventListener("sessionstart",t=>{this.webXRActive=!0}),this.renderer.xr.addEventListener("sessionend",t=>{this.webXRActive=!1}),this.renderer.xr.enabled=!0,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt))}setupControls(){if(this.useBuiltInControls&&this.webXRMode===Lr.None){this.usingExternalCamera?this.camera.isOrthographicCamera?this.orthographicControls=new ec(this.camera,this.renderer.domElement):this.perspectiveControls=new ec(this.camera,this.renderer.domElement):(this.perspectiveControls=new ec(this.perspectiveCamera,this.renderer.domElement),this.orthographicControls=new ec(this.orthographicCamera,this.renderer.domElement));for(let e of[this.orthographicControls,this.perspectiveControls])e&&(e.listenToKeyEvents(window),e.rotateSpeed=.5,e.maxPolarAngle=Math.PI*.75,e.minPolarAngle=.1,e.enableDamping=!0,e.dampingFactor=.05,e.target.copy(this.initialCameraLookAt),e.update());this.controls=this.camera.isOrthographicCamera?this.orthographicControls:this.perspectiveControls,this.controls.update()}}setupEventHandlers(){this.useBuiltInControls&&this.webXRMode===Lr.None&&(this.mouseMoveListener=this.onMouseMove.bind(this),this.renderer.domElement.addEventListener("pointermove",this.mouseMoveListener,!1),this.mouseDownListener=this.onMouseDown.bind(this),this.renderer.domElement.addEventListener("pointerdown",this.mouseDownListener,!1),this.mouseUpListener=this.onMouseUp.bind(this),this.renderer.domElement.addEventListener("pointerup",this.mouseUpListener,!1),this.keyDownListener=this.onKeyDown.bind(this),window.addEventListener("keydown",this.keyDownListener,!1))}removeEventHandlers(){this.useBuiltInControls&&(this.renderer.domElement.removeEventListener("pointermove",this.mouseMoveListener),this.mouseMoveListener=null,this.renderer.domElement.removeEventListener("pointerdown",this.mouseDownListener),this.mouseDownListener=null,this.renderer.domElement.removeEventListener("pointerup",this.mouseUpListener),this.mouseUpListener=null,window.removeEventListener("keydown",this.keyDownListener),this.keyDownListener=null)}setRenderMode(e){this.renderMode=e}setActiveSphericalHarmonicsDegrees(e){this.splatMesh.material.uniforms.sphericalHarmonicsDegree.value=e,this.splatMesh.material.uniformsNeedUpdate=!0}onSplatMeshChanged(e){this.onSplatMeshChangedCallback=e}onKeyDown=(function(){const e=new B,t=new st,n=new st;return function(s){switch(e.set(0,0,-1),e.transformDirection(this.camera.matrixWorld),t.makeRotationAxis(e,Math.PI/128),n.makeRotationAxis(e,-Math.PI/128),s.code){case"KeyG":this.focalAdjustment+=.02,this.forceRenderNextFrame();break;case"KeyF":this.focalAdjustment-=.02,this.forceRenderNextFrame();break;case"ArrowLeft":this.camera.up.transformDirection(t);break;case"ArrowRight":this.camera.up.transformDirection(n);break;case"KeyC":this.showMeshCursor=!this.showMeshCursor;break;case"KeyU":this.showControlPlane=!this.showControlPlane;break;case"KeyI":this.showInfo=!this.showInfo,this.showInfo?this.infoPanel.show():this.infoPanel.hide();break;case"KeyO":this.usingExternalCamera||this.setOrthographicMode(!this.camera.isOrthographicCamera);break;case"KeyP":this.usingExternalCamera||this.splatMesh.setPointCloudModeEnabled(!this.splatMesh.getPointCloudModeEnabled());break;case"Equal":this.usingExternalCamera||this.splatMesh.setSplatScale(this.splatMesh.getSplatScale()+.05);break;case"Minus":this.usingExternalCamera||this.splatMesh.setSplatScale(Math.max(this.splatMesh.getSplatScale()-.05,0));break}}})();onMouseMove(e){this.mousePosition.set(e.offsetX,e.offsetY)}onMouseDown(){this.mouseDownPosition.copy(this.mousePosition),this.mouseDownTime=Ao()}onMouseUp=(function(){const e=new Ke;return function(t){e.copy(this.mousePosition).sub(this.mouseDownPosition),Ao()-this.mouseDownTime<.5&&e.length()<2&&this.onMouseClick(t)}})();onMouseClick(e){this.mousePosition.set(e.offsetX,e.offsetY),this.checkForFocalPointChange()}checkForFocalPointChange=(function(){const e=new Ke,t=new B,n=[];return function(){if(!this.transitioningCameraTarget&&(this.getRenderDimensions(e),n.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,e),this.raycaster.intersectSplatMesh(this.splatMesh,n),n.length>0)){const r=n[0].origin;t.copy(r).sub(this.camera.position),t.length()>JE&&(this.previousCameraTarget.copy(this.controls.target),this.nextCameraTarget.copy(r),this.transitioningCameraTarget=!0,this.transitioningCameraTargetStartTime=Ao())}}})();getRenderDimensions(e){this.rootElement?(e.x=this.rootElement.offsetWidth,e.y=this.rootElement.offsetHeight):this.renderer.getSize(e)}setOrthographicMode(e){if(e===this.camera.isOrthographicCamera)return;const t=this.camera,n=e?this.orthographicCamera:this.perspectiveCamera;if(n.position.copy(t.position),n.up.copy(t.up),n.rotation.copy(t.rotation),n.quaternion.copy(t.quaternion),n.matrix.copy(t.matrix),this.camera=n,this.controls){const s=a=>{a.saveState(),a.reset()},r=this.controls,o=e?this.orthographicControls:this.perspectiveControls;s(o),s(r),o.target.copy(r.target),e?To.setCameraZoomFromPosition(n,t,r):To.setCameraPositionFromZoom(n,t,o),this.controls=o,this.camera.lookAt(this.controls.target)}}static setCameraPositionFromZoom=(function(){const e=new B;return function(t,n,s){const r=1/(n.zoom*.001);e.copy(s.target).sub(t.position).normalize().multiplyScalar(r).negate(),t.position.copy(s.target).add(e)}})();static setCameraZoomFromPosition=(function(){const e=new B;return function(t,n,s){const r=e.copy(s.target).sub(n.position).length();t.zoom=1/(r*.001)}})();updateSplatMesh=(function(){const e=new Ke;return function(){if(!this.splatMesh)return;if(this.splatMesh.getSplatCount()>0){this.splatMesh.updateVisibleRegionFadeDistance(this.sceneRevealMode),this.splatMesh.updateTransforms(),this.getRenderDimensions(e);const n=this.camera.projectionMatrix.elements[0]*.5*this.devicePixelRatio*e.x,s=this.camera.projectionMatrix.elements[5]*.5*this.devicePixelRatio*e.y,r=this.camera.isOrthographicCamera?1/this.devicePixelRatio:1,o=this.focalAdjustment*r,a=1/o;this.adjustForWebXRStereo(e),this.splatMesh.updateUniforms(e,n*o,s*o,this.camera.isOrthographicCamera,this.camera.zoom||1,a)}}})();adjustForWebXRStereo(e){if(this.camera&&this.webXRActive){const n=this.renderer.xr.getCamera().projectionMatrix.elements[0],s=this.camera.projectionMatrix.elements[0];e.x*=s/n}}isLoadingOrUnloading(){return Object.keys(this.splatSceneDownloadPromises).length>0||this.splatSceneDownloadAndBuildPromise!==null||this.splatSceneRemovalPromise!==null}isDisposingOrDisposed(){return this.disposing||this.disposed}addSplatSceneDownloadPromise(e){this.splatSceneDownloadPromises[e.id]=e}removeSplatSceneDownloadPromise(e){delete this.splatSceneDownloadPromises[e.id]}setSplatSceneDownloadAndBuildPromise(e){this.splatSceneDownloadAndBuildPromise=e}clearSplatSceneDownloadAndBuildPromise(){this.splatSceneDownloadAndBuildPromise=null}addSplatScene(e,t={}){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");t.progressiveLoad&&this.splatMesh.scenes&&this.splatMesh.scenes.length>0&&(console.log('addSplatScene(): "progressiveLoad" option ignore because there are multiple splat scenes'),t.progressiveLoad=!1);const n=t.format!==void 0&&t.format!==null?t.format:Xm(e),s=To.isProgressivelyLoadable(n)&&t.progressiveLoad,r=t.showLoadingUI!==void 0&&t.showLoadingUI!==null?t.showLoadingUI:!0;let o=null;r&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=()=>{this.loadingProgressBar.hide(),this.loadingSpinner.removeAllTasks()},l=(p,g,m)=>{if(r)if(m===un.Downloading)if(p==100)this.loadingSpinner.setMessageForTask(o,"Download complete!");else if(s)this.loadingSpinner.setMessageForTask(o,"Downloading splats...");else{const _=g?`: ${g}`:"...";this.loadingSpinner.setMessageForTask(o,`Downloading${_}`)}else m===un.Processing&&this.loadingSpinner.setMessageForTask(o,"Processing splats...")};let c=!1,u=0;const f=(p,g)=>{r&&((p&&s||g&&!s)&&(this.loadingSpinner.removeTask(o),!g&&!c&&this.loadingProgressBar.show()),s&&(g?(c=!0,this.loadingProgressBar.hide()):this.loadingProgressBar.setProgress(u)))},d=(p,g,m)=>{u=p,l(p,g,m),t.onProgress&&t.onProgress(p,g,m)},h=(p,g,m)=>{!s&&t.onProgress&&t.onProgress(0,"0%",un.Processing);const _={rotation:t.rotation||t.orientation,position:t.position,scale:t.scale,splatAlphaRemovalThreshold:t.splatAlphaRemovalThreshold};return this.addSplatBuffers([p],[_],m,g&&r,r,s,s).then(()=>{!s&&t.onProgress&&t.onProgress(100,"100%",un.Processing),f(g,m)})};return(s?this.downloadAndBuildSingleSplatSceneProgressiveLoad.bind(this):this.downloadAndBuildSingleSplatSceneStandardLoad.bind(this))(e,n,t.splatAlphaRemovalThreshold,h.bind(this),d,a.bind(this),t.headers)}downloadAndBuildSingleSplatSceneStandardLoad(e,t,n,s,r,o,a){const l=this.downloadSplatSceneToSplatBuffer(e,n,r,!1,void 0,t,a),c=Ku(l.abortHandler);return l.then(u=>(this.removeSplatSceneDownloadPromise(l),s(u,!0,!0).then(()=>{c.resolve(),this.clearSplatSceneDownloadAndBuildPromise()}))).catch(u=>{o&&o(),this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(l),c.reject(this.updateError(u,`Viewer::addSplatScene -> Could not load file ${e}`))}),this.addSplatSceneDownloadPromise(l),this.setSplatSceneDownloadAndBuildPromise(c.promise),c.promise}downloadAndBuildSingleSplatSceneProgressiveLoad(e,t,n,s,r,o,a){let l=0,c=!1;const u=[],f=()=>{if(u.length>0&&!c&&!this.isDisposingOrDisposed()){c=!0;const g=u.shift();s(g.splatBuffer,g.firstBuild,g.finalBuild).then(()=>{c=!1,g.firstBuild?x.resolve():g.finalBuild&&(p.resolve(),this.clearSplatSceneDownloadAndBuildPromise()),u.length>0&&pi(()=>f())})}},d=(g,m)=>{this.isDisposingOrDisposed()||(m||u.length===0||g.getSplatCount()>u[0].splatBuffer.getSplatCount())&&(u.push({splatBuffer:g,firstBuild:l===0,finalBuild:m}),l++,f())},h=this.downloadSplatSceneToSplatBuffer(e,n,r,!0,d,t,a),x=Ku(h.abortHandler),p=Ku();return this.addSplatSceneDownloadPromise(h),this.setSplatSceneDownloadAndBuildPromise(p.promise),h.then(()=>{this.removeSplatSceneDownloadPromise(h)}).catch(g=>{this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(h);const m=this.updateError(g,"Viewer::addSplatScene -> Could not load one or more scenes");x.reject(m),o&&o(m)}),x.promise}addSplatScenes(e,t=!0,n=void 0){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");const s=e.length,r=[];let o;t&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=(f,d,h,x)=>{r[f]=d;let p=0;for(let g=0;g<s;g++)p+=r[g]||0;p=p/s,h=`${p.toFixed(2)}%`,t&&x===un.Downloading&&this.loadingSpinner.setMessageForTask(o,p==100?"Download complete!":`Downloading: ${h}`),n&&n(p,h,x)},l=[],c=[];for(let f=0;f<e.length;f++){const d=e[f],h=d.format!==void 0&&d.format!==null?d.format:Xm(d.path),x=this.downloadSplatSceneToSplatBuffer(d.path,d.splatAlphaRemovalThreshold,a.bind(this,f),!1,void 0,h,d.headers);l.push(x),c.push(x.promise)}const u=new Zs((f,d)=>{Promise.all(c).then(h=>{t&&this.loadingSpinner.removeTask(o),n&&n(0,"0%",un.Processing),this.addSplatBuffers(h,e,!0,t,t,!1,!1).then(()=>{n&&n(100,"100%",un.Processing),this.clearSplatSceneDownloadAndBuildPromise(),f()})}).catch(h=>{t&&this.loadingSpinner.removeTask(o),this.clearSplatSceneDownloadAndBuildPromise(),d(this.updateError(h,"Viewer::addSplatScenes -> Could not load one or more splat scenes."))}).finally(()=>{this.removeSplatSceneDownloadPromise(u)})},f=>{for(let d of l)d.abort(f)});return this.addSplatSceneDownloadPromise(u),this.setSplatSceneDownloadAndBuildPromise(u),u}downloadSplatSceneToSplatBuffer(e,t=1,n=void 0,s=!1,r=void 0,o,a){try{if(o===Zn.Splat||o===Zn.KSplat||o===Zn.Ply){const l=s?!1:this.optimizeSplatData;if(o===Zn.Splat)return yh.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,a);if(o===Zn.KSplat)return Ua.loadFromURL(e,n,s,r,a);if(o===Zn.Ply)return Ah.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,this.sphericalHarmonicsDegree,a)}else if(o===Zn.Spz)return Sh.loadFromURL(e,n,t,this.inMemoryCompressionLevel,this.optimizeSplatData,this.sphericalHarmonicsDegree,a)}catch(l){throw this.updateError(l,null)}throw new Error(`Viewer::downloadSplatSceneToSplatBuffer -> File format not supported: ${e}`)}static isProgressivelyLoadable(e){return e===Zn.Splat||e===Zn.KSplat||e===Zn.Ply}addSplatBuffers=(function(){return function(e,t=[],n=!0,s=!0,r=!0,o=!1,a=!1,l=!0){if(this.isDisposingOrDisposed())return Promise.resolve();let c=null;const u=()=>{c!==null&&(this.loadingSpinner.removeTask(c),c=null)};return this.splatRenderReady=!1,new Promise(f=>{s&&(c=this.loadingSpinner.addTask("Processing splats...")),pi(()=>{if(this.isDisposingOrDisposed())f();else{const d=this.addSplatBuffersToMesh(e,t,n,r,o,l),h=this.splatMesh.getMaxSplatCount();this.sortWorker&&this.sortWorker.maxSplatCount!==h&&this.disposeSortWorker(),this.gpuAcceleratedSort||this.preSortMessages.push({centers:d.centers.buffer,sceneIndexes:d.sceneIndexes.buffer,range:{from:d.from,to:d.to,count:d.count}}),(!this.sortWorker&&h>0?this.setupSortWorker(this.splatMesh):Promise.resolve()).then(()=>{this.isDisposingOrDisposed()||this.runSplatSort(!0,!0).then(p=>{!this.sortWorker||!p?(this.splatRenderReady=!0,u(),f()):(a?this.splatRenderReady=!0:this.runAfterNextSort.push(()=>{this.splatRenderReady=!0}),this.runAfterNextSort.push(()=>{u(),f()}))})})}},!0)})}})();addSplatBuffersToMesh=(function(){let e;return function(t,n,s=!0,r=!1,o=!1,a=!0){if(this.isDisposingOrDisposed())return;let l=[],c=[];o||(l=this.splatMesh.scenes.map(h=>h.splatBuffer)||[],c=this.splatMesh.sceneOptions?this.splatMesh.sceneOptions.map(h=>h):[]),l.push(...t),c.push(...n),this.renderer&&this.splatMesh.setRenderer(this.renderer);const u=h=>{if(this.isDisposingOrDisposed())return;const x=this.splatMesh.getSplatCount();r&&x>=ew&&!h&&!e&&(this.loadingSpinner.setMinimized(!0,!0),e=this.loadingSpinner.addTask("Optimizing data structures..."))},f=h=>{this.isDisposingOrDisposed()||h&&e&&(this.loadingSpinner.removeTask(e),e=null)},d=this.splatMesh.build(l,c,!0,s,u,f,a);return s&&this.freeIntermediateSplatData&&this.splatMesh.freeIntermediateSplatData(),d}})();setupSortWorker(e){if(!this.isDisposingOrDisposed())return new Promise(t=>{const n=this.integerBasedSort?Int32Array:Float32Array,s=e.getSplatCount(),r=e.getMaxSplatCount();this.sortWorker=jE(r,this.sharedMemoryForWorkers,this.enableSIMDInSort,this.integerBasedSort,this.splatMesh.dynamicMode,this.splatSortDistanceMapPrecision),this.sortWorker.onmessage=o=>{if(o.data.sortDone){if(this.sortRunning=!1,this.sharedMemoryForWorkers)this.splatMesh.updateRenderIndexes(this.sortWorkerSortedIndexes,o.data.splatRenderCount);else{const a=new Uint32Array(o.data.sortedIndexes.buffer,0,o.data.splatRenderCount);this.splatMesh.updateRenderIndexes(a,o.data.splatRenderCount)}this.lastSplatSortCount=this.splatSortCount,this.lastSortTime=o.data.sortTime,this.sortPromiseResolver(),this.sortPromiseResolver=null,this.forceRenderNextFrame(),this.runAfterNextSort.length>0&&(this.runAfterNextSort.forEach(a=>{a()}),this.runAfterNextSort.length=0)}else if(o.data.sortCanceled)this.sortRunning=!1;else if(o.data.sortSetupPhase1Complete){this.logLevel>=No.Info&&console.log("Sorting web worker WASM setup complete."),this.sharedMemoryForWorkers?(this.sortWorkerSortedIndexes=new Uint32Array(o.data.sortedIndexesBuffer,o.data.sortedIndexesOffset,r),this.sortWorkerIndexesToSort=new Uint32Array(o.data.indexesToSortBuffer,o.data.indexesToSortOffset,r),this.sortWorkerPrecomputedDistances=new n(o.data.precomputedDistancesBuffer,o.data.precomputedDistancesOffset,r),this.sortWorkerTransforms=new Float32Array(o.data.transformsBuffer,o.data.transformsOffset,Nt.MaxScenes*16)):(this.sortWorkerIndexesToSort=new Uint32Array(r),this.sortWorkerPrecomputedDistances=new n(r),this.sortWorkerTransforms=new Float32Array(Nt.MaxScenes*16));for(let a=0;a<s;a++)this.sortWorkerIndexesToSort[a]=a;if(this.sortWorker.maxSplatCount=r,this.logLevel>=No.Info){console.log("Sorting web worker ready.");const a=this.splatMesh.getSplatDataTextures(),l=a.covariances.size,c=a.centerColors.size;console.log("Covariances texture size: "+l.x+" x "+l.y),console.log("Centers/colors texture size: "+c.x+" x "+c.y)}t()}}})}updateError(e,t){return e instanceof sx?e:e instanceof Ec?new Error("File type or server does not support progressive loading."):t?new Error(t):e}disposeSortWorker(){this.sortWorker&&this.sortWorker.terminate(),this.sortWorker=null,this.sortPromise=null,this.sortPromiseResolver&&(this.sortPromiseResolver(),this.sortPromiseResolver=null),this.preSortMessages=[],this.sortRunning=!1}removeSplatScene(e,t=!0){return this.removeSplatScenes([e],t)}removeSplatScenes(e,t=!0){if(this.isLoadingOrUnloading())throw new Error("Cannot remove splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot remove splat scene after dispose() is called.");let n;return this.splatSceneRemovalPromise=new Promise((s,r)=>{let o;t&&(this.loadingSpinner.removeAllTasks(),this.loadingSpinner.show(),o=this.loadingSpinner.addTask("Removing splat scene..."));const a=()=>{t&&(this.loadingSpinner.hide(),this.loadingSpinner.removeTask(o))},l=u=>{a(),this.splatSceneRemovalPromise=null,u?r(u):s()},c=()=>this.isDisposingOrDisposed()?(l(),!0):!1;n=this.sortPromise||Promise.resolve(),n.then(()=>{if(c())return;const u=[],f=[],d=[];for(let h=0;h<this.splatMesh.scenes.length;h++){let x=!1;for(let p of e)if(p===h){x=!0;break}if(!x){const p=this.splatMesh.scenes[h];u.push(p.splatBuffer),f.push(this.splatMesh.sceneOptions[h]),d.push({position:p.position.clone(),quaternion:p.quaternion.clone(),scale:p.scale.clone()})}}this.disposeSortWorker(),this.splatMesh.dispose(),this.sceneRevealMode=za.Instant,this.createSplatMesh(),this.addSplatBuffers(u,f,!0,!1,!0).then(()=>{c()||(a(),this.splatMesh.scenes.forEach((h,x)=>{h.position.copy(d[x].position),h.quaternion.copy(d[x].quaternion),h.scale.copy(d[x].scale)}),this.splatMesh.updateTransforms(),this.splatRenderReady=!1,this.runSplatSort(!0).then(()=>{if(c()){this.splatRenderReady=!0;return}n=this.sortPromise||Promise.resolve(),n.then(()=>{this.splatRenderReady=!0,l()})}))}).catch(h=>{l(h)})})}),this.splatSceneRemovalPromise}start(){if(this.selfDrivenMode)this.webXRMode?this.renderer.setAnimationLoop(this.selfDrivenUpdateFunc):this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc),this.selfDrivenModeRunning=!0;else throw new Error("Cannot start viewer unless it is in self driven mode.")}stop(){this.selfDrivenMode&&this.selfDrivenModeRunning&&(this.webXRMode?this.renderer.setAnimationLoop(null):cancelAnimationFrame(this.requestFrameId),this.selfDrivenModeRunning=!1)}async dispose(){if(this.isDisposingOrDisposed())return this.disposePromise;let e=[],t=[];for(let n in this.splatSceneDownloadPromises)if(this.splatSceneDownloadPromises.hasOwnProperty(n)){const s=this.splatSceneDownloadPromises[n];t.push(s),e.push(s.promise)}return this.sortPromise&&e.push(this.sortPromise),this.disposing=!0,this.disposePromise=Promise.all(e).finally(()=>{this.stop(),this.orthographicControls&&(this.orthographicControls.dispose(),this.orthographicControls=null),this.perspectiveControls&&(this.perspectiveControls.dispose(),this.perspectiveControls=null),this.controls=null,this.splatMesh&&(this.splatMesh.dispose(),this.splatMesh=null),this.sceneHelper&&(this.sceneHelper.dispose(),this.sceneHelper=null),this.resizeObserver&&(this.resizeObserver.unobserve(this.rootElement),this.resizeObserver=null),this.disposeSortWorker(),this.removeEventHandlers(),this.loadingSpinner.removeAllTasks(),this.loadingSpinner.setContainer(null),this.loadingProgressBar.hide(),this.loadingProgressBar.setContainer(null),this.infoPanel.setContainer(null),this.camera=null,this.threeScene=null,this.splatRenderReady=!1,this.initialized=!1,this.renderer&&(this.usingExternalRenderer||(this.rootElement.removeChild(this.renderer.domElement),this.renderer.dispose()),this.renderer=null),this.usingExternalRenderer||document.body.removeChild(this.rootElement),this.sortWorkerSortedIndexes=null,this.sortWorkerIndexesToSort=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.disposed=!0,this.disposing=!1,this.disposePromise=null}),t.forEach(n=>{n.abort("Scene disposed")}),this.disposePromise}selfDrivenUpdate(){this.selfDrivenMode&&!this.webXRMode&&(this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc)),this.update(),this.shouldRender()?(this.render(),this.consecutiveRenderFrames++):this.consecutiveRenderFrames=0,this.renderNextFrame=!1}forceRenderNextFrame(){this.renderNextFrame=!0}shouldRender=(function(){let e=0;const t=new B,n=new Vt,s=1e-4;return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return!1;let r=!1,o=!1;if(this.camera){const a=this.camera.position,l=this.camera.quaternion;o=Math.abs(a.x-t.x)>s||Math.abs(a.y-t.y)>s||Math.abs(a.z-t.z)>s||Math.abs(l.x-n.x)>s||Math.abs(l.y-n.y)>s||Math.abs(l.z-n.z)>s||Math.abs(l.w-n.w)>s}return r=this.renderMode!==sf.Never&&(e===0||this.splatMesh.visibleRegionChanging||o||this.renderMode===sf.Always||this.dynamicMode===!0||this.renderNextFrame),this.camera&&(t.copy(this.camera.position),n.copy(this.camera.quaternion)),e++,r}})();render=(function(){return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return;const e=n=>{for(let s of n.children)if(s.visible)return!0;return!1},t=this.renderer.autoClear;e(this.threeScene)&&(this.renderer.render(this.threeScene,this.camera),this.renderer.autoClear=!1),this.renderer.render(this.splatMesh,this.camera),this.renderer.autoClear=!1,this.sceneHelper.getFocusMarkerOpacity()>0&&this.renderer.render(this.sceneHelper.focusMarker,this.camera),this.showControlPlane&&this.renderer.render(this.sceneHelper.controlPlane,this.camera),this.renderer.autoClear=t}})();update(e,t){this.dropInMode&&this.updateForDropInMode(e,t),!(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())&&(this.controls&&(this.controls.update(),this.camera.isOrthographicCamera&&!this.usingExternalCamera&&To.setCameraPositionFromZoom(this.camera,this.camera,this.controls)),this.runSplatSort(),this.updateForRendererSizeChanges(),this.updateSplatMesh(),this.updateMeshCursor(),this.updateFPS(),this.timingSensitiveUpdates(),this.updateInfoPanel(),this.updateControlPlane())}updateForDropInMode(e,t){this.renderer=e,this.splatMesh&&this.splatMesh.setRenderer(this.renderer),this.camera=t,this.controls&&(this.controls.object=t),this.init()}updateFPS=(function(){let e=Ao(),t=0;return function(){if(this.consecutiveRenderFrames>iw){const n=Ao();n-e>=1?(this.currentFPS=t,t=0,e=n):t++}else this.currentFPS=null}})();updateForRendererSizeChanges=(function(){const e=new Ke,t=new Ke;let n;return function(){this.usingExternalCamera||(this.renderer.getSize(t),(n===void 0||n!==this.camera.isOrthographicCamera||t.x!==e.x||t.y!==e.y)&&(this.camera.isOrthographicCamera?(this.camera.left=-t.x/2,this.camera.right=t.x/2,this.camera.top=t.y/2,this.camera.bottom=-t.y/2):this.camera.aspect=t.x/t.y,this.camera.updateProjectionMatrix(),e.copy(t),n=this.camera.isOrthographicCamera))}})();timingSensitiveUpdates=(function(){let e;return function(){const t=Ao();e||(e=t);const n=t-e;this.updateCameraTransition(t),this.updateFocusMarker(n),e=t}})();updateCameraTransition=(function(){let e=new B,t=new B,n=new B;return function(s){if(this.transitioningCameraTarget){t.copy(this.previousCameraTarget).sub(this.camera.position).normalize(),n.copy(this.nextCameraTarget).sub(this.camera.position).normalize();const r=Math.acos(t.dot(n)),a=(r/(Math.PI/3)*.65+.3)/r*(s-this.transitioningCameraTargetStartTime);e.copy(this.previousCameraTarget).lerp(this.nextCameraTarget,a),this.camera.lookAt(e),this.controls.target.copy(e),a>=1&&(this.transitioningCameraTarget=!1)}}})();updateFocusMarker=(function(){const e=new Ke;let t=!1;return function(n){if(this.getRenderDimensions(e),this.transitioningCameraTarget){this.sceneHelper.setFocusMarkerVisibility(!0);const s=Math.max(this.sceneHelper.getFocusMarkerOpacity(),0);let r=Math.min(s+tw*n,1);this.sceneHelper.setFocusMarkerOpacity(r),this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e),t=!0,this.forceRenderNextFrame()}else{let s;if(t?s=1:s=Math.min(this.sceneHelper.getFocusMarkerOpacity(),1),s>0){this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e);let r=Math.max(s-nw*n,0);this.sceneHelper.setFocusMarkerOpacity(r),r===0&&this.sceneHelper.setFocusMarkerVisibility(!1)}s>0&&this.forceRenderNextFrame(),t=!1}}})();updateMeshCursor=(function(){const e=[],t=new Ke;return function(){this.showMeshCursor?(this.forceRenderNextFrame(),this.getRenderDimensions(t),e.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,t),this.raycaster.intersectSplatMesh(this.splatMesh,e),e.length>0?(this.sceneHelper.setMeshCursorVisibility(!0),this.sceneHelper.positionAndOrientMeshCursor(e[0].origin,this.camera)):this.sceneHelper.setMeshCursorVisibility(!1)):(this.sceneHelper.getMeschCursorVisibility()&&this.forceRenderNextFrame(),this.sceneHelper.setMeshCursorVisibility(!1))}})();updateInfoPanel=(function(){const e=new Ke;return function(){if(!this.showInfo)return;const t=this.splatMesh.getSplatCount();this.getRenderDimensions(e);const n=this.controls?this.controls.target:null,s=this.showMeshCursor?this.sceneHelper.meshCursor.position:null,r=t>0?this.splatRenderCount/t*100:0;this.infoPanel.update(e,this.camera.position,n,this.camera.up,this.camera.isOrthographicCamera,s,this.currentFPS||"N/A",t,this.splatRenderCount,r,this.lastSortTime,this.focalAdjustment,this.splatMesh.getSplatScale(),this.splatMesh.getPointCloudModeEnabled())}})();updateControlPlane(){this.showControlPlane?(this.sceneHelper.setControlPlaneVisibility(!0),this.sceneHelper.positionAndOrientControlPlane(this.controls.target,this.camera.up)):this.sceneHelper.setControlPlaneVisibility(!1)}runSplatSort=(function(){const e=new st,t=[],n=new B(0,0,-1),s=new B(0,0,-1),r=new B,o=new B,a=[],l=[{angleThreshold:.55,sortFractions:[.125,.33333,.75]},{angleThreshold:.65,sortFractions:[.33333,.66667]},{angleThreshold:.8,sortFractions:[.5]}];return function(c=!1,u=!1){if(!this.initialized)return Promise.resolve(!1);if(this.sortRunning)return Promise.resolve(!0);if(this.splatMesh.getSplatCount()<=0)return this.splatRenderCount=0,Promise.resolve(!1);let f=0,d=0,h=!1,x=!1;if(s.set(0,0,-1).applyQuaternion(this.camera.quaternion),f=s.dot(n),d=o.copy(this.camera.position).sub(r).length(),!c&&!this.splatMesh.dynamicMode&&a.length===0&&(f<=.99&&(h=!0),d>=1&&(x=!0),!h&&!x))return Promise.resolve(!1);this.sortRunning=!0;let{splatRenderCount:p,shouldSortAll:g}=this.gatherSceneNodesForSort();g=g||u,this.splatRenderCount=p,e.copy(this.camera.matrixWorld).invert();const m=this.perspectiveCamera||this.camera;e.premultiply(m.projectionMatrix),this.splatMesh.dynamicMode||e.multiply(this.splatMesh.matrixWorld);let _=Promise.resolve(!0);return this.gpuAcceleratedSort&&(a.length<=1||a.length%2===0)&&(_=this.splatMesh.computeDistancesOnGPU(e,this.sortWorkerPrecomputedDistances)),_.then(()=>{if(a.length===0)if(this.splatMesh.dynamicMode||g)a.push(this.splatRenderCount);else{for(let S of l)if(f<S.angleThreshold){for(let y of S.sortFractions)a.push(Math.floor(this.splatRenderCount*y));break}a.push(this.splatRenderCount)}let A=Math.min(a.shift(),this.splatRenderCount);this.splatSortCount=A,t[0]=this.camera.position.x,t[1]=this.camera.position.y,t[2]=this.camera.position.z;const v={modelViewProj:e.elements,cameraPosition:t,splatRenderCount:this.splatRenderCount,splatSortCount:A,usePrecomputedDistances:this.gpuAcceleratedSort};return this.splatMesh.dynamicMode&&this.splatMesh.fillTransformsArray(this.sortWorkerTransforms),this.sharedMemoryForWorkers||(v.indexesToSort=this.sortWorkerIndexesToSort,v.transforms=this.sortWorkerTransforms,this.gpuAcceleratedSort&&(v.precomputedDistances=this.sortWorkerPrecomputedDistances)),this.sortPromise=new Promise(S=>{this.sortPromiseResolver=S}),this.preSortMessages.length>0&&(this.preSortMessages.forEach(S=>{this.sortWorker.postMessage(S)}),this.preSortMessages=[]),this.sortWorker.postMessage({sort:v}),a.length===0&&(r.copy(this.camera.position),n.copy(s)),!0}),_}})();gatherSceneNodesForSort=(function(){const e=[];let t=null;const n=new B,s=new B,r=new B,o=new st,a=new st,l=new st,c=new B,u=new B(0,0,-1),f=new B,d=h=>f.copy(h.max).sub(h.min).length();return function(h=!1){this.getRenderDimensions(c);const x=c.y/2/Math.tan(this.camera.fov/2*Sn.DEG2RAD),p=Math.atan(c.x/2/x),g=Math.atan(c.y/2/x),m=Math.cos(p),_=Math.cos(g),A=this.splatMesh.getSplatTree();if(A){a.copy(this.camera.matrixWorld).invert(),this.splatMesh.dynamicMode||a.multiply(this.splatMesh.matrixWorld);let v=0,S=0;for(let M=0;M<A.subTrees.length;M++){const E=A.subTrees[M];o.copy(a),this.splatMesh.dynamicMode&&(this.splatMesh.getSceneTransform(M,l),o.multiply(l));const b=E.nodesWithIndexes.length;for(let C=0;C<b;C++){const D=E.nodesWithIndexes[C];if(!D.data||!D.data.indexes||D.data.indexes.length===0)continue;r.copy(D.center).applyMatrix4(o);const F=r.length();r.normalize(),n.copy(r).setX(0).normalize(),s.copy(r).setY(0).normalize();const O=u.dot(s),z=u.dot(n),V=d(D),H=z<_-.6,q=O<m-.6;!h&&(q||H)&&F>V||(S+=D.data.indexes.length,e[v]=D,D.data.distanceToNode=F,v++)}}e.length=v,e.sort((M,E)=>M.data.distanceToNode<E.data.distanceToNode?-1:1);let y=S*Nt.BytesPerInt;for(let M=0;M<v;M++){const E=e[M],b=E.data.indexes.length,C=b*Nt.BytesPerInt;new Uint32Array(this.sortWorkerIndexesToSort.buffer,y-C,b).set(E.data.indexes),y-=C}return{splatRenderCount:S,shouldSortAll:!1}}else{const v=this.splatMesh.getSplatCount();if(!t||t.length!==v){t=new Uint32Array(v);for(let S=0;S<v;S++)t[S]=S}return this.sortWorkerIndexesToSort.set(t),{splatRenderCount:v,shouldSortAll:!0}}}})();getSplatMesh(){return this.splatMesh}getSplatScene(e){return this.splatMesh.getScene(e)}getSceneCount(){return this.splatMesh.getSceneCount()}isMobile(){return navigator.userAgent.includes("Mobi")}}function Ms(i){if(i===void 0)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return i}function hx(i,e){i.prototype=Object.create(e.prototype),i.prototype.constructor=i,i.__proto__=e}var _i={autoSleep:120,force3D:"auto",nullTargetWarn:1,units:{lineHeight:""}},Zo={duration:.5,overwrite:!1,delay:0},Th,Tn,Zt,Di=1e8,kt=1/Di,md=Math.PI*2,sw=md/4,rw=0,px=Math.sqrt,ow=Math.cos,aw=Math.sin,yn=function(e){return typeof e=="string"},an=function(e){return typeof e=="function"},Us=function(e){return typeof e=="number"},Eh=function(e){return typeof e>"u"},ls=function(e){return typeof e=="object"},ei=function(e){return e!==!1},wh=function(){return typeof window<"u"},nc=function(e){return an(e)||yn(e)},mx=typeof ArrayBuffer=="function"&&ArrayBuffer.isView||function(){},Bn=Array.isArray,lw=/random\([^)]+\)/g,cw=/,\s*/g,t0=/(?:-?\.?\d|\.)+/gi,gx=/[-+=.]*\d+[.e\-+]*\d*[e\-+]*\d*/g,Eo=/[-+=.]*\d+[.e-]*\d*[a-z%]*/g,rf=/[-+=.]*\d+\.?\d*(?:e-|e\+)?\d*/gi,xx=/[+-]=-?[.\d]+/,uw=/[^,'"\[\]\s]+/gi,fw=/^[+\-=e\s\d]*\d+[.\d]*([a-z]*|%)\s*$/i,nn,Yi,gd,Rh,vi={},Ic={},_x,vx=function(e){return(Ic=Jo(e,vi))&&si},Ih=function(e,t){return console.warn("Invalid property",e,"set to",t,"Missing plugin? gsap.registerPlugin()")},il=function(e,t){return!t&&console.warn(e)},Ax=function(e,t){return e&&(vi[e]=t)&&Ic&&(Ic[e]=t)||vi},sl=function(){return 0},dw={suppressEvents:!0,isStart:!0,kill:!1},hc={suppressEvents:!0,kill:!1},hw={suppressEvents:!0},Dh={},rr=[],xd={},Sx,fi={},of={},n0=30,pc=[],Ph="",Fh=function(e){var t=e[0],n,s;if(ls(t)||an(t)||(e=[e]),!(n=(t._gsap||{}).harness)){for(s=pc.length;s--&&!pc[s].targetTest(t););n=pc[s]}for(s=e.length;s--;)e[s]&&(e[s]._gsap||(e[s]._gsap=new Xx(e[s],n)))||e.splice(s,1);return e},Xr=function(e){return e._gsap||Fh(Pi(e))[0]._gsap},yx=function(e,t,n){return(n=e[t])&&an(n)?e[t]():Eh(n)&&e.getAttribute&&e.getAttribute(t)||n},ti=function(e,t){return(e=e.split(",")).forEach(t)||e},cn=function(e){return Math.round(e*1e5)/1e5||0},tn=function(e){return Math.round(e*1e7)/1e7||0},zo=function(e,t){var n=t.charAt(0),s=parseFloat(t.substr(2));return e=parseFloat(e),n==="+"?e+s:n==="-"?e-s:n==="*"?e*s:e/s},pw=function(e,t){for(var n=t.length,s=0;e.indexOf(t[s])<0&&++s<n;);return s<n},Dc=function(){var e=rr.length,t=rr.slice(0),n,s;for(xd={},rr.length=0,n=0;n<e;n++)s=t[n],s&&s._lazy&&(s.render(s._lazy[0],s._lazy[1],!0)._lazy=0)},Lh=function(e){return!!(e._initted||e._startAt||e.add)},bx=function(e,t,n,s){rr.length&&!Tn&&Dc(),e.render(t,n,!!(Tn&&t<0&&Lh(e))),rr.length&&!Tn&&Dc()},Mx=function(e){var t=parseFloat(e);return(t||t===0)&&(e+"").match(uw).length<2?t:yn(e)?e.trim():e},Cx=function(e){return e},Ai=function(e,t){for(var n in t)n in e||(e[n]=t[n]);return e},mw=function(e){return function(t,n){for(var s in n)s in t||s==="duration"&&e||s==="ease"||(t[s]=n[s])}},Jo=function(e,t){for(var n in t)e[n]=t[n];return e},i0=function i(e,t){for(var n in t)n!=="__proto__"&&n!=="constructor"&&n!=="prototype"&&(e[n]=ls(t[n])?i(e[n]||(e[n]={}),t[n]):t[n]);return e},Pc=function(e,t){var n={},s;for(s in e)s in t||(n[s]=e[s]);return n},ka=function(e){var t=e.parent||nn,n=e.keyframes?mw(Bn(e.keyframes)):Ai;if(ei(e.inherit))for(;t;)n(e,t.vars.defaults),t=t.parent||t._dp;return e},gw=function(e,t){for(var n=e.length,s=n===t.length;s&&n--&&e[n]===t[n];);return n<0},Tx=function(e,t,n,s,r){var o=e[s],a;if(r)for(a=t[r];o&&o[r]>a;)o=o._prev;return o?(t._next=o._next,o._next=t):(t._next=e[n],e[n]=t),t._next?t._next._prev=t:e[s]=t,t._prev=o,t.parent=t._dp=e,t},eu=function(e,t,n,s){n===void 0&&(n="_first"),s===void 0&&(s="_last");var r=t._prev,o=t._next;r?r._next=o:e[n]===t&&(e[n]=o),o?o._prev=r:e[s]===t&&(e[s]=r),t._next=t._prev=t.parent=null},ur=function(e,t){e.parent&&(!t||e.parent.autoRemoveChildren)&&e.parent.remove&&e.parent.remove(e),e._act=0},qr=function(e,t){if(e&&(!t||t._end>e._dur||t._start<0))for(var n=e;n;)n._dirty=1,n=n.parent;return e},xw=function(e){for(var t=e.parent;t&&t.parent;)t._dirty=1,t.totalDuration(),t=t.parent;return e},_d=function(e,t,n,s){return e._startAt&&(Tn?e._startAt.revert(hc):e.vars.immediateRender&&!e.vars.autoRevert||e._startAt.render(t,!0,s))},_w=function i(e){return!e||e._ts&&i(e.parent)},s0=function(e){return e._repeat?ea(e._tTime,e=e.duration()+e._rDelay)*e:0},ea=function(e,t){var n=Math.floor(e=tn(e/t));return e&&n===e?n-1:n},Fc=function(e,t){return(e-t._start)*t._ts+(t._ts>=0?0:t._dirty?t.totalDuration():t._tDur)},tu=function(e){return e._end=tn(e._start+(e._tDur/Math.abs(e._ts||e._rts||kt)||0))},nu=function(e,t){var n=e._dp;return n&&n.smoothChildTiming&&e._ts&&(e._start=tn(n._time-(e._ts>0?t/e._ts:((e._dirty?e.totalDuration():e._tDur)-t)/-e._ts)),tu(e),n._dirty||qr(n,e)),e},Ex=function(e,t){var n;if((t._time||!t._dur&&t._initted||t._start<e._time&&(t._dur||!t.add))&&(n=Fc(e.rawTime(),t),(!t._dur||ml(0,t.totalDuration(),n)-t._tTime>kt)&&t.render(n,!0)),qr(e,t)._dp&&e._initted&&e._time>=e._dur&&e._ts){if(e._dur<e.duration())for(n=e;n._dp;)n.rawTime()>=0&&n.totalTime(n._tTime),n=n._dp;e._zTime=-kt}},ji=function(e,t,n,s){return t.parent&&ur(t),t._start=tn((Us(n)?n:n||e!==nn?bi(e,n,t):e._time)+t._delay),t._end=tn(t._start+(t.totalDuration()/Math.abs(t.timeScale())||0)),Tx(e,t,"_first","_last",e._sort?"_start":0),vd(t)||(e._recent=t),s||Ex(e,t),e._ts<0&&nu(e,e._tTime),e},wx=function(e,t){return(vi.ScrollTrigger||Ih("scrollTrigger",t))&&vi.ScrollTrigger.create(t,e)},Rx=function(e,t,n,s,r){if(Uh(e,t,r),!e._initted)return 1;if(!n&&e._pt&&!Tn&&(e._dur&&e.vars.lazy!==!1||!e._dur&&e.vars.lazy)&&Sx!==di.frame)return rr.push(e),e._lazy=[r,s],1},vw=function i(e){var t=e.parent;return t&&t._ts&&t._initted&&!t._lock&&(t.rawTime()<0||i(t))},vd=function(e){var t=e.data;return t==="isFromStart"||t==="isStart"},Aw=function(e,t,n,s){var r=e.ratio,o=t<0||!t&&(!e._start&&vw(e)&&!(!e._initted&&vd(e))||(e._ts<0||e._dp._ts<0)&&!vd(e))?0:1,a=e._rDelay,l=0,c,u,f;if(a&&e._repeat&&(l=ml(0,e._tDur,t),u=ea(l,a),e._yoyo&&u&1&&(o=1-o),u!==ea(e._tTime,a)&&(r=1-o,e.vars.repeatRefresh&&e._initted&&e.invalidate())),o!==r||Tn||s||e._zTime===kt||!t&&e._zTime){if(!e._initted&&Rx(e,t,s,n,l))return;for(f=e._zTime,e._zTime=t||(n?kt:0),n||(n=t&&!f),e.ratio=o,e._from&&(o=1-o),e._time=0,e._tTime=l,c=e._pt;c;)c.r(o,c.d),c=c._next;t<0&&_d(e,t,n,!0),e._onUpdate&&!n&&mi(e,"onUpdate"),l&&e._repeat&&!n&&e.parent&&mi(e,"onRepeat"),(t>=e._tDur||t<0)&&e.ratio===o&&(o&&ur(e,1),!n&&!Tn&&(mi(e,o?"onComplete":"onReverseComplete",!0),e._prom&&e._prom()))}else e._zTime||(e._zTime=t)},Sw=function(e,t,n){var s;if(n>t)for(s=e._first;s&&s._start<=n;){if(s.data==="isPause"&&s._start>t)return s;s=s._next}else for(s=e._last;s&&s._start>=n;){if(s.data==="isPause"&&s._start<t)return s;s=s._prev}},ta=function(e,t,n,s){var r=e._repeat,o=tn(t)||0,a=e._tTime/e._tDur;return a&&!s&&(e._time*=o/e._dur),e._dur=o,e._tDur=r?r<0?1e10:tn(o*(r+1)+e._rDelay*r):o,a>0&&!s&&nu(e,e._tTime=e._tDur*a),e.parent&&tu(e),n||qr(e.parent,e),e},r0=function(e){return e instanceof Wn?qr(e):ta(e,e._dur)},yw={_start:0,endTime:sl,totalDuration:sl},bi=function i(e,t,n){var s=e.labels,r=e._recent||yw,o=e.duration()>=Di?r.endTime(!1):e._dur,a,l,c;return yn(t)&&(isNaN(t)||t in s)?(l=t.charAt(0),c=t.substr(-1)==="%",a=t.indexOf("="),l==="<"||l===">"?(a>=0&&(t=t.replace(/=/,"")),(l==="<"?r._start:r.endTime(r._repeat>=0))+(parseFloat(t.substr(1))||0)*(c?(a<0?r:n).totalDuration()/100:1)):a<0?(t in s||(s[t]=o),s[t]):(l=parseFloat(t.charAt(a-1)+t.substr(a+1)),c&&n&&(l=l/100*(Bn(n)?n[0]:n).totalDuration()),a>1?i(e,t.substr(0,a-1),n)+l:o+l)):t==null?o:+t},Ha=function(e,t,n){var s=Us(t[1]),r=(s?2:1)+(e<2?0:1),o=t[r],a,l;if(s&&(o.duration=t[1]),o.parent=n,e){for(a=o,l=n;l&&!("immediateRender"in a);)a=l.vars.defaults||{},l=ei(l.vars.inherit)&&l.parent;o.immediateRender=ei(a.immediateRender),e<2?o.runBackwards=1:o.startAt=t[r-1]}return new pn(t[0],o,t[r+1])},pr=function(e,t){return e||e===0?t(e):t},ml=function(e,t,n){return n<e?e:n>t?t:n},Pn=function(e,t){return!yn(e)||!(t=fw.exec(e))?"":t[1]},bw=function(e,t,n){return pr(n,function(s){return ml(e,t,s)})},Ad=[].slice,Ix=function(e,t){return e&&ls(e)&&"length"in e&&(!t&&!e.length||e.length-1 in e&&ls(e[0]))&&!e.nodeType&&e!==Yi},Mw=function(e,t,n){return n===void 0&&(n=[]),e.forEach(function(s){var r;return yn(s)&&!t||Ix(s,1)?(r=n).push.apply(r,Pi(s)):n.push(s)})||n},Pi=function(e,t,n){return Zt&&!t&&Zt.selector?Zt.selector(e):yn(e)&&!n&&(gd||!na())?Ad.call((t||Rh).querySelectorAll(e),0):Bn(e)?Mw(e,n):Ix(e)?Ad.call(e,0):e?[e]:[]},Sd=function(e){return e=Pi(e)[0]||il("Invalid scope")||{},function(t){var n=e.current||e.nativeElement||e;return Pi(t,n.querySelectorAll?n:n===e?il("Invalid scope")||Rh.createElement("div"):e)}},Dx=function(e){return e.sort(function(){return .5-Math.random()})},Px=function(e){if(an(e))return e;var t=ls(e)?e:{each:e},n=Yr(t.ease),s=t.from||0,r=parseFloat(t.base)||0,o={},a=s>0&&s<1,l=isNaN(s)||a,c=t.axis,u=s,f=s;return yn(s)?u=f={center:.5,edges:.5,end:1}[s]||0:!a&&l&&(u=s[0],f=s[1]),function(d,h,x){var p=(x||t).length,g=o[p],m,_,A,v,S,y,M,E,b;if(!g){if(b=t.grid==="auto"?0:(t.grid||[1,Di])[1],!b){for(M=-Di;M<(M=x[b++].getBoundingClientRect().left)&&b<p;);b<p&&b--}for(g=o[p]=[],m=l?Math.min(b,p)*u-.5:s%b,_=b===Di?0:l?p*f/b-.5:s/b|0,M=0,E=Di,y=0;y<p;y++)A=y%b-m,v=_-(y/b|0),g[y]=S=c?Math.abs(c==="y"?v:A):px(A*A+v*v),S>M&&(M=S),S<E&&(E=S);s==="random"&&Dx(g),g.max=M-E,g.min=E,g.v=p=(parseFloat(t.amount)||parseFloat(t.each)*(b>p?p-1:c?c==="y"?p/b:b:Math.max(b,p/b))||0)*(s==="edges"?-1:1),g.b=p<0?r-p:r,g.u=Pn(t.amount||t.each)||0,n=n&&p<0?Vx(n):n}return p=(g[d]-g.min)/g.max||0,tn(g.b+(n?n(p):p)*g.v)+g.u}},yd=function(e){var t=Math.pow(10,((e+"").split(".")[1]||"").length);return function(n){var s=tn(Math.round(parseFloat(n)/e)*e*t);return(s-s%1)/t+(Us(n)?0:Pn(n))}},Fx=function(e,t){var n=Bn(e),s,r;return!n&&ls(e)&&(s=n=e.radius||Di,e.values?(e=Pi(e.values),(r=!Us(e[0]))&&(s*=s)):e=yd(e.increment)),pr(t,n?an(e)?function(o){return r=e(o),Math.abs(r-o)<=s?r:o}:function(o){for(var a=parseFloat(r?o.x:o),l=parseFloat(r?o.y:0),c=Di,u=0,f=e.length,d,h;f--;)r?(d=e[f].x-a,h=e[f].y-l,d=d*d+h*h):d=Math.abs(e[f]-a),d<c&&(c=d,u=f);return u=!s||c<=s?e[u]:o,r||u===o||Us(o)?u:u+Pn(o)}:yd(e))},Lx=function(e,t,n,s){return pr(Bn(e)?!t:n===!0?!!(n=0):!s,function(){return Bn(e)?e[~~(Math.random()*e.length)]:(n=n||1e-5)&&(s=n<1?Math.pow(10,(n+"").length-2):1)&&Math.floor(Math.round((e-n/2+Math.random()*(t-e+n*.99))/n)*n*s)/s})},Cw=function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(s){return t.reduce(function(r,o){return o(r)},s)}},Tw=function(e,t){return function(n){return e(parseFloat(n))+(t||Pn(n))}},Ew=function(e,t,n){return Ux(e,t,0,1,n)},Bx=function(e,t,n){return pr(n,function(s){return e[~~t(s)]})},ww=function i(e,t,n){var s=t-e;return Bn(e)?Bx(e,i(0,e.length),t):pr(n,function(r){return(s+(r-e)%s)%s+e})},Rw=function i(e,t,n){var s=t-e,r=s*2;return Bn(e)?Bx(e,i(0,e.length-1),t):pr(n,function(o){return o=(r+(o-e)%r)%r||0,e+(o>s?r-o:o)})},rl=function(e){return e.replace(lw,function(t){var n=t.indexOf("[")+1,s=t.substring(n||7,n?t.indexOf("]"):t.length-1).split(cw);return Lx(n?s:+s[0],n?0:+s[1],+s[2]||1e-5)})},Ux=function(e,t,n,s,r){var o=t-e,a=s-n;return pr(r,function(l){return n+((l-e)/o*a||0)})},Iw=function i(e,t,n,s){var r=isNaN(e+t)?0:function(h){return(1-h)*e+h*t};if(!r){var o=yn(e),a={},l,c,u,f,d;if(n===!0&&(s=1)&&(n=null),o)e={p:e},t={p:t};else if(Bn(e)&&!Bn(t)){for(u=[],f=e.length,d=f-2,c=1;c<f;c++)u.push(i(e[c-1],e[c]));f--,r=function(x){x*=f;var p=Math.min(d,~~x);return u[p](x-p)},n=t}else s||(e=Jo(Bn(e)?[]:{},e));if(!u){for(l in t)Bh.call(a,e,l,"get",t[l]);r=function(x){return zh(x,a)||(o?e.p:e)}}}return pr(n,r)},o0=function(e,t,n){var s=e.labels,r=Di,o,a,l;for(o in s)a=s[o]-t,a<0==!!n&&a&&r>(a=Math.abs(a))&&(l=o,r=a);return l},mi=function(e,t,n){var s=e.vars,r=s[t],o=Zt,a=e._ctx,l,c,u;if(r)return l=s[t+"Params"],c=s.callbackScope||e,n&&rr.length&&Dc(),a&&(Zt=a),u=l?r.apply(c,l):r.call(c),Zt=o,u},ba=function(e){return ur(e),e.scrollTrigger&&e.scrollTrigger.kill(!!Tn),e.progress()<1&&mi(e,"onInterrupt"),e},wo,Ox=[],Nx=function(e){if(e)if(e=!e.name&&e.default||e,wh()||e.headless){var t=e.name,n=an(e),s=t&&!n&&e.init?function(){this._props=[]}:e,r={init:sl,render:zh,add:Bh,kill:qw,modifier:Xw,rawVars:0},o={targetTest:0,get:0,getSetter:Nh,aliases:{},register:0};if(na(),e!==s){if(fi[t])return;Ai(s,Ai(Pc(e,r),o)),Jo(s.prototype,Jo(r,Pc(e,o))),fi[s.prop=t]=s,e.targetTest&&(pc.push(s),Dh[t]=1),t=(t==="css"?"CSS":t.charAt(0).toUpperCase()+t.substr(1))+"Plugin"}Ax(t,s),e.register&&e.register(si,s,ni)}else Ox.push(e)},zt=255,Ma={aqua:[0,zt,zt],lime:[0,zt,0],silver:[192,192,192],black:[0,0,0],maroon:[128,0,0],teal:[0,128,128],blue:[0,0,zt],navy:[0,0,128],white:[zt,zt,zt],olive:[128,128,0],yellow:[zt,zt,0],orange:[zt,165,0],gray:[128,128,128],purple:[128,0,128],green:[0,128,0],red:[zt,0,0],pink:[zt,192,203],cyan:[0,zt,zt],transparent:[zt,zt,zt,0]},af=function(e,t,n){return e+=e<0?1:e>1?-1:0,(e*6<1?t+(n-t)*e*6:e<.5?n:e*3<2?t+(n-t)*(2/3-e)*6:t)*zt+.5|0},zx=function(e,t,n){var s=e?Us(e)?[e>>16,e>>8&zt,e&zt]:0:Ma.black,r,o,a,l,c,u,f,d,h,x;if(!s){if(e.substr(-1)===","&&(e=e.substr(0,e.length-1)),Ma[e])s=Ma[e];else if(e.charAt(0)==="#"){if(e.length<6&&(r=e.charAt(1),o=e.charAt(2),a=e.charAt(3),e="#"+r+r+o+o+a+a+(e.length===5?e.charAt(4)+e.charAt(4):"")),e.length===9)return s=parseInt(e.substr(1,6),16),[s>>16,s>>8&zt,s&zt,parseInt(e.substr(7),16)/255];e=parseInt(e.substr(1),16),s=[e>>16,e>>8&zt,e&zt]}else if(e.substr(0,3)==="hsl"){if(s=x=e.match(t0),!t)l=+s[0]%360/360,c=+s[1]/100,u=+s[2]/100,o=u<=.5?u*(c+1):u+c-u*c,r=u*2-o,s.length>3&&(s[3]*=1),s[0]=af(l+1/3,r,o),s[1]=af(l,r,o),s[2]=af(l-1/3,r,o);else if(~e.indexOf("="))return s=e.match(gx),n&&s.length<4&&(s[3]=1),s}else s=e.match(t0)||Ma.transparent;s=s.map(Number)}return t&&!x&&(r=s[0]/zt,o=s[1]/zt,a=s[2]/zt,f=Math.max(r,o,a),d=Math.min(r,o,a),u=(f+d)/2,f===d?l=c=0:(h=f-d,c=u>.5?h/(2-f-d):h/(f+d),l=f===r?(o-a)/h+(o<a?6:0):f===o?(a-r)/h+2:(r-o)/h+4,l*=60),s[0]=~~(l+.5),s[1]=~~(c*100+.5),s[2]=~~(u*100+.5)),n&&s.length<4&&(s[3]=1),s},kx=function(e){var t=[],n=[],s=-1;return e.split(or).forEach(function(r){var o=r.match(Eo)||[];t.push.apply(t,o),n.push(s+=o.length+1)}),t.c=n,t},a0=function(e,t,n){var s="",r=(e+s).match(or),o=t?"hsla(":"rgba(",a=0,l,c,u,f;if(!r)return e;if(r=r.map(function(d){return(d=zx(d,t,1))&&o+(t?d[0]+","+d[1]+"%,"+d[2]+"%,"+d[3]:d.join(","))+")"}),n&&(u=kx(e),l=n.c,l.join(s)!==u.c.join(s)))for(c=e.replace(or,"1").split(Eo),f=c.length-1;a<f;a++)s+=c[a]+(~l.indexOf(a)?r.shift()||o+"0,0,0,0)":(u.length?u:r.length?r:n).shift());if(!c)for(c=e.split(or),f=c.length-1;a<f;a++)s+=c[a]+r[a];return s+c[f]},or=(function(){var i="(?:\\b(?:(?:rgb|rgba|hsl|hsla)\\(.+?\\))|\\B#(?:[0-9a-f]{3,4}){1,2}\\b",e;for(e in Ma)i+="|"+e+"\\b";return new RegExp(i+")","gi")})(),Dw=/hsl[a]?\(/,Hx=function(e){var t=e.join(" "),n;if(or.lastIndex=0,or.test(t))return n=Dw.test(t),e[1]=a0(e[1],n),e[0]=a0(e[0],n,kx(e[1])),!0},ol,di=(function(){var i=Date.now,e=500,t=33,n=i(),s=n,r=1e3/240,o=r,a=[],l,c,u,f,d,h,x=function p(g){var m=i()-s,_=g===!0,A,v,S,y;if((m>e||m<0)&&(n+=m-t),s+=m,S=s-n,A=S-o,(A>0||_)&&(y=++f.frame,d=S-f.time*1e3,f.time=S=S/1e3,o+=A+(A>=r?4:r-A),v=1),_||(l=c(p)),v)for(h=0;h<a.length;h++)a[h](S,d,y,g)};return f={time:0,frame:0,tick:function(){x(!0)},deltaRatio:function(g){return d/(1e3/(g||60))},wake:function(){_x&&(!gd&&wh()&&(Yi=gd=window,Rh=Yi.document||{},vi.gsap=si,(Yi.gsapVersions||(Yi.gsapVersions=[])).push(si.version),vx(Ic||Yi.GreenSockGlobals||!Yi.gsap&&Yi||{}),Ox.forEach(Nx)),u=typeof requestAnimationFrame<"u"&&requestAnimationFrame,l&&f.sleep(),c=u||function(g){return setTimeout(g,o-f.time*1e3+1|0)},ol=1,x(2))},sleep:function(){(u?cancelAnimationFrame:clearTimeout)(l),ol=0,c=sl},lagSmoothing:function(g,m){e=g||1/0,t=Math.min(m||33,e)},fps:function(g){r=1e3/(g||240),o=f.time*1e3+r},add:function(g,m,_){var A=m?function(v,S,y,M){g(v,S,y,M),f.remove(A)}:g;return f.remove(g),a[_?"unshift":"push"](A),na(),A},remove:function(g,m){~(m=a.indexOf(g))&&a.splice(m,1)&&h>=m&&h--},_listeners:a},f})(),na=function(){return!ol&&di.wake()},At={},Pw=/^[\d.\-M][\d.\-,\s]/,Fw=/["']/g,Lw=function(e){for(var t={},n=e.substr(1,e.length-3).split(":"),s=n[0],r=1,o=n.length,a,l,c;r<o;r++)l=n[r],a=r!==o-1?l.lastIndexOf(","):l.length,c=l.substr(0,a),t[s]=isNaN(c)?c.replace(Fw,"").trim():+c,s=l.substr(a+1).trim();return t},Bw=function(e){var t=e.indexOf("(")+1,n=e.indexOf(")"),s=e.indexOf("(",t);return e.substring(t,~s&&s<n?e.indexOf(")",n+1):n)},Uw=function(e){var t=(e+"").split("("),n=At[t[0]];return n&&t.length>1&&n.config?n.config.apply(null,~e.indexOf("{")?[Lw(t[1])]:Bw(e).split(",").map(Mx)):At._CE&&Pw.test(e)?At._CE("",e):n},Vx=function(e){return function(t){return 1-e(1-t)}},Gx=function i(e,t){for(var n=e._first,s;n;)n instanceof Wn?i(n,t):n.vars.yoyoEase&&(!n._yoyo||!n._repeat)&&n._yoyo!==t&&(n.timeline?i(n.timeline,t):(s=n._ease,n._ease=n._yEase,n._yEase=s,n._yoyo=t)),n=n._next},Yr=function(e,t){return e&&(an(e)?e:At[e]||Uw(e))||t},Zr=function(e,t,n,s){n===void 0&&(n=function(l){return 1-t(1-l)}),s===void 0&&(s=function(l){return l<.5?t(l*2)/2:1-t((1-l)*2)/2});var r={easeIn:t,easeOut:n,easeInOut:s},o;return ti(e,function(a){At[a]=vi[a]=r,At[o=a.toLowerCase()]=n;for(var l in r)At[o+(l==="easeIn"?".in":l==="easeOut"?".out":".inOut")]=At[a+"."+l]=r[l]}),r},Wx=function(e){return function(t){return t<.5?(1-e(1-t*2))/2:.5+e((t-.5)*2)/2}},lf=function i(e,t,n){var s=t>=1?t:1,r=(n||(e?.3:.45))/(t<1?t:1),o=r/md*(Math.asin(1/s)||0),a=function(u){return u===1?1:s*Math.pow(2,-10*u)*aw((u-o)*r)+1},l=e==="out"?a:e==="in"?function(c){return 1-a(1-c)}:Wx(a);return r=md/r,l.config=function(c,u){return i(e,c,u)},l},cf=function i(e,t){t===void 0&&(t=1.70158);var n=function(o){return o?--o*o*((t+1)*o+t)+1:0},s=e==="out"?n:e==="in"?function(r){return 1-n(1-r)}:Wx(n);return s.config=function(r){return i(e,r)},s};ti("Linear,Quad,Cubic,Quart,Quint,Strong",function(i,e){var t=e<5?e+1:e;Zr(i+",Power"+(t-1),e?function(n){return Math.pow(n,t)}:function(n){return n},function(n){return 1-Math.pow(1-n,t)},function(n){return n<.5?Math.pow(n*2,t)/2:1-Math.pow((1-n)*2,t)/2})});At.Linear.easeNone=At.none=At.Linear.easeIn;Zr("Elastic",lf("in"),lf("out"),lf());(function(i,e){var t=1/e,n=2*t,s=2.5*t,r=function(a){return a<t?i*a*a:a<n?i*Math.pow(a-1.5/e,2)+.75:a<s?i*(a-=2.25/e)*a+.9375:i*Math.pow(a-2.625/e,2)+.984375};Zr("Bounce",function(o){return 1-r(1-o)},r)})(7.5625,2.75);Zr("Expo",function(i){return Math.pow(2,10*(i-1))*i+i*i*i*i*i*i*(1-i)});Zr("Circ",function(i){return-(px(1-i*i)-1)});Zr("Sine",function(i){return i===1?1:-ow(i*sw)+1});Zr("Back",cf("in"),cf("out"),cf());At.SteppedEase=At.steps=vi.SteppedEase={config:function(e,t){e===void 0&&(e=1);var n=1/e,s=e+(t?0:1),r=t?1:0,o=1-kt;return function(a){return((s*ml(0,o,a)|0)+r)*n}}};Zo.ease=At["quad.out"];ti("onComplete,onUpdate,onStart,onRepeat,onReverseComplete,onInterrupt",function(i){return Ph+=i+","+i+"Params,"});var Xx=function(e,t){this.id=rw++,e._gsap=this,this.target=e,this.harness=t,this.get=t?t.get:yx,this.set=t?t.getSetter:Nh},al=(function(){function i(t){this.vars=t,this._delay=+t.delay||0,(this._repeat=t.repeat===1/0?-2:t.repeat||0)&&(this._rDelay=t.repeatDelay||0,this._yoyo=!!t.yoyo||!!t.yoyoEase),this._ts=1,ta(this,+t.duration,1,1),this.data=t.data,Zt&&(this._ctx=Zt,Zt.data.push(this)),ol||di.wake()}var e=i.prototype;return e.delay=function(n){return n||n===0?(this.parent&&this.parent.smoothChildTiming&&this.startTime(this._start+n-this._delay),this._delay=n,this):this._delay},e.duration=function(n){return arguments.length?this.totalDuration(this._repeat>0?n+(n+this._rDelay)*this._repeat:n):this.totalDuration()&&this._dur},e.totalDuration=function(n){return arguments.length?(this._dirty=0,ta(this,this._repeat<0?n:(n-this._repeat*this._rDelay)/(this._repeat+1))):this._tDur},e.totalTime=function(n,s){if(na(),!arguments.length)return this._tTime;var r=this._dp;if(r&&r.smoothChildTiming&&this._ts){for(nu(this,n),!r._dp||r.parent||Ex(r,this);r&&r.parent;)r.parent._time!==r._start+(r._ts>=0?r._tTime/r._ts:(r.totalDuration()-r._tTime)/-r._ts)&&r.totalTime(r._tTime,!0),r=r.parent;!this.parent&&this._dp.autoRemoveChildren&&(this._ts>0&&n<this._tDur||this._ts<0&&n>0||!this._tDur&&!n)&&ji(this._dp,this,this._start-this._delay)}return(this._tTime!==n||!this._dur&&!s||this._initted&&Math.abs(this._zTime)===kt||!this._initted&&this._dur&&n||!n&&!this._initted&&(this.add||this._ptLookup))&&(this._ts||(this._pTime=n),bx(this,n,s)),this},e.time=function(n,s){return arguments.length?this.totalTime(Math.min(this.totalDuration(),n+s0(this))%(this._dur+this._rDelay)||(n?this._dur:0),s):this._time},e.totalProgress=function(n,s){return arguments.length?this.totalTime(this.totalDuration()*n,s):this.totalDuration()?Math.min(1,this._tTime/this._tDur):this.rawTime()>=0&&this._initted?1:0},e.progress=function(n,s){return arguments.length?this.totalTime(this.duration()*(this._yoyo&&!(this.iteration()&1)?1-n:n)+s0(this),s):this.duration()?Math.min(1,this._time/this._dur):this.rawTime()>0?1:0},e.iteration=function(n,s){var r=this.duration()+this._rDelay;return arguments.length?this.totalTime(this._time+(n-1)*r,s):this._repeat?ea(this._tTime,r)+1:1},e.timeScale=function(n,s){if(!arguments.length)return this._rts===-kt?0:this._rts;if(this._rts===n)return this;var r=this.parent&&this._ts?Fc(this.parent._time,this):this._tTime;return this._rts=+n||0,this._ts=this._ps||n===-kt?0:this._rts,this.totalTime(ml(-Math.abs(this._delay),this.totalDuration(),r),s!==!1),tu(this),xw(this)},e.paused=function(n){return arguments.length?(this._ps!==n&&(this._ps=n,n?(this._pTime=this._tTime||Math.max(-this._delay,this.rawTime()),this._ts=this._act=0):(na(),this._ts=this._rts,this.totalTime(this.parent&&!this.parent.smoothChildTiming?this.rawTime():this._tTime||this._pTime,this.progress()===1&&Math.abs(this._zTime)!==kt&&(this._tTime-=kt)))),this):this._ps},e.startTime=function(n){if(arguments.length){this._start=tn(n);var s=this.parent||this._dp;return s&&(s._sort||!this.parent)&&ji(s,this,this._start-this._delay),this}return this._start},e.endTime=function(n){return this._start+(ei(n)?this.totalDuration():this.duration())/Math.abs(this._ts||1)},e.rawTime=function(n){var s=this.parent||this._dp;return s?n&&(!this._ts||this._repeat&&this._time&&this.totalProgress()<1)?this._tTime%(this._dur+this._rDelay):this._ts?Fc(s.rawTime(n),this):this._tTime:this._tTime},e.revert=function(n){n===void 0&&(n=hw);var s=Tn;return Tn=n,Lh(this)&&(this.timeline&&this.timeline.revert(n),this.totalTime(-.01,n.suppressEvents)),this.data!=="nested"&&n.kill!==!1&&this.kill(),Tn=s,this},e.globalTime=function(n){for(var s=this,r=arguments.length?n:s.rawTime();s;)r=s._start+r/(Math.abs(s._ts)||1),s=s._dp;return!this.parent&&this._sat?this._sat.globalTime(n):r},e.repeat=function(n){return arguments.length?(this._repeat=n===1/0?-2:n,r0(this)):this._repeat===-2?1/0:this._repeat},e.repeatDelay=function(n){if(arguments.length){var s=this._time;return this._rDelay=n,r0(this),s?this.time(s):this}return this._rDelay},e.yoyo=function(n){return arguments.length?(this._yoyo=n,this):this._yoyo},e.seek=function(n,s){return this.totalTime(bi(this,n),ei(s))},e.restart=function(n,s){return this.play().totalTime(n?-this._delay:0,ei(s)),this._dur||(this._zTime=-kt),this},e.play=function(n,s){return n!=null&&this.seek(n,s),this.reversed(!1).paused(!1)},e.reverse=function(n,s){return n!=null&&this.seek(n||this.totalDuration(),s),this.reversed(!0).paused(!1)},e.pause=function(n,s){return n!=null&&this.seek(n,s),this.paused(!0)},e.resume=function(){return this.paused(!1)},e.reversed=function(n){return arguments.length?(!!n!==this.reversed()&&this.timeScale(-this._rts||(n?-kt:0)),this):this._rts<0},e.invalidate=function(){return this._initted=this._act=0,this._zTime=-kt,this},e.isActive=function(){var n=this.parent||this._dp,s=this._start,r;return!!(!n||this._ts&&this._initted&&n.isActive()&&(r=n.rawTime(!0))>=s&&r<this.endTime(!0)-kt)},e.eventCallback=function(n,s,r){var o=this.vars;return arguments.length>1?(s?(o[n]=s,r&&(o[n+"Params"]=r),n==="onUpdate"&&(this._onUpdate=s)):delete o[n],this):o[n]},e.then=function(n){var s=this,r=s._prom;return new Promise(function(o){var a=an(n)?n:Cx,l=function(){var u=s.then;s.then=null,r&&r(),an(a)&&(a=a(s))&&(a.then||a===s)&&(s.then=u),o(a),s.then=u};s._initted&&s.totalProgress()===1&&s._ts>=0||!s._tTime&&s._ts<0?l():s._prom=l})},e.kill=function(){ba(this)},i})();Ai(al.prototype,{_time:0,_start:0,_end:0,_tTime:0,_tDur:0,_dirty:0,_repeat:0,_yoyo:!1,parent:null,_initted:!1,_rDelay:0,_ts:1,_dp:0,ratio:0,_zTime:-kt,_prom:0,_ps:!1,_rts:1});var Wn=(function(i){hx(e,i);function e(n,s){var r;return n===void 0&&(n={}),r=i.call(this,n)||this,r.labels={},r.smoothChildTiming=!!n.smoothChildTiming,r.autoRemoveChildren=!!n.autoRemoveChildren,r._sort=ei(n.sortChildren),nn&&ji(n.parent||nn,Ms(r),s),n.reversed&&r.reverse(),n.paused&&r.paused(!0),n.scrollTrigger&&wx(Ms(r),n.scrollTrigger),r}var t=e.prototype;return t.to=function(s,r,o){return Ha(0,arguments,this),this},t.from=function(s,r,o){return Ha(1,arguments,this),this},t.fromTo=function(s,r,o,a){return Ha(2,arguments,this),this},t.set=function(s,r,o){return r.duration=0,r.parent=this,ka(r).repeatDelay||(r.repeat=0),r.immediateRender=!!r.immediateRender,new pn(s,r,bi(this,o),1),this},t.call=function(s,r,o){return ji(this,pn.delayedCall(0,s,r),o)},t.staggerTo=function(s,r,o,a,l,c,u){return o.duration=r,o.stagger=o.stagger||a,o.onComplete=c,o.onCompleteParams=u,o.parent=this,new pn(s,o,bi(this,l)),this},t.staggerFrom=function(s,r,o,a,l,c,u){return o.runBackwards=1,ka(o).immediateRender=ei(o.immediateRender),this.staggerTo(s,r,o,a,l,c,u)},t.staggerFromTo=function(s,r,o,a,l,c,u,f){return a.startAt=o,ka(a).immediateRender=ei(a.immediateRender),this.staggerTo(s,r,a,l,c,u,f)},t.render=function(s,r,o){var a=this._time,l=this._dirty?this.totalDuration():this._tDur,c=this._dur,u=s<=0?0:tn(s),f=this._zTime<0!=s<0&&(this._initted||!c),d,h,x,p,g,m,_,A,v,S,y,M;if(this!==nn&&u>l&&s>=0&&(u=l),u!==this._tTime||o||f){if(a!==this._time&&c&&(u+=this._time-a,s+=this._time-a),d=u,v=this._start,A=this._ts,m=!A,f&&(c||(a=this._zTime),(s||!r)&&(this._zTime=s)),this._repeat){if(y=this._yoyo,g=c+this._rDelay,this._repeat<-1&&s<0)return this.totalTime(g*100+s,r,o);if(d=tn(u%g),u===l?(p=this._repeat,d=c):(S=tn(u/g),p=~~S,p&&p===S&&(d=c,p--),d>c&&(d=c)),S=ea(this._tTime,g),!a&&this._tTime&&S!==p&&this._tTime-S*g-this._dur<=0&&(S=p),y&&p&1&&(d=c-d,M=1),p!==S&&!this._lock){var E=y&&S&1,b=E===(y&&p&1);if(p<S&&(E=!E),a=E?0:u%c?c:u,this._lock=1,this.render(a||(M?0:tn(p*g)),r,!c)._lock=0,this._tTime=u,!r&&this.parent&&mi(this,"onRepeat"),this.vars.repeatRefresh&&!M&&(this.invalidate()._lock=1,S=p),a&&a!==this._time||m!==!this._ts||this.vars.onRepeat&&!this.parent&&!this._act)return this;if(c=this._dur,l=this._tDur,b&&(this._lock=2,a=E?c:-1e-4,this.render(a,!0),this.vars.repeatRefresh&&!M&&this.invalidate()),this._lock=0,!this._ts&&!m)return this;Gx(this,M)}}if(this._hasPause&&!this._forcing&&this._lock<2&&(_=Sw(this,tn(a),tn(d)),_&&(u-=d-(d=_._start))),this._tTime=u,this._time=d,this._act=!A,this._initted||(this._onUpdate=this.vars.onUpdate,this._initted=1,this._zTime=s,a=0),!a&&u&&c&&!r&&!S&&(mi(this,"onStart"),this._tTime!==u))return this;if(d>=a&&s>=0)for(h=this._first;h;){if(x=h._next,(h._act||d>=h._start)&&h._ts&&_!==h){if(h.parent!==this)return this.render(s,r,o);if(h.render(h._ts>0?(d-h._start)*h._ts:(h._dirty?h.totalDuration():h._tDur)+(d-h._start)*h._ts,r,o),d!==this._time||!this._ts&&!m){_=0,x&&(u+=this._zTime=-kt);break}}h=x}else{h=this._last;for(var C=s<0?s:d;h;){if(x=h._prev,(h._act||C<=h._end)&&h._ts&&_!==h){if(h.parent!==this)return this.render(s,r,o);if(h.render(h._ts>0?(C-h._start)*h._ts:(h._dirty?h.totalDuration():h._tDur)+(C-h._start)*h._ts,r,o||Tn&&Lh(h)),d!==this._time||!this._ts&&!m){_=0,x&&(u+=this._zTime=C?-kt:kt);break}}h=x}}if(_&&!r&&(this.pause(),_.render(d>=a?0:-kt)._zTime=d>=a?1:-1,this._ts))return this._start=v,tu(this),this.render(s,r,o);this._onUpdate&&!r&&mi(this,"onUpdate",!0),(u===l&&this._tTime>=this.totalDuration()||!u&&a)&&(v===this._start||Math.abs(A)!==Math.abs(this._ts))&&(this._lock||((s||!c)&&(u===l&&this._ts>0||!u&&this._ts<0)&&ur(this,1),!r&&!(s<0&&!a)&&(u||a||!l)&&(mi(this,u===l&&s>=0?"onComplete":"onReverseComplete",!0),this._prom&&!(u<l&&this.timeScale()>0)&&this._prom())))}return this},t.add=function(s,r){var o=this;if(Us(r)||(r=bi(this,r,s)),!(s instanceof al)){if(Bn(s))return s.forEach(function(a){return o.add(a,r)}),this;if(yn(s))return this.addLabel(s,r);if(an(s))s=pn.delayedCall(0,s);else return this}return this!==s?ji(this,s,r):this},t.getChildren=function(s,r,o,a){s===void 0&&(s=!0),r===void 0&&(r=!0),o===void 0&&(o=!0),a===void 0&&(a=-Di);for(var l=[],c=this._first;c;)c._start>=a&&(c instanceof pn?r&&l.push(c):(o&&l.push(c),s&&l.push.apply(l,c.getChildren(!0,r,o)))),c=c._next;return l},t.getById=function(s){for(var r=this.getChildren(1,1,1),o=r.length;o--;)if(r[o].vars.id===s)return r[o]},t.remove=function(s){return yn(s)?this.removeLabel(s):an(s)?this.killTweensOf(s):(s.parent===this&&eu(this,s),s===this._recent&&(this._recent=this._last),qr(this))},t.totalTime=function(s,r){return arguments.length?(this._forcing=1,!this._dp&&this._ts&&(this._start=tn(di.time-(this._ts>0?s/this._ts:(this.totalDuration()-s)/-this._ts))),i.prototype.totalTime.call(this,s,r),this._forcing=0,this):this._tTime},t.addLabel=function(s,r){return this.labels[s]=bi(this,r),this},t.removeLabel=function(s){return delete this.labels[s],this},t.addPause=function(s,r,o){var a=pn.delayedCall(0,r||sl,o);return a.data="isPause",this._hasPause=1,ji(this,a,bi(this,s))},t.removePause=function(s){var r=this._first;for(s=bi(this,s);r;)r._start===s&&r.data==="isPause"&&ur(r),r=r._next},t.killTweensOf=function(s,r,o){for(var a=this.getTweensOf(s,o),l=a.length;l--;)Js!==a[l]&&a[l].kill(s,r);return this},t.getTweensOf=function(s,r){for(var o=[],a=Pi(s),l=this._first,c=Us(r),u;l;)l instanceof pn?pw(l._targets,a)&&(c?(!Js||l._initted&&l._ts)&&l.globalTime(0)<=r&&l.globalTime(l.totalDuration())>r:!r||l.isActive())&&o.push(l):(u=l.getTweensOf(a,r)).length&&o.push.apply(o,u),l=l._next;return o},t.tweenTo=function(s,r){r=r||{};var o=this,a=bi(o,s),l=r,c=l.startAt,u=l.onStart,f=l.onStartParams,d=l.immediateRender,h,x=pn.to(o,Ai({ease:r.ease||"none",lazy:!1,immediateRender:!1,time:a,overwrite:"auto",duration:r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale())||kt,onStart:function(){if(o.pause(),!h){var g=r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale());x._dur!==g&&ta(x,g,0,1).render(x._time,!0,!0),h=1}u&&u.apply(x,f||[])}},r));return d?x.render(0):x},t.tweenFromTo=function(s,r,o){return this.tweenTo(r,Ai({startAt:{time:bi(this,s)}},o))},t.recent=function(){return this._recent},t.nextLabel=function(s){return s===void 0&&(s=this._time),o0(this,bi(this,s))},t.previousLabel=function(s){return s===void 0&&(s=this._time),o0(this,bi(this,s),1)},t.currentLabel=function(s){return arguments.length?this.seek(s,!0):this.previousLabel(this._time+kt)},t.shiftChildren=function(s,r,o){o===void 0&&(o=0);var a=this._first,l=this.labels,c;for(s=tn(s);a;)a._start>=o&&(a._start+=s,a._end+=s),a=a._next;if(r)for(c in l)l[c]>=o&&(l[c]+=s);return qr(this)},t.invalidate=function(s){var r=this._first;for(this._lock=0;r;)r.invalidate(s),r=r._next;return i.prototype.invalidate.call(this,s)},t.clear=function(s){s===void 0&&(s=!0);for(var r=this._first,o;r;)o=r._next,this.remove(r),r=o;return this._dp&&(this._time=this._tTime=this._pTime=0),s&&(this.labels={}),qr(this)},t.totalDuration=function(s){var r=0,o=this,a=o._last,l=Di,c,u,f;if(arguments.length)return o.timeScale((o._repeat<0?o.duration():o.totalDuration())/(o.reversed()?-s:s));if(o._dirty){for(f=o.parent;a;)c=a._prev,a._dirty&&a.totalDuration(),u=a._start,u>l&&o._sort&&a._ts&&!o._lock?(o._lock=1,ji(o,a,u-a._delay,1)._lock=0):l=u,u<0&&a._ts&&(r-=u,(!f&&!o._dp||f&&f.smoothChildTiming)&&(o._start+=tn(u/o._ts),o._time-=u,o._tTime-=u),o.shiftChildren(-u,!1,-1/0),l=0),a._end>r&&a._ts&&(r=a._end),a=c;ta(o,o===nn&&o._time>r?o._time:r,1,1),o._dirty=0}return o._tDur},e.updateRoot=function(s){if(nn._ts&&(bx(nn,Fc(s,nn)),Sx=di.frame),di.frame>=n0){n0+=_i.autoSleep||120;var r=nn._first;if((!r||!r._ts)&&_i.autoSleep&&di._listeners.length<2){for(;r&&!r._ts;)r=r._next;r||di.sleep()}}},e})(al);Ai(Wn.prototype,{_lock:0,_hasPause:0,_forcing:0});var Ow=function(e,t,n,s,r,o,a){var l=new ni(this._pt,e,t,0,1,$x,null,r),c=0,u=0,f,d,h,x,p,g,m,_;for(l.b=n,l.e=s,n+="",s+="",(m=~s.indexOf("random("))&&(s=rl(s)),o&&(_=[n,s],o(_,e,t),n=_[0],s=_[1]),d=n.match(rf)||[];f=rf.exec(s);)x=f[0],p=s.substring(c,f.index),h?h=(h+1)%5:p.substr(-5)==="rgba("&&(h=1),x!==d[u++]&&(g=parseFloat(d[u-1])||0,l._pt={_next:l._pt,p:p||u===1?p:",",s:g,c:x.charAt(1)==="="?zo(g,x)-g:parseFloat(x)-g,m:h&&h<4?Math.round:0},c=rf.lastIndex);return l.c=c<s.length?s.substring(c,s.length):"",l.fp=a,(xx.test(s)||m)&&(l.e=0),this._pt=l,l},Bh=function(e,t,n,s,r,o,a,l,c,u){an(s)&&(s=s(r||0,e,o));var f=e[t],d=n!=="get"?n:an(f)?c?e[t.indexOf("set")||!an(e["get"+t.substr(3)])?t:"get"+t.substr(3)](c):e[t]():f,h=an(f)?c?Vw:Kx:Oh,x;if(yn(s)&&(~s.indexOf("random(")&&(s=rl(s)),s.charAt(1)==="="&&(x=zo(d,s)+(Pn(d)||0),(x||x===0)&&(s=x))),!u||d!==s||bd)return!isNaN(d*s)&&s!==""?(x=new ni(this._pt,e,t,+d||0,s-(d||0),typeof f=="boolean"?Ww:jx,0,h),c&&(x.fp=c),a&&x.modifier(a,this,e),this._pt=x):(!f&&!(t in e)&&Ih(t,s),Ow.call(this,e,t,d,s,h,l||_i.stringFilter,c))},Nw=function(e,t,n,s,r){if(an(e)&&(e=Va(e,r,t,n,s)),!ls(e)||e.style&&e.nodeType||Bn(e)||mx(e))return yn(e)?Va(e,r,t,n,s):e;var o={},a;for(a in e)o[a]=Va(e[a],r,t,n,s);return o},qx=function(e,t,n,s,r,o){var a,l,c,u;if(fi[e]&&(a=new fi[e]).init(r,a.rawVars?t[e]:Nw(t[e],s,r,o,n),n,s,o)!==!1&&(n._pt=l=new ni(n._pt,r,e,0,1,a.render,a,0,a.priority),n!==wo))for(c=n._ptLookup[n._targets.indexOf(r)],u=a._props.length;u--;)c[a._props[u]]=l;return a},Js,bd,Uh=function i(e,t,n){var s=e.vars,r=s.ease,o=s.startAt,a=s.immediateRender,l=s.lazy,c=s.onUpdate,u=s.runBackwards,f=s.yoyoEase,d=s.keyframes,h=s.autoRevert,x=e._dur,p=e._startAt,g=e._targets,m=e.parent,_=m&&m.data==="nested"?m.vars.targets:g,A=e._overwrite==="auto"&&!Th,v=e.timeline,S,y,M,E,b,C,D,F,O,z,V,H,q;if(v&&(!d||!r)&&(r="none"),e._ease=Yr(r,Zo.ease),e._yEase=f?Vx(Yr(f===!0?r:f,Zo.ease)):0,f&&e._yoyo&&!e._repeat&&(f=e._yEase,e._yEase=e._ease,e._ease=f),e._from=!v&&!!s.runBackwards,!v||d&&!s.stagger){if(F=g[0]?Xr(g[0]).harness:0,H=F&&s[F.prop],S=Pc(s,Dh),p&&(p._zTime<0&&p.progress(1),t<0&&u&&a&&!h?p.render(-1,!0):p.revert(u&&x?hc:dw),p._lazy=0),o){if(ur(e._startAt=pn.set(g,Ai({data:"isStart",overwrite:!1,parent:m,immediateRender:!0,lazy:!p&&ei(l),startAt:null,delay:0,onUpdate:c&&function(){return mi(e,"onUpdate")},stagger:0},o))),e._startAt._dp=0,e._startAt._sat=e,t<0&&(Tn||!a&&!h)&&e._startAt.revert(hc),a&&x&&t<=0&&n<=0){t&&(e._zTime=t);return}}else if(u&&x&&!p){if(t&&(a=!1),M=Ai({overwrite:!1,data:"isFromStart",lazy:a&&!p&&ei(l),immediateRender:a,stagger:0,parent:m},S),H&&(M[F.prop]=H),ur(e._startAt=pn.set(g,M)),e._startAt._dp=0,e._startAt._sat=e,t<0&&(Tn?e._startAt.revert(hc):e._startAt.render(-1,!0)),e._zTime=t,!a)i(e._startAt,kt,kt);else if(!t)return}for(e._pt=e._ptCache=0,l=x&&ei(l)||l&&!x,y=0;y<g.length;y++){if(b=g[y],D=b._gsap||Fh(g)[y]._gsap,e._ptLookup[y]=z={},xd[D.id]&&rr.length&&Dc(),V=_===g?y:_.indexOf(b),F&&(O=new F).init(b,H||S,e,V,_)!==!1&&(e._pt=E=new ni(e._pt,b,O.name,0,1,O.render,O,0,O.priority),O._props.forEach(function(G){z[G]=E}),O.priority&&(C=1)),!F||H)for(M in S)fi[M]&&(O=qx(M,S,e,V,b,_))?O.priority&&(C=1):z[M]=E=Bh.call(e,b,M,"get",S[M],V,_,0,s.stringFilter);e._op&&e._op[y]&&e.kill(b,e._op[y]),A&&e._pt&&(Js=e,nn.killTweensOf(b,z,e.globalTime(t)),q=!e.parent,Js=0),e._pt&&l&&(xd[D.id]=1)}C&&Zx(e),e._onInit&&e._onInit(e)}e._onUpdate=c,e._initted=(!e._op||e._pt)&&!q,d&&t<=0&&v.render(Di,!0,!0)},zw=function(e,t,n,s,r,o,a,l){var c=(e._pt&&e._ptCache||(e._ptCache={}))[t],u,f,d,h;if(!c)for(c=e._ptCache[t]=[],d=e._ptLookup,h=e._targets.length;h--;){if(u=d[h][t],u&&u.d&&u.d._pt)for(u=u.d._pt;u&&u.p!==t&&u.fp!==t;)u=u._next;if(!u)return bd=1,e.vars[t]="+=0",Uh(e,a),bd=0,l?il(t+" not eligible for reset"):1;c.push(u)}for(h=c.length;h--;)f=c[h],u=f._pt||f,u.s=(s||s===0)&&!r?s:u.s+(s||0)+o*u.c,u.c=n-u.s,f.e&&(f.e=cn(n)+Pn(f.e)),f.b&&(f.b=u.s+Pn(f.b))},kw=function(e,t){var n=e[0]?Xr(e[0]).harness:0,s=n&&n.aliases,r,o,a,l;if(!s)return t;r=Jo({},t);for(o in s)if(o in r)for(l=s[o].split(","),a=l.length;a--;)r[l[a]]=r[o];return r},Hw=function(e,t,n,s){var r=t.ease||s||"power1.inOut",o,a;if(Bn(t))a=n[e]||(n[e]=[]),t.forEach(function(l,c){return a.push({t:c/(t.length-1)*100,v:l,e:r})});else for(o in t)a=n[o]||(n[o]=[]),o==="ease"||a.push({t:parseFloat(e),v:t[o],e:r})},Va=function(e,t,n,s,r){return an(e)?e.call(t,n,s,r):yn(e)&&~e.indexOf("random(")?rl(e):e},Yx=Ph+"repeat,repeatDelay,yoyo,repeatRefresh,yoyoEase,autoRevert",Qx={};ti(Yx+",id,stagger,delay,duration,paused,scrollTrigger",function(i){return Qx[i]=1});var pn=(function(i){hx(e,i);function e(n,s,r,o){var a;typeof s=="number"&&(r.duration=s,s=r,r=null),a=i.call(this,o?s:ka(s))||this;var l=a.vars,c=l.duration,u=l.delay,f=l.immediateRender,d=l.stagger,h=l.overwrite,x=l.keyframes,p=l.defaults,g=l.scrollTrigger,m=l.yoyoEase,_=s.parent||nn,A=(Bn(n)||mx(n)?Us(n[0]):"length"in s)?[n]:Pi(n),v,S,y,M,E,b,C,D;if(a._targets=A.length?Fh(A):il("GSAP target "+n+" not found. https://gsap.com",!_i.nullTargetWarn)||[],a._ptLookup=[],a._overwrite=h,x||d||nc(c)||nc(u)){if(s=a.vars,v=a.timeline=new Wn({data:"nested",defaults:p||{},targets:_&&_.data==="nested"?_.vars.targets:A}),v.kill(),v.parent=v._dp=Ms(a),v._start=0,d||nc(c)||nc(u)){if(M=A.length,C=d&&Px(d),ls(d))for(E in d)~Yx.indexOf(E)&&(D||(D={}),D[E]=d[E]);for(S=0;S<M;S++)y=Pc(s,Qx),y.stagger=0,m&&(y.yoyoEase=m),D&&Jo(y,D),b=A[S],y.duration=+Va(c,Ms(a),S,b,A),y.delay=(+Va(u,Ms(a),S,b,A)||0)-a._delay,!d&&M===1&&y.delay&&(a._delay=u=y.delay,a._start+=u,y.delay=0),v.to(b,y,C?C(S,b,A):0),v._ease=At.none;v.duration()?c=u=0:a.timeline=0}else if(x){ka(Ai(v.vars.defaults,{ease:"none"})),v._ease=Yr(x.ease||s.ease||"none");var F=0,O,z,V;if(Bn(x))x.forEach(function(H){return v.to(A,H,">")}),v.duration();else{y={};for(E in x)E==="ease"||E==="easeEach"||Hw(E,x[E],y,x.easeEach);for(E in y)for(O=y[E].sort(function(H,q){return H.t-q.t}),F=0,S=0;S<O.length;S++)z=O[S],V={ease:z.e,duration:(z.t-(S?O[S-1].t:0))/100*c},V[E]=z.v,v.to(A,V,F),F+=V.duration;v.duration()<c&&v.to({},{duration:c-v.duration()})}}c||a.duration(c=v.duration())}else a.timeline=0;return h===!0&&!Th&&(Js=Ms(a),nn.killTweensOf(A),Js=0),ji(_,Ms(a),r),s.reversed&&a.reverse(),s.paused&&a.paused(!0),(f||!c&&!x&&a._start===tn(_._time)&&ei(f)&&_w(Ms(a))&&_.data!=="nested")&&(a._tTime=-kt,a.render(Math.max(0,-u)||0)),g&&wx(Ms(a),g),a}var t=e.prototype;return t.render=function(s,r,o){var a=this._time,l=this._tDur,c=this._dur,u=s<0,f=s>l-kt&&!u?l:s<kt?0:s,d,h,x,p,g,m,_,A,v;if(!c)Aw(this,s,r,o);else if(f!==this._tTime||!s||o||!this._initted&&this._tTime||this._startAt&&this._zTime<0!==u||this._lazy){if(d=f,A=this.timeline,this._repeat){if(p=c+this._rDelay,this._repeat<-1&&u)return this.totalTime(p*100+s,r,o);if(d=tn(f%p),f===l?(x=this._repeat,d=c):(g=tn(f/p),x=~~g,x&&x===g?(d=c,x--):d>c&&(d=c)),m=this._yoyo&&x&1,m&&(v=this._yEase,d=c-d),g=ea(this._tTime,p),d===a&&!o&&this._initted&&x===g)return this._tTime=f,this;x!==g&&(A&&this._yEase&&Gx(A,m),this.vars.repeatRefresh&&!m&&!this._lock&&d!==p&&this._initted&&(this._lock=o=1,this.render(tn(p*x),!0).invalidate()._lock=0))}if(!this._initted){if(Rx(this,u?s:d,o,r,f))return this._tTime=0,this;if(a!==this._time&&!(o&&this.vars.repeatRefresh&&x!==g))return this;if(c!==this._dur)return this.render(s,r,o)}if(this._tTime=f,this._time=d,!this._act&&this._ts&&(this._act=1,this._lazy=0),this.ratio=_=(v||this._ease)(d/c),this._from&&(this.ratio=_=1-_),!a&&f&&!r&&!g&&(mi(this,"onStart"),this._tTime!==f))return this;for(h=this._pt;h;)h.r(_,h.d),h=h._next;A&&A.render(s<0?s:A._dur*A._ease(d/this._dur),r,o)||this._startAt&&(this._zTime=s),this._onUpdate&&!r&&(u&&_d(this,s,r,o),mi(this,"onUpdate")),this._repeat&&x!==g&&this.vars.onRepeat&&!r&&this.parent&&mi(this,"onRepeat"),(f===this._tDur||!f)&&this._tTime===f&&(u&&!this._onUpdate&&_d(this,s,!0,!0),(s||!c)&&(f===this._tDur&&this._ts>0||!f&&this._ts<0)&&ur(this,1),!r&&!(u&&!a)&&(f||a||m)&&(mi(this,f===l?"onComplete":"onReverseComplete",!0),this._prom&&!(f<l&&this.timeScale()>0)&&this._prom()))}return this},t.targets=function(){return this._targets},t.invalidate=function(s){return(!s||!this.vars.runBackwards)&&(this._startAt=0),this._pt=this._op=this._onUpdate=this._lazy=this.ratio=0,this._ptLookup=[],this.timeline&&this.timeline.invalidate(s),i.prototype.invalidate.call(this,s)},t.resetTo=function(s,r,o,a,l){ol||di.wake(),this._ts||this.play();var c=Math.min(this._dur,(this._dp._time-this._start)*this._ts),u;return this._initted||Uh(this,c),u=this._ease(c/this._dur),zw(this,s,r,o,a,u,c,l)?this.resetTo(s,r,o,a,1):(nu(this,0),this.parent||Tx(this._dp,this,"_first","_last",this._dp._sort?"_start":0),this.render(0))},t.kill=function(s,r){if(r===void 0&&(r="all"),!s&&(!r||r==="all"))return this._lazy=this._pt=0,this.parent?ba(this):this.scrollTrigger&&this.scrollTrigger.kill(!!Tn),this;if(this.timeline){var o=this.timeline.totalDuration();return this.timeline.killTweensOf(s,r,Js&&Js.vars.overwrite!==!0)._first||ba(this),this.parent&&o!==this.timeline.totalDuration()&&ta(this,this._dur*this.timeline._tDur/o,0,1),this}var a=this._targets,l=s?Pi(s):a,c=this._ptLookup,u=this._pt,f,d,h,x,p,g,m;if((!r||r==="all")&&gw(a,l))return r==="all"&&(this._pt=0),ba(this);for(f=this._op=this._op||[],r!=="all"&&(yn(r)&&(p={},ti(r,function(_){return p[_]=1}),r=p),r=kw(a,r)),m=a.length;m--;)if(~l.indexOf(a[m])){d=c[m],r==="all"?(f[m]=r,x=d,h={}):(h=f[m]=f[m]||{},x=r);for(p in x)g=d&&d[p],g&&((!("kill"in g.d)||g.d.kill(p)===!0)&&eu(this,g,"_pt"),delete d[p]),h!=="all"&&(h[p]=1)}return this._initted&&!this._pt&&u&&ba(this),this},e.to=function(s,r){return new e(s,r,arguments[2])},e.from=function(s,r){return Ha(1,arguments)},e.delayedCall=function(s,r,o,a){return new e(r,0,{immediateRender:!1,lazy:!1,overwrite:!1,delay:s,onComplete:r,onReverseComplete:r,onCompleteParams:o,onReverseCompleteParams:o,callbackScope:a})},e.fromTo=function(s,r,o){return Ha(2,arguments)},e.set=function(s,r){return r.duration=0,r.repeatDelay||(r.repeat=0),new e(s,r)},e.killTweensOf=function(s,r,o){return nn.killTweensOf(s,r,o)},e})(al);Ai(pn.prototype,{_targets:[],_lazy:0,_startAt:0,_op:0,_onInit:0});ti("staggerTo,staggerFrom,staggerFromTo",function(i){pn[i]=function(){var e=new Wn,t=Ad.call(arguments,0);return t.splice(i==="staggerFromTo"?5:4,0,0),e[i].apply(e,t)}});var Oh=function(e,t,n){return e[t]=n},Kx=function(e,t,n){return e[t](n)},Vw=function(e,t,n,s){return e[t](s.fp,n)},Gw=function(e,t,n){return e.setAttribute(t,n)},Nh=function(e,t){return an(e[t])?Kx:Eh(e[t])&&e.setAttribute?Gw:Oh},jx=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e6)/1e6,t)},Ww=function(e,t){return t.set(t.t,t.p,!!(t.s+t.c*e),t)},$x=function(e,t){var n=t._pt,s="";if(!e&&t.b)s=t.b;else if(e===1&&t.e)s=t.e;else{for(;n;)s=n.p+(n.m?n.m(n.s+n.c*e):Math.round((n.s+n.c*e)*1e4)/1e4)+s,n=n._next;s+=t.c}t.set(t.t,t.p,s,t)},zh=function(e,t){for(var n=t._pt;n;)n.r(e,n.d),n=n._next},Xw=function(e,t,n,s){for(var r=this._pt,o;r;)o=r._next,r.p===s&&r.modifier(e,t,n),r=o},qw=function(e){for(var t=this._pt,n,s;t;)s=t._next,t.p===e&&!t.op||t.op===e?eu(this,t,"_pt"):t.dep||(n=1),t=s;return!n},Yw=function(e,t,n,s){s.mSet(e,t,s.m.call(s.tween,n,s.mt),s)},Zx=function(e){for(var t=e._pt,n,s,r,o;t;){for(n=t._next,s=r;s&&s.pr>t.pr;)s=s._next;(t._prev=s?s._prev:o)?t._prev._next=t:r=t,(t._next=s)?s._prev=t:o=t,t=n}e._pt=r},ni=(function(){function i(t,n,s,r,o,a,l,c,u){this.t=n,this.s=r,this.c=o,this.p=s,this.r=a||jx,this.d=l||this,this.set=c||Oh,this.pr=u||0,this._next=t,t&&(t._prev=this)}var e=i.prototype;return e.modifier=function(n,s,r){this.mSet=this.mSet||this.set,this.set=Yw,this.m=n,this.mt=r,this.tween=s},i})();ti(Ph+"parent,duration,ease,delay,overwrite,runBackwards,startAt,yoyo,immediateRender,repeat,repeatDelay,data,paused,reversed,lazy,callbackScope,stringFilter,id,yoyoEase,stagger,inherit,repeatRefresh,keyframes,autoRevert,scrollTrigger",function(i){return Dh[i]=1});vi.TweenMax=vi.TweenLite=pn;vi.TimelineLite=vi.TimelineMax=Wn;nn=new Wn({sortChildren:!1,defaults:Zo,autoRemoveChildren:!0,id:"root",smoothChildTiming:!0});_i.stringFilter=Hx;var Qr=[],mc={},Qw=[],l0=0,Kw=0,uf=function(e){return(mc[e]||Qw).map(function(t){return t()})},Md=function(){var e=Date.now(),t=[];e-l0>2&&(uf("matchMediaInit"),Qr.forEach(function(n){var s=n.queries,r=n.conditions,o,a,l,c;for(a in s)o=Yi.matchMedia(s[a]).matches,o&&(l=1),o!==r[a]&&(r[a]=o,c=1);c&&(n.revert(),l&&t.push(n))}),uf("matchMediaRevert"),t.forEach(function(n){return n.onMatch(n,function(s){return n.add(null,s)})}),l0=e,uf("matchMedia"))},Jx=(function(){function i(t,n){this.selector=n&&Sd(n),this.data=[],this._r=[],this.isReverted=!1,this.id=Kw++,t&&this.add(t)}var e=i.prototype;return e.add=function(n,s,r){an(n)&&(r=s,s=n,n=an);var o=this,a=function(){var c=Zt,u=o.selector,f;return c&&c!==o&&c.data.push(o),r&&(o.selector=Sd(r)),Zt=o,f=s.apply(o,arguments),an(f)&&o._r.push(f),Zt=c,o.selector=u,o.isReverted=!1,f};return o.last=a,n===an?a(o,function(l){return o.add(null,l)}):n?o[n]=a:a},e.ignore=function(n){var s=Zt;Zt=null,n(this),Zt=s},e.getTweens=function(){var n=[];return this.data.forEach(function(s){return s instanceof i?n.push.apply(n,s.getTweens()):s instanceof pn&&!(s.parent&&s.parent.data==="nested")&&n.push(s)}),n},e.clear=function(){this._r.length=this.data.length=0},e.kill=function(n,s){var r=this;if(n?(function(){for(var a=r.getTweens(),l=r.data.length,c;l--;)c=r.data[l],c.data==="isFlip"&&(c.revert(),c.getChildren(!0,!0,!1).forEach(function(u){return a.splice(a.indexOf(u),1)}));for(a.map(function(u){return{g:u._dur||u._delay||u._sat&&!u._sat.vars.immediateRender?u.globalTime(0):-1/0,t:u}}).sort(function(u,f){return f.g-u.g||-1/0}).forEach(function(u){return u.t.revert(n)}),l=r.data.length;l--;)c=r.data[l],c instanceof Wn?c.data!=="nested"&&(c.scrollTrigger&&c.scrollTrigger.revert(),c.kill()):!(c instanceof pn)&&c.revert&&c.revert(n);r._r.forEach(function(u){return u(n,r)}),r.isReverted=!0})():this.data.forEach(function(a){return a.kill&&a.kill()}),this.clear(),s)for(var o=Qr.length;o--;)Qr[o].id===this.id&&Qr.splice(o,1)},e.revert=function(n){this.kill(n||{})},i})(),jw=(function(){function i(t){this.contexts=[],this.scope=t,Zt&&Zt.data.push(this)}var e=i.prototype;return e.add=function(n,s,r){ls(n)||(n={matches:n});var o=new Jx(0,r||this.scope),a=o.conditions={},l,c,u;Zt&&!o.selector&&(o.selector=Zt.selector),this.contexts.push(o),s=o.add("onMatch",s),o.queries=n;for(c in n)c==="all"?u=1:(l=Yi.matchMedia(n[c]),l&&(Qr.indexOf(o)<0&&Qr.push(o),(a[c]=l.matches)&&(u=1),l.addListener?l.addListener(Md):l.addEventListener("change",Md)));return u&&s(o,function(f){return o.add(null,f)}),this},e.revert=function(n){this.kill(n||{})},e.kill=function(n){this.contexts.forEach(function(s){return s.kill(n,!0)})},i})(),Lc={registerPlugin:function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];t.forEach(function(s){return Nx(s)})},timeline:function(e){return new Wn(e)},getTweensOf:function(e,t){return nn.getTweensOf(e,t)},getProperty:function(e,t,n,s){yn(e)&&(e=Pi(e)[0]);var r=Xr(e||{}).get,o=n?Cx:Mx;return n==="native"&&(n=""),e&&(t?o((fi[t]&&fi[t].get||r)(e,t,n,s)):function(a,l,c){return o((fi[a]&&fi[a].get||r)(e,a,l,c))})},quickSetter:function(e,t,n){if(e=Pi(e),e.length>1){var s=e.map(function(u){return si.quickSetter(u,t,n)}),r=s.length;return function(u){for(var f=r;f--;)s[f](u)}}e=e[0]||{};var o=fi[t],a=Xr(e),l=a.harness&&(a.harness.aliases||{})[t]||t,c=o?function(u){var f=new o;wo._pt=0,f.init(e,n?u+n:u,wo,0,[e]),f.render(1,f),wo._pt&&zh(1,wo)}:a.set(e,l);return o?c:function(u){return c(e,l,n?u+n:u,a,1)}},quickTo:function(e,t,n){var s,r=si.to(e,Ai((s={},s[t]="+=0.1",s.paused=!0,s.stagger=0,s),n||{})),o=function(l,c,u){return r.resetTo(t,l,c,u)};return o.tween=r,o},isTweening:function(e){return nn.getTweensOf(e,!0).length>0},defaults:function(e){return e&&e.ease&&(e.ease=Yr(e.ease,Zo.ease)),i0(Zo,e||{})},config:function(e){return i0(_i,e||{})},registerEffect:function(e){var t=e.name,n=e.effect,s=e.plugins,r=e.defaults,o=e.extendTimeline;(s||"").split(",").forEach(function(a){return a&&!fi[a]&&!vi[a]&&il(t+" effect requires "+a+" plugin.")}),of[t]=function(a,l,c){return n(Pi(a),Ai(l||{},r),c)},o&&(Wn.prototype[t]=function(a,l,c){return this.add(of[t](a,ls(l)?l:(c=l)&&{},this),c)})},registerEase:function(e,t){At[e]=Yr(t)},parseEase:function(e,t){return arguments.length?Yr(e,t):At},getById:function(e){return nn.getById(e)},exportRoot:function(e,t){e===void 0&&(e={});var n=new Wn(e),s,r;for(n.smoothChildTiming=ei(e.smoothChildTiming),nn.remove(n),n._dp=0,n._time=n._tTime=nn._time,s=nn._first;s;)r=s._next,(t||!(!s._dur&&s instanceof pn&&s.vars.onComplete===s._targets[0]))&&ji(n,s,s._start-s._delay),s=r;return ji(nn,n,0),n},context:function(e,t){return e?new Jx(e,t):Zt},matchMedia:function(e){return new jw(e)},matchMediaRefresh:function(){return Qr.forEach(function(e){var t=e.conditions,n,s;for(s in t)t[s]&&(t[s]=!1,n=1);n&&e.revert()})||Md()},addEventListener:function(e,t){var n=mc[e]||(mc[e]=[]);~n.indexOf(t)||n.push(t)},removeEventListener:function(e,t){var n=mc[e],s=n&&n.indexOf(t);s>=0&&n.splice(s,1)},utils:{wrap:ww,wrapYoyo:Rw,distribute:Px,random:Lx,snap:Fx,normalize:Ew,getUnit:Pn,clamp:bw,splitColor:zx,toArray:Pi,selector:Sd,mapRange:Ux,pipe:Cw,unitize:Tw,interpolate:Iw,shuffle:Dx},install:vx,effects:of,ticker:di,updateRoot:Wn.updateRoot,plugins:fi,globalTimeline:nn,core:{PropTween:ni,globals:Ax,Tween:pn,Timeline:Wn,Animation:al,getCache:Xr,_removeLinkedListItem:eu,reverting:function(){return Tn},context:function(e){return e&&Zt&&(Zt.data.push(e),e._ctx=Zt),Zt},suppressOverwrites:function(e){return Th=e}}};ti("to,from,fromTo,delayedCall,set,killTweensOf",function(i){return Lc[i]=pn[i]});di.add(Wn.updateRoot);wo=Lc.to({},{duration:0});var $w=function(e,t){for(var n=e._pt;n&&n.p!==t&&n.op!==t&&n.fp!==t;)n=n._next;return n},Zw=function(e,t){var n=e._targets,s,r,o;for(s in t)for(r=n.length;r--;)o=e._ptLookup[r][s],o&&(o=o.d)&&(o._pt&&(o=$w(o,s)),o&&o.modifier&&o.modifier(t[s],e,n[r],s))},ff=function(e,t){return{name:e,headless:1,rawVars:1,init:function(s,r,o){o._onInit=function(a){var l,c;if(yn(r)&&(l={},ti(r,function(u){return l[u]=1}),r=l),t){l={};for(c in r)l[c]=t(r[c]);r=l}Zw(a,r)}}}},si=Lc.registerPlugin({name:"attr",init:function(e,t,n,s,r){var o,a,l;this.tween=n;for(o in t)l=e.getAttribute(o)||"",a=this.add(e,"setAttribute",(l||0)+"",t[o],s,r,0,0,o),a.op=o,a.b=l,this._props.push(o)},render:function(e,t){for(var n=t._pt;n;)Tn?n.set(n.t,n.p,n.b,n):n.r(e,n.d),n=n._next}},{name:"endArray",headless:1,init:function(e,t){for(var n=t.length;n--;)this.add(e,n,e[n]||0,t[n],0,0,0,0,0,1)}},ff("roundProps",yd),ff("modifiers"),ff("snap",Fx))||Lc;pn.version=Wn.version=si.version="3.14.2";_x=1;wh()&&na();At.Power0;At.Power1;At.Power2;At.Power3;At.Power4;At.Linear;At.Quad;At.Cubic;At.Quart;At.Quint;At.Strong;At.Elastic;At.Back;At.SteppedEase;At.Bounce;At.Sine;At.Expo;At.Circ;var c0,er,ko,kh,Vr,u0,Hh,Jw=function(){return typeof window<"u"},Os={},Br=180/Math.PI,Ho=Math.PI/180,yo=Math.atan2,f0=1e8,Vh=/([A-Z])/g,e3=/(left|right|width|margin|padding|x)/i,t3=/[\s,\(]\S/,ts={autoAlpha:"opacity,visibility",scale:"scaleX,scaleY",alpha:"opacity"},Cd=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},n3=function(e,t){return t.set(t.t,t.p,e===1?t.e:Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},i3=function(e,t){return t.set(t.t,t.p,e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},s3=function(e,t){return t.set(t.t,t.p,e===1?t.e:e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},r3=function(e,t){var n=t.s+t.c*e;t.set(t.t,t.p,~~(n+(n<0?-.5:.5))+t.u,t)},e_=function(e,t){return t.set(t.t,t.p,e?t.e:t.b,t)},t_=function(e,t){return t.set(t.t,t.p,e!==1?t.b:t.e,t)},o3=function(e,t,n){return e.style[t]=n},a3=function(e,t,n){return e.style.setProperty(t,n)},l3=function(e,t,n){return e._gsap[t]=n},c3=function(e,t,n){return e._gsap.scaleX=e._gsap.scaleY=n},u3=function(e,t,n,s,r){var o=e._gsap;o.scaleX=o.scaleY=n,o.renderTransform(r,o)},f3=function(e,t,n,s,r){var o=e._gsap;o[t]=n,o.renderTransform(r,o)},sn="transform",ii=sn+"Origin",d3=function i(e,t){var n=this,s=this.target,r=s.style,o=s._gsap;if(e in Os&&r){if(this.tfm=this.tfm||{},e!=="transform")e=ts[e]||e,~e.indexOf(",")?e.split(",").forEach(function(a){return n.tfm[a]=Ts(s,a)}):this.tfm[e]=o.x?o[e]:Ts(s,e),e===ii&&(this.tfm.zOrigin=o.zOrigin);else return ts.transform.split(",").forEach(function(a){return i.call(n,a,t)});if(this.props.indexOf(sn)>=0)return;o.svg&&(this.svgo=s.getAttribute("data-svg-origin"),this.props.push(ii,t,"")),e=sn}(r||t)&&this.props.push(e,t,r[e])},n_=function(e){e.translate&&(e.removeProperty("translate"),e.removeProperty("scale"),e.removeProperty("rotate"))},h3=function(){var e=this.props,t=this.target,n=t.style,s=t._gsap,r,o;for(r=0;r<e.length;r+=3)e[r+1]?e[r+1]===2?t[e[r]](e[r+2]):t[e[r]]=e[r+2]:e[r+2]?n[e[r]]=e[r+2]:n.removeProperty(e[r].substr(0,2)==="--"?e[r]:e[r].replace(Vh,"-$1").toLowerCase());if(this.tfm){for(o in this.tfm)s[o]=this.tfm[o];s.svg&&(s.renderTransform(),t.setAttribute("data-svg-origin",this.svgo||"")),r=Hh(),(!r||!r.isStart)&&!n[sn]&&(n_(n),s.zOrigin&&n[ii]&&(n[ii]+=" "+s.zOrigin+"px",s.zOrigin=0,s.renderTransform()),s.uncache=1)}},i_=function(e,t){var n={target:e,props:[],revert:h3,save:d3};return e._gsap||si.core.getCache(e),t&&e.style&&e.nodeType&&t.split(",").forEach(function(s){return n.save(s)}),n},s_,Td=function(e,t){var n=er.createElementNS?er.createElementNS((t||"http://www.w3.org/1999/xhtml").replace(/^https/,"http"),e):er.createElement(e);return n&&n.style?n:er.createElement(e)},gi=function i(e,t,n){var s=getComputedStyle(e);return s[t]||s.getPropertyValue(t.replace(Vh,"-$1").toLowerCase())||s.getPropertyValue(t)||!n&&i(e,ia(t)||t,1)||""},d0="O,Moz,ms,Ms,Webkit".split(","),ia=function(e,t,n){var s=t||Vr,r=s.style,o=5;if(e in r&&!n)return e;for(e=e.charAt(0).toUpperCase()+e.substr(1);o--&&!(d0[o]+e in r););return o<0?null:(o===3?"ms":o>=0?d0[o]:"")+e},Ed=function(){Jw()&&window.document&&(c0=window,er=c0.document,ko=er.documentElement,Vr=Td("div")||{style:{}},Td("div"),sn=ia(sn),ii=sn+"Origin",Vr.style.cssText="border-width:0;line-height:0;position:absolute;padding:0",s_=!!ia("perspective"),Hh=si.core.reverting,kh=1)},h0=function(e){var t=e.ownerSVGElement,n=Td("svg",t&&t.getAttribute("xmlns")||"http://www.w3.org/2000/svg"),s=e.cloneNode(!0),r;s.style.display="block",n.appendChild(s),ko.appendChild(n);try{r=s.getBBox()}catch{}return n.removeChild(s),ko.removeChild(n),r},p0=function(e,t){for(var n=t.length;n--;)if(e.hasAttribute(t[n]))return e.getAttribute(t[n])},r_=function(e){var t,n;try{t=e.getBBox()}catch{t=h0(e),n=1}return t&&(t.width||t.height)||n||(t=h0(e)),t&&!t.width&&!t.x&&!t.y?{x:+p0(e,["x","cx","x1"])||0,y:+p0(e,["y","cy","y1"])||0,width:0,height:0}:t},o_=function(e){return!!(e.getCTM&&(!e.parentNode||e.ownerSVGElement)&&r_(e))},fr=function(e,t){if(t){var n=e.style,s;t in Os&&t!==ii&&(t=sn),n.removeProperty?(s=t.substr(0,2),(s==="ms"||t.substr(0,6)==="webkit")&&(t="-"+t),n.removeProperty(s==="--"?t:t.replace(Vh,"-$1").toLowerCase())):n.removeAttribute(t)}},tr=function(e,t,n,s,r,o){var a=new ni(e._pt,t,n,0,1,o?t_:e_);return e._pt=a,a.b=s,a.e=r,e._props.push(n),a},m0={deg:1,rad:1,turn:1},p3={grid:1,flex:1},dr=function i(e,t,n,s){var r=parseFloat(n)||0,o=(n+"").trim().substr((r+"").length)||"px",a=Vr.style,l=e3.test(t),c=e.tagName.toLowerCase()==="svg",u=(c?"client":"offset")+(l?"Width":"Height"),f=100,d=s==="px",h=s==="%",x,p,g,m;if(s===o||!r||m0[s]||m0[o])return r;if(o!=="px"&&!d&&(r=i(e,t,n,"px")),m=e.getCTM&&o_(e),(h||o==="%")&&(Os[t]||~t.indexOf("adius")))return x=m?e.getBBox()[l?"width":"height"]:e[u],cn(h?r/x*f:r/100*x);if(a[l?"width":"height"]=f+(d?o:s),p=s!=="rem"&&~t.indexOf("adius")||s==="em"&&e.appendChild&&!c?e:e.parentNode,m&&(p=(e.ownerSVGElement||{}).parentNode),(!p||p===er||!p.appendChild)&&(p=er.body),g=p._gsap,g&&h&&g.width&&l&&g.time===di.time&&!g.uncache)return cn(r/g.width*f);if(h&&(t==="height"||t==="width")){var _=e.style[t];e.style[t]=f+s,x=e[u],_?e.style[t]=_:fr(e,t)}else(h||o==="%")&&!p3[gi(p,"display")]&&(a.position=gi(e,"position")),p===e&&(a.position="static"),p.appendChild(Vr),x=Vr[u],p.removeChild(Vr),a.position="absolute";return l&&h&&(g=Xr(p),g.time=di.time,g.width=p[u]),cn(d?x*r/f:x&&r?f/x*r:0)},Ts=function(e,t,n,s){var r;return kh||Ed(),t in ts&&t!=="transform"&&(t=ts[t],~t.indexOf(",")&&(t=t.split(",")[0])),Os[t]&&t!=="transform"?(r=cl(e,s),r=t!=="transformOrigin"?r[t]:r.svg?r.origin:Uc(gi(e,ii))+" "+r.zOrigin+"px"):(r=e.style[t],(!r||r==="auto"||s||~(r+"").indexOf("calc("))&&(r=Bc[t]&&Bc[t](e,t,n)||gi(e,t)||yx(e,t)||(t==="opacity"?1:0))),n&&!~(r+"").trim().indexOf(" ")?dr(e,t,r,n)+n:r},m3=function(e,t,n,s){if(!n||n==="none"){var r=ia(t,e,1),o=r&&gi(e,r,1);o&&o!==n?(t=r,n=o):t==="borderColor"&&(n=gi(e,"borderTopColor"))}var a=new ni(this._pt,e.style,t,0,1,$x),l=0,c=0,u,f,d,h,x,p,g,m,_,A,v,S;if(a.b=n,a.e=s,n+="",s+="",s.substring(0,6)==="var(--"&&(s=gi(e,s.substring(4,s.indexOf(")")))),s==="auto"&&(p=e.style[t],e.style[t]=s,s=gi(e,t)||s,p?e.style[t]=p:fr(e,t)),u=[n,s],Hx(u),n=u[0],s=u[1],d=n.match(Eo)||[],S=s.match(Eo)||[],S.length){for(;f=Eo.exec(s);)g=f[0],_=s.substring(l,f.index),x?x=(x+1)%5:(_.substr(-5)==="rgba("||_.substr(-5)==="hsla(")&&(x=1),g!==(p=d[c++]||"")&&(h=parseFloat(p)||0,v=p.substr((h+"").length),g.charAt(1)==="="&&(g=zo(h,g)+v),m=parseFloat(g),A=g.substr((m+"").length),l=Eo.lastIndex-A.length,A||(A=A||_i.units[t]||v,l===s.length&&(s+=A,a.e+=A)),v!==A&&(h=dr(e,t,p,A)||0),a._pt={_next:a._pt,p:_||c===1?_:",",s:h,c:m-h,m:x&&x<4||t==="zIndex"?Math.round:0});a.c=l<s.length?s.substring(l,s.length):""}else a.r=t==="display"&&s==="none"?t_:e_;return xx.test(s)&&(a.e=0),this._pt=a,a},g0={top:"0%",bottom:"100%",left:"0%",right:"100%",center:"50%"},g3=function(e){var t=e.split(" "),n=t[0],s=t[1]||"50%";return(n==="top"||n==="bottom"||s==="left"||s==="right")&&(e=n,n=s,s=e),t[0]=g0[n]||n,t[1]=g0[s]||s,t.join(" ")},x3=function(e,t){if(t.tween&&t.tween._time===t.tween._dur){var n=t.t,s=n.style,r=t.u,o=n._gsap,a,l,c;if(r==="all"||r===!0)s.cssText="",l=1;else for(r=r.split(","),c=r.length;--c>-1;)a=r[c],Os[a]&&(l=1,a=a==="transformOrigin"?ii:sn),fr(n,a);l&&(fr(n,sn),o&&(o.svg&&n.removeAttribute("transform"),s.scale=s.rotate=s.translate="none",cl(n,1),o.uncache=1,n_(s)))}},Bc={clearProps:function(e,t,n,s,r){if(r.data!=="isFromStart"){var o=e._pt=new ni(e._pt,t,n,0,0,x3);return o.u=s,o.pr=-10,o.tween=r,e._props.push(n),1}}},ll=[1,0,0,1,0,0],a_={},l_=function(e){return e==="matrix(1, 0, 0, 1, 0, 0)"||e==="none"||!e},x0=function(e){var t=gi(e,sn);return l_(t)?ll:t.substr(7).match(gx).map(cn)},Gh=function(e,t){var n=e._gsap||Xr(e),s=e.style,r=x0(e),o,a,l,c;return n.svg&&e.getAttribute("transform")?(l=e.transform.baseVal.consolidate().matrix,r=[l.a,l.b,l.c,l.d,l.e,l.f],r.join(",")==="1,0,0,1,0,0"?ll:r):(r===ll&&!e.offsetParent&&e!==ko&&!n.svg&&(l=s.display,s.display="block",o=e.parentNode,(!o||!e.offsetParent&&!e.getBoundingClientRect().width)&&(c=1,a=e.nextElementSibling,ko.appendChild(e)),r=x0(e),l?s.display=l:fr(e,"display"),c&&(a?o.insertBefore(e,a):o?o.appendChild(e):ko.removeChild(e))),t&&r.length>6?[r[0],r[1],r[4],r[5],r[12],r[13]]:r)},wd=function(e,t,n,s,r,o){var a=e._gsap,l=r||Gh(e,!0),c=a.xOrigin||0,u=a.yOrigin||0,f=a.xOffset||0,d=a.yOffset||0,h=l[0],x=l[1],p=l[2],g=l[3],m=l[4],_=l[5],A=t.split(" "),v=parseFloat(A[0])||0,S=parseFloat(A[1])||0,y,M,E,b;n?l!==ll&&(M=h*g-x*p)&&(E=v*(g/M)+S*(-p/M)+(p*_-g*m)/M,b=v*(-x/M)+S*(h/M)-(h*_-x*m)/M,v=E,S=b):(y=r_(e),v=y.x+(~A[0].indexOf("%")?v/100*y.width:v),S=y.y+(~(A[1]||A[0]).indexOf("%")?S/100*y.height:S)),s||s!==!1&&a.smooth?(m=v-c,_=S-u,a.xOffset=f+(m*h+_*p)-m,a.yOffset=d+(m*x+_*g)-_):a.xOffset=a.yOffset=0,a.xOrigin=v,a.yOrigin=S,a.smooth=!!s,a.origin=t,a.originIsAbsolute=!!n,e.style[ii]="0px 0px",o&&(tr(o,a,"xOrigin",c,v),tr(o,a,"yOrigin",u,S),tr(o,a,"xOffset",f,a.xOffset),tr(o,a,"yOffset",d,a.yOffset)),e.setAttribute("data-svg-origin",v+" "+S)},cl=function(e,t){var n=e._gsap||new Xx(e);if("x"in n&&!t&&!n.uncache)return n;var s=e.style,r=n.scaleX<0,o="px",a="deg",l=getComputedStyle(e),c=gi(e,ii)||"0",u,f,d,h,x,p,g,m,_,A,v,S,y,M,E,b,C,D,F,O,z,V,H,q,G,$,fe,Y,we,ze,ke,We;return u=f=d=p=g=m=_=A=v=0,h=x=1,n.svg=!!(e.getCTM&&o_(e)),l.translate&&((l.translate!=="none"||l.scale!=="none"||l.rotate!=="none")&&(s[sn]=(l.translate!=="none"?"translate3d("+(l.translate+" 0 0").split(" ").slice(0,3).join(", ")+") ":"")+(l.rotate!=="none"?"rotate("+l.rotate+") ":"")+(l.scale!=="none"?"scale("+l.scale.split(" ").join(",")+") ":"")+(l[sn]!=="none"?l[sn]:"")),s.scale=s.rotate=s.translate="none"),M=Gh(e,n.svg),n.svg&&(n.uncache?(G=e.getBBox(),c=n.xOrigin-G.x+"px "+(n.yOrigin-G.y)+"px",q=""):q=!t&&e.getAttribute("data-svg-origin"),wd(e,q||c,!!q||n.originIsAbsolute,n.smooth!==!1,M)),S=n.xOrigin||0,y=n.yOrigin||0,M!==ll&&(D=M[0],F=M[1],O=M[2],z=M[3],u=V=M[4],f=H=M[5],M.length===6?(h=Math.sqrt(D*D+F*F),x=Math.sqrt(z*z+O*O),p=D||F?yo(F,D)*Br:0,_=O||z?yo(O,z)*Br+p:0,_&&(x*=Math.abs(Math.cos(_*Ho))),n.svg&&(u-=S-(S*D+y*O),f-=y-(S*F+y*z))):(We=M[6],ze=M[7],fe=M[8],Y=M[9],we=M[10],ke=M[11],u=M[12],f=M[13],d=M[14],E=yo(We,we),g=E*Br,E&&(b=Math.cos(-E),C=Math.sin(-E),q=V*b+fe*C,G=H*b+Y*C,$=We*b+we*C,fe=V*-C+fe*b,Y=H*-C+Y*b,we=We*-C+we*b,ke=ze*-C+ke*b,V=q,H=G,We=$),E=yo(-O,we),m=E*Br,E&&(b=Math.cos(-E),C=Math.sin(-E),q=D*b-fe*C,G=F*b-Y*C,$=O*b-we*C,ke=z*C+ke*b,D=q,F=G,O=$),E=yo(F,D),p=E*Br,E&&(b=Math.cos(E),C=Math.sin(E),q=D*b+F*C,G=V*b+H*C,F=F*b-D*C,H=H*b-V*C,D=q,V=G),g&&Math.abs(g)+Math.abs(p)>359.9&&(g=p=0,m=180-m),h=cn(Math.sqrt(D*D+F*F+O*O)),x=cn(Math.sqrt(H*H+We*We)),E=yo(V,H),_=Math.abs(E)>2e-4?E*Br:0,v=ke?1/(ke<0?-ke:ke):0),n.svg&&(q=e.getAttribute("transform"),n.forceCSS=e.setAttribute("transform","")||!l_(gi(e,sn)),q&&e.setAttribute("transform",q))),Math.abs(_)>90&&Math.abs(_)<270&&(r?(h*=-1,_+=p<=0?180:-180,p+=p<=0?180:-180):(x*=-1,_+=_<=0?180:-180)),t=t||n.uncache,n.x=u-((n.xPercent=u&&(!t&&n.xPercent||(Math.round(e.offsetWidth/2)===Math.round(-u)?-50:0)))?e.offsetWidth*n.xPercent/100:0)+o,n.y=f-((n.yPercent=f&&(!t&&n.yPercent||(Math.round(e.offsetHeight/2)===Math.round(-f)?-50:0)))?e.offsetHeight*n.yPercent/100:0)+o,n.z=d+o,n.scaleX=cn(h),n.scaleY=cn(x),n.rotation=cn(p)+a,n.rotationX=cn(g)+a,n.rotationY=cn(m)+a,n.skewX=_+a,n.skewY=A+a,n.transformPerspective=v+o,(n.zOrigin=parseFloat(c.split(" ")[2])||!t&&n.zOrigin||0)&&(s[ii]=Uc(c)),n.xOffset=n.yOffset=0,n.force3D=_i.force3D,n.renderTransform=n.svg?v3:s_?c_:_3,n.uncache=0,n},Uc=function(e){return(e=e.split(" "))[0]+" "+e[1]},df=function(e,t,n){var s=Pn(t);return cn(parseFloat(t)+parseFloat(dr(e,"x",n+"px",s)))+s},_3=function(e,t){t.z="0px",t.rotationY=t.rotationX="0deg",t.force3D=0,c_(e,t)},Ir="0deg",va="0px",Dr=") ",c_=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.z,c=n.rotation,u=n.rotationY,f=n.rotationX,d=n.skewX,h=n.skewY,x=n.scaleX,p=n.scaleY,g=n.transformPerspective,m=n.force3D,_=n.target,A=n.zOrigin,v="",S=m==="auto"&&e&&e!==1||m===!0;if(A&&(f!==Ir||u!==Ir)){var y=parseFloat(u)*Ho,M=Math.sin(y),E=Math.cos(y),b;y=parseFloat(f)*Ho,b=Math.cos(y),o=df(_,o,M*b*-A),a=df(_,a,-Math.sin(y)*-A),l=df(_,l,E*b*-A+A)}g!==va&&(v+="perspective("+g+Dr),(s||r)&&(v+="translate("+s+"%, "+r+"%) "),(S||o!==va||a!==va||l!==va)&&(v+=l!==va||S?"translate3d("+o+", "+a+", "+l+") ":"translate("+o+", "+a+Dr),c!==Ir&&(v+="rotate("+c+Dr),u!==Ir&&(v+="rotateY("+u+Dr),f!==Ir&&(v+="rotateX("+f+Dr),(d!==Ir||h!==Ir)&&(v+="skew("+d+", "+h+Dr),(x!==1||p!==1)&&(v+="scale("+x+", "+p+Dr),_.style[sn]=v||"translate(0, 0)"},v3=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.rotation,c=n.skewX,u=n.skewY,f=n.scaleX,d=n.scaleY,h=n.target,x=n.xOrigin,p=n.yOrigin,g=n.xOffset,m=n.yOffset,_=n.forceCSS,A=parseFloat(o),v=parseFloat(a),S,y,M,E,b;l=parseFloat(l),c=parseFloat(c),u=parseFloat(u),u&&(u=parseFloat(u),c+=u,l+=u),l||c?(l*=Ho,c*=Ho,S=Math.cos(l)*f,y=Math.sin(l)*f,M=Math.sin(l-c)*-d,E=Math.cos(l-c)*d,c&&(u*=Ho,b=Math.tan(c-u),b=Math.sqrt(1+b*b),M*=b,E*=b,u&&(b=Math.tan(u),b=Math.sqrt(1+b*b),S*=b,y*=b)),S=cn(S),y=cn(y),M=cn(M),E=cn(E)):(S=f,E=d,y=M=0),(A&&!~(o+"").indexOf("px")||v&&!~(a+"").indexOf("px"))&&(A=dr(h,"x",o,"px"),v=dr(h,"y",a,"px")),(x||p||g||m)&&(A=cn(A+x-(x*S+p*M)+g),v=cn(v+p-(x*y+p*E)+m)),(s||r)&&(b=h.getBBox(),A=cn(A+s/100*b.width),v=cn(v+r/100*b.height)),b="matrix("+S+","+y+","+M+","+E+","+A+","+v+")",h.setAttribute("transform",b),_&&(h.style[sn]=b)},A3=function(e,t,n,s,r){var o=360,a=yn(r),l=parseFloat(r)*(a&&~r.indexOf("rad")?Br:1),c=l-s,u=s+c+"deg",f,d;return a&&(f=r.split("_")[1],f==="short"&&(c%=o,c!==c%(o/2)&&(c+=c<0?o:-o)),f==="cw"&&c<0?c=(c+o*f0)%o-~~(c/o)*o:f==="ccw"&&c>0&&(c=(c-o*f0)%o-~~(c/o)*o)),e._pt=d=new ni(e._pt,t,n,s,c,n3),d.e=u,d.u="deg",e._props.push(n),d},_0=function(e,t){for(var n in t)e[n]=t[n];return e},S3=function(e,t,n){var s=_0({},n._gsap),r="perspective,force3D,transformOrigin,svgOrigin",o=n.style,a,l,c,u,f,d,h,x;s.svg?(c=n.getAttribute("transform"),n.setAttribute("transform",""),o[sn]=t,a=cl(n,1),fr(n,sn),n.setAttribute("transform",c)):(c=getComputedStyle(n)[sn],o[sn]=t,a=cl(n,1),o[sn]=c);for(l in Os)c=s[l],u=a[l],c!==u&&r.indexOf(l)<0&&(h=Pn(c),x=Pn(u),f=h!==x?dr(n,l,c,x):parseFloat(c),d=parseFloat(u),e._pt=new ni(e._pt,a,l,f,d-f,Cd),e._pt.u=x||0,e._props.push(l));_0(a,s)};ti("padding,margin,Width,Radius",function(i,e){var t="Top",n="Right",s="Bottom",r="Left",o=(e<3?[t,n,s,r]:[t+r,t+n,s+n,s+r]).map(function(a){return e<2?i+a:"border"+a+i});Bc[e>1?"border"+i:i]=function(a,l,c,u,f){var d,h;if(arguments.length<4)return d=o.map(function(x){return Ts(a,x,c)}),h=d.join(" "),h.split(d[0]).length===5?d[0]:h;d=(u+"").split(" "),h={},o.forEach(function(x,p){return h[x]=d[p]=d[p]||d[(p-1)/2|0]}),a.init(l,h,f)}});var u_={name:"css",register:Ed,targetTest:function(e){return e.style&&e.nodeType},init:function(e,t,n,s,r){var o=this._props,a=e.style,l=n.vars.startAt,c,u,f,d,h,x,p,g,m,_,A,v,S,y,M,E,b;kh||Ed(),this.styles=this.styles||i_(e),E=this.styles.props,this.tween=n;for(p in t)if(p!=="autoRound"&&(u=t[p],!(fi[p]&&qx(p,t,n,s,e,r)))){if(h=typeof u,x=Bc[p],h==="function"&&(u=u.call(n,s,e,r),h=typeof u),h==="string"&&~u.indexOf("random(")&&(u=rl(u)),x)x(this,e,p,u,n)&&(M=1);else if(p.substr(0,2)==="--")c=(getComputedStyle(e).getPropertyValue(p)+"").trim(),u+="",or.lastIndex=0,or.test(c)||(g=Pn(c),m=Pn(u),m?g!==m&&(c=dr(e,p,c,m)+m):g&&(u+=g)),this.add(a,"setProperty",c,u,s,r,0,0,p),o.push(p),E.push(p,0,a[p]);else if(h!=="undefined"){if(l&&p in l?(c=typeof l[p]=="function"?l[p].call(n,s,e,r):l[p],yn(c)&&~c.indexOf("random(")&&(c=rl(c)),Pn(c+"")||c==="auto"||(c+=_i.units[p]||Pn(Ts(e,p))||""),(c+"").charAt(1)==="="&&(c=Ts(e,p))):c=Ts(e,p),d=parseFloat(c),_=h==="string"&&u.charAt(1)==="="&&u.substr(0,2),_&&(u=u.substr(2)),f=parseFloat(u),p in ts&&(p==="autoAlpha"&&(d===1&&Ts(e,"visibility")==="hidden"&&f&&(d=0),E.push("visibility",0,a.visibility),tr(this,a,"visibility",d?"inherit":"hidden",f?"inherit":"hidden",!f)),p!=="scale"&&p!=="transform"&&(p=ts[p],~p.indexOf(",")&&(p=p.split(",")[0]))),A=p in Os,A){if(this.styles.save(p),b=u,h==="string"&&u.substring(0,6)==="var(--"){if(u=gi(e,u.substring(4,u.indexOf(")"))),u.substring(0,5)==="calc("){var C=e.style.perspective;e.style.perspective=u,u=gi(e,"perspective"),C?e.style.perspective=C:fr(e,"perspective")}f=parseFloat(u)}if(v||(S=e._gsap,S.renderTransform&&!t.parseTransform||cl(e,t.parseTransform),y=t.smoothOrigin!==!1&&S.smooth,v=this._pt=new ni(this._pt,a,sn,0,1,S.renderTransform,S,0,-1),v.dep=1),p==="scale")this._pt=new ni(this._pt,S,"scaleY",S.scaleY,(_?zo(S.scaleY,_+f):f)-S.scaleY||0,Cd),this._pt.u=0,o.push("scaleY",p),p+="X";else if(p==="transformOrigin"){E.push(ii,0,a[ii]),u=g3(u),S.svg?wd(e,u,0,y,0,this):(m=parseFloat(u.split(" ")[2])||0,m!==S.zOrigin&&tr(this,S,"zOrigin",S.zOrigin,m),tr(this,a,p,Uc(c),Uc(u)));continue}else if(p==="svgOrigin"){wd(e,u,1,y,0,this);continue}else if(p in a_){A3(this,S,p,d,_?zo(d,_+u):u);continue}else if(p==="smoothOrigin"){tr(this,S,"smooth",S.smooth,u);continue}else if(p==="force3D"){S[p]=u;continue}else if(p==="transform"){S3(this,u,e);continue}}else p in a||(p=ia(p)||p);if(A||(f||f===0)&&(d||d===0)&&!t3.test(u)&&p in a)g=(c+"").substr((d+"").length),f||(f=0),m=Pn(u)||(p in _i.units?_i.units[p]:g),g!==m&&(d=dr(e,p,c,m)),this._pt=new ni(this._pt,A?S:a,p,d,(_?zo(d,_+f):f)-d,!A&&(m==="px"||p==="zIndex")&&t.autoRound!==!1?r3:Cd),this._pt.u=m||0,A&&b!==u?(this._pt.b=c,this._pt.e=b,this._pt.r=s3):g!==m&&m!=="%"&&(this._pt.b=c,this._pt.r=i3);else if(p in a)m3.call(this,e,p,c,_?_+u:u);else if(p in e)this.add(e,p,c||e[p],_?_+u:u,s,r);else if(p!=="parseTransform"){Ih(p,u);continue}A||(p in a?E.push(p,0,a[p]):typeof e[p]=="function"?E.push(p,2,e[p]()):E.push(p,1,c||e[p])),o.push(p)}}M&&Zx(this)},render:function(e,t){if(t.tween._time||!Hh())for(var n=t._pt;n;)n.r(e,n.d),n=n._next;else t.styles.revert()},get:Ts,aliases:ts,getSetter:function(e,t,n){var s=ts[t];return s&&s.indexOf(",")<0&&(t=s),t in Os&&t!==ii&&(e._gsap.x||Ts(e,"x"))?n&&u0===n?t==="scale"?c3:l3:(u0=n||{})&&(t==="scale"?u3:f3):e.style&&!Eh(e.style[t])?o3:~t.indexOf("-")?a3:Nh(e,t)},core:{_removeProperty:fr,_getMatrix:Gh}};si.utils.checkPrefix=ia;si.core.getStyleSaver=i_;(function(i,e,t,n){var s=ti(i+","+e+","+t,function(r){Os[r]=1});ti(e,function(r){_i.units[r]="deg",a_[r]=1}),ts[s[13]]=i+","+e,ti(n,function(r){var o=r.split(":");ts[o[1]]=s[o[0]]})})("x,y,z,scale,scaleX,scaleY,xPercent,yPercent","rotation,rotationX,rotationY,skewX,skewY","transform,transformOrigin,svgOrigin,force3D,smoothOrigin,transformPerspective","0:translateX,1:translateY,2:translateZ,8:rotate,8:rotationZ,8:rotateZ,9:rotateX,10:rotateY");ti("x,y,z,top,right,bottom,left,width,height,fontSize,padding,margin,perspective",function(i){_i.units[i]="px"});si.registerPlugin(u_);var Ur=si.registerPlugin(u_)||si;Ur.core.Tween;const f_=(i,e)=>{const t=i.__vccOpts||i;for(const[n,s]of e)t[n]=s;return t},y3={key:0,class:"bs-tabs"},b3={class:"bs-track-wrap"},M3=["onClick"],C3=["src"],T3={key:1,class:"bs-thumb bs-thumb--empty"},E3={key:2,class:"bs-tag"},w3=["onClick"],R3=["src"],I3={key:1,class:"bs-thumb bs-thumb--empty"},D3={class:"bs-time"},P3={__name:"BottomSelector",props:{models:{type:Array,default:()=>[]},activeModelId:{type:String,default:""},poses:{type:Array,default:()=>[]},activePoseId:{type:String,default:""},searchQuery:{type:String,default:""},getPosePresentationId:{type:Function,default:i=>i.id},hasModels:{type:Boolean,default:!1},hasPoses:{type:Boolean,default:!1}},emits:["selectModel","selectPose"],setup(i,{emit:e}){const t=i,n=e,s=yt("pose"),r=yt(null);let o=!1,a=0,l=0,c=!1;const u=Mi(()=>[...t.models].sort((A,v)=>{const S=new Date(A.createdAt||0).getTime();return new Date(v.createdAt||0).getTime()-S})),f=Mi(()=>t.hasModels);Da([()=>t.hasModels,()=>t.hasPoses],()=>{s.value==="model"&&!t.hasModels&&(s.value="pose")},{immediate:!0});function d(A){if(!A)return"";const v=new Date(A),S=String(v.getMonth()+1).padStart(2,"0"),y=String(v.getDate()).padStart(2,"0"),M=String(v.getHours()).padStart(2,"0"),E=String(v.getMinutes()).padStart(2,"0");return`${S}/${y} ${M}:${E}`}function h(A){c||A.id!==t.activeModelId&&n("selectModel",A)}function x(A){c||n("selectPose",A)}function p(){if(!r.value)return;const A=r.value.querySelector(".bs-item--active");A&&A.scrollIntoView({behavior:"smooth",block:"nearest",inline:"center"})}Da([()=>t.activePoseId,()=>t.activeModelId,s],()=>{K0(p)});function g(A){o=!0,c=!1,a=A.clientX||A.touches&&A.touches[0].clientX||0,l=r.value?r.value.scrollLeft:0}function m(A){if(!o||!r.value)return;const S=(A.clientX||A.touches&&A.touches[0].clientX||0)-a;Math.abs(S)>3&&(c=!0),r.value.scrollLeft=l-S}function _(){o=!1}return Xd(()=>{window.addEventListener("mouseup",_),window.addEventListener("touchend",_)}),qd(()=>{window.removeEventListener("mouseup",_),window.removeEventListener("touchend",_)}),(A,v)=>(jt(),on("div",{class:"bs-root",onMousedown:v[2]||(v[2]=Rt(()=>{},["stop"])),onTouchstart:v[3]||(v[3]=Rt(()=>{},["stop"])),onTouchmove:v[4]||(v[4]=Rt(()=>{},["stop"])),onTouchend:v[5]||(v[5]=Rt(()=>{},["stop"])),onWheel:v[6]||(v[6]=Rt(()=>{},["stop"]))},[f.value?(jt(),on("div",y3,[De("button",{class:$i(["bs-tab",{"bs-tab--active":s.value==="pose"}]),onClick:v[0]||(v[0]=S=>s.value="pose")},"空间",2),De("button",{class:$i(["bs-tab",{"bs-tab--active":s.value==="model"}]),onClick:v[1]||(v[1]=S=>s.value="model")},"时间",2)])):kn("",!0),De("div",b3,[De("div",{class:"bs-track",ref_key:"scrollRef",ref:r,onMousedown:g,onMousemove:m,onTouchstart:g,onTouchmove:m},[s.value==="pose"?(jt(!0),on(zi,{key:0},Jh(i.poses,S=>(jt(),on("div",{key:S.id,class:$i(["bs-item",{"bs-item--active":i.activePoseId===i.getPosePresentationId(S)}]),onClick:y=>x(S)},[S.image_url?(jt(),on("img",{key:0,src:S.image_url,class:"bs-thumb",draggable:"false"},null,8,C3)):(jt(),on("div",T3,[...v[7]||(v[7]=[De("span",null,"未命名",-1)])])),S.tag?(jt(),on("div",E3,Mn(S.tag),1)):kn("",!0)],10,M3))),128)):kn("",!0),s.value==="model"?(jt(!0),on(zi,{key:1},Jh(u.value,S=>(jt(),on("div",{key:S.id,class:$i(["bs-item",{"bs-item--active":S.id===i.activeModelId}]),onClick:y=>h(S)},[S.previewImg?(jt(),on("img",{key:0,src:S.previewImg,class:"bs-thumb",draggable:"false"},null,8,R3)):(jt(),on("div",I3,[...v[8]||(v[8]=[De("svg",{viewBox:"0 0 24 24",width:"18",height:"18",fill:"none",stroke:"currentColor","stroke-width":"1.5"},[De("path",{d:"M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"})],-1)])])),De("div",D3,Mn(d(S.createdAt)),1)],10,w3))),128)):kn("",!0)],544),v[9]||(v[9]=De("div",{class:"bs-fade bs-fade--left"},null,-1)),v[10]||(v[10]=De("div",{class:"bs-fade bs-fade--right"},null,-1))])],32))}},F3=f_(P3,[["__scopeId","data-v-9f718e10"]]),L3={class:"top-hud"},B3={class:"top-actions"},U3={class:"cinematic-head"},O3={class:"cinematic-head-actions"},N3={class:"cinematic-loop-toggle"},z3={class:"cinematic-actions"},k3=["disabled"],H3={class:"cinematic-progress-row"},V3=["value"],G3={class:"cinematic-progress-row"},W3={class:"cinematic-progress-row"},X3={class:"cinematic-focus-toggle"},q3={key:2,class:"fps-counter"},Y3={key:1,class:"loading-overlay"},Q3={key:2,class:"error-overlay"},K3={class:"error-card"},j3={class:"error-msg"},$3=["min","max"],Z3={class:"focal-row"},J3=["min","max"],e2={class:"focal-row"},t2={class:"focal-row"},n2=["src"],i2={key:0,class:"ref-info"},s2={class:"info-tag info-tag--accent"},r2={key:1,class:"ref-info"},o2={class:"info-tag"},a2={class:"info-tag"},l2={class:"info-tag"},Pr=380,v0=.065,A0=.0022,ic=.08,c2=1,S0=.0055,y0=.0042,u2=1,f2=.35,d2=1.2,h2=8,p2=.26,m2=.1,hf=18,Aa=6,g2=.45,x2={__name:"GaussianViewer",setup(i){const e=yt(null);yt(!1);const t=yt(!1),n=yt(!1),s=yt(!1),r={FREE:"free",ORBIT:"orbit"},o=yt(r.FREE),a=yt([]),l=yt(""),c=yt(""),u=yt(""),f=yt(""),d=yt({}),h=yt({x:0,y:0,z:0}),x=yt({x:0,y:0,z:0}),p=yt(""),g=yt(0),m=yt(!1),_=yt(0),A=yt(0),v=yt(null),S=yt(1),y=yt(0),M=yt(!0),E=yt(!1),b=yt(!1),C=yt(.68),D=yt(!0),F=yt(!1),O=yt([]),z=yt(""),V=Mi(()=>O.value.length>1||!G.value&&$.value.length>0),H=Mi(()=>O.value.length>1),q=Mi(()=>!G.value&&$.value.length>0),G=Mi(()=>o.value===r.ORBIT),$=Mi(()=>{if(!l.value.trim()){const L=a.value.filter(le=>le.tag);return L.length>0?L:a.value.slice(0,60)}const w=l.value.trim().toLowerCase();return a.value.filter(L=>L.tag&&L.tag.toLowerCase().includes(w))}),fe=()=>{$.value.length>0?Ce($.value[0]):alert("场景中没有找到符合该描述的视角哦~")};let Y,we;const ze=new B(0,1,0);let ke=null,We=!1,ne=!1,ue=!1,Se=0;const he={trajectory:null,phase:"main",startTimeMs:0,elapsedMs:0,lastNearestPoseIndex:-1,filteredSample:null},Ee=yt({x:0,y:0}),Ze=Mi(()=>a.value.length>=2),U=Mi(()=>E.value?"暂停运镜":b.value?"继续运镜":"开始运镜"),N=(w,L)=>!w||!L?null:2*Math.atan(L/2/w)*(180/Math.PI),K=(w,L)=>{if(!w||!L)return null;const le=w*Math.PI/180/2;return le<=0?null:L/2/Math.tan(le)},R=()=>{if(!Y||!Y.camera)return;const w=d.value.h||e.value?.clientHeight||window.innerHeight;if(_.value=Number(Y.camera.fov||0),w&&_.value>0&&_.value<179){const L=K(_.value,w);A.value=L?Number(L.toFixed(1)):0}},te=(w,L={})=>{if(!Y||!Y.camera)return;const le=d.value.h||e.value?.clientHeight||window.innerHeight;if(!le||!w)return;const _e=N(w,le);if(!_e||!Number.isFinite(_e))return;const Te=Y.camera,Oe=L.duration??0;if(Oe>0)Ur.to(Te,{fov:_e,duration:Oe,ease:L.ease||"power2.out",onUpdate:()=>{Te.updateProjectionMatrix();try{Y.update(),Y.render()}catch{}R()}});else{Te.fov=_e,Te.updateProjectionMatrix();try{Y.update(),Y.render()}catch{}R()}},oe=w=>Number.isFinite(w)?Math.min(se.value,Math.max(me.value,w)):null,pe=()=>{const w=Number(v.value||A.value||d.value.fl_y||Pr);return oe(w)},ie=w=>{if(!Y||!Y.camera||!Number.isFinite(w)||w<=0)return;const L=pe();if(!L)return;const le=oe(L*w);le&&(v.value=Number(le.toFixed(1)),te(le))},me=Mi(()=>{const w=Number(d.value.fl_y||0);return w>0?Math.max(50,Math.floor(w*.4)):50}),se=Mi(()=>{const w=Number(d.value.fl_y||0);return w>0?Math.max(500,Math.ceil(w*2.5)):3e3}),ve=()=>{m.value=!m.value,m.value&&!v.value&&(v.value=Number((A.value||d.value.fl_y||Pr).toFixed(1)))},I=()=>{const w=Number(v.value);!Number.isFinite(w)||w<=0||te(w)},T=()=>{const w=Number(d.value.fl_y||0);w&&(v.value=Number(w.toFixed(1)),te(w,{duration:.5,ease:"power2.inOut"}))},X=()=>{if(!Y||!Y.camera)return;const w=new Wi().setFromQuaternion(Y.camera.quaternion,"YXZ");h.value={x:(w.x*180/Math.PI).toFixed(1),y:(w.y*180/Math.PI).toFixed(1),z:(w.z*180/Math.PI).toFixed(1)},R()},re=()=>ot.uCenter.value.clone(),de=()=>{const w=Number(ot.uMaxRadius.value||0);return w>0?w:1},ee=()=>{Se&&(cancelAnimationFrame(Se),Se=0)},Ue=()=>{!Y||!Y.camera||(Ur.killTweensOf(Y.camera.position),Ur.killTweensOf(Y.camera.quaternion),Ur.killTweensOf(Y.camera))},ye=w=>{k(w)},Xe=w=>w?String(w.id||w.image_id||w.imageId||gr(w)||JSON.stringify(mr(w.matrix)||[])):"",k=(w,L={})=>{f.value=Xe(w),L.updateReference!==!1&&(c.value=w?.image_url||gr(w),u.value=w?.tag||"")},Z=(w={})=>{ee(),he.trajectory=null,he.startTimeMs=0,he.elapsedMs=0,he.lastNearestPoseIndex=-1,he.filteredSample=null,E.value=!1,b.value=!1,w.resetProgress!==!1&&(y.value=0)},xe=()=>{!E.value&&!b.value||Z({resetProgress:!1})},Re=(w,L)=>{if(!Array.isArray(w)||w.length<3)return w.map(Qe=>Qe.clone());const le=Sn.clamp(Number(L)||0,0,1),_e=Math.max(1,Math.round(1+le*3)),Te=.12+le*.26;let Oe=w.map(Qe=>Qe.clone());for(let Qe=0;Qe<_e;Qe+=1)Oe=Oe.map((Ve,Ie)=>{if(Ie===0||Ie===Oe.length-1)return Ve.clone();const et=Oe[Ie-1].clone().add(Oe[Ie].clone().multiplyScalar(2)).add(Oe[Ie+1]).multiplyScalar(.25);return Ve.clone().lerp(et,Te)});return Oe},Be=(w,L)=>{if(!Array.isArray(w)||w.length<3)return w.slice();const le=Sn.clamp(Number(L)||0,0,1),_e=Math.max(1,Math.round(1+le*2)),Te=.1+le*.28;let Oe=w.slice();for(let Qe=0;Qe<_e;Qe+=1)Oe=Oe.map((Ve,Ie)=>{if(Ie===0||Ie===Oe.length-1)return Ve;const et=(Oe[Ie-1]+Oe[Ie]*2+Oe[Ie+1])/4;return Sn.lerp(Ve,et,Te)});return Oe},Fe=(w,L)=>{if(L.clone().sub(w).lengthSq()<1e-8)return new Vt;const _e=new st().lookAt(w,L,ze);return new Vt().setFromRotationMatrix(_e)},je=(w,L,le)=>{const _e=be(L);let Te=w.clone().add(_e);return le&&_e.lengthSq()<1e-8&&(Te=le.clone()),Fe(w,Te)},W=w=>{if(!Array.isArray(w)||w.length===0)return[];const L=[w[0].clone().normalize()];for(let le=1;le<w.length;le+=1){const _e=w[le].clone().normalize();L[le-1].dot(_e)<0&&(_e.x*=-1,_e.y*=-1,_e.z*=-1,_e.w*=-1),L.push(_e)}return L},Le=(w,L)=>{if(!Array.isArray(w)||w.length<3)return W(w||[]);const le=Sn.clamp(Number(L)||0,0,1),_e=Math.max(1,Math.round(1+le*2)),Te=.16+le*.22;let Oe=W(w);for(let Qe=0;Qe<_e;Qe+=1){const nt=Oe.map((Ve,Ie)=>{if(Ie===0||Ie===Oe.length-1)return Ve.clone();const et=Oe[Ie-1].clone(),ut=Oe[Ie].clone(),Bt=Oe[Ie+1].clone(),pt=et.slerp(Bt,.5);return ut.slerp(pt,Te).normalize()});Oe=W(nt)}return Oe},Me=w=>{if(!w)return 0;const L=new B(0,1,0).applyQuaternion(w).normalize();return Math.abs(L.dot(ze))},be=w=>new B(0,0,-1).applyQuaternion(w).normalize(),Ae=w=>{if(!Array.isArray(w)||w.length<=hf)return w;const L=w.filter(Ve=>Me(Ve.quaternion)>=g2),le=L.length>=Aa?L:w.slice();if(le.length<=hf)return le;const _e=le.map((Ve,Ie,et)=>{const ut=et[Math.max(0,Ie-1)],Bt=et[Math.min(et.length-1,Ie+1)],pt=Me(Ve.quaternion),rn=Ie>0?Ve.position.distanceTo(ut.position):0,Ct=Ie<et.length-1?Ve.position.distanceTo(Bt.position):0,jn=(rn+Ct)*.5,wn=be(ut.quaternion),oi=be(Ve.quaternion),ds=be(Bt.quaternion),Sl=Ie>0&&Ie<et.length-1?Math.max(0,wn.dot(oi))*.5+Math.max(0,oi.dot(ds))*.5:1;return{frame:Ve,index:Ie,score:pt*2.2+Sl*1.4+Math.min(jn,1.5)*.4}}),Te=new Set([0,le.length-1]),Oe=Math.max(Aa,Math.min(hf,le.length)),Qe=_e.filter(({index:Ve})=>Te.has(Ve)).map(({frame:Ve})=>Ve),nt=_e.filter(({index:Ve})=>!Te.has(Ve)).sort((Ve,Ie)=>Ie.score-Ve.score);for(const Ve of nt){if(Qe.length>=Oe)break;Qe.push(Ve.frame)}if(Qe.sort((Ve,Ie)=>Ve.index-Ie.index),Qe.length<Aa){const Ve=Math.max(1,Math.floor(le.length/Aa));for(let Ie=0;Ie<le.length&&Qe.length<Aa;Ie+=Ve){const et=le[Ie];Qe.includes(et)||Qe.push(et)}Qe.sort((Ie,et)=>Ie.index-et.index)}return Qe},ge=({keyframes:w,positions:L,targets:le,focals:_e,durationMs:Te})=>{const Oe=de(),Qe=Le(L.map((ut,Bt)=>je(ut,w[Bt].quaternion,le[Bt])),C.value),nt=w.map((ut,Bt)=>({...ut,position:L[Bt],target:le[Bt],stabilizedQuaternion:Qe[Bt],fl_y:_e[Bt]||ut.fl_y})),Ve=new tm(nt.map(ut=>ut.position.clone()),!1,"centripetal"),Ie=new tm(nt.map(ut=>ut.target.clone()),!1,"centripetal"),et=[0];for(let ut=1;ut<nt.length;ut+=1){const Bt=nt[ut-1],pt=nt[ut];et.push(et[ut-1]+Bt.position.distanceTo(pt.position))}return{keyframes:nt,curve:Ve,lookCurve:Ie,cumulativeDistances:et,totalDistance:Math.max(et[et.length-1],1e-5),durationMs:Te,lookAheadDistance:Sn.clamp(Oe*(.4+C.value*.45),d2,h2)}},qe=(w,L)=>{if(!w?.keyframes||w.keyframes.length<2)return null;const le=de(),_e=w.keyframes[0],Te=w.keyframes[w.keyframes.length-1],Oe=Te.position.distanceTo(_e.position);if(Oe<1e-4)return null;const Qe=Math.max(le*.55,Oe*.22,.9),nt=Math.max(le*.18,Oe*.08,.35),Ve=Te.position.clone().sub(L).setY(0),Ie=_e.position.clone().sub(L).setY(0);Ve.lengthSq()<1e-6&&Ve.set(1,0,0),Ie.lengthSq()<1e-6&&Ie.set(-1,0,0),Ve.normalize().multiplyScalar(nt),Ie.normalize().multiplyScalar(nt);const et=L.clone().add(new B(0,le*.15,0)),ut=[Te.position.clone(),Te.position.clone().add(new B(0,Qe,0)).add(Ve),_e.position.clone().add(new B(0,Qe*.86,0)).add(Ie),_e.position.clone()],Bt=[Te.target.clone().lerp(et,.4),et.clone(),et.clone(),_e.target.clone().lerp(et,.28)],pt=Math.max(0,Number(Te.fl_y||_e.fl_y||d.value.fl_y||Pr)),rn=Sn.clamp(Oe*1350+1800,2400,6200)/S.value;return ge({keyframes:[{index:Te.index,pose:Te.pose,quaternion:Te.stabilizedQuaternion||Te.quaternion,fl_y:pt,h:Te.h},{index:Te.index,pose:Te.pose,quaternion:Te.stabilizedQuaternion||Te.quaternion,fl_y:pt,h:Te.h},{index:_e.index,pose:_e.pose,quaternion:_e.stabilizedQuaternion||_e.quaternion,fl_y:pt,h:_e.h},{index:_e.index,pose:_e.pose,quaternion:_e.stabilizedQuaternion||_e.quaternion,fl_y:pt,h:_e.h}],positions:ut,targets:Bt,focals:[pt,pt,pt,pt],durationMs:rn})},Je={FLY_IN:0,DIFFUSION:1,COLORING:2,FINISHED:3},rt={isLoaded:!1,lastFrameTime:0,phase:Je.FLY_IN,flyDuration:1.5,diffusionDuration:1,colorDuration:4},ot={uTime:{value:0},uCenter:{value:new B(0,0,0)},uGeoRadius:{value:0},uColorRadius:{value:0},uMaxRadius:{value:50},uParticleProgress:{value:0}},Si=w=>{if(!Y)return;const L=w.getSplatCount();w.updateMatrixWorld();let le=1/0,_e=1/0,Te=1/0,Oe=-1/0,Qe=-1/0,nt=-1/0;const Ve=new B,Ie=Math.max(1,Math.floor(L/1e3));for(let Ar=0;Ar<L;Ar+=Ie)w.getSplatCenter(Ar,Ve),Ve.applyMatrix4(w.matrixWorld),Ve.x<le&&(le=Ve.x),Ve.x>Oe&&(Oe=Ve.x),Ve.y<_e&&(_e=Ve.y),Ve.y>Qe&&(Qe=Ve.y),Ve.z<Te&&(Te=Ve.z),Ve.z>nt&&(nt=Ve.z);const et=(le+Oe)/2,ut=(_e+Qe)/2,Bt=(Te+nt)/2,pt=Math.max(Oe-le,Qe-_e,nt-Te);ot.uCenter.value.set(et,ut,Bt),ot.uMaxRadius.value=pt*.7;let rn=6e4;L<4e4?rn=L:L>1e6&&(rn=4e5);const Ct=Math.ceil(L/rn);let jn=pt/200*window.devicePixelRatio;jn<.5&&(jn=.5);const wn=pt*1;console.log(`[Adaptive] MaxDim: ${pt.toFixed(2)}, Particles: ~${Math.floor(L/Ct)}, Size: ${jn.toFixed(2)}`);const oi=new Kn,ds=[],Sl=[],Xh=[];for(let Ar=0;Ar<L;Ar+=Ct){w.getSplatCenter(Ar,Ve),Ve.applyMatrix4(w.matrixWorld),Sl.push(Ve.x,Ve.y,Ve.z);const ru=wn+Math.random()*(pt*.5),qh=Math.random()*Math.PI*2,ou=Math.acos(2*Math.random()-1),x_=et+ru*Math.sin(ou)*Math.cos(qh),__=ut+ru*Math.sin(ou)*Math.sin(qh),v_=Bt+ru*Math.cos(ou);ds.push(x_,__,v_),Xh.push(Math.random())}oi.setAttribute("position",new Ln(ds,3)),oi.setAttribute("aTarget",new Ln(Sl,3)),oi.setAttribute("aRandom",new Ln(Xh,1));const g_=new Yn({uniforms:{uProgress:ot.uParticleProgress,uSize:{value:jn},uColor:{value:new bt(.6,.6,.6)}},vertexShader:`
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
    `,transparent:!0,opacity:1,depthTest:!0,depthWrite:!1});we=new Iy(oi,g_),we.frustumCulled=!1,Y.threeScene.add(we)},ri=w=>{if(!w||!w.material)return;const L=w.material;L.uniforms=L.uniforms||{},L.uniforms.uGeoRadius=ot.uGeoRadius,L.uniforms.uColorRadius=ot.uColorRadius,L.uniforms.uMaxRadius=ot.uMaxRadius,L.uniforms.uCenter=ot.uCenter,L.vertexShader=`varying vec3 vWorldPosition;
`+L.vertexShader;const le=L.vertexShader.lastIndexOf("}");if(le!==-1){const Oe=`vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
`;L.vertexShader=L.vertexShader.substring(0,le)+Oe+"}"}const _e=`
    uniform float uGeoRadius;
    uniform float uColorRadius;
    uniform float uMaxRadius;
    uniform vec3 uCenter;
    varying vec3 vWorldPosition;
  `;L.fragmentShader=_e+L.fragmentShader;const Te=L.fragmentShader.lastIndexOf("}");if(Te!==-1){const Oe=L.fragmentShader.substring(0,Te),Qe=`
      float distFromCenter = distance(vWorldPosition, uCenter);
      
      if (distFromCenter > uGeoRadius) {
          discard;
      }
      if (distFromCenter > uColorRadius) {
          if (gl_FragColor.a < 0.8) discard; 
          gl_FragColor.a = 1.0; 
          gl_FragColor.rgb = vec3(0.6, 0.6, 0.6);
      } 
    `;L.fragmentShader=Oe+Qe+"}"}L.needsUpdate=!0},mr=w=>{if(!Array.isArray(w))return null;if(w.length===16){const L=w.map(le=>Number(le));return L.every(Number.isFinite)?L:null}if(w.length===4&&w.every(L=>Array.isArray(L)&&L.length===4)){const L=w.flat().map(le=>Number(le));return L.every(Number.isFinite)?L:null}return null},Jr=w=>{if(w==null)return"";let L=String(w).trim();if(!L)return"";try{L=decodeURIComponent(L)}catch{}L=L.replace(/\\/g,"/");const le=L.split("/");return(le[le.length-1]||"").trim().toLowerCase()},gl=w=>/^https?:\/\//i.test(String(w||"")),cs=w=>{if(typeof w!="string"||!w.trim())return"";const L=w.trim();return gl(L)&&window.location.origin.startsWith("http://127.0.0.1:")?`${window.location.origin}/proxy/${encodeURIComponent(L)}`:L},gr=w=>{if(!w)return"";const L=w.id||w.image_id||w.imageId;if(L)return Jr(L);const le=w.image_url;if(typeof le!="string"||le.length===0)return"";const _e=le.split("?")[0];return Jr(_e)},xl=w=>{if(!w||a.value.length===0)return null;const L=Jr(w.imageId);if(L){const Oe=a.value.find(Qe=>gr(Qe)===L);if(Oe)return Oe}const le=mr(w.matrix);if(!le)return null;let _e=null,Te=Number.POSITIVE_INFINITY;for(const Oe of a.value){const Qe=mr(Oe.matrix);if(!Qe)continue;let nt=0;for(let Ve=0;Ve<16;Ve+=1){const Ie=Math.abs(Qe[Ve]-le[Ve]);if(Ie>nt&&(nt=Ie),nt>=Te)break}nt<Te&&(Te=nt,_e=Oe)}return Te<=1e-4?_e:null},us=(w=!1)=>{if(!ke||We)return;if(!ke.imageId){const _e=aa();if(_e){We=!0,Ce(_e);return}}const L=xl(ke);if(L){We=!0,Ce(L);return}if(!w||ke.imageId&&!ue)return;const le=mr(ke.matrix);le&&(We=!0,Ce({matrix:le,image_url:ke.imageId||""}))},xr=w=>{const L=w?.image_url;return typeof L=="string"&&L.trim().length>0},aa=()=>{if(!Array.isArray(a.value)||a.value.length===0)return null;const w=a.value.find(le=>xr(le)&&le.tag);if(w)return w;const L=a.value.find(le=>xr(le));return L||a.value[0]||null},eo=()=>{if(!Array.isArray($.value)||$.value.length===0)return a.value;const w=$.value.filter(le=>typeof le?.tag=="string"&&le.tag.trim().length>0);if(w.length>=2)return w.slice(0,12);if($.value.length>=2)return $.value.slice(0,12);const L=a.value.filter(le=>typeof le?.tag=="string"&&le.tag.trim().length>0);return L.length>=2?L.slice(0,12):a.value.slice(0,12)},to=()=>{if(ke||We||ne)return;const w=aa();w&&(ne=!0,Ce(w))},la=w=>{if(!Y||!Y.camera)return null;const L=mr(w?.matrix);if(!L)return null;const le=Y.getSplatMesh(),_e=new st().fromArray(L),Te=new st;le?(le.updateMatrixWorld(),Te.copy(le.matrixWorld).multiply(_e)):Te.copy(_e);const Oe=new B,Qe=new Vt,nt=new B;return Te.decompose(Oe,Qe,nt),{position:Oe,quaternion:Qe,fl_y:Number(w?.fl_y||d.value.fl_y||0),h:Number(w?.h||d.value.h||0)}},_l=()=>{const w=eo();if(!Y||!Array.isArray(w)||w.length<2)return null;const L=re(),le=w.map((Ct,jn)=>{const wn=la(Ct);if(!wn)return null;const oi=je(wn.position,wn.quaternion,L);return{index:jn,pose:Ct,position:wn.position,quaternion:oi,fl_y:wn.fl_y,h:wn.h}}).filter(Boolean);if(le.length<2)return null;const _e=[le[0]];for(let Ct=1;Ct<le.length;Ct+=1){const jn=_e[_e.length-1],wn=le[Ct],oi=jn.position.distanceToSquared(wn.position)<1e-6,ds=Math.abs(jn.quaternion.dot(wn.quaternion))>.999999;oi&&ds||_e.push(wn)}if(_e.length<2)return null;const Te=Ae(_e);if(Te.length<2)return null;const Oe=Te,Qe=Oe.map(Ct=>Ct.position.clone()),nt=Re(Qe,C.value),Ve=Be(Oe.map(Ct=>Ct.fl_y||0),C.value),Ie=Oe.map((Ct,jn)=>{const wn=new B(0,0,-1).applyQuaternion(Ct.quaternion).normalize(),oi=Math.max(.8,Qe[jn].distanceTo(L)),ds=Qe[jn].clone().add(wn.multiplyScalar(Math.max(2.2,oi*.9)));return D.value?ds.lerp(L,Sn.clamp(.48+C.value*.26,0,.9)):ds}),et=Re(Ie,C.value);let ut=0;for(let Ct=1;Ct<nt.length;Ct+=1)ut+=nt[Ct-1].distanceTo(nt[Ct]);const Bt=Oe.length-1,pt=Sn.clamp(ut*1600+Bt*260,7e3,42e3)/S.value,rn=ge({keyframes:Oe,positions:nt,targets:et,focals:Ve,durationMs:pt});return{...rn,worldCenter:L.clone(),loopBridge:qe(rn,L)}},ca=(w,L)=>{if(!w)return null;const le=Sn.clamp(L,0,1),_e=w.totalDistance*le;let Te=w.keyframes.length-2;for(let Ct=0;Ct<w.cumulativeDistances.length-1;Ct+=1)if(_e<=w.cumulativeDistances[Ct+1]){Te=Ct;break}const Oe=w.cumulativeDistances[Te],Qe=w.cumulativeDistances[Te+1],nt=Math.max(Qe-Oe,1e-5),Ve=Sn.smootherstep((_e-Oe)/nt,0,1),Ie=w.keyframes[Te],et=w.keyframes[Te+1],ut=w.curve.getPointAt(le),Bt=Ie.stabilizedQuaternion.clone().slerp(et.stabilizedQuaternion,Ve).normalize(),pt=Ie.target.clone().lerp(et.target,Ve);return{position:ut,quaternion:Bt,target:pt,fl_y:Ie.fl_y&&et.fl_y?Sn.lerp(Ie.fl_y,et.fl_y,Ve):Ie.fl_y||et.fl_y||0,h:Ie.h||et.h||d.value.h||0,nearestPoseIndex:Ve<.5?Ie.index:et.index}},ua=w=>{if(!w||!Y||!Y.camera)return;const L=Sn.lerp(p2,m2,C.value);he.filteredSample?(he.filteredSample.position.lerp(w.position,L),he.filteredSample.quaternion.slerp(w.quaternion,L).normalize(),w.fl_y&&(he.filteredSample.fl_y=Sn.lerp(he.filteredSample.fl_y||w.fl_y,w.fl_y,L*.85)),w.h&&(he.filteredSample.h=w.h)):he.filteredSample={position:w.position.clone(),quaternion:w.quaternion.clone(),fl_y:Number(w.fl_y||0),h:Number(w.h||d.value.h||0)};const le=Y.camera;if(le.position.copy(he.filteredSample.position),le.quaternion.copy(he.filteredSample.quaternion),he.filteredSample.fl_y&&he.filteredSample.h?(d.value.h=he.filteredSample.h,v.value=Number(he.filteredSample.fl_y.toFixed(1)),te(he.filteredSample.fl_y)):Mt(),w.nearestPoseIndex!==he.lastNearestPoseIndex){he.lastNearestPoseIndex=w.nearestPoseIndex;const _e=a.value[w.nearestPoseIndex];_e&&k(_e,{updateReference:!1})}},vl=w=>{if(!he.trajectory||!Y||!Y.camera){Z({resetProgress:!1});return}const L=he.phase==="loop-bridge"&&he.trajectory.loopBridge?he.trajectory.loopBridge:he.trajectory,le=Math.max(L.durationMs,1),_e=Math.max(0,w-he.startTimeMs);he.elapsedMs=_e;let Te=_e/le;if(Te>=1&&(he.phase==="loop-bridge"?(he.startTimeMs=w,he.elapsedMs=0,he.phase="main",he.lastNearestPoseIndex=-1,Te=0):M.value&&he.trajectory.loopBridge?(he.startTimeMs=w,he.elapsedMs=0,he.phase="loop-bridge",he.lastNearestPoseIndex=-1,Te=0):M.value?(he.startTimeMs=w,he.elapsedMs=0,he.phase="main",he.lastNearestPoseIndex=-1,Te=0):Te=1),y.value=he.phase==="main"?Te:1,ua(ca(L,Te)),!M.value&&he.phase==="main"&&Te>=1){Z({resetProgress:!1}),y.value=1;return}Se=requestAnimationFrame(vl)},iu=(w={})=>{if(!Y||!Y.camera)return;const L=_l();L&&(Ue(),ee(),he.trajectory=L,he.phase="main",he.filteredSample=null,he.elapsedMs=w.resume?he.elapsedMs:0,he.startTimeMs=performance.now()-he.elapsedMs,he.lastNearestPoseIndex=-1,E.value=!0,b.value=!1,w.resume||(y.value=0,ua(ca(L,0))),Se=requestAnimationFrame(vl))},su=()=>{E.value&&(ee(),he.elapsedMs=Math.max(0,performance.now()-he.startTimeMs),E.value=!1,b.value=!0)},P=()=>{if(Ze.value){if(E.value){su();return}iu({resume:b.value})}},Q=()=>{Ze.value&&(F.value=!F.value)},ae=()=>{const w=_l();w&&(he.trajectory=w,he.phase="main",he.lastNearestPoseIndex=-1,ua(ca(w,y.value)),E.value?(he.elapsedMs=w.durationMs*y.value,he.startTimeMs=performance.now()-he.elapsedMs):b.value&&(he.elapsedMs=w.durationMs*y.value))},ce=()=>{S.value=Number(Sn.clamp(Number(S.value)||1,.25,3).toFixed(2)),(E.value||b.value)&&ae()},j=()=>{C.value=Number(Sn.clamp(Number(C.value)||.68,0,1).toFixed(2)),(E.value||b.value)&&ae()},Ce=(w,L={})=>{if(!Y||!Y.camera)return;const le=la(w);if(!le){console.warn("[Viewer] Skip invalid pose matrix:",w);return}L.keepCinematic||xe();const _e=Y.camera,Te=le.position,Oe=le.quaternion;ye(w);const Qe=le.fl_y,nt=le.h;Qe&&nt&&(d.value.h=nt,v.value=Number(Qe.toFixed(1)),te(Qe,{duration:1.5,ease:"power3.inOut"})),_e.near>.001&&(_e.near=.001,_e.updateProjectionMatrix()),t.value=!1,Y.controls&&(Y.controls.enabled=!1);const Ve=_e.position.clone(),Ie=_e.quaternion.clone(),et={t:0};Ue(),Ur.killTweensOf(et),Ur.to(et,{t:1,duration:1.5,ease:"power3.inOut",onUpdate:()=>{_e.position.lerpVectors(Ve,Te,et.t),_e.quaternion.slerpQuaternions(Ie,Oe,et.t)},onComplete:()=>{const ut=new Wi().setFromQuaternion(_e.quaternion,"YXZ");x.value={x:(ut.x*180/Math.PI).toFixed(1),y:(ut.y*180/Math.PI).toFixed(1),z:(ut.z*180/Math.PI).toFixed(1)},Ee.value={x:0,y:0},wt.roll=0,X(),Y.controls&&(Y.controls.enabled=!0)}})},Ge=()=>{const w=/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);return{rootElement:e.value,cameraUp:[0,1,0],initialCameraPosition:[0,0,5],initialCameraLookAt:[0,0,0],useBuiltInControls:!1,gpuAcceleratedSort:!1,webXRMode:Lr.None,sharedMemoryForWorkers:!1,antialiased:!w}};let Ye="/models/scene_auto_sync.ply",He="/models/webgl_poses_with_tags.json",it=!1;const at=()=>{const w=new URLSearchParams(window.location.search),L=w.get("payload");if(L)try{const nt=JSON.parse(decodeURIComponent(L));return{ply:nt.ply||null,poses:nt.poses||null,matrix:nt.matrix||null,imageId:nt.imageId||null}}catch(nt){console.warn("[Viewer] 无法解析 payload 查询参数:",nt)}const le=w.get("ply"),_e=w.get("poses"),Te=w.get("matrix"),Oe=w.get("imageId");let Qe=null;if(Te)try{Qe=JSON.parse(decodeURIComponent(Te))}catch(nt){console.warn("[Viewer] 无法解析 matrix 查询参数:",nt)}return le||_e||Qe?{ply:le||null,poses:_e||null,matrix:Qe,imageId:Oe||null}:null},$e=async(w,L,le)=>{if(!n.value){n.value=!0,Z(),a.value=[],f.value="",c.value="",u.value="",d.value={},w&&(Ye=w),L&&(He=L);try{if(Y){try{Y.renderer&&Y.renderer.setAnimationLoop(null)}catch(Ie){console.warn("[Viewer] renderer cleanup:",Ie)}try{Y.dispose&&await Y.dispose()}catch(Ie){console.warn("[Viewer] dispose:",Ie)}Y=null}if(e.value)for(;e.value.firstChild;)e.value.removeChild(e.value.firstChild);rt.isLoaded=!1,rt.phase=Je.FLY_IN,ot.uParticleProgress.value=0,ot.uGeoRadius.value=0,ot.uColorRadius.value=0,ke=null,We=!1,ne=!1,ue=!1;const _e=Ge();Y=new To(_e),window.viewer=Y,v.value=Pr,console.log(`[Viewer] 加载模型: ${Ye}`),await Y.addSplatScene(Ye,{showLoadingUI:!0,progressiveLoad:!1,rotation:[0,0,0,1]}),n.value=!1,window.BrainDanceChannel&&window.BrainDanceChannel.postMessage(JSON.stringify({status:"success",msg:"模型加载完成"})),console.log(`[Viewer] 加载位姿: ${He}`),fetch(He).then(Ie=>Ie.json()).then(Ie=>{ue=!0,Ie.frames?(d.value={w:Ie.w,h:Ie.h,fl_x:Ie.fl_x,fl_y:Ie.fl_y},v.value=Number((Ie.fl_y||0).toFixed(1)),a.value=Ie.frames.map(et=>{let ut=et.image_url;if(ut&&!ut.startsWith("http")&&He.startsWith("http")){const Bt=He.substring(0,He.lastIndexOf("/"));let pt=ut;const rn=pt.indexOf("images/");rn!==-1?pt=pt.substring(rn):pt.startsWith("/models/")?pt=pt.substring(8):pt.startsWith("/")&&(pt=pt.substring(1)),ut=`${Bt}/${pt}`}return ut=cs(ut),{id:et.id,matrix:et.matrix,image_url:ut,tag:et.tag,fl_x:et.fl_x,fl_y:et.fl_y,w:et.w||Ie.w,h:et.h||Ie.h}}),d.value.fl_y&&d.value.h?te(d.value.fl_y):te(Pr),us(!0),to()):(a.value=Ie,te(Pr),us(!0),to())}).catch(Ie=>{ue=!0,console.error("加载位姿失败:",Ie),te(Pr),us(!0)});const Te=Y.getSplatMesh();Te.visible=!1,setTimeout(()=>{Te&&(Si(Te),ri(Te),le&&(le.matrix||le.imageId)?(ke={matrix:le.matrix||null,imageId:le.imageId||null},us(ue),setTimeout(()=>{us(!1)},50),le.imageId||setTimeout(()=>{us(!0)},800)):setTimeout(()=>{to()},80),rt.lastFrameTime=Date.now(),rt.startTime=Date.now(),rt.isLoaded=!0)},200);let Oe=performance.now();const Qe=1e3/120;let nt=0,Ve=performance.now();Y.renderer.setAnimationLoop(()=>{const Ie=performance.now(),et=Ie-Oe;if(et<Qe||(Oe=Ie-et%Qe,Y.update(),Y.render(),nt++,Ie-Ve>=1e3&&(g.value=nt,nt=0,Ve=Ie),!rt.isLoaded||rt.phase===Je.FINISHED))return;const ut=Date.now(),Bt=(ut-rt.lastFrameTime)/1e3||.016;if(rt.lastFrameTime=ut,rt.phase===Je.FLY_IN){const pt=1/rt.flyDuration;let rn=ot.uParticleProgress.value+Bt*pt;if(rn>=1.2){rn=1.2;const Ct=Y.getSplatMesh();Ct&&(Ct.visible=!0),rt.phase=Je.DIFFUSION,rt.diffuseTime=0}ot.uParticleProgress.value=rn}else if(rt.phase===Je.DIFFUSION){rt.diffuseTime+=Bt;const pt=Math.min(rt.diffuseTime/rt.diffusionDuration,1),rn=ot.uMaxRadius.value;ot.uGeoRadius.value=pt*(rn*1.5),we&&we.material&&(we.material.opacity=1-pt),pt>=1&&(we&&(we.visible=!1),ot.uGeoRadius.value=99999,rt.phase=Je.COLORING,rt.colorStartTime=ut)}else if(rt.phase===Je.COLORING){const pt=(ut-rt.colorStartTime)/1e3,rn=ot.uMaxRadius.value,Ct=pt/rt.colorDuration;ot.uColorRadius.value=Ct*(rn*1.5),Ct>=1&&(rt.phase=Je.FINISHED,ot.uColorRadius.value=99999)}}),fs()}catch(_e){console.error("error:",_e),p.value=_e&&(_e.message||String(_e))||"模型加载失败，请检查模型 URL 是否正确可访问"}finally{n.value=!1}}},vt=()=>{!Y||!Y.controls||(Y.controls.dispose(),Y.controls=null)},Mt=()=>{if(!(!Y||!Y.camera)){Y.camera.updateProjectionMatrix(),R(),X();try{Y.update(),Y.render()}catch{}}},Xt=(w,L)=>{!Y||!Y.camera||(Y.camera.rotateOnWorldAxis(ze,-w),Y.camera.rotateX(-L),Mt())},Qt=w=>{!Y||!Y.camera||!Number.isFinite(w)||(Y.camera.rotateZ(w*u2),Mt())},Dt=w=>{if(!Y||!Y.camera||!Number.isFinite(w)||w<=0)return;const L=Math.max(.3,Y.camera.position.distanceTo(re())),le=Sn.clamp((1-w)*L*f2,-L*.25,L*.25);Y.camera.translateZ(le),Mt()},tt=(w,L)=>Math.atan2(L.clientY-w.clientY,L.clientX-w.clientX),qt=w=>w>Math.PI?w-Math.PI*2:w<-Math.PI?w+Math.PI*2:w,St=()=>{Y&&vt()},Un=()=>{Y&&(vt(),wt.roll=0)},fs=()=>{Y&&(G.value?Un():St())},En=w=>{w!==r.FREE&&w!==r.ORBIT||o.value!==w&&(o.value=w,fs(),G.value)},_r=()=>{const w=window.location.hostname==="localhost"||window.location.hostname==="127.0.0.1",L=window.location.protocol==="https:";s.value=w||L},dt=yt(!1),ht={x:0,y:0},mt={active:!1,distance:0},wt={active:!1,angle:0,roll:0},ln=(w,L)=>{const le=w.clientX-L.clientX,_e=w.clientY-L.clientY;return Math.hypot(le,_e)},vr=w=>{if(xe(),G.value){if(w.button!==0)return;dt.value=!0,mt.active=!1,wt.active=!1,ht.x=w.clientX,ht.y=w.clientY;return}dt.value=!0,mt.active=!1,ht.x=w.clientX,ht.y=w.clientY},Bi=w=>{if(G.value){if(!dt.value||!Y||!Y.camera)return;const Te=w.clientX-ht.x,Oe=w.clientY-ht.y;Xt(Te*S0,Oe*y0),ht.x=w.clientX,ht.y=w.clientY;return}if(!dt.value||!Y||!Y.camera)return;const L=w.clientX-ht.x,_e=(w.clientY-ht.y)*v0;Y.camera.rotateX(_e*Math.PI/180),Y.camera.translateX(-L*A0),Y.camera.updateProjectionMatrix(),X(),ht.x=w.clientX,ht.y=w.clientY},Al=()=>{if(G.value){dt.value=!1,mt.active=!1,wt.active=!1;return}dt.value=!1,mt.active=!1},d_=w=>{if(!Y||!Y.camera)return;if(xe(),G.value){const le=w.deltaY<0?1+ic:1/(1+ic);Dt(le);return}const L=w.deltaY<0?1+ic:1/(1+ic);ie(L)},h_=w=>{if(xe(),G.value){if(w.touches.length>=2){dt.value=!1,mt.active=!0,mt.distance=ln(w.touches[0],w.touches[1]),wt.active=!0,wt.angle=tt(w.touches[0],w.touches[1]);return}mt.active=!1,wt.active=!1,w.touches.length===1&&(dt.value=!0,ht.x=w.touches[0].clientX,ht.y=w.touches[0].clientY);return}if(w.touches.length>=2){dt.value=!1,mt.active=!0,mt.distance=ln(w.touches[0],w.touches[1]);return}mt.active=!1,w.touches.length===1&&(dt.value=!0,ht.x=w.touches[0].clientX,ht.y=w.touches[0].clientY)},p_=w=>{if(G.value){if(!Y||!Y.camera||w.touches.length===0)return;if(w.touches.length>=2){const Qe=ln(w.touches[0],w.touches[1]),nt=tt(w.touches[0],w.touches[1]);mt.active&&mt.distance>0&&Qe>0&&Dt(Qe/mt.distance),wt.active&&Qt(qt(nt-wt.angle)),mt.active=!0,mt.distance=Qe,wt.active=!0,wt.angle=nt,dt.value=!1;return}if(!dt.value)return;const Te=w.touches[0].clientX-ht.x,Oe=w.touches[0].clientY-ht.y;Xt(Te*S0,Oe*y0),ht.x=w.touches[0].clientX,ht.y=w.touches[0].clientY;return}if(!Y||!Y.camera||w.touches.length===0)return;if(w.touches.length>=2){const Te=ln(w.touches[0],w.touches[1]);if(mt.active&&mt.distance>0&&Te>0){const Oe=Te/mt.distance;ie(1+(Oe-1)*c2)}mt.active=!0,mt.distance=Te,dt.value=!1;return}if(!dt.value)return;const L=w.touches[0].clientX-ht.x,_e=(w.touches[0].clientY-ht.y)*v0;Ee.value.x+=_e,Y.camera.rotateX(_e*Math.PI/180),Y.camera.translateX(-L*A0),Y.camera.updateProjectionMatrix(),X(),ht.x=w.touches[0].clientX,ht.y=w.touches[0].clientY},Wh=w=>{if(G.value){if(w.touches.length>=2){mt.active=!0,mt.distance=ln(w.touches[0],w.touches[1]),wt.active=!0,wt.angle=tt(w.touches[0],w.touches[1]),dt.value=!1;return}mt.active=!1,mt.distance=0,wt.active=!1,wt.angle=0,dt.value=!1,w.touches.length===1&&(ht.x=w.touches[0].clientX,ht.y=w.touches[0].clientY,dt.value=!0);return}if(w.touches.length>=2){mt.active=!0,mt.distance=ln(w.touches[0],w.touches[1]),dt.value=!1;return}mt.active=!1,mt.distance=0,dt.value=!1,w.touches.length===1&&(ht.x=w.touches[0].clientX,ht.y=w.touches[0].clientY,dt.value=!0)};function m_(w){z.value=w.id,window.BrainDanceChannel?window.BrainDanceChannel.postMessage(JSON.stringify({action:"switchModel",modelId:w.id,ply:w.ply||"",poses:w.poses||""})):(n.value=!1,Z(),$e(w.ply||null,w.poses||null,null))}return Xd(()=>{if(e.value){if(_r(),window.setModelListForTimePeeling=(w,L)=>{console.log("[Flutter->WebGL] 收到模型列表:",w,"当前模型:",L),Array.isArray(w)&&(O.value=w,L?z.value=L:w.length>0&&!z.value&&(z.value=w[0].id||""))},window.loadModelFromFlutter=w=>{console.log("[Flutter->WebGL] 收到加载请求:",w),typeof w=="string"?$e(w,null,null):typeof w=="object"&&w!==null?$e(w.ply||null,w.poses||null,{matrix:w.matrix||null,imageId:w.imageId||null}):$e(null,null,null)},window.BrainDanceChannel)window.BrainDanceChannel.postMessage(JSON.stringify({status:"ready"}));else{const w=at();w&&!it?(it=!0,$e(w.ply,w.poses,{matrix:w.matrix||null,imageId:w.imageId||null})):$e(null,null)}window.addEventListener("mousedown",vr),window.addEventListener("mousemove",Bi),window.addEventListener("mouseup",Al)}}),qd(async()=>{if(window.removeEventListener("mousedown",vr),window.removeEventListener("mousemove",Bi),window.removeEventListener("mouseup",Al),Z(),Y){try{Y.renderer&&Y.renderer.setAnimationLoop(null)}catch{}try{await Y.dispose()}catch{}Y=null}}),(w,L)=>(jt(),on("div",{class:"app-container",onMousedown:vr,onMousemove:Bi,onMouseup:Al,onWheel:Rt(d_,["prevent"]),onMouseleave:Al,onTouchstart:h_,onTouchmove:Rt(p_,["prevent"]),onTouchend:Wh,onTouchcancel:Wh},[De("div",{ref_key:"containerRef",ref:e,class:"viewer-container"},null,512),L[53]||(L[53]=De("div",{class:"viewer-vignette"},null,-1)),V.value?(jt(),yg(F3,{key:0,models:O.value,activeModelId:z.value,poses:$.value,activePoseId:f.value,searchQuery:l.value,getPosePresentationId:Xe,hasModels:H.value,hasPoses:q.value,onSelectModel:m_,onSelectPose:Ce},null,8,["models","activeModelId","poses","activePoseId","searchQuery","hasModels","hasPoses"])):kn("",!0),De("div",L3,[De("div",{class:"search-panel archive-card",onMousedown:L[1]||(L[1]=Rt(()=>{},["stop"])),onTouchstart:L[2]||(L[2]=Rt(()=>{},["stop"])),onTouchmove:L[3]||(L[3]=Rt(()=>{},["stop"])),onTouchend:L[4]||(L[4]=Rt(()=>{},["stop"]))},[Sr(De("input",{type:"text","onUpdate:modelValue":L[0]||(L[0]=le=>l.value=le),onKeyup:zA(fe,["enter"]),placeholder:"例如：门口、桌面左侧、正面特写",class:"search-input"},null,544),[[ha,l.value]]),De("button",{onClick:fe,class:"archive-btn archive-btn--solid search-btn"},"检索视角")],32),De("div",B3,[De("div",{class:"view-mode-switch archive-card",onMousedown:L[7]||(L[7]=Rt(()=>{},["stop"])),onTouchstart:L[8]||(L[8]=Rt(()=>{},["stop"])),onTouchmove:L[9]||(L[9]=Rt(()=>{},["stop"])),onTouchend:L[10]||(L[10]=Rt(()=>{},["stop"]))},[De("button",{class:$i(["mode-chip",{active:o.value===r.FREE}]),onClick:L[5]||(L[5]=le=>En(r.FREE))}," 自由模式 ",2),De("button",{class:$i(["mode-chip",{active:o.value===r.ORBIT}]),onClick:L[6]||(L[6]=le=>En(r.ORBIT))}," Orbit 模式 ",2)],32),De("button",{class:"archive-btn archive-btn--ghost focal-settings-toggle",onClick:ve,onMousedown:L[11]||(L[11]=Rt(()=>{},["stop"])),onTouchstart:L[12]||(L[12]=Rt(()=>{},["stop"])),onTouchend:L[13]||(L[13]=Rt(()=>{},["stop"]))},Mn(m.value?"收起焦距":"焦距设置"),33),Ze.value?(jt(),on("button",{key:0,class:$i(["cinematic-trigger archive-btn archive-btn--ghost",{active:F.value}]),onClick:Q,onMousedown:L[14]||(L[14]=Rt(()=>{},["stop"])),onTouchstart:L[15]||(L[15]=Rt(()=>{},["stop"])),onTouchend:L[16]||(L[16]=Rt(()=>{},["stop"]))},[...L[37]||(L[37]=[De("span",{class:"cinematic-trigger-icon","aria-hidden":"true"},[De("svg",{viewBox:"0 0 24 24",focusable:"false"},[De("path",{d:"M4 7.5a1.5 1.5 0 0 1 1.5-1.5h7A1.5 1.5 0 0 1 14 7.5v9a1.5 1.5 0 0 1-1.5 1.5h-7A1.5 1.5 0 0 1 4 16.5v-9Zm11 2.1 4.83-2.76A.75.75 0 0 1 21 7.5v9a.75.75 0 0 1-1.17.66L15 14.4V9.6Z"})])],-1),De("span",null,"运镜",-1)])],34)):kn("",!0),Ze.value&&F.value?(jt(),on("div",{key:1,class:"cinematic-panel archive-card",onMousedown:L[23]||(L[23]=Rt(()=>{},["stop"])),onTouchstart:L[24]||(L[24]=Rt(()=>{},["stop"])),onTouchmove:L[25]||(L[25]=Rt(()=>{},["stop"])),onTouchend:L[26]||(L[26]=Rt(()=>{},["stop"])),onTouchcancel:L[27]||(L[27]=Rt(()=>{},["stop"]))},[De("div",U3,[L[39]||(L[39]=De("div",null,[De("div",{class:"eyebrow"},"Camera Move"),De("div",{class:"cinematic-title"},"自动运镜")],-1)),De("div",O3,[De("label",N3,[Sr(De("input",{type:"checkbox","onUpdate:modelValue":L[17]||(L[17]=le=>M.value=le)},null,512),[[Mp,M.value]]),L[38]||(L[38]=De("span",null,"循环",-1))]),De("button",{class:"cinematic-close",onClick:L[18]||(L[18]=le=>F.value=!1),"aria-label":"收起运镜面板"}," × ")])]),De("div",z3,[De("button",{class:"archive-btn archive-btn--solid cinematic-primary",onClick:P},Mn(U.value),1),De("button",{class:"archive-btn archive-btn--ghost cinematic-secondary",onClick:L[19]||(L[19]=le=>Z()),disabled:!E.value&&!b.value&&y.value===0}," 停止 ",8,k3)]),De("div",H3,[L[40]||(L[40]=De("span",null,"进度",-1)),De("span",null,Mn(Math.round(y.value*100))+"%",1)]),De("input",{class:"cinematic-progress",type:"range",value:y.value*100,min:"0",max:"100",step:"1",disabled:""},null,8,V3),De("div",G3,[L[41]||(L[41]=De("span",null,"速度",-1)),De("span",null,Mn(S.value.toFixed(2))+"x",1)]),Sr(De("input",{class:"cinematic-speed",type:"range","onUpdate:modelValue":L[20]||(L[20]=le=>S.value=le),min:"0.25",max:"3",step:"0.05",onInput:ce},null,544),[[ha,S.value,void 0,{number:!0}]]),De("div",W3,[L[42]||(L[42]=De("span",null,"平滑",-1)),De("span",null,Mn(Math.round(C.value*100))+"%",1)]),Sr(De("input",{class:"cinematic-speed",type:"range","onUpdate:modelValue":L[21]||(L[21]=le=>C.value=le),min:"0",max:"1",step:"0.05",onInput:j},null,544),[[ha,C.value,void 0,{number:!0}]]),De("label",X3,[Sr(De("input",{type:"checkbox","onUpdate:modelValue":L[22]||(L[22]=le=>D.value=le),onChange:j},null,544),[[Mp,D.value]]),L[43]||(L[43]=De("span",null,"主体锁定",-1))])],32)):kn("",!0),g.value>0?(jt(),on("div",q3,"FPS "+Mn(g.value),1)):kn("",!0)])]),n.value?(jt(),on("div",Y3,[...L[44]||(L[44]=[De("div",{class:"loading-card"},[De("div",{class:"loading-dot"}),De("div",{class:"loading-title"},"场景正在展开"),De("div",{class:"loading-copy"},"模型与参考镜头正在同步到工作台。")],-1)])])):kn("",!0),p.value?(jt(),on("div",Q3,[De("div",K3,[L[45]||(L[45]=De("div",{class:"eyebrow"},"Load Failed",-1)),L[46]||(L[46]=De("div",{class:"error-title"},"模型未能正常打开",-1)),De("div",j3,Mn(p.value),1),De("button",{class:"archive-btn archive-btn--solid",onClick:L[28]||(L[28]=le=>$e(vf(Ye),vf(He),null))}," 重新载入 ")])])):kn("",!0),kn("",!0),m.value?(jt(),on("div",{key:4,class:"focal-settings-panel",onMousedown:L[31]||(L[31]=Rt(()=>{},["stop"])),onTouchstart:L[32]||(L[32]=Rt(()=>{},["stop"])),onTouchmove:L[33]||(L[33]=Rt(()=>{},["stop"])),onTouchend:L[34]||(L[34]=Rt(()=>{},["stop"])),onTouchcancel:L[35]||(L[35]=Rt(()=>{},["stop"]))},[L[48]||(L[48]=De("div",{class:"eyebrow"},"Lens Control",-1)),L[49]||(L[49]=De("div",{class:"focal-title"},"镜头焦距",-1)),Sr(De("input",{type:"range","onUpdate:modelValue":L[29]||(L[29]=le=>v.value=le),min:me.value,max:se.value,step:"1",onInput:I},null,40,$3),[[ha,v.value,void 0,{number:!0}]]),De("div",Z3,[Sr(De("input",{class:"focal-number-input",type:"number","onUpdate:modelValue":L[30]||(L[30]=le=>v.value=le),min:me.value,max:se.value,step:"1",onChange:I},null,40,J3),[[ha,v.value,void 0,{number:!0}]]),L[47]||(L[47]=De("span",null,"px",-1))]),De("div",e2,[De("span",null,"当前 FOV: "+Mn(_.value.toFixed(1))+"°",1)]),De("div",t2,[De("span",null,"当前焦距: "+Mn(A.value.toFixed(1))+" px",1)]),De("button",{class:"archive-btn archive-btn--solid focal-reset-btn",onClick:T},"恢复拍摄焦距")],32)):kn("",!0),c.value?(jt(),on("div",{key:5,class:"reference-overlay",onClick:L[36]||(L[36]=le=>{c.value="",u.value=""})},[L[50]||(L[50]=De("div",{class:"eyebrow"},"Reference Still",-1)),L[51]||(L[51]=De("div",{class:"ref-title"},"参考原图",-1)),De("img",{src:c.value,class:"ref-img"},null,8,n2),u.value?(jt(),on("div",i2,[De("span",s2,Mn(u.value),1)])):kn("",!0),d.value.fl_y?(jt(),on("div",r2,[De("span",o2,"焦距: "+Mn(d.value.fl_y.toFixed(1))+" px",1),De("span",a2,"FOV: "+Mn((2*Math.atan(d.value.h/(2*d.value.fl_y))*(180/Math.PI)).toFixed(1))+"°",1),De("span",l2,"分辨率: "+Mn(d.value.w)+"x"+Mn(d.value.h),1)])):kn("",!0),L[52]||(L[52]=De("div",{class:"ref-hint"},"点击关闭对比",-1))])):kn("",!0)],32))}},_2=f_(x2,[["__scopeId","data-v-a132b8d5"]]),v2={__name:"App",setup(i){return(e,t)=>(jt(),on("main",null,[is(_2)]))}};VA(v2).mount("#app");
