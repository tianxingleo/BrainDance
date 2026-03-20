(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))n(s);new MutationObserver(s=>{for(const r of s)if(r.type==="childList")for(const o of r.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&n(o)}).observe(document,{childList:!0,subtree:!0});function t(s){const r={};return s.integrity&&(r.integrity=s.integrity),s.referrerPolicy&&(r.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?r.credentials="include":s.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function n(s){if(s.ep)return;s.ep=!0;const r=t(s);fetch(s.href,r)}})();function nd(i){const e=Object.create(null);for(const t of i.split(","))e[t]=1;return t=>t in e}const Et={},oo=[],Ni=()=>{},Gm=()=>!1,hc=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&(i.charCodeAt(2)>122||i.charCodeAt(2)<97),id=i=>i.startsWith("onUpdate:"),yn=Object.assign,sd=(i,e)=>{const t=i.indexOf(e);t>-1&&i.splice(t,1)},Bx=Object.prototype.hasOwnProperty,vt=(i,e)=>Bx.call(i,e),Je=Array.isArray,ao=i=>Va(i)==="[object Map]",pc=i=>Va(i)==="[object Set]",mh=i=>Va(i)==="[object Date]",it=i=>typeof i=="function",$t=i=>typeof i=="string",ki=i=>typeof i=="symbol",Mt=i=>i!==null&&typeof i=="object",Wm=i=>(Mt(i)||it(i))&&it(i.then)&&it(i.catch),Xm=Object.prototype.toString,Va=i=>Xm.call(i),Ux=i=>Va(i).slice(8,-1),qm=i=>Va(i)==="[object Object]",rd=i=>$t(i)&&i!=="NaN"&&i[0]!=="-"&&""+parseInt(i,10)===i,sa=nd(",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"),mc=i=>{const e=Object.create(null);return(t=>e[t]||(e[t]=i(t)))},Ox=/-\w/g,Ws=mc(i=>i.replace(Ox,e=>e.slice(1).toUpperCase())),Nx=/\B([A-Z])/g,js=mc(i=>i.replace(Nx,"-$1").toLowerCase()),Ym=mc(i=>i.charAt(0).toUpperCase()+i.slice(1)),Lc=mc(i=>i?`on${Ym(i)}`:""),zs=(i,e)=>!Object.is(i,e),Ll=(i,...e)=>{for(let t=0;t<i.length;t++)i[t](...e)},Qm=(i,e,t,n=!1)=>{Object.defineProperty(i,e,{configurable:!0,enumerable:!1,writable:n,value:t})},od=i=>{const e=parseFloat(i);return isNaN(e)?i:e};let gh;const gc=()=>gh||(gh=typeof globalThis<"u"?globalThis:typeof self<"u"?self:typeof window<"u"?window:typeof global<"u"?global:{});function ad(i){if(Je(i)){const e={};for(let t=0;t<i.length;t++){const n=i[t],s=$t(n)?Vx(n):ad(n);if(s)for(const r in s)e[r]=s[r]}return e}else if($t(i)||Mt(i))return i}const zx=/;(?![^(]*\))/g,kx=/:([^]+)/,Hx=/\/\*[^]*?\*\//g;function Vx(i){const e={};return i.replace(Hx,"").split(zx).forEach(t=>{if(t){const n=t.split(kx);n.length>1&&(e[n[0].trim()]=n[1].trim())}}),e}function lo(i){let e="";if($t(i))e=i;else if(Je(i))for(let t=0;t<i.length;t++){const n=lo(i[t]);n&&(e+=n+" ")}else if(Mt(i))for(const t in i)i[t]&&(e+=t+" ");return e.trim()}const Gx="itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly",Wx=nd(Gx);function Km(i){return!!i||i===""}function Xx(i,e){if(i.length!==e.length)return!1;let t=!0;for(let n=0;t&&n<i.length;n++)t=Ga(i[n],e[n]);return t}function Ga(i,e){if(i===e)return!0;let t=mh(i),n=mh(e);if(t||n)return t&&n?i.getTime()===e.getTime():!1;if(t=ki(i),n=ki(e),t||n)return i===e;if(t=Je(i),n=Je(e),t||n)return t&&n?Xx(i,e):!1;if(t=Mt(i),n=Mt(e),t||n){if(!t||!n)return!1;const s=Object.keys(i).length,r=Object.keys(e).length;if(s!==r)return!1;for(const o in i){const a=i.hasOwnProperty(o),l=e.hasOwnProperty(o);if(a&&!l||!a&&l||!Ga(i[o],e[o]))return!1}}return String(i)===String(e)}function jm(i,e){return i.findIndex(t=>Ga(t,e))}const $m=i=>!!(i&&i.__v_isRef===!0),mn=i=>$t(i)?i:i==null?"":Je(i)||Mt(i)&&(i.toString===Xm||!it(i.toString))?$m(i)?mn(i.value):JSON.stringify(i,Zm,2):String(i),Zm=(i,e)=>$m(e)?Zm(i,e.value):ao(e)?{[`Map(${e.size})`]:[...e.entries()].reduce((t,[n,s],r)=>(t[Bc(n,r)+" =>"]=s,t),{})}:pc(e)?{[`Set(${e.size})`]:[...e.values()].map(t=>Bc(t))}:ki(e)?Bc(e):Mt(e)&&!Je(e)&&!qm(e)?String(e):e,Bc=(i,e="")=>{var t;return ki(i)?`Symbol(${(t=i.description)!=null?t:e})`:i};let Nn;class qx{constructor(e=!1){this.detached=e,this._active=!0,this._on=0,this.effects=[],this.cleanups=[],this._isPaused=!1,this.__v_skip=!0,this.parent=Nn,!e&&Nn&&(this.index=(Nn.scopes||(Nn.scopes=[])).push(this)-1)}get active(){return this._active}pause(){if(this._active){this._isPaused=!0;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].pause();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].pause()}}resume(){if(this._active&&this._isPaused){this._isPaused=!1;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].resume();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].resume()}}run(e){if(this._active){const t=Nn;try{return Nn=this,e()}finally{Nn=t}}}on(){++this._on===1&&(this.prevScope=Nn,Nn=this)}off(){this._on>0&&--this._on===0&&(Nn=this.prevScope,this.prevScope=void 0)}stop(e){if(this._active){this._active=!1;let t,n;for(t=0,n=this.effects.length;t<n;t++)this.effects[t].stop();for(this.effects.length=0,t=0,n=this.cleanups.length;t<n;t++)this.cleanups[t]();if(this.cleanups.length=0,this.scopes){for(t=0,n=this.scopes.length;t<n;t++)this.scopes[t].stop(!0);this.scopes.length=0}if(!this.detached&&this.parent&&!e){const s=this.parent.scopes.pop();s&&s!==this&&(this.parent.scopes[this.index]=s,s.index=this.index)}this.parent=void 0}}}function Yx(){return Nn}let Rt;const Uc=new WeakSet;class Jm{constructor(e){this.fn=e,this.deps=void 0,this.depsTail=void 0,this.flags=5,this.next=void 0,this.cleanup=void 0,this.scheduler=void 0,Nn&&Nn.active&&Nn.effects.push(this)}pause(){this.flags|=64}resume(){this.flags&64&&(this.flags&=-65,Uc.has(this)&&(Uc.delete(this),this.trigger()))}notify(){this.flags&2&&!(this.flags&32)||this.flags&8||t0(this)}run(){if(!(this.flags&1))return this.fn();this.flags|=2,xh(this),n0(this);const e=Rt,t=Mi;Rt=this,Mi=!0;try{return this.fn()}finally{i0(this),Rt=e,Mi=t,this.flags&=-3}}stop(){if(this.flags&1){for(let e=this.deps;e;e=e.nextDep)ud(e);this.deps=this.depsTail=void 0,xh(this),this.onStop&&this.onStop(),this.flags&=-2}}trigger(){this.flags&64?Uc.add(this):this.scheduler?this.scheduler():this.runIfDirty()}runIfDirty(){ku(this)&&this.run()}get dirty(){return ku(this)}}let e0=0,ra,oa;function t0(i,e=!1){if(i.flags|=8,e){i.next=oa,oa=i;return}i.next=ra,ra=i}function ld(){e0++}function cd(){if(--e0>0)return;if(oa){let e=oa;for(oa=void 0;e;){const t=e.next;e.next=void 0,e.flags&=-9,e=t}}let i;for(;ra;){let e=ra;for(ra=void 0;e;){const t=e.next;if(e.next=void 0,e.flags&=-9,e.flags&1)try{e.trigger()}catch(n){i||(i=n)}e=t}}if(i)throw i}function n0(i){for(let e=i.deps;e;e=e.nextDep)e.version=-1,e.prevActiveLink=e.dep.activeLink,e.dep.activeLink=e}function i0(i){let e,t=i.depsTail,n=t;for(;n;){const s=n.prevDep;n.version===-1?(n===t&&(t=s),ud(n),Qx(n)):e=n,n.dep.activeLink=n.prevActiveLink,n.prevActiveLink=void 0,n=s}i.deps=e,i.depsTail=t}function ku(i){for(let e=i.deps;e;e=e.nextDep)if(e.dep.version!==e.version||e.dep.computed&&(s0(e.dep.computed)||e.dep.version!==e.version))return!0;return!!i._dirty}function s0(i){if(i.flags&4&&!(i.flags&16)||(i.flags&=-17,i.globalVersion===Sa)||(i.globalVersion=Sa,!i.isSSR&&i.flags&128&&(!i.deps&&!i._dirty||!ku(i))))return;i.flags|=2;const e=i.dep,t=Rt,n=Mi;Rt=i,Mi=!0;try{n0(i);const s=i.fn(i._value);(e.version===0||zs(s,i._value))&&(i.flags|=128,i._value=s,e.version++)}catch(s){throw e.version++,s}finally{Rt=t,Mi=n,i0(i),i.flags&=-3}}function ud(i,e=!1){const{dep:t,prevSub:n,nextSub:s}=i;if(n&&(n.nextSub=s,i.prevSub=void 0),s&&(s.prevSub=n,i.nextSub=void 0),t.subs===i&&(t.subs=n,!n&&t.computed)){t.computed.flags&=-5;for(let r=t.computed.deps;r;r=r.nextDep)ud(r,!0)}!e&&!--t.sc&&t.map&&t.map.delete(t.key)}function Qx(i){const{prevDep:e,nextDep:t}=i;e&&(e.nextDep=t,i.prevDep=void 0),t&&(t.prevDep=e,i.nextDep=void 0)}let Mi=!0;const r0=[];function ps(){r0.push(Mi),Mi=!1}function ms(){const i=r0.pop();Mi=i===void 0?!0:i}function xh(i){const{cleanup:e}=i;if(i.cleanup=void 0,e){const t=Rt;Rt=void 0;try{e()}finally{Rt=t}}}let Sa=0;class Kx{constructor(e,t){this.sub=e,this.dep=t,this.version=t.version,this.nextDep=this.prevDep=this.nextSub=this.prevSub=this.prevActiveLink=void 0}}class fd{constructor(e){this.computed=e,this.version=0,this.activeLink=void 0,this.subs=void 0,this.map=void 0,this.key=void 0,this.sc=0,this.__v_skip=!0}track(e){if(!Rt||!Mi||Rt===this.computed)return;let t=this.activeLink;if(t===void 0||t.sub!==Rt)t=this.activeLink=new Kx(Rt,this),Rt.deps?(t.prevDep=Rt.depsTail,Rt.depsTail.nextDep=t,Rt.depsTail=t):Rt.deps=Rt.depsTail=t,o0(t);else if(t.version===-1&&(t.version=this.version,t.nextDep)){const n=t.nextDep;n.prevDep=t.prevDep,t.prevDep&&(t.prevDep.nextDep=n),t.prevDep=Rt.depsTail,t.nextDep=void 0,Rt.depsTail.nextDep=t,Rt.depsTail=t,Rt.deps===t&&(Rt.deps=n)}return t}trigger(e){this.version++,Sa++,this.notify(e)}notify(e){ld();try{for(let t=this.subs;t;t=t.prevSub)t.sub.notify()&&t.sub.dep.notify()}finally{cd()}}}function o0(i){if(i.dep.sc++,i.sub.flags&4){const e=i.dep.computed;if(e&&!i.dep.subs){e.flags|=20;for(let n=e.deps;n;n=n.nextDep)o0(n)}const t=i.dep.subs;t!==i&&(i.prevSub=t,t&&(t.nextSub=i)),i.dep.subs=i}}const Hu=new WeakMap,Sr=Symbol(""),Vu=Symbol(""),ya=Symbol("");function xn(i,e,t){if(Mi&&Rt){let n=Hu.get(i);n||Hu.set(i,n=new Map);let s=n.get(t);s||(n.set(t,s=new fd),s.map=n,s.key=t),s.track()}}function ls(i,e,t,n,s,r){const o=Hu.get(i);if(!o){Sa++;return}const a=l=>{l&&l.trigger()};if(ld(),e==="clear")o.forEach(a);else{const l=Je(i),c=l&&rd(t);if(l&&t==="length"){const u=Number(n);o.forEach((f,d)=>{(d==="length"||d===ya||!ki(d)&&d>=u)&&a(f)})}else switch((t!==void 0||o.has(void 0))&&a(o.get(t)),c&&a(o.get(ya)),e){case"add":l?c&&a(o.get("length")):(a(o.get(Sr)),ao(i)&&a(o.get(Vu)));break;case"delete":l||(a(o.get(Sr)),ao(i)&&a(o.get(Vu)));break;case"set":ao(i)&&a(o.get(Sr));break}}cd()}function Fr(i){const e=_t(i);return e===i?e:(xn(e,"iterate",ya),gi(i)?e:e.map(Ci))}function xc(i){return xn(i=_t(i),"iterate",ya),i}function Rs(i,e){return gs(i)?So(yr(i)?Ci(e):e):Ci(e)}const jx={__proto__:null,[Symbol.iterator](){return Oc(this,Symbol.iterator,i=>Rs(this,i))},concat(...i){return Fr(this).concat(...i.map(e=>Je(e)?Fr(e):e))},entries(){return Oc(this,"entries",i=>(i[1]=Rs(this,i[1]),i))},every(i,e){return Yi(this,"every",i,e,void 0,arguments)},filter(i,e){return Yi(this,"filter",i,e,t=>t.map(n=>Rs(this,n)),arguments)},find(i,e){return Yi(this,"find",i,e,t=>Rs(this,t),arguments)},findIndex(i,e){return Yi(this,"findIndex",i,e,void 0,arguments)},findLast(i,e){return Yi(this,"findLast",i,e,t=>Rs(this,t),arguments)},findLastIndex(i,e){return Yi(this,"findLastIndex",i,e,void 0,arguments)},forEach(i,e){return Yi(this,"forEach",i,e,void 0,arguments)},includes(...i){return Nc(this,"includes",i)},indexOf(...i){return Nc(this,"indexOf",i)},join(i){return Fr(this).join(i)},lastIndexOf(...i){return Nc(this,"lastIndexOf",i)},map(i,e){return Yi(this,"map",i,e,void 0,arguments)},pop(){return Xo(this,"pop")},push(...i){return Xo(this,"push",i)},reduce(i,...e){return _h(this,"reduce",i,e)},reduceRight(i,...e){return _h(this,"reduceRight",i,e)},shift(){return Xo(this,"shift")},some(i,e){return Yi(this,"some",i,e,void 0,arguments)},splice(...i){return Xo(this,"splice",i)},toReversed(){return Fr(this).toReversed()},toSorted(i){return Fr(this).toSorted(i)},toSpliced(...i){return Fr(this).toSpliced(...i)},unshift(...i){return Xo(this,"unshift",i)},values(){return Oc(this,"values",i=>Rs(this,i))}};function Oc(i,e,t){const n=xc(i),s=n[e]();return n!==i&&!gi(i)&&(s._next=s.next,s.next=()=>{const r=s._next();return r.done||(r.value=t(r.value)),r}),s}const $x=Array.prototype;function Yi(i,e,t,n,s,r){const o=xc(i),a=o!==i&&!gi(i),l=o[e];if(l!==$x[e]){const f=l.apply(i,r);return a?Ci(f):f}let c=t;o!==i&&(a?c=function(f,d){return t.call(this,Rs(i,f),d,i)}:t.length>2&&(c=function(f,d){return t.call(this,f,d,i)}));const u=l.call(o,c,n);return a&&s?s(u):u}function _h(i,e,t,n){const s=xc(i);let r=t;return s!==i&&(gi(i)?t.length>3&&(r=function(o,a,l){return t.call(this,o,a,l,i)}):r=function(o,a,l){return t.call(this,o,Rs(i,a),l,i)}),s[e](r,...n)}function Nc(i,e,t){const n=_t(i);xn(n,"iterate",ya);const s=n[e](...t);return(s===-1||s===!1)&&md(t[0])?(t[0]=_t(t[0]),n[e](...t)):s}function Xo(i,e,t=[]){ps(),ld();const n=_t(i)[e].apply(i,t);return cd(),ms(),n}const Zx=nd("__proto__,__v_isRef,__isVue"),a0=new Set(Object.getOwnPropertyNames(Symbol).filter(i=>i!=="arguments"&&i!=="caller").map(i=>Symbol[i]).filter(ki));function Jx(i){ki(i)||(i=String(i));const e=_t(this);return xn(e,"has",i),e.hasOwnProperty(i)}class l0{constructor(e=!1,t=!1){this._isReadonly=e,this._isShallow=t}get(e,t,n){if(t==="__v_skip")return e.__v_skip;const s=this._isReadonly,r=this._isShallow;if(t==="__v_isReactive")return!s;if(t==="__v_isReadonly")return s;if(t==="__v_isShallow")return r;if(t==="__v_raw")return n===(s?r?c_:d0:r?f0:u0).get(e)||Object.getPrototypeOf(e)===Object.getPrototypeOf(n)?e:void 0;const o=Je(e);if(!s){let l;if(o&&(l=jx[t]))return l;if(t==="hasOwnProperty")return Jx}const a=Reflect.get(e,t,vn(e)?e:n);if((ki(t)?a0.has(t):Zx(t))||(s||xn(e,"get",t),r))return a;if(vn(a)){const l=o&&rd(t)?a:a.value;return s&&Mt(l)?Wu(l):l}return Mt(a)?s?Wu(a):hd(a):a}}class c0 extends l0{constructor(e=!1){super(!1,e)}set(e,t,n,s){let r=e[t];const o=Je(e)&&rd(t);if(!this._isShallow){const c=gs(r);if(!gi(n)&&!gs(n)&&(r=_t(r),n=_t(n)),!o&&vn(r)&&!vn(n))return c||(r.value=n),!0}const a=o?Number(t)<e.length:vt(e,t),l=Reflect.set(e,t,n,vn(e)?e:s);return e===_t(s)&&(a?zs(n,r)&&ls(e,"set",t,n):ls(e,"add",t,n)),l}deleteProperty(e,t){const n=vt(e,t);e[t];const s=Reflect.deleteProperty(e,t);return s&&n&&ls(e,"delete",t,void 0),s}has(e,t){const n=Reflect.has(e,t);return(!ki(t)||!a0.has(t))&&xn(e,"has",t),n}ownKeys(e){return xn(e,"iterate",Je(e)?"length":Sr),Reflect.ownKeys(e)}}class e_ extends l0{constructor(e=!1){super(!0,e)}set(e,t){return!0}deleteProperty(e,t){return!0}}const t_=new c0,n_=new e_,i_=new c0(!0);const Gu=i=>i,$a=i=>Reflect.getPrototypeOf(i);function s_(i,e,t){return function(...n){const s=this.__v_raw,r=_t(s),o=ao(r),a=i==="entries"||i===Symbol.iterator&&o,l=i==="keys"&&o,c=s[i](...n),u=t?Gu:e?So:Ci;return!e&&xn(r,"iterate",l?Vu:Sr),yn(Object.create(c),{next(){const{value:f,done:d}=c.next();return d?{value:f,done:d}:{value:a?[u(f[0]),u(f[1])]:u(f),done:d}}})}}function Za(i){return function(...e){return i==="delete"?!1:i==="clear"?void 0:this}}function r_(i,e){const t={get(s){const r=this.__v_raw,o=_t(r),a=_t(s);i||(zs(s,a)&&xn(o,"get",s),xn(o,"get",a));const{has:l}=$a(o),c=e?Gu:i?So:Ci;if(l.call(o,s))return c(r.get(s));if(l.call(o,a))return c(r.get(a));r!==o&&r.get(s)},get size(){const s=this.__v_raw;return!i&&xn(_t(s),"iterate",Sr),s.size},has(s){const r=this.__v_raw,o=_t(r),a=_t(s);return i||(zs(s,a)&&xn(o,"has",s),xn(o,"has",a)),s===a?r.has(s):r.has(s)||r.has(a)},forEach(s,r){const o=this,a=o.__v_raw,l=_t(a),c=e?Gu:i?So:Ci;return!i&&xn(l,"iterate",Sr),a.forEach((u,f)=>s.call(r,c(u),c(f),o))}};return yn(t,i?{add:Za("add"),set:Za("set"),delete:Za("delete"),clear:Za("clear")}:{add(s){!e&&!gi(s)&&!gs(s)&&(s=_t(s));const r=_t(this);return $a(r).has.call(r,s)||(r.add(s),ls(r,"add",s,s)),this},set(s,r){!e&&!gi(r)&&!gs(r)&&(r=_t(r));const o=_t(this),{has:a,get:l}=$a(o);let c=a.call(o,s);c||(s=_t(s),c=a.call(o,s));const u=l.call(o,s);return o.set(s,r),c?zs(r,u)&&ls(o,"set",s,r):ls(o,"add",s,r),this},delete(s){const r=_t(this),{has:o,get:a}=$a(r);let l=o.call(r,s);l||(s=_t(s),l=o.call(r,s)),a&&a.call(r,s);const c=r.delete(s);return l&&ls(r,"delete",s,void 0),c},clear(){const s=_t(this),r=s.size!==0,o=s.clear();return r&&ls(s,"clear",void 0,void 0),o}}),["keys","values","entries",Symbol.iterator].forEach(s=>{t[s]=s_(s,i,e)}),t}function dd(i,e){const t=r_(i,e);return(n,s,r)=>s==="__v_isReactive"?!i:s==="__v_isReadonly"?i:s==="__v_raw"?n:Reflect.get(vt(t,s)&&s in n?t:n,s,r)}const o_={get:dd(!1,!1)},a_={get:dd(!1,!0)},l_={get:dd(!0,!1)};const u0=new WeakMap,f0=new WeakMap,d0=new WeakMap,c_=new WeakMap;function u_(i){switch(i){case"Object":case"Array":return 1;case"Map":case"Set":case"WeakMap":case"WeakSet":return 2;default:return 0}}function f_(i){return i.__v_skip||!Object.isExtensible(i)?0:u_(Ux(i))}function hd(i){return gs(i)?i:pd(i,!1,t_,o_,u0)}function d_(i){return pd(i,!1,i_,a_,f0)}function Wu(i){return pd(i,!0,n_,l_,d0)}function pd(i,e,t,n,s){if(!Mt(i)||i.__v_raw&&!(e&&i.__v_isReactive))return i;const r=f_(i);if(r===0)return i;const o=s.get(i);if(o)return o;const a=new Proxy(i,r===2?n:t);return s.set(i,a),a}function yr(i){return gs(i)?yr(i.__v_raw):!!(i&&i.__v_isReactive)}function gs(i){return!!(i&&i.__v_isReadonly)}function gi(i){return!!(i&&i.__v_isShallow)}function md(i){return i?!!i.__v_raw:!1}function _t(i){const e=i&&i.__v_raw;return e?_t(e):i}function h_(i){return!vt(i,"__v_skip")&&Object.isExtensible(i)&&Qm(i,"__v_skip",!0),i}const Ci=i=>Mt(i)?hd(i):i,So=i=>Mt(i)?Wu(i):i;function vn(i){return i?i.__v_isRef===!0:!1}function Dt(i){return p_(i,!1)}function p_(i,e){return vn(i)?i:new m_(i,e)}class m_{constructor(e,t){this.dep=new fd,this.__v_isRef=!0,this.__v_isShallow=!1,this._rawValue=t?e:_t(e),this._value=t?e:Ci(e),this.__v_isShallow=t}get value(){return this.dep.track(),this._value}set value(e){const t=this._rawValue,n=this.__v_isShallow||gi(e)||gs(e);e=n?e:_t(e),zs(e,t)&&(this._rawValue=e,this._value=n?e:Ci(e),this.dep.trigger())}}function Xu(i){return vn(i)?i.value:i}const g_={get:(i,e,t)=>e==="__v_raw"?i:Xu(Reflect.get(i,e,t)),set:(i,e,t,n)=>{const s=i[e];return vn(s)&&!vn(t)?(s.value=t,!0):Reflect.set(i,e,t,n)}};function h0(i){return yr(i)?i:new Proxy(i,g_)}class x_{constructor(e,t,n){this.fn=e,this.setter=t,this._value=void 0,this.dep=new fd(this),this.__v_isRef=!0,this.deps=void 0,this.depsTail=void 0,this.flags=16,this.globalVersion=Sa-1,this.next=void 0,this.effect=this,this.__v_isReadonly=!t,this.isSSR=n}notify(){if(this.flags|=16,!(this.flags&8)&&Rt!==this)return t0(this,!0),!0}get value(){const e=this.dep.track();return s0(this),e&&(e.version=this.dep.version),this._value}set value(e){this.setter&&this.setter(e)}}function __(i,e,t=!1){let n,s;return it(i)?n=i:(n=i.get,s=i.set),new x_(n,s,t)}const Ja={},ql=new WeakMap;let ur;function v_(i,e=!1,t=ur){if(t){let n=ql.get(t);n||ql.set(t,n=[]),n.push(i)}}function A_(i,e,t=Et){const{immediate:n,deep:s,once:r,scheduler:o,augmentJob:a,call:l}=t,c=A=>s?A:gi(A)||s===!1||s===0?cs(A,1):cs(A);let u,f,d,h,x=!1,p=!1;if(vn(i)?(f=()=>i.value,x=gi(i)):yr(i)?(f=()=>c(i),x=!0):Je(i)?(p=!0,x=i.some(A=>yr(A)||gi(A)),f=()=>i.map(A=>{if(vn(A))return A.value;if(yr(A))return c(A);if(it(A))return l?l(A,2):A()})):it(i)?e?f=l?()=>l(i,2):i:f=()=>{if(d){ps();try{d()}finally{ms()}}const A=ur;ur=u;try{return l?l(i,3,[h]):i(h)}finally{ur=A}}:f=Ni,e&&s){const A=f,S=s===!0?1/0:s;f=()=>cs(A(),S)}const g=Yx(),m=()=>{u.stop(),g&&g.active&&sd(g.effects,u)};if(r&&e){const A=e;e=(...S)=>{A(...S),m()}}let _=p?new Array(i.length).fill(Ja):Ja;const v=A=>{if(!(!(u.flags&1)||!u.dirty&&!A))if(e){const S=u.run();if(s||x||(p?S.some((b,M)=>zs(b,_[M])):zs(S,_))){d&&d();const b=ur;ur=u;try{const M=[S,_===Ja?void 0:p&&_[0]===Ja?[]:_,h];_=S,l?l(e,3,M):e(...M)}finally{ur=b}}}else u.run()};return a&&a(v),u=new Jm(f),u.scheduler=o?()=>o(v,!1):v,h=A=>v_(A,!1,u),d=u.onStop=()=>{const A=ql.get(u);if(A){if(l)l(A,4);else for(const S of A)S();ql.delete(u)}},e?n?v(!0):_=u.run():o?o(v.bind(null,!0),!0):u.run(),m.pause=u.pause.bind(u),m.resume=u.resume.bind(u),m.stop=m,m}function cs(i,e=1/0,t){if(e<=0||!Mt(i)||i.__v_skip||(t=t||new Map,(t.get(i)||0)>=e))return i;if(t.set(i,e),e--,vn(i))cs(i.value,e,t);else if(Je(i))for(let n=0;n<i.length;n++)cs(i[n],e,t);else if(pc(i)||ao(i))i.forEach(n=>{cs(n,e,t)});else if(qm(i)){for(const n in i)cs(i[n],e,t);for(const n of Object.getOwnPropertySymbols(i))Object.prototype.propertyIsEnumerable.call(i,n)&&cs(i[n],e,t)}return i}function Wa(i,e,t,n){try{return n?i(...n):i()}catch(s){_c(s,e,t)}}function Hi(i,e,t,n){if(it(i)){const s=Wa(i,e,t,n);return s&&Wm(s)&&s.catch(r=>{_c(r,e,t)}),s}if(Je(i)){const s=[];for(let r=0;r<i.length;r++)s.push(Hi(i[r],e,t,n));return s}}function _c(i,e,t,n=!0){const s=e?e.vnode:null,{errorHandler:r,throwUnhandledErrorInProduction:o}=e&&e.appContext.config||Et;if(e){let a=e.parent;const l=e.proxy,c=`https://vuejs.org/error-reference/#runtime-${t}`;for(;a;){const u=a.ec;if(u){for(let f=0;f<u.length;f++)if(u[f](i,l,c)===!1)return}a=a.parent}if(r){ps(),Wa(r,null,10,[i,l,c]),ms();return}}S_(i,t,s,n,o)}function S_(i,e,t,n=!0,s=!1){if(s)throw i;console.error(i)}const wn=[];let wi=-1;const co=[];let Is=null,eo=0;const p0=Promise.resolve();let Yl=null;function y_(i){const e=Yl||p0;return i?e.then(this?i.bind(this):i):e}function b_(i){let e=wi+1,t=wn.length;for(;e<t;){const n=e+t>>>1,s=wn[n],r=ba(s);r<i||r===i&&s.flags&2?e=n+1:t=n}return e}function gd(i){if(!(i.flags&1)){const e=ba(i),t=wn[wn.length-1];!t||!(i.flags&2)&&e>=ba(t)?wn.push(i):wn.splice(b_(e),0,i),i.flags|=1,m0()}}function m0(){Yl||(Yl=p0.then(x0))}function M_(i){Je(i)?co.push(...i):Is&&i.id===-1?Is.splice(eo+1,0,i):i.flags&1||(co.push(i),i.flags|=1),m0()}function vh(i,e,t=wi+1){for(;t<wn.length;t++){const n=wn[t];if(n&&n.flags&2){if(i&&n.id!==i.uid)continue;wn.splice(t,1),t--,n.flags&4&&(n.flags&=-2),n(),n.flags&4||(n.flags&=-2)}}}function g0(i){if(co.length){const e=[...new Set(co)].sort((t,n)=>ba(t)-ba(n));if(co.length=0,Is){Is.push(...e);return}for(Is=e,eo=0;eo<Is.length;eo++){const t=Is[eo];t.flags&4&&(t.flags&=-2),t.flags&8||t(),t.flags&=-2}Is=null,eo=0}}const ba=i=>i.id==null?i.flags&2?-1:1/0:i.id;function x0(i){try{for(wi=0;wi<wn.length;wi++){const e=wn[wi];e&&!(e.flags&8)&&(e.flags&4&&(e.flags&=-2),Wa(e,e.i,e.i?15:14),e.flags&4||(e.flags&=-2))}}finally{for(;wi<wn.length;wi++){const e=wn[wi];e&&(e.flags&=-2)}wi=-1,wn.length=0,g0(),Yl=null,(wn.length||co.length)&&x0()}}let fi=null,_0=null;function Ql(i){const e=fi;return fi=i,_0=i&&i.type.__scopeId||null,e}function C_(i,e=fi,t){if(!e||i._n)return i;const n=(...s)=>{n._d&&Ih(-1);const r=Ql(e);let o;try{o=i(...s)}finally{Ql(r),n._d&&Ih(1)}return o};return n._n=!0,n._c=!0,n._d=!0,n}function qo(i,e){if(fi===null)return i;const t=yc(fi),n=i.dirs||(i.dirs=[]);for(let s=0;s<e.length;s++){let[r,o,a,l=Et]=e[s];r&&(it(r)&&(r={mounted:r,updated:r}),r.deep&&cs(o),n.push({dir:r,instance:t,value:o,oldValue:void 0,arg:a,modifiers:l}))}return i}function er(i,e,t,n){const s=i.dirs,r=e&&e.dirs;for(let o=0;o<s.length;o++){const a=s[o];r&&(a.oldValue=r[o].value);let l=a.dir[n];l&&(ps(),Hi(l,t,8,[i.el,a,i,e]),ms())}}function T_(i,e){if(In){let t=In.provides;const n=In.parent&&In.parent.provides;n===t&&(t=In.provides=Object.create(n)),t[i]=e}}function Bl(i,e,t=!1){const n=Tv();if(n||uo){let s=uo?uo._context.provides:n?n.parent==null||n.ce?n.vnode.appContext&&n.vnode.appContext.provides:n.parent.provides:void 0;if(s&&i in s)return s[i];if(arguments.length>1)return t&&it(e)?e.call(n&&n.proxy):e}}const E_=Symbol.for("v-scx"),w_=()=>Bl(E_);function zc(i,e,t){return v0(i,e,t)}function v0(i,e,t=Et){const{immediate:n,deep:s,flush:r,once:o}=t,a=yn({},t),l=e&&n||!e&&r!=="post";let c;if(Ca){if(r==="sync"){const h=w_();c=h.__watcherHandles||(h.__watcherHandles=[])}else if(!l){const h=()=>{};return h.stop=Ni,h.resume=Ni,h.pause=Ni,h}}const u=In;a.call=(h,x,p)=>Hi(h,u,x,p);let f=!1;r==="post"?a.scheduler=h=>{On(h,u&&u.suspense)}:r!=="sync"&&(f=!0,a.scheduler=(h,x)=>{x?h():gd(h)}),a.augmentJob=h=>{e&&(h.flags|=4),f&&(h.flags|=2,u&&(h.id=u.uid,h.i=u))};const d=A_(i,e,a);return Ca&&(c?c.push(d):l&&d()),d}function R_(i,e,t){const n=this.proxy,s=$t(i)?i.includes(".")?A0(n,i):()=>n[i]:i.bind(n,n);let r;it(e)?r=e:(r=e.handler,t=e);const o=Xa(this),a=v0(s,r.bind(n),t);return o(),a}function A0(i,e){const t=e.split(".");return()=>{let n=i;for(let s=0;s<t.length&&n;s++)n=n[t[s]];return n}}const I_=Symbol("_vte"),D_=i=>i.__isTeleport,P_=Symbol("_leaveCb");function xd(i,e){i.shapeFlag&6&&i.component?(i.transition=e,xd(i.component.subTree,e)):i.shapeFlag&128?(i.ssContent.transition=e.clone(i.ssContent),i.ssFallback.transition=e.clone(i.ssFallback)):i.transition=e}function S0(i){i.ids=[i.ids[0]+i.ids[2]+++"-",0,0]}function Ah(i,e){let t;return!!((t=Object.getOwnPropertyDescriptor(i,e))&&!t.configurable)}const Kl=new WeakMap;function aa(i,e,t,n,s=!1){if(Je(i)){i.forEach((p,g)=>aa(p,e&&(Je(e)?e[g]:e),t,n,s));return}if(la(n)&&!s){n.shapeFlag&512&&n.type.__asyncResolved&&n.component.subTree.component&&aa(i,e,t,n.component.subTree);return}const r=n.shapeFlag&4?yc(n.component):n.el,o=s?null:r,{i:a,r:l}=i,c=e&&e.r,u=a.refs===Et?a.refs={}:a.refs,f=a.setupState,d=_t(f),h=f===Et?Gm:p=>Ah(u,p)?!1:vt(d,p),x=(p,g)=>!(g&&Ah(u,g));if(c!=null&&c!==l){if(Sh(e),$t(c))u[c]=null,h(c)&&(f[c]=null);else if(vn(c)){const p=e;x(c,p.k)&&(c.value=null),p.k&&(u[p.k]=null)}}if(it(l))Wa(l,a,12,[o,u]);else{const p=$t(l),g=vn(l);if(p||g){const m=()=>{if(i.f){const _=p?h(l)?f[l]:u[l]:x()||!i.k?l.value:u[i.k];if(s)Je(_)&&sd(_,r);else if(Je(_))_.includes(r)||_.push(r);else if(p)u[l]=[r],h(l)&&(f[l]=u[l]);else{const v=[r];x(l,i.k)&&(l.value=v),i.k&&(u[i.k]=v)}}else p?(u[l]=o,h(l)&&(f[l]=o)):g&&(x(l,i.k)&&(l.value=o),i.k&&(u[i.k]=o))};if(o){const _=()=>{m(),Kl.delete(i)};_.id=-1,Kl.set(i,_),On(_,t)}else Sh(i),m()}}}function Sh(i){const e=Kl.get(i);e&&(e.flags|=8,Kl.delete(i))}gc().requestIdleCallback;gc().cancelIdleCallback;const la=i=>!!i.type.__asyncLoader,y0=i=>i.type.__isKeepAlive;function F_(i,e){b0(i,"a",e)}function L_(i,e){b0(i,"da",e)}function b0(i,e,t=In){const n=i.__wdc||(i.__wdc=()=>{let s=t;for(;s;){if(s.isDeactivated)return;s=s.parent}return i()});if(vc(e,n,t),t){let s=t.parent;for(;s&&s.parent;)y0(s.parent.vnode)&&B_(n,e,t,s),s=s.parent}}function B_(i,e,t,n){const s=vc(e,i,n,!0);T0(()=>{sd(n[e],s)},t)}function vc(i,e,t=In,n=!1){if(t){const s=t[i]||(t[i]=[]),r=e.__weh||(e.__weh=(...o)=>{ps();const a=Xa(t),l=Hi(e,t,i,o);return a(),ms(),l});return n?s.unshift(r):s.push(r),r}}const vs=i=>(e,t=In)=>{(!Ca||i==="sp")&&vc(i,(...n)=>e(...n),t)},U_=vs("bm"),M0=vs("m"),O_=vs("bu"),N_=vs("u"),C0=vs("bum"),T0=vs("um"),z_=vs("sp"),k_=vs("rtg"),H_=vs("rtc");function V_(i,e=In){vc("ec",i,e)}const G_=Symbol.for("v-ndc");function W_(i,e,t,n){let s;const r=t,o=Je(i);if(o||$t(i)){const a=o&&yr(i);let l=!1,c=!1;a&&(l=!gi(i),c=gs(i),i=xc(i)),s=new Array(i.length);for(let u=0,f=i.length;u<f;u++)s[u]=e(l?c?So(Ci(i[u])):Ci(i[u]):i[u],u,void 0,r)}else if(typeof i=="number"){s=new Array(i);for(let a=0;a<i;a++)s[a]=e(a+1,a,void 0,r)}else if(Mt(i))if(i[Symbol.iterator])s=Array.from(i,(a,l)=>e(a,l,void 0,r));else{const a=Object.keys(i);s=new Array(a.length);for(let l=0,c=a.length;l<c;l++){const u=a[l];s[l]=e(i[u],u,l,r)}}else s=[];return s}const qu=i=>i?Y0(i)?yc(i):qu(i.parent):null,ca=yn(Object.create(null),{$:i=>i,$el:i=>i.vnode.el,$data:i=>i.data,$props:i=>i.props,$attrs:i=>i.attrs,$slots:i=>i.slots,$refs:i=>i.refs,$parent:i=>qu(i.parent),$root:i=>qu(i.root),$host:i=>i.ce,$emit:i=>i.emit,$options:i=>w0(i),$forceUpdate:i=>i.f||(i.f=()=>{gd(i.update)}),$nextTick:i=>i.n||(i.n=y_.bind(i.proxy)),$watch:i=>R_.bind(i)}),kc=(i,e)=>i!==Et&&!i.__isScriptSetup&&vt(i,e),X_={get({_:i},e){if(e==="__v_skip")return!0;const{ctx:t,setupState:n,data:s,props:r,accessCache:o,type:a,appContext:l}=i;if(e[0]!=="$"){const d=o[e];if(d!==void 0)switch(d){case 1:return n[e];case 2:return s[e];case 4:return t[e];case 3:return r[e]}else{if(kc(n,e))return o[e]=1,n[e];if(s!==Et&&vt(s,e))return o[e]=2,s[e];if(vt(r,e))return o[e]=3,r[e];if(t!==Et&&vt(t,e))return o[e]=4,t[e];Yu&&(o[e]=0)}}const c=ca[e];let u,f;if(c)return e==="$attrs"&&xn(i.attrs,"get",""),c(i);if((u=a.__cssModules)&&(u=u[e]))return u;if(t!==Et&&vt(t,e))return o[e]=4,t[e];if(f=l.config.globalProperties,vt(f,e))return f[e]},set({_:i},e,t){const{data:n,setupState:s,ctx:r}=i;return kc(s,e)?(s[e]=t,!0):n!==Et&&vt(n,e)?(n[e]=t,!0):vt(i.props,e)||e[0]==="$"&&e.slice(1)in i?!1:(r[e]=t,!0)},has({_:{data:i,setupState:e,accessCache:t,ctx:n,appContext:s,props:r,type:o}},a){let l;return!!(t[a]||i!==Et&&a[0]!=="$"&&vt(i,a)||kc(e,a)||vt(r,a)||vt(n,a)||vt(ca,a)||vt(s.config.globalProperties,a)||(l=o.__cssModules)&&l[a])},defineProperty(i,e,t){return t.get!=null?i._.accessCache[e]=0:vt(t,"value")&&this.set(i,e,t.value,null),Reflect.defineProperty(i,e,t)}};function yh(i){return Je(i)?i.reduce((e,t)=>(e[t]=null,e),{}):i}let Yu=!0;function q_(i){const e=w0(i),t=i.proxy,n=i.ctx;Yu=!1,e.beforeCreate&&bh(e.beforeCreate,i,"bc");const{data:s,computed:r,methods:o,watch:a,provide:l,inject:c,created:u,beforeMount:f,mounted:d,beforeUpdate:h,updated:x,activated:p,deactivated:g,beforeDestroy:m,beforeUnmount:_,destroyed:v,unmounted:A,render:S,renderTracked:b,renderTriggered:M,errorCaptured:R,serverPrefetch:y,expose:C,inheritAttrs:F,components:I,directives:O,filters:z}=e;if(c&&Y_(c,n,null),o)for(const q in o){const G=o[q];it(G)&&(n[q]=G.bind(t))}if(s){const q=s.call(t,t);Mt(q)&&(i.data=hd(q))}if(Yu=!0,r)for(const q in r){const G=r[q],Y=it(G)?G.bind(t,t):it(G.get)?G.get.bind(t,t):Ni,ae=!it(G)&&it(G.set)?G.set.bind(t):Ni,Ae=fr({get:Y,set:ae});Object.defineProperty(n,q,{enumerable:!0,configurable:!0,get:()=>Ae.value,set:Pe=>Ae.value=Pe})}if(a)for(const q in a)E0(a[q],n,t,q);if(l){const q=it(l)?l.call(t):l;Reflect.ownKeys(q).forEach(G=>{T_(G,q[G])})}u&&bh(u,i,"c");function V(q,G){Je(G)?G.forEach(Y=>q(Y.bind(t))):G&&q(G.bind(t))}if(V(U_,f),V(M0,d),V(O_,h),V(N_,x),V(F_,p),V(L_,g),V(V_,R),V(H_,b),V(k_,M),V(C0,_),V(T0,A),V(z_,y),Je(C))if(C.length){const q=i.exposed||(i.exposed={});C.forEach(G=>{Object.defineProperty(q,G,{get:()=>t[G],set:Y=>t[G]=Y,enumerable:!0})})}else i.exposed||(i.exposed={});S&&i.render===Ni&&(i.render=S),F!=null&&(i.inheritAttrs=F),I&&(i.components=I),O&&(i.directives=O),y&&S0(i)}function Y_(i,e,t=Ni){Je(i)&&(i=Qu(i));for(const n in i){const s=i[n];let r;Mt(s)?"default"in s?r=Bl(s.from||n,s.default,!0):r=Bl(s.from||n):r=Bl(s),vn(r)?Object.defineProperty(e,n,{enumerable:!0,configurable:!0,get:()=>r.value,set:o=>r.value=o}):e[n]=r}}function bh(i,e,t){Hi(Je(i)?i.map(n=>n.bind(e.proxy)):i.bind(e.proxy),e,t)}function E0(i,e,t,n){let s=n.includes(".")?A0(t,n):()=>t[n];if($t(i)){const r=e[i];it(r)&&zc(s,r)}else if(it(i))zc(s,i.bind(t));else if(Mt(i))if(Je(i))i.forEach(r=>E0(r,e,t,n));else{const r=it(i.handler)?i.handler.bind(t):e[i.handler];it(r)&&zc(s,r,i)}}function w0(i){const e=i.type,{mixins:t,extends:n}=e,{mixins:s,optionsCache:r,config:{optionMergeStrategies:o}}=i.appContext,a=r.get(e);let l;return a?l=a:!s.length&&!t&&!n?l=e:(l={},s.length&&s.forEach(c=>jl(l,c,o,!0)),jl(l,e,o)),Mt(e)&&r.set(e,l),l}function jl(i,e,t,n=!1){const{mixins:s,extends:r}=e;r&&jl(i,r,t,!0),s&&s.forEach(o=>jl(i,o,t,!0));for(const o in e)if(!(n&&o==="expose")){const a=Q_[o]||t&&t[o];i[o]=a?a(i[o],e[o]):e[o]}return i}const Q_={data:Mh,props:Ch,emits:Ch,methods:ea,computed:ea,beforeCreate:Cn,created:Cn,beforeMount:Cn,mounted:Cn,beforeUpdate:Cn,updated:Cn,beforeDestroy:Cn,beforeUnmount:Cn,destroyed:Cn,unmounted:Cn,activated:Cn,deactivated:Cn,errorCaptured:Cn,serverPrefetch:Cn,components:ea,directives:ea,watch:j_,provide:Mh,inject:K_};function Mh(i,e){return e?i?function(){return yn(it(i)?i.call(this,this):i,it(e)?e.call(this,this):e)}:e:i}function K_(i,e){return ea(Qu(i),Qu(e))}function Qu(i){if(Je(i)){const e={};for(let t=0;t<i.length;t++)e[i[t]]=i[t];return e}return i}function Cn(i,e){return i?[...new Set([].concat(i,e))]:e}function ea(i,e){return i?yn(Object.create(null),i,e):e}function Ch(i,e){return i?Je(i)&&Je(e)?[...new Set([...i,...e])]:yn(Object.create(null),yh(i),yh(e??{})):e}function j_(i,e){if(!i)return e;if(!e)return i;const t=yn(Object.create(null),i);for(const n in e)t[n]=Cn(i[n],e[n]);return t}function R0(){return{app:null,config:{isNativeTag:Gm,performance:!1,globalProperties:{},optionMergeStrategies:{},errorHandler:void 0,warnHandler:void 0,compilerOptions:{}},mixins:[],components:{},directives:{},provides:Object.create(null),optionsCache:new WeakMap,propsCache:new WeakMap,emitsCache:new WeakMap}}let $_=0;function Z_(i,e){return function(n,s=null){it(n)||(n=yn({},n)),s!=null&&!Mt(s)&&(s=null);const r=R0(),o=new WeakSet,a=[];let l=!1;const c=r.app={_uid:$_++,_component:n,_props:s,_container:null,_context:r,_instance:null,version:Pv,get config(){return r.config},set config(u){},use(u,...f){return o.has(u)||(u&&it(u.install)?(o.add(u),u.install(c,...f)):it(u)&&(o.add(u),u(c,...f))),c},mixin(u){return r.mixins.includes(u)||r.mixins.push(u),c},component(u,f){return f?(r.components[u]=f,c):r.components[u]},directive(u,f){return f?(r.directives[u]=f,c):r.directives[u]},mount(u,f,d){if(!l){const h=c._ceVNode||zi(n,s);return h.appContext=r,d===!0?d="svg":d===!1&&(d=void 0),i(h,u,d),l=!0,c._container=u,u.__vue_app__=c,yc(h.component)}},onUnmount(u){a.push(u)},unmount(){l&&(Hi(a,c._instance,16),i(null,c._container),delete c._container.__vue_app__)},provide(u,f){return r.provides[u]=f,c},runWithContext(u){const f=uo;uo=c;try{return u()}finally{uo=f}}};return c}}let uo=null;const J_=(i,e)=>e==="modelValue"||e==="model-value"?i.modelModifiers:i[`${e}Modifiers`]||i[`${Ws(e)}Modifiers`]||i[`${js(e)}Modifiers`];function ev(i,e,...t){if(i.isUnmounted)return;const n=i.vnode.props||Et;let s=t;const r=e.startsWith("update:"),o=r&&J_(n,e.slice(7));o&&(o.trim&&(s=t.map(u=>$t(u)?u.trim():u)),o.number&&(s=t.map(od)));let a,l=n[a=Lc(e)]||n[a=Lc(Ws(e))];!l&&r&&(l=n[a=Lc(js(e))]),l&&Hi(l,i,6,s);const c=n[a+"Once"];if(c){if(!i.emitted)i.emitted={};else if(i.emitted[a])return;i.emitted[a]=!0,Hi(c,i,6,s)}}const tv=new WeakMap;function I0(i,e,t=!1){const n=t?tv:e.emitsCache,s=n.get(i);if(s!==void 0)return s;const r=i.emits;let o={},a=!1;if(!it(i)){const l=c=>{const u=I0(c,e,!0);u&&(a=!0,yn(o,u))};!t&&e.mixins.length&&e.mixins.forEach(l),i.extends&&l(i.extends),i.mixins&&i.mixins.forEach(l)}return!r&&!a?(Mt(i)&&n.set(i,null),null):(Je(r)?r.forEach(l=>o[l]=null):yn(o,r),Mt(i)&&n.set(i,o),o)}function Ac(i,e){return!i||!hc(e)?!1:(e=e.slice(2).replace(/Once$/,""),vt(i,e[0].toLowerCase()+e.slice(1))||vt(i,js(e))||vt(i,e))}function Th(i){const{type:e,vnode:t,proxy:n,withProxy:s,propsOptions:[r],slots:o,attrs:a,emit:l,render:c,renderCache:u,props:f,data:d,setupState:h,ctx:x,inheritAttrs:p}=i,g=Ql(i);let m,_;try{if(t.shapeFlag&4){const A=s||n,S=A;m=Di(c.call(S,A,u,f,h,d,x)),_=a}else{const A=e;m=Di(A.length>1?A(f,{attrs:a,slots:o,emit:l}):A(f,null)),_=e.props?a:nv(a)}}catch(A){ua.length=0,_c(A,i,1),m=zi(Xs)}let v=m;if(_&&p!==!1){const A=Object.keys(_),{shapeFlag:S}=v;A.length&&S&7&&(r&&A.some(id)&&(_=iv(_,r)),v=yo(v,_,!1,!0))}return t.dirs&&(v=yo(v,null,!1,!0),v.dirs=v.dirs?v.dirs.concat(t.dirs):t.dirs),t.transition&&xd(v,t.transition),m=v,Ql(g),m}const nv=i=>{let e;for(const t in i)(t==="class"||t==="style"||hc(t))&&((e||(e={}))[t]=i[t]);return e},iv=(i,e)=>{const t={};for(const n in i)(!id(n)||!(n.slice(9)in e))&&(t[n]=i[n]);return t};function sv(i,e,t){const{props:n,children:s,component:r}=i,{props:o,children:a,patchFlag:l}=e,c=r.emitsOptions;if(e.dirs||e.transition)return!0;if(t&&l>=0){if(l&1024)return!0;if(l&16)return n?Eh(n,o,c):!!o;if(l&8){const u=e.dynamicProps;for(let f=0;f<u.length;f++){const d=u[f];if(D0(o,n,d)&&!Ac(c,d))return!0}}}else return(s||a)&&(!a||!a.$stable)?!0:n===o?!1:n?o?Eh(n,o,c):!0:!!o;return!1}function Eh(i,e,t){const n=Object.keys(e);if(n.length!==Object.keys(i).length)return!0;for(let s=0;s<n.length;s++){const r=n[s];if(D0(e,i,r)&&!Ac(t,r))return!0}return!1}function D0(i,e,t){const n=i[t],s=e[t];return t==="style"&&Mt(n)&&Mt(s)?!Ga(n,s):n!==s}function rv({vnode:i,parent:e},t){for(;e;){const n=e.subTree;if(n.suspense&&n.suspense.activeBranch===i&&(n.el=i.el),n===i)(i=e.vnode).el=t,e=e.parent;else break}}const P0={},F0=()=>Object.create(P0),L0=i=>Object.getPrototypeOf(i)===P0;function ov(i,e,t,n=!1){const s={},r=F0();i.propsDefaults=Object.create(null),B0(i,e,s,r);for(const o in i.propsOptions[0])o in s||(s[o]=void 0);t?i.props=n?s:d_(s):i.type.props?i.props=s:i.props=r,i.attrs=r}function av(i,e,t,n){const{props:s,attrs:r,vnode:{patchFlag:o}}=i,a=_t(s),[l]=i.propsOptions;let c=!1;if((n||o>0)&&!(o&16)){if(o&8){const u=i.vnode.dynamicProps;for(let f=0;f<u.length;f++){let d=u[f];if(Ac(i.emitsOptions,d))continue;const h=e[d];if(l)if(vt(r,d))h!==r[d]&&(r[d]=h,c=!0);else{const x=Ws(d);s[x]=Ku(l,a,x,h,i,!1)}else h!==r[d]&&(r[d]=h,c=!0)}}}else{B0(i,e,s,r)&&(c=!0);let u;for(const f in a)(!e||!vt(e,f)&&((u=js(f))===f||!vt(e,u)))&&(l?t&&(t[f]!==void 0||t[u]!==void 0)&&(s[f]=Ku(l,a,f,void 0,i,!0)):delete s[f]);if(r!==a)for(const f in r)(!e||!vt(e,f))&&(delete r[f],c=!0)}c&&ls(i.attrs,"set","")}function B0(i,e,t,n){const[s,r]=i.propsOptions;let o=!1,a;if(e)for(let l in e){if(sa(l))continue;const c=e[l];let u;s&&vt(s,u=Ws(l))?!r||!r.includes(u)?t[u]=c:(a||(a={}))[u]=c:Ac(i.emitsOptions,l)||(!(l in n)||c!==n[l])&&(n[l]=c,o=!0)}if(r){const l=_t(t),c=a||Et;for(let u=0;u<r.length;u++){const f=r[u];t[f]=Ku(s,l,f,c[f],i,!vt(c,f))}}return o}function Ku(i,e,t,n,s,r){const o=i[t];if(o!=null){const a=vt(o,"default");if(a&&n===void 0){const l=o.default;if(o.type!==Function&&!o.skipFactory&&it(l)){const{propsDefaults:c}=s;if(t in c)n=c[t];else{const u=Xa(s);n=c[t]=l.call(null,e),u()}}else n=l;s.ce&&s.ce._setProp(t,n)}o[0]&&(r&&!a?n=!1:o[1]&&(n===""||n===js(t))&&(n=!0))}return n}const lv=new WeakMap;function U0(i,e,t=!1){const n=t?lv:e.propsCache,s=n.get(i);if(s)return s;const r=i.props,o={},a=[];let l=!1;if(!it(i)){const u=f=>{l=!0;const[d,h]=U0(f,e,!0);yn(o,d),h&&a.push(...h)};!t&&e.mixins.length&&e.mixins.forEach(u),i.extends&&u(i.extends),i.mixins&&i.mixins.forEach(u)}if(!r&&!l)return Mt(i)&&n.set(i,oo),oo;if(Je(r))for(let u=0;u<r.length;u++){const f=Ws(r[u]);wh(f)&&(o[f]=Et)}else if(r)for(const u in r){const f=Ws(u);if(wh(f)){const d=r[u],h=o[f]=Je(d)||it(d)?{type:d}:yn({},d),x=h.type;let p=!1,g=!0;if(Je(x))for(let m=0;m<x.length;++m){const _=x[m],v=it(_)&&_.name;if(v==="Boolean"){p=!0;break}else v==="String"&&(g=!1)}else p=it(x)&&x.name==="Boolean";h[0]=p,h[1]=g,(p||vt(h,"default"))&&a.push(f)}}const c=[o,a];return Mt(i)&&n.set(i,c),c}function wh(i){return i[0]!=="$"&&!sa(i)}const _d=i=>i==="_"||i==="_ctx"||i==="$stable",vd=i=>Je(i)?i.map(Di):[Di(i)],cv=(i,e,t)=>{if(e._n)return e;const n=C_((...s)=>vd(e(...s)),t);return n._c=!1,n},O0=(i,e,t)=>{const n=i._ctx;for(const s in i){if(_d(s))continue;const r=i[s];if(it(r))e[s]=cv(s,r,n);else if(r!=null){const o=vd(r);e[s]=()=>o}}},N0=(i,e)=>{const t=vd(e);i.slots.default=()=>t},z0=(i,e,t)=>{for(const n in e)(t||!_d(n))&&(i[n]=e[n])},uv=(i,e,t)=>{const n=i.slots=F0();if(i.vnode.shapeFlag&32){const s=e._;s?(z0(n,e,t),t&&Qm(n,"_",s,!0)):O0(e,n)}else e&&N0(i,e)},fv=(i,e,t)=>{const{vnode:n,slots:s}=i;let r=!0,o=Et;if(n.shapeFlag&32){const a=e._;a?t&&a===1?r=!1:z0(s,e,t):(r=!e.$stable,O0(e,s)),o=e}else e&&(N0(i,e),o={default:1});if(r)for(const a in s)!_d(a)&&o[a]==null&&delete s[a]},On=gv;function dv(i){return hv(i)}function hv(i,e){const t=gc();t.__VUE__=!0;const{insert:n,remove:s,patchProp:r,createElement:o,createText:a,createComment:l,setText:c,setElementText:u,parentNode:f,nextSibling:d,setScopeId:h=Ni,insertStaticContent:x}=i,p=(B,N,j,P=null,ie=null,re=null,fe=void 0,ne=null,he=!!N.dynamicChildren)=>{if(B===N)return;B&&!Yo(B,N)&&(P=le(B),Pe(B,ie,re,!0),B=null),N.patchFlag===-2&&(he=!1,N.dynamicChildren=null);const{type:te,ref:ge,shapeFlag:D}=N;switch(te){case Sc:g(B,N,j,P);break;case Xs:m(B,N,j,P);break;case Vc:B==null&&_(N,j,P,fe);break;case Ii:I(B,N,j,P,ie,re,fe,ne,he);break;default:D&1?S(B,N,j,P,ie,re,fe,ne,he):D&6?O(B,N,j,P,ie,re,fe,ne,he):(D&64||D&128)&&te.process(B,N,j,P,ie,re,fe,ne,he,Ce)}ge!=null&&ie?aa(ge,B&&B.ref,re,N||B,!N):ge==null&&B&&B.ref!=null&&aa(B.ref,null,re,B,!0)},g=(B,N,j,P)=>{if(B==null)n(N.el=a(N.children),j,P);else{const ie=N.el=B.el;N.children!==B.children&&c(ie,N.children)}},m=(B,N,j,P)=>{B==null?n(N.el=l(N.children||""),j,P):N.el=B.el},_=(B,N,j,P)=>{[B.el,B.anchor]=x(B.children,N,j,P,B.el,B.anchor)},v=({el:B,anchor:N},j,P)=>{let ie;for(;B&&B!==N;)ie=d(B),n(B,j,P),B=ie;n(N,j,P)},A=({el:B,anchor:N})=>{let j;for(;B&&B!==N;)j=d(B),s(B),B=j;s(N)},S=(B,N,j,P,ie,re,fe,ne,he)=>{if(N.type==="svg"?fe="svg":N.type==="math"&&(fe="mathml"),B==null)b(N,j,P,ie,re,fe,ne,he);else{const te=B.el&&B.el._isVueCE?B.el:null;try{te&&te._beginPatch(),y(B,N,ie,re,fe,ne,he)}finally{te&&te._endPatch()}}},b=(B,N,j,P,ie,re,fe,ne)=>{let he,te;const{props:ge,shapeFlag:D,transition:T,dirs:X}=B;if(he=B.el=o(B.type,re,ge&&ge.is,ge),D&8?u(he,B.children):D&16&&R(B.children,he,null,P,ie,Hc(B,re),fe,ne),X&&er(B,null,P,"created"),M(he,B,B.scopeId,fe,P),ge){for(const ce in ge)ce!=="value"&&!sa(ce)&&r(he,ce,null,ge[ce],re,P);"value"in ge&&r(he,"value",null,ge.value,re),(te=ge.onVnodeBeforeMount)&&Ei(te,P,B)}X&&er(B,null,P,"beforeMount");const se=pv(ie,T);se&&T.beforeEnter(he),n(he,N,j),((te=ge&&ge.onVnodeMounted)||se||X)&&On(()=>{te&&Ei(te,P,B),se&&T.enter(he),X&&er(B,null,P,"mounted")},ie)},M=(B,N,j,P,ie)=>{if(j&&h(B,j),P)for(let re=0;re<P.length;re++)h(B,P[re]);if(ie){let re=ie.subTree;if(N===re||G0(re.type)&&(re.ssContent===N||re.ssFallback===N)){const fe=ie.vnode;M(B,fe,fe.scopeId,fe.slotScopeIds,ie.parent)}}},R=(B,N,j,P,ie,re,fe,ne,he=0)=>{for(let te=he;te<B.length;te++){const ge=B[te]=ne?rs(B[te]):Di(B[te]);p(null,ge,N,j,P,ie,re,fe,ne)}},y=(B,N,j,P,ie,re,fe)=>{const ne=N.el=B.el;let{patchFlag:he,dynamicChildren:te,dirs:ge}=N;he|=B.patchFlag&16;const D=B.props||Et,T=N.props||Et;let X;if(j&&tr(j,!1),(X=T.onVnodeBeforeUpdate)&&Ei(X,j,N,B),ge&&er(N,B,j,"beforeUpdate"),j&&tr(j,!0),(D.innerHTML&&T.innerHTML==null||D.textContent&&T.textContent==null)&&u(ne,""),te?C(B.dynamicChildren,te,ne,j,P,Hc(N,ie),re):fe||G(B,N,ne,null,j,P,Hc(N,ie),re,!1),he>0){if(he&16)F(ne,D,T,j,ie);else if(he&2&&D.class!==T.class&&r(ne,"class",null,T.class,ie),he&4&&r(ne,"style",D.style,T.style,ie),he&8){const se=N.dynamicProps;for(let ce=0;ce<se.length;ce++){const Z=se[ce],Be=D[Z],Se=T[Z];(Se!==Be||Z==="value")&&r(ne,Z,Be,Se,ie,j)}}he&1&&B.children!==N.children&&u(ne,N.children)}else!fe&&te==null&&F(ne,D,T,j,ie);((X=T.onVnodeUpdated)||ge)&&On(()=>{X&&Ei(X,j,N,B),ge&&er(N,B,j,"updated")},P)},C=(B,N,j,P,ie,re,fe)=>{for(let ne=0;ne<N.length;ne++){const he=B[ne],te=N[ne],ge=he.el&&(he.type===Ii||!Yo(he,te)||he.shapeFlag&198)?f(he.el):j;p(he,te,ge,null,P,ie,re,fe,!0)}},F=(B,N,j,P,ie)=>{if(N!==j){if(N!==Et)for(const re in N)!sa(re)&&!(re in j)&&r(B,re,N[re],null,ie,P);for(const re in j){if(sa(re))continue;const fe=j[re],ne=N[re];fe!==ne&&re!=="value"&&r(B,re,ne,fe,ie,P)}"value"in j&&r(B,"value",N.value,j.value,ie)}},I=(B,N,j,P,ie,re,fe,ne,he)=>{const te=N.el=B?B.el:a(""),ge=N.anchor=B?B.anchor:a("");let{patchFlag:D,dynamicChildren:T,slotScopeIds:X}=N;X&&(ne=ne?ne.concat(X):X),B==null?(n(te,j,P),n(ge,j,P),R(N.children||[],j,ge,ie,re,fe,ne,he)):D>0&&D&64&&T&&B.dynamicChildren&&B.dynamicChildren.length===T.length?(C(B.dynamicChildren,T,j,ie,re,fe,ne),(N.key!=null||ie&&N===ie.subTree)&&k0(B,N,!0)):G(B,N,j,ge,ie,re,fe,ne,he)},O=(B,N,j,P,ie,re,fe,ne,he)=>{N.slotScopeIds=ne,B==null?N.shapeFlag&512?ie.ctx.activate(N,j,P,fe,he):z(N,j,P,ie,re,fe,he):H(B,N,he)},z=(B,N,j,P,ie,re,fe)=>{const ne=B.component=Cv(B,P,ie);if(y0(B)&&(ne.ctx.renderer=Ce),Ev(ne,!1,fe),ne.asyncDep){if(ie&&ie.registerDep(ne,V,fe),!B.el){const he=ne.subTree=zi(Xs);m(null,he,N,j),B.placeholder=he.el}}else V(ne,B,N,j,ie,re,fe)},H=(B,N,j)=>{const P=N.component=B.component;if(sv(B,N,j))if(P.asyncDep&&!P.asyncResolved){q(P,N,j);return}else P.next=N,P.update();else N.el=B.el,P.vnode=N},V=(B,N,j,P,ie,re,fe)=>{const ne=()=>{if(B.isMounted){let{next:D,bu:T,u:X,parent:se,vnode:ce}=B;{const k=H0(B);if(k){D&&(D.el=ce.el,q(B,D,fe)),k.asyncDep.then(()=>{On(()=>{B.isUnmounted||te()},ie)});return}}let Z=D,Be;tr(B,!1),D?(D.el=ce.el,q(B,D,fe)):D=ce,T&&Ll(T),(Be=D.props&&D.props.onVnodeBeforeUpdate)&&Ei(Be,se,D,ce),tr(B,!0);const Se=Th(B),Oe=B.subTree;B.subTree=Se,p(Oe,Se,f(Oe.el),le(Oe),B,ie,re),D.el=Se.el,Z===null&&rv(B,Se.el),X&&On(X,ie),(Be=D.props&&D.props.onVnodeUpdated)&&On(()=>Ei(Be,se,D,ce),ie)}else{let D;const{el:T,props:X}=N,{bm:se,m:ce,parent:Z,root:Be,type:Se}=B,Oe=la(N);tr(B,!1),se&&Ll(se),!Oe&&(D=X&&X.onVnodeBeforeMount)&&Ei(D,Z,N),tr(B,!0);{Be.ce&&Be.ce._hasShadowRoot()&&Be.ce._injectChildStyle(Se);const k=B.subTree=Th(B);p(null,k,j,P,B,ie,re),N.el=k.el}if(ce&&On(ce,ie),!Oe&&(D=X&&X.onVnodeMounted)){const k=N;On(()=>Ei(D,Z,k),ie)}(N.shapeFlag&256||Z&&la(Z.vnode)&&Z.vnode.shapeFlag&256)&&B.a&&On(B.a,ie),B.isMounted=!0,N=j=P=null}};B.scope.on();const he=B.effect=new Jm(ne);B.scope.off();const te=B.update=he.run.bind(he),ge=B.job=he.runIfDirty.bind(he);ge.i=B,ge.id=B.uid,he.scheduler=()=>gd(ge),tr(B,!0),te()},q=(B,N,j)=>{N.component=B;const P=B.vnode.props;B.vnode=N,B.next=null,av(B,N.props,P,j),fv(B,N.children,j),ps(),vh(B),ms()},G=(B,N,j,P,ie,re,fe,ne,he=!1)=>{const te=B&&B.children,ge=B?B.shapeFlag:0,D=N.children,{patchFlag:T,shapeFlag:X}=N;if(T>0){if(T&128){ae(te,D,j,P,ie,re,fe,ne,he);return}else if(T&256){Y(te,D,j,P,ie,re,fe,ne,he);return}}X&8?(ge&16&&J(te,ie,re),D!==te&&u(j,D)):ge&16?X&16?ae(te,D,j,P,ie,re,fe,ne,he):J(te,ie,re,!0):(ge&8&&u(j,""),X&16&&R(D,j,P,ie,re,fe,ne,he))},Y=(B,N,j,P,ie,re,fe,ne,he)=>{B=B||oo,N=N||oo;const te=B.length,ge=N.length,D=Math.min(te,ge);let T;for(T=0;T<D;T++){const X=N[T]=he?rs(N[T]):Di(N[T]);p(B[T],X,j,null,ie,re,fe,ne,he)}te>ge?J(B,ie,re,!0,!1,D):R(N,j,P,ie,re,fe,ne,he,D)},ae=(B,N,j,P,ie,re,fe,ne,he)=>{let te=0;const ge=N.length;let D=B.length-1,T=ge-1;for(;te<=D&&te<=T;){const X=B[te],se=N[te]=he?rs(N[te]):Di(N[te]);if(Yo(X,se))p(X,se,j,null,ie,re,fe,ne,he);else break;te++}for(;te<=D&&te<=T;){const X=B[D],se=N[T]=he?rs(N[T]):Di(N[T]);if(Yo(X,se))p(X,se,j,null,ie,re,fe,ne,he);else break;D--,T--}if(te>D){if(te<=T){const X=T+1,se=X<ge?N[X].el:P;for(;te<=T;)p(null,N[te]=he?rs(N[te]):Di(N[te]),j,se,ie,re,fe,ne,he),te++}}else if(te>T)for(;te<=D;)Pe(B[te],ie,re,!0),te++;else{const X=te,se=te,ce=new Map;for(te=se;te<=T;te++){const Te=N[te]=he?rs(N[te]):Di(N[te]);Te.key!=null&&ce.set(Te.key,te)}let Z,Be=0;const Se=T-se+1;let Oe=!1,k=0;const ee=new Array(Se);for(te=0;te<Se;te++)ee[te]=0;for(te=X;te<=D;te++){const Te=B[te];if(Be>=Se){Pe(Te,ie,re,!0);continue}let Fe;if(Te.key!=null)Fe=ce.get(Te.key);else for(Z=se;Z<=T;Z++)if(ee[Z-se]===0&&Yo(Te,N[Z])){Fe=Z;break}Fe===void 0?Pe(Te,ie,re,!0):(ee[Fe-se]=te+1,Fe>=k?k=Fe:Oe=!0,p(Te,N[Fe],j,null,ie,re,fe,ne,he),Be++)}const me=Oe?mv(ee):oo;for(Z=me.length-1,te=Se-1;te>=0;te--){const Te=se+te,Fe=N[Te],Re=N[Te+1],Ye=Te+1<ge?Re.el||V0(Re):P;ee[te]===0?p(null,Fe,j,Ye,ie,re,fe,ne,he):Oe&&(Z<0||te!==me[Z]?Ae(Fe,j,Ye,2):Z--)}}},Ae=(B,N,j,P,ie=null)=>{const{el:re,type:fe,transition:ne,children:he,shapeFlag:te}=B;if(te&6){Ae(B.component.subTree,N,j,P);return}if(te&128){B.suspense.move(N,j,P);return}if(te&64){fe.move(B,N,j,Ce);return}if(fe===Ii){n(re,N,j);for(let D=0;D<he.length;D++)Ae(he[D],N,j,P);n(B.anchor,N,j);return}if(fe===Vc){v(B,N,j);return}if(P!==2&&te&1&&ne)if(P===0)ne.beforeEnter(re),n(re,N,j),On(()=>ne.enter(re),ie);else{const{leave:D,delayLeave:T,afterLeave:X}=ne,se=()=>{B.ctx.isUnmounted?s(re):n(re,N,j)},ce=()=>{re._isLeaving&&re[P_](!0),D(re,()=>{se(),X&&X()})};T?T(re,se,ce):ce()}else n(re,N,j)},Pe=(B,N,j,P=!1,ie=!1)=>{const{type:re,props:fe,ref:ne,children:he,dynamicChildren:te,shapeFlag:ge,patchFlag:D,dirs:T,cacheIndex:X}=B;if(D===-2&&(ie=!1),ne!=null&&(ps(),aa(ne,null,j,B,!0),ms()),X!=null&&(N.renderCache[X]=void 0),ge&256){N.ctx.deactivate(B);return}const se=ge&1&&T,ce=!la(B);let Z;if(ce&&(Z=fe&&fe.onVnodeBeforeUnmount)&&Ei(Z,N,B),ge&6)He(B.component,j,P);else{if(ge&128){B.suspense.unmount(j,P);return}se&&er(B,null,N,"beforeUnmount"),ge&64?B.type.remove(B,N,j,Ce,P):te&&!te.hasOnce&&(re!==Ii||D>0&&D&64)?J(te,N,j,!1,!0):(re===Ii&&D&384||!ie&&ge&16)&&J(he,N,j),P&&Ne(B)}(ce&&(Z=fe&&fe.onVnodeUnmounted)||se)&&On(()=>{Z&&Ei(Z,N,B),se&&er(B,null,N,"unmounted")},j)},Ne=B=>{const{type:N,el:j,anchor:P,transition:ie}=B;if(N===Ii){We(j,P);return}if(N===Vc){A(B);return}const re=()=>{s(j),ie&&!ie.persisted&&ie.afterLeave&&ie.afterLeave()};if(B.shapeFlag&1&&ie&&!ie.persisted){const{leave:fe,delayLeave:ne}=ie,he=()=>fe(j,re);ne?ne(B.el,re,he):he()}else re()},We=(B,N)=>{let j;for(;B!==N;)j=d(B),s(B),B=j;s(N)},He=(B,N,j)=>{const{bum:P,scope:ie,job:re,subTree:fe,um:ne,m:he,a:te}=B;Rh(he),Rh(te),P&&Ll(P),ie.stop(),re&&(re.flags|=8,Pe(fe,B,N,j)),ne&&On(ne,N),On(()=>{B.isUnmounted=!0},N)},J=(B,N,j,P=!1,ie=!1,re=0)=>{for(let fe=re;fe<B.length;fe++)Pe(B[fe],N,j,P,ie)},le=B=>{if(B.shapeFlag&6)return le(B.component.subTree);if(B.shapeFlag&128)return B.suspense.next();const N=d(B.anchor||B.el),j=N&&N[I_];return j?d(j):N};let be=!1;const ze=(B,N,j)=>{let P;B==null?N._vnode&&(Pe(N._vnode,null,null,!0),P=N._vnode.component):p(N._vnode||null,B,N,null,null,null,j),N._vnode=B,be||(be=!0,vh(P),g0(),be=!1)},Ce={p,um:Pe,m:Ae,r:Ne,mt:z,mc:R,pc:G,pbc:C,n:le,o:i};return{render:ze,hydrate:void 0,createApp:Z_(ze)}}function Hc({type:i,props:e},t){return t==="svg"&&i==="foreignObject"||t==="mathml"&&i==="annotation-xml"&&e&&e.encoding&&e.encoding.includes("html")?void 0:t}function tr({effect:i,job:e},t){t?(i.flags|=32,e.flags|=4):(i.flags&=-33,e.flags&=-5)}function pv(i,e){return(!i||i&&!i.pendingBranch)&&e&&!e.persisted}function k0(i,e,t=!1){const n=i.children,s=e.children;if(Je(n)&&Je(s))for(let r=0;r<n.length;r++){const o=n[r];let a=s[r];a.shapeFlag&1&&!a.dynamicChildren&&((a.patchFlag<=0||a.patchFlag===32)&&(a=s[r]=rs(s[r]),a.el=o.el),!t&&a.patchFlag!==-2&&k0(o,a)),a.type===Sc&&(a.patchFlag===-1&&(a=s[r]=rs(a)),a.el=o.el),a.type===Xs&&!a.el&&(a.el=o.el)}}function mv(i){const e=i.slice(),t=[0];let n,s,r,o,a;const l=i.length;for(n=0;n<l;n++){const c=i[n];if(c!==0){if(s=t[t.length-1],i[s]<c){e[n]=s,t.push(n);continue}for(r=0,o=t.length-1;r<o;)a=r+o>>1,i[t[a]]<c?r=a+1:o=a;c<i[t[r]]&&(r>0&&(e[n]=t[r-1]),t[r]=n)}}for(r=t.length,o=t[r-1];r-- >0;)t[r]=o,o=e[o];return t}function H0(i){const e=i.subTree.component;if(e)return e.asyncDep&&!e.asyncResolved?e:H0(e)}function Rh(i){if(i)for(let e=0;e<i.length;e++)i[e].flags|=8}function V0(i){if(i.placeholder)return i.placeholder;const e=i.component;return e?V0(e.subTree):null}const G0=i=>i.__isSuspense;function gv(i,e){e&&e.pendingBranch?Je(i)?e.effects.push(...i):e.effects.push(i):M_(i)}const Ii=Symbol.for("v-fgt"),Sc=Symbol.for("v-txt"),Xs=Symbol.for("v-cmt"),Vc=Symbol.for("v-stc"),ua=[];let $n=null;function an(i=!1){ua.push($n=i?null:[])}function xv(){ua.pop(),$n=ua[ua.length-1]||null}let Ma=1;function Ih(i,e=!1){Ma+=i,i<0&&$n&&e&&($n.hasOnce=!0)}function W0(i){return i.dynamicChildren=Ma>0?$n||oo:null,xv(),Ma>0&&$n&&$n.push(i),i}function gn(i,e,t,n,s,r){return W0(qe(i,e,t,n,s,r,!0))}function _v(i,e,t,n,s){return W0(zi(i,e,t,n,s,!0))}function X0(i){return i?i.__v_isVNode===!0:!1}function Yo(i,e){return i.type===e.type&&i.key===e.key}const q0=({key:i})=>i??null,Ul=({ref:i,ref_key:e,ref_for:t})=>(typeof i=="number"&&(i=""+i),i!=null?$t(i)||vn(i)||it(i)?{i:fi,r:i,k:e,f:!!t}:i:null);function qe(i,e=null,t=null,n=0,s=null,r=i===Ii?0:1,o=!1,a=!1){const l={__v_isVNode:!0,__v_skip:!0,type:i,props:e,key:e&&q0(e),ref:e&&Ul(e),scopeId:_0,slotScopeIds:null,children:t,component:null,suspense:null,ssContent:null,ssFallback:null,dirs:null,transition:null,el:null,anchor:null,target:null,targetStart:null,targetAnchor:null,staticCount:0,shapeFlag:r,patchFlag:n,dynamicProps:s,dynamicChildren:null,appContext:null,ctx:fi};return a?(Ad(l,t),r&128&&i.normalize(l)):t&&(l.shapeFlag|=$t(t)?8:16),Ma>0&&!o&&$n&&(l.patchFlag>0||r&6)&&l.patchFlag!==32&&$n.push(l),l}const zi=vv;function vv(i,e=null,t=null,n=0,s=null,r=!1){if((!i||i===G_)&&(i=Xs),X0(i)){const a=yo(i,e,!0);return t&&Ad(a,t),Ma>0&&!r&&$n&&(a.shapeFlag&6?$n[$n.indexOf(i)]=a:$n.push(a)),a.patchFlag=-2,a}if(Dv(i)&&(i=i.__vccOpts),e){e=Av(e);let{class:a,style:l}=e;a&&!$t(a)&&(e.class=lo(a)),Mt(l)&&(md(l)&&!Je(l)&&(l=yn({},l)),e.style=ad(l))}const o=$t(i)?1:G0(i)?128:D_(i)?64:Mt(i)?4:it(i)?2:0;return qe(i,e,t,n,s,o,r,!0)}function Av(i){return i?md(i)||L0(i)?yn({},i):i:null}function yo(i,e,t=!1,n=!1){const{props:s,ref:r,patchFlag:o,children:a,transition:l}=i,c=e?yv(s||{},e):s,u={__v_isVNode:!0,__v_skip:!0,type:i.type,props:c,key:c&&q0(c),ref:e&&e.ref?t&&r?Je(r)?r.concat(Ul(e)):[r,Ul(e)]:Ul(e):r,scopeId:i.scopeId,slotScopeIds:i.slotScopeIds,children:a,target:i.target,targetStart:i.targetStart,targetAnchor:i.targetAnchor,staticCount:i.staticCount,shapeFlag:i.shapeFlag,patchFlag:e&&i.type!==Ii?o===-1?16:o|16:o,dynamicProps:i.dynamicProps,dynamicChildren:i.dynamicChildren,appContext:i.appContext,dirs:i.dirs,transition:l,component:i.component,suspense:i.suspense,ssContent:i.ssContent&&yo(i.ssContent),ssFallback:i.ssFallback&&yo(i.ssFallback),placeholder:i.placeholder,el:i.el,anchor:i.anchor,ctx:i.ctx,ce:i.ce};return l&&n&&xd(u,l.clone(u)),u}function Sv(i=" ",e=0){return zi(Sc,null,i,e)}function ri(i="",e=!1){return e?(an(),_v(Xs,null,i)):zi(Xs,null,i)}function Di(i){return i==null||typeof i=="boolean"?zi(Xs):Je(i)?zi(Ii,null,i.slice()):X0(i)?rs(i):zi(Sc,null,String(i))}function rs(i){return i.el===null&&i.patchFlag!==-1||i.memo?i:yo(i)}function Ad(i,e){let t=0;const{shapeFlag:n}=i;if(e==null)e=null;else if(Je(e))t=16;else if(typeof e=="object")if(n&65){const s=e.default;s&&(s._c&&(s._d=!1),Ad(i,s()),s._c&&(s._d=!0));return}else{t=32;const s=e._;!s&&!L0(e)?e._ctx=fi:s===3&&fi&&(fi.slots._===1?e._=1:(e._=2,i.patchFlag|=1024))}else it(e)?(e={default:e,_ctx:fi},t=32):(e=String(e),n&64?(t=16,e=[Sv(e)]):t=8);i.children=e,i.shapeFlag|=t}function yv(...i){const e={};for(let t=0;t<i.length;t++){const n=i[t];for(const s in n)if(s==="class")e.class!==n.class&&(e.class=lo([e.class,n.class]));else if(s==="style")e.style=ad([e.style,n.style]);else if(hc(s)){const r=e[s],o=n[s];o&&r!==o&&!(Je(r)&&r.includes(o))&&(e[s]=r?[].concat(r,o):o)}else s!==""&&(e[s]=n[s])}return e}function Ei(i,e,t,n=null){Hi(i,e,7,[t,n])}const bv=R0();let Mv=0;function Cv(i,e,t){const n=i.type,s=(e?e.appContext:i.appContext)||bv,r={uid:Mv++,vnode:i,type:n,parent:e,appContext:s,root:null,next:null,subTree:null,effect:null,update:null,job:null,scope:new qx(!0),render:null,proxy:null,exposed:null,exposeProxy:null,withProxy:null,provides:e?e.provides:Object.create(s.provides),ids:e?e.ids:["",0,0],accessCache:null,renderCache:[],components:null,directives:null,propsOptions:U0(n,s),emitsOptions:I0(n,s),emit:null,emitted:null,propsDefaults:Et,inheritAttrs:n.inheritAttrs,ctx:Et,data:Et,props:Et,attrs:Et,slots:Et,refs:Et,setupState:Et,setupContext:null,suspense:t,suspenseId:t?t.pendingId:0,asyncDep:null,asyncResolved:!1,isMounted:!1,isUnmounted:!1,isDeactivated:!1,bc:null,c:null,bm:null,m:null,bu:null,u:null,um:null,bum:null,da:null,a:null,rtg:null,rtc:null,ec:null,sp:null};return r.ctx={_:r},r.root=e?e.root:r,r.emit=ev.bind(null,r),i.ce&&i.ce(r),r}let In=null;const Tv=()=>In||fi;let $l,ju;{const i=gc(),e=(t,n)=>{let s;return(s=i[t])||(s=i[t]=[]),s.push(n),r=>{s.length>1?s.forEach(o=>o(r)):s[0](r)}};$l=e("__VUE_INSTANCE_SETTERS__",t=>In=t),ju=e("__VUE_SSR_SETTERS__",t=>Ca=t)}const Xa=i=>{const e=In;return $l(i),i.scope.on(),()=>{i.scope.off(),$l(e)}},Dh=()=>{In&&In.scope.off(),$l(null)};function Y0(i){return i.vnode.shapeFlag&4}let Ca=!1;function Ev(i,e=!1,t=!1){e&&ju(e);const{props:n,children:s}=i.vnode,r=Y0(i);ov(i,n,r,e),uv(i,s,t||e);const o=r?wv(i,e):void 0;return e&&ju(!1),o}function wv(i,e){const t=i.type;i.accessCache=Object.create(null),i.proxy=new Proxy(i.ctx,X_);const{setup:n}=t;if(n){ps();const s=i.setupContext=n.length>1?Iv(i):null,r=Xa(i),o=Wa(n,i,0,[i.props,s]),a=Wm(o);if(ms(),r(),(a||i.sp)&&!la(i)&&S0(i),a){if(o.then(Dh,Dh),e)return o.then(l=>{Ph(i,l)}).catch(l=>{_c(l,i,0)});i.asyncDep=o}else Ph(i,o)}else Q0(i)}function Ph(i,e,t){it(e)?i.type.__ssrInlineRender?i.ssrRender=e:i.render=e:Mt(e)&&(i.setupState=h0(e)),Q0(i)}function Q0(i,e,t){const n=i.type;i.render||(i.render=n.render||Ni);{const s=Xa(i);ps();try{q_(i)}finally{ms(),s()}}}const Rv={get(i,e){return xn(i,"get",""),i[e]}};function Iv(i){const e=t=>{i.exposed=t||{}};return{attrs:new Proxy(i.attrs,Rv),slots:i.slots,emit:i.emit,expose:e}}function yc(i){return i.exposed?i.exposeProxy||(i.exposeProxy=new Proxy(h0(h_(i.exposed)),{get(e,t){if(t in e)return e[t];if(t in ca)return ca[t](i)},has(e,t){return t in e||t in ca}})):i.proxy}function Dv(i){return it(i)&&"__vccOpts"in i}const fr=(i,e)=>__(i,e,Ca),Pv="3.5.28";let $u;const Fh=typeof window<"u"&&window.trustedTypes;if(Fh)try{$u=Fh.createPolicy("vue",{createHTML:i=>i})}catch{}const K0=$u?i=>$u.createHTML(i):i=>i,Fv="http://www.w3.org/2000/svg",Lv="http://www.w3.org/1998/Math/MathML",is=typeof document<"u"?document:null,Lh=is&&is.createElement("template"),Bv={insert:(i,e,t)=>{e.insertBefore(i,t||null)},remove:i=>{const e=i.parentNode;e&&e.removeChild(i)},createElement:(i,e,t,n)=>{const s=e==="svg"?is.createElementNS(Fv,i):e==="mathml"?is.createElementNS(Lv,i):t?is.createElement(i,{is:t}):is.createElement(i);return i==="select"&&n&&n.multiple!=null&&s.setAttribute("multiple",n.multiple),s},createText:i=>is.createTextNode(i),createComment:i=>is.createComment(i),setText:(i,e)=>{i.nodeValue=e},setElementText:(i,e)=>{i.textContent=e},parentNode:i=>i.parentNode,nextSibling:i=>i.nextSibling,querySelector:i=>is.querySelector(i),setScopeId(i,e){i.setAttribute(e,"")},insertStaticContent(i,e,t,n,s,r){const o=t?t.previousSibling:e.lastChild;if(s&&(s===r||s.nextSibling))for(;e.insertBefore(s.cloneNode(!0),t),!(s===r||!(s=s.nextSibling)););else{Lh.innerHTML=K0(n==="svg"?`<svg>${i}</svg>`:n==="mathml"?`<math>${i}</math>`:i);const a=Lh.content;if(n==="svg"||n==="mathml"){const l=a.firstChild;for(;l.firstChild;)a.appendChild(l.firstChild);a.removeChild(l)}e.insertBefore(a,t)}return[o?o.nextSibling:e.firstChild,t?t.previousSibling:e.lastChild]}},Uv=Symbol("_vtc");function Ov(i,e,t){const n=i[Uv];n&&(e=(e?[e,...n]:[...n]).join(" ")),e==null?i.removeAttribute("class"):t?i.setAttribute("class",e):i.className=e}const Bh=Symbol("_vod"),Nv=Symbol("_vsh"),zv=Symbol(""),kv=/(?:^|;)\s*display\s*:/;function Hv(i,e,t){const n=i.style,s=$t(t);let r=!1;if(t&&!s){if(e)if($t(e))for(const o of e.split(";")){const a=o.slice(0,o.indexOf(":")).trim();t[a]==null&&Ol(n,a,"")}else for(const o in e)t[o]==null&&Ol(n,o,"");for(const o in t)o==="display"&&(r=!0),Ol(n,o,t[o])}else if(s){if(e!==t){const o=n[zv];o&&(t+=";"+o),n.cssText=t,r=kv.test(t)}}else e&&i.removeAttribute("style");Bh in i&&(i[Bh]=r?n.display:"",i[Nv]&&(n.display="none"))}const Uh=/\s*!important$/;function Ol(i,e,t){if(Je(t))t.forEach(n=>Ol(i,e,n));else if(t==null&&(t=""),e.startsWith("--"))i.setProperty(e,t);else{const n=Vv(i,e);Uh.test(t)?i.setProperty(js(n),t.replace(Uh,""),"important"):i[n]=t}}const Oh=["Webkit","Moz","ms"],Gc={};function Vv(i,e){const t=Gc[e];if(t)return t;let n=Ws(e);if(n!=="filter"&&n in i)return Gc[e]=n;n=Ym(n);for(let s=0;s<Oh.length;s++){const r=Oh[s]+n;if(r in i)return Gc[e]=r}return e}const Nh="http://www.w3.org/1999/xlink";function zh(i,e,t,n,s,r=Wx(e)){n&&e.startsWith("xlink:")?t==null?i.removeAttributeNS(Nh,e.slice(6,e.length)):i.setAttributeNS(Nh,e,t):t==null||r&&!Km(t)?i.removeAttribute(e):i.setAttribute(e,r?"":ki(t)?String(t):t)}function kh(i,e,t,n,s){if(e==="innerHTML"||e==="textContent"){t!=null&&(i[e]=e==="innerHTML"?K0(t):t);return}const r=i.tagName;if(e==="value"&&r!=="PROGRESS"&&!r.includes("-")){const a=r==="OPTION"?i.getAttribute("value")||"":i.value,l=t==null?i.type==="checkbox"?"on":"":String(t);(a!==l||!("_value"in i))&&(i.value=l),t==null&&i.removeAttribute(e),i._value=t;return}let o=!1;if(t===""||t==null){const a=typeof i[e];a==="boolean"?t=Km(t):t==null&&a==="string"?(t="",o=!0):a==="number"&&(t=0,o=!0)}try{i[e]=t}catch{}o&&i.removeAttribute(s||e)}function mr(i,e,t,n){i.addEventListener(e,t,n)}function Gv(i,e,t,n){i.removeEventListener(e,t,n)}const Hh=Symbol("_vei");function Wv(i,e,t,n,s=null){const r=i[Hh]||(i[Hh]={}),o=r[e];if(n&&o)o.value=n;else{const[a,l]=Xv(e);if(n){const c=r[e]=Qv(n,s);mr(i,a,c,l)}else o&&(Gv(i,a,o,l),r[e]=void 0)}}const Vh=/(?:Once|Passive|Capture)$/;function Xv(i){let e;if(Vh.test(i)){e={};let n;for(;n=i.match(Vh);)i=i.slice(0,i.length-n[0].length),e[n[0].toLowerCase()]=!0}return[i[2]===":"?i.slice(3):js(i.slice(2)),e]}let Wc=0;const qv=Promise.resolve(),Yv=()=>Wc||(qv.then(()=>Wc=0),Wc=Date.now());function Qv(i,e){const t=n=>{if(!n._vts)n._vts=Date.now();else if(n._vts<=t.attached)return;Hi(Kv(n,t.value),e,5,[n])};return t.value=i,t.attached=Yv(),t}function Kv(i,e){if(Je(e)){const t=i.stopImmediatePropagation;return i.stopImmediatePropagation=()=>{t.call(i),i._stopped=!0},e.map(n=>s=>!s._stopped&&n&&n(s))}else return e}const Gh=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&i.charCodeAt(2)>96&&i.charCodeAt(2)<123,jv=(i,e,t,n,s,r)=>{const o=s==="svg";e==="class"?Ov(i,n,o):e==="style"?Hv(i,t,n):hc(e)?id(e)||Wv(i,e,t,n,r):(e[0]==="."?(e=e.slice(1),!0):e[0]==="^"?(e=e.slice(1),!1):$v(i,e,n,o))?(kh(i,e,n),!i.tagName.includes("-")&&(e==="value"||e==="checked"||e==="selected")&&zh(i,e,n,o,r,e!=="value")):i._isVueCE&&(/[A-Z]/.test(e)||!$t(n))?kh(i,Ws(e),n,r,e):(e==="true-value"?i._trueValue=n:e==="false-value"&&(i._falseValue=n),zh(i,e,n,o))};function $v(i,e,t,n){if(n)return!!(e==="innerHTML"||e==="textContent"||e in i&&Gh(e)&&it(t));if(e==="spellcheck"||e==="draggable"||e==="translate"||e==="autocorrect"||e==="sandbox"&&i.tagName==="IFRAME"||e==="form"||e==="list"&&i.tagName==="INPUT"||e==="type"&&i.tagName==="TEXTAREA")return!1;if(e==="width"||e==="height"){const s=i.tagName;if(s==="IMG"||s==="VIDEO"||s==="CANVAS"||s==="SOURCE")return!1}return Gh(e)&&$t(t)?!1:e in i}const Zl=i=>{const e=i.props["onUpdate:modelValue"]||!1;return Je(e)?t=>Ll(e,t):e};function Zv(i){i.target.composing=!0}function Wh(i){const e=i.target;e.composing&&(e.composing=!1,e.dispatchEvent(new Event("input")))}const fo=Symbol("_assign");function Xh(i,e,t){return e&&(i=i.trim()),t&&(i=od(i)),i}const el={created(i,{modifiers:{lazy:e,trim:t,number:n}},s){i[fo]=Zl(s);const r=n||s.props&&s.props.type==="number";mr(i,e?"change":"input",o=>{o.target.composing||i[fo](Xh(i.value,t,r))}),(t||r)&&mr(i,"change",()=>{i.value=Xh(i.value,t,r)}),e||(mr(i,"compositionstart",Zv),mr(i,"compositionend",Wh),mr(i,"change",Wh))},mounted(i,{value:e}){i.value=e??""},beforeUpdate(i,{value:e,oldValue:t,modifiers:{lazy:n,trim:s,number:r}},o){if(i[fo]=Zl(o),i.composing)return;const a=(r||i.type==="number")&&!/^0\d/.test(i.value)?od(i.value):i.value,l=e??"";a!==l&&(document.activeElement===i&&i.type!=="range"&&(n&&e===t||s&&i.value.trim()===l)||(i.value=l))}},Jv={deep:!0,created(i,e,t){i[fo]=Zl(t),mr(i,"change",()=>{const n=i._modelValue,s=eA(i),r=i.checked,o=i[fo];if(Je(n)){const a=jm(n,s),l=a!==-1;if(r&&!l)o(n.concat(s));else if(!r&&l){const c=[...n];c.splice(a,1),o(c)}}else if(pc(n)){const a=new Set(n);r?a.add(s):a.delete(s),o(a)}else o(j0(i,r))})},mounted:qh,beforeUpdate(i,e,t){i[fo]=Zl(t),qh(i,e,t)}};function qh(i,{value:e,oldValue:t},n){i._modelValue=e;let s;if(Je(e))s=jm(e,n.props.value)>-1;else if(pc(e))s=e.has(n.props.value);else{if(e===t)return;s=Ga(e,j0(i,!0))}i.checked!==s&&(i.checked=s)}function eA(i){return"_value"in i?i._value:i.value}function j0(i,e){const t=e?"_trueValue":"_falseValue";return t in i?i[t]:e}const tA=["ctrl","shift","alt","meta"],nA={stop:i=>i.stopPropagation(),prevent:i=>i.preventDefault(),self:i=>i.target!==i.currentTarget,ctrl:i=>!i.ctrlKey,shift:i=>!i.shiftKey,alt:i=>!i.altKey,meta:i=>!i.metaKey,left:i=>"button"in i&&i.button!==0,middle:i=>"button"in i&&i.button!==1,right:i=>"button"in i&&i.button!==2,exact:(i,e)=>tA.some(t=>i[`${t}Key`]&&!e.includes(t))},At=(i,e)=>{if(!i)return i;const t=i._withMods||(i._withMods={}),n=e.join(".");return t[n]||(t[n]=((s,...r)=>{for(let o=0;o<e.length;o++){const a=nA[e[o]];if(a&&a(s,e))return}return i(s,...r)}))},iA={esc:"escape",space:" ",up:"arrow-up",left:"arrow-left",right:"arrow-right",down:"arrow-down",delete:"backspace"},sA=(i,e)=>{const t=i._withKeys||(i._withKeys={}),n=e.join(".");return t[n]||(t[n]=(s=>{if(!("key"in s))return;const r=js(s.key);if(e.some(o=>o===r||iA[o]===r))return i(s)}))},rA=yn({patchProp:jv},Bv);let Yh;function oA(){return Yh||(Yh=dv(rA))}const aA=((...i)=>{const e=oA().createApp(...i),{mount:t}=e;return e.mount=n=>{const s=cA(n);if(!s)return;const r=e._component;!it(r)&&!r.render&&!r.template&&(r.template=s.innerHTML),s.nodeType===1&&(s.textContent="");const o=t(s,!1,lA(s));return s instanceof Element&&(s.removeAttribute("v-cloak"),s.setAttribute("data-v-app","")),o},e});function lA(i){if(i instanceof SVGElement)return"svg";if(typeof MathMLElement=="function"&&i instanceof MathMLElement)return"mathml"}function cA(i){return $t(i)?document.querySelector(i):i}const Sd="181",Lr={ROTATE:0,DOLLY:1,PAN:2},Br={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},uA=0,Qh=1,fA=2,$0=1,dA=2,ts=3,Vi=0,kn=1,ui=2,ds=0,ks=1,Kh=2,jh=3,$h=4,Z0=5,gr=100,hA=101,pA=102,mA=103,gA=104,xA=200,_A=201,vA=202,AA=203,Ta=204,Ea=205,SA=206,yA=207,bA=208,MA=209,CA=210,TA=211,EA=212,wA=213,RA=214,Zu=0,Ju=1,ef=2,bo=3,tf=4,nf=5,sf=6,rf=7,J0=0,IA=1,DA=2,Hs=0,PA=1,FA=2,LA=3,BA=4,UA=5,OA=6,NA=7,eg=300,Mo=301,Co=302,of=303,af=304,bc=306,lf=1e3,fs=1001,cf=1002,ti=1003,zA=1004,tl=1005,di=1006,Xc=1007,_r=1008,Gi=1009,tg=1010,ng=1011,wa=1012,yd=1013,hi=1014,bi=1015,wr=1016,bd=1017,Md=1018,Ra=1020,ig=35902,sg=35899,rg=1021,og=1022,Pn=1023,To=1026,Ia=1027,ag=1028,Mc=1029,Cd=1030,Td=1031,ho=1033,Nl=33776,zl=33777,kl=33778,Hl=33779,uf=35840,ff=35841,df=35842,hf=35843,pf=36196,mf=37492,gf=37496,xf=37808,_f=37809,vf=37810,Af=37811,Sf=37812,yf=37813,bf=37814,Mf=37815,Cf=37816,Tf=37817,Ef=37818,wf=37819,Rf=37820,If=37821,Df=36492,Pf=36494,Ff=36495,Lf=36283,Bf=36284,Uf=36285,Of=36286,kA=3200,HA=3201,VA=0,GA=1,Fs="",li="srgb",Eo="srgb-linear",Jl="linear",yt="srgb",Ur=7680,Zh=519,WA=512,XA=513,qA=514,lg=515,YA=516,QA=517,KA=518,jA=519,Jh=35044,$A=35048,ep="300 es",Li=2e3,ec=2001;function cg(i){for(let e=i.length-1;e>=0;--e)if(i[e]>=65535)return!0;return!1}function tc(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function ZA(){const i=tc("canvas");return i.style.display="block",i}const tp={};function np(...i){const e="THREE."+i.shift();console.log(e,...i)}function et(...i){const e="THREE."+i.shift();console.warn(e,...i)}function Xt(...i){const e="THREE."+i.shift();console.error(e,...i)}function Da(...i){const e=i.join(" ");e in tp||(tp[e]=!0,et(...i))}function JA(i,e,t){return new Promise(function(n,s){function r(){switch(i.clientWaitSync(e,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:s();break;case i.TIMEOUT_EXPIRED:setTimeout(r,t);break;default:n()}}setTimeout(r,t)})}class Rr{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[e]===void 0&&(n[e]=[]),n[e].indexOf(t)===-1&&n[e].push(t)}hasEventListener(e,t){const n=this._listeners;return n===void 0?!1:n[e]!==void 0&&n[e].indexOf(t)!==-1}removeEventListener(e,t){const n=this._listeners;if(n===void 0)return;const s=n[e];if(s!==void 0){const r=s.indexOf(t);r!==-1&&s.splice(r,1)}}dispatchEvent(e){const t=this._listeners;if(t===void 0)return;const n=t[e.type];if(n!==void 0){e.target=this;const s=n.slice(0);for(let r=0,o=s.length;r<o;r++)s[r].call(this,e);e.target=null}}}const hn=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"];let ip=1234567;const fa=Math.PI/180,Pa=180/Math.PI;function Oo(){const i=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(hn[i&255]+hn[i>>8&255]+hn[i>>16&255]+hn[i>>24&255]+"-"+hn[e&255]+hn[e>>8&255]+"-"+hn[e>>16&15|64]+hn[e>>24&255]+"-"+hn[t&63|128]+hn[t>>8&255]+"-"+hn[t>>16&255]+hn[t>>24&255]+hn[n&255]+hn[n>>8&255]+hn[n>>16&255]+hn[n>>24&255]).toLowerCase()}function nt(i,e,t){return Math.max(e,Math.min(t,i))}function Ed(i,e){return(i%e+e)%e}function eS(i,e,t,n,s){return n+(i-e)*(s-n)/(t-e)}function tS(i,e,t){return i!==e?(t-i)/(e-i):0}function da(i,e,t){return(1-t)*i+t*e}function nS(i,e,t,n){return da(i,e,1-Math.exp(-t*n))}function iS(i,e=1){return e-Math.abs(Ed(i,e*2)-e)}function sS(i,e,t){return i<=e?0:i>=t?1:(i=(i-e)/(t-e),i*i*(3-2*i))}function rS(i,e,t){return i<=e?0:i>=t?1:(i=(i-e)/(t-e),i*i*i*(i*(i*6-15)+10))}function oS(i,e){return i+Math.floor(Math.random()*(e-i+1))}function aS(i,e){return i+Math.random()*(e-i)}function lS(i){return i*(.5-Math.random())}function cS(i){i!==void 0&&(ip=i);let e=ip+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=e+Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}function uS(i){return i*fa}function fS(i){return i*Pa}function dS(i){return(i&i-1)===0&&i!==0}function hS(i){return Math.pow(2,Math.ceil(Math.log(i)/Math.LN2))}function pS(i){return Math.pow(2,Math.floor(Math.log(i)/Math.LN2))}function mS(i,e,t,n,s){const r=Math.cos,o=Math.sin,a=r(t/2),l=o(t/2),c=r((e+n)/2),u=o((e+n)/2),f=r((e-n)/2),d=o((e-n)/2),h=r((n-e)/2),x=o((n-e)/2);switch(s){case"XYX":i.set(a*u,l*f,l*d,a*c);break;case"YZY":i.set(l*d,a*u,l*f,a*c);break;case"ZXZ":i.set(l*f,l*d,a*u,a*c);break;case"XZX":i.set(a*u,l*x,l*h,a*c);break;case"YXY":i.set(l*h,a*u,l*x,a*c);break;case"ZYZ":i.set(l*x,l*h,a*u,a*c);break;default:et("MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+s)}}function to(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return i/4294967295;case Uint16Array:return i/65535;case Uint8Array:return i/255;case Int32Array:return Math.max(i/2147483647,-1);case Int16Array:return Math.max(i/32767,-1);case Int8Array:return Math.max(i/127,-1);default:throw new Error("Invalid component type.")}}function Tn(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return Math.round(i*4294967295);case Uint16Array:return Math.round(i*65535);case Uint8Array:return Math.round(i*255);case Int32Array:return Math.round(i*2147483647);case Int16Array:return Math.round(i*32767);case Int8Array:return Math.round(i*127);default:throw new Error("Invalid component type.")}}const Ds={DEG2RAD:fa,RAD2DEG:Pa,generateUUID:Oo,clamp:nt,euclideanModulo:Ed,mapLinear:eS,inverseLerp:tS,lerp:da,damp:nS,pingpong:iS,smoothstep:sS,smootherstep:rS,randInt:oS,randFloat:aS,randFloatSpread:lS,seededRandom:cS,degToRad:uS,radToDeg:fS,isPowerOfTwo:dS,ceilPowerOfTwo:hS,floorPowerOfTwo:pS,setQuaternionFromProperEuler:mS,normalize:Tn,denormalize:to};class Xe{constructor(e=0,t=0){Xe.prototype.isVector2=!0,this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const t=this.x,n=this.y,s=e.elements;return this.x=s[0]*t+s[3]*n+s[6],this.y=s[1]*t+s[4]*n+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=nt(this.x,e.x,t.x),this.y=nt(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=nt(this.x,e,t),this.y=nt(this.y,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(nt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(nt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y;return t*t+n*n}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){const n=Math.cos(t),s=Math.sin(t),r=this.x-e.x,o=this.y-e.y;return this.x=r*n-o*s+e.x,this.y=r*s+o*n+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class Ft{constructor(e=0,t=0,n=0,s=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=n,this._w=s}static slerpFlat(e,t,n,s,r,o,a){let l=n[s+0],c=n[s+1],u=n[s+2],f=n[s+3],d=r[o+0],h=r[o+1],x=r[o+2],p=r[o+3];if(a<=0){e[t+0]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f;return}if(a>=1){e[t+0]=d,e[t+1]=h,e[t+2]=x,e[t+3]=p;return}if(f!==p||l!==d||c!==h||u!==x){let g=l*d+c*h+u*x+f*p;g<0&&(d=-d,h=-h,x=-x,p=-p,g=-g);let m=1-a;if(g<.9995){const _=Math.acos(g),v=Math.sin(_);m=Math.sin(m*_)/v,a=Math.sin(a*_)/v,l=l*m+d*a,c=c*m+h*a,u=u*m+x*a,f=f*m+p*a}else{l=l*m+d*a,c=c*m+h*a,u=u*m+x*a,f=f*m+p*a;const _=1/Math.sqrt(l*l+c*c+u*u+f*f);l*=_,c*=_,u*=_,f*=_}}e[t]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f}static multiplyQuaternionsFlat(e,t,n,s,r,o){const a=n[s],l=n[s+1],c=n[s+2],u=n[s+3],f=r[o],d=r[o+1],h=r[o+2],x=r[o+3];return e[t]=a*x+u*f+l*h-c*d,e[t+1]=l*x+u*d+c*f-a*h,e[t+2]=c*x+u*h+a*d-l*f,e[t+3]=u*x-a*f-l*d-c*h,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,n,s){return this._x=e,this._y=t,this._z=n,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){const n=e._x,s=e._y,r=e._z,o=e._order,a=Math.cos,l=Math.sin,c=a(n/2),u=a(s/2),f=a(r/2),d=l(n/2),h=l(s/2),x=l(r/2);switch(o){case"XYZ":this._x=d*u*f+c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f-d*h*x;break;case"YXZ":this._x=d*u*f+c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f+d*h*x;break;case"ZXY":this._x=d*u*f-c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f-d*h*x;break;case"ZYX":this._x=d*u*f-c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f+d*h*x;break;case"YZX":this._x=d*u*f+c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f-d*h*x;break;case"XZY":this._x=d*u*f-c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f+d*h*x;break;default:et("Quaternion: .setFromEuler() encountered an unknown order: "+o)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){const n=t/2,s=Math.sin(n);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(e){const t=e.elements,n=t[0],s=t[4],r=t[8],o=t[1],a=t[5],l=t[9],c=t[2],u=t[6],f=t[10],d=n+a+f;if(d>0){const h=.5/Math.sqrt(d+1);this._w=.25/h,this._x=(u-l)*h,this._y=(r-c)*h,this._z=(o-s)*h}else if(n>a&&n>f){const h=2*Math.sqrt(1+n-a-f);this._w=(u-l)/h,this._x=.25*h,this._y=(s+o)/h,this._z=(r+c)/h}else if(a>f){const h=2*Math.sqrt(1+a-n-f);this._w=(r-c)/h,this._x=(s+o)/h,this._y=.25*h,this._z=(l+u)/h}else{const h=2*Math.sqrt(1+f-n-a);this._w=(o-s)/h,this._x=(r+c)/h,this._y=(l+u)/h,this._z=.25*h}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let n=e.dot(t)+1;return n<1e-8?(n=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=n):(this._x=0,this._y=-e.z,this._z=e.y,this._w=n)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=n),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(nt(this.dot(e),-1,1)))}rotateTowards(e,t){const n=this.angleTo(e);if(n===0)return this;const s=Math.min(1,t/n);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){const n=e._x,s=e._y,r=e._z,o=e._w,a=t._x,l=t._y,c=t._z,u=t._w;return this._x=n*u+o*a+s*c-r*l,this._y=s*u+o*l+r*a-n*c,this._z=r*u+o*c+n*l-s*a,this._w=o*u-n*a-s*l-r*c,this._onChangeCallback(),this}slerp(e,t){if(t<=0)return this;if(t>=1)return this.copy(e);let n=e._x,s=e._y,r=e._z,o=e._w,a=this.dot(e);a<0&&(n=-n,s=-s,r=-r,o=-o,a=-a);let l=1-t;if(a<.9995){const c=Math.acos(a),u=Math.sin(c);l=Math.sin(l*c)/u,t=Math.sin(t*c)/u,this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this._onChangeCallback()}else this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this.normalize();return this}slerpQuaternions(e,t,n){return this.copy(e).slerp(t,n)}random(){const e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),n=Math.random(),s=Math.sqrt(1-n),r=Math.sqrt(n);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(t),r*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class U{constructor(e=0,t=0,n=0){U.prototype.isVector3=!0,this.x=e,this.y=t,this.z=n}set(e,t,n){return n===void 0&&(n=this.z),this.x=e,this.y=t,this.z=n,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(sp.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(sp.setFromAxisAngle(e,t))}applyMatrix3(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[3]*n+r[6]*s,this.y=r[1]*t+r[4]*n+r[7]*s,this.z=r[2]*t+r[5]*n+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=e.elements,o=1/(r[3]*t+r[7]*n+r[11]*s+r[15]);return this.x=(r[0]*t+r[4]*n+r[8]*s+r[12])*o,this.y=(r[1]*t+r[5]*n+r[9]*s+r[13])*o,this.z=(r[2]*t+r[6]*n+r[10]*s+r[14])*o,this}applyQuaternion(e){const t=this.x,n=this.y,s=this.z,r=e.x,o=e.y,a=e.z,l=e.w,c=2*(o*s-a*n),u=2*(a*t-r*s),f=2*(r*n-o*t);return this.x=t+l*c+o*f-a*u,this.y=n+l*u+a*c-r*f,this.z=s+l*f+r*u-o*c,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[4]*n+r[8]*s,this.y=r[1]*t+r[5]*n+r[9]*s,this.z=r[2]*t+r[6]*n+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=nt(this.x,e.x,t.x),this.y=nt(this.y,e.y,t.y),this.z=nt(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=nt(this.x,e,t),this.y=nt(this.y,e,t),this.z=nt(this.z,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(nt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){const n=e.x,s=e.y,r=e.z,o=t.x,a=t.y,l=t.z;return this.x=s*l-r*a,this.y=r*o-n*l,this.z=n*a-s*o,this}projectOnVector(e){const t=e.lengthSq();if(t===0)return this.set(0,0,0);const n=e.dot(this)/t;return this.copy(e).multiplyScalar(n)}projectOnPlane(e){return qc.copy(this).projectOnVector(e),this.sub(qc)}reflect(e){return this.sub(qc.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(nt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y,s=this.z-e.z;return t*t+n*n+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,n){const s=Math.sin(t)*e;return this.x=s*Math.sin(n),this.y=Math.cos(t)*e,this.z=s*Math.cos(n),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,n){return this.x=e*Math.sin(t),this.y=n,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){const t=this.setFromMatrixColumn(e,0).length(),n=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=n,this.z=s,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,t=Math.random()*2-1,n=Math.sqrt(1-t*t);return this.x=n*Math.cos(e),this.y=t,this.z=n*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const qc=new U,sp=new Ft;class Ze{constructor(e,t,n,s,r,o,a,l,c){Ze.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c)}set(e,t,n,s,r,o,a,l,c){const u=this.elements;return u[0]=e,u[1]=s,u[2]=a,u[3]=t,u[4]=r,u[5]=l,u[6]=n,u[7]=o,u[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],this}extractBasis(e,t,n){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[3],l=n[6],c=n[1],u=n[4],f=n[7],d=n[2],h=n[5],x=n[8],p=s[0],g=s[3],m=s[6],_=s[1],v=s[4],A=s[7],S=s[2],b=s[5],M=s[8];return r[0]=o*p+a*_+l*S,r[3]=o*g+a*v+l*b,r[6]=o*m+a*A+l*M,r[1]=c*p+u*_+f*S,r[4]=c*g+u*v+f*b,r[7]=c*m+u*A+f*M,r[2]=d*p+h*_+x*S,r[5]=d*g+h*v+x*b,r[8]=d*m+h*A+x*M,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8];return t*o*u-t*a*c-n*r*u+n*a*l+s*r*c-s*o*l}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=u*o-a*c,d=a*l-u*r,h=c*r-o*l,x=t*f+n*d+s*h;if(x===0)return this.set(0,0,0,0,0,0,0,0,0);const p=1/x;return e[0]=f*p,e[1]=(s*c-u*n)*p,e[2]=(a*n-s*o)*p,e[3]=d*p,e[4]=(u*t-s*l)*p,e[5]=(s*r-a*t)*p,e[6]=h*p,e[7]=(n*l-c*t)*p,e[8]=(o*t-n*r)*p,this}transpose(){let e;const t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,n,s,r,o,a){const l=Math.cos(r),c=Math.sin(r);return this.set(n*l,n*c,-n*(l*o+c*a)+o+e,-s*c,s*l,-s*(-c*o+l*a)+a+t,0,0,1),this}scale(e,t){return this.premultiply(Yc.makeScale(e,t)),this}rotate(e){return this.premultiply(Yc.makeRotation(-e)),this}translate(e,t){return this.premultiply(Yc.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,n,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<9;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<9;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e}clone(){return new this.constructor().fromArray(this.elements)}}const Yc=new Ze,rp=new Ze().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),op=new Ze().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function gS(){const i={enabled:!0,workingColorSpace:Eo,spaces:{},convert:function(s,r,o){return this.enabled===!1||r===o||!r||!o||(this.spaces[r].transfer===yt&&(s.r=hs(s.r),s.g=hs(s.g),s.b=hs(s.b)),this.spaces[r].primaries!==this.spaces[o].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[o].fromXYZ)),this.spaces[o].transfer===yt&&(s.r=po(s.r),s.g=po(s.g),s.b=po(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===Fs?Jl:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,o){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[o].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return Da("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),i.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return Da("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),i.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],n=[.3127,.329];return i.define({[Eo]:{primaries:e,whitePoint:n,transfer:Jl,toXYZ:rp,fromXYZ:op,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:li},outputColorSpaceConfig:{drawingBufferColorSpace:li}},[li]:{primaries:e,whitePoint:n,transfer:yt,toXYZ:rp,fromXYZ:op,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:li}}}),i}const dt=gS();function hs(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function po(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}let Or;class xS{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let n;if(e instanceof HTMLCanvasElement)n=e;else{Or===void 0&&(Or=tc("canvas")),Or.width=e.width,Or.height=e.height;const s=Or.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),n=Or}return n.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const t=tc("canvas");t.width=e.width,t.height=e.height;const n=t.getContext("2d");n.drawImage(e,0,0,e.width,e.height);const s=n.getImageData(0,0,e.width,e.height),r=s.data;for(let o=0;o<r.length;o++)r[o]=hs(r[o]/255)*255;return n.putImageData(s,0,0),t}else if(e.data){const t=e.data.slice(0);for(let n=0;n<t.length;n++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[n]=Math.floor(hs(t[n]/255)*255):t[n]=hs(t[n]);return{data:t,width:e.width,height:e.height}}else return et("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let _S=0;class wd{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:_S++}),this.uuid=Oo(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):t instanceof VideoFrame?e.set(t.displayHeight,t.displayWidth,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const n={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let o=0,a=s.length;o<a;o++)s[o].isDataTexture?r.push(Qc(s[o].image)):r.push(Qc(s[o]))}else r=Qc(s);n.url=r}return t||(e.images[this.uuid]=n),n}}function Qc(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?xS.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(et("Texture: Unable to serialize Texture."),{})}let vS=0;const Kc=new U;class Fn extends Rr{constructor(e=Fn.DEFAULT_IMAGE,t=Fn.DEFAULT_MAPPING,n=fs,s=fs,r=di,o=_r,a=Pn,l=Gi,c=Fn.DEFAULT_ANISOTROPY,u=Fs){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:vS++}),this.uuid=Oo(),this.name="",this.source=new wd(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=n,this.wrapT=s,this.magFilter=r,this.minFilter=o,this.anisotropy=c,this.format=a,this.internalFormat=null,this.type=l,this.offset=new Xe(0,0),this.repeat=new Xe(1,1),this.center=new Xe(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Ze,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=u,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0}get width(){return this.source.getSize(Kc).x}get height(){return this.source.getSize(Kc).y}get depth(){return this.source.getSize(Kc).z}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const t in e){const n=e[t];if(n===void 0){et(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){et(`Texture.setValues(): property '${t}' does not exist.`);continue}s&&n&&s.isVector2&&n.isVector2||s&&n&&s.isVector3&&n.isVector3||s&&n&&s.isMatrix3&&n.isMatrix3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const n={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),t||(e.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==eg)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case lf:e.x=e.x-Math.floor(e.x);break;case fs:e.x=e.x<0?0:1;break;case cf:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case lf:e.y=e.y-Math.floor(e.y);break;case fs:e.y=e.y<0?0:1;break;case cf:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}Fn.DEFAULT_IMAGE=null;Fn.DEFAULT_MAPPING=eg;Fn.DEFAULT_ANISOTROPY=1;class Ot{constructor(e=0,t=0,n=0,s=1){Ot.prototype.isVector4=!0,this.x=e,this.y=t,this.z=n,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,n,s){return this.x=e,this.y=t,this.z=n,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=this.w,o=e.elements;return this.x=o[0]*t+o[4]*n+o[8]*s+o[12]*r,this.y=o[1]*t+o[5]*n+o[9]*s+o[13]*r,this.z=o[2]*t+o[6]*n+o[10]*s+o[14]*r,this.w=o[3]*t+o[7]*n+o[11]*s+o[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,n,s,r;const l=e.elements,c=l[0],u=l[4],f=l[8],d=l[1],h=l[5],x=l[9],p=l[2],g=l[6],m=l[10];if(Math.abs(u-d)<.01&&Math.abs(f-p)<.01&&Math.abs(x-g)<.01){if(Math.abs(u+d)<.1&&Math.abs(f+p)<.1&&Math.abs(x+g)<.1&&Math.abs(c+h+m-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;const v=(c+1)/2,A=(h+1)/2,S=(m+1)/2,b=(u+d)/4,M=(f+p)/4,R=(x+g)/4;return v>A&&v>S?v<.01?(n=0,s=.707106781,r=.707106781):(n=Math.sqrt(v),s=b/n,r=M/n):A>S?A<.01?(n=.707106781,s=0,r=.707106781):(s=Math.sqrt(A),n=b/s,r=R/s):S<.01?(n=.707106781,s=.707106781,r=0):(r=Math.sqrt(S),n=M/r,s=R/r),this.set(n,s,r,t),this}let _=Math.sqrt((g-x)*(g-x)+(f-p)*(f-p)+(d-u)*(d-u));return Math.abs(_)<.001&&(_=1),this.x=(g-x)/_,this.y=(f-p)/_,this.z=(d-u)/_,this.w=Math.acos((c+h+m-1)/2),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=nt(this.x,e.x,t.x),this.y=nt(this.y,e.y,t.y),this.z=nt(this.z,e.z,t.z),this.w=nt(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=nt(this.x,e,t),this.y=nt(this.y,e,t),this.z=nt(this.z,e,t),this.w=nt(this.w,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(nt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this.w=e.w+(t.w-e.w)*n,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class AS extends Rr{constructor(e=1,t=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:di,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},n),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=n.depth,this.scissor=new Ot(0,0,e,t),this.scissorTest=!1,this.viewport=new Ot(0,0,e,t);const s={width:e,height:t,depth:n.depth},r=new Fn(s);this.textures=[];const o=n.count;for(let a=0;a<o;a++)this.textures[a]=r.clone(),this.textures[a].isRenderTargetTexture=!0,this.textures[a].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview}_setTextureOptions(e={}){const t={minFilter:di,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let n=0;n<this.textures.length;n++)this.textures[n].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,n=1){if(this.width!==e||this.height!==t||this.depth!==n){this.width=e,this.height=t,this.depth=n;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=t,this.textures[s].image.depth=n,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,n=e.textures.length;t<n;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;const s=Object.assign({},e.textures[t].image);this.textures[t].source=new wd(s)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class qs extends AS{constructor(e=1,t=1,n={}){super(e,t,n),this.isWebGLRenderTarget=!0}}class ug extends Fn{constructor(e=null,t=1,n=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=ti,this.minFilter=ti,this.wrapR=fs,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class SS extends Fn{constructor(e=null,t=1,n=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=ti,this.minFilter=ti,this.wrapR=fs,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class Bi{constructor(e=new U(1/0,1/0,1/0),t=new U(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t+=3)this.expandByPoint(vi.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,n=e.count;t<n;t++)this.expandByPoint(vi.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){const n=vi.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(n),this.max.copy(e).add(n),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);const n=e.geometry;if(n!==void 0){const r=n.getAttribute("position");if(t===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let o=0,a=r.count;o<a;o++)e.isMesh===!0?e.getVertexPosition(o,vi):vi.fromBufferAttribute(r,o),vi.applyMatrix4(e.matrixWorld),this.expandByPoint(vi);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),nl.copy(e.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),nl.copy(n.boundingBox)),nl.applyMatrix4(e.matrixWorld),this.union(nl)}const s=e.children;for(let r=0,o=s.length;r<o;r++)this.expandByObject(s[r],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,vi),vi.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,n;return e.normal.x>0?(t=e.normal.x*this.min.x,n=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,n=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,n+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,n+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,n+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,n+=e.normal.z*this.min.z),t<=-e.constant&&n>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(Qo),il.subVectors(this.max,Qo),Nr.subVectors(e.a,Qo),zr.subVectors(e.b,Qo),kr.subVectors(e.c,Qo),ys.subVectors(zr,Nr),bs.subVectors(kr,zr),nr.subVectors(Nr,kr);let t=[0,-ys.z,ys.y,0,-bs.z,bs.y,0,-nr.z,nr.y,ys.z,0,-ys.x,bs.z,0,-bs.x,nr.z,0,-nr.x,-ys.y,ys.x,0,-bs.y,bs.x,0,-nr.y,nr.x,0];return!jc(t,Nr,zr,kr,il)||(t=[1,0,0,0,1,0,0,0,1],!jc(t,Nr,zr,kr,il))?!1:(sl.crossVectors(ys,bs),t=[sl.x,sl.y,sl.z],jc(t,Nr,zr,kr,il))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,vi).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(vi).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Qi[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Qi[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Qi[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Qi[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Qi[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Qi[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Qi[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Qi[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Qi),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const Qi=[new U,new U,new U,new U,new U,new U,new U,new U],vi=new U,nl=new Bi,Nr=new U,zr=new U,kr=new U,ys=new U,bs=new U,nr=new U,Qo=new U,il=new U,sl=new U,ir=new U;function jc(i,e,t,n,s){for(let r=0,o=i.length-3;r<=o;r+=3){ir.fromArray(i,r);const a=s.x*Math.abs(ir.x)+s.y*Math.abs(ir.y)+s.z*Math.abs(ir.z),l=e.dot(ir),c=t.dot(ir),u=n.dot(ir);if(Math.max(-Math.max(l,c,u),Math.min(l,c,u))>a)return!1}return!0}const yS=new Bi,Ko=new U,$c=new U;class Cc{constructor(e=new U,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){const n=this.center;t!==void 0?n.copy(t):yS.setFromPoints(e).getCenter(n);let s=0;for(let r=0,o=e.length;r<o;r++)s=Math.max(s,n.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){const n=this.center.distanceToSquared(e);return t.copy(e),n>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;Ko.subVectors(e,this.center);const t=Ko.lengthSq();if(t>this.radius*this.radius){const n=Math.sqrt(t),s=(n-this.radius)*.5;this.center.addScaledVector(Ko,s/n),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):($c.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(Ko.copy(e.center).add($c)),this.expandByPoint(Ko.copy(e.center).sub($c))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}const Ki=new U,Zc=new U,rl=new U,Ms=new U,Jc=new U,ol=new U,eu=new U;let Rd=class{constructor(e=new U,t=new U(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,Ki)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);const n=t.dot(this.direction);return n<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const t=Ki.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(Ki.copy(this.origin).addScaledVector(this.direction,t),Ki.distanceToSquared(e))}distanceSqToSegment(e,t,n,s){Zc.copy(e).add(t).multiplyScalar(.5),rl.copy(t).sub(e).normalize(),Ms.copy(this.origin).sub(Zc);const r=e.distanceTo(t)*.5,o=-this.direction.dot(rl),a=Ms.dot(this.direction),l=-Ms.dot(rl),c=Ms.lengthSq(),u=Math.abs(1-o*o);let f,d,h,x;if(u>0)if(f=o*l-a,d=o*a-l,x=r*u,f>=0)if(d>=-x)if(d<=x){const p=1/u;f*=p,d*=p,h=f*(f+o*d+2*a)+d*(o*f+d+2*l)+c}else d=r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;else d=-r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;else d<=-x?(f=Math.max(0,-(-o*r+a)),d=f>0?-r:Math.min(Math.max(-r,-l),r),h=-f*f+d*(d+2*l)+c):d<=x?(f=0,d=Math.min(Math.max(-r,-l),r),h=d*(d+2*l)+c):(f=Math.max(0,-(o*r+a)),d=f>0?r:Math.min(Math.max(-r,-l),r),h=-f*f+d*(d+2*l)+c);else d=o>0?-r:r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;return n&&n.copy(this.origin).addScaledVector(this.direction,f),s&&s.copy(Zc).addScaledVector(rl,d),h}intersectSphere(e,t){Ki.subVectors(e.center,this.origin);const n=Ki.dot(this.direction),s=Ki.dot(Ki)-n*n,r=e.radius*e.radius;if(s>r)return null;const o=Math.sqrt(r-s),a=n-o,l=n+o;return l<0?null:a<0?this.at(l,t):this.at(a,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(e.normal)+e.constant)/t;return n>=0?n:null}intersectPlane(e,t){const n=this.distanceToPlane(e);return n===null?null:this.at(n,t)}intersectsPlane(e){const t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let n,s,r,o,a,l;const c=1/this.direction.x,u=1/this.direction.y,f=1/this.direction.z,d=this.origin;return c>=0?(n=(e.min.x-d.x)*c,s=(e.max.x-d.x)*c):(n=(e.max.x-d.x)*c,s=(e.min.x-d.x)*c),u>=0?(r=(e.min.y-d.y)*u,o=(e.max.y-d.y)*u):(r=(e.max.y-d.y)*u,o=(e.min.y-d.y)*u),n>o||r>s||((r>n||isNaN(n))&&(n=r),(o<s||isNaN(s))&&(s=o),f>=0?(a=(e.min.z-d.z)*f,l=(e.max.z-d.z)*f):(a=(e.max.z-d.z)*f,l=(e.min.z-d.z)*f),n>l||a>s)||((a>n||n!==n)&&(n=a),(l<s||s!==s)&&(s=l),s<0)?null:this.at(n>=0?n:s,t)}intersectsBox(e){return this.intersectBox(e,Ki)!==null}intersectTriangle(e,t,n,s,r){Jc.subVectors(t,e),ol.subVectors(n,e),eu.crossVectors(Jc,ol);let o=this.direction.dot(eu),a;if(o>0){if(s)return null;a=1}else if(o<0)a=-1,o=-o;else return null;Ms.subVectors(this.origin,e);const l=a*this.direction.dot(ol.crossVectors(Ms,ol));if(l<0)return null;const c=a*this.direction.dot(Jc.cross(Ms));if(c<0||l+c>o)return null;const u=-a*Ms.dot(eu);return u<0?null:this.at(u/o,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}};class $e{constructor(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g){$e.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g)}set(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g){const m=this.elements;return m[0]=e,m[4]=t,m[8]=n,m[12]=s,m[1]=r,m[5]=o,m[9]=a,m[13]=l,m[2]=c,m[6]=u,m[10]=f,m[14]=d,m[3]=h,m[7]=x,m[11]=p,m[15]=g,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new $e().fromArray(this.elements)}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],this}copyPosition(e){const t=this.elements,n=e.elements;return t[12]=n[12],t[13]=n[13],t[14]=n[14],this}setFromMatrix3(e){const t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,n){return e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this}makeBasis(e,t,n){return this.set(e.x,t.x,n.x,0,e.y,t.y,n.y,0,e.z,t.z,n.z,0,0,0,0,1),this}extractRotation(e){const t=this.elements,n=e.elements,s=1/Hr.setFromMatrixColumn(e,0).length(),r=1/Hr.setFromMatrixColumn(e,1).length(),o=1/Hr.setFromMatrixColumn(e,2).length();return t[0]=n[0]*s,t[1]=n[1]*s,t[2]=n[2]*s,t[3]=0,t[4]=n[4]*r,t[5]=n[5]*r,t[6]=n[6]*r,t[7]=0,t[8]=n[8]*o,t[9]=n[9]*o,t[10]=n[10]*o,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){const t=this.elements,n=e.x,s=e.y,r=e.z,o=Math.cos(n),a=Math.sin(n),l=Math.cos(s),c=Math.sin(s),u=Math.cos(r),f=Math.sin(r);if(e.order==="XYZ"){const d=o*u,h=o*f,x=a*u,p=a*f;t[0]=l*u,t[4]=-l*f,t[8]=c,t[1]=h+x*c,t[5]=d-p*c,t[9]=-a*l,t[2]=p-d*c,t[6]=x+h*c,t[10]=o*l}else if(e.order==="YXZ"){const d=l*u,h=l*f,x=c*u,p=c*f;t[0]=d+p*a,t[4]=x*a-h,t[8]=o*c,t[1]=o*f,t[5]=o*u,t[9]=-a,t[2]=h*a-x,t[6]=p+d*a,t[10]=o*l}else if(e.order==="ZXY"){const d=l*u,h=l*f,x=c*u,p=c*f;t[0]=d-p*a,t[4]=-o*f,t[8]=x+h*a,t[1]=h+x*a,t[5]=o*u,t[9]=p-d*a,t[2]=-o*c,t[6]=a,t[10]=o*l}else if(e.order==="ZYX"){const d=o*u,h=o*f,x=a*u,p=a*f;t[0]=l*u,t[4]=x*c-h,t[8]=d*c+p,t[1]=l*f,t[5]=p*c+d,t[9]=h*c-x,t[2]=-c,t[6]=a*l,t[10]=o*l}else if(e.order==="YZX"){const d=o*l,h=o*c,x=a*l,p=a*c;t[0]=l*u,t[4]=p-d*f,t[8]=x*f+h,t[1]=f,t[5]=o*u,t[9]=-a*u,t[2]=-c*u,t[6]=h*f+x,t[10]=d-p*f}else if(e.order==="XZY"){const d=o*l,h=o*c,x=a*l,p=a*c;t[0]=l*u,t[4]=-f,t[8]=c*u,t[1]=d*f+p,t[5]=o*u,t[9]=h*f-x,t[2]=x*f-h,t[6]=a*u,t[10]=p*f+d}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(bS,e,MS)}lookAt(e,t,n){const s=this.elements;return qn.subVectors(e,t),qn.lengthSq()===0&&(qn.z=1),qn.normalize(),Cs.crossVectors(n,qn),Cs.lengthSq()===0&&(Math.abs(n.z)===1?qn.x+=1e-4:qn.z+=1e-4,qn.normalize(),Cs.crossVectors(n,qn)),Cs.normalize(),al.crossVectors(qn,Cs),s[0]=Cs.x,s[4]=al.x,s[8]=qn.x,s[1]=Cs.y,s[5]=al.y,s[9]=qn.y,s[2]=Cs.z,s[6]=al.z,s[10]=qn.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[4],l=n[8],c=n[12],u=n[1],f=n[5],d=n[9],h=n[13],x=n[2],p=n[6],g=n[10],m=n[14],_=n[3],v=n[7],A=n[11],S=n[15],b=s[0],M=s[4],R=s[8],y=s[12],C=s[1],F=s[5],I=s[9],O=s[13],z=s[2],H=s[6],V=s[10],q=s[14],G=s[3],Y=s[7],ae=s[11],Ae=s[15];return r[0]=o*b+a*C+l*z+c*G,r[4]=o*M+a*F+l*H+c*Y,r[8]=o*R+a*I+l*V+c*ae,r[12]=o*y+a*O+l*q+c*Ae,r[1]=u*b+f*C+d*z+h*G,r[5]=u*M+f*F+d*H+h*Y,r[9]=u*R+f*I+d*V+h*ae,r[13]=u*y+f*O+d*q+h*Ae,r[2]=x*b+p*C+g*z+m*G,r[6]=x*M+p*F+g*H+m*Y,r[10]=x*R+p*I+g*V+m*ae,r[14]=x*y+p*O+g*q+m*Ae,r[3]=_*b+v*C+A*z+S*G,r[7]=_*M+v*F+A*H+S*Y,r[11]=_*R+v*I+A*V+S*ae,r[15]=_*y+v*O+A*q+S*Ae,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[4],s=e[8],r=e[12],o=e[1],a=e[5],l=e[9],c=e[13],u=e[2],f=e[6],d=e[10],h=e[14],x=e[3],p=e[7],g=e[11],m=e[15];return x*(+r*l*f-s*c*f-r*a*d+n*c*d+s*a*h-n*l*h)+p*(+t*l*h-t*c*d+r*o*d-s*o*h+s*c*u-r*l*u)+g*(+t*c*f-t*a*h-r*o*f+n*o*h+r*a*u-n*c*u)+m*(-s*a*u-t*l*f+t*a*d+s*o*f-n*o*d+n*l*u)}transpose(){const e=this.elements;let t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,n){const s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=t,s[14]=n),this}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=e[9],d=e[10],h=e[11],x=e[12],p=e[13],g=e[14],m=e[15],_=f*g*c-p*d*c+p*l*h-a*g*h-f*l*m+a*d*m,v=x*d*c-u*g*c-x*l*h+o*g*h+u*l*m-o*d*m,A=u*p*c-x*f*c+x*a*h-o*p*h-u*a*m+o*f*m,S=x*f*l-u*p*l-x*a*d+o*p*d+u*a*g-o*f*g,b=t*_+n*v+s*A+r*S;if(b===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const M=1/b;return e[0]=_*M,e[1]=(p*d*r-f*g*r-p*s*h+n*g*h+f*s*m-n*d*m)*M,e[2]=(a*g*r-p*l*r+p*s*c-n*g*c-a*s*m+n*l*m)*M,e[3]=(f*l*r-a*d*r-f*s*c+n*d*c+a*s*h-n*l*h)*M,e[4]=v*M,e[5]=(u*g*r-x*d*r+x*s*h-t*g*h-u*s*m+t*d*m)*M,e[6]=(x*l*r-o*g*r-x*s*c+t*g*c+o*s*m-t*l*m)*M,e[7]=(o*d*r-u*l*r+u*s*c-t*d*c-o*s*h+t*l*h)*M,e[8]=A*M,e[9]=(x*f*r-u*p*r-x*n*h+t*p*h+u*n*m-t*f*m)*M,e[10]=(o*p*r-x*a*r+x*n*c-t*p*c-o*n*m+t*a*m)*M,e[11]=(u*a*r-o*f*r-u*n*c+t*f*c+o*n*h-t*a*h)*M,e[12]=S*M,e[13]=(u*p*s-x*f*s+x*n*d-t*p*d-u*n*g+t*f*g)*M,e[14]=(x*a*s-o*p*s-x*n*l+t*p*l+o*n*g-t*a*g)*M,e[15]=(o*f*s-u*a*s+u*n*l-t*f*l-o*n*d+t*a*d)*M,this}scale(e){const t=this.elements,n=e.x,s=e.y,r=e.z;return t[0]*=n,t[4]*=s,t[8]*=r,t[1]*=n,t[5]*=s,t[9]*=r,t[2]*=n,t[6]*=s,t[10]*=r,t[3]*=n,t[7]*=s,t[11]*=r,this}getMaxScaleOnAxis(){const e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],n=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,n,s))}makeTranslation(e,t,n){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,n,0,0,0,1),this}makeRotationX(e){const t=Math.cos(e),n=Math.sin(e);return this.set(1,0,0,0,0,t,-n,0,0,n,t,0,0,0,0,1),this}makeRotationY(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,0,n,0,0,1,0,0,-n,0,t,0,0,0,0,1),this}makeRotationZ(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,0,n,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){const n=Math.cos(t),s=Math.sin(t),r=1-n,o=e.x,a=e.y,l=e.z,c=r*o,u=r*a;return this.set(c*o+n,c*a-s*l,c*l+s*a,0,c*a+s*l,u*a+n,u*l-s*o,0,c*l-s*a,u*l+s*o,r*l*l+n,0,0,0,0,1),this}makeScale(e,t,n){return this.set(e,0,0,0,0,t,0,0,0,0,n,0,0,0,0,1),this}makeShear(e,t,n,s,r,o){return this.set(1,n,r,0,e,1,o,0,t,s,1,0,0,0,0,1),this}compose(e,t,n){const s=this.elements,r=t._x,o=t._y,a=t._z,l=t._w,c=r+r,u=o+o,f=a+a,d=r*c,h=r*u,x=r*f,p=o*u,g=o*f,m=a*f,_=l*c,v=l*u,A=l*f,S=n.x,b=n.y,M=n.z;return s[0]=(1-(p+m))*S,s[1]=(h+A)*S,s[2]=(x-v)*S,s[3]=0,s[4]=(h-A)*b,s[5]=(1-(d+m))*b,s[6]=(g+_)*b,s[7]=0,s[8]=(x+v)*M,s[9]=(g-_)*M,s[10]=(1-(d+p))*M,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,t,n){const s=this.elements;let r=Hr.set(s[0],s[1],s[2]).length();const o=Hr.set(s[4],s[5],s[6]).length(),a=Hr.set(s[8],s[9],s[10]).length();this.determinant()<0&&(r=-r),e.x=s[12],e.y=s[13],e.z=s[14],Ai.copy(this);const c=1/r,u=1/o,f=1/a;return Ai.elements[0]*=c,Ai.elements[1]*=c,Ai.elements[2]*=c,Ai.elements[4]*=u,Ai.elements[5]*=u,Ai.elements[6]*=u,Ai.elements[8]*=f,Ai.elements[9]*=f,Ai.elements[10]*=f,t.setFromRotationMatrix(Ai),n.x=r,n.y=o,n.z=a,this}makePerspective(e,t,n,s,r,o,a=Li,l=!1){const c=this.elements,u=2*r/(t-e),f=2*r/(n-s),d=(t+e)/(t-e),h=(n+s)/(n-s);let x,p;if(l)x=r/(o-r),p=o*r/(o-r);else if(a===Li)x=-(o+r)/(o-r),p=-2*o*r/(o-r);else if(a===ec)x=-o/(o-r),p=-o*r/(o-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=d,c[12]=0,c[1]=0,c[5]=f,c[9]=h,c[13]=0,c[2]=0,c[6]=0,c[10]=x,c[14]=p,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(e,t,n,s,r,o,a=Li,l=!1){const c=this.elements,u=2/(t-e),f=2/(n-s),d=-(t+e)/(t-e),h=-(n+s)/(n-s);let x,p;if(l)x=1/(o-r),p=o/(o-r);else if(a===Li)x=-2/(o-r),p=-(o+r)/(o-r);else if(a===ec)x=-1/(o-r),p=-r/(o-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=0,c[12]=d,c[1]=0,c[5]=f,c[9]=0,c[13]=h,c[2]=0,c[6]=0,c[10]=x,c[14]=p,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<16;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<16;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e[t+9]=n[9],e[t+10]=n[10],e[t+11]=n[11],e[t+12]=n[12],e[t+13]=n[13],e[t+14]=n[14],e[t+15]=n[15],e}}const Hr=new U,Ai=new $e,bS=new U(0,0,0),MS=new U(1,1,1),Cs=new U,al=new U,qn=new U,ap=new $e,lp=new Ft;class Ti{constructor(e=0,t=0,n=0,s=Ti.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=n,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,n,s=this._order){return this._x=e,this._y=t,this._z=n,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,n=!0){const s=e.elements,r=s[0],o=s[4],a=s[8],l=s[1],c=s[5],u=s[9],f=s[2],d=s[6],h=s[10];switch(t){case"XYZ":this._y=Math.asin(nt(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-u,h),this._z=Math.atan2(-o,r)):(this._x=Math.atan2(d,c),this._z=0);break;case"YXZ":this._x=Math.asin(-nt(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(a,h),this._z=Math.atan2(l,c)):(this._y=Math.atan2(-f,r),this._z=0);break;case"ZXY":this._x=Math.asin(nt(d,-1,1)),Math.abs(d)<.9999999?(this._y=Math.atan2(-f,h),this._z=Math.atan2(-o,c)):(this._y=0,this._z=Math.atan2(l,r));break;case"ZYX":this._y=Math.asin(-nt(f,-1,1)),Math.abs(f)<.9999999?(this._x=Math.atan2(d,h),this._z=Math.atan2(l,r)):(this._x=0,this._z=Math.atan2(-o,c));break;case"YZX":this._z=Math.asin(nt(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(-u,c),this._y=Math.atan2(-f,r)):(this._x=0,this._y=Math.atan2(a,h));break;case"XZY":this._z=Math.asin(-nt(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(d,c),this._y=Math.atan2(a,r)):(this._x=Math.atan2(-u,h),this._y=0);break;default:et("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,n===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,n){return ap.makeRotationFromQuaternion(e),this.setFromRotationMatrix(ap,t,n)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return lp.setFromEuler(this),this.setFromQuaternion(lp,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}Ti.DEFAULT_ORDER="XYZ";class fg{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let CS=0;const cp=new U,Vr=new Ft,ji=new $e,ll=new U,jo=new U,TS=new U,ES=new Ft,up=new U(1,0,0),fp=new U(0,1,0),dp=new U(0,0,1),hp={type:"added"},wS={type:"removed"},Gr={type:"childadded",child:null},tu={type:"childremoved",child:null};class jt extends Rr{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:CS++}),this.uuid=Oo(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=jt.DEFAULT_UP.clone();const e=new U,t=new Ti,n=new Ft,s=new U(1,1,1);function r(){n.setFromEuler(t,!1)}function o(){t.setFromQuaternion(n,void 0,!1)}t._onChange(r),n._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new $e},normalMatrix:{value:new Ze}}),this.matrix=new $e,this.matrixWorld=new $e,this.matrixAutoUpdate=jt.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=jt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new fg,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return Vr.setFromAxisAngle(e,t),this.quaternion.multiply(Vr),this}rotateOnWorldAxis(e,t){return Vr.setFromAxisAngle(e,t),this.quaternion.premultiply(Vr),this}rotateX(e){return this.rotateOnAxis(up,e)}rotateY(e){return this.rotateOnAxis(fp,e)}rotateZ(e){return this.rotateOnAxis(dp,e)}translateOnAxis(e,t){return cp.copy(e).applyQuaternion(this.quaternion),this.position.add(cp.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(up,e)}translateY(e){return this.translateOnAxis(fp,e)}translateZ(e){return this.translateOnAxis(dp,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(ji.copy(this.matrixWorld).invert())}lookAt(e,t,n){e.isVector3?ll.copy(e):ll.set(e,t,n);const s=this.parent;this.updateWorldMatrix(!0,!1),jo.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?ji.lookAt(jo,ll,this.up):ji.lookAt(ll,jo,this.up),this.quaternion.setFromRotationMatrix(ji),s&&(ji.extractRotation(s.matrixWorld),Vr.setFromRotationMatrix(ji),this.quaternion.premultiply(Vr.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(Xt("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(hp),Gr.child=e,this.dispatchEvent(Gr),Gr.child=null):Xt("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(wS),tu.child=e,this.dispatchEvent(tu),tu.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),ji.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),ji.multiply(e.parent.matrixWorld)),e.applyMatrix4(ji),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(hp),Gr.child=e,this.dispatchEvent(Gr),Gr.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let n=0,s=this.children.length;n<s;n++){const o=this.children[n].getObjectByProperty(e,t);if(o!==void 0)return o}}getObjectsByProperty(e,t,n=[]){this[e]===t&&n.push(this);const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].getObjectsByProperty(e,t,n);return n}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(jo,e,TS),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(jo,ES,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverseVisible(e)}traverseAncestors(e){const t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].updateMatrixWorld(e)}updateWorldMatrix(e,t){const n=this.parent;if(e===!0&&n!==null&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),t===!0){const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].updateWorldMatrix(!1,!0)}}toJSON(e){const t=e===void 0||typeof e=="string",n={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(a=>({...a,boundingBox:a.boundingBox?a.boundingBox.toJSON():void 0,boundingSphere:a.boundingSphere?a.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(a=>({...a})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(a,l){return a[l.uuid]===void 0&&(a[l.uuid]=l.toJSON(e)),l.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);const a=this.geometry.parameters;if(a!==void 0&&a.shapes!==void 0){const l=a.shapes;if(Array.isArray(l))for(let c=0,u=l.length;c<u;c++){const f=l[c];r(e.shapes,f)}else r(e.shapes,l)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const a=[];for(let l=0,c=this.material.length;l<c;l++)a.push(r(e.materials,this.material[l]));s.material=a}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let a=0;a<this.children.length;a++)s.children.push(this.children[a].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let a=0;a<this.animations.length;a++){const l=this.animations[a];s.animations.push(r(e.animations,l))}}if(t){const a=o(e.geometries),l=o(e.materials),c=o(e.textures),u=o(e.images),f=o(e.shapes),d=o(e.skeletons),h=o(e.animations),x=o(e.nodes);a.length>0&&(n.geometries=a),l.length>0&&(n.materials=l),c.length>0&&(n.textures=c),u.length>0&&(n.images=u),f.length>0&&(n.shapes=f),d.length>0&&(n.skeletons=d),h.length>0&&(n.animations=h),x.length>0&&(n.nodes=x)}return n.object=s,n;function o(a){const l=[];for(const c in a){const u=a[c];delete u.metadata,l.push(u)}return l}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let n=0;n<e.children.length;n++){const s=e.children[n];this.add(s.clone())}return this}}jt.DEFAULT_UP=new U(0,1,0);jt.DEFAULT_MATRIX_AUTO_UPDATE=!0;jt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const Si=new U,$i=new U,nu=new U,Zi=new U,Wr=new U,Xr=new U,pp=new U,iu=new U,su=new U,ru=new U,ou=new Ot,au=new Ot,lu=new Ot;class yi{constructor(e=new U,t=new U,n=new U){this.a=e,this.b=t,this.c=n}static getNormal(e,t,n,s){s.subVectors(n,t),Si.subVectors(e,t),s.cross(Si);const r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,t,n,s,r){Si.subVectors(s,t),$i.subVectors(n,t),nu.subVectors(e,t);const o=Si.dot(Si),a=Si.dot($i),l=Si.dot(nu),c=$i.dot($i),u=$i.dot(nu),f=o*c-a*a;if(f===0)return r.set(0,0,0),null;const d=1/f,h=(c*l-a*u)*d,x=(o*u-a*l)*d;return r.set(1-h-x,x,h)}static containsPoint(e,t,n,s){return this.getBarycoord(e,t,n,s,Zi)===null?!1:Zi.x>=0&&Zi.y>=0&&Zi.x+Zi.y<=1}static getInterpolation(e,t,n,s,r,o,a,l){return this.getBarycoord(e,t,n,s,Zi)===null?(l.x=0,l.y=0,"z"in l&&(l.z=0),"w"in l&&(l.w=0),null):(l.setScalar(0),l.addScaledVector(r,Zi.x),l.addScaledVector(o,Zi.y),l.addScaledVector(a,Zi.z),l)}static getInterpolatedAttribute(e,t,n,s,r,o){return ou.setScalar(0),au.setScalar(0),lu.setScalar(0),ou.fromBufferAttribute(e,t),au.fromBufferAttribute(e,n),lu.fromBufferAttribute(e,s),o.setScalar(0),o.addScaledVector(ou,r.x),o.addScaledVector(au,r.y),o.addScaledVector(lu,r.z),o}static isFrontFacing(e,t,n,s){return Si.subVectors(n,t),$i.subVectors(e,t),Si.cross($i).dot(s)<0}set(e,t,n){return this.a.copy(e),this.b.copy(t),this.c.copy(n),this}setFromPointsAndIndices(e,t,n,s){return this.a.copy(e[t]),this.b.copy(e[n]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,t,n,s){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,n),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return Si.subVectors(this.c,this.b),$i.subVectors(this.a,this.b),Si.cross($i).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return yi.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return yi.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,n,s,r){return yi.getInterpolation(e,this.a,this.b,this.c,t,n,s,r)}containsPoint(e){return yi.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return yi.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){const n=this.a,s=this.b,r=this.c;let o,a;Wr.subVectors(s,n),Xr.subVectors(r,n),iu.subVectors(e,n);const l=Wr.dot(iu),c=Xr.dot(iu);if(l<=0&&c<=0)return t.copy(n);su.subVectors(e,s);const u=Wr.dot(su),f=Xr.dot(su);if(u>=0&&f<=u)return t.copy(s);const d=l*f-u*c;if(d<=0&&l>=0&&u<=0)return o=l/(l-u),t.copy(n).addScaledVector(Wr,o);ru.subVectors(e,r);const h=Wr.dot(ru),x=Xr.dot(ru);if(x>=0&&h<=x)return t.copy(r);const p=h*c-l*x;if(p<=0&&c>=0&&x<=0)return a=c/(c-x),t.copy(n).addScaledVector(Xr,a);const g=u*x-h*f;if(g<=0&&f-u>=0&&h-x>=0)return pp.subVectors(r,s),a=(f-u)/(f-u+(h-x)),t.copy(s).addScaledVector(pp,a);const m=1/(g+p+d);return o=p*m,a=d*m,t.copy(n).addScaledVector(Wr,o).addScaledVector(Xr,a)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}const dg={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Ts={h:0,s:0,l:0},cl={h:0,s:0,l:0};function cu(i,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?i+(e-i)*6*t:t<1/2?e:t<2/3?i+(e-i)*6*(2/3-t):i}class lt{constructor(e,t,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,n)}set(e,t,n){if(t===void 0&&n===void 0){const s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,t,n);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=li){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,dt.colorSpaceToWorking(this,t),this}setRGB(e,t,n,s=dt.workingColorSpace){return this.r=e,this.g=t,this.b=n,dt.colorSpaceToWorking(this,s),this}setHSL(e,t,n,s=dt.workingColorSpace){if(e=Ed(e,1),t=nt(t,0,1),n=nt(n,0,1),t===0)this.r=this.g=this.b=n;else{const r=n<=.5?n*(1+t):n+t-n*t,o=2*n-r;this.r=cu(o,r,e+1/3),this.g=cu(o,r,e),this.b=cu(o,r,e-1/3)}return dt.colorSpaceToWorking(this,s),this}setStyle(e,t=li){function n(r){r!==void 0&&parseFloat(r)<1&&et("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r;const o=s[1],a=s[2];switch(o){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,t);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,t);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,t);break;default:et("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){const r=s[1],o=r.length;if(o===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,t);if(o===6)return this.setHex(parseInt(r,16),t);et("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=li){const n=dg[e.toLowerCase()];return n!==void 0?this.setHex(n,t):et("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=hs(e.r),this.g=hs(e.g),this.b=hs(e.b),this}copyLinearToSRGB(e){return this.r=po(e.r),this.g=po(e.g),this.b=po(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=li){return dt.workingToColorSpace(pn.copy(this),e),Math.round(nt(pn.r*255,0,255))*65536+Math.round(nt(pn.g*255,0,255))*256+Math.round(nt(pn.b*255,0,255))}getHexString(e=li){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=dt.workingColorSpace){dt.workingToColorSpace(pn.copy(this),t);const n=pn.r,s=pn.g,r=pn.b,o=Math.max(n,s,r),a=Math.min(n,s,r);let l,c;const u=(a+o)/2;if(a===o)l=0,c=0;else{const f=o-a;switch(c=u<=.5?f/(o+a):f/(2-o-a),o){case n:l=(s-r)/f+(s<r?6:0);break;case s:l=(r-n)/f+2;break;case r:l=(n-s)/f+4;break}l/=6}return e.h=l,e.s=c,e.l=u,e}getRGB(e,t=dt.workingColorSpace){return dt.workingToColorSpace(pn.copy(this),t),e.r=pn.r,e.g=pn.g,e.b=pn.b,e}getStyle(e=li){dt.workingToColorSpace(pn.copy(this),e);const t=pn.r,n=pn.g,s=pn.b;return e!==li?`color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(n*255)},${Math.round(s*255)})`}offsetHSL(e,t,n){return this.getHSL(Ts),this.setHSL(Ts.h+e,Ts.s+t,Ts.l+n)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,n){return this.r=e.r+(t.r-e.r)*n,this.g=e.g+(t.g-e.g)*n,this.b=e.b+(t.b-e.b)*n,this}lerpHSL(e,t){this.getHSL(Ts),e.getHSL(cl);const n=da(Ts.h,cl.h,t),s=da(Ts.s,cl.s,t),r=da(Ts.l,cl.l,t);return this.setHSL(n,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const t=this.r,n=this.g,s=this.b,r=e.elements;return this.r=r[0]*t+r[3]*n+r[6]*s,this.g=r[1]*t+r[4]*n+r[7]*s,this.b=r[2]*t+r[5]*n+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const pn=new lt;lt.NAMES=dg;let RS=0;class qa extends Rr{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:RS++}),this.uuid=Oo(),this.name="",this.type="Material",this.blending=ks,this.side=Vi,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=Ta,this.blendDst=Ea,this.blendEquation=gr,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new lt(0,0,0),this.blendAlpha=0,this.depthFunc=bo,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Zh,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Ur,this.stencilZFail=Ur,this.stencilZPass=Ur,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const t in e){const n=e[t];if(n===void 0){et(`Material: parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){et(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(n):s&&s.isVector3&&n&&n.isVector3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});const n={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(e).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(e).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(e).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(e).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(e).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==ks&&(n.blending=this.blending),this.side!==Vi&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==Ta&&(n.blendSrc=this.blendSrc),this.blendDst!==Ea&&(n.blendDst=this.blendDst),this.blendEquation!==gr&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==bo&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==Zh&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Ur&&(n.stencilFail=this.stencilFail),this.stencilZFail!==Ur&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==Ur&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function s(r){const o=[];for(const a in r){const l=r[a];delete l.metadata,o.push(l)}return o}if(t){const r=s(e.textures),o=s(e.images);r.length>0&&(n.textures=r),o.length>0&&(n.images=o)}return n}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const t=e.clippingPlanes;let n=null;if(t!==null){const s=t.length;n=new Array(s);for(let r=0;r!==s;++r)n[r]=t[r].clone()}return this.clippingPlanes=n,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}class Er extends qa{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new lt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Ti,this.combine=J0,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const us=IS();function IS(){const i=new ArrayBuffer(4),e=new Float32Array(i),t=new Uint32Array(i),n=new Uint32Array(512),s=new Uint32Array(512);for(let l=0;l<256;++l){const c=l-127;c<-27?(n[l]=0,n[l|256]=32768,s[l]=24,s[l|256]=24):c<-14?(n[l]=1024>>-c-14,n[l|256]=1024>>-c-14|32768,s[l]=-c-1,s[l|256]=-c-1):c<=15?(n[l]=c+15<<10,n[l|256]=c+15<<10|32768,s[l]=13,s[l|256]=13):c<128?(n[l]=31744,n[l|256]=64512,s[l]=24,s[l|256]=24):(n[l]=31744,n[l|256]=64512,s[l]=13,s[l|256]=13)}const r=new Uint32Array(2048),o=new Uint32Array(64),a=new Uint32Array(64);for(let l=1;l<1024;++l){let c=l<<13,u=0;for(;(c&8388608)===0;)c<<=1,u-=8388608;c&=-8388609,u+=947912704,r[l]=c|u}for(let l=1024;l<2048;++l)r[l]=939524096+(l-1024<<13);for(let l=1;l<31;++l)o[l]=l<<23;o[31]=1199570944,o[32]=2147483648;for(let l=33;l<63;++l)o[l]=2147483648+(l-32<<23);o[63]=3347054592;for(let l=1;l<64;++l)l!==32&&(a[l]=1024);return{floatView:e,uint32View:t,baseTable:n,shiftTable:s,mantissaTable:r,exponentTable:o,offsetTable:a}}function DS(i){Math.abs(i)>65504&&et("DataUtils.toHalfFloat(): Value out of range."),i=nt(i,-65504,65504),us.floatView[0]=i;const e=us.uint32View[0],t=e>>23&511;return us.baseTable[t]+((e&8388607)>>us.shiftTable[t])}function PS(i){const e=i>>10;return us.uint32View[0]=us.mantissaTable[us.offsetTable[e]+(i&1023)]+us.exponentTable[e],us.floatView[0]}class Fa{static toHalfFloat(e){return DS(e)}static fromHalfFloat(e){return PS(e)}}const Yt=new U,ul=new Xe;let FS=0;class xi{constructor(e,t,n=!1){if(Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:FS++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=n,this.usage=Jh,this.updateRanges=[],this.gpuType=bi,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,n){e*=this.itemSize,n*=t.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=t.array[n+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,n=this.count;t<n;t++)ul.fromBufferAttribute(this,t),ul.applyMatrix3(e),this.setXY(t,ul.x,ul.y);else if(this.itemSize===3)for(let t=0,n=this.count;t<n;t++)Yt.fromBufferAttribute(this,t),Yt.applyMatrix3(e),this.setXYZ(t,Yt.x,Yt.y,Yt.z);return this}applyMatrix4(e){for(let t=0,n=this.count;t<n;t++)Yt.fromBufferAttribute(this,t),Yt.applyMatrix4(e),this.setXYZ(t,Yt.x,Yt.y,Yt.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)Yt.fromBufferAttribute(this,t),Yt.applyNormalMatrix(e),this.setXYZ(t,Yt.x,Yt.y,Yt.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)Yt.fromBufferAttribute(this,t),Yt.transformDirection(e),this.setXYZ(t,Yt.x,Yt.y,Yt.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let n=this.array[e*this.itemSize+t];return this.normalized&&(n=to(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=Tn(n,this.array)),this.array[e*this.itemSize+t]=n,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=to(t,this.array)),t}setX(e,t){return this.normalized&&(t=Tn(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=to(t,this.array)),t}setY(e,t){return this.normalized&&(t=Tn(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=to(t,this.array)),t}setZ(e,t){return this.normalized&&(t=Tn(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=to(t,this.array)),t}setW(e,t){return this.normalized&&(t=Tn(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,n){return e*=this.itemSize,this.normalized&&(t=Tn(t,this.array),n=Tn(n,this.array)),this.array[e+0]=t,this.array[e+1]=n,this}setXYZ(e,t,n,s){return e*=this.itemSize,this.normalized&&(t=Tn(t,this.array),n=Tn(n,this.array),s=Tn(s,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this}setXYZW(e,t,n,s,r){return e*=this.itemSize,this.normalized&&(t=Tn(t,this.array),n=Tn(n,this.array),s=Tn(s,this.array),r=Tn(r,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==Jh&&(e.usage=this.usage),e}}class hg extends xi{constructor(e,t,n){super(new Uint16Array(e),t,n)}}class pg extends xi{constructor(e,t,n){super(new Uint32Array(e),t,n)}}class An extends xi{constructor(e,t,n){super(new Float32Array(e),t,n)}}let LS=0;const oi=new $e,uu=new jt,qr=new U,Yn=new Bi,$o=new Bi,nn=new U;class Bn extends Rr{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:LS++}),this.uuid=Oo(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(cg(e)?pg:hg)(e,1):this.index=e,this}setIndirect(e){return this.indirect=e,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,n=0){this.groups.push({start:e,count:t,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){const t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const r=new Ze().getNormalMatrix(e);n.applyNormalMatrix(r),n.needsUpdate=!0}const s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return oi.makeRotationFromQuaternion(e),this.applyMatrix4(oi),this}rotateX(e){return oi.makeRotationX(e),this.applyMatrix4(oi),this}rotateY(e){return oi.makeRotationY(e),this.applyMatrix4(oi),this}rotateZ(e){return oi.makeRotationZ(e),this.applyMatrix4(oi),this}translate(e,t,n){return oi.makeTranslation(e,t,n),this.applyMatrix4(oi),this}scale(e,t,n){return oi.makeScale(e,t,n),this.applyMatrix4(oi),this}lookAt(e){return uu.lookAt(e),uu.updateMatrix(),this.applyMatrix4(uu.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(qr).negate(),this.translate(qr.x,qr.y,qr.z),this}setFromPoints(e){const t=this.getAttribute("position");if(t===void 0){const n=[];for(let s=0,r=e.length;s<r;s++){const o=e[s];n.push(o.x,o.y,o.z||0)}this.setAttribute("position",new An(n,3))}else{const n=Math.min(e.length,t.count);for(let s=0;s<n;s++){const r=e[s];t.setXYZ(s,r.x,r.y,r.z||0)}e.length>t.count&&et("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new Bi);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Xt("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new U(-1/0,-1/0,-1/0),new U(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let n=0,s=t.length;n<s;n++){const r=t[n];Yn.setFromBufferAttribute(r),this.morphTargetsRelative?(nn.addVectors(this.boundingBox.min,Yn.min),this.boundingBox.expandByPoint(nn),nn.addVectors(this.boundingBox.max,Yn.max),this.boundingBox.expandByPoint(nn)):(this.boundingBox.expandByPoint(Yn.min),this.boundingBox.expandByPoint(Yn.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&Xt('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Cc);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Xt("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new U,1/0);return}if(e){const n=this.boundingSphere.center;if(Yn.setFromBufferAttribute(e),t)for(let r=0,o=t.length;r<o;r++){const a=t[r];$o.setFromBufferAttribute(a),this.morphTargetsRelative?(nn.addVectors(Yn.min,$o.min),Yn.expandByPoint(nn),nn.addVectors(Yn.max,$o.max),Yn.expandByPoint(nn)):(Yn.expandByPoint($o.min),Yn.expandByPoint($o.max))}Yn.getCenter(n);let s=0;for(let r=0,o=e.count;r<o;r++)nn.fromBufferAttribute(e,r),s=Math.max(s,n.distanceToSquared(nn));if(t)for(let r=0,o=t.length;r<o;r++){const a=t[r],l=this.morphTargetsRelative;for(let c=0,u=a.count;c<u;c++)nn.fromBufferAttribute(a,c),l&&(qr.fromBufferAttribute(e,c),nn.add(qr)),s=Math.max(s,n.distanceToSquared(nn))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&Xt('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){Xt("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=t.position,s=t.normal,r=t.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new xi(new Float32Array(4*n.count),4));const o=this.getAttribute("tangent"),a=[],l=[];for(let R=0;R<n.count;R++)a[R]=new U,l[R]=new U;const c=new U,u=new U,f=new U,d=new Xe,h=new Xe,x=new Xe,p=new U,g=new U;function m(R,y,C){c.fromBufferAttribute(n,R),u.fromBufferAttribute(n,y),f.fromBufferAttribute(n,C),d.fromBufferAttribute(r,R),h.fromBufferAttribute(r,y),x.fromBufferAttribute(r,C),u.sub(c),f.sub(c),h.sub(d),x.sub(d);const F=1/(h.x*x.y-x.x*h.y);isFinite(F)&&(p.copy(u).multiplyScalar(x.y).addScaledVector(f,-h.y).multiplyScalar(F),g.copy(f).multiplyScalar(h.x).addScaledVector(u,-x.x).multiplyScalar(F),a[R].add(p),a[y].add(p),a[C].add(p),l[R].add(g),l[y].add(g),l[C].add(g))}let _=this.groups;_.length===0&&(_=[{start:0,count:e.count}]);for(let R=0,y=_.length;R<y;++R){const C=_[R],F=C.start,I=C.count;for(let O=F,z=F+I;O<z;O+=3)m(e.getX(O+0),e.getX(O+1),e.getX(O+2))}const v=new U,A=new U,S=new U,b=new U;function M(R){S.fromBufferAttribute(s,R),b.copy(S);const y=a[R];v.copy(y),v.sub(S.multiplyScalar(S.dot(y))).normalize(),A.crossVectors(b,y);const F=A.dot(l[R])<0?-1:1;o.setXYZW(R,v.x,v.y,v.z,F)}for(let R=0,y=_.length;R<y;++R){const C=_[R],F=C.start,I=C.count;for(let O=F,z=F+I;O<z;O+=3)M(e.getX(O+0)),M(e.getX(O+1)),M(e.getX(O+2))}}computeVertexNormals(){const e=this.index,t=this.getAttribute("position");if(t!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new xi(new Float32Array(t.count*3),3),this.setAttribute("normal",n);else for(let d=0,h=n.count;d<h;d++)n.setXYZ(d,0,0,0);const s=new U,r=new U,o=new U,a=new U,l=new U,c=new U,u=new U,f=new U;if(e)for(let d=0,h=e.count;d<h;d+=3){const x=e.getX(d+0),p=e.getX(d+1),g=e.getX(d+2);s.fromBufferAttribute(t,x),r.fromBufferAttribute(t,p),o.fromBufferAttribute(t,g),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),a.fromBufferAttribute(n,x),l.fromBufferAttribute(n,p),c.fromBufferAttribute(n,g),a.add(u),l.add(u),c.add(u),n.setXYZ(x,a.x,a.y,a.z),n.setXYZ(p,l.x,l.y,l.z),n.setXYZ(g,c.x,c.y,c.z)}else for(let d=0,h=t.count;d<h;d+=3)s.fromBufferAttribute(t,d+0),r.fromBufferAttribute(t,d+1),o.fromBufferAttribute(t,d+2),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),n.setXYZ(d+0,u.x,u.y,u.z),n.setXYZ(d+1,u.x,u.y,u.z),n.setXYZ(d+2,u.x,u.y,u.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let t=0,n=e.count;t<n;t++)nn.fromBufferAttribute(e,t),nn.normalize(),e.setXYZ(t,nn.x,nn.y,nn.z)}toNonIndexed(){function e(a,l){const c=a.array,u=a.itemSize,f=a.normalized,d=new c.constructor(l.length*u);let h=0,x=0;for(let p=0,g=l.length;p<g;p++){a.isInterleavedBufferAttribute?h=l[p]*a.data.stride+a.offset:h=l[p]*u;for(let m=0;m<u;m++)d[x++]=c[h++]}return new xi(d,u,f)}if(this.index===null)return et("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const t=new Bn,n=this.index.array,s=this.attributes;for(const a in s){const l=s[a],c=e(l,n);t.setAttribute(a,c)}const r=this.morphAttributes;for(const a in r){const l=[],c=r[a];for(let u=0,f=c.length;u<f;u++){const d=c[u],h=e(d,n);l.push(h)}t.morphAttributes[a]=l}t.morphTargetsRelative=this.morphTargetsRelative;const o=this.groups;for(let a=0,l=o.length;a<l;a++){const c=o[a];t.addGroup(c.start,c.count,c.materialIndex)}return t}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const l=this.parameters;for(const c in l)l[c]!==void 0&&(e[c]=l[c]);return e}e.data={attributes:{}};const t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});const n=this.attributes;for(const l in n){const c=n[l];e.data.attributes[l]=c.toJSON(e.data)}const s={};let r=!1;for(const l in this.morphAttributes){const c=this.morphAttributes[l],u=[];for(let f=0,d=c.length;f<d;f++){const h=c[f];u.push(h.toJSON(e.data))}u.length>0&&(s[l]=u,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);const o=this.groups;o.length>0&&(e.data.groups=JSON.parse(JSON.stringify(o)));const a=this.boundingSphere;return a!==null&&(e.data.boundingSphere=a.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const t={};this.name=e.name;const n=e.index;n!==null&&this.setIndex(n.clone());const s=e.attributes;for(const c in s){const u=s[c];this.setAttribute(c,u.clone(t))}const r=e.morphAttributes;for(const c in r){const u=[],f=r[c];for(let d=0,h=f.length;d<h;d++)u.push(f[d].clone(t));this.morphAttributes[c]=u}this.morphTargetsRelative=e.morphTargetsRelative;const o=e.groups;for(let c=0,u=o.length;c<u;c++){const f=o[c];this.addGroup(f.start,f.count,f.materialIndex)}const a=e.boundingBox;a!==null&&(this.boundingBox=a.clone());const l=e.boundingSphere;return l!==null&&(this.boundingSphere=l.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const mp=new $e,sr=new Rd,fl=new Cc,gp=new U,dl=new U,hl=new U,pl=new U,fu=new U,ml=new U,xp=new U,gl=new U;class Qt extends jt{constructor(e=new Bn,t=new Er){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}getVertexPosition(e,t){const n=this.geometry,s=n.attributes.position,r=n.morphAttributes.position,o=n.morphTargetsRelative;t.fromBufferAttribute(s,e);const a=this.morphTargetInfluences;if(r&&a){ml.set(0,0,0);for(let l=0,c=r.length;l<c;l++){const u=a[l],f=r[l];u!==0&&(fu.fromBufferAttribute(f,e),o?ml.addScaledVector(fu,u):ml.addScaledVector(fu.sub(t),u))}t.add(ml)}return t}raycast(e,t){const n=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),fl.copy(n.boundingSphere),fl.applyMatrix4(r),sr.copy(e.ray).recast(e.near),!(fl.containsPoint(sr.origin)===!1&&(sr.intersectSphere(fl,gp)===null||sr.origin.distanceToSquared(gp)>(e.far-e.near)**2))&&(mp.copy(r).invert(),sr.copy(e.ray).applyMatrix4(mp),!(n.boundingBox!==null&&sr.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(e,t,sr)))}_computeIntersections(e,t,n){let s;const r=this.geometry,o=this.material,a=r.index,l=r.attributes.position,c=r.attributes.uv,u=r.attributes.uv1,f=r.attributes.normal,d=r.groups,h=r.drawRange;if(a!==null)if(Array.isArray(o))for(let x=0,p=d.length;x<p;x++){const g=d[x],m=o[g.materialIndex],_=Math.max(g.start,h.start),v=Math.min(a.count,Math.min(g.start+g.count,h.start+h.count));for(let A=_,S=v;A<S;A+=3){const b=a.getX(A),M=a.getX(A+1),R=a.getX(A+2);s=xl(this,m,e,n,c,u,f,b,M,R),s&&(s.faceIndex=Math.floor(A/3),s.face.materialIndex=g.materialIndex,t.push(s))}}else{const x=Math.max(0,h.start),p=Math.min(a.count,h.start+h.count);for(let g=x,m=p;g<m;g+=3){const _=a.getX(g),v=a.getX(g+1),A=a.getX(g+2);s=xl(this,o,e,n,c,u,f,_,v,A),s&&(s.faceIndex=Math.floor(g/3),t.push(s))}}else if(l!==void 0)if(Array.isArray(o))for(let x=0,p=d.length;x<p;x++){const g=d[x],m=o[g.materialIndex],_=Math.max(g.start,h.start),v=Math.min(l.count,Math.min(g.start+g.count,h.start+h.count));for(let A=_,S=v;A<S;A+=3){const b=A,M=A+1,R=A+2;s=xl(this,m,e,n,c,u,f,b,M,R),s&&(s.faceIndex=Math.floor(A/3),s.face.materialIndex=g.materialIndex,t.push(s))}}else{const x=Math.max(0,h.start),p=Math.min(l.count,h.start+h.count);for(let g=x,m=p;g<m;g+=3){const _=g,v=g+1,A=g+2;s=xl(this,o,e,n,c,u,f,_,v,A),s&&(s.faceIndex=Math.floor(g/3),t.push(s))}}}}function BS(i,e,t,n,s,r,o,a){let l;if(e.side===kn?l=n.intersectTriangle(o,r,s,!0,a):l=n.intersectTriangle(s,r,o,e.side===Vi,a),l===null)return null;gl.copy(a),gl.applyMatrix4(i.matrixWorld);const c=t.ray.origin.distanceTo(gl);return c<t.near||c>t.far?null:{distance:c,point:gl.clone(),object:i}}function xl(i,e,t,n,s,r,o,a,l,c){i.getVertexPosition(a,dl),i.getVertexPosition(l,hl),i.getVertexPosition(c,pl);const u=BS(i,e,t,n,dl,hl,pl,xp);if(u){const f=new U;yi.getBarycoord(xp,dl,hl,pl,f),s&&(u.uv=yi.getInterpolatedAttribute(s,a,l,c,f,new Xe)),r&&(u.uv1=yi.getInterpolatedAttribute(r,a,l,c,f,new Xe)),o&&(u.normal=yi.getInterpolatedAttribute(o,a,l,c,f,new U),u.normal.dot(n.direction)>0&&u.normal.multiplyScalar(-1));const d={a,b:l,c,normal:new U,materialIndex:0};yi.getNormal(dl,hl,pl,d.normal),u.face=d,u.barycoord=f}return u}class No extends Bn{constructor(e=1,t=1,n=1,s=1,r=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:n,widthSegments:s,heightSegments:r,depthSegments:o};const a=this;s=Math.floor(s),r=Math.floor(r),o=Math.floor(o);const l=[],c=[],u=[],f=[];let d=0,h=0;x("z","y","x",-1,-1,n,t,e,o,r,0),x("z","y","x",1,-1,n,t,-e,o,r,1),x("x","z","y",1,1,e,n,t,s,o,2),x("x","z","y",1,-1,e,n,-t,s,o,3),x("x","y","z",1,-1,e,t,n,s,r,4),x("x","y","z",-1,-1,e,t,-n,s,r,5),this.setIndex(l),this.setAttribute("position",new An(c,3)),this.setAttribute("normal",new An(u,3)),this.setAttribute("uv",new An(f,2));function x(p,g,m,_,v,A,S,b,M,R,y){const C=A/M,F=S/R,I=A/2,O=S/2,z=b/2,H=M+1,V=R+1;let q=0,G=0;const Y=new U;for(let ae=0;ae<V;ae++){const Ae=ae*F-O;for(let Pe=0;Pe<H;Pe++){const Ne=Pe*C-I;Y[p]=Ne*_,Y[g]=Ae*v,Y[m]=z,c.push(Y.x,Y.y,Y.z),Y[p]=0,Y[g]=0,Y[m]=b>0?1:-1,u.push(Y.x,Y.y,Y.z),f.push(Pe/M),f.push(1-ae/R),q+=1}}for(let ae=0;ae<R;ae++)for(let Ae=0;Ae<M;Ae++){const Pe=d+Ae+H*ae,Ne=d+Ae+H*(ae+1),We=d+(Ae+1)+H*(ae+1),He=d+(Ae+1)+H*ae;l.push(Pe,Ne,He),l.push(Ne,We,He),G+=6}a.addGroup(h,G,y),h+=G,d+=q}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new No(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}function wo(i){const e={};for(const t in i){e[t]={};for(const n in i[t]){const s=i[t][n];s&&(s.isColor||s.isMatrix3||s.isMatrix4||s.isVector2||s.isVector3||s.isVector4||s.isTexture||s.isQuaternion)?s.isRenderTargetTexture?(et("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][n]=null):e[t][n]=s.clone():Array.isArray(s)?e[t][n]=s.slice():e[t][n]=s}}return e}function En(i){const e={};for(let t=0;t<i.length;t++){const n=wo(i[t]);for(const s in n)e[s]=n[s]}return e}function US(i){const e=[];for(let t=0;t<i.length;t++)e.push(i[t].clone());return e}function mg(i){const e=i.getRenderTarget();return e===null?i.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:dt.workingColorSpace}const OS={clone:wo,merge:En};var NS=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,zS=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class Ln extends qa{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=NS,this.fragmentShader=zS,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=wo(e.uniforms),this.uniformsGroups=US(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this}toJSON(e){const t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(const s in this.uniforms){const o=this.uniforms[s].value;o&&o.isTexture?t.uniforms[s]={type:"t",value:o.toJSON(e).uuid}:o&&o.isColor?t.uniforms[s]={type:"c",value:o.getHex()}:o&&o.isVector2?t.uniforms[s]={type:"v2",value:o.toArray()}:o&&o.isVector3?t.uniforms[s]={type:"v3",value:o.toArray()}:o&&o.isVector4?t.uniforms[s]={type:"v4",value:o.toArray()}:o&&o.isMatrix3?t.uniforms[s]={type:"m3",value:o.toArray()}:o&&o.isMatrix4?t.uniforms[s]={type:"m4",value:o.toArray()}:t.uniforms[s]={value:o}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;const n={};for(const s in this.extensions)this.extensions[s]===!0&&(n[s]=!0);return Object.keys(n).length>0&&(t.extensions=n),t}}class gg extends jt{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new $e,this.projectionMatrix=new $e,this.projectionMatrixInverse=new $e,this.coordinateSystem=Li,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const Es=new U,_p=new Xe,vp=new Xe;class ci extends gg{constructor(e=50,t=1,n=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=n,this.far=s,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const t=.5*this.getFilmHeight()/e;this.fov=Pa*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(fa*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return Pa*2*Math.atan(Math.tan(fa*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,n){Es.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(Es.x,Es.y).multiplyScalar(-e/Es.z),Es.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(Es.x,Es.y).multiplyScalar(-e/Es.z)}getViewSize(e,t){return this.getViewBounds(e,_p,vp),t.subVectors(vp,_p)}setViewOffset(e,t,n,s,r,o){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let t=e*Math.tan(fa*.5*this.fov)/this.zoom,n=2*t,s=this.aspect*n,r=-.5*s;const o=this.view;if(this.view!==null&&this.view.enabled){const l=o.fullWidth,c=o.fullHeight;r+=o.offsetX*s/l,t-=o.offsetY*n/c,s*=o.width/l,n*=o.height/c}const a=this.filmOffset;a!==0&&(r+=e*a/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,t,t-n,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}}const Yr=-90,Qr=1;class kS extends jt{constructor(e,t,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const s=new ci(Yr,Qr,e,t);s.layers=this.layers,this.add(s);const r=new ci(Yr,Qr,e,t);r.layers=this.layers,this.add(r);const o=new ci(Yr,Qr,e,t);o.layers=this.layers,this.add(o);const a=new ci(Yr,Qr,e,t);a.layers=this.layers,this.add(a);const l=new ci(Yr,Qr,e,t);l.layers=this.layers,this.add(l);const c=new ci(Yr,Qr,e,t);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){const e=this.coordinateSystem,t=this.children.concat(),[n,s,r,o,a,l]=t;for(const c of t)this.remove(c);if(e===Li)n.up.set(0,1,0),n.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),o.up.set(0,0,1),o.lookAt(0,-1,0),a.up.set(0,1,0),a.lookAt(0,0,1),l.up.set(0,1,0),l.lookAt(0,0,-1);else if(e===ec)n.up.set(0,-1,0),n.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),o.up.set(0,0,-1),o.lookAt(0,-1,0),a.up.set(0,-1,0),a.lookAt(0,0,1),l.up.set(0,-1,0),l.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const c of t)this.add(c),c.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[r,o,a,l,c,u]=this.children,f=e.getRenderTarget(),d=e.getActiveCubeFace(),h=e.getActiveMipmapLevel(),x=e.xr.enabled;e.xr.enabled=!1;const p=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,e.setRenderTarget(n,0,s),e.render(t,r),e.setRenderTarget(n,1,s),e.render(t,o),e.setRenderTarget(n,2,s),e.render(t,a),e.setRenderTarget(n,3,s),e.render(t,l),e.setRenderTarget(n,4,s),e.render(t,c),n.texture.generateMipmaps=p,e.setRenderTarget(n,5,s),e.render(t,u),e.setRenderTarget(f,d,h),e.xr.enabled=x,n.texture.needsPMREMUpdate=!0}}class xg extends Fn{constructor(e=[],t=Mo,n,s,r,o,a,l,c,u){super(e,t,n,s,r,o,a,l,c,u),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class HS extends qs{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;const n={width:e,height:e,depth:1},s=[n,n,n,n,n,n];this.texture=new xg(s),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

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
			`},s=new No(5,5,5),r=new Ln({name:"CubemapFromEquirect",uniforms:wo(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:kn,blending:ds});r.uniforms.tEquirect.value=t;const o=new Qt(s,r),a=t.minFilter;return t.minFilter===_r&&(t.minFilter=di),new kS(1,10,this).update(e,o),t.minFilter=a,o.geometry.dispose(),o.material.dispose(),this}clear(e,t=!0,n=!0,s=!0){const r=e.getRenderTarget();for(let o=0;o<6;o++)e.setRenderTarget(this,o),e.clear(t,n,s);e.setRenderTarget(r)}}class _l extends jt{constructor(){super(),this.isGroup=!0,this.type="Group"}}const VS={type:"move"};class du{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new _l,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new _l,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new U,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new U),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new _l,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new U,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new U),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const t=this._hand;if(t)for(const n of e.hand.values())this._getHandJoint(t,n)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,n){let s=null,r=null,o=null;const a=this._targetRay,l=this._grip,c=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(c&&e.hand){o=!0;for(const p of e.hand.values()){const g=t.getJointPose(p,n),m=this._getHandJoint(c,p);g!==null&&(m.matrix.fromArray(g.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,m.jointRadius=g.radius),m.visible=g!==null}const u=c.joints["index-finger-tip"],f=c.joints["thumb-tip"],d=u.position.distanceTo(f.position),h=.02,x=.005;c.inputState.pinching&&d>h+x?(c.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!c.inputState.pinching&&d<=h-x&&(c.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else l!==null&&e.gripSpace&&(r=t.getPose(e.gripSpace,n),r!==null&&(l.matrix.fromArray(r.transform.matrix),l.matrix.decompose(l.position,l.rotation,l.scale),l.matrixWorldNeedsUpdate=!0,r.linearVelocity?(l.hasLinearVelocity=!0,l.linearVelocity.copy(r.linearVelocity)):l.hasLinearVelocity=!1,r.angularVelocity?(l.hasAngularVelocity=!0,l.angularVelocity.copy(r.angularVelocity)):l.hasAngularVelocity=!1));a!==null&&(s=t.getPose(e.targetRaySpace,n),s===null&&r!==null&&(s=r),s!==null&&(a.matrix.fromArray(s.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),a.matrixWorldNeedsUpdate=!0,s.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(s.linearVelocity)):a.hasLinearVelocity=!1,s.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(s.angularVelocity)):a.hasAngularVelocity=!1,this.dispatchEvent(VS)))}return a!==null&&(a.visible=s!==null),l!==null&&(l.visible=r!==null),c!==null&&(c.visible=o!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){const n=new _l;n.matrixAutoUpdate=!1,n.visible=!1,e.joints[t.jointName]=n,e.add(n)}return e.joints[t.jointName]}}class GS extends jt{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Ti,this.environmentIntensity=1,this.environmentRotation=new Ti,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}}class ns extends Fn{constructor(e=null,t=1,n=1,s,r,o,a,l,c=ti,u=ti,f,d){super(null,o,a,l,c,u,s,r,f,d),this.isDataTexture=!0,this.image={data:e,width:t,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class WS extends xi{constructor(e,t,n,s=1){super(e,t,n),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=s}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}toJSON(){const e=super.toJSON();return e.meshPerAttribute=this.meshPerAttribute,e.isInstancedBufferAttribute=!0,e}}const hu=new U,XS=new U,qS=new Ze;class Ps{constructor(e=new U(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,n,s){return this.normal.set(e,t,n),this.constant=s,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,n){const s=hu.subVectors(n,t).cross(XS.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t){const n=e.delta(hu),s=this.normal.dot(n);if(s===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;const r=-(e.start.dot(this.normal)+this.constant)/s;return r<0||r>1?null:t.copy(e.start).addScaledVector(n,r)}intersectsLine(e){const t=this.distanceToPoint(e.start),n=this.distanceToPoint(e.end);return t<0&&n>0||n<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){const n=t||qS.getNormalMatrix(e),s=this.coplanarPoint(hu).applyMatrix4(e),r=this.normal.applyMatrix3(n).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const rr=new Cc,YS=new Xe(.5,.5),vl=new U;class _g{constructor(e=new Ps,t=new Ps,n=new Ps,s=new Ps,r=new Ps,o=new Ps){this.planes=[e,t,n,s,r,o]}set(e,t,n,s,r,o){const a=this.planes;return a[0].copy(e),a[1].copy(t),a[2].copy(n),a[3].copy(s),a[4].copy(r),a[5].copy(o),this}copy(e){const t=this.planes;for(let n=0;n<6;n++)t[n].copy(e.planes[n]);return this}setFromProjectionMatrix(e,t=Li,n=!1){const s=this.planes,r=e.elements,o=r[0],a=r[1],l=r[2],c=r[3],u=r[4],f=r[5],d=r[6],h=r[7],x=r[8],p=r[9],g=r[10],m=r[11],_=r[12],v=r[13],A=r[14],S=r[15];if(s[0].setComponents(c-o,h-u,m-x,S-_).normalize(),s[1].setComponents(c+o,h+u,m+x,S+_).normalize(),s[2].setComponents(c+a,h+f,m+p,S+v).normalize(),s[3].setComponents(c-a,h-f,m-p,S-v).normalize(),n)s[4].setComponents(l,d,g,A).normalize(),s[5].setComponents(c-l,h-d,m-g,S-A).normalize();else if(s[4].setComponents(c-l,h-d,m-g,S-A).normalize(),t===Li)s[5].setComponents(c+l,h+d,m+g,S+A).normalize();else if(t===ec)s[5].setComponents(l,d,g,A).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),rr.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),rr.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(rr)}intersectsSprite(e){rr.center.set(0,0,0);const t=YS.distanceTo(e.center);return rr.radius=.7071067811865476+t,rr.applyMatrix4(e.matrixWorld),this.intersectsSphere(rr)}intersectsSphere(e){const t=this.planes,n=e.center,s=-e.radius;for(let r=0;r<6;r++)if(t[r].distanceToPoint(n)<s)return!1;return!0}intersectsBox(e){const t=this.planes;for(let n=0;n<6;n++){const s=t[n];if(vl.x=s.normal.x>0?e.max.x:e.min.x,vl.y=s.normal.y>0?e.max.y:e.min.y,vl.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(vl)<0)return!1}return!0}containsPoint(e){const t=this.planes;for(let n=0;n<6;n++)if(t[n].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class QS extends qa{constructor(e){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new lt(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.size=e.size,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}}const Ap=new $e,Nf=new Rd,Al=new Cc,Sl=new U;class KS extends jt{constructor(e=new Bn,t=new QS){super(),this.isPoints=!0,this.type="Points",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}raycast(e,t){const n=this.geometry,s=this.matrixWorld,r=e.params.Points.threshold,o=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),Al.copy(n.boundingSphere),Al.applyMatrix4(s),Al.radius+=r,e.ray.intersectsSphere(Al)===!1)return;Ap.copy(s).invert(),Nf.copy(e.ray).applyMatrix4(Ap);const a=r/((this.scale.x+this.scale.y+this.scale.z)/3),l=a*a,c=n.index,f=n.attributes.position;if(c!==null){const d=Math.max(0,o.start),h=Math.min(c.count,o.start+o.count);for(let x=d,p=h;x<p;x++){const g=c.getX(x);Sl.fromBufferAttribute(f,g),Sp(Sl,g,l,s,e,t,this)}}else{const d=Math.max(0,o.start),h=Math.min(f.count,o.start+o.count);for(let x=d,p=h;x<p;x++)Sl.fromBufferAttribute(f,x),Sp(Sl,x,l,s,e,t,this)}}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}}function Sp(i,e,t,n,s,r,o){const a=Nf.distanceSqToPoint(i);if(a<t){const l=new U;Nf.closestPointToPoint(i,l),l.applyMatrix4(n);const c=s.ray.origin.distanceTo(l);if(c<s.near||c>s.far)return;r.push({distance:c,distanceToRay:Math.sqrt(a),point:l,index:e,face:null,faceIndex:null,barycoord:null,object:o})}}class Id extends Fn{constructor(e,t,n=hi,s,r,o,a=ti,l=ti,c,u=To,f=1){if(u!==To&&u!==Ia)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const d={width:e,height:t,depth:f};super(d,s,r,o,a,l,u,n,c),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new wd(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}}class vg extends Fn{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class La extends Bn{constructor(e=1,t=1,n=1,s=32,r=1,o=!1,a=0,l=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:n,radialSegments:s,heightSegments:r,openEnded:o,thetaStart:a,thetaLength:l};const c=this;s=Math.floor(s),r=Math.floor(r);const u=[],f=[],d=[],h=[];let x=0;const p=[],g=n/2;let m=0;_(),o===!1&&(e>0&&v(!0),t>0&&v(!1)),this.setIndex(u),this.setAttribute("position",new An(f,3)),this.setAttribute("normal",new An(d,3)),this.setAttribute("uv",new An(h,2));function _(){const A=new U,S=new U;let b=0;const M=(t-e)/n;for(let R=0;R<=r;R++){const y=[],C=R/r,F=C*(t-e)+e;for(let I=0;I<=s;I++){const O=I/s,z=O*l+a,H=Math.sin(z),V=Math.cos(z);S.x=F*H,S.y=-C*n+g,S.z=F*V,f.push(S.x,S.y,S.z),A.set(H,M,V).normalize(),d.push(A.x,A.y,A.z),h.push(O,1-C),y.push(x++)}p.push(y)}for(let R=0;R<s;R++)for(let y=0;y<r;y++){const C=p[y][R],F=p[y+1][R],I=p[y+1][R+1],O=p[y][R+1];(e>0||y!==0)&&(u.push(C,F,O),b+=3),(t>0||y!==r-1)&&(u.push(F,I,O),b+=3)}c.addGroup(m,b,0),m+=b}function v(A){const S=x,b=new Xe,M=new U;let R=0;const y=A===!0?e:t,C=A===!0?1:-1;for(let I=1;I<=s;I++)f.push(0,g*C,0),d.push(0,C,0),h.push(.5,.5),x++;const F=x;for(let I=0;I<=s;I++){const z=I/s*l+a,H=Math.cos(z),V=Math.sin(z);M.x=y*V,M.y=g*C,M.z=y*H,f.push(M.x,M.y,M.z),d.push(0,C,0),b.x=H*.5+.5,b.y=V*.5*C+.5,h.push(b.x,b.y),x++}for(let I=0;I<s;I++){const O=S+I,z=F+I;A===!0?u.push(z,z+1,O):u.push(z+1,z,O),R+=3}c.addGroup(m,R,A===!0?1:2),m+=R}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new La(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class Dd extends La{constructor(e=1,t=1,n=32,s=1,r=!1,o=0,a=Math.PI*2){super(0,e,t,n,s,r,o,a),this.type="ConeGeometry",this.parameters={radius:e,height:t,radialSegments:n,heightSegments:s,openEnded:r,thetaStart:o,thetaLength:a}}static fromJSON(e){return new Dd(e.radius,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class jS{constructor(){this.type="Curve",this.arcLengthDivisions=200,this.needsUpdate=!1,this.cacheArcLengths=null}getPoint(){et("Curve: .getPoint() not implemented.")}getPointAt(e,t){const n=this.getUtoTmapping(e);return this.getPoint(n,t)}getPoints(e=5){const t=[];for(let n=0;n<=e;n++)t.push(this.getPoint(n/e));return t}getSpacedPoints(e=5){const t=[];for(let n=0;n<=e;n++)t.push(this.getPointAt(n/e));return t}getLength(){const e=this.getLengths();return e[e.length-1]}getLengths(e=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===e+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;const t=[];let n,s=this.getPoint(0),r=0;t.push(0);for(let o=1;o<=e;o++)n=this.getPoint(o/e),r+=n.distanceTo(s),t.push(r),s=n;return this.cacheArcLengths=t,t}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(e,t=null){const n=this.getLengths();let s=0;const r=n.length;let o;t?o=t:o=e*n[r-1];let a=0,l=r-1,c;for(;a<=l;)if(s=Math.floor(a+(l-a)/2),c=n[s]-o,c<0)a=s+1;else if(c>0)l=s-1;else{l=s;break}if(s=l,n[s]===o)return s/(r-1);const u=n[s],d=n[s+1]-u,h=(o-u)/d;return(s+h)/(r-1)}getTangent(e,t){let s=e-1e-4,r=e+1e-4;s<0&&(s=0),r>1&&(r=1);const o=this.getPoint(s),a=this.getPoint(r),l=t||(o.isVector2?new Xe:new U);return l.copy(a).sub(o).normalize(),l}getTangentAt(e,t){const n=this.getUtoTmapping(e);return this.getTangent(n,t)}computeFrenetFrames(e,t=!1){const n=new U,s=[],r=[],o=[],a=new U,l=new $e;for(let h=0;h<=e;h++){const x=h/e;s[h]=this.getTangentAt(x,new U)}r[0]=new U,o[0]=new U;let c=Number.MAX_VALUE;const u=Math.abs(s[0].x),f=Math.abs(s[0].y),d=Math.abs(s[0].z);u<=c&&(c=u,n.set(1,0,0)),f<=c&&(c=f,n.set(0,1,0)),d<=c&&n.set(0,0,1),a.crossVectors(s[0],n).normalize(),r[0].crossVectors(s[0],a),o[0].crossVectors(s[0],r[0]);for(let h=1;h<=e;h++){if(r[h]=r[h-1].clone(),o[h]=o[h-1].clone(),a.crossVectors(s[h-1],s[h]),a.length()>Number.EPSILON){a.normalize();const x=Math.acos(nt(s[h-1].dot(s[h]),-1,1));r[h].applyMatrix4(l.makeRotationAxis(a,x))}o[h].crossVectors(s[h],r[h])}if(t===!0){let h=Math.acos(nt(r[0].dot(r[e]),-1,1));h/=e,s[0].dot(a.crossVectors(r[0],r[e]))>0&&(h=-h);for(let x=1;x<=e;x++)r[x].applyMatrix4(l.makeRotationAxis(s[x],h*x)),o[x].crossVectors(s[x],r[x])}return{tangents:s,normals:r,binormals:o}}clone(){return new this.constructor().copy(this)}copy(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}toJSON(){const e={metadata:{version:4.7,type:"Curve",generator:"Curve.toJSON"}};return e.arcLengthDivisions=this.arcLengthDivisions,e.type=this.type,e}fromJSON(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}}function Pd(){let i=0,e=0,t=0,n=0;function s(r,o,a,l){i=r,e=a,t=-3*r+3*o-2*a-l,n=2*r-2*o+a+l}return{initCatmullRom:function(r,o,a,l,c){s(o,a,c*(a-r),c*(l-o))},initNonuniformCatmullRom:function(r,o,a,l,c,u,f){let d=(o-r)/c-(a-r)/(c+u)+(a-o)/u,h=(a-o)/u-(l-o)/(u+f)+(l-a)/f;d*=u,h*=u,s(o,a,d,h)},calc:function(r){const o=r*r,a=o*r;return i+e*r+t*o+n*a}}}const yl=new U,pu=new Pd,mu=new Pd,gu=new Pd;class $S extends jS{constructor(e=[],t=!1,n="centripetal",s=.5){super(),this.isCatmullRomCurve3=!0,this.type="CatmullRomCurve3",this.points=e,this.closed=t,this.curveType=n,this.tension=s}getPoint(e,t=new U){const n=t,s=this.points,r=s.length,o=(r-(this.closed?0:1))*e;let a=Math.floor(o),l=o-a;this.closed?a+=a>0?0:(Math.floor(Math.abs(a)/r)+1)*r:l===0&&a===r-1&&(a=r-2,l=1);let c,u;this.closed||a>0?c=s[(a-1)%r]:(yl.subVectors(s[0],s[1]).add(s[0]),c=yl);const f=s[a%r],d=s[(a+1)%r];if(this.closed||a+2<r?u=s[(a+2)%r]:(yl.subVectors(s[r-1],s[r-2]).add(s[r-1]),u=yl),this.curveType==="centripetal"||this.curveType==="chordal"){const h=this.curveType==="chordal"?.5:.25;let x=Math.pow(c.distanceToSquared(f),h),p=Math.pow(f.distanceToSquared(d),h),g=Math.pow(d.distanceToSquared(u),h);p<1e-4&&(p=1),x<1e-4&&(x=p),g<1e-4&&(g=p),pu.initNonuniformCatmullRom(c.x,f.x,d.x,u.x,x,p,g),mu.initNonuniformCatmullRom(c.y,f.y,d.y,u.y,x,p,g),gu.initNonuniformCatmullRom(c.z,f.z,d.z,u.z,x,p,g)}else this.curveType==="catmullrom"&&(pu.initCatmullRom(c.x,f.x,d.x,u.x,this.tension),mu.initCatmullRom(c.y,f.y,d.y,u.y,this.tension),gu.initCatmullRom(c.z,f.z,d.z,u.z,this.tension));return n.set(pu.calc(l),mu.calc(l),gu.calc(l)),n}copy(e){super.copy(e),this.points=[];for(let t=0,n=e.points.length;t<n;t++){const s=e.points[t];this.points.push(s.clone())}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}toJSON(){const e=super.toJSON();e.points=[];for(let t=0,n=this.points.length;t<n;t++){const s=this.points[t];e.points.push(s.toArray())}return e.closed=this.closed,e.curveType=this.curveType,e.tension=this.tension,e}fromJSON(e){super.fromJSON(e),this.points=[];for(let t=0,n=e.points.length;t<n;t++){const s=e.points[t];this.points.push(new U().fromArray(s))}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}}class Ro extends Bn{constructor(e=1,t=1,n=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:n,heightSegments:s};const r=e/2,o=t/2,a=Math.floor(n),l=Math.floor(s),c=a+1,u=l+1,f=e/a,d=t/l,h=[],x=[],p=[],g=[];for(let m=0;m<u;m++){const _=m*d-o;for(let v=0;v<c;v++){const A=v*f-r;x.push(A,-_,0),p.push(0,0,1),g.push(v/a),g.push(1-m/l)}}for(let m=0;m<l;m++)for(let _=0;_<a;_++){const v=_+c*m,A=_+c*(m+1),S=_+1+c*(m+1),b=_+1+c*m;h.push(v,A,b),h.push(A,S,b)}this.setIndex(h),this.setAttribute("position",new An(x,3)),this.setAttribute("normal",new An(p,3)),this.setAttribute("uv",new An(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Ro(e.width,e.height,e.widthSegments,e.heightSegments)}}class nc extends Bn{constructor(e=1,t=32,n=16,s=0,r=Math.PI*2,o=0,a=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:e,widthSegments:t,heightSegments:n,phiStart:s,phiLength:r,thetaStart:o,thetaLength:a},t=Math.max(3,Math.floor(t)),n=Math.max(2,Math.floor(n));const l=Math.min(o+a,Math.PI);let c=0;const u=[],f=new U,d=new U,h=[],x=[],p=[],g=[];for(let m=0;m<=n;m++){const _=[],v=m/n;let A=0;m===0&&o===0?A=.5/t:m===n&&l===Math.PI&&(A=-.5/t);for(let S=0;S<=t;S++){const b=S/t;f.x=-e*Math.cos(s+b*r)*Math.sin(o+v*a),f.y=e*Math.cos(o+v*a),f.z=e*Math.sin(s+b*r)*Math.sin(o+v*a),x.push(f.x,f.y,f.z),d.copy(f).normalize(),p.push(d.x,d.y,d.z),g.push(b+A,1-v),_.push(c++)}u.push(_)}for(let m=0;m<n;m++)for(let _=0;_<t;_++){const v=u[m][_+1],A=u[m][_],S=u[m+1][_],b=u[m+1][_+1];(m!==0||o>0)&&h.push(v,A,b),(m!==n-1||l<Math.PI)&&h.push(A,S,b)}this.setIndex(h),this.setAttribute("position",new An(x,3)),this.setAttribute("normal",new An(p,3)),this.setAttribute("uv",new An(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new nc(e.radius,e.widthSegments,e.heightSegments,e.phiStart,e.phiLength,e.thetaStart,e.thetaLength)}}class ZS extends qa{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=kA,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class JS extends qa{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}class Fd extends gg{constructor(e=-1,t=1,n=1,s=-1,r=.1,o=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=n,this.bottom=s,this.near=r,this.far=o,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,n,s,r,o){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,s=(this.top+this.bottom)/2;let r=n-e,o=n+e,a=s+t,l=s-t;if(this.view!==null&&this.view.enabled){const c=(this.right-this.left)/this.view.fullWidth/this.zoom,u=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=c*this.view.offsetX,o=r+c*this.view.width,a-=u*this.view.offsetY,l=a-u*this.view.height}this.projectionMatrix.makeOrthographic(r,o,a,l,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}}class ey extends Bn{constructor(){super(),this.isInstancedBufferGeometry=!0,this.type="InstancedBufferGeometry",this.instanceCount=1/0}copy(e){return super.copy(e),this.instanceCount=e.instanceCount,this}toJSON(){const e=super.toJSON();return e.instanceCount=this.instanceCount,e.isInstancedBufferGeometry=!0,e}}class ty extends ci{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}class yp{constructor(e=1,t=0,n=0){this.radius=e,this.phi=t,this.theta=n}set(e,t,n){return this.radius=e,this.phi=t,this.theta=n,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=nt(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,n){return this.radius=Math.sqrt(e*e+t*t+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,n),this.phi=Math.acos(nt(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}function bp(i,e,t,n){const s=ny(n);switch(t){case rg:return i*e;case ag:return i*e/s.components*s.byteLength;case Mc:return i*e/s.components*s.byteLength;case Cd:return i*e*2/s.components*s.byteLength;case Td:return i*e*2/s.components*s.byteLength;case og:return i*e*3/s.components*s.byteLength;case Pn:return i*e*4/s.components*s.byteLength;case ho:return i*e*4/s.components*s.byteLength;case Nl:case zl:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case kl:case Hl:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case ff:case hf:return Math.max(i,16)*Math.max(e,8)/4;case uf:case df:return Math.max(i,8)*Math.max(e,8)/2;case pf:case mf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case gf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case xf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case _f:return Math.floor((i+4)/5)*Math.floor((e+3)/4)*16;case vf:return Math.floor((i+4)/5)*Math.floor((e+4)/5)*16;case Af:return Math.floor((i+5)/6)*Math.floor((e+4)/5)*16;case Sf:return Math.floor((i+5)/6)*Math.floor((e+5)/6)*16;case yf:return Math.floor((i+7)/8)*Math.floor((e+4)/5)*16;case bf:return Math.floor((i+7)/8)*Math.floor((e+5)/6)*16;case Mf:return Math.floor((i+7)/8)*Math.floor((e+7)/8)*16;case Cf:return Math.floor((i+9)/10)*Math.floor((e+4)/5)*16;case Tf:return Math.floor((i+9)/10)*Math.floor((e+5)/6)*16;case Ef:return Math.floor((i+9)/10)*Math.floor((e+7)/8)*16;case wf:return Math.floor((i+9)/10)*Math.floor((e+9)/10)*16;case Rf:return Math.floor((i+11)/12)*Math.floor((e+9)/10)*16;case If:return Math.floor((i+11)/12)*Math.floor((e+11)/12)*16;case Df:case Pf:case Ff:return Math.ceil(i/4)*Math.ceil(e/4)*16;case Lf:case Bf:return Math.ceil(i/4)*Math.ceil(e/4)*8;case Uf:case Of:return Math.ceil(i/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function ny(i){switch(i){case Gi:case tg:return{byteLength:1,components:1};case wa:case ng:case wr:return{byteLength:2,components:1};case bd:case Md:return{byteLength:2,components:4};case hi:case yd:case bi:return{byteLength:4,components:1};case ig:case sg:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${i}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:Sd}}));typeof window<"u"&&(window.__THREE__?et("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=Sd);function Ag(){let i=null,e=!1,t=null,n=null;function s(r,o){t(r,o),n=i.requestAnimationFrame(s)}return{start:function(){e!==!0&&t!==null&&(n=i.requestAnimationFrame(s),e=!0)},stop:function(){i.cancelAnimationFrame(n),e=!1},setAnimationLoop:function(r){t=r},setContext:function(r){i=r}}}function iy(i){const e=new WeakMap;function t(a,l){const c=a.array,u=a.usage,f=c.byteLength,d=i.createBuffer();i.bindBuffer(l,d),i.bufferData(l,c,u),a.onUploadCallback();let h;if(c instanceof Float32Array)h=i.FLOAT;else if(typeof Float16Array<"u"&&c instanceof Float16Array)h=i.HALF_FLOAT;else if(c instanceof Uint16Array)a.isFloat16BufferAttribute?h=i.HALF_FLOAT:h=i.UNSIGNED_SHORT;else if(c instanceof Int16Array)h=i.SHORT;else if(c instanceof Uint32Array)h=i.UNSIGNED_INT;else if(c instanceof Int32Array)h=i.INT;else if(c instanceof Int8Array)h=i.BYTE;else if(c instanceof Uint8Array)h=i.UNSIGNED_BYTE;else if(c instanceof Uint8ClampedArray)h=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+c);return{buffer:d,type:h,bytesPerElement:c.BYTES_PER_ELEMENT,version:a.version,size:f}}function n(a,l,c){const u=l.array,f=l.updateRanges;if(i.bindBuffer(c,a),f.length===0)i.bufferSubData(c,0,u);else{f.sort((h,x)=>h.start-x.start);let d=0;for(let h=1;h<f.length;h++){const x=f[d],p=f[h];p.start<=x.start+x.count+1?x.count=Math.max(x.count,p.start+p.count-x.start):(++d,f[d]=p)}f.length=d+1;for(let h=0,x=f.length;h<x;h++){const p=f[h];i.bufferSubData(c,p.start*u.BYTES_PER_ELEMENT,u,p.start,p.count)}l.clearUpdateRanges()}l.onUploadCallback()}function s(a){return a.isInterleavedBufferAttribute&&(a=a.data),e.get(a)}function r(a){a.isInterleavedBufferAttribute&&(a=a.data);const l=e.get(a);l&&(i.deleteBuffer(l.buffer),e.delete(a))}function o(a,l){if(a.isInterleavedBufferAttribute&&(a=a.data),a.isGLBufferAttribute){const u=e.get(a);(!u||u.version<a.version)&&e.set(a,{buffer:a.buffer,type:a.type,bytesPerElement:a.elementSize,version:a.version});return}const c=e.get(a);if(c===void 0)e.set(a,t(a,l));else if(c.version<a.version){if(c.size!==a.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(c.buffer,a,l),c.version=a.version}}return{get:s,remove:r,update:o}}var sy=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,ry=`#ifdef USE_ALPHAHASH
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
#endif`,oy=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,ay=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,ly=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,cy=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,uy=`#ifdef USE_AOMAP
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
#endif`,fy=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,dy=`#ifdef USE_BATCHING
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
#endif`,hy=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,py=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,my=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,gy=`float G_BlinnPhong_Implicit( ) {
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
} // validated`,xy=`#ifdef USE_IRIDESCENCE
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
#endif`,_y=`#ifdef USE_BUMPMAP
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
#endif`,vy=`#if NUM_CLIPPING_PLANES > 0
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
#endif`,Ay=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,Sy=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,yy=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,by=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,My=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,Cy=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,Ty=`#if defined( USE_COLOR_ALPHA )
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
#endif`,Ey=`#define PI 3.141592653589793
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
} // validated`,wy=`#ifdef ENVMAP_TYPE_CUBE_UV
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
#endif`,Ry=`vec3 transformedNormal = objectNormal;
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
#endif`,Iy=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,Dy=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,Py=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,Fy=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,Ly="gl_FragColor = linearToOutputTexel( gl_FragColor );",By=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,Uy=`#ifdef USE_ENVMAP
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
#endif`,Oy=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,Ny=`#ifdef USE_ENVMAP
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
#endif`,zy=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,ky=`#ifdef USE_ENVMAP
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
#endif`,Hy=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,Vy=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,Gy=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,Wy=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,Xy=`#ifdef USE_GRADIENTMAP
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
}`,qy=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,Yy=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,Qy=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,Ky=`uniform bool receiveShadow;
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
#endif`,jy=`#ifdef USE_ENVMAP
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
#endif`,$y=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,Zy=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,Jy=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,eb=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,tb=`PhysicalMaterial material;
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
#endif`,nb=`uniform sampler2D dfgLUT;
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
}`,ib=`
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
#endif`,sb=`#if defined( RE_IndirectDiffuse )
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
#endif`,rb=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,ob=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,ab=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,lb=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,cb=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,ub=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,fb=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,db=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
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
#endif`,hb=`#if defined( USE_POINTS_UV )
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
#endif`,pb=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,mb=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,gb=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,xb=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,_b=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,vb=`#ifdef USE_MORPHTARGETS
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
#endif`,Ab=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Sb=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
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
vec3 nonPerturbedNormal = normal;`,yb=`#ifdef USE_NORMALMAP_OBJECTSPACE
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
#endif`,bb=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Mb=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Cb=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,Tb=`#ifdef USE_NORMALMAP
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
#endif`,Eb=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,wb=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,Rb=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,Ib=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,Db=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,Pb=`vec3 packNormalToRGB( const in vec3 normal ) {
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
}`,Fb=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,Lb=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,Bb=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,Ub=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,Ob=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,Nb=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,zb=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,kb=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,Hb=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
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
#endif`,Vb=`float getShadowMask() {
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
}`,Gb=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,Wb=`#ifdef USE_SKINNING
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
#endif`,Xb=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,qb=`#ifdef USE_SKINNING
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
#endif`,Yb=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,Qb=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,Kb=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,jb=`#ifndef saturate
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
vec3 CustomToneMapping( vec3 color ) { return color; }`,$b=`#ifdef USE_TRANSMISSION
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
#endif`,Zb=`#ifdef USE_TRANSMISSION
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
#endif`,Jb=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,eM=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,tM=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,nM=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const iM=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,sM=`uniform sampler2D t2D;
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
}`,rM=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,oM=`#ifdef ENVMAP_TYPE_CUBE
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
}`,aM=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,lM=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,cM=`#include <common>
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
}`,uM=`#if DEPTH_PACKING == 3200
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
}`,fM=`#define DISTANCE
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
}`,dM=`#define DISTANCE
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
}`,hM=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,pM=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,mM=`uniform float scale;
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
}`,gM=`uniform vec3 diffuse;
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
}`,xM=`#include <common>
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
}`,_M=`uniform vec3 diffuse;
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
}`,vM=`#define LAMBERT
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
}`,AM=`#define LAMBERT
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
}`,SM=`#define MATCAP
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
}`,yM=`#define MATCAP
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
}`,bM=`#define NORMAL
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
}`,MM=`#define NORMAL
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
}`,CM=`#define PHONG
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
}`,TM=`#define PHONG
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
}`,EM=`#define STANDARD
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
}`,wM=`#define STANDARD
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
}`,RM=`#define TOON
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
}`,IM=`#define TOON
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
}`,DM=`uniform float size;
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
}`,PM=`uniform vec3 diffuse;
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
}`,FM=`#include <common>
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
}`,LM=`uniform vec3 color;
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
}`,BM=`uniform float rotation;
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
}`,UM=`uniform vec3 diffuse;
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
}`,st={alphahash_fragment:sy,alphahash_pars_fragment:ry,alphamap_fragment:oy,alphamap_pars_fragment:ay,alphatest_fragment:ly,alphatest_pars_fragment:cy,aomap_fragment:uy,aomap_pars_fragment:fy,batching_pars_vertex:dy,batching_vertex:hy,begin_vertex:py,beginnormal_vertex:my,bsdfs:gy,iridescence_fragment:xy,bumpmap_pars_fragment:_y,clipping_planes_fragment:vy,clipping_planes_pars_fragment:Ay,clipping_planes_pars_vertex:Sy,clipping_planes_vertex:yy,color_fragment:by,color_pars_fragment:My,color_pars_vertex:Cy,color_vertex:Ty,common:Ey,cube_uv_reflection_fragment:wy,defaultnormal_vertex:Ry,displacementmap_pars_vertex:Iy,displacementmap_vertex:Dy,emissivemap_fragment:Py,emissivemap_pars_fragment:Fy,colorspace_fragment:Ly,colorspace_pars_fragment:By,envmap_fragment:Uy,envmap_common_pars_fragment:Oy,envmap_pars_fragment:Ny,envmap_pars_vertex:zy,envmap_physical_pars_fragment:jy,envmap_vertex:ky,fog_vertex:Hy,fog_pars_vertex:Vy,fog_fragment:Gy,fog_pars_fragment:Wy,gradientmap_pars_fragment:Xy,lightmap_pars_fragment:qy,lights_lambert_fragment:Yy,lights_lambert_pars_fragment:Qy,lights_pars_begin:Ky,lights_toon_fragment:$y,lights_toon_pars_fragment:Zy,lights_phong_fragment:Jy,lights_phong_pars_fragment:eb,lights_physical_fragment:tb,lights_physical_pars_fragment:nb,lights_fragment_begin:ib,lights_fragment_maps:sb,lights_fragment_end:rb,logdepthbuf_fragment:ob,logdepthbuf_pars_fragment:ab,logdepthbuf_pars_vertex:lb,logdepthbuf_vertex:cb,map_fragment:ub,map_pars_fragment:fb,map_particle_fragment:db,map_particle_pars_fragment:hb,metalnessmap_fragment:pb,metalnessmap_pars_fragment:mb,morphinstance_vertex:gb,morphcolor_vertex:xb,morphnormal_vertex:_b,morphtarget_pars_vertex:vb,morphtarget_vertex:Ab,normal_fragment_begin:Sb,normal_fragment_maps:yb,normal_pars_fragment:bb,normal_pars_vertex:Mb,normal_vertex:Cb,normalmap_pars_fragment:Tb,clearcoat_normal_fragment_begin:Eb,clearcoat_normal_fragment_maps:wb,clearcoat_pars_fragment:Rb,iridescence_pars_fragment:Ib,opaque_fragment:Db,packing:Pb,premultiplied_alpha_fragment:Fb,project_vertex:Lb,dithering_fragment:Bb,dithering_pars_fragment:Ub,roughnessmap_fragment:Ob,roughnessmap_pars_fragment:Nb,shadowmap_pars_fragment:zb,shadowmap_pars_vertex:kb,shadowmap_vertex:Hb,shadowmask_pars_fragment:Vb,skinbase_vertex:Gb,skinning_pars_vertex:Wb,skinning_vertex:Xb,skinnormal_vertex:qb,specularmap_fragment:Yb,specularmap_pars_fragment:Qb,tonemapping_fragment:Kb,tonemapping_pars_fragment:jb,transmission_fragment:$b,transmission_pars_fragment:Zb,uv_pars_fragment:Jb,uv_pars_vertex:eM,uv_vertex:tM,worldpos_vertex:nM,background_vert:iM,background_frag:sM,backgroundCube_vert:rM,backgroundCube_frag:oM,cube_vert:aM,cube_frag:lM,depth_vert:cM,depth_frag:uM,distanceRGBA_vert:fM,distanceRGBA_frag:dM,equirect_vert:hM,equirect_frag:pM,linedashed_vert:mM,linedashed_frag:gM,meshbasic_vert:xM,meshbasic_frag:_M,meshlambert_vert:vM,meshlambert_frag:AM,meshmatcap_vert:SM,meshmatcap_frag:yM,meshnormal_vert:bM,meshnormal_frag:MM,meshphong_vert:CM,meshphong_frag:TM,meshphysical_vert:EM,meshphysical_frag:wM,meshtoon_vert:RM,meshtoon_frag:IM,points_vert:DM,points_frag:PM,shadow_vert:FM,shadow_frag:LM,sprite_vert:BM,sprite_frag:UM},Ue={common:{diffuse:{value:new lt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Ze},alphaMap:{value:null},alphaMapTransform:{value:new Ze},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Ze}},envmap:{envMap:{value:null},envMapRotation:{value:new Ze},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Ze}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Ze}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Ze},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Ze},normalScale:{value:new Xe(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Ze},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Ze}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Ze}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Ze}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new lt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new lt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Ze},alphaTest:{value:0},uvTransform:{value:new Ze}},sprite:{diffuse:{value:new lt(16777215)},opacity:{value:1},center:{value:new Xe(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Ze},alphaMap:{value:null},alphaMapTransform:{value:new Ze},alphaTest:{value:0}}},Pi={basic:{uniforms:En([Ue.common,Ue.specularmap,Ue.envmap,Ue.aomap,Ue.lightmap,Ue.fog]),vertexShader:st.meshbasic_vert,fragmentShader:st.meshbasic_frag},lambert:{uniforms:En([Ue.common,Ue.specularmap,Ue.envmap,Ue.aomap,Ue.lightmap,Ue.emissivemap,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,Ue.fog,Ue.lights,{emissive:{value:new lt(0)}}]),vertexShader:st.meshlambert_vert,fragmentShader:st.meshlambert_frag},phong:{uniforms:En([Ue.common,Ue.specularmap,Ue.envmap,Ue.aomap,Ue.lightmap,Ue.emissivemap,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,Ue.fog,Ue.lights,{emissive:{value:new lt(0)},specular:{value:new lt(1118481)},shininess:{value:30}}]),vertexShader:st.meshphong_vert,fragmentShader:st.meshphong_frag},standard:{uniforms:En([Ue.common,Ue.envmap,Ue.aomap,Ue.lightmap,Ue.emissivemap,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,Ue.roughnessmap,Ue.metalnessmap,Ue.fog,Ue.lights,{emissive:{value:new lt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:st.meshphysical_vert,fragmentShader:st.meshphysical_frag},toon:{uniforms:En([Ue.common,Ue.aomap,Ue.lightmap,Ue.emissivemap,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,Ue.gradientmap,Ue.fog,Ue.lights,{emissive:{value:new lt(0)}}]),vertexShader:st.meshtoon_vert,fragmentShader:st.meshtoon_frag},matcap:{uniforms:En([Ue.common,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,Ue.fog,{matcap:{value:null}}]),vertexShader:st.meshmatcap_vert,fragmentShader:st.meshmatcap_frag},points:{uniforms:En([Ue.points,Ue.fog]),vertexShader:st.points_vert,fragmentShader:st.points_frag},dashed:{uniforms:En([Ue.common,Ue.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:st.linedashed_vert,fragmentShader:st.linedashed_frag},depth:{uniforms:En([Ue.common,Ue.displacementmap]),vertexShader:st.depth_vert,fragmentShader:st.depth_frag},normal:{uniforms:En([Ue.common,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,{opacity:{value:1}}]),vertexShader:st.meshnormal_vert,fragmentShader:st.meshnormal_frag},sprite:{uniforms:En([Ue.sprite,Ue.fog]),vertexShader:st.sprite_vert,fragmentShader:st.sprite_frag},background:{uniforms:{uvTransform:{value:new Ze},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:st.background_vert,fragmentShader:st.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Ze}},vertexShader:st.backgroundCube_vert,fragmentShader:st.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:st.cube_vert,fragmentShader:st.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:st.equirect_vert,fragmentShader:st.equirect_frag},distanceRGBA:{uniforms:En([Ue.common,Ue.displacementmap,{referencePosition:{value:new U},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:st.distanceRGBA_vert,fragmentShader:st.distanceRGBA_frag},shadow:{uniforms:En([Ue.lights,Ue.fog,{color:{value:new lt(0)},opacity:{value:1}}]),vertexShader:st.shadow_vert,fragmentShader:st.shadow_frag}};Pi.physical={uniforms:En([Pi.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Ze},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Ze},clearcoatNormalScale:{value:new Xe(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Ze},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Ze},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Ze},sheen:{value:0},sheenColor:{value:new lt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Ze},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Ze},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Ze},transmissionSamplerSize:{value:new Xe},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Ze},attenuationDistance:{value:0},attenuationColor:{value:new lt(0)},specularColor:{value:new lt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Ze},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Ze},anisotropyVector:{value:new Xe},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Ze}}]),vertexShader:st.meshphysical_vert,fragmentShader:st.meshphysical_frag};const bl={r:0,b:0,g:0},or=new Ti,OM=new $e;function NM(i,e,t,n,s,r,o){const a=new lt(0);let l=r===!0?0:1,c,u,f=null,d=0,h=null;function x(v){let A=v.isScene===!0?v.background:null;return A&&A.isTexture&&(A=(v.backgroundBlurriness>0?t:e).get(A)),A}function p(v){let A=!1;const S=x(v);S===null?m(a,l):S&&S.isColor&&(m(S,1),A=!0);const b=i.xr.getEnvironmentBlendMode();b==="additive"?n.buffers.color.setClear(0,0,0,1,o):b==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,o),(i.autoClear||A)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function g(v,A){const S=x(A);S&&(S.isCubeTexture||S.mapping===bc)?(u===void 0&&(u=new Qt(new No(1,1,1),new Ln({name:"BackgroundCubeMaterial",uniforms:wo(Pi.backgroundCube.uniforms),vertexShader:Pi.backgroundCube.vertexShader,fragmentShader:Pi.backgroundCube.fragmentShader,side:kn,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),u.geometry.deleteAttribute("normal"),u.geometry.deleteAttribute("uv"),u.onBeforeRender=function(b,M,R){this.matrixWorld.copyPosition(R.matrixWorld)},Object.defineProperty(u.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),s.update(u)),or.copy(A.backgroundRotation),or.x*=-1,or.y*=-1,or.z*=-1,S.isCubeTexture&&S.isRenderTargetTexture===!1&&(or.y*=-1,or.z*=-1),u.material.uniforms.envMap.value=S,u.material.uniforms.flipEnvMap.value=S.isCubeTexture&&S.isRenderTargetTexture===!1?-1:1,u.material.uniforms.backgroundBlurriness.value=A.backgroundBlurriness,u.material.uniforms.backgroundIntensity.value=A.backgroundIntensity,u.material.uniforms.backgroundRotation.value.setFromMatrix4(OM.makeRotationFromEuler(or)),u.material.toneMapped=dt.getTransfer(S.colorSpace)!==yt,(f!==S||d!==S.version||h!==i.toneMapping)&&(u.material.needsUpdate=!0,f=S,d=S.version,h=i.toneMapping),u.layers.enableAll(),v.unshift(u,u.geometry,u.material,0,0,null)):S&&S.isTexture&&(c===void 0&&(c=new Qt(new Ro(2,2),new Ln({name:"BackgroundMaterial",uniforms:wo(Pi.background.uniforms),vertexShader:Pi.background.vertexShader,fragmentShader:Pi.background.fragmentShader,side:Vi,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),s.update(c)),c.material.uniforms.t2D.value=S,c.material.uniforms.backgroundIntensity.value=A.backgroundIntensity,c.material.toneMapped=dt.getTransfer(S.colorSpace)!==yt,S.matrixAutoUpdate===!0&&S.updateMatrix(),c.material.uniforms.uvTransform.value.copy(S.matrix),(f!==S||d!==S.version||h!==i.toneMapping)&&(c.material.needsUpdate=!0,f=S,d=S.version,h=i.toneMapping),c.layers.enableAll(),v.unshift(c,c.geometry,c.material,0,0,null))}function m(v,A){v.getRGB(bl,mg(i)),n.buffers.color.setClear(bl.r,bl.g,bl.b,A,o)}function _(){u!==void 0&&(u.geometry.dispose(),u.material.dispose(),u=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return a},setClearColor:function(v,A=1){a.set(v),l=A,m(a,l)},getClearAlpha:function(){return l},setClearAlpha:function(v){l=v,m(a,l)},render:p,addToRenderList:g,dispose:_}}function zM(i,e){const t=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},s=d(null);let r=s,o=!1;function a(C,F,I,O,z){let H=!1;const V=f(O,I,F);r!==V&&(r=V,c(r.object)),H=h(C,O,I,z),H&&x(C,O,I,z),z!==null&&e.update(z,i.ELEMENT_ARRAY_BUFFER),(H||o)&&(o=!1,A(C,F,I,O),z!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,e.get(z).buffer))}function l(){return i.createVertexArray()}function c(C){return i.bindVertexArray(C)}function u(C){return i.deleteVertexArray(C)}function f(C,F,I){const O=I.wireframe===!0;let z=n[C.id];z===void 0&&(z={},n[C.id]=z);let H=z[F.id];H===void 0&&(H={},z[F.id]=H);let V=H[O];return V===void 0&&(V=d(l()),H[O]=V),V}function d(C){const F=[],I=[],O=[];for(let z=0;z<t;z++)F[z]=0,I[z]=0,O[z]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:F,enabledAttributes:I,attributeDivisors:O,object:C,attributes:{},index:null}}function h(C,F,I,O){const z=r.attributes,H=F.attributes;let V=0;const q=I.getAttributes();for(const G in q)if(q[G].location>=0){const ae=z[G];let Ae=H[G];if(Ae===void 0&&(G==="instanceMatrix"&&C.instanceMatrix&&(Ae=C.instanceMatrix),G==="instanceColor"&&C.instanceColor&&(Ae=C.instanceColor)),ae===void 0||ae.attribute!==Ae||Ae&&ae.data!==Ae.data)return!0;V++}return r.attributesNum!==V||r.index!==O}function x(C,F,I,O){const z={},H=F.attributes;let V=0;const q=I.getAttributes();for(const G in q)if(q[G].location>=0){let ae=H[G];ae===void 0&&(G==="instanceMatrix"&&C.instanceMatrix&&(ae=C.instanceMatrix),G==="instanceColor"&&C.instanceColor&&(ae=C.instanceColor));const Ae={};Ae.attribute=ae,ae&&ae.data&&(Ae.data=ae.data),z[G]=Ae,V++}r.attributes=z,r.attributesNum=V,r.index=O}function p(){const C=r.newAttributes;for(let F=0,I=C.length;F<I;F++)C[F]=0}function g(C){m(C,0)}function m(C,F){const I=r.newAttributes,O=r.enabledAttributes,z=r.attributeDivisors;I[C]=1,O[C]===0&&(i.enableVertexAttribArray(C),O[C]=1),z[C]!==F&&(i.vertexAttribDivisor(C,F),z[C]=F)}function _(){const C=r.newAttributes,F=r.enabledAttributes;for(let I=0,O=F.length;I<O;I++)F[I]!==C[I]&&(i.disableVertexAttribArray(I),F[I]=0)}function v(C,F,I,O,z,H,V){V===!0?i.vertexAttribIPointer(C,F,I,z,H):i.vertexAttribPointer(C,F,I,O,z,H)}function A(C,F,I,O){p();const z=O.attributes,H=I.getAttributes(),V=F.defaultAttributeValues;for(const q in H){const G=H[q];if(G.location>=0){let Y=z[q];if(Y===void 0&&(q==="instanceMatrix"&&C.instanceMatrix&&(Y=C.instanceMatrix),q==="instanceColor"&&C.instanceColor&&(Y=C.instanceColor)),Y!==void 0){const ae=Y.normalized,Ae=Y.itemSize,Pe=e.get(Y);if(Pe===void 0)continue;const Ne=Pe.buffer,We=Pe.type,He=Pe.bytesPerElement,J=We===i.INT||We===i.UNSIGNED_INT||Y.gpuType===yd;if(Y.isInterleavedBufferAttribute){const le=Y.data,be=le.stride,ze=Y.offset;if(le.isInstancedInterleavedBuffer){for(let Ce=0;Ce<G.locationSize;Ce++)m(G.location+Ce,le.meshPerAttribute);C.isInstancedMesh!==!0&&O._maxInstanceCount===void 0&&(O._maxInstanceCount=le.meshPerAttribute*le.count)}else for(let Ce=0;Ce<G.locationSize;Ce++)g(G.location+Ce);i.bindBuffer(i.ARRAY_BUFFER,Ne);for(let Ce=0;Ce<G.locationSize;Ce++)v(G.location+Ce,Ae/G.locationSize,We,ae,be*He,(ze+Ae/G.locationSize*Ce)*He,J)}else{if(Y.isInstancedBufferAttribute){for(let le=0;le<G.locationSize;le++)m(G.location+le,Y.meshPerAttribute);C.isInstancedMesh!==!0&&O._maxInstanceCount===void 0&&(O._maxInstanceCount=Y.meshPerAttribute*Y.count)}else for(let le=0;le<G.locationSize;le++)g(G.location+le);i.bindBuffer(i.ARRAY_BUFFER,Ne);for(let le=0;le<G.locationSize;le++)v(G.location+le,Ae/G.locationSize,We,ae,Ae*He,Ae/G.locationSize*le*He,J)}}else if(V!==void 0){const ae=V[q];if(ae!==void 0)switch(ae.length){case 2:i.vertexAttrib2fv(G.location,ae);break;case 3:i.vertexAttrib3fv(G.location,ae);break;case 4:i.vertexAttrib4fv(G.location,ae);break;default:i.vertexAttrib1fv(G.location,ae)}}}}_()}function S(){R();for(const C in n){const F=n[C];for(const I in F){const O=F[I];for(const z in O)u(O[z].object),delete O[z];delete F[I]}delete n[C]}}function b(C){if(n[C.id]===void 0)return;const F=n[C.id];for(const I in F){const O=F[I];for(const z in O)u(O[z].object),delete O[z];delete F[I]}delete n[C.id]}function M(C){for(const F in n){const I=n[F];if(I[C.id]===void 0)continue;const O=I[C.id];for(const z in O)u(O[z].object),delete O[z];delete I[C.id]}}function R(){y(),o=!0,r!==s&&(r=s,c(r.object))}function y(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:a,reset:R,resetDefaultState:y,dispose:S,releaseStatesOfGeometry:b,releaseStatesOfProgram:M,initAttributes:p,enableAttribute:g,disableUnusedAttributes:_}}function kM(i,e,t){let n;function s(c){n=c}function r(c,u){i.drawArrays(n,c,u),t.update(u,n,1)}function o(c,u,f){f!==0&&(i.drawArraysInstanced(n,c,u,f),t.update(u,n,f))}function a(c,u,f){if(f===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,c,0,u,0,f);let h=0;for(let x=0;x<f;x++)h+=u[x];t.update(h,n,1)}function l(c,u,f,d){if(f===0)return;const h=e.get("WEBGL_multi_draw");if(h===null)for(let x=0;x<c.length;x++)o(c[x],u[x],d[x]);else{h.multiDrawArraysInstancedWEBGL(n,c,0,u,0,d,0,f);let x=0;for(let p=0;p<f;p++)x+=u[p]*d[p];t.update(x,n,1)}}this.setMode=s,this.render=r,this.renderInstances=o,this.renderMultiDraw=a,this.renderMultiDrawInstances=l}function HM(i,e,t,n){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){const M=e.get("EXT_texture_filter_anisotropic");s=i.getParameter(M.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function o(M){return!(M!==Pn&&n.convert(M)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function a(M){const R=M===wr&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(M!==Gi&&n.convert(M)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE)&&M!==bi&&!R)}function l(M){if(M==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";M="mediump"}return M==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let c=t.precision!==void 0?t.precision:"highp";const u=l(c);u!==c&&(et("WebGLRenderer:",c,"not supported, using",u,"instead."),c=u);const f=t.logarithmicDepthBuffer===!0,d=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),h=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),x=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),p=i.getParameter(i.MAX_TEXTURE_SIZE),g=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),m=i.getParameter(i.MAX_VERTEX_ATTRIBS),_=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),v=i.getParameter(i.MAX_VARYING_VECTORS),A=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),S=x>0,b=i.getParameter(i.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:l,textureFormatReadable:o,textureTypeReadable:a,precision:c,logarithmicDepthBuffer:f,reversedDepthBuffer:d,maxTextures:h,maxVertexTextures:x,maxTextureSize:p,maxCubemapSize:g,maxAttributes:m,maxVertexUniforms:_,maxVaryings:v,maxFragmentUniforms:A,vertexTextures:S,maxSamples:b}}function VM(i){const e=this;let t=null,n=0,s=!1,r=!1;const o=new Ps,a=new Ze,l={value:null,needsUpdate:!1};this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(f,d){const h=f.length!==0||d||n!==0||s;return s=d,n=f.length,h},this.beginShadows=function(){r=!0,u(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(f,d){t=u(f,d,0)},this.setState=function(f,d,h){const x=f.clippingPlanes,p=f.clipIntersection,g=f.clipShadows,m=i.get(f);if(!s||x===null||x.length===0||r&&!g)r?u(null):c();else{const _=r?0:n,v=_*4;let A=m.clippingState||null;l.value=A,A=u(x,d,v,h);for(let S=0;S!==v;++S)A[S]=t[S];m.clippingState=A,this.numIntersection=p?this.numPlanes:0,this.numPlanes+=_}};function c(){l.value!==t&&(l.value=t,l.needsUpdate=n>0),e.numPlanes=n,e.numIntersection=0}function u(f,d,h,x){const p=f!==null?f.length:0;let g=null;if(p!==0){if(g=l.value,x!==!0||g===null){const m=h+p*4,_=d.matrixWorldInverse;a.getNormalMatrix(_),(g===null||g.length<m)&&(g=new Float32Array(m));for(let v=0,A=h;v!==p;++v,A+=4)o.copy(f[v]).applyMatrix4(_,a),o.normal.toArray(g,A),g[A+3]=o.constant}l.value=g,l.needsUpdate=!0}return e.numPlanes=p,e.numIntersection=0,g}}function GM(i){let e=new WeakMap;function t(o,a){return a===of?o.mapping=Mo:a===af&&(o.mapping=Co),o}function n(o){if(o&&o.isTexture){const a=o.mapping;if(a===of||a===af)if(e.has(o)){const l=e.get(o).texture;return t(l,o.mapping)}else{const l=o.image;if(l&&l.height>0){const c=new HS(l.height);return c.fromEquirectangularTexture(i,o),e.set(o,c),o.addEventListener("dispose",s),t(c.texture,o.mapping)}else return null}}return o}function s(o){const a=o.target;a.removeEventListener("dispose",s);const l=e.get(a);l!==void 0&&(e.delete(a),l.dispose())}function r(){e=new WeakMap}return{get:n,dispose:r}}const Ls=4,Mp=[.125,.215,.35,.446,.526,.582],xr=20,WM=256,Zo=new Fd,Cp=new lt;let xu=null,_u=0,vu=0,Au=!1;const XM=new U;class Tp{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,n=.1,s=100,r={}){const{size:o=256,position:a=XM}=r;xu=this._renderer.getRenderTarget(),_u=this._renderer.getActiveCubeFace(),vu=this._renderer.getActiveMipmapLevel(),Au=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(o);const l=this._allocateTargets();return l.depthBuffer=!0,this._sceneToCubeUV(e,n,s,l,a),t>0&&this._blur(l,0,0,t),this._applyPMREM(l),this._cleanup(l),l}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Rp(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=wp(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(xu,_u,vu),this._renderer.xr.enabled=Au,e.scissorTest=!1,Kr(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===Mo||e.mapping===Co?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),xu=this._renderer.getRenderTarget(),_u=this._renderer.getActiveCubeFace(),vu=this._renderer.getActiveMipmapLevel(),Au=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const n=t||this._allocateTargets();return this._textureToCubeUV(e,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,n={magFilter:di,minFilter:di,generateMipmaps:!1,type:wr,format:Pn,colorSpace:Eo,depthBuffer:!1},s=Ep(e,t,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=Ep(e,t,n);const{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=qM(r)),this._blurMaterial=QM(r,e,t),this._ggxMaterial=YM(r,e,t)}return s}_compileMaterial(e){const t=new Qt(new Bn,e);this._renderer.compile(t,Zo)}_sceneToCubeUV(e,t,n,s,r){const l=new ci(90,1,t,n),c=[1,-1,1,1,1,1],u=[1,1,1,-1,-1,-1],f=this._renderer,d=f.autoClear,h=f.toneMapping;f.getClearColor(Cp),f.toneMapping=Hs,f.autoClear=!1,f.state.buffers.depth.getReversed()&&(f.setRenderTarget(s),f.clearDepth(),f.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new Qt(new No,new Er({name:"PMREM.Background",side:kn,depthWrite:!1,depthTest:!1})));const p=this._backgroundBox,g=p.material;let m=!1;const _=e.background;_?_.isColor&&(g.color.copy(_),e.background=null,m=!0):(g.color.copy(Cp),m=!0);for(let v=0;v<6;v++){const A=v%3;A===0?(l.up.set(0,c[v],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x+u[v],r.y,r.z)):A===1?(l.up.set(0,0,c[v]),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y+u[v],r.z)):(l.up.set(0,c[v],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y,r.z+u[v]));const S=this._cubeSize;Kr(s,A*S,v>2?S:0,S,S),f.setRenderTarget(s),m&&f.render(p,l),f.render(e,l)}f.toneMapping=h,f.autoClear=d,e.background=_}_textureToCubeUV(e,t){const n=this._renderer,s=e.mapping===Mo||e.mapping===Co;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=Rp()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=wp());const r=s?this._cubemapMaterial:this._equirectMaterial,o=this._lodMeshes[0];o.material=r;const a=r.uniforms;a.envMap.value=e;const l=this._cubeSize;Kr(t,0,0,3*l,2*l),n.setRenderTarget(t),n.render(o,Zo)}_applyPMREM(e){const t=this._renderer,n=t.autoClear;t.autoClear=!1;const s=this._lodMeshes.length;for(let r=1;r<s;r++)this._applyGGXFilter(e,r-1,r);t.autoClear=n}_applyGGXFilter(e,t,n){const s=this._renderer,r=this._pingPongRenderTarget,o=this._ggxMaterial,a=this._lodMeshes[n];a.material=o;const l=o.uniforms,c=n/(this._lodMeshes.length-1),u=t/(this._lodMeshes.length-1),f=Math.sqrt(c*c-u*u),d=.05+c*.95,h=f*d,{_lodMax:x}=this,p=this._sizeLods[n],g=3*p*(n>x-Ls?n-x+Ls:0),m=4*(this._cubeSize-p);l.envMap.value=e.texture,l.roughness.value=h,l.mipInt.value=x-t,Kr(r,g,m,3*p,2*p),s.setRenderTarget(r),s.render(a,Zo),l.envMap.value=r.texture,l.roughness.value=0,l.mipInt.value=x-n,Kr(e,g,m,3*p,2*p),s.setRenderTarget(e),s.render(a,Zo)}_blur(e,t,n,s,r){const o=this._pingPongRenderTarget;this._halfBlur(e,o,t,n,s,"latitudinal",r),this._halfBlur(o,e,n,n,s,"longitudinal",r)}_halfBlur(e,t,n,s,r,o,a){const l=this._renderer,c=this._blurMaterial;o!=="latitudinal"&&o!=="longitudinal"&&Xt("blur direction must be either latitudinal or longitudinal!");const u=3,f=this._lodMeshes[s];f.material=c;const d=c.uniforms,h=this._sizeLods[n]-1,x=isFinite(r)?Math.PI/(2*h):2*Math.PI/(2*xr-1),p=r/x,g=isFinite(r)?1+Math.floor(u*p):xr;g>xr&&et(`sigmaRadians, ${r}, is too large and will clip, as it requested ${g} samples when the maximum is set to ${xr}`);const m=[];let _=0;for(let M=0;M<xr;++M){const R=M/p,y=Math.exp(-R*R/2);m.push(y),M===0?_+=y:M<g&&(_+=2*y)}for(let M=0;M<m.length;M++)m[M]=m[M]/_;d.envMap.value=e.texture,d.samples.value=g,d.weights.value=m,d.latitudinal.value=o==="latitudinal",a&&(d.poleAxis.value=a);const{_lodMax:v}=this;d.dTheta.value=x,d.mipInt.value=v-n;const A=this._sizeLods[s],S=3*A*(s>v-Ls?s-v+Ls:0),b=4*(this._cubeSize-A);Kr(t,S,b,3*A,2*A),l.setRenderTarget(t),l.render(f,Zo)}}function qM(i){const e=[],t=[],n=[];let s=i;const r=i-Ls+1+Mp.length;for(let o=0;o<r;o++){const a=Math.pow(2,s);e.push(a);let l=1/a;o>i-Ls?l=Mp[o-i+Ls-1]:o===0&&(l=0),t.push(l);const c=1/(a-2),u=-c,f=1+c,d=[u,u,f,u,f,f,u,u,f,f,u,f],h=6,x=6,p=3,g=2,m=1,_=new Float32Array(p*x*h),v=new Float32Array(g*x*h),A=new Float32Array(m*x*h);for(let b=0;b<h;b++){const M=b%3*2/3-1,R=b>2?0:-1,y=[M,R,0,M+2/3,R,0,M+2/3,R+1,0,M,R,0,M+2/3,R+1,0,M,R+1,0];_.set(y,p*x*b),v.set(d,g*x*b);const C=[b,b,b,b,b,b];A.set(C,m*x*b)}const S=new Bn;S.setAttribute("position",new xi(_,p)),S.setAttribute("uv",new xi(v,g)),S.setAttribute("faceIndex",new xi(A,m)),n.push(new Qt(S,null)),s>Ls&&s--}return{lodMeshes:n,sizeLods:e,sigmas:t}}function Ep(i,e,t){const n=new qs(i,e,t);return n.texture.mapping=bc,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function Kr(i,e,t,n,s){i.viewport.set(e,t,n,s),i.scissor.set(e,t,n,s)}function YM(i,e,t){return new Ln({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:WM,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:Tc(),fragmentShader:`

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
		`,blending:ds,depthTest:!1,depthWrite:!1})}function QM(i,e,t){const n=new Float32Array(xr),s=new U(0,1,0);return new Ln({name:"SphericalGaussianBlur",defines:{n:xr,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:Tc(),fragmentShader:`

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
		`,blending:ds,depthTest:!1,depthWrite:!1})}function wp(){return new Ln({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Tc(),fragmentShader:`

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
		`,blending:ds,depthTest:!1,depthWrite:!1})}function Rp(){return new Ln({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Tc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:ds,depthTest:!1,depthWrite:!1})}function Tc(){return`

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
	`}function KM(i){let e=new WeakMap,t=null;function n(a){if(a&&a.isTexture){const l=a.mapping,c=l===of||l===af,u=l===Mo||l===Co;if(c||u){let f=e.get(a);const d=f!==void 0?f.texture.pmremVersion:0;if(a.isRenderTargetTexture&&a.pmremVersion!==d)return t===null&&(t=new Tp(i)),f=c?t.fromEquirectangular(a,f):t.fromCubemap(a,f),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),f.texture;if(f!==void 0)return f.texture;{const h=a.image;return c&&h&&h.height>0||u&&h&&s(h)?(t===null&&(t=new Tp(i)),f=c?t.fromEquirectangular(a):t.fromCubemap(a),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),a.addEventListener("dispose",r),f.texture):null}}}return a}function s(a){let l=0;const c=6;for(let u=0;u<c;u++)a[u]!==void 0&&l++;return l===c}function r(a){const l=a.target;l.removeEventListener("dispose",r);const c=e.get(l);c!==void 0&&(e.delete(l),c.dispose())}function o(){e=new WeakMap,t!==null&&(t.dispose(),t=null)}return{get:n,dispose:o}}function jM(i){const e={};function t(n){if(e[n]!==void 0)return e[n];const s=i.getExtension(n);return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(n){const s=t(n);return s===null&&Da("WebGLRenderer: "+n+" extension not supported."),s}}}function $M(i,e,t,n){const s={},r=new WeakMap;function o(f){const d=f.target;d.index!==null&&e.remove(d.index);for(const x in d.attributes)e.remove(d.attributes[x]);d.removeEventListener("dispose",o),delete s[d.id];const h=r.get(d);h&&(e.remove(h),r.delete(d)),n.releaseStatesOfGeometry(d),d.isInstancedBufferGeometry===!0&&delete d._maxInstanceCount,t.memory.geometries--}function a(f,d){return s[d.id]===!0||(d.addEventListener("dispose",o),s[d.id]=!0,t.memory.geometries++),d}function l(f){const d=f.attributes;for(const h in d)e.update(d[h],i.ARRAY_BUFFER)}function c(f){const d=[],h=f.index,x=f.attributes.position;let p=0;if(h!==null){const _=h.array;p=h.version;for(let v=0,A=_.length;v<A;v+=3){const S=_[v+0],b=_[v+1],M=_[v+2];d.push(S,b,b,M,M,S)}}else if(x!==void 0){const _=x.array;p=x.version;for(let v=0,A=_.length/3-1;v<A;v+=3){const S=v+0,b=v+1,M=v+2;d.push(S,b,b,M,M,S)}}else return;const g=new(cg(d)?pg:hg)(d,1);g.version=p;const m=r.get(f);m&&e.remove(m),r.set(f,g)}function u(f){const d=r.get(f);if(d){const h=f.index;h!==null&&d.version<h.version&&c(f)}else c(f);return r.get(f)}return{get:a,update:l,getWireframeAttribute:u}}function ZM(i,e,t){let n;function s(d){n=d}let r,o;function a(d){r=d.type,o=d.bytesPerElement}function l(d,h){i.drawElements(n,h,r,d*o),t.update(h,n,1)}function c(d,h,x){x!==0&&(i.drawElementsInstanced(n,h,r,d*o,x),t.update(h,n,x))}function u(d,h,x){if(x===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,h,0,r,d,0,x);let g=0;for(let m=0;m<x;m++)g+=h[m];t.update(g,n,1)}function f(d,h,x,p){if(x===0)return;const g=e.get("WEBGL_multi_draw");if(g===null)for(let m=0;m<d.length;m++)c(d[m]/o,h[m],p[m]);else{g.multiDrawElementsInstancedWEBGL(n,h,0,r,d,0,p,0,x);let m=0;for(let _=0;_<x;_++)m+=h[_]*p[_];t.update(m,n,1)}}this.setMode=s,this.setIndex=a,this.render=l,this.renderInstances=c,this.renderMultiDraw=u,this.renderMultiDrawInstances=f}function JM(i){const e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function n(r,o,a){switch(t.calls++,o){case i.TRIANGLES:t.triangles+=a*(r/3);break;case i.LINES:t.lines+=a*(r/2);break;case i.LINE_STRIP:t.lines+=a*(r-1);break;case i.LINE_LOOP:t.lines+=a*r;break;case i.POINTS:t.points+=a*r;break;default:Xt("WebGLInfo: Unknown draw mode:",o);break}}function s(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:s,update:n}}function eC(i,e,t){const n=new WeakMap,s=new Ot;function r(o,a,l){const c=o.morphTargetInfluences,u=a.morphAttributes.position||a.morphAttributes.normal||a.morphAttributes.color,f=u!==void 0?u.length:0;let d=n.get(a);if(d===void 0||d.count!==f){let C=function(){R.dispose(),n.delete(a),a.removeEventListener("dispose",C)};var h=C;d!==void 0&&d.texture.dispose();const x=a.morphAttributes.position!==void 0,p=a.morphAttributes.normal!==void 0,g=a.morphAttributes.color!==void 0,m=a.morphAttributes.position||[],_=a.morphAttributes.normal||[],v=a.morphAttributes.color||[];let A=0;x===!0&&(A=1),p===!0&&(A=2),g===!0&&(A=3);let S=a.attributes.position.count*A,b=1;S>e.maxTextureSize&&(b=Math.ceil(S/e.maxTextureSize),S=e.maxTextureSize);const M=new Float32Array(S*b*4*f),R=new ug(M,S,b,f);R.type=bi,R.needsUpdate=!0;const y=A*4;for(let F=0;F<f;F++){const I=m[F],O=_[F],z=v[F],H=S*b*4*F;for(let V=0;V<I.count;V++){const q=V*y;x===!0&&(s.fromBufferAttribute(I,V),M[H+q+0]=s.x,M[H+q+1]=s.y,M[H+q+2]=s.z,M[H+q+3]=0),p===!0&&(s.fromBufferAttribute(O,V),M[H+q+4]=s.x,M[H+q+5]=s.y,M[H+q+6]=s.z,M[H+q+7]=0),g===!0&&(s.fromBufferAttribute(z,V),M[H+q+8]=s.x,M[H+q+9]=s.y,M[H+q+10]=s.z,M[H+q+11]=z.itemSize===4?s.w:1)}}d={count:f,texture:R,size:new Xe(S,b)},n.set(a,d),a.addEventListener("dispose",C)}if(o.isInstancedMesh===!0&&o.morphTexture!==null)l.getUniforms().setValue(i,"morphTexture",o.morphTexture,t);else{let x=0;for(let g=0;g<c.length;g++)x+=c[g];const p=a.morphTargetsRelative?1:1-x;l.getUniforms().setValue(i,"morphTargetBaseInfluence",p),l.getUniforms().setValue(i,"morphTargetInfluences",c)}l.getUniforms().setValue(i,"morphTargetsTexture",d.texture,t),l.getUniforms().setValue(i,"morphTargetsTextureSize",d.size)}return{update:r}}function tC(i,e,t,n){let s=new WeakMap;function r(l){const c=n.render.frame,u=l.geometry,f=e.get(l,u);if(s.get(f)!==c&&(e.update(f),s.set(f,c)),l.isInstancedMesh&&(l.hasEventListener("dispose",a)===!1&&l.addEventListener("dispose",a),s.get(l)!==c&&(t.update(l.instanceMatrix,i.ARRAY_BUFFER),l.instanceColor!==null&&t.update(l.instanceColor,i.ARRAY_BUFFER),s.set(l,c))),l.isSkinnedMesh){const d=l.skeleton;s.get(d)!==c&&(d.update(),s.set(d,c))}return f}function o(){s=new WeakMap}function a(l){const c=l.target;c.removeEventListener("dispose",a),t.remove(c.instanceMatrix),c.instanceColor!==null&&t.remove(c.instanceColor)}return{update:r,dispose:o}}const Sg=new Fn,Ip=new Id(1,1),yg=new ug,bg=new SS,Mg=new xg,Dp=[],Pp=[],Fp=new Float32Array(16),Lp=new Float32Array(9),Bp=new Float32Array(4);function zo(i,e,t){const n=i[0];if(n<=0||n>0)return i;const s=e*t;let r=Dp[s];if(r===void 0&&(r=new Float32Array(s),Dp[s]=r),e!==0){n.toArray(r,0);for(let o=1,a=0;o!==e;++o)a+=t,i[o].toArray(r,a)}return r}function en(i,e){if(i.length!==e.length)return!1;for(let t=0,n=i.length;t<n;t++)if(i[t]!==e[t])return!1;return!0}function tn(i,e){for(let t=0,n=e.length;t<n;t++)i[t]=e[t]}function Ec(i,e){let t=Pp[e];t===void 0&&(t=new Int32Array(e),Pp[e]=t);for(let n=0;n!==e;++n)t[n]=i.allocateTextureUnit();return t}function nC(i,e){const t=this.cache;t[0]!==e&&(i.uniform1f(this.addr,e),t[0]=e)}function iC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(en(t,e))return;i.uniform2fv(this.addr,e),tn(t,e)}}function sC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(i.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(en(t,e))return;i.uniform3fv(this.addr,e),tn(t,e)}}function rC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(en(t,e))return;i.uniform4fv(this.addr,e),tn(t,e)}}function oC(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(en(t,e))return;i.uniformMatrix2fv(this.addr,!1,e),tn(t,e)}else{if(en(t,n))return;Bp.set(n),i.uniformMatrix2fv(this.addr,!1,Bp),tn(t,n)}}function aC(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(en(t,e))return;i.uniformMatrix3fv(this.addr,!1,e),tn(t,e)}else{if(en(t,n))return;Lp.set(n),i.uniformMatrix3fv(this.addr,!1,Lp),tn(t,n)}}function lC(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(en(t,e))return;i.uniformMatrix4fv(this.addr,!1,e),tn(t,e)}else{if(en(t,n))return;Fp.set(n),i.uniformMatrix4fv(this.addr,!1,Fp),tn(t,n)}}function cC(i,e){const t=this.cache;t[0]!==e&&(i.uniform1i(this.addr,e),t[0]=e)}function uC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(en(t,e))return;i.uniform2iv(this.addr,e),tn(t,e)}}function fC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(en(t,e))return;i.uniform3iv(this.addr,e),tn(t,e)}}function dC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(en(t,e))return;i.uniform4iv(this.addr,e),tn(t,e)}}function hC(i,e){const t=this.cache;t[0]!==e&&(i.uniform1ui(this.addr,e),t[0]=e)}function pC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(en(t,e))return;i.uniform2uiv(this.addr,e),tn(t,e)}}function mC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(en(t,e))return;i.uniform3uiv(this.addr,e),tn(t,e)}}function gC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(en(t,e))return;i.uniform4uiv(this.addr,e),tn(t,e)}}function xC(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s);let r;this.type===i.SAMPLER_2D_SHADOW?(Ip.compareFunction=lg,r=Ip):r=Sg,t.setTexture2D(e||r,s)}function _C(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture3D(e||bg,s)}function vC(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTextureCube(e||Mg,s)}function AC(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture2DArray(e||yg,s)}function SC(i){switch(i){case 5126:return nC;case 35664:return iC;case 35665:return sC;case 35666:return rC;case 35674:return oC;case 35675:return aC;case 35676:return lC;case 5124:case 35670:return cC;case 35667:case 35671:return uC;case 35668:case 35672:return fC;case 35669:case 35673:return dC;case 5125:return hC;case 36294:return pC;case 36295:return mC;case 36296:return gC;case 35678:case 36198:case 36298:case 36306:case 35682:return xC;case 35679:case 36299:case 36307:return _C;case 35680:case 36300:case 36308:case 36293:return vC;case 36289:case 36303:case 36311:case 36292:return AC}}function yC(i,e){i.uniform1fv(this.addr,e)}function bC(i,e){const t=zo(e,this.size,2);i.uniform2fv(this.addr,t)}function MC(i,e){const t=zo(e,this.size,3);i.uniform3fv(this.addr,t)}function CC(i,e){const t=zo(e,this.size,4);i.uniform4fv(this.addr,t)}function TC(i,e){const t=zo(e,this.size,4);i.uniformMatrix2fv(this.addr,!1,t)}function EC(i,e){const t=zo(e,this.size,9);i.uniformMatrix3fv(this.addr,!1,t)}function wC(i,e){const t=zo(e,this.size,16);i.uniformMatrix4fv(this.addr,!1,t)}function RC(i,e){i.uniform1iv(this.addr,e)}function IC(i,e){i.uniform2iv(this.addr,e)}function DC(i,e){i.uniform3iv(this.addr,e)}function PC(i,e){i.uniform4iv(this.addr,e)}function FC(i,e){i.uniform1uiv(this.addr,e)}function LC(i,e){i.uniform2uiv(this.addr,e)}function BC(i,e){i.uniform3uiv(this.addr,e)}function UC(i,e){i.uniform4uiv(this.addr,e)}function OC(i,e,t){const n=this.cache,s=e.length,r=Ec(t,s);en(n,r)||(i.uniform1iv(this.addr,r),tn(n,r));for(let o=0;o!==s;++o)t.setTexture2D(e[o]||Sg,r[o])}function NC(i,e,t){const n=this.cache,s=e.length,r=Ec(t,s);en(n,r)||(i.uniform1iv(this.addr,r),tn(n,r));for(let o=0;o!==s;++o)t.setTexture3D(e[o]||bg,r[o])}function zC(i,e,t){const n=this.cache,s=e.length,r=Ec(t,s);en(n,r)||(i.uniform1iv(this.addr,r),tn(n,r));for(let o=0;o!==s;++o)t.setTextureCube(e[o]||Mg,r[o])}function kC(i,e,t){const n=this.cache,s=e.length,r=Ec(t,s);en(n,r)||(i.uniform1iv(this.addr,r),tn(n,r));for(let o=0;o!==s;++o)t.setTexture2DArray(e[o]||yg,r[o])}function HC(i){switch(i){case 5126:return yC;case 35664:return bC;case 35665:return MC;case 35666:return CC;case 35674:return TC;case 35675:return EC;case 35676:return wC;case 5124:case 35670:return RC;case 35667:case 35671:return IC;case 35668:case 35672:return DC;case 35669:case 35673:return PC;case 5125:return FC;case 36294:return LC;case 36295:return BC;case 36296:return UC;case 35678:case 36198:case 36298:case 36306:case 35682:return OC;case 35679:case 36299:case 36307:return NC;case 35680:case 36300:case 36308:case 36293:return zC;case 36289:case 36303:case 36311:case 36292:return kC}}class VC{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.setValue=SC(t.type)}}class GC{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=HC(t.type)}}class WC{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,n){const s=this.seq;for(let r=0,o=s.length;r!==o;++r){const a=s[r];a.setValue(e,t[a.id],n)}}}const Su=/(\w+)(\])?(\[|\.)?/g;function Up(i,e){i.seq.push(e),i.map[e.id]=e}function XC(i,e,t){const n=i.name,s=n.length;for(Su.lastIndex=0;;){const r=Su.exec(n),o=Su.lastIndex;let a=r[1];const l=r[2]==="]",c=r[3];if(l&&(a=a|0),c===void 0||c==="["&&o+2===s){Up(t,c===void 0?new VC(a,i,e):new GC(a,i,e));break}else{let f=t.map[a];f===void 0&&(f=new WC(a),Up(t,f)),t=f}}}class Vl{constructor(e,t){this.seq=[],this.map={};const n=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let s=0;s<n;++s){const r=e.getActiveUniform(t,s),o=e.getUniformLocation(t,r.name);XC(r,o,this)}}setValue(e,t,n,s){const r=this.map[t];r!==void 0&&r.setValue(e,n,s)}setOptional(e,t,n){const s=t[n];s!==void 0&&this.setValue(e,n,s)}static upload(e,t,n,s){for(let r=0,o=t.length;r!==o;++r){const a=t[r],l=n[a.id];l.needsUpdate!==!1&&a.setValue(e,l.value,s)}}static seqWithValue(e,t){const n=[];for(let s=0,r=e.length;s!==r;++s){const o=e[s];o.id in t&&n.push(o)}return n}}function Op(i,e,t){const n=i.createShader(e);return i.shaderSource(n,t),i.compileShader(n),n}const qC=37297;let YC=0;function QC(i,e){const t=i.split(`
`),n=[],s=Math.max(e-6,0),r=Math.min(e+6,t.length);for(let o=s;o<r;o++){const a=o+1;n.push(`${a===e?">":" "} ${a}: ${t[o]}`)}return n.join(`
`)}const Np=new Ze;function KC(i){dt._getMatrix(Np,dt.workingColorSpace,i);const e=`mat3( ${Np.elements.map(t=>t.toFixed(4))} )`;switch(dt.getTransfer(i)){case Jl:return[e,"LinearTransferOETF"];case yt:return[e,"sRGBTransferOETF"];default:return et("WebGLProgram: Unsupported color space: ",i),[e,"LinearTransferOETF"]}}function zp(i,e,t){const n=i.getShaderParameter(e,i.COMPILE_STATUS),r=(i.getShaderInfoLog(e)||"").trim();if(n&&r==="")return"";const o=/ERROR: 0:(\d+)/.exec(r);if(o){const a=parseInt(o[1]);return t.toUpperCase()+`

`+r+`

`+QC(i.getShaderSource(e),a)}else return r}function jC(i,e){const t=KC(e);return[`vec4 ${i}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}function $C(i,e){let t;switch(e){case PA:t="Linear";break;case FA:t="Reinhard";break;case LA:t="Cineon";break;case BA:t="ACESFilmic";break;case OA:t="AgX";break;case NA:t="Neutral";break;case UA:t="Custom";break;default:et("WebGLProgram: Unsupported toneMapping:",e),t="Linear"}return"vec3 "+i+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}const Ml=new U;function ZC(){dt.getLuminanceCoefficients(Ml);const i=Ml.x.toFixed(4),e=Ml.y.toFixed(4),t=Ml.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function JC(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(ta).join(`
`)}function eT(i){const e=[];for(const t in i){const n=i[t];n!==!1&&e.push("#define "+t+" "+n)}return e.join(`
`)}function tT(i,e){const t={},n=i.getProgramParameter(e,i.ACTIVE_ATTRIBUTES);for(let s=0;s<n;s++){const r=i.getActiveAttrib(e,s),o=r.name;let a=1;r.type===i.FLOAT_MAT2&&(a=2),r.type===i.FLOAT_MAT3&&(a=3),r.type===i.FLOAT_MAT4&&(a=4),t[o]={type:r.type,location:i.getAttribLocation(e,o),locationSize:a}}return t}function ta(i){return i!==""}function kp(i,e){const t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return i.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function Hp(i,e){return i.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const nT=/^[ \t]*#include +<([\w\d./]+)>/gm;function zf(i){return i.replace(nT,sT)}const iT=new Map;function sT(i,e){let t=st[e];if(t===void 0){const n=iT.get(e);if(n!==void 0)t=st[n],et('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,n);else throw new Error("Can not resolve #include <"+e+">")}return zf(t)}const rT=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function Vp(i){return i.replace(rT,oT)}function oT(i,e,t,n){let s="";for(let r=parseInt(e);r<parseInt(t);r++)s+=n.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function Gp(i){let e=`precision ${i.precision} float;
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
#define LOW_PRECISION`),e}function aT(i){let e="SHADOWMAP_TYPE_BASIC";return i.shadowMapType===$0?e="SHADOWMAP_TYPE_PCF":i.shadowMapType===dA?e="SHADOWMAP_TYPE_PCF_SOFT":i.shadowMapType===ts&&(e="SHADOWMAP_TYPE_VSM"),e}function lT(i){let e="ENVMAP_TYPE_CUBE";if(i.envMap)switch(i.envMapMode){case Mo:case Co:e="ENVMAP_TYPE_CUBE";break;case bc:e="ENVMAP_TYPE_CUBE_UV";break}return e}function cT(i){let e="ENVMAP_MODE_REFLECTION";return i.envMap&&i.envMapMode===Co&&(e="ENVMAP_MODE_REFRACTION"),e}function uT(i){let e="ENVMAP_BLENDING_NONE";if(i.envMap)switch(i.combine){case J0:e="ENVMAP_BLENDING_MULTIPLY";break;case IA:e="ENVMAP_BLENDING_MIX";break;case DA:e="ENVMAP_BLENDING_ADD";break}return e}function fT(i){const e=i.envMapCubeUVHeight;if(e===null)return null;const t=Math.log2(e)-2,n=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:n,maxMip:t}}function dT(i,e,t,n){const s=i.getContext(),r=t.defines;let o=t.vertexShader,a=t.fragmentShader;const l=aT(t),c=lT(t),u=cT(t),f=uT(t),d=fT(t),h=JC(t),x=eT(r),p=s.createProgram();let g,m,_=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(g=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x].filter(ta).join(`
`),g.length>0&&(g+=`
`),m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x].filter(ta).join(`
`),m.length>0&&(m+=`
`)):(g=[Gp(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+u:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(ta).join(`
`),m=[Gp(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+c:"",t.envMap?"#define "+u:"",t.envMap?"#define "+f:"",d?"#define CUBEUV_TEXEL_WIDTH "+d.texelWidth:"",d?"#define CUBEUV_TEXEL_HEIGHT "+d.texelHeight:"",d?"#define CUBEUV_MAX_MIP "+d.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor||t.batchingColor?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==Hs?"#define TONE_MAPPING":"",t.toneMapping!==Hs?st.tonemapping_pars_fragment:"",t.toneMapping!==Hs?$C("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",st.colorspace_pars_fragment,jC("linearToOutputTexel",t.outputColorSpace),ZC(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(ta).join(`
`)),o=zf(o),o=kp(o,t),o=Hp(o,t),a=zf(a),a=kp(a,t),a=Hp(a,t),o=Vp(o),a=Vp(a),t.isRawShaderMaterial!==!0&&(_=`#version 300 es
`,g=[h,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+g,m=["#define varying in",t.glslVersion===ep?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===ep?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+m);const v=_+g+o,A=_+m+a,S=Op(s,s.VERTEX_SHADER,v),b=Op(s,s.FRAGMENT_SHADER,A);s.attachShader(p,S),s.attachShader(p,b),t.index0AttributeName!==void 0?s.bindAttribLocation(p,0,t.index0AttributeName):t.morphTargets===!0&&s.bindAttribLocation(p,0,"position"),s.linkProgram(p);function M(F){if(i.debug.checkShaderErrors){const I=s.getProgramInfoLog(p)||"",O=s.getShaderInfoLog(S)||"",z=s.getShaderInfoLog(b)||"",H=I.trim(),V=O.trim(),q=z.trim();let G=!0,Y=!0;if(s.getProgramParameter(p,s.LINK_STATUS)===!1)if(G=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(s,p,S,b);else{const ae=zp(s,S,"vertex"),Ae=zp(s,b,"fragment");Xt("THREE.WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(p,s.VALIDATE_STATUS)+`

Material Name: `+F.name+`
Material Type: `+F.type+`

Program Info Log: `+H+`
`+ae+`
`+Ae)}else H!==""?et("WebGLProgram: Program Info Log:",H):(V===""||q==="")&&(Y=!1);Y&&(F.diagnostics={runnable:G,programLog:H,vertexShader:{log:V,prefix:g},fragmentShader:{log:q,prefix:m}})}s.deleteShader(S),s.deleteShader(b),R=new Vl(s,p),y=tT(s,p)}let R;this.getUniforms=function(){return R===void 0&&M(this),R};let y;this.getAttributes=function(){return y===void 0&&M(this),y};let C=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return C===!1&&(C=s.getProgramParameter(p,qC)),C},this.destroy=function(){n.releaseStatesOfProgram(this),s.deleteProgram(p),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=YC++,this.cacheKey=e,this.usedTimes=1,this.program=p,this.vertexShader=S,this.fragmentShader=b,this}let hT=0;class pT{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const t=e.vertexShader,n=e.fragmentShader,s=this._getShaderStage(t),r=this._getShaderStage(n),o=this._getShaderCacheForMaterial(e);return o.has(s)===!1&&(o.add(s),s.usedTimes++),o.has(r)===!1&&(o.add(r),r.usedTimes++),this}remove(e){const t=this.materialCache.get(e);for(const n of t)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const t=this.materialCache;let n=t.get(e);return n===void 0&&(n=new Set,t.set(e,n)),n}_getShaderStage(e){const t=this.shaderCache;let n=t.get(e);return n===void 0&&(n=new mT(e),t.set(e,n)),n}}class mT{constructor(e){this.id=hT++,this.code=e,this.usedTimes=0}}function gT(i,e,t,n,s,r,o){const a=new fg,l=new pT,c=new Set,u=[],f=s.logarithmicDepthBuffer,d=s.vertexTextures;let h=s.precision;const x={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function p(y){return c.add(y),y===0?"uv":`uv${y}`}function g(y,C,F,I,O){const z=I.fog,H=O.geometry,V=y.isMeshStandardMaterial?I.environment:null,q=(y.isMeshStandardMaterial?t:e).get(y.envMap||V),G=q&&q.mapping===bc?q.image.height:null,Y=x[y.type];y.precision!==null&&(h=s.getMaxPrecision(y.precision),h!==y.precision&&et("WebGLProgram.getParameters:",y.precision,"not supported, using",h,"instead."));const ae=H.morphAttributes.position||H.morphAttributes.normal||H.morphAttributes.color,Ae=ae!==void 0?ae.length:0;let Pe=0;H.morphAttributes.position!==void 0&&(Pe=1),H.morphAttributes.normal!==void 0&&(Pe=2),H.morphAttributes.color!==void 0&&(Pe=3);let Ne,We,He,J;if(Y){const ht=Pi[Y];Ne=ht.vertexShader,We=ht.fragmentShader}else Ne=y.vertexShader,We=y.fragmentShader,l.update(y),He=l.getVertexShaderID(y),J=l.getFragmentShaderID(y);const le=i.getRenderTarget(),be=i.state.buffers.depth.getReversed(),ze=O.isInstancedMesh===!0,Ce=O.isBatchedMesh===!0,Qe=!!y.map,B=!!y.matcap,N=!!q,j=!!y.aoMap,P=!!y.lightMap,ie=!!y.bumpMap,re=!!y.normalMap,fe=!!y.displacementMap,ne=!!y.emissiveMap,he=!!y.metalnessMap,te=!!y.roughnessMap,ge=y.anisotropy>0,D=y.clearcoat>0,T=y.dispersion>0,X=y.iridescence>0,se=y.sheen>0,ce=y.transmission>0,Z=ge&&!!y.anisotropyMap,Be=D&&!!y.clearcoatMap,Se=D&&!!y.clearcoatNormalMap,Oe=D&&!!y.clearcoatRoughnessMap,k=X&&!!y.iridescenceMap,ee=X&&!!y.iridescenceThicknessMap,me=se&&!!y.sheenColorMap,Te=se&&!!y.sheenRoughnessMap,Fe=!!y.specularMap,Re=!!y.specularColorMap,Ye=!!y.specularIntensityMap,W=ce&&!!y.transmissionMap,De=ce&&!!y.thicknessMap,ye=!!y.gradientMap,Me=!!y.alphaMap,ve=y.alphaTest>0,de=!!y.alphaHash,ke=!!y.extensions;let je=Hs;y.toneMapped&&(le===null||le.isXRRenderTarget===!0)&&(je=i.toneMapping);const ct={shaderID:Y,shaderType:y.type,shaderName:y.name,vertexShader:Ne,fragmentShader:We,defines:y.defines,customVertexShaderID:He,customFragmentShaderID:J,isRawShaderMaterial:y.isRawShaderMaterial===!0,glslVersion:y.glslVersion,precision:h,batching:Ce,batchingColor:Ce&&O._colorsTexture!==null,instancing:ze,instancingColor:ze&&O.instanceColor!==null,instancingMorph:ze&&O.morphTexture!==null,supportsVertexTextures:d,outputColorSpace:le===null?i.outputColorSpace:le.isXRRenderTarget===!0?le.texture.colorSpace:Eo,alphaToCoverage:!!y.alphaToCoverage,map:Qe,matcap:B,envMap:N,envMapMode:N&&q.mapping,envMapCubeUVHeight:G,aoMap:j,lightMap:P,bumpMap:ie,normalMap:re,displacementMap:d&&fe,emissiveMap:ne,normalMapObjectSpace:re&&y.normalMapType===GA,normalMapTangentSpace:re&&y.normalMapType===VA,metalnessMap:he,roughnessMap:te,anisotropy:ge,anisotropyMap:Z,clearcoat:D,clearcoatMap:Be,clearcoatNormalMap:Se,clearcoatRoughnessMap:Oe,dispersion:T,iridescence:X,iridescenceMap:k,iridescenceThicknessMap:ee,sheen:se,sheenColorMap:me,sheenRoughnessMap:Te,specularMap:Fe,specularColorMap:Re,specularIntensityMap:Ye,transmission:ce,transmissionMap:W,thicknessMap:De,gradientMap:ye,opaque:y.transparent===!1&&y.blending===ks&&y.alphaToCoverage===!1,alphaMap:Me,alphaTest:ve,alphaHash:de,combine:y.combine,mapUv:Qe&&p(y.map.channel),aoMapUv:j&&p(y.aoMap.channel),lightMapUv:P&&p(y.lightMap.channel),bumpMapUv:ie&&p(y.bumpMap.channel),normalMapUv:re&&p(y.normalMap.channel),displacementMapUv:fe&&p(y.displacementMap.channel),emissiveMapUv:ne&&p(y.emissiveMap.channel),metalnessMapUv:he&&p(y.metalnessMap.channel),roughnessMapUv:te&&p(y.roughnessMap.channel),anisotropyMapUv:Z&&p(y.anisotropyMap.channel),clearcoatMapUv:Be&&p(y.clearcoatMap.channel),clearcoatNormalMapUv:Se&&p(y.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Oe&&p(y.clearcoatRoughnessMap.channel),iridescenceMapUv:k&&p(y.iridescenceMap.channel),iridescenceThicknessMapUv:ee&&p(y.iridescenceThicknessMap.channel),sheenColorMapUv:me&&p(y.sheenColorMap.channel),sheenRoughnessMapUv:Te&&p(y.sheenRoughnessMap.channel),specularMapUv:Fe&&p(y.specularMap.channel),specularColorMapUv:Re&&p(y.specularColorMap.channel),specularIntensityMapUv:Ye&&p(y.specularIntensityMap.channel),transmissionMapUv:W&&p(y.transmissionMap.channel),thicknessMapUv:De&&p(y.thicknessMap.channel),alphaMapUv:Me&&p(y.alphaMap.channel),vertexTangents:!!H.attributes.tangent&&(re||ge),vertexColors:y.vertexColors,vertexAlphas:y.vertexColors===!0&&!!H.attributes.color&&H.attributes.color.itemSize===4,pointsUvs:O.isPoints===!0&&!!H.attributes.uv&&(Qe||Me),fog:!!z,useFog:y.fog===!0,fogExp2:!!z&&z.isFogExp2,flatShading:y.flatShading===!0&&y.wireframe===!1,sizeAttenuation:y.sizeAttenuation===!0,logarithmicDepthBuffer:f,reversedDepthBuffer:be,skinning:O.isSkinnedMesh===!0,morphTargets:H.morphAttributes.position!==void 0,morphNormals:H.morphAttributes.normal!==void 0,morphColors:H.morphAttributes.color!==void 0,morphTargetsCount:Ae,morphTextureStride:Pe,numDirLights:C.directional.length,numPointLights:C.point.length,numSpotLights:C.spot.length,numSpotLightMaps:C.spotLightMap.length,numRectAreaLights:C.rectArea.length,numHemiLights:C.hemi.length,numDirLightShadows:C.directionalShadowMap.length,numPointLightShadows:C.pointShadowMap.length,numSpotLightShadows:C.spotShadowMap.length,numSpotLightShadowsWithMaps:C.numSpotLightShadowsWithMaps,numLightProbes:C.numLightProbes,numClippingPlanes:o.numPlanes,numClipIntersection:o.numIntersection,dithering:y.dithering,shadowMapEnabled:i.shadowMap.enabled&&F.length>0,shadowMapType:i.shadowMap.type,toneMapping:je,decodeVideoTexture:Qe&&y.map.isVideoTexture===!0&&dt.getTransfer(y.map.colorSpace)===yt,decodeVideoTextureEmissive:ne&&y.emissiveMap.isVideoTexture===!0&&dt.getTransfer(y.emissiveMap.colorSpace)===yt,premultipliedAlpha:y.premultipliedAlpha,doubleSided:y.side===ui,flipSided:y.side===kn,useDepthPacking:y.depthPacking>=0,depthPacking:y.depthPacking||0,index0AttributeName:y.index0AttributeName,extensionClipCullDistance:ke&&y.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(ke&&y.extensions.multiDraw===!0||Ce)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:y.customProgramCacheKey()};return ct.vertexUv1s=c.has(1),ct.vertexUv2s=c.has(2),ct.vertexUv3s=c.has(3),c.clear(),ct}function m(y){const C=[];if(y.shaderID?C.push(y.shaderID):(C.push(y.customVertexShaderID),C.push(y.customFragmentShaderID)),y.defines!==void 0)for(const F in y.defines)C.push(F),C.push(y.defines[F]);return y.isRawShaderMaterial===!1&&(_(C,y),v(C,y),C.push(i.outputColorSpace)),C.push(y.customProgramCacheKey),C.join()}function _(y,C){y.push(C.precision),y.push(C.outputColorSpace),y.push(C.envMapMode),y.push(C.envMapCubeUVHeight),y.push(C.mapUv),y.push(C.alphaMapUv),y.push(C.lightMapUv),y.push(C.aoMapUv),y.push(C.bumpMapUv),y.push(C.normalMapUv),y.push(C.displacementMapUv),y.push(C.emissiveMapUv),y.push(C.metalnessMapUv),y.push(C.roughnessMapUv),y.push(C.anisotropyMapUv),y.push(C.clearcoatMapUv),y.push(C.clearcoatNormalMapUv),y.push(C.clearcoatRoughnessMapUv),y.push(C.iridescenceMapUv),y.push(C.iridescenceThicknessMapUv),y.push(C.sheenColorMapUv),y.push(C.sheenRoughnessMapUv),y.push(C.specularMapUv),y.push(C.specularColorMapUv),y.push(C.specularIntensityMapUv),y.push(C.transmissionMapUv),y.push(C.thicknessMapUv),y.push(C.combine),y.push(C.fogExp2),y.push(C.sizeAttenuation),y.push(C.morphTargetsCount),y.push(C.morphAttributeCount),y.push(C.numDirLights),y.push(C.numPointLights),y.push(C.numSpotLights),y.push(C.numSpotLightMaps),y.push(C.numHemiLights),y.push(C.numRectAreaLights),y.push(C.numDirLightShadows),y.push(C.numPointLightShadows),y.push(C.numSpotLightShadows),y.push(C.numSpotLightShadowsWithMaps),y.push(C.numLightProbes),y.push(C.shadowMapType),y.push(C.toneMapping),y.push(C.numClippingPlanes),y.push(C.numClipIntersection),y.push(C.depthPacking)}function v(y,C){a.disableAll(),C.supportsVertexTextures&&a.enable(0),C.instancing&&a.enable(1),C.instancingColor&&a.enable(2),C.instancingMorph&&a.enable(3),C.matcap&&a.enable(4),C.envMap&&a.enable(5),C.normalMapObjectSpace&&a.enable(6),C.normalMapTangentSpace&&a.enable(7),C.clearcoat&&a.enable(8),C.iridescence&&a.enable(9),C.alphaTest&&a.enable(10),C.vertexColors&&a.enable(11),C.vertexAlphas&&a.enable(12),C.vertexUv1s&&a.enable(13),C.vertexUv2s&&a.enable(14),C.vertexUv3s&&a.enable(15),C.vertexTangents&&a.enable(16),C.anisotropy&&a.enable(17),C.alphaHash&&a.enable(18),C.batching&&a.enable(19),C.dispersion&&a.enable(20),C.batchingColor&&a.enable(21),C.gradientMap&&a.enable(22),y.push(a.mask),a.disableAll(),C.fog&&a.enable(0),C.useFog&&a.enable(1),C.flatShading&&a.enable(2),C.logarithmicDepthBuffer&&a.enable(3),C.reversedDepthBuffer&&a.enable(4),C.skinning&&a.enable(5),C.morphTargets&&a.enable(6),C.morphNormals&&a.enable(7),C.morphColors&&a.enable(8),C.premultipliedAlpha&&a.enable(9),C.shadowMapEnabled&&a.enable(10),C.doubleSided&&a.enable(11),C.flipSided&&a.enable(12),C.useDepthPacking&&a.enable(13),C.dithering&&a.enable(14),C.transmission&&a.enable(15),C.sheen&&a.enable(16),C.opaque&&a.enable(17),C.pointsUvs&&a.enable(18),C.decodeVideoTexture&&a.enable(19),C.decodeVideoTextureEmissive&&a.enable(20),C.alphaToCoverage&&a.enable(21),y.push(a.mask)}function A(y){const C=x[y.type];let F;if(C){const I=Pi[C];F=OS.clone(I.uniforms)}else F=y.uniforms;return F}function S(y,C){let F;for(let I=0,O=u.length;I<O;I++){const z=u[I];if(z.cacheKey===C){F=z,++F.usedTimes;break}}return F===void 0&&(F=new dT(i,C,y,r),u.push(F)),F}function b(y){if(--y.usedTimes===0){const C=u.indexOf(y);u[C]=u[u.length-1],u.pop(),y.destroy()}}function M(y){l.remove(y)}function R(){l.dispose()}return{getParameters:g,getProgramCacheKey:m,getUniforms:A,acquireProgram:S,releaseProgram:b,releaseShaderCache:M,programs:u,dispose:R}}function xT(){let i=new WeakMap;function e(o){return i.has(o)}function t(o){let a=i.get(o);return a===void 0&&(a={},i.set(o,a)),a}function n(o){i.delete(o)}function s(o,a,l){i.get(o)[a]=l}function r(){i=new WeakMap}return{has:e,get:t,remove:n,update:s,dispose:r}}function _T(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.material.id!==e.material.id?i.material.id-e.material.id:i.z!==e.z?i.z-e.z:i.id-e.id}function Wp(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.z!==e.z?e.z-i.z:i.id-e.id}function Xp(){const i=[];let e=0;const t=[],n=[],s=[];function r(){e=0,t.length=0,n.length=0,s.length=0}function o(f,d,h,x,p,g){let m=i[e];return m===void 0?(m={id:f.id,object:f,geometry:d,material:h,groupOrder:x,renderOrder:f.renderOrder,z:p,group:g},i[e]=m):(m.id=f.id,m.object=f,m.geometry=d,m.material=h,m.groupOrder=x,m.renderOrder=f.renderOrder,m.z=p,m.group=g),e++,m}function a(f,d,h,x,p,g){const m=o(f,d,h,x,p,g);h.transmission>0?n.push(m):h.transparent===!0?s.push(m):t.push(m)}function l(f,d,h,x,p,g){const m=o(f,d,h,x,p,g);h.transmission>0?n.unshift(m):h.transparent===!0?s.unshift(m):t.unshift(m)}function c(f,d){t.length>1&&t.sort(f||_T),n.length>1&&n.sort(d||Wp),s.length>1&&s.sort(d||Wp)}function u(){for(let f=e,d=i.length;f<d;f++){const h=i[f];if(h.id===null)break;h.id=null,h.object=null,h.geometry=null,h.material=null,h.group=null}}return{opaque:t,transmissive:n,transparent:s,init:r,push:a,unshift:l,finish:u,sort:c}}function vT(){let i=new WeakMap;function e(n,s){const r=i.get(n);let o;return r===void 0?(o=new Xp,i.set(n,[o])):s>=r.length?(o=new Xp,r.push(o)):o=r[s],o}function t(){i=new WeakMap}return{get:e,dispose:t}}function AT(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new U,color:new lt};break;case"SpotLight":t={position:new U,direction:new U,color:new lt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new U,color:new lt,distance:0,decay:0};break;case"HemisphereLight":t={direction:new U,skyColor:new lt,groundColor:new lt};break;case"RectAreaLight":t={color:new lt,position:new U,halfWidth:new U,halfHeight:new U};break}return i[e.id]=t,t}}}function ST(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Xe};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Xe};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Xe,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[e.id]=t,t}}}let yT=0;function bT(i,e){return(e.castShadow?2:0)-(i.castShadow?2:0)+(e.map?1:0)-(i.map?1:0)}function MT(i){const e=new AT,t=ST(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)n.probe.push(new U);const s=new U,r=new $e,o=new $e;function a(c){let u=0,f=0,d=0;for(let y=0;y<9;y++)n.probe[y].set(0,0,0);let h=0,x=0,p=0,g=0,m=0,_=0,v=0,A=0,S=0,b=0,M=0;c.sort(bT);for(let y=0,C=c.length;y<C;y++){const F=c[y],I=F.color,O=F.intensity,z=F.distance,H=F.shadow&&F.shadow.map?F.shadow.map.texture:null;if(F.isAmbientLight)u+=I.r*O,f+=I.g*O,d+=I.b*O;else if(F.isLightProbe){for(let V=0;V<9;V++)n.probe[V].addScaledVector(F.sh.coefficients[V],O);M++}else if(F.isDirectionalLight){const V=e.get(F);if(V.color.copy(F.color).multiplyScalar(F.intensity),F.castShadow){const q=F.shadow,G=t.get(F);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,n.directionalShadow[h]=G,n.directionalShadowMap[h]=H,n.directionalShadowMatrix[h]=F.shadow.matrix,_++}n.directional[h]=V,h++}else if(F.isSpotLight){const V=e.get(F);V.position.setFromMatrixPosition(F.matrixWorld),V.color.copy(I).multiplyScalar(O),V.distance=z,V.coneCos=Math.cos(F.angle),V.penumbraCos=Math.cos(F.angle*(1-F.penumbra)),V.decay=F.decay,n.spot[p]=V;const q=F.shadow;if(F.map&&(n.spotLightMap[S]=F.map,S++,q.updateMatrices(F),F.castShadow&&b++),n.spotLightMatrix[p]=q.matrix,F.castShadow){const G=t.get(F);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,n.spotShadow[p]=G,n.spotShadowMap[p]=H,A++}p++}else if(F.isRectAreaLight){const V=e.get(F);V.color.copy(I).multiplyScalar(O),V.halfWidth.set(F.width*.5,0,0),V.halfHeight.set(0,F.height*.5,0),n.rectArea[g]=V,g++}else if(F.isPointLight){const V=e.get(F);if(V.color.copy(F.color).multiplyScalar(F.intensity),V.distance=F.distance,V.decay=F.decay,F.castShadow){const q=F.shadow,G=t.get(F);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,G.shadowCameraNear=q.camera.near,G.shadowCameraFar=q.camera.far,n.pointShadow[x]=G,n.pointShadowMap[x]=H,n.pointShadowMatrix[x]=F.shadow.matrix,v++}n.point[x]=V,x++}else if(F.isHemisphereLight){const V=e.get(F);V.skyColor.copy(F.color).multiplyScalar(O),V.groundColor.copy(F.groundColor).multiplyScalar(O),n.hemi[m]=V,m++}}g>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=Ue.LTC_FLOAT_1,n.rectAreaLTC2=Ue.LTC_FLOAT_2):(n.rectAreaLTC1=Ue.LTC_HALF_1,n.rectAreaLTC2=Ue.LTC_HALF_2)),n.ambient[0]=u,n.ambient[1]=f,n.ambient[2]=d;const R=n.hash;(R.directionalLength!==h||R.pointLength!==x||R.spotLength!==p||R.rectAreaLength!==g||R.hemiLength!==m||R.numDirectionalShadows!==_||R.numPointShadows!==v||R.numSpotShadows!==A||R.numSpotMaps!==S||R.numLightProbes!==M)&&(n.directional.length=h,n.spot.length=p,n.rectArea.length=g,n.point.length=x,n.hemi.length=m,n.directionalShadow.length=_,n.directionalShadowMap.length=_,n.pointShadow.length=v,n.pointShadowMap.length=v,n.spotShadow.length=A,n.spotShadowMap.length=A,n.directionalShadowMatrix.length=_,n.pointShadowMatrix.length=v,n.spotLightMatrix.length=A+S-b,n.spotLightMap.length=S,n.numSpotLightShadowsWithMaps=b,n.numLightProbes=M,R.directionalLength=h,R.pointLength=x,R.spotLength=p,R.rectAreaLength=g,R.hemiLength=m,R.numDirectionalShadows=_,R.numPointShadows=v,R.numSpotShadows=A,R.numSpotMaps=S,R.numLightProbes=M,n.version=yT++)}function l(c,u){let f=0,d=0,h=0,x=0,p=0;const g=u.matrixWorldInverse;for(let m=0,_=c.length;m<_;m++){const v=c[m];if(v.isDirectionalLight){const A=n.directional[f];A.direction.setFromMatrixPosition(v.matrixWorld),s.setFromMatrixPosition(v.target.matrixWorld),A.direction.sub(s),A.direction.transformDirection(g),f++}else if(v.isSpotLight){const A=n.spot[h];A.position.setFromMatrixPosition(v.matrixWorld),A.position.applyMatrix4(g),A.direction.setFromMatrixPosition(v.matrixWorld),s.setFromMatrixPosition(v.target.matrixWorld),A.direction.sub(s),A.direction.transformDirection(g),h++}else if(v.isRectAreaLight){const A=n.rectArea[x];A.position.setFromMatrixPosition(v.matrixWorld),A.position.applyMatrix4(g),o.identity(),r.copy(v.matrixWorld),r.premultiply(g),o.extractRotation(r),A.halfWidth.set(v.width*.5,0,0),A.halfHeight.set(0,v.height*.5,0),A.halfWidth.applyMatrix4(o),A.halfHeight.applyMatrix4(o),x++}else if(v.isPointLight){const A=n.point[d];A.position.setFromMatrixPosition(v.matrixWorld),A.position.applyMatrix4(g),d++}else if(v.isHemisphereLight){const A=n.hemi[p];A.direction.setFromMatrixPosition(v.matrixWorld),A.direction.transformDirection(g),p++}}}return{setup:a,setupView:l,state:n}}function qp(i){const e=new MT(i),t=[],n=[];function s(u){c.camera=u,t.length=0,n.length=0}function r(u){t.push(u)}function o(u){n.push(u)}function a(){e.setup(t)}function l(u){e.setupView(t,u)}const c={lightsArray:t,shadowsArray:n,camera:null,lights:e,transmissionRenderTarget:{}};return{init:s,state:c,setupLights:a,setupLightsView:l,pushLight:r,pushShadow:o}}function CT(i){let e=new WeakMap;function t(s,r=0){const o=e.get(s);let a;return o===void 0?(a=new qp(i),e.set(s,[a])):r>=o.length?(a=new qp(i),o.push(a)):a=o[r],a}function n(){e=new WeakMap}return{get:t,dispose:n}}const TT=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,ET=`uniform sampler2D shadow_pass;
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
}`;function wT(i,e,t){let n=new _g;const s=new Xe,r=new Xe,o=new Ot,a=new ZS({depthPacking:HA}),l=new JS,c={},u=t.maxTextureSize,f={[Vi]:kn,[kn]:Vi,[ui]:ui},d=new Ln({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Xe},radius:{value:4}},vertexShader:TT,fragmentShader:ET}),h=d.clone();h.defines.HORIZONTAL_PASS=1;const x=new Bn;x.setAttribute("position",new xi(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const p=new Qt(x,d),g=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=$0;let m=this.type;this.render=function(b,M,R){if(g.enabled===!1||g.autoUpdate===!1&&g.needsUpdate===!1||b.length===0)return;const y=i.getRenderTarget(),C=i.getActiveCubeFace(),F=i.getActiveMipmapLevel(),I=i.state;I.setBlending(ds),I.buffers.depth.getReversed()===!0?I.buffers.color.setClear(0,0,0,0):I.buffers.color.setClear(1,1,1,1),I.buffers.depth.setTest(!0),I.setScissorTest(!1);const O=m!==ts&&this.type===ts,z=m===ts&&this.type!==ts;for(let H=0,V=b.length;H<V;H++){const q=b[H],G=q.shadow;if(G===void 0){et("WebGLShadowMap:",q,"has no shadow.");continue}if(G.autoUpdate===!1&&G.needsUpdate===!1)continue;s.copy(G.mapSize);const Y=G.getFrameExtents();if(s.multiply(Y),r.copy(G.mapSize),(s.x>u||s.y>u)&&(s.x>u&&(r.x=Math.floor(u/Y.x),s.x=r.x*Y.x,G.mapSize.x=r.x),s.y>u&&(r.y=Math.floor(u/Y.y),s.y=r.y*Y.y,G.mapSize.y=r.y)),G.map===null||O===!0||z===!0){const Ae=this.type!==ts?{minFilter:ti,magFilter:ti}:{};G.map!==null&&G.map.dispose(),G.map=new qs(s.x,s.y,Ae),G.map.texture.name=q.name+".shadowMap",G.camera.updateProjectionMatrix()}i.setRenderTarget(G.map),i.clear();const ae=G.getViewportCount();for(let Ae=0;Ae<ae;Ae++){const Pe=G.getViewport(Ae);o.set(r.x*Pe.x,r.y*Pe.y,r.x*Pe.z,r.y*Pe.w),I.viewport(o),G.updateMatrices(q,Ae),n=G.getFrustum(),A(M,R,G.camera,q,this.type)}G.isPointLightShadow!==!0&&this.type===ts&&_(G,R),G.needsUpdate=!1}m=this.type,g.needsUpdate=!1,i.setRenderTarget(y,C,F)};function _(b,M){const R=e.update(p);d.defines.VSM_SAMPLES!==b.blurSamples&&(d.defines.VSM_SAMPLES=b.blurSamples,h.defines.VSM_SAMPLES=b.blurSamples,d.needsUpdate=!0,h.needsUpdate=!0),b.mapPass===null&&(b.mapPass=new qs(s.x,s.y)),d.uniforms.shadow_pass.value=b.map.texture,d.uniforms.resolution.value=b.mapSize,d.uniforms.radius.value=b.radius,i.setRenderTarget(b.mapPass),i.clear(),i.renderBufferDirect(M,null,R,d,p,null),h.uniforms.shadow_pass.value=b.mapPass.texture,h.uniforms.resolution.value=b.mapSize,h.uniforms.radius.value=b.radius,i.setRenderTarget(b.map),i.clear(),i.renderBufferDirect(M,null,R,h,p,null)}function v(b,M,R,y){let C=null;const F=R.isPointLight===!0?b.customDistanceMaterial:b.customDepthMaterial;if(F!==void 0)C=F;else if(C=R.isPointLight===!0?l:a,i.localClippingEnabled&&M.clipShadows===!0&&Array.isArray(M.clippingPlanes)&&M.clippingPlanes.length!==0||M.displacementMap&&M.displacementScale!==0||M.alphaMap&&M.alphaTest>0||M.map&&M.alphaTest>0||M.alphaToCoverage===!0){const I=C.uuid,O=M.uuid;let z=c[I];z===void 0&&(z={},c[I]=z);let H=z[O];H===void 0&&(H=C.clone(),z[O]=H,M.addEventListener("dispose",S)),C=H}if(C.visible=M.visible,C.wireframe=M.wireframe,y===ts?C.side=M.shadowSide!==null?M.shadowSide:M.side:C.side=M.shadowSide!==null?M.shadowSide:f[M.side],C.alphaMap=M.alphaMap,C.alphaTest=M.alphaToCoverage===!0?.5:M.alphaTest,C.map=M.map,C.clipShadows=M.clipShadows,C.clippingPlanes=M.clippingPlanes,C.clipIntersection=M.clipIntersection,C.displacementMap=M.displacementMap,C.displacementScale=M.displacementScale,C.displacementBias=M.displacementBias,C.wireframeLinewidth=M.wireframeLinewidth,C.linewidth=M.linewidth,R.isPointLight===!0&&C.isMeshDistanceMaterial===!0){const I=i.properties.get(C);I.light=R}return C}function A(b,M,R,y,C){if(b.visible===!1)return;if(b.layers.test(M.layers)&&(b.isMesh||b.isLine||b.isPoints)&&(b.castShadow||b.receiveShadow&&C===ts)&&(!b.frustumCulled||n.intersectsObject(b))){b.modelViewMatrix.multiplyMatrices(R.matrixWorldInverse,b.matrixWorld);const O=e.update(b),z=b.material;if(Array.isArray(z)){const H=O.groups;for(let V=0,q=H.length;V<q;V++){const G=H[V],Y=z[G.materialIndex];if(Y&&Y.visible){const ae=v(b,Y,y,C);b.onBeforeShadow(i,b,M,R,O,ae,G),i.renderBufferDirect(R,null,O,ae,b,G),b.onAfterShadow(i,b,M,R,O,ae,G)}}}else if(z.visible){const H=v(b,z,y,C);b.onBeforeShadow(i,b,M,R,O,H,null),i.renderBufferDirect(R,null,O,H,b,null),b.onAfterShadow(i,b,M,R,O,H,null)}}const I=b.children;for(let O=0,z=I.length;O<z;O++)A(I[O],M,R,y,C)}function S(b){b.target.removeEventListener("dispose",S);for(const R in c){const y=c[R],C=b.target.uuid;C in y&&(y[C].dispose(),delete y[C])}}}const RT={[Zu]:Ju,[ef]:sf,[tf]:rf,[bo]:nf,[Ju]:Zu,[sf]:ef,[rf]:tf,[nf]:bo};function IT(i,e){function t(){let W=!1;const De=new Ot;let ye=null;const Me=new Ot(0,0,0,0);return{setMask:function(ve){ye!==ve&&!W&&(i.colorMask(ve,ve,ve,ve),ye=ve)},setLocked:function(ve){W=ve},setClear:function(ve,de,ke,je,ct){ct===!0&&(ve*=je,de*=je,ke*=je),De.set(ve,de,ke,je),Me.equals(De)===!1&&(i.clearColor(ve,de,ke,je),Me.copy(De))},reset:function(){W=!1,ye=null,Me.set(-1,0,0,0)}}}function n(){let W=!1,De=!1,ye=null,Me=null,ve=null;return{setReversed:function(de){if(De!==de){const ke=e.get("EXT_clip_control");de?ke.clipControlEXT(ke.LOWER_LEFT_EXT,ke.ZERO_TO_ONE_EXT):ke.clipControlEXT(ke.LOWER_LEFT_EXT,ke.NEGATIVE_ONE_TO_ONE_EXT),De=de;const je=ve;ve=null,this.setClear(je)}},getReversed:function(){return De},setTest:function(de){de?le(i.DEPTH_TEST):be(i.DEPTH_TEST)},setMask:function(de){ye!==de&&!W&&(i.depthMask(de),ye=de)},setFunc:function(de){if(De&&(de=RT[de]),Me!==de){switch(de){case Zu:i.depthFunc(i.NEVER);break;case Ju:i.depthFunc(i.ALWAYS);break;case ef:i.depthFunc(i.LESS);break;case bo:i.depthFunc(i.LEQUAL);break;case tf:i.depthFunc(i.EQUAL);break;case nf:i.depthFunc(i.GEQUAL);break;case sf:i.depthFunc(i.GREATER);break;case rf:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}Me=de}},setLocked:function(de){W=de},setClear:function(de){ve!==de&&(De&&(de=1-de),i.clearDepth(de),ve=de)},reset:function(){W=!1,ye=null,Me=null,ve=null,De=!1}}}function s(){let W=!1,De=null,ye=null,Me=null,ve=null,de=null,ke=null,je=null,ct=null;return{setTest:function(ht){W||(ht?le(i.STENCIL_TEST):be(i.STENCIL_TEST))},setMask:function(ht){De!==ht&&!W&&(i.stencilMask(ht),De=ht)},setFunc:function(ht,bn,Un){(ye!==ht||Me!==bn||ve!==Un)&&(i.stencilFunc(ht,bn,Un),ye=ht,Me=bn,ve=Un)},setOp:function(ht,bn,Un){(de!==ht||ke!==bn||je!==Un)&&(i.stencilOp(ht,bn,Un),de=ht,ke=bn,je=Un)},setLocked:function(ht){W=ht},setClear:function(ht){ct!==ht&&(i.clearStencil(ht),ct=ht)},reset:function(){W=!1,De=null,ye=null,Me=null,ve=null,de=null,ke=null,je=null,ct=null}}}const r=new t,o=new n,a=new s,l=new WeakMap,c=new WeakMap;let u={},f={},d=new WeakMap,h=[],x=null,p=!1,g=null,m=null,_=null,v=null,A=null,S=null,b=null,M=new lt(0,0,0),R=0,y=!1,C=null,F=null,I=null,O=null,z=null;const H=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let V=!1,q=0;const G=i.getParameter(i.VERSION);G.indexOf("WebGL")!==-1?(q=parseFloat(/^WebGL (\d)/.exec(G)[1]),V=q>=1):G.indexOf("OpenGL ES")!==-1&&(q=parseFloat(/^OpenGL ES (\d)/.exec(G)[1]),V=q>=2);let Y=null,ae={};const Ae=i.getParameter(i.SCISSOR_BOX),Pe=i.getParameter(i.VIEWPORT),Ne=new Ot().fromArray(Ae),We=new Ot().fromArray(Pe);function He(W,De,ye,Me){const ve=new Uint8Array(4),de=i.createTexture();i.bindTexture(W,de),i.texParameteri(W,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(W,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let ke=0;ke<ye;ke++)W===i.TEXTURE_3D||W===i.TEXTURE_2D_ARRAY?i.texImage3D(De,0,i.RGBA,1,1,Me,0,i.RGBA,i.UNSIGNED_BYTE,ve):i.texImage2D(De+ke,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,ve);return de}const J={};J[i.TEXTURE_2D]=He(i.TEXTURE_2D,i.TEXTURE_2D,1),J[i.TEXTURE_CUBE_MAP]=He(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),J[i.TEXTURE_2D_ARRAY]=He(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),J[i.TEXTURE_3D]=He(i.TEXTURE_3D,i.TEXTURE_3D,1,1),r.setClear(0,0,0,1),o.setClear(1),a.setClear(0),le(i.DEPTH_TEST),o.setFunc(bo),ie(!1),re(Qh),le(i.CULL_FACE),j(ds);function le(W){u[W]!==!0&&(i.enable(W),u[W]=!0)}function be(W){u[W]!==!1&&(i.disable(W),u[W]=!1)}function ze(W,De){return f[W]!==De?(i.bindFramebuffer(W,De),f[W]=De,W===i.DRAW_FRAMEBUFFER&&(f[i.FRAMEBUFFER]=De),W===i.FRAMEBUFFER&&(f[i.DRAW_FRAMEBUFFER]=De),!0):!1}function Ce(W,De){let ye=h,Me=!1;if(W){ye=d.get(De),ye===void 0&&(ye=[],d.set(De,ye));const ve=W.textures;if(ye.length!==ve.length||ye[0]!==i.COLOR_ATTACHMENT0){for(let de=0,ke=ve.length;de<ke;de++)ye[de]=i.COLOR_ATTACHMENT0+de;ye.length=ve.length,Me=!0}}else ye[0]!==i.BACK&&(ye[0]=i.BACK,Me=!0);Me&&i.drawBuffers(ye)}function Qe(W){return x!==W?(i.useProgram(W),x=W,!0):!1}const B={[gr]:i.FUNC_ADD,[hA]:i.FUNC_SUBTRACT,[pA]:i.FUNC_REVERSE_SUBTRACT};B[mA]=i.MIN,B[gA]=i.MAX;const N={[xA]:i.ZERO,[_A]:i.ONE,[vA]:i.SRC_COLOR,[Ta]:i.SRC_ALPHA,[CA]:i.SRC_ALPHA_SATURATE,[bA]:i.DST_COLOR,[SA]:i.DST_ALPHA,[AA]:i.ONE_MINUS_SRC_COLOR,[Ea]:i.ONE_MINUS_SRC_ALPHA,[MA]:i.ONE_MINUS_DST_COLOR,[yA]:i.ONE_MINUS_DST_ALPHA,[TA]:i.CONSTANT_COLOR,[EA]:i.ONE_MINUS_CONSTANT_COLOR,[wA]:i.CONSTANT_ALPHA,[RA]:i.ONE_MINUS_CONSTANT_ALPHA};function j(W,De,ye,Me,ve,de,ke,je,ct,ht){if(W===ds){p===!0&&(be(i.BLEND),p=!1);return}if(p===!1&&(le(i.BLEND),p=!0),W!==Z0){if(W!==g||ht!==y){if((m!==gr||A!==gr)&&(i.blendEquation(i.FUNC_ADD),m=gr,A=gr),ht)switch(W){case ks:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Kh:i.blendFunc(i.ONE,i.ONE);break;case jh:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case $h:i.blendFuncSeparate(i.DST_COLOR,i.ONE_MINUS_SRC_ALPHA,i.ZERO,i.ONE);break;default:Xt("WebGLState: Invalid blending: ",W);break}else switch(W){case ks:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Kh:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE,i.ONE,i.ONE);break;case jh:Xt("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case $h:Xt("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:Xt("WebGLState: Invalid blending: ",W);break}_=null,v=null,S=null,b=null,M.set(0,0,0),R=0,g=W,y=ht}return}ve=ve||De,de=de||ye,ke=ke||Me,(De!==m||ve!==A)&&(i.blendEquationSeparate(B[De],B[ve]),m=De,A=ve),(ye!==_||Me!==v||de!==S||ke!==b)&&(i.blendFuncSeparate(N[ye],N[Me],N[de],N[ke]),_=ye,v=Me,S=de,b=ke),(je.equals(M)===!1||ct!==R)&&(i.blendColor(je.r,je.g,je.b,ct),M.copy(je),R=ct),g=W,y=!1}function P(W,De){W.side===ui?be(i.CULL_FACE):le(i.CULL_FACE);let ye=W.side===kn;De&&(ye=!ye),ie(ye),W.blending===ks&&W.transparent===!1?j(ds):j(W.blending,W.blendEquation,W.blendSrc,W.blendDst,W.blendEquationAlpha,W.blendSrcAlpha,W.blendDstAlpha,W.blendColor,W.blendAlpha,W.premultipliedAlpha),o.setFunc(W.depthFunc),o.setTest(W.depthTest),o.setMask(W.depthWrite),r.setMask(W.colorWrite);const Me=W.stencilWrite;a.setTest(Me),Me&&(a.setMask(W.stencilWriteMask),a.setFunc(W.stencilFunc,W.stencilRef,W.stencilFuncMask),a.setOp(W.stencilFail,W.stencilZFail,W.stencilZPass)),ne(W.polygonOffset,W.polygonOffsetFactor,W.polygonOffsetUnits),W.alphaToCoverage===!0?le(i.SAMPLE_ALPHA_TO_COVERAGE):be(i.SAMPLE_ALPHA_TO_COVERAGE)}function ie(W){C!==W&&(W?i.frontFace(i.CW):i.frontFace(i.CCW),C=W)}function re(W){W!==uA?(le(i.CULL_FACE),W!==F&&(W===Qh?i.cullFace(i.BACK):W===fA?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):be(i.CULL_FACE),F=W}function fe(W){W!==I&&(V&&i.lineWidth(W),I=W)}function ne(W,De,ye){W?(le(i.POLYGON_OFFSET_FILL),(O!==De||z!==ye)&&(i.polygonOffset(De,ye),O=De,z=ye)):be(i.POLYGON_OFFSET_FILL)}function he(W){W?le(i.SCISSOR_TEST):be(i.SCISSOR_TEST)}function te(W){W===void 0&&(W=i.TEXTURE0+H-1),Y!==W&&(i.activeTexture(W),Y=W)}function ge(W,De,ye){ye===void 0&&(Y===null?ye=i.TEXTURE0+H-1:ye=Y);let Me=ae[ye];Me===void 0&&(Me={type:void 0,texture:void 0},ae[ye]=Me),(Me.type!==W||Me.texture!==De)&&(Y!==ye&&(i.activeTexture(ye),Y=ye),i.bindTexture(W,De||J[W]),Me.type=W,Me.texture=De)}function D(){const W=ae[Y];W!==void 0&&W.type!==void 0&&(i.bindTexture(W.type,null),W.type=void 0,W.texture=void 0)}function T(){try{i.compressedTexImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function X(){try{i.compressedTexImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function se(){try{i.texSubImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function ce(){try{i.texSubImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function Z(){try{i.compressedTexSubImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function Be(){try{i.compressedTexSubImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function Se(){try{i.texStorage2D(...arguments)}catch(W){W("WebGLState:",W)}}function Oe(){try{i.texStorage3D(...arguments)}catch(W){W("WebGLState:",W)}}function k(){try{i.texImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function ee(){try{i.texImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function me(W){Ne.equals(W)===!1&&(i.scissor(W.x,W.y,W.z,W.w),Ne.copy(W))}function Te(W){We.equals(W)===!1&&(i.viewport(W.x,W.y,W.z,W.w),We.copy(W))}function Fe(W,De){let ye=c.get(De);ye===void 0&&(ye=new WeakMap,c.set(De,ye));let Me=ye.get(W);Me===void 0&&(Me=i.getUniformBlockIndex(De,W.name),ye.set(W,Me))}function Re(W,De){const Me=c.get(De).get(W);l.get(De)!==Me&&(i.uniformBlockBinding(De,Me,W.__bindingPointIndex),l.set(De,Me))}function Ye(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),o.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),u={},Y=null,ae={},f={},d=new WeakMap,h=[],x=null,p=!1,g=null,m=null,_=null,v=null,A=null,S=null,b=null,M=new lt(0,0,0),R=0,y=!1,C=null,F=null,I=null,O=null,z=null,Ne.set(0,0,i.canvas.width,i.canvas.height),We.set(0,0,i.canvas.width,i.canvas.height),r.reset(),o.reset(),a.reset()}return{buffers:{color:r,depth:o,stencil:a},enable:le,disable:be,bindFramebuffer:ze,drawBuffers:Ce,useProgram:Qe,setBlending:j,setMaterial:P,setFlipSided:ie,setCullFace:re,setLineWidth:fe,setPolygonOffset:ne,setScissorTest:he,activeTexture:te,bindTexture:ge,unbindTexture:D,compressedTexImage2D:T,compressedTexImage3D:X,texImage2D:k,texImage3D:ee,updateUBOMapping:Fe,uniformBlockBinding:Re,texStorage2D:Se,texStorage3D:Oe,texSubImage2D:se,texSubImage3D:ce,compressedTexSubImage2D:Z,compressedTexSubImage3D:Be,scissor:me,viewport:Te,reset:Ye}}function DT(i,e,t,n,s,r,o){const a=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new Xe,u=new WeakMap;let f;const d=new WeakMap;let h=!1;try{h=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function x(D,T){return h?new OffscreenCanvas(D,T):tc("canvas")}function p(D,T,X){let se=1;const ce=ge(D);if((ce.width>X||ce.height>X)&&(se=X/Math.max(ce.width,ce.height)),se<1)if(typeof HTMLImageElement<"u"&&D instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&D instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&D instanceof ImageBitmap||typeof VideoFrame<"u"&&D instanceof VideoFrame){const Z=Math.floor(se*ce.width),Be=Math.floor(se*ce.height);f===void 0&&(f=x(Z,Be));const Se=T?x(Z,Be):f;return Se.width=Z,Se.height=Be,Se.getContext("2d").drawImage(D,0,0,Z,Be),et("WebGLRenderer: Texture has been resized from ("+ce.width+"x"+ce.height+") to ("+Z+"x"+Be+")."),Se}else return"data"in D&&et("WebGLRenderer: Image in DataTexture is too big ("+ce.width+"x"+ce.height+")."),D;return D}function g(D){return D.generateMipmaps}function m(D){i.generateMipmap(D)}function _(D){return D.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:D.isWebGL3DRenderTarget?i.TEXTURE_3D:D.isWebGLArrayRenderTarget||D.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function v(D,T,X,se,ce=!1){if(D!==null){if(i[D]!==void 0)return i[D];et("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+D+"'")}let Z=T;if(T===i.RED&&(X===i.FLOAT&&(Z=i.R32F),X===i.HALF_FLOAT&&(Z=i.R16F),X===i.UNSIGNED_BYTE&&(Z=i.R8)),T===i.RED_INTEGER&&(X===i.UNSIGNED_BYTE&&(Z=i.R8UI),X===i.UNSIGNED_SHORT&&(Z=i.R16UI),X===i.UNSIGNED_INT&&(Z=i.R32UI),X===i.BYTE&&(Z=i.R8I),X===i.SHORT&&(Z=i.R16I),X===i.INT&&(Z=i.R32I)),T===i.RG&&(X===i.FLOAT&&(Z=i.RG32F),X===i.HALF_FLOAT&&(Z=i.RG16F),X===i.UNSIGNED_BYTE&&(Z=i.RG8)),T===i.RG_INTEGER&&(X===i.UNSIGNED_BYTE&&(Z=i.RG8UI),X===i.UNSIGNED_SHORT&&(Z=i.RG16UI),X===i.UNSIGNED_INT&&(Z=i.RG32UI),X===i.BYTE&&(Z=i.RG8I),X===i.SHORT&&(Z=i.RG16I),X===i.INT&&(Z=i.RG32I)),T===i.RGB_INTEGER&&(X===i.UNSIGNED_BYTE&&(Z=i.RGB8UI),X===i.UNSIGNED_SHORT&&(Z=i.RGB16UI),X===i.UNSIGNED_INT&&(Z=i.RGB32UI),X===i.BYTE&&(Z=i.RGB8I),X===i.SHORT&&(Z=i.RGB16I),X===i.INT&&(Z=i.RGB32I)),T===i.RGBA_INTEGER&&(X===i.UNSIGNED_BYTE&&(Z=i.RGBA8UI),X===i.UNSIGNED_SHORT&&(Z=i.RGBA16UI),X===i.UNSIGNED_INT&&(Z=i.RGBA32UI),X===i.BYTE&&(Z=i.RGBA8I),X===i.SHORT&&(Z=i.RGBA16I),X===i.INT&&(Z=i.RGBA32I)),T===i.RGB&&(X===i.UNSIGNED_INT_5_9_9_9_REV&&(Z=i.RGB9_E5),X===i.UNSIGNED_INT_10F_11F_11F_REV&&(Z=i.R11F_G11F_B10F)),T===i.RGBA){const Be=ce?Jl:dt.getTransfer(se);X===i.FLOAT&&(Z=i.RGBA32F),X===i.HALF_FLOAT&&(Z=i.RGBA16F),X===i.UNSIGNED_BYTE&&(Z=Be===yt?i.SRGB8_ALPHA8:i.RGBA8),X===i.UNSIGNED_SHORT_4_4_4_4&&(Z=i.RGBA4),X===i.UNSIGNED_SHORT_5_5_5_1&&(Z=i.RGB5_A1)}return(Z===i.R16F||Z===i.R32F||Z===i.RG16F||Z===i.RG32F||Z===i.RGBA16F||Z===i.RGBA32F)&&e.get("EXT_color_buffer_float"),Z}function A(D,T){let X;return D?T===null||T===hi||T===Ra?X=i.DEPTH24_STENCIL8:T===bi?X=i.DEPTH32F_STENCIL8:T===wa&&(X=i.DEPTH24_STENCIL8,et("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):T===null||T===hi||T===Ra?X=i.DEPTH_COMPONENT24:T===bi?X=i.DEPTH_COMPONENT32F:T===wa&&(X=i.DEPTH_COMPONENT16),X}function S(D,T){return g(D)===!0||D.isFramebufferTexture&&D.minFilter!==ti&&D.minFilter!==di?Math.log2(Math.max(T.width,T.height))+1:D.mipmaps!==void 0&&D.mipmaps.length>0?D.mipmaps.length:D.isCompressedTexture&&Array.isArray(D.image)?T.mipmaps.length:1}function b(D){const T=D.target;T.removeEventListener("dispose",b),R(T),T.isVideoTexture&&u.delete(T)}function M(D){const T=D.target;T.removeEventListener("dispose",M),C(T)}function R(D){const T=n.get(D);if(T.__webglInit===void 0)return;const X=D.source,se=d.get(X);if(se){const ce=se[T.__cacheKey];ce.usedTimes--,ce.usedTimes===0&&y(D),Object.keys(se).length===0&&d.delete(X)}n.remove(D)}function y(D){const T=n.get(D);i.deleteTexture(T.__webglTexture);const X=D.source,se=d.get(X);delete se[T.__cacheKey],o.memory.textures--}function C(D){const T=n.get(D);if(D.depthTexture&&(D.depthTexture.dispose(),n.remove(D.depthTexture)),D.isWebGLCubeRenderTarget)for(let se=0;se<6;se++){if(Array.isArray(T.__webglFramebuffer[se]))for(let ce=0;ce<T.__webglFramebuffer[se].length;ce++)i.deleteFramebuffer(T.__webglFramebuffer[se][ce]);else i.deleteFramebuffer(T.__webglFramebuffer[se]);T.__webglDepthbuffer&&i.deleteRenderbuffer(T.__webglDepthbuffer[se])}else{if(Array.isArray(T.__webglFramebuffer))for(let se=0;se<T.__webglFramebuffer.length;se++)i.deleteFramebuffer(T.__webglFramebuffer[se]);else i.deleteFramebuffer(T.__webglFramebuffer);if(T.__webglDepthbuffer&&i.deleteRenderbuffer(T.__webglDepthbuffer),T.__webglMultisampledFramebuffer&&i.deleteFramebuffer(T.__webglMultisampledFramebuffer),T.__webglColorRenderbuffer)for(let se=0;se<T.__webglColorRenderbuffer.length;se++)T.__webglColorRenderbuffer[se]&&i.deleteRenderbuffer(T.__webglColorRenderbuffer[se]);T.__webglDepthRenderbuffer&&i.deleteRenderbuffer(T.__webglDepthRenderbuffer)}const X=D.textures;for(let se=0,ce=X.length;se<ce;se++){const Z=n.get(X[se]);Z.__webglTexture&&(i.deleteTexture(Z.__webglTexture),o.memory.textures--),n.remove(X[se])}n.remove(D)}let F=0;function I(){F=0}function O(){const D=F;return D>=s.maxTextures&&et("WebGLTextures: Trying to use "+D+" texture units while this GPU supports only "+s.maxTextures),F+=1,D}function z(D){const T=[];return T.push(D.wrapS),T.push(D.wrapT),T.push(D.wrapR||0),T.push(D.magFilter),T.push(D.minFilter),T.push(D.anisotropy),T.push(D.internalFormat),T.push(D.format),T.push(D.type),T.push(D.generateMipmaps),T.push(D.premultiplyAlpha),T.push(D.flipY),T.push(D.unpackAlignment),T.push(D.colorSpace),T.join()}function H(D,T){const X=n.get(D);if(D.isVideoTexture&&he(D),D.isRenderTargetTexture===!1&&D.isExternalTexture!==!0&&D.version>0&&X.__version!==D.version){const se=D.image;if(se===null)et("WebGLRenderer: Texture marked for update but no image data found.");else if(se.complete===!1)et("WebGLRenderer: Texture marked for update but image is incomplete");else{J(X,D,T);return}}else D.isExternalTexture&&(X.__webglTexture=D.sourceTexture?D.sourceTexture:null);t.bindTexture(i.TEXTURE_2D,X.__webglTexture,i.TEXTURE0+T)}function V(D,T){const X=n.get(D);if(D.isRenderTargetTexture===!1&&D.version>0&&X.__version!==D.version){J(X,D,T);return}else D.isExternalTexture&&(X.__webglTexture=D.sourceTexture?D.sourceTexture:null);t.bindTexture(i.TEXTURE_2D_ARRAY,X.__webglTexture,i.TEXTURE0+T)}function q(D,T){const X=n.get(D);if(D.isRenderTargetTexture===!1&&D.version>0&&X.__version!==D.version){J(X,D,T);return}t.bindTexture(i.TEXTURE_3D,X.__webglTexture,i.TEXTURE0+T)}function G(D,T){const X=n.get(D);if(D.version>0&&X.__version!==D.version){le(X,D,T);return}t.bindTexture(i.TEXTURE_CUBE_MAP,X.__webglTexture,i.TEXTURE0+T)}const Y={[lf]:i.REPEAT,[fs]:i.CLAMP_TO_EDGE,[cf]:i.MIRRORED_REPEAT},ae={[ti]:i.NEAREST,[zA]:i.NEAREST_MIPMAP_NEAREST,[tl]:i.NEAREST_MIPMAP_LINEAR,[di]:i.LINEAR,[Xc]:i.LINEAR_MIPMAP_NEAREST,[_r]:i.LINEAR_MIPMAP_LINEAR},Ae={[WA]:i.NEVER,[jA]:i.ALWAYS,[XA]:i.LESS,[lg]:i.LEQUAL,[qA]:i.EQUAL,[KA]:i.GEQUAL,[YA]:i.GREATER,[QA]:i.NOTEQUAL};function Pe(D,T){if(T.type===bi&&e.has("OES_texture_float_linear")===!1&&(T.magFilter===di||T.magFilter===Xc||T.magFilter===tl||T.magFilter===_r||T.minFilter===di||T.minFilter===Xc||T.minFilter===tl||T.minFilter===_r)&&et("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(D,i.TEXTURE_WRAP_S,Y[T.wrapS]),i.texParameteri(D,i.TEXTURE_WRAP_T,Y[T.wrapT]),(D===i.TEXTURE_3D||D===i.TEXTURE_2D_ARRAY)&&i.texParameteri(D,i.TEXTURE_WRAP_R,Y[T.wrapR]),i.texParameteri(D,i.TEXTURE_MAG_FILTER,ae[T.magFilter]),i.texParameteri(D,i.TEXTURE_MIN_FILTER,ae[T.minFilter]),T.compareFunction&&(i.texParameteri(D,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(D,i.TEXTURE_COMPARE_FUNC,Ae[T.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(T.magFilter===ti||T.minFilter!==tl&&T.minFilter!==_r||T.type===bi&&e.has("OES_texture_float_linear")===!1)return;if(T.anisotropy>1||n.get(T).__currentAnisotropy){const X=e.get("EXT_texture_filter_anisotropic");i.texParameterf(D,X.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(T.anisotropy,s.getMaxAnisotropy())),n.get(T).__currentAnisotropy=T.anisotropy}}}function Ne(D,T){let X=!1;D.__webglInit===void 0&&(D.__webglInit=!0,T.addEventListener("dispose",b));const se=T.source;let ce=d.get(se);ce===void 0&&(ce={},d.set(se,ce));const Z=z(T);if(Z!==D.__cacheKey){ce[Z]===void 0&&(ce[Z]={texture:i.createTexture(),usedTimes:0},o.memory.textures++,X=!0),ce[Z].usedTimes++;const Be=ce[D.__cacheKey];Be!==void 0&&(ce[D.__cacheKey].usedTimes--,Be.usedTimes===0&&y(T)),D.__cacheKey=Z,D.__webglTexture=ce[Z].texture}return X}function We(D,T,X){return Math.floor(Math.floor(D/X)/T)}function He(D,T,X,se){const Z=D.updateRanges;if(Z.length===0)t.texSubImage2D(i.TEXTURE_2D,0,0,0,T.width,T.height,X,se,T.data);else{Z.sort((ee,me)=>ee.start-me.start);let Be=0;for(let ee=1;ee<Z.length;ee++){const me=Z[Be],Te=Z[ee],Fe=me.start+me.count,Re=We(Te.start,T.width,4),Ye=We(me.start,T.width,4);Te.start<=Fe+1&&Re===Ye&&We(Te.start+Te.count-1,T.width,4)===Re?me.count=Math.max(me.count,Te.start+Te.count-me.start):(++Be,Z[Be]=Te)}Z.length=Be+1;const Se=i.getParameter(i.UNPACK_ROW_LENGTH),Oe=i.getParameter(i.UNPACK_SKIP_PIXELS),k=i.getParameter(i.UNPACK_SKIP_ROWS);i.pixelStorei(i.UNPACK_ROW_LENGTH,T.width);for(let ee=0,me=Z.length;ee<me;ee++){const Te=Z[ee],Fe=Math.floor(Te.start/4),Re=Math.ceil(Te.count/4),Ye=Fe%T.width,W=Math.floor(Fe/T.width),De=Re,ye=1;i.pixelStorei(i.UNPACK_SKIP_PIXELS,Ye),i.pixelStorei(i.UNPACK_SKIP_ROWS,W),t.texSubImage2D(i.TEXTURE_2D,0,Ye,W,De,ye,X,se,T.data)}D.clearUpdateRanges(),i.pixelStorei(i.UNPACK_ROW_LENGTH,Se),i.pixelStorei(i.UNPACK_SKIP_PIXELS,Oe),i.pixelStorei(i.UNPACK_SKIP_ROWS,k)}}function J(D,T,X){let se=i.TEXTURE_2D;(T.isDataArrayTexture||T.isCompressedArrayTexture)&&(se=i.TEXTURE_2D_ARRAY),T.isData3DTexture&&(se=i.TEXTURE_3D);const ce=Ne(D,T),Z=T.source;t.bindTexture(se,D.__webglTexture,i.TEXTURE0+X);const Be=n.get(Z);if(Z.version!==Be.__version||ce===!0){t.activeTexture(i.TEXTURE0+X);const Se=dt.getPrimaries(dt.workingColorSpace),Oe=T.colorSpace===Fs?null:dt.getPrimaries(T.colorSpace),k=T.colorSpace===Fs||Se===Oe?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,T.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,T.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,k);let ee=p(T.image,!1,s.maxTextureSize);ee=te(T,ee);const me=r.convert(T.format,T.colorSpace),Te=r.convert(T.type);let Fe=v(T.internalFormat,me,Te,T.colorSpace,T.isVideoTexture);Pe(se,T);let Re;const Ye=T.mipmaps,W=T.isVideoTexture!==!0,De=Be.__version===void 0||ce===!0,ye=Z.dataReady,Me=S(T,ee);if(T.isDepthTexture)Fe=A(T.format===Ia,T.type),De&&(W?t.texStorage2D(i.TEXTURE_2D,1,Fe,ee.width,ee.height):t.texImage2D(i.TEXTURE_2D,0,Fe,ee.width,ee.height,0,me,Te,null));else if(T.isDataTexture)if(Ye.length>0){W&&De&&t.texStorage2D(i.TEXTURE_2D,Me,Fe,Ye[0].width,Ye[0].height);for(let ve=0,de=Ye.length;ve<de;ve++)Re=Ye[ve],W?ye&&t.texSubImage2D(i.TEXTURE_2D,ve,0,0,Re.width,Re.height,me,Te,Re.data):t.texImage2D(i.TEXTURE_2D,ve,Fe,Re.width,Re.height,0,me,Te,Re.data);T.generateMipmaps=!1}else W?(De&&t.texStorage2D(i.TEXTURE_2D,Me,Fe,ee.width,ee.height),ye&&He(T,ee,me,Te)):t.texImage2D(i.TEXTURE_2D,0,Fe,ee.width,ee.height,0,me,Te,ee.data);else if(T.isCompressedTexture)if(T.isCompressedArrayTexture){W&&De&&t.texStorage3D(i.TEXTURE_2D_ARRAY,Me,Fe,Ye[0].width,Ye[0].height,ee.depth);for(let ve=0,de=Ye.length;ve<de;ve++)if(Re=Ye[ve],T.format!==Pn)if(me!==null)if(W){if(ye)if(T.layerUpdates.size>0){const ke=bp(Re.width,Re.height,T.format,T.type);for(const je of T.layerUpdates){const ct=Re.data.subarray(je*ke/Re.data.BYTES_PER_ELEMENT,(je+1)*ke/Re.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,ve,0,0,je,Re.width,Re.height,1,me,ct)}T.clearLayerUpdates()}else t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,ve,0,0,0,Re.width,Re.height,ee.depth,me,Re.data)}else t.compressedTexImage3D(i.TEXTURE_2D_ARRAY,ve,Fe,Re.width,Re.height,ee.depth,0,Re.data,0,0);else et("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else W?ye&&t.texSubImage3D(i.TEXTURE_2D_ARRAY,ve,0,0,0,Re.width,Re.height,ee.depth,me,Te,Re.data):t.texImage3D(i.TEXTURE_2D_ARRAY,ve,Fe,Re.width,Re.height,ee.depth,0,me,Te,Re.data)}else{W&&De&&t.texStorage2D(i.TEXTURE_2D,Me,Fe,Ye[0].width,Ye[0].height);for(let ve=0,de=Ye.length;ve<de;ve++)Re=Ye[ve],T.format!==Pn?me!==null?W?ye&&t.compressedTexSubImage2D(i.TEXTURE_2D,ve,0,0,Re.width,Re.height,me,Re.data):t.compressedTexImage2D(i.TEXTURE_2D,ve,Fe,Re.width,Re.height,0,Re.data):et("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):W?ye&&t.texSubImage2D(i.TEXTURE_2D,ve,0,0,Re.width,Re.height,me,Te,Re.data):t.texImage2D(i.TEXTURE_2D,ve,Fe,Re.width,Re.height,0,me,Te,Re.data)}else if(T.isDataArrayTexture)if(W){if(De&&t.texStorage3D(i.TEXTURE_2D_ARRAY,Me,Fe,ee.width,ee.height,ee.depth),ye)if(T.layerUpdates.size>0){const ve=bp(ee.width,ee.height,T.format,T.type);for(const de of T.layerUpdates){const ke=ee.data.subarray(de*ve/ee.data.BYTES_PER_ELEMENT,(de+1)*ve/ee.data.BYTES_PER_ELEMENT);t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,de,ee.width,ee.height,1,me,Te,ke)}T.clearLayerUpdates()}else t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,ee.width,ee.height,ee.depth,me,Te,ee.data)}else t.texImage3D(i.TEXTURE_2D_ARRAY,0,Fe,ee.width,ee.height,ee.depth,0,me,Te,ee.data);else if(T.isData3DTexture)W?(De&&t.texStorage3D(i.TEXTURE_3D,Me,Fe,ee.width,ee.height,ee.depth),ye&&t.texSubImage3D(i.TEXTURE_3D,0,0,0,0,ee.width,ee.height,ee.depth,me,Te,ee.data)):t.texImage3D(i.TEXTURE_3D,0,Fe,ee.width,ee.height,ee.depth,0,me,Te,ee.data);else if(T.isFramebufferTexture){if(De)if(W)t.texStorage2D(i.TEXTURE_2D,Me,Fe,ee.width,ee.height);else{let ve=ee.width,de=ee.height;for(let ke=0;ke<Me;ke++)t.texImage2D(i.TEXTURE_2D,ke,Fe,ve,de,0,me,Te,null),ve>>=1,de>>=1}}else if(Ye.length>0){if(W&&De){const ve=ge(Ye[0]);t.texStorage2D(i.TEXTURE_2D,Me,Fe,ve.width,ve.height)}for(let ve=0,de=Ye.length;ve<de;ve++)Re=Ye[ve],W?ye&&t.texSubImage2D(i.TEXTURE_2D,ve,0,0,me,Te,Re):t.texImage2D(i.TEXTURE_2D,ve,Fe,me,Te,Re);T.generateMipmaps=!1}else if(W){if(De){const ve=ge(ee);t.texStorage2D(i.TEXTURE_2D,Me,Fe,ve.width,ve.height)}ye&&t.texSubImage2D(i.TEXTURE_2D,0,0,0,me,Te,ee)}else t.texImage2D(i.TEXTURE_2D,0,Fe,me,Te,ee);g(T)&&m(se),Be.__version=Z.version,T.onUpdate&&T.onUpdate(T)}D.__version=T.version}function le(D,T,X){if(T.image.length!==6)return;const se=Ne(D,T),ce=T.source;t.bindTexture(i.TEXTURE_CUBE_MAP,D.__webglTexture,i.TEXTURE0+X);const Z=n.get(ce);if(ce.version!==Z.__version||se===!0){t.activeTexture(i.TEXTURE0+X);const Be=dt.getPrimaries(dt.workingColorSpace),Se=T.colorSpace===Fs?null:dt.getPrimaries(T.colorSpace),Oe=T.colorSpace===Fs||Be===Se?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,T.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,T.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,Oe);const k=T.isCompressedTexture||T.image[0].isCompressedTexture,ee=T.image[0]&&T.image[0].isDataTexture,me=[];for(let de=0;de<6;de++)!k&&!ee?me[de]=p(T.image[de],!0,s.maxCubemapSize):me[de]=ee?T.image[de].image:T.image[de],me[de]=te(T,me[de]);const Te=me[0],Fe=r.convert(T.format,T.colorSpace),Re=r.convert(T.type),Ye=v(T.internalFormat,Fe,Re,T.colorSpace),W=T.isVideoTexture!==!0,De=Z.__version===void 0||se===!0,ye=ce.dataReady;let Me=S(T,Te);Pe(i.TEXTURE_CUBE_MAP,T);let ve;if(k){W&&De&&t.texStorage2D(i.TEXTURE_CUBE_MAP,Me,Ye,Te.width,Te.height);for(let de=0;de<6;de++){ve=me[de].mipmaps;for(let ke=0;ke<ve.length;ke++){const je=ve[ke];T.format!==Pn?Fe!==null?W?ye&&t.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,ke,0,0,je.width,je.height,Fe,je.data):t.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,ke,Ye,je.width,je.height,0,je.data):et("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):W?ye&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,ke,0,0,je.width,je.height,Fe,Re,je.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,ke,Ye,je.width,je.height,0,Fe,Re,je.data)}}}else{if(ve=T.mipmaps,W&&De){ve.length>0&&Me++;const de=ge(me[0]);t.texStorage2D(i.TEXTURE_CUBE_MAP,Me,Ye,de.width,de.height)}for(let de=0;de<6;de++)if(ee){W?ye&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,0,0,me[de].width,me[de].height,Fe,Re,me[de].data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,Ye,me[de].width,me[de].height,0,Fe,Re,me[de].data);for(let ke=0;ke<ve.length;ke++){const ct=ve[ke].image[de].image;W?ye&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,ke+1,0,0,ct.width,ct.height,Fe,Re,ct.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,ke+1,Ye,ct.width,ct.height,0,Fe,Re,ct.data)}}else{W?ye&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,0,0,Fe,Re,me[de]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,0,Ye,Fe,Re,me[de]);for(let ke=0;ke<ve.length;ke++){const je=ve[ke];W?ye&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,ke+1,0,0,Fe,Re,je.image[de]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+de,ke+1,Ye,Fe,Re,je.image[de])}}}g(T)&&m(i.TEXTURE_CUBE_MAP),Z.__version=ce.version,T.onUpdate&&T.onUpdate(T)}D.__version=T.version}function be(D,T,X,se,ce,Z){const Be=r.convert(X.format,X.colorSpace),Se=r.convert(X.type),Oe=v(X.internalFormat,Be,Se,X.colorSpace),k=n.get(T),ee=n.get(X);if(ee.__renderTarget=T,!k.__hasExternalTextures){const me=Math.max(1,T.width>>Z),Te=Math.max(1,T.height>>Z);ce===i.TEXTURE_3D||ce===i.TEXTURE_2D_ARRAY?t.texImage3D(ce,Z,Oe,me,Te,T.depth,0,Be,Se,null):t.texImage2D(ce,Z,Oe,me,Te,0,Be,Se,null)}t.bindFramebuffer(i.FRAMEBUFFER,D),ne(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,se,ce,ee.__webglTexture,0,fe(T)):(ce===i.TEXTURE_2D||ce>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&ce<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,se,ce,ee.__webglTexture,Z),t.bindFramebuffer(i.FRAMEBUFFER,null)}function ze(D,T,X){if(i.bindRenderbuffer(i.RENDERBUFFER,D),T.depthBuffer){const se=T.depthTexture,ce=se&&se.isDepthTexture?se.type:null,Z=A(T.stencilBuffer,ce),Be=T.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Se=fe(T);ne(T)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,Se,Z,T.width,T.height):X?i.renderbufferStorageMultisample(i.RENDERBUFFER,Se,Z,T.width,T.height):i.renderbufferStorage(i.RENDERBUFFER,Z,T.width,T.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,Be,i.RENDERBUFFER,D)}else{const se=T.textures;for(let ce=0;ce<se.length;ce++){const Z=se[ce],Be=r.convert(Z.format,Z.colorSpace),Se=r.convert(Z.type),Oe=v(Z.internalFormat,Be,Se,Z.colorSpace),k=fe(T);X&&ne(T)===!1?i.renderbufferStorageMultisample(i.RENDERBUFFER,k,Oe,T.width,T.height):ne(T)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,k,Oe,T.width,T.height):i.renderbufferStorage(i.RENDERBUFFER,Oe,T.width,T.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function Ce(D,T){if(T&&T.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(t.bindFramebuffer(i.FRAMEBUFFER,D),!(T.depthTexture&&T.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const se=n.get(T.depthTexture);se.__renderTarget=T,(!se.__webglTexture||T.depthTexture.image.width!==T.width||T.depthTexture.image.height!==T.height)&&(T.depthTexture.image.width=T.width,T.depthTexture.image.height=T.height,T.depthTexture.needsUpdate=!0),H(T.depthTexture,0);const ce=se.__webglTexture,Z=fe(T);if(T.depthTexture.format===To)ne(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,ce,0,Z):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,ce,0);else if(T.depthTexture.format===Ia)ne(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,ce,0,Z):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,ce,0);else throw new Error("Unknown depthTexture format")}function Qe(D){const T=n.get(D),X=D.isWebGLCubeRenderTarget===!0;if(T.__boundDepthTexture!==D.depthTexture){const se=D.depthTexture;if(T.__depthDisposeCallback&&T.__depthDisposeCallback(),se){const ce=()=>{delete T.__boundDepthTexture,delete T.__depthDisposeCallback,se.removeEventListener("dispose",ce)};se.addEventListener("dispose",ce),T.__depthDisposeCallback=ce}T.__boundDepthTexture=se}if(D.depthTexture&&!T.__autoAllocateDepthBuffer){if(X)throw new Error("target.depthTexture not supported in Cube render targets");const se=D.texture.mipmaps;se&&se.length>0?Ce(T.__webglFramebuffer[0],D):Ce(T.__webglFramebuffer,D)}else if(X){T.__webglDepthbuffer=[];for(let se=0;se<6;se++)if(t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer[se]),T.__webglDepthbuffer[se]===void 0)T.__webglDepthbuffer[se]=i.createRenderbuffer(),ze(T.__webglDepthbuffer[se],D,!1);else{const ce=D.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Z=T.__webglDepthbuffer[se];i.bindRenderbuffer(i.RENDERBUFFER,Z),i.framebufferRenderbuffer(i.FRAMEBUFFER,ce,i.RENDERBUFFER,Z)}}else{const se=D.texture.mipmaps;if(se&&se.length>0?t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer[0]):t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer),T.__webglDepthbuffer===void 0)T.__webglDepthbuffer=i.createRenderbuffer(),ze(T.__webglDepthbuffer,D,!1);else{const ce=D.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Z=T.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,Z),i.framebufferRenderbuffer(i.FRAMEBUFFER,ce,i.RENDERBUFFER,Z)}}t.bindFramebuffer(i.FRAMEBUFFER,null)}function B(D,T,X){const se=n.get(D);T!==void 0&&be(se.__webglFramebuffer,D,D.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),X!==void 0&&Qe(D)}function N(D){const T=D.texture,X=n.get(D),se=n.get(T);D.addEventListener("dispose",M);const ce=D.textures,Z=D.isWebGLCubeRenderTarget===!0,Be=ce.length>1;if(Be||(se.__webglTexture===void 0&&(se.__webglTexture=i.createTexture()),se.__version=T.version,o.memory.textures++),Z){X.__webglFramebuffer=[];for(let Se=0;Se<6;Se++)if(T.mipmaps&&T.mipmaps.length>0){X.__webglFramebuffer[Se]=[];for(let Oe=0;Oe<T.mipmaps.length;Oe++)X.__webglFramebuffer[Se][Oe]=i.createFramebuffer()}else X.__webglFramebuffer[Se]=i.createFramebuffer()}else{if(T.mipmaps&&T.mipmaps.length>0){X.__webglFramebuffer=[];for(let Se=0;Se<T.mipmaps.length;Se++)X.__webglFramebuffer[Se]=i.createFramebuffer()}else X.__webglFramebuffer=i.createFramebuffer();if(Be)for(let Se=0,Oe=ce.length;Se<Oe;Se++){const k=n.get(ce[Se]);k.__webglTexture===void 0&&(k.__webglTexture=i.createTexture(),o.memory.textures++)}if(D.samples>0&&ne(D)===!1){X.__webglMultisampledFramebuffer=i.createFramebuffer(),X.__webglColorRenderbuffer=[],t.bindFramebuffer(i.FRAMEBUFFER,X.__webglMultisampledFramebuffer);for(let Se=0;Se<ce.length;Se++){const Oe=ce[Se];X.__webglColorRenderbuffer[Se]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,X.__webglColorRenderbuffer[Se]);const k=r.convert(Oe.format,Oe.colorSpace),ee=r.convert(Oe.type),me=v(Oe.internalFormat,k,ee,Oe.colorSpace,D.isXRRenderTarget===!0),Te=fe(D);i.renderbufferStorageMultisample(i.RENDERBUFFER,Te,me,D.width,D.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+Se,i.RENDERBUFFER,X.__webglColorRenderbuffer[Se])}i.bindRenderbuffer(i.RENDERBUFFER,null),D.depthBuffer&&(X.__webglDepthRenderbuffer=i.createRenderbuffer(),ze(X.__webglDepthRenderbuffer,D,!0)),t.bindFramebuffer(i.FRAMEBUFFER,null)}}if(Z){t.bindTexture(i.TEXTURE_CUBE_MAP,se.__webglTexture),Pe(i.TEXTURE_CUBE_MAP,T);for(let Se=0;Se<6;Se++)if(T.mipmaps&&T.mipmaps.length>0)for(let Oe=0;Oe<T.mipmaps.length;Oe++)be(X.__webglFramebuffer[Se][Oe],D,T,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+Se,Oe);else be(X.__webglFramebuffer[Se],D,T,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+Se,0);g(T)&&m(i.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(Be){for(let Se=0,Oe=ce.length;Se<Oe;Se++){const k=ce[Se],ee=n.get(k);let me=i.TEXTURE_2D;(D.isWebGL3DRenderTarget||D.isWebGLArrayRenderTarget)&&(me=D.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(me,ee.__webglTexture),Pe(me,k),be(X.__webglFramebuffer,D,k,i.COLOR_ATTACHMENT0+Se,me,0),g(k)&&m(me)}t.unbindTexture()}else{let Se=i.TEXTURE_2D;if((D.isWebGL3DRenderTarget||D.isWebGLArrayRenderTarget)&&(Se=D.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(Se,se.__webglTexture),Pe(Se,T),T.mipmaps&&T.mipmaps.length>0)for(let Oe=0;Oe<T.mipmaps.length;Oe++)be(X.__webglFramebuffer[Oe],D,T,i.COLOR_ATTACHMENT0,Se,Oe);else be(X.__webglFramebuffer,D,T,i.COLOR_ATTACHMENT0,Se,0);g(T)&&m(Se),t.unbindTexture()}D.depthBuffer&&Qe(D)}function j(D){const T=D.textures;for(let X=0,se=T.length;X<se;X++){const ce=T[X];if(g(ce)){const Z=_(D),Be=n.get(ce).__webglTexture;t.bindTexture(Z,Be),m(Z),t.unbindTexture()}}}const P=[],ie=[];function re(D){if(D.samples>0){if(ne(D)===!1){const T=D.textures,X=D.width,se=D.height;let ce=i.COLOR_BUFFER_BIT;const Z=D.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Be=n.get(D),Se=T.length>1;if(Se)for(let k=0;k<T.length;k++)t.bindFramebuffer(i.FRAMEBUFFER,Be.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.RENDERBUFFER,null),t.bindFramebuffer(i.FRAMEBUFFER,Be.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.TEXTURE_2D,null,0);t.bindFramebuffer(i.READ_FRAMEBUFFER,Be.__webglMultisampledFramebuffer);const Oe=D.texture.mipmaps;Oe&&Oe.length>0?t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Be.__webglFramebuffer[0]):t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Be.__webglFramebuffer);for(let k=0;k<T.length;k++){if(D.resolveDepthBuffer&&(D.depthBuffer&&(ce|=i.DEPTH_BUFFER_BIT),D.stencilBuffer&&D.resolveStencilBuffer&&(ce|=i.STENCIL_BUFFER_BIT)),Se){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,Be.__webglColorRenderbuffer[k]);const ee=n.get(T[k]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,ee,0)}i.blitFramebuffer(0,0,X,se,0,0,X,se,ce,i.NEAREST),l===!0&&(P.length=0,ie.length=0,P.push(i.COLOR_ATTACHMENT0+k),D.depthBuffer&&D.resolveDepthBuffer===!1&&(P.push(Z),ie.push(Z),i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,ie)),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,P))}if(t.bindFramebuffer(i.READ_FRAMEBUFFER,null),t.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),Se)for(let k=0;k<T.length;k++){t.bindFramebuffer(i.FRAMEBUFFER,Be.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.RENDERBUFFER,Be.__webglColorRenderbuffer[k]);const ee=n.get(T[k]).__webglTexture;t.bindFramebuffer(i.FRAMEBUFFER,Be.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.TEXTURE_2D,ee,0)}t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Be.__webglMultisampledFramebuffer)}else if(D.depthBuffer&&D.resolveDepthBuffer===!1&&l){const T=D.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[T])}}}function fe(D){return Math.min(s.maxSamples,D.samples)}function ne(D){const T=n.get(D);return D.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&T.__useRenderToTexture!==!1}function he(D){const T=o.render.frame;u.get(D)!==T&&(u.set(D,T),D.update())}function te(D,T){const X=D.colorSpace,se=D.format,ce=D.type;return D.isCompressedTexture===!0||D.isVideoTexture===!0||X!==Eo&&X!==Fs&&(dt.getTransfer(X)===yt?(se!==Pn||ce!==Gi)&&et("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):Xt("WebGLTextures: Unsupported texture color space:",X)),T}function ge(D){return typeof HTMLImageElement<"u"&&D instanceof HTMLImageElement?(c.width=D.naturalWidth||D.width,c.height=D.naturalHeight||D.height):typeof VideoFrame<"u"&&D instanceof VideoFrame?(c.width=D.displayWidth,c.height=D.displayHeight):(c.width=D.width,c.height=D.height),c}this.allocateTextureUnit=O,this.resetTextureUnits=I,this.setTexture2D=H,this.setTexture2DArray=V,this.setTexture3D=q,this.setTextureCube=G,this.rebindTextures=B,this.setupRenderTarget=N,this.updateRenderTargetMipmap=j,this.updateMultisampleRenderTarget=re,this.setupDepthRenderbuffer=Qe,this.setupFrameBufferTexture=be,this.useMultisampledRTT=ne}function Cg(i,e){function t(n,s=Fs){let r;const o=dt.getTransfer(s);if(n===Gi)return i.UNSIGNED_BYTE;if(n===bd)return i.UNSIGNED_SHORT_4_4_4_4;if(n===Md)return i.UNSIGNED_SHORT_5_5_5_1;if(n===ig)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===sg)return i.UNSIGNED_INT_10F_11F_11F_REV;if(n===tg)return i.BYTE;if(n===ng)return i.SHORT;if(n===wa)return i.UNSIGNED_SHORT;if(n===yd)return i.INT;if(n===hi)return i.UNSIGNED_INT;if(n===bi)return i.FLOAT;if(n===wr)return i.HALF_FLOAT;if(n===rg)return i.ALPHA;if(n===og)return i.RGB;if(n===Pn)return i.RGBA;if(n===To)return i.DEPTH_COMPONENT;if(n===Ia)return i.DEPTH_STENCIL;if(n===ag)return i.RED;if(n===Mc)return i.RED_INTEGER;if(n===Cd)return i.RG;if(n===Td)return i.RG_INTEGER;if(n===ho)return i.RGBA_INTEGER;if(n===Nl||n===zl||n===kl||n===Hl)if(o===yt)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(n===Nl)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===zl)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===kl)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===Hl)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(n===Nl)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===zl)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===kl)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===Hl)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===uf||n===ff||n===df||n===hf)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(n===uf)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===ff)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===df)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===hf)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===pf||n===mf||n===gf)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(n===pf||n===mf)return o===yt?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(n===gf)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(n===xf||n===_f||n===vf||n===Af||n===Sf||n===yf||n===bf||n===Mf||n===Cf||n===Tf||n===Ef||n===wf||n===Rf||n===If)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(n===xf)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===_f)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===vf)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===Af)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===Sf)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===yf)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===bf)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===Mf)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===Cf)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===Tf)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===Ef)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===wf)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===Rf)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===If)return o===yt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===Df||n===Pf||n===Ff)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(n===Df)return o===yt?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===Pf)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===Ff)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===Lf||n===Bf||n===Uf||n===Of)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(n===Lf)return r.COMPRESSED_RED_RGTC1_EXT;if(n===Bf)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===Uf)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===Of)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===Ra?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:t}}const PT=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,FT=`
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

}`;class LT{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){const n=new vg(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=n}}getMesh(e){if(this.texture!==null&&this.mesh===null){const t=e.cameras[0].viewport,n=new Ln({vertexShader:PT,fragmentShader:FT,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new Qt(new Ro(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class BT extends Rr{constructor(e,t){super();const n=this;let s=null,r=1,o=null,a="local-floor",l=1,c=null,u=null,f=null,d=null,h=null,x=null;const p=typeof XRWebGLBinding<"u",g=new LT,m={},_=t.getContextAttributes();let v=null,A=null;const S=[],b=[],M=new Xe;let R=null;const y=new ci;y.viewport=new Ot;const C=new ci;C.viewport=new Ot;const F=[y,C],I=new ty;let O=null,z=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(J){let le=S[J];return le===void 0&&(le=new du,S[J]=le),le.getTargetRaySpace()},this.getControllerGrip=function(J){let le=S[J];return le===void 0&&(le=new du,S[J]=le),le.getGripSpace()},this.getHand=function(J){let le=S[J];return le===void 0&&(le=new du,S[J]=le),le.getHandSpace()};function H(J){const le=b.indexOf(J.inputSource);if(le===-1)return;const be=S[le];be!==void 0&&(be.update(J.inputSource,J.frame,c||o),be.dispatchEvent({type:J.type,data:J.inputSource}))}function V(){s.removeEventListener("select",H),s.removeEventListener("selectstart",H),s.removeEventListener("selectend",H),s.removeEventListener("squeeze",H),s.removeEventListener("squeezestart",H),s.removeEventListener("squeezeend",H),s.removeEventListener("end",V),s.removeEventListener("inputsourceschange",q);for(let J=0;J<S.length;J++){const le=b[J];le!==null&&(b[J]=null,S[J].disconnect(le))}O=null,z=null,g.reset();for(const J in m)delete m[J];e.setRenderTarget(v),h=null,d=null,f=null,s=null,A=null,He.stop(),n.isPresenting=!1,e.setPixelRatio(R),e.setSize(M.width,M.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(J){r=J,n.isPresenting===!0&&et("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(J){a=J,n.isPresenting===!0&&et("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return c||o},this.setReferenceSpace=function(J){c=J},this.getBaseLayer=function(){return d!==null?d:h},this.getBinding=function(){return f===null&&p&&(f=new XRWebGLBinding(s,t)),f},this.getFrame=function(){return x},this.getSession=function(){return s},this.setSession=async function(J){if(s=J,s!==null){if(v=e.getRenderTarget(),s.addEventListener("select",H),s.addEventListener("selectstart",H),s.addEventListener("selectend",H),s.addEventListener("squeeze",H),s.addEventListener("squeezestart",H),s.addEventListener("squeezeend",H),s.addEventListener("end",V),s.addEventListener("inputsourceschange",q),_.xrCompatible!==!0&&await t.makeXRCompatible(),R=e.getPixelRatio(),e.getSize(M),p&&"createProjectionLayer"in XRWebGLBinding.prototype){let be=null,ze=null,Ce=null;_.depth&&(Ce=_.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,be=_.stencil?Ia:To,ze=_.stencil?Ra:hi);const Qe={colorFormat:t.RGBA8,depthFormat:Ce,scaleFactor:r};f=this.getBinding(),d=f.createProjectionLayer(Qe),s.updateRenderState({layers:[d]}),e.setPixelRatio(1),e.setSize(d.textureWidth,d.textureHeight,!1),A=new qs(d.textureWidth,d.textureHeight,{format:Pn,type:Gi,depthTexture:new Id(d.textureWidth,d.textureHeight,ze,void 0,void 0,void 0,void 0,void 0,void 0,be),stencilBuffer:_.stencil,colorSpace:e.outputColorSpace,samples:_.antialias?4:0,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1})}else{const be={antialias:_.antialias,alpha:!0,depth:_.depth,stencil:_.stencil,framebufferScaleFactor:r};h=new XRWebGLLayer(s,t,be),s.updateRenderState({baseLayer:h}),e.setPixelRatio(1),e.setSize(h.framebufferWidth,h.framebufferHeight,!1),A=new qs(h.framebufferWidth,h.framebufferHeight,{format:Pn,type:Gi,colorSpace:e.outputColorSpace,stencilBuffer:_.stencil,resolveDepthBuffer:h.ignoreDepthValues===!1,resolveStencilBuffer:h.ignoreDepthValues===!1})}A.isXRRenderTarget=!0,this.setFoveation(l),c=null,o=await s.requestReferenceSpace(a),He.setContext(s),He.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return g.getDepthTexture()};function q(J){for(let le=0;le<J.removed.length;le++){const be=J.removed[le],ze=b.indexOf(be);ze>=0&&(b[ze]=null,S[ze].disconnect(be))}for(let le=0;le<J.added.length;le++){const be=J.added[le];let ze=b.indexOf(be);if(ze===-1){for(let Qe=0;Qe<S.length;Qe++)if(Qe>=b.length){b.push(be),ze=Qe;break}else if(b[Qe]===null){b[Qe]=be,ze=Qe;break}if(ze===-1)break}const Ce=S[ze];Ce&&Ce.connect(be)}}const G=new U,Y=new U;function ae(J,le,be){G.setFromMatrixPosition(le.matrixWorld),Y.setFromMatrixPosition(be.matrixWorld);const ze=G.distanceTo(Y),Ce=le.projectionMatrix.elements,Qe=be.projectionMatrix.elements,B=Ce[14]/(Ce[10]-1),N=Ce[14]/(Ce[10]+1),j=(Ce[9]+1)/Ce[5],P=(Ce[9]-1)/Ce[5],ie=(Ce[8]-1)/Ce[0],re=(Qe[8]+1)/Qe[0],fe=B*ie,ne=B*re,he=ze/(-ie+re),te=he*-ie;if(le.matrixWorld.decompose(J.position,J.quaternion,J.scale),J.translateX(te),J.translateZ(he),J.matrixWorld.compose(J.position,J.quaternion,J.scale),J.matrixWorldInverse.copy(J.matrixWorld).invert(),Ce[10]===-1)J.projectionMatrix.copy(le.projectionMatrix),J.projectionMatrixInverse.copy(le.projectionMatrixInverse);else{const ge=B+he,D=N+he,T=fe-te,X=ne+(ze-te),se=j*N/D*ge,ce=P*N/D*ge;J.projectionMatrix.makePerspective(T,X,se,ce,ge,D),J.projectionMatrixInverse.copy(J.projectionMatrix).invert()}}function Ae(J,le){le===null?J.matrixWorld.copy(J.matrix):J.matrixWorld.multiplyMatrices(le.matrixWorld,J.matrix),J.matrixWorldInverse.copy(J.matrixWorld).invert()}this.updateCamera=function(J){if(s===null)return;let le=J.near,be=J.far;g.texture!==null&&(g.depthNear>0&&(le=g.depthNear),g.depthFar>0&&(be=g.depthFar)),I.near=C.near=y.near=le,I.far=C.far=y.far=be,(O!==I.near||z!==I.far)&&(s.updateRenderState({depthNear:I.near,depthFar:I.far}),O=I.near,z=I.far),I.layers.mask=J.layers.mask|6,y.layers.mask=I.layers.mask&3,C.layers.mask=I.layers.mask&5;const ze=J.parent,Ce=I.cameras;Ae(I,ze);for(let Qe=0;Qe<Ce.length;Qe++)Ae(Ce[Qe],ze);Ce.length===2?ae(I,y,C):I.projectionMatrix.copy(y.projectionMatrix),Pe(J,I,ze)};function Pe(J,le,be){be===null?J.matrix.copy(le.matrixWorld):(J.matrix.copy(be.matrixWorld),J.matrix.invert(),J.matrix.multiply(le.matrixWorld)),J.matrix.decompose(J.position,J.quaternion,J.scale),J.updateMatrixWorld(!0),J.projectionMatrix.copy(le.projectionMatrix),J.projectionMatrixInverse.copy(le.projectionMatrixInverse),J.isPerspectiveCamera&&(J.fov=Pa*2*Math.atan(1/J.projectionMatrix.elements[5]),J.zoom=1)}this.getCamera=function(){return I},this.getFoveation=function(){if(!(d===null&&h===null))return l},this.setFoveation=function(J){l=J,d!==null&&(d.fixedFoveation=J),h!==null&&h.fixedFoveation!==void 0&&(h.fixedFoveation=J)},this.hasDepthSensing=function(){return g.texture!==null},this.getDepthSensingMesh=function(){return g.getMesh(I)},this.getCameraTexture=function(J){return m[J]};let Ne=null;function We(J,le){if(u=le.getViewerPose(c||o),x=le,u!==null){const be=u.views;h!==null&&(e.setRenderTargetFramebuffer(A,h.framebuffer),e.setRenderTarget(A));let ze=!1;be.length!==I.cameras.length&&(I.cameras.length=0,ze=!0);for(let N=0;N<be.length;N++){const j=be[N];let P=null;if(h!==null)P=h.getViewport(j);else{const re=f.getViewSubImage(d,j);P=re.viewport,N===0&&(e.setRenderTargetTextures(A,re.colorTexture,re.depthStencilTexture),e.setRenderTarget(A))}let ie=F[N];ie===void 0&&(ie=new ci,ie.layers.enable(N),ie.viewport=new Ot,F[N]=ie),ie.matrix.fromArray(j.transform.matrix),ie.matrix.decompose(ie.position,ie.quaternion,ie.scale),ie.projectionMatrix.fromArray(j.projectionMatrix),ie.projectionMatrixInverse.copy(ie.projectionMatrix).invert(),ie.viewport.set(P.x,P.y,P.width,P.height),N===0&&(I.matrix.copy(ie.matrix),I.matrix.decompose(I.position,I.quaternion,I.scale)),ze===!0&&I.cameras.push(ie)}const Ce=s.enabledFeatures;if(Ce&&Ce.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&p){f=n.getBinding();const N=f.getDepthInformation(be[0]);N&&N.isValid&&N.texture&&g.init(N,s.renderState)}if(Ce&&Ce.includes("camera-access")&&p){e.state.unbindTexture(),f=n.getBinding();for(let N=0;N<be.length;N++){const j=be[N].camera;if(j){let P=m[j];P||(P=new vg,m[j]=P);const ie=f.getCameraImage(j);P.sourceTexture=ie}}}}for(let be=0;be<S.length;be++){const ze=b[be],Ce=S[be];ze!==null&&Ce!==void 0&&Ce.update(ze,le,c||o)}Ne&&Ne(J,le),le.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:le}),x=null}const He=new Ag;He.setAnimationLoop(We),this.setAnimationLoop=function(J){Ne=J},this.dispose=function(){}}}const ar=new Ti,UT=new $e;function OT(i,e){function t(g,m){g.matrixAutoUpdate===!0&&g.updateMatrix(),m.value.copy(g.matrix)}function n(g,m){m.color.getRGB(g.fogColor.value,mg(i)),m.isFog?(g.fogNear.value=m.near,g.fogFar.value=m.far):m.isFogExp2&&(g.fogDensity.value=m.density)}function s(g,m,_,v,A){m.isMeshBasicMaterial||m.isMeshLambertMaterial?r(g,m):m.isMeshToonMaterial?(r(g,m),f(g,m)):m.isMeshPhongMaterial?(r(g,m),u(g,m)):m.isMeshStandardMaterial?(r(g,m),d(g,m),m.isMeshPhysicalMaterial&&h(g,m,A)):m.isMeshMatcapMaterial?(r(g,m),x(g,m)):m.isMeshDepthMaterial?r(g,m):m.isMeshDistanceMaterial?(r(g,m),p(g,m)):m.isMeshNormalMaterial?r(g,m):m.isLineBasicMaterial?(o(g,m),m.isLineDashedMaterial&&a(g,m)):m.isPointsMaterial?l(g,m,_,v):m.isSpriteMaterial?c(g,m):m.isShadowMaterial?(g.color.value.copy(m.color),g.opacity.value=m.opacity):m.isShaderMaterial&&(m.uniformsNeedUpdate=!1)}function r(g,m){g.opacity.value=m.opacity,m.color&&g.diffuse.value.copy(m.color),m.emissive&&g.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity),m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.bumpMap&&(g.bumpMap.value=m.bumpMap,t(m.bumpMap,g.bumpMapTransform),g.bumpScale.value=m.bumpScale,m.side===kn&&(g.bumpScale.value*=-1)),m.normalMap&&(g.normalMap.value=m.normalMap,t(m.normalMap,g.normalMapTransform),g.normalScale.value.copy(m.normalScale),m.side===kn&&g.normalScale.value.negate()),m.displacementMap&&(g.displacementMap.value=m.displacementMap,t(m.displacementMap,g.displacementMapTransform),g.displacementScale.value=m.displacementScale,g.displacementBias.value=m.displacementBias),m.emissiveMap&&(g.emissiveMap.value=m.emissiveMap,t(m.emissiveMap,g.emissiveMapTransform)),m.specularMap&&(g.specularMap.value=m.specularMap,t(m.specularMap,g.specularMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest);const _=e.get(m),v=_.envMap,A=_.envMapRotation;v&&(g.envMap.value=v,ar.copy(A),ar.x*=-1,ar.y*=-1,ar.z*=-1,v.isCubeTexture&&v.isRenderTargetTexture===!1&&(ar.y*=-1,ar.z*=-1),g.envMapRotation.value.setFromMatrix4(UT.makeRotationFromEuler(ar)),g.flipEnvMap.value=v.isCubeTexture&&v.isRenderTargetTexture===!1?-1:1,g.reflectivity.value=m.reflectivity,g.ior.value=m.ior,g.refractionRatio.value=m.refractionRatio),m.lightMap&&(g.lightMap.value=m.lightMap,g.lightMapIntensity.value=m.lightMapIntensity,t(m.lightMap,g.lightMapTransform)),m.aoMap&&(g.aoMap.value=m.aoMap,g.aoMapIntensity.value=m.aoMapIntensity,t(m.aoMap,g.aoMapTransform))}function o(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform))}function a(g,m){g.dashSize.value=m.dashSize,g.totalSize.value=m.dashSize+m.gapSize,g.scale.value=m.scale}function l(g,m,_,v){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.size.value=m.size*_,g.scale.value=v*.5,m.map&&(g.map.value=m.map,t(m.map,g.uvTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function c(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.rotation.value=m.rotation,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function u(g,m){g.specular.value.copy(m.specular),g.shininess.value=Math.max(m.shininess,1e-4)}function f(g,m){m.gradientMap&&(g.gradientMap.value=m.gradientMap)}function d(g,m){g.metalness.value=m.metalness,m.metalnessMap&&(g.metalnessMap.value=m.metalnessMap,t(m.metalnessMap,g.metalnessMapTransform)),g.roughness.value=m.roughness,m.roughnessMap&&(g.roughnessMap.value=m.roughnessMap,t(m.roughnessMap,g.roughnessMapTransform)),m.envMap&&(g.envMapIntensity.value=m.envMapIntensity)}function h(g,m,_){g.ior.value=m.ior,m.sheen>0&&(g.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen),g.sheenRoughness.value=m.sheenRoughness,m.sheenColorMap&&(g.sheenColorMap.value=m.sheenColorMap,t(m.sheenColorMap,g.sheenColorMapTransform)),m.sheenRoughnessMap&&(g.sheenRoughnessMap.value=m.sheenRoughnessMap,t(m.sheenRoughnessMap,g.sheenRoughnessMapTransform))),m.clearcoat>0&&(g.clearcoat.value=m.clearcoat,g.clearcoatRoughness.value=m.clearcoatRoughness,m.clearcoatMap&&(g.clearcoatMap.value=m.clearcoatMap,t(m.clearcoatMap,g.clearcoatMapTransform)),m.clearcoatRoughnessMap&&(g.clearcoatRoughnessMap.value=m.clearcoatRoughnessMap,t(m.clearcoatRoughnessMap,g.clearcoatRoughnessMapTransform)),m.clearcoatNormalMap&&(g.clearcoatNormalMap.value=m.clearcoatNormalMap,t(m.clearcoatNormalMap,g.clearcoatNormalMapTransform),g.clearcoatNormalScale.value.copy(m.clearcoatNormalScale),m.side===kn&&g.clearcoatNormalScale.value.negate())),m.dispersion>0&&(g.dispersion.value=m.dispersion),m.iridescence>0&&(g.iridescence.value=m.iridescence,g.iridescenceIOR.value=m.iridescenceIOR,g.iridescenceThicknessMinimum.value=m.iridescenceThicknessRange[0],g.iridescenceThicknessMaximum.value=m.iridescenceThicknessRange[1],m.iridescenceMap&&(g.iridescenceMap.value=m.iridescenceMap,t(m.iridescenceMap,g.iridescenceMapTransform)),m.iridescenceThicknessMap&&(g.iridescenceThicknessMap.value=m.iridescenceThicknessMap,t(m.iridescenceThicknessMap,g.iridescenceThicknessMapTransform))),m.transmission>0&&(g.transmission.value=m.transmission,g.transmissionSamplerMap.value=_.texture,g.transmissionSamplerSize.value.set(_.width,_.height),m.transmissionMap&&(g.transmissionMap.value=m.transmissionMap,t(m.transmissionMap,g.transmissionMapTransform)),g.thickness.value=m.thickness,m.thicknessMap&&(g.thicknessMap.value=m.thicknessMap,t(m.thicknessMap,g.thicknessMapTransform)),g.attenuationDistance.value=m.attenuationDistance,g.attenuationColor.value.copy(m.attenuationColor)),m.anisotropy>0&&(g.anisotropyVector.value.set(m.anisotropy*Math.cos(m.anisotropyRotation),m.anisotropy*Math.sin(m.anisotropyRotation)),m.anisotropyMap&&(g.anisotropyMap.value=m.anisotropyMap,t(m.anisotropyMap,g.anisotropyMapTransform))),g.specularIntensity.value=m.specularIntensity,g.specularColor.value.copy(m.specularColor),m.specularColorMap&&(g.specularColorMap.value=m.specularColorMap,t(m.specularColorMap,g.specularColorMapTransform)),m.specularIntensityMap&&(g.specularIntensityMap.value=m.specularIntensityMap,t(m.specularIntensityMap,g.specularIntensityMapTransform))}function x(g,m){m.matcap&&(g.matcap.value=m.matcap)}function p(g,m){const _=e.get(m).light;g.referencePosition.value.setFromMatrixPosition(_.matrixWorld),g.nearDistance.value=_.shadow.camera.near,g.farDistance.value=_.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:s}}function NT(i,e,t,n){let s={},r={},o=[];const a=i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);function l(_,v){const A=v.program;n.uniformBlockBinding(_,A)}function c(_,v){let A=s[_.id];A===void 0&&(x(_),A=u(_),s[_.id]=A,_.addEventListener("dispose",g));const S=v.program;n.updateUBOMapping(_,S);const b=e.render.frame;r[_.id]!==b&&(d(_),r[_.id]=b)}function u(_){const v=f();_.__bindingPointIndex=v;const A=i.createBuffer(),S=_.__size,b=_.usage;return i.bindBuffer(i.UNIFORM_BUFFER,A),i.bufferData(i.UNIFORM_BUFFER,S,b),i.bindBuffer(i.UNIFORM_BUFFER,null),i.bindBufferBase(i.UNIFORM_BUFFER,v,A),A}function f(){for(let _=0;_<a;_++)if(o.indexOf(_)===-1)return o.push(_),_;return Xt("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function d(_){const v=s[_.id],A=_.uniforms,S=_.__cache;i.bindBuffer(i.UNIFORM_BUFFER,v);for(let b=0,M=A.length;b<M;b++){const R=Array.isArray(A[b])?A[b]:[A[b]];for(let y=0,C=R.length;y<C;y++){const F=R[y];if(h(F,b,y,S)===!0){const I=F.__offset,O=Array.isArray(F.value)?F.value:[F.value];let z=0;for(let H=0;H<O.length;H++){const V=O[H],q=p(V);typeof V=="number"||typeof V=="boolean"?(F.__data[0]=V,i.bufferSubData(i.UNIFORM_BUFFER,I+z,F.__data)):V.isMatrix3?(F.__data[0]=V.elements[0],F.__data[1]=V.elements[1],F.__data[2]=V.elements[2],F.__data[3]=0,F.__data[4]=V.elements[3],F.__data[5]=V.elements[4],F.__data[6]=V.elements[5],F.__data[7]=0,F.__data[8]=V.elements[6],F.__data[9]=V.elements[7],F.__data[10]=V.elements[8],F.__data[11]=0):(V.toArray(F.__data,z),z+=q.storage/Float32Array.BYTES_PER_ELEMENT)}i.bufferSubData(i.UNIFORM_BUFFER,I,F.__data)}}}i.bindBuffer(i.UNIFORM_BUFFER,null)}function h(_,v,A,S){const b=_.value,M=v+"_"+A;if(S[M]===void 0)return typeof b=="number"||typeof b=="boolean"?S[M]=b:S[M]=b.clone(),!0;{const R=S[M];if(typeof b=="number"||typeof b=="boolean"){if(R!==b)return S[M]=b,!0}else if(R.equals(b)===!1)return R.copy(b),!0}return!1}function x(_){const v=_.uniforms;let A=0;const S=16;for(let M=0,R=v.length;M<R;M++){const y=Array.isArray(v[M])?v[M]:[v[M]];for(let C=0,F=y.length;C<F;C++){const I=y[C],O=Array.isArray(I.value)?I.value:[I.value];for(let z=0,H=O.length;z<H;z++){const V=O[z],q=p(V),G=A%S,Y=G%q.boundary,ae=G+Y;A+=Y,ae!==0&&S-ae<q.storage&&(A+=S-ae),I.__data=new Float32Array(q.storage/Float32Array.BYTES_PER_ELEMENT),I.__offset=A,A+=q.storage}}}const b=A%S;return b>0&&(A+=S-b),_.__size=A,_.__cache={},this}function p(_){const v={boundary:0,storage:0};return typeof _=="number"||typeof _=="boolean"?(v.boundary=4,v.storage=4):_.isVector2?(v.boundary=8,v.storage=8):_.isVector3||_.isColor?(v.boundary=16,v.storage=12):_.isVector4?(v.boundary=16,v.storage=16):_.isMatrix3?(v.boundary=48,v.storage=48):_.isMatrix4?(v.boundary=64,v.storage=64):_.isTexture?et("WebGLRenderer: Texture samplers can not be part of an uniforms group."):et("WebGLRenderer: Unsupported uniform value type.",_),v}function g(_){const v=_.target;v.removeEventListener("dispose",g);const A=o.indexOf(v.__bindingPointIndex);o.splice(A,1),i.deleteBuffer(s[v.id]),delete s[v.id],delete r[v.id]}function m(){for(const _ in s)i.deleteBuffer(s[_]);o=[],s={},r={}}return{bind:l,update:c,dispose:m}}const zT=new Uint16Array([11481,15204,11534,15171,11808,15015,12385,14843,12894,14716,13396,14600,13693,14483,13976,14366,14237,14171,14405,13961,14511,13770,14605,13598,14687,13444,14760,13305,14822,13066,14876,12857,14923,12675,14963,12517,14997,12379,15025,12230,15049,12023,15070,11843,15086,11687,15100,11551,15111,11433,15120,11330,15127,11217,15132,11060,15135,10922,15138,10801,15139,10695,15139,10600,13012,14923,13020,14917,13064,14886,13176,14800,13349,14666,13513,14526,13724,14398,13960,14230,14200,14020,14383,13827,14488,13651,14583,13491,14667,13348,14740,13132,14803,12908,14856,12713,14901,12542,14938,12394,14968,12241,14992,12017,15010,11822,15024,11654,15034,11507,15041,11380,15044,11269,15044,11081,15042,10913,15037,10764,15031,10635,15023,10520,15014,10419,15003,10330,13657,14676,13658,14673,13670,14660,13698,14622,13750,14547,13834,14442,13956,14317,14112,14093,14291,13889,14407,13704,14499,13538,14586,13389,14664,13201,14733,12966,14792,12758,14842,12577,14882,12418,14915,12272,14940,12033,14959,11826,14972,11646,14980,11490,14983,11355,14983,11212,14979,11008,14971,10830,14961,10675,14950,10540,14936,10420,14923,10315,14909,10204,14894,10041,14089,14460,14090,14459,14096,14452,14112,14431,14141,14388,14186,14305,14252,14130,14341,13941,14399,13756,14467,13585,14539,13430,14610,13272,14677,13026,14737,12808,14790,12617,14833,12449,14869,12303,14896,12065,14916,11845,14929,11655,14937,11490,14939,11347,14936,11184,14930,10970,14921,10783,14912,10621,14900,10480,14885,10356,14867,10247,14848,10062,14827,9894,14805,9745,14400,14208,14400,14206,14402,14198,14406,14174,14415,14122,14427,14035,14444,13913,14469,13767,14504,13613,14548,13463,14598,13324,14651,13082,14704,12858,14752,12658,14795,12483,14831,12330,14860,12106,14881,11875,14895,11675,14903,11501,14905,11351,14903,11178,14900,10953,14892,10757,14880,10589,14865,10442,14847,10313,14827,10162,14805,9965,14782,9792,14757,9642,14731,9507,14562,13883,14562,13883,14563,13877,14566,13862,14570,13830,14576,13773,14584,13689,14595,13582,14613,13461,14637,13336,14668,13120,14704,12897,14741,12695,14776,12516,14808,12358,14835,12150,14856,11910,14870,11701,14878,11519,14882,11361,14884,11187,14880,10951,14871,10748,14858,10572,14842,10418,14823,10286,14801,10099,14777,9897,14751,9722,14725,9567,14696,9430,14666,9309,14702,13604,14702,13604,14702,13600,14703,13591,14705,13570,14707,13533,14709,13477,14712,13400,14718,13305,14727,13106,14743,12907,14762,12716,14784,12539,14807,12380,14827,12190,14844,11943,14855,11727,14863,11539,14870,11376,14871,11204,14868,10960,14858,10748,14845,10565,14829,10406,14809,10269,14786,10058,14761,9852,14734,9671,14705,9512,14674,9374,14641,9253,14608,9076,14821,13366,14821,13365,14821,13364,14821,13358,14821,13344,14821,13320,14819,13252,14817,13145,14815,13011,14814,12858,14817,12698,14823,12539,14832,12389,14841,12214,14850,11968,14856,11750,14861,11558,14866,11390,14867,11226,14862,10972,14853,10754,14840,10565,14823,10401,14803,10259,14780,10032,14754,9820,14725,9635,14694,9473,14661,9333,14627,9203,14593,8988,14557,8798,14923,13014,14922,13014,14922,13012,14922,13004,14920,12987,14919,12957,14915,12907,14909,12834,14902,12738,14894,12623,14888,12498,14883,12370,14880,12203,14878,11970,14875,11759,14873,11569,14874,11401,14872,11243,14865,10986,14855,10762,14842,10568,14825,10401,14804,10255,14781,10017,14754,9799,14725,9611,14692,9445,14658,9301,14623,9139,14587,8920,14548,8729,14509,8562,15008,12672,15008,12672,15008,12671,15007,12667,15005,12656,15001,12637,14997,12605,14989,12556,14978,12490,14966,12407,14953,12313,14940,12136,14927,11934,14914,11742,14903,11563,14896,11401,14889,11247,14879,10992,14866,10767,14851,10570,14833,10400,14812,10252,14789,10007,14761,9784,14731,9592,14698,9424,14663,9279,14627,9088,14588,8868,14548,8676,14508,8508,14467,8360,15080,12386,15080,12386,15079,12385,15078,12383,15076,12378,15072,12367,15066,12347,15057,12315,15045,12253,15030,12138,15012,11998,14993,11845,14972,11685,14951,11530,14935,11383,14920,11228,14904,10981,14887,10762,14870,10567,14850,10397,14827,10248,14803,9997,14774,9771,14743,9578,14710,9407,14674,9259,14637,9048,14596,8826,14555,8632,14514,8464,14471,8317,14427,8182,15139,12008,15139,12008,15138,12008,15137,12007,15135,12003,15130,11990,15124,11969,15115,11929,15102,11872,15086,11794,15064,11693,15041,11581,15013,11459,14987,11336,14966,11170,14944,10944,14921,10738,14898,10552,14875,10387,14850,10239,14824,9983,14794,9758,14762,9563,14728,9392,14692,9244,14653,9014,14611,8791,14569,8597,14526,8427,14481,8281,14436,8110,14391,7885,15188,11617,15188,11617,15187,11617,15186,11618,15183,11617,15179,11612,15173,11601,15163,11581,15150,11546,15133,11495,15110,11427,15083,11346,15051,11246,15024,11057,14996,10868,14967,10687,14938,10517,14911,10362,14882,10206,14853,9956,14821,9737,14787,9543,14752,9375,14715,9228,14675,8980,14632,8760,14589,8565,14544,8395,14498,8248,14451,8049,14404,7824,14357,7630,15228,11298,15228,11298,15227,11299,15226,11301,15223,11303,15219,11302,15213,11299,15204,11290,15191,11271,15174,11217,15150,11129,15119,11015,15087,10886,15057,10744,15024,10599,14990,10455,14957,10318,14924,10143,14891,9911,14856,9701,14820,9516,14782,9352,14744,9200,14703,8946,14659,8725,14615,8533,14568,8366,14521,8220,14472,7992,14423,7770,14374,7578,14315,7408,15260,10819,15260,10819,15259,10822,15258,10826,15256,10832,15251,10836,15246,10841,15237,10838,15225,10821,15207,10788,15183,10734,15151,10660,15120,10571,15087,10469,15049,10359,15012,10249,14974,10041,14937,9837,14900,9647,14860,9475,14820,9320,14779,9147,14736,8902,14691,8688,14646,8499,14598,8335,14549,8189,14499,7940,14448,7720,14397,7529,14347,7363,14256,7218,15285,10410,15285,10411,15285,10413,15284,10418,15282,10425,15278,10434,15272,10442,15264,10449,15252,10445,15235,10433,15210,10403,15179,10358,15149,10301,15113,10218,15073,10059,15033,9894,14991,9726,14951,9565,14909,9413,14865,9273,14822,9073,14777,8845,14730,8641,14682,8459,14633,8300,14583,8129,14531,7883,14479,7670,14426,7482,14373,7321,14305,7176,14201,6939,15305,9939,15305,9940,15305,9945,15304,9955,15302,9967,15298,9989,15293,10010,15286,10033,15274,10044,15258,10045,15233,10022,15205,9975,15174,9903,15136,9808,15095,9697,15053,9578,15009,9451,14965,9327,14918,9198,14871,8973,14825,8766,14775,8579,14725,8408,14675,8259,14622,8058,14569,7821,14515,7615,14460,7435,14405,7276,14350,7108,14256,6866,14149,6653,15321,9444,15321,9445,15321,9448,15320,9458,15317,9470,15314,9490,15310,9515,15302,9540,15292,9562,15276,9579,15251,9577,15226,9559,15195,9519,15156,9463,15116,9389,15071,9304,15025,9208,14978,9023,14927,8838,14878,8661,14827,8496,14774,8344,14722,8206,14667,7973,14612,7749,14556,7555,14499,7382,14443,7229,14385,7025,14322,6791,14210,6588,14100,6409,15333,8920,15333,8921,15332,8927,15332,8943,15329,8965,15326,9002,15322,9048,15316,9106,15307,9162,15291,9204,15267,9221,15244,9221,15212,9196,15175,9134,15133,9043,15088,8930,15040,8801,14990,8665,14938,8526,14886,8391,14830,8261,14775,8087,14719,7866,14661,7664,14603,7482,14544,7322,14485,7178,14426,6936,14367,6713,14281,6517,14166,6348,14054,6198,15341,8360,15341,8361,15341,8366,15341,8379,15339,8399,15336,8431,15332,8473,15326,8527,15318,8585,15302,8632,15281,8670,15258,8690,15227,8690,15191,8664,15149,8612,15104,8543,15055,8456,15001,8360,14948,8259,14892,8122,14834,7923,14776,7734,14716,7558,14656,7397,14595,7250,14534,7070,14472,6835,14410,6628,14350,6443,14243,6283,14125,6135,14010,5889,15348,7715,15348,7717,15348,7725,15347,7745,15345,7780,15343,7836,15339,7905,15334,8e3,15326,8103,15310,8193,15293,8239,15270,8270,15240,8287,15204,8283,15163,8260,15118,8223,15067,8143,15014,8014,14958,7873,14899,7723,14839,7573,14778,7430,14715,7293,14652,7164,14588,6931,14524,6720,14460,6531,14396,6362,14330,6210,14207,6015,14086,5781,13969,5576,15352,7114,15352,7116,15352,7128,15352,7159,15350,7195,15348,7237,15345,7299,15340,7374,15332,7457,15317,7544,15301,7633,15280,7703,15251,7754,15216,7775,15176,7767,15131,7733,15079,7670,15026,7588,14967,7492,14906,7387,14844,7278,14779,7171,14714,6965,14648,6770,14581,6587,14515,6420,14448,6269,14382,6123,14299,5881,14172,5665,14049,5477,13929,5310,15355,6329,15355,6330,15355,6339,15355,6362,15353,6410,15351,6472,15349,6572,15344,6688,15337,6835,15323,6985,15309,7142,15287,7220,15260,7277,15226,7310,15188,7326,15142,7318,15090,7285,15036,7239,14976,7177,14914,7045,14849,6892,14782,6736,14714,6581,14645,6433,14576,6293,14506,6164,14438,5946,14369,5733,14270,5540,14140,5369,14014,5216,13892,5043,15357,5483,15357,5484,15357,5496,15357,5528,15356,5597,15354,5692,15351,5835,15347,6011,15339,6195,15328,6317,15314,6446,15293,6566,15268,6668,15235,6746,15197,6796,15152,6811,15101,6790,15046,6748,14985,6673,14921,6583,14854,6479,14785,6371,14714,6259,14643,6149,14571,5946,14499,5750,14428,5567,14358,5401,14242,5250,14109,5111,13980,4870,13856,4657,15359,4555,15359,4557,15358,4573,15358,4633,15357,4715,15355,4841,15353,5061,15349,5216,15342,5391,15331,5577,15318,5770,15299,5967,15274,6150,15243,6223,15206,6280,15161,6310,15111,6317,15055,6300,14994,6262,14928,6208,14860,6141,14788,5994,14715,5838,14641,5684,14566,5529,14492,5384,14418,5247,14346,5121,14216,4892,14079,4682,13948,4496,13822,4330,15359,3498,15359,3501,15359,3520,15359,3598,15358,3719,15356,3860,15355,4137,15351,4305,15344,4563,15334,4809,15321,5116,15303,5273,15280,5418,15250,5547,15214,5653,15170,5722,15120,5761,15064,5763,15002,5733,14935,5673,14865,5597,14792,5504,14716,5400,14640,5294,14563,5185,14486,5041,14410,4841,14335,4655,14191,4482,14051,4325,13918,4183,13790,4012,15360,2282,15360,2285,15360,2306,15360,2401,15359,2547,15357,2748,15355,3103,15352,3349,15345,3675,15336,4020,15324,4272,15307,4496,15285,4716,15255,4908,15220,5086,15178,5170,15128,5214,15072,5234,15010,5231,14943,5206,14871,5166,14796,5102,14718,4971,14639,4833,14559,4687,14480,4541,14402,4401,14315,4268,14167,4142,14025,3958,13888,3747,13759,3556,15360,923,15360,925,15360,946,15360,1052,15359,1214,15357,1494,15356,1892,15352,2274,15346,2663,15338,3099,15326,3393,15309,3679,15288,3980,15260,4183,15226,4325,15185,4437,15136,4517,15080,4570,15018,4591,14950,4581,14877,4545,14800,4485,14720,4411,14638,4325,14556,4231,14475,4136,14395,3988,14297,3803,14145,3628,13999,3465,13861,3314,13729,3177,15360,263,15360,264,15360,272,15360,325,15359,407,15358,548,15356,780,15352,1144,15347,1580,15339,2099,15328,2425,15312,2795,15292,3133,15264,3329,15232,3517,15191,3689,15143,3819,15088,3923,15025,3978,14956,3999,14882,3979,14804,3931,14722,3855,14639,3756,14554,3645,14470,3529,14388,3409,14279,3289,14124,3173,13975,3055,13834,2848,13701,2658,15360,49,15360,49,15360,52,15360,75,15359,111,15358,201,15356,283,15353,519,15348,726,15340,1045,15329,1415,15314,1795,15295,2173,15269,2410,15237,2649,15197,2866,15150,3054,15095,3140,15032,3196,14963,3228,14888,3236,14808,3224,14725,3191,14639,3146,14553,3088,14466,2976,14382,2836,14262,2692,14103,2549,13952,2409,13808,2278,13674,2154,15360,4,15360,4,15360,4,15360,13,15359,33,15358,59,15357,112,15353,199,15348,302,15341,456,15331,628,15316,827,15297,1082,15272,1332,15241,1601,15202,1851,15156,2069,15101,2172,15039,2256,14970,2314,14894,2348,14813,2358,14728,2344,14640,2311,14551,2263,14463,2203,14376,2133,14247,2059,14084,1915,13930,1761,13784,1609,13648,1464,15360,0,15360,0,15360,0,15360,3,15359,18,15358,26,15357,53,15354,80,15348,97,15341,165,15332,238,15318,326,15299,427,15275,529,15245,654,15207,771,15161,885,15108,994,15046,1089,14976,1170,14900,1229,14817,1266,14731,1284,14641,1282,14550,1260,14460,1223,14370,1174,14232,1116,14066,1050,13909,981,13761,910,13623,839]);let Ji=null;function kT(){return Ji===null&&(Ji=new ns(zT,32,32,Cd,wr),Ji.minFilter=di,Ji.magFilter=di,Ji.wrapS=fs,Ji.wrapT=fs,Ji.generateMipmaps=!1,Ji.needsUpdate=!0),Ji}class HT{constructor(e={}){const{canvas:t=ZA(),context:n=null,depth:s=!0,stencil:r=!1,alpha:o=!1,antialias:a=!1,premultipliedAlpha:l=!0,preserveDrawingBuffer:c=!1,powerPreference:u="default",failIfMajorPerformanceCaveat:f=!1,reversedDepthBuffer:d=!1}=e;this.isWebGLRenderer=!0;let h;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");h=n.getContextAttributes().alpha}else h=o;const x=new Set([ho,Td,Mc]),p=new Set([Gi,hi,wa,Ra,bd,Md]),g=new Uint32Array(4),m=new Int32Array(4);let _=null,v=null;const A=[],S=[];this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=Hs,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const b=this;let M=!1;this._outputColorSpace=li;let R=0,y=0,C=null,F=-1,I=null;const O=new Ot,z=new Ot;let H=null;const V=new lt(0);let q=0,G=t.width,Y=t.height,ae=1,Ae=null,Pe=null;const Ne=new Ot(0,0,G,Y),We=new Ot(0,0,G,Y);let He=!1;const J=new _g;let le=!1,be=!1;const ze=new $e,Ce=new U,Qe=new Ot,B={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let N=!1;function j(){return C===null?ae:1}let P=n;function ie(L,K){return t.getContext(L,K)}try{const L={alpha:!0,depth:s,stencil:r,antialias:a,premultipliedAlpha:l,preserveDrawingBuffer:c,powerPreference:u,failIfMajorPerformanceCaveat:f};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${Sd}`),t.addEventListener("webglcontextlost",ve,!1),t.addEventListener("webglcontextrestored",de,!1),t.addEventListener("webglcontextcreationerror",ke,!1),P===null){const K="webgl2";if(P=ie(K,L),P===null)throw ie(K)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(L){throw L("WebGLRenderer: "+L.message),L}let re,fe,ne,he,te,ge,D,T,X,se,ce,Z,Be,Se,Oe,k,ee,me,Te,Fe,Re,Ye,W,De;function ye(){re=new jM(P),re.init(),Ye=new Cg(P,re),fe=new HM(P,re,e,Ye),ne=new IT(P,re),fe.reversedDepthBuffer&&d&&ne.buffers.depth.setReversed(!0),he=new JM(P),te=new xT,ge=new DT(P,re,ne,te,fe,Ye,he),D=new GM(b),T=new KM(b),X=new iy(P),W=new zM(P,X),se=new $M(P,X,he,W),ce=new tC(P,se,X,he),Te=new eC(P,fe,ge),k=new VM(te),Z=new gT(b,D,T,re,fe,W,k),Be=new OT(b,te),Se=new vT,Oe=new CT(re),me=new NM(b,D,T,ne,ce,h,l),ee=new wT(b,ce,fe),De=new NT(P,he,fe,ne),Fe=new kM(P,re,he),Re=new ZM(P,re,he),he.programs=Z.programs,b.capabilities=fe,b.extensions=re,b.properties=te,b.renderLists=Se,b.shadowMap=ee,b.state=ne,b.info=he}ye();const Me=new BT(b,P);this.xr=Me,this.getContext=function(){return P},this.getContextAttributes=function(){return P.getContextAttributes()},this.forceContextLoss=function(){const L=re.get("WEBGL_lose_context");L&&L.loseContext()},this.forceContextRestore=function(){const L=re.get("WEBGL_lose_context");L&&L.restoreContext()},this.getPixelRatio=function(){return ae},this.setPixelRatio=function(L){L!==void 0&&(ae=L,this.setSize(G,Y,!1))},this.getSize=function(L){return L.set(G,Y)},this.setSize=function(L,K,oe=!0){if(Me.isPresenting){et("WebGLRenderer: Can't change size while VR device is presenting.");return}G=L,Y=K,t.width=Math.floor(L*ae),t.height=Math.floor(K*ae),oe===!0&&(t.style.width=L+"px",t.style.height=K+"px"),this.setViewport(0,0,L,K)},this.getDrawingBufferSize=function(L){return L.set(G*ae,Y*ae).floor()},this.setDrawingBufferSize=function(L,K,oe){G=L,Y=K,ae=oe,t.width=Math.floor(L*oe),t.height=Math.floor(K*oe),this.setViewport(0,0,L,K)},this.getCurrentViewport=function(L){return L.copy(O)},this.getViewport=function(L){return L.copy(Ne)},this.setViewport=function(L,K,oe,w){L.isVector4?Ne.set(L.x,L.y,L.z,L.w):Ne.set(L,K,oe,w),ne.viewport(O.copy(Ne).multiplyScalar(ae).round())},this.getScissor=function(L){return L.copy(We)},this.setScissor=function(L,K,oe,w){L.isVector4?We.set(L.x,L.y,L.z,L.w):We.set(L,K,oe,w),ne.scissor(z.copy(We).multiplyScalar(ae).round())},this.getScissorTest=function(){return He},this.setScissorTest=function(L){ne.setScissorTest(He=L)},this.setOpaqueSort=function(L){Ae=L},this.setTransparentSort=function(L){Pe=L},this.getClearColor=function(L){return L.copy(me.getClearColor())},this.setClearColor=function(){me.setClearColor(...arguments)},this.getClearAlpha=function(){return me.getClearAlpha()},this.setClearAlpha=function(){me.setClearAlpha(...arguments)},this.clear=function(L=!0,K=!0,oe=!0){let w=0;if(L){let E=!1;if(C!==null){const Q=C.texture.format;E=x.has(Q)}if(E){const Q=C.texture.type,ue=p.has(Q),pe=me.getClearColor(),xe=me.getClearAlpha(),Ee=pe.r,we=pe.g,_e=pe.b;ue?(g[0]=Ee,g[1]=we,g[2]=_e,g[3]=xe,P.clearBufferuiv(P.COLOR,0,g)):(m[0]=Ee,m[1]=we,m[2]=_e,m[3]=xe,P.clearBufferiv(P.COLOR,0,m))}else w|=P.COLOR_BUFFER_BIT}K&&(w|=P.DEPTH_BUFFER_BIT),oe&&(w|=P.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),P.clear(w)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){t.removeEventListener("webglcontextlost",ve,!1),t.removeEventListener("webglcontextrestored",de,!1),t.removeEventListener("webglcontextcreationerror",ke,!1),me.dispose(),Se.dispose(),Oe.dispose(),te.dispose(),D.dispose(),T.dispose(),ce.dispose(),W.dispose(),De.dispose(),Z.dispose(),Me.dispose(),Me.removeEventListener("sessionstart",ko),Me.removeEventListener("sessionend",Dr),Xi.stop()};function ve(L){L.preventDefault(),np("WebGLRenderer: Context Lost."),M=!0}function de(){np("WebGLRenderer: Context Restored."),M=!1;const L=he.autoReset,K=ee.enabled,oe=ee.autoUpdate,w=ee.needsUpdate,E=ee.type;ye(),he.autoReset=L,ee.enabled=K,ee.autoUpdate=oe,ee.needsUpdate=w,ee.type=E}function ke(L){Xt("WebGLRenderer: A WebGL context could not be created. Reason: ",L.statusMessage)}function je(L){const K=L.target;K.removeEventListener("dispose",je),ct(K)}function ct(L){ht(L),te.remove(L)}function ht(L){const K=te.get(L).programs;K!==void 0&&(K.forEach(function(oe){Z.releaseProgram(oe)}),L.isShaderMaterial&&Z.releaseShaderCache(L))}this.renderBufferDirect=function(L,K,oe,w,E,Q){K===null&&(K=B);const ue=E.isMesh&&E.matrixWorld.determinant()<0,pe=rn(L,K,oe,w,E);ne.setMaterial(w,ue);let xe=oe.index,Ee=1;if(w.wireframe===!0){if(xe=se.getWireframeAttribute(oe),xe===void 0)return;Ee=2}const we=oe.drawRange,_e=oe.attributes.position;let Le=we.start*Ee,Ve=(we.start+we.count)*Ee;Q!==null&&(Le=Math.max(Le,Q.start*Ee),Ve=Math.min(Ve,(Q.start+Q.count)*Ee)),xe!==null?(Le=Math.max(Le,0),Ve=Math.min(Ve,xe.count)):_e!=null&&(Le=Math.max(Le,0),Ve=Math.min(Ve,_e.count));const tt=Ve-Le;if(tt<0||tt===1/0)return;W.setup(E,w,pe,oe,xe);let rt,Ke=Fe;if(xe!==null&&(rt=X.get(xe),Ke=Re,Ke.setIndex(rt)),E.isMesh)w.wireframe===!0?(ne.setLineWidth(w.wireframeLinewidth*j()),Ke.setMode(P.LINES)):Ke.setMode(P.TRIANGLES);else if(E.isLine){let Ge=w.linewidth;Ge===void 0&&(Ge=1),ne.setLineWidth(Ge*j()),E.isLineSegments?Ke.setMode(P.LINES):E.isLineLoop?Ke.setMode(P.LINE_LOOP):Ke.setMode(P.LINE_STRIP)}else E.isPoints?Ke.setMode(P.POINTS):E.isSprite&&Ke.setMode(P.TRIANGLES);if(E.isBatchedMesh)if(E._multiDrawInstances!==null)Da("WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."),Ke.renderMultiDrawInstances(E._multiDrawStarts,E._multiDrawCounts,E._multiDrawCount,E._multiDrawInstances);else if(re.get("WEBGL_multi_draw"))Ke.renderMultiDraw(E._multiDrawStarts,E._multiDrawCounts,E._multiDrawCount);else{const Ge=E._multiDrawStarts,ut=E._multiDrawCounts,ot=E._multiDrawCount,Mn=xe?X.get(xe).bytesPerElement:1,_i=te.get(w).currentProgram.getUniforms();for(let fn=0;fn<ot;fn++)_i.setValue(P,"_gl_DrawID",fn),Ke.render(Ge[fn]/Mn,ut[fn])}else if(E.isInstancedMesh)Ke.renderInstances(Le,tt,E.count);else if(oe.isInstancedBufferGeometry){const Ge=oe._maxInstanceCount!==void 0?oe._maxInstanceCount:1/0,ut=Math.min(oe.instanceCount,Ge);Ke.renderInstances(Le,tt,ut)}else Ke.render(Le,tt)};function bn(L,K,oe){L.transparent===!0&&L.side===ui&&L.forceSinglePass===!1?(L.side=kn,L.needsUpdate=!0,It(L,K,oe),L.side=Vi,L.needsUpdate=!0,It(L,K,oe),L.side=ui):It(L,K,oe)}this.compile=function(L,K,oe=null){oe===null&&(oe=L),v=Oe.get(oe),v.init(K),S.push(v),oe.traverseVisible(function(E){E.isLight&&E.layers.test(K.layers)&&(v.pushLight(E),E.castShadow&&v.pushShadow(E))}),L!==oe&&L.traverseVisible(function(E){E.isLight&&E.layers.test(K.layers)&&(v.pushLight(E),E.castShadow&&v.pushShadow(E))}),v.setupLights();const w=new Set;return L.traverse(function(E){if(!(E.isMesh||E.isPoints||E.isLine||E.isSprite))return;const Q=E.material;if(Q)if(Array.isArray(Q))for(let ue=0;ue<Q.length;ue++){const pe=Q[ue];bn(pe,oe,E),w.add(pe)}else bn(Q,oe,E),w.add(Q)}),v=S.pop(),w},this.compileAsync=function(L,K,oe=null){const w=this.compile(L,K,oe);return new Promise(E=>{function Q(){if(w.forEach(function(ue){te.get(ue).currentProgram.isReady()&&w.delete(ue)}),w.size===0){E(L);return}setTimeout(Q,10)}re.get("KHR_parallel_shader_compile")!==null?Q():setTimeout(Q,10)})};let Un=null;function Pc(L){Un&&Un(L)}function ko(){Xi.stop()}function Dr(){Xi.start()}const Xi=new Ag;Xi.setAnimationLoop(Pc),typeof self<"u"&&Xi.setContext(self),this.setAnimationLoop=function(L){Un=L,Me.setAnimationLoop(L),L===null?Xi.stop():Xi.start()},Me.addEventListener("sessionstart",ko),Me.addEventListener("sessionend",Dr),this.render=function(L,K){if(K!==void 0&&K.isCamera!==!0){Xt("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(M===!0)return;if(L.matrixWorldAutoUpdate===!0&&L.updateMatrixWorld(),K.parent===null&&K.matrixWorldAutoUpdate===!0&&K.updateMatrixWorld(),Me.enabled===!0&&Me.isPresenting===!0&&(Me.cameraAutoUpdate===!0&&Me.updateCamera(K),K=Me.getCamera()),L.isScene===!0&&L.onBeforeRender(b,L,K,C),v=Oe.get(L,S.length),v.init(K),S.push(v),ze.multiplyMatrices(K.projectionMatrix,K.matrixWorldInverse),J.setFromProjectionMatrix(ze,Li,K.reversedDepth),be=this.localClippingEnabled,le=k.init(this.clippingPlanes,be),_=Se.get(L,A.length),_.init(),A.push(_),Me.enabled===!0&&Me.isPresenting===!0){const Q=b.xr.getDepthSensingMesh();Q!==null&&Ho(Q,K,-1/0,b.sortObjects)}Ho(L,K,0,b.sortObjects),_.finish(),b.sortObjects===!0&&_.sort(Ae,Pe),N=Me.enabled===!1||Me.isPresenting===!1||Me.hasDepthSensing()===!1,N&&me.addToRenderList(_,L),this.info.render.frame++,le===!0&&k.beginShadows();const oe=v.state.shadowsArray;ee.render(oe,L,K),le===!0&&k.endShadows(),this.info.autoReset===!0&&this.info.reset();const w=_.opaque,E=_.transmissive;if(v.setupLights(),K.isArrayCamera){const Q=K.cameras;if(E.length>0)for(let ue=0,pe=Q.length;ue<pe;ue++){const xe=Q[ue];Vo(w,E,L,xe)}N&&me.render(L);for(let ue=0,pe=Q.length;ue<pe;ue++){const xe=Q[ue];Ka(_,L,xe,xe.viewport)}}else E.length>0&&Vo(w,E,L,K),N&&me.render(L),Ka(_,L,K);C!==null&&y===0&&(ge.updateMultisampleRenderTarget(C),ge.updateRenderTargetMipmap(C)),L.isScene===!0&&L.onAfterRender(b,L,K),W.resetDefaultState(),F=-1,I=null,S.pop(),S.length>0?(v=S[S.length-1],le===!0&&k.setGlobalState(b.clippingPlanes,v.state.camera)):v=null,A.pop(),A.length>0?_=A[A.length-1]:_=null};function Ho(L,K,oe,w){if(L.visible===!1)return;if(L.layers.test(K.layers)){if(L.isGroup)oe=L.renderOrder;else if(L.isLOD)L.autoUpdate===!0&&L.update(K);else if(L.isLight)v.pushLight(L),L.castShadow&&v.pushShadow(L);else if(L.isSprite){if(!L.frustumCulled||J.intersectsSprite(L)){w&&Qe.setFromMatrixPosition(L.matrixWorld).applyMatrix4(ze);const ue=ce.update(L),pe=L.material;pe.visible&&_.push(L,ue,pe,oe,Qe.z,null)}}else if((L.isMesh||L.isLine||L.isPoints)&&(!L.frustumCulled||J.intersectsObject(L))){const ue=ce.update(L),pe=L.material;if(w&&(L.boundingSphere!==void 0?(L.boundingSphere===null&&L.computeBoundingSphere(),Qe.copy(L.boundingSphere.center)):(ue.boundingSphere===null&&ue.computeBoundingSphere(),Qe.copy(ue.boundingSphere.center)),Qe.applyMatrix4(L.matrixWorld).applyMatrix4(ze)),Array.isArray(pe)){const xe=ue.groups;for(let Ee=0,we=xe.length;Ee<we;Ee++){const _e=xe[Ee],Le=pe[_e.materialIndex];Le&&Le.visible&&_.push(L,ue,Le,oe,Qe.z,_e)}}else pe.visible&&_.push(L,ue,pe,oe,Qe.z,null)}}const Q=L.children;for(let ue=0,pe=Q.length;ue<pe;ue++)Ho(Q[ue],K,oe,w)}function Ka(L,K,oe,w){const{opaque:E,transmissive:Q,transparent:ue}=L;v.setupLightsView(oe),le===!0&&k.setGlobalState(b.clippingPlanes,oe),w&&ne.viewport(O.copy(w)),E.length>0&&Zs(E,K,oe),Q.length>0&&Zs(Q,K,oe),ue.length>0&&Zs(ue,K,oe),ne.buffers.depth.setTest(!0),ne.buffers.depth.setMask(!0),ne.buffers.color.setMask(!0),ne.setPolygonOffset(!1)}function Vo(L,K,oe,w){if((oe.isScene===!0?oe.overrideMaterial:null)!==null)return;v.state.transmissionRenderTarget[w.id]===void 0&&(v.state.transmissionRenderTarget[w.id]=new qs(1,1,{generateMipmaps:!0,type:re.has("EXT_color_buffer_half_float")||re.has("EXT_color_buffer_float")?wr:Gi,minFilter:_r,samples:4,stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:dt.workingColorSpace}));const Q=v.state.transmissionRenderTarget[w.id],ue=w.viewport||O;Q.setSize(ue.z*b.transmissionResolutionScale,ue.w*b.transmissionResolutionScale);const pe=b.getRenderTarget(),xe=b.getActiveCubeFace(),Ee=b.getActiveMipmapLevel();b.setRenderTarget(Q),b.getClearColor(V),q=b.getClearAlpha(),q<1&&b.setClearColor(16777215,.5),b.clear(),N&&me.render(oe);const we=b.toneMapping;b.toneMapping=Hs;const _e=w.viewport;if(w.viewport!==void 0&&(w.viewport=void 0),v.setupLightsView(w),le===!0&&k.setGlobalState(b.clippingPlanes,w),Zs(L,oe,w),ge.updateMultisampleRenderTarget(Q),ge.updateRenderTargetMipmap(Q),re.has("WEBGL_multisampled_render_to_texture")===!1){let Le=!1;for(let Ve=0,tt=K.length;Ve<tt;Ve++){const rt=K[Ve],{object:Ke,geometry:Ge,material:ut,group:ot}=rt;if(ut.side===ui&&Ke.layers.test(w.layers)){const Mn=ut.side;ut.side=kn,ut.needsUpdate=!0,ja(Ke,oe,w,Ge,ut,ot),ut.side=Mn,ut.needsUpdate=!0,Le=!0}}Le===!0&&(ge.updateMultisampleRenderTarget(Q),ge.updateRenderTargetMipmap(Q))}b.setRenderTarget(pe,xe,Ee),b.setClearColor(V,q),_e!==void 0&&(w.viewport=_e),b.toneMapping=we}function Zs(L,K,oe){const w=K.isScene===!0?K.overrideMaterial:null;for(let E=0,Q=L.length;E<Q;E++){const ue=L[E],{object:pe,geometry:xe,group:Ee}=ue;let we=ue.material;we.allowOverride===!0&&w!==null&&(we=w),pe.layers.test(oe.layers)&&ja(pe,K,oe,xe,we,Ee)}}function ja(L,K,oe,w,E,Q){L.onBeforeRender(b,K,oe,w,E,Q),L.modelViewMatrix.multiplyMatrices(oe.matrixWorldInverse,L.matrixWorld),L.normalMatrix.getNormalMatrix(L.modelViewMatrix),E.onBeforeRender(b,K,oe,w,L,Q),E.transparent===!0&&E.side===ui&&E.forceSinglePass===!1?(E.side=kn,E.needsUpdate=!0,b.renderBufferDirect(oe,K,w,E,L,Q),E.side=Vi,E.needsUpdate=!0,b.renderBufferDirect(oe,K,w,E,L,Q),E.side=ui):b.renderBufferDirect(oe,K,w,E,L,Q),L.onAfterRender(b,K,oe,w,E,Q)}function It(L,K,oe){K.isScene!==!0&&(K=B);const w=te.get(L),E=v.state.lights,Q=v.state.shadowsArray,ue=E.state.version,pe=Z.getParameters(L,E.state,Q,K,oe),xe=Z.getProgramCacheKey(pe);let Ee=w.programs;w.environment=L.isMeshStandardMaterial?K.environment:null,w.fog=K.fog,w.envMap=(L.isMeshStandardMaterial?T:D).get(L.envMap||w.environment),w.envMapRotation=w.environment!==null&&L.envMap===null?K.environmentRotation:L.envMapRotation,Ee===void 0&&(L.addEventListener("dispose",je),Ee=new Map,w.programs=Ee);let we=Ee.get(xe);if(we!==void 0){if(w.currentProgram===we&&w.lightsStateVersion===ue)return mt(L,pe),we}else pe.uniforms=Z.getUniforms(L),L.onBeforeCompile(pe,b),we=Z.acquireProgram(pe,xe),Ee.set(xe,we),w.uniforms=pe.uniforms;const _e=w.uniforms;return(!L.isShaderMaterial&&!L.isRawShaderMaterial||L.clipping===!0)&&(_e.clippingPlanes=k.uniform),mt(L,pe),w.needsLights=Go(L),w.lightsStateVersion=ue,w.needsLights&&(_e.ambientLightColor.value=E.state.ambient,_e.lightProbe.value=E.state.probe,_e.directionalLights.value=E.state.directional,_e.directionalLightShadows.value=E.state.directionalShadow,_e.spotLights.value=E.state.spot,_e.spotLightShadows.value=E.state.spotShadow,_e.rectAreaLights.value=E.state.rectArea,_e.ltc_1.value=E.state.rectAreaLTC1,_e.ltc_2.value=E.state.rectAreaLTC2,_e.pointLights.value=E.state.point,_e.pointLightShadows.value=E.state.pointShadow,_e.hemisphereLights.value=E.state.hemi,_e.directionalShadowMap.value=E.state.directionalShadowMap,_e.directionalShadowMatrix.value=E.state.directionalShadowMatrix,_e.spotShadowMap.value=E.state.spotShadowMap,_e.spotLightMatrix.value=E.state.spotLightMatrix,_e.spotLightMap.value=E.state.spotLightMap,_e.pointShadowMap.value=E.state.pointShadowMap,_e.pointShadowMatrix.value=E.state.pointShadowMatrix),w.currentProgram=we,w.uniformsList=null,we}function pt(L){if(L.uniformsList===null){const K=L.currentProgram.getUniforms();L.uniformsList=Vl.seqWithValue(K.seq,L.uniforms)}return L.uniformsList}function mt(L,K){const oe=te.get(L);oe.outputColorSpace=K.outputColorSpace,oe.batching=K.batching,oe.batchingColor=K.batchingColor,oe.instancing=K.instancing,oe.instancingColor=K.instancingColor,oe.instancingMorph=K.instancingMorph,oe.skinning=K.skinning,oe.morphTargets=K.morphTargets,oe.morphNormals=K.morphNormals,oe.morphColors=K.morphColors,oe.morphTargetsCount=K.morphTargetsCount,oe.numClippingPlanes=K.numClippingPlanes,oe.numIntersection=K.numClipIntersection,oe.vertexAlphas=K.vertexAlphas,oe.vertexTangents=K.vertexTangents,oe.toneMapping=K.toneMapping}function rn(L,K,oe,w,E){K.isScene!==!0&&(K=B),ge.resetTextureUnits();const Q=K.fog,ue=w.isMeshStandardMaterial?K.environment:null,pe=C===null?b.outputColorSpace:C.isXRRenderTarget===!0?C.texture.colorSpace:Eo,xe=(w.isMeshStandardMaterial?T:D).get(w.envMap||ue),Ee=w.vertexColors===!0&&!!oe.attributes.color&&oe.attributes.color.itemSize===4,we=!!oe.attributes.tangent&&(!!w.normalMap||w.anisotropy>0),_e=!!oe.morphAttributes.position,Le=!!oe.morphAttributes.normal,Ve=!!oe.morphAttributes.color;let tt=Hs;w.toneMapped&&(C===null||C.isXRRenderTarget===!0)&&(tt=b.toneMapping);const rt=oe.morphAttributes.position||oe.morphAttributes.normal||oe.morphAttributes.color,Ke=rt!==void 0?rt.length:0,Ge=te.get(w),ut=v.state.lights;if(le===!0&&(be===!0||L!==I)){const Zt=L===I&&w.id===F;k.setState(w,L,Zt)}let ot=!1;w.version===Ge.__version?(Ge.needsLights&&Ge.lightsStateVersion!==ut.state.version||Ge.outputColorSpace!==pe||E.isBatchedMesh&&Ge.batching===!1||!E.isBatchedMesh&&Ge.batching===!0||E.isBatchedMesh&&Ge.batchingColor===!0&&E.colorTexture===null||E.isBatchedMesh&&Ge.batchingColor===!1&&E.colorTexture!==null||E.isInstancedMesh&&Ge.instancing===!1||!E.isInstancedMesh&&Ge.instancing===!0||E.isSkinnedMesh&&Ge.skinning===!1||!E.isSkinnedMesh&&Ge.skinning===!0||E.isInstancedMesh&&Ge.instancingColor===!0&&E.instanceColor===null||E.isInstancedMesh&&Ge.instancingColor===!1&&E.instanceColor!==null||E.isInstancedMesh&&Ge.instancingMorph===!0&&E.morphTexture===null||E.isInstancedMesh&&Ge.instancingMorph===!1&&E.morphTexture!==null||Ge.envMap!==xe||w.fog===!0&&Ge.fog!==Q||Ge.numClippingPlanes!==void 0&&(Ge.numClippingPlanes!==k.numPlanes||Ge.numIntersection!==k.numIntersection)||Ge.vertexAlphas!==Ee||Ge.vertexTangents!==we||Ge.morphTargets!==_e||Ge.morphNormals!==Le||Ge.morphColors!==Ve||Ge.toneMapping!==tt||Ge.morphTargetsCount!==Ke)&&(ot=!0):(ot=!0,Ge.__version=w.version);let Mn=Ge.currentProgram;ot===!0&&(Mn=It(w,K,E));let _i=!1,fn=!1,Ss=!1;const wt=Mn.getUniforms(),dn=Ge.uniforms;if(ne.useProgram(Mn.program)&&(_i=!0,fn=!0,Ss=!0),w.id!==F&&(F=w.id,fn=!0),_i||I!==L){ne.buffers.depth.getReversed()&&L.reversedDepth!==!0&&(L._reversedDepth=!0,L.updateProjectionMatrix()),wt.setValue(P,"projectionMatrix",L.projectionMatrix),wt.setValue(P,"viewMatrix",L.matrixWorldInverse);const on=wt.map.cameraPosition;on!==void 0&&on.setValue(P,Ce.setFromMatrixPosition(L.matrixWorld)),fe.logarithmicDepthBuffer&&wt.setValue(P,"logDepthBufFC",2/(Math.log(L.far+1)/Math.LN2)),(w.isMeshPhongMaterial||w.isMeshToonMaterial||w.isMeshLambertMaterial||w.isMeshBasicMaterial||w.isMeshStandardMaterial||w.isShaderMaterial)&&wt.setValue(P,"isOrthographic",L.isOrthographicCamera===!0),I!==L&&(I=L,fn=!0,Ss=!0)}if(E.isSkinnedMesh){wt.setOptional(P,E,"bindMatrix"),wt.setOptional(P,E,"bindMatrixInverse");const Zt=E.skeleton;Zt&&(Zt.boneTexture===null&&Zt.computeBoneTexture(),wt.setValue(P,"boneTexture",Zt.boneTexture,ge))}E.isBatchedMesh&&(wt.setOptional(P,E,"batchingTexture"),wt.setValue(P,"batchingTexture",E._matricesTexture,ge),wt.setOptional(P,E,"batchingIdTexture"),wt.setValue(P,"batchingIdTexture",E._indirectTexture,ge),wt.setOptional(P,E,"batchingColorTexture"),E._colorsTexture!==null&&wt.setValue(P,"batchingColorTexture",E._colorsTexture,ge));const qt=oe.morphAttributes;if((qt.position!==void 0||qt.normal!==void 0||qt.color!==void 0)&&Te.update(E,oe,Mn),(fn||Ge.receiveShadow!==E.receiveShadow)&&(Ge.receiveShadow=E.receiveShadow,wt.setValue(P,"receiveShadow",E.receiveShadow)),w.isMeshGouraudMaterial&&w.envMap!==null&&(dn.envMap.value=xe,dn.flipEnvMap.value=xe.isCubeTexture&&xe.isRenderTargetTexture===!1?-1:1),w.isMeshStandardMaterial&&w.envMap===null&&K.environment!==null&&(dn.envMapIntensity.value=K.environmentIntensity),dn.dfgLUT!==void 0&&(dn.dfgLUT.value=kT()),fn&&(wt.setValue(P,"toneMappingExposure",b.toneMappingExposure),Ge.needsLights&&As(dn,Ss),Q&&w.fog===!0&&Be.refreshFogUniforms(dn,Q),Be.refreshMaterialUniforms(dn,w,ae,Y,v.state.transmissionRenderTarget[L.id]),Vl.upload(P,pt(Ge),dn,ge)),w.isShaderMaterial&&w.uniformsNeedUpdate===!0&&(Vl.upload(P,pt(Ge),dn,ge),w.uniformsNeedUpdate=!1),w.isSpriteMaterial&&wt.setValue(P,"center",E.center),wt.setValue(P,"modelViewMatrix",E.modelViewMatrix),wt.setValue(P,"normalMatrix",E.normalMatrix),wt.setValue(P,"modelMatrix",E.matrixWorld),w.isShaderMaterial||w.isRawShaderMaterial){const Zt=w.uniformsGroups;for(let on=0,Js=Zt.length;on<Js;on++){const qi=Zt[on];De.update(qi,Mn),De.bind(qi,Mn)}}return Mn}function As(L,K){L.ambientLightColor.needsUpdate=K,L.lightProbe.needsUpdate=K,L.directionalLights.needsUpdate=K,L.directionalLightShadows.needsUpdate=K,L.pointLights.needsUpdate=K,L.pointLightShadows.needsUpdate=K,L.spotLights.needsUpdate=K,L.spotLightShadows.needsUpdate=K,L.rectAreaLights.needsUpdate=K,L.hemisphereLights.needsUpdate=K}function Go(L){return L.isMeshLambertMaterial||L.isMeshToonMaterial||L.isMeshPhongMaterial||L.isMeshStandardMaterial||L.isShadowMaterial||L.isShaderMaterial&&L.lights===!0}this.getActiveCubeFace=function(){return R},this.getActiveMipmapLevel=function(){return y},this.getRenderTarget=function(){return C},this.setRenderTargetTextures=function(L,K,oe){const w=te.get(L);w.__autoAllocateDepthBuffer=L.resolveDepthBuffer===!1,w.__autoAllocateDepthBuffer===!1&&(w.__useRenderToTexture=!1),te.get(L.texture).__webglTexture=K,te.get(L.depthTexture).__webglTexture=w.__autoAllocateDepthBuffer?void 0:oe,w.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(L,K){const oe=te.get(L);oe.__webglFramebuffer=K,oe.__useDefaultFramebuffer=K===void 0};const Wo=P.createFramebuffer();this.setRenderTarget=function(L,K=0,oe=0){C=L,R=K,y=oe;let w=!0,E=null,Q=!1,ue=!1;if(L){const xe=te.get(L);if(xe.__useDefaultFramebuffer!==void 0)ne.bindFramebuffer(P.FRAMEBUFFER,null),w=!1;else if(xe.__webglFramebuffer===void 0)ge.setupRenderTarget(L);else if(xe.__hasExternalTextures)ge.rebindTextures(L,te.get(L.texture).__webglTexture,te.get(L.depthTexture).__webglTexture);else if(L.depthBuffer){const _e=L.depthTexture;if(xe.__boundDepthTexture!==_e){if(_e!==null&&te.has(_e)&&(L.width!==_e.image.width||L.height!==_e.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");ge.setupDepthRenderbuffer(L)}}const Ee=L.texture;(Ee.isData3DTexture||Ee.isDataArrayTexture||Ee.isCompressedArrayTexture)&&(ue=!0);const we=te.get(L).__webglFramebuffer;L.isWebGLCubeRenderTarget?(Array.isArray(we[K])?E=we[K][oe]:E=we[K],Q=!0):L.samples>0&&ge.useMultisampledRTT(L)===!1?E=te.get(L).__webglMultisampledFramebuffer:Array.isArray(we)?E=we[oe]:E=we,O.copy(L.viewport),z.copy(L.scissor),H=L.scissorTest}else O.copy(Ne).multiplyScalar(ae).floor(),z.copy(We).multiplyScalar(ae).floor(),H=He;if(oe!==0&&(E=Wo),ne.bindFramebuffer(P.FRAMEBUFFER,E)&&w&&ne.drawBuffers(L,E),ne.viewport(O),ne.scissor(z),ne.setScissorTest(H),Q){const xe=te.get(L.texture);P.framebufferTexture2D(P.FRAMEBUFFER,P.COLOR_ATTACHMENT0,P.TEXTURE_CUBE_MAP_POSITIVE_X+K,xe.__webglTexture,oe)}else if(ue){const xe=K;for(let Ee=0;Ee<L.textures.length;Ee++){const we=te.get(L.textures[Ee]);P.framebufferTextureLayer(P.FRAMEBUFFER,P.COLOR_ATTACHMENT0+Ee,we.__webglTexture,oe,xe)}}else if(L!==null&&oe!==0){const xe=te.get(L.texture);P.framebufferTexture2D(P.FRAMEBUFFER,P.COLOR_ATTACHMENT0,P.TEXTURE_2D,xe.__webglTexture,oe)}F=-1},this.readRenderTargetPixels=function(L,K,oe,w,E,Q,ue,pe=0){if(!(L&&L.isWebGLRenderTarget)){Xt("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let xe=te.get(L).__webglFramebuffer;if(L.isWebGLCubeRenderTarget&&ue!==void 0&&(xe=xe[ue]),xe){ne.bindFramebuffer(P.FRAMEBUFFER,xe);try{const Ee=L.textures[pe],we=Ee.format,_e=Ee.type;if(!fe.textureFormatReadable(we)){Xt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!fe.textureTypeReadable(_e)){Xt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}K>=0&&K<=L.width-w&&oe>=0&&oe<=L.height-E&&(L.textures.length>1&&P.readBuffer(P.COLOR_ATTACHMENT0+pe),P.readPixels(K,oe,w,E,Ye.convert(we),Ye.convert(_e),Q))}finally{const Ee=C!==null?te.get(C).__webglFramebuffer:null;ne.bindFramebuffer(P.FRAMEBUFFER,Ee)}}},this.readRenderTargetPixelsAsync=async function(L,K,oe,w,E,Q,ue,pe=0){if(!(L&&L.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let xe=te.get(L).__webglFramebuffer;if(L.isWebGLCubeRenderTarget&&ue!==void 0&&(xe=xe[ue]),xe)if(K>=0&&K<=L.width-w&&oe>=0&&oe<=L.height-E){ne.bindFramebuffer(P.FRAMEBUFFER,xe);const Ee=L.textures[pe],we=Ee.format,_e=Ee.type;if(!fe.textureFormatReadable(we))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!fe.textureTypeReadable(_e))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const Le=P.createBuffer();P.bindBuffer(P.PIXEL_PACK_BUFFER,Le),P.bufferData(P.PIXEL_PACK_BUFFER,Q.byteLength,P.STREAM_READ),L.textures.length>1&&P.readBuffer(P.COLOR_ATTACHMENT0+pe),P.readPixels(K,oe,w,E,Ye.convert(we),Ye.convert(_e),0);const Ve=C!==null?te.get(C).__webglFramebuffer:null;ne.bindFramebuffer(P.FRAMEBUFFER,Ve);const tt=P.fenceSync(P.SYNC_GPU_COMMANDS_COMPLETE,0);return P.flush(),await JA(P,tt,4),P.bindBuffer(P.PIXEL_PACK_BUFFER,Le),P.getBufferSubData(P.PIXEL_PACK_BUFFER,0,Q),P.deleteBuffer(Le),P.deleteSync(tt),Q}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(L,K=null,oe=0){const w=Math.pow(2,-oe),E=Math.floor(L.image.width*w),Q=Math.floor(L.image.height*w),ue=K!==null?K.x:0,pe=K!==null?K.y:0;ge.setTexture2D(L,0),P.copyTexSubImage2D(P.TEXTURE_2D,oe,0,0,ue,pe,E,Q),ne.unbindTexture()};const Pr=P.createFramebuffer(),Fc=P.createFramebuffer();this.copyTextureToTexture=function(L,K,oe=null,w=null,E=0,Q=null){Q===null&&(E!==0?(Da("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."),Q=E,E=0):Q=0);let ue,pe,xe,Ee,we,_e,Le,Ve,tt;const rt=L.isCompressedTexture?L.mipmaps[Q]:L.image;if(oe!==null)ue=oe.max.x-oe.min.x,pe=oe.max.y-oe.min.y,xe=oe.isBox3?oe.max.z-oe.min.z:1,Ee=oe.min.x,we=oe.min.y,_e=oe.isBox3?oe.min.z:0;else{const qt=Math.pow(2,-E);ue=Math.floor(rt.width*qt),pe=Math.floor(rt.height*qt),L.isDataArrayTexture?xe=rt.depth:L.isData3DTexture?xe=Math.floor(rt.depth*qt):xe=1,Ee=0,we=0,_e=0}w!==null?(Le=w.x,Ve=w.y,tt=w.z):(Le=0,Ve=0,tt=0);const Ke=Ye.convert(K.format),Ge=Ye.convert(K.type);let ut;K.isData3DTexture?(ge.setTexture3D(K,0),ut=P.TEXTURE_3D):K.isDataArrayTexture||K.isCompressedArrayTexture?(ge.setTexture2DArray(K,0),ut=P.TEXTURE_2D_ARRAY):(ge.setTexture2D(K,0),ut=P.TEXTURE_2D),P.pixelStorei(P.UNPACK_FLIP_Y_WEBGL,K.flipY),P.pixelStorei(P.UNPACK_PREMULTIPLY_ALPHA_WEBGL,K.premultiplyAlpha),P.pixelStorei(P.UNPACK_ALIGNMENT,K.unpackAlignment);const ot=P.getParameter(P.UNPACK_ROW_LENGTH),Mn=P.getParameter(P.UNPACK_IMAGE_HEIGHT),_i=P.getParameter(P.UNPACK_SKIP_PIXELS),fn=P.getParameter(P.UNPACK_SKIP_ROWS),Ss=P.getParameter(P.UNPACK_SKIP_IMAGES);P.pixelStorei(P.UNPACK_ROW_LENGTH,rt.width),P.pixelStorei(P.UNPACK_IMAGE_HEIGHT,rt.height),P.pixelStorei(P.UNPACK_SKIP_PIXELS,Ee),P.pixelStorei(P.UNPACK_SKIP_ROWS,we),P.pixelStorei(P.UNPACK_SKIP_IMAGES,_e);const wt=L.isDataArrayTexture||L.isData3DTexture,dn=K.isDataArrayTexture||K.isData3DTexture;if(L.isDepthTexture){const qt=te.get(L),Zt=te.get(K),on=te.get(qt.__renderTarget),Js=te.get(Zt.__renderTarget);ne.bindFramebuffer(P.READ_FRAMEBUFFER,on.__webglFramebuffer),ne.bindFramebuffer(P.DRAW_FRAMEBUFFER,Js.__webglFramebuffer);for(let qi=0;qi<xe;qi++)wt&&(P.framebufferTextureLayer(P.READ_FRAMEBUFFER,P.COLOR_ATTACHMENT0,te.get(L).__webglTexture,E,_e+qi),P.framebufferTextureLayer(P.DRAW_FRAMEBUFFER,P.COLOR_ATTACHMENT0,te.get(K).__webglTexture,Q,tt+qi)),P.blitFramebuffer(Ee,we,ue,pe,Le,Ve,ue,pe,P.DEPTH_BUFFER_BIT,P.NEAREST);ne.bindFramebuffer(P.READ_FRAMEBUFFER,null),ne.bindFramebuffer(P.DRAW_FRAMEBUFFER,null)}else if(E!==0||L.isRenderTargetTexture||te.has(L)){const qt=te.get(L),Zt=te.get(K);ne.bindFramebuffer(P.READ_FRAMEBUFFER,Pr),ne.bindFramebuffer(P.DRAW_FRAMEBUFFER,Fc);for(let on=0;on<xe;on++)wt?P.framebufferTextureLayer(P.READ_FRAMEBUFFER,P.COLOR_ATTACHMENT0,qt.__webglTexture,E,_e+on):P.framebufferTexture2D(P.READ_FRAMEBUFFER,P.COLOR_ATTACHMENT0,P.TEXTURE_2D,qt.__webglTexture,E),dn?P.framebufferTextureLayer(P.DRAW_FRAMEBUFFER,P.COLOR_ATTACHMENT0,Zt.__webglTexture,Q,tt+on):P.framebufferTexture2D(P.DRAW_FRAMEBUFFER,P.COLOR_ATTACHMENT0,P.TEXTURE_2D,Zt.__webglTexture,Q),E!==0?P.blitFramebuffer(Ee,we,ue,pe,Le,Ve,ue,pe,P.COLOR_BUFFER_BIT,P.NEAREST):dn?P.copyTexSubImage3D(ut,Q,Le,Ve,tt+on,Ee,we,ue,pe):P.copyTexSubImage2D(ut,Q,Le,Ve,Ee,we,ue,pe);ne.bindFramebuffer(P.READ_FRAMEBUFFER,null),ne.bindFramebuffer(P.DRAW_FRAMEBUFFER,null)}else dn?L.isDataTexture||L.isData3DTexture?P.texSubImage3D(ut,Q,Le,Ve,tt,ue,pe,xe,Ke,Ge,rt.data):K.isCompressedArrayTexture?P.compressedTexSubImage3D(ut,Q,Le,Ve,tt,ue,pe,xe,Ke,rt.data):P.texSubImage3D(ut,Q,Le,Ve,tt,ue,pe,xe,Ke,Ge,rt):L.isDataTexture?P.texSubImage2D(P.TEXTURE_2D,Q,Le,Ve,ue,pe,Ke,Ge,rt.data):L.isCompressedTexture?P.compressedTexSubImage2D(P.TEXTURE_2D,Q,Le,Ve,rt.width,rt.height,Ke,rt.data):P.texSubImage2D(P.TEXTURE_2D,Q,Le,Ve,ue,pe,Ke,Ge,rt);P.pixelStorei(P.UNPACK_ROW_LENGTH,ot),P.pixelStorei(P.UNPACK_IMAGE_HEIGHT,Mn),P.pixelStorei(P.UNPACK_SKIP_PIXELS,_i),P.pixelStorei(P.UNPACK_SKIP_ROWS,fn),P.pixelStorei(P.UNPACK_SKIP_IMAGES,Ss),Q===0&&K.generateMipmaps&&P.generateMipmap(ut),ne.unbindTexture()},this.initRenderTarget=function(L){te.get(L).__webglFramebuffer===void 0&&ge.setupRenderTarget(L)},this.initTexture=function(L){L.isCubeTexture?ge.setTextureCube(L,0):L.isData3DTexture?ge.setTexture3D(L,0):L.isDataArrayTexture||L.isCompressedArrayTexture?ge.setTexture2DArray(L,0):ge.setTexture2D(L,0),ne.unbindTexture()},this.resetState=function(){R=0,y=0,C=null,ne.reset(),W.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Li}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const t=this.getContext();t.drawingBufferColorSpace=dt._getDrawingBufferColorSpace(e),t.unpackColorSpace=dt._getUnpackColorSpace()}}class Bs{static idGen=0;constructor(e,t){let n,s;this.promise=new Promise((c,u)=>{n=c,s=u});const r=n.bind(this),o=s.bind(this),a=(...c)=>{r(...c)},l=c=>{o(c)};e(a.bind(this),l.bind(this)),this.abortHandler=t,this.id=Bs.idGen++}then(e){return new Bs((t,n)=>{this.promise=this.promise.then((...s)=>{const r=e(...s);r instanceof Promise||r instanceof Bs?r.then((...o)=>{t(...o)}):t(r)}).catch(s=>{n(s)})},this.abortHandler)}catch(e){return new Bs(t=>{this.promise=this.promise.then((...n)=>{t(...n)}).catch(e)},this.abortHandler)}abort(e){this.abortHandler&&this.abortHandler(e)}}class Tg extends Error{constructor(e){super(e)}}(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){i[0]=t;const n=e[0];let s=n>>16&32768,r=n>>12&2047;const o=n>>23&255;return o<103?s:o>142?(s|=31744,s|=(o==255?0:1)&&n&8388607,s):o<113?(r|=2048,s|=(r>>114-o)+(r>>113-o&1),s):(s|=o-112<<10|r>>1,s+=r&1,s)}})();const yu=(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){return i[0]=t,e[0]}})(),VT=function(i,e){return i[e]+(i[e+1]<<8)+(i[e+2]<<16)+(i[e+3]<<24)},wc=function(i,e,t=!0,n){const s=new AbortController,r=s.signal;let o=!1;const a=u=>{s.abort(u),o=!0};let l=!1;const c=(u,f,d,h)=>{e&&!l&&(e(u,f,d,h),u===100&&(l=!0))};return new Bs((u,f)=>{const d={signal:r};n&&(d.headers=n),fetch(i,d).then(async h=>{if(!h.ok){const v=await h.text();f(new Error(`Fetch failed: ${h.status} ${h.statusText} ${v}`));return}const x=h.body.getReader();let p=0,g=h.headers.get("Content-Length"),m=g?parseInt(g):void 0;const _=[];for(;!o;)try{const{value:v,done:A}=await x.read();if(A){if(c(100,"100%",v,m),t){const M=new Blob(_).arrayBuffer();u(M)}else u();break}p+=v.length;let S,b;m!==void 0&&(S=p/m*100,b=`${S.toFixed(2)}%`),t&&_.push(v),c(S,b,v,m)}catch(v){f(v);return}}).catch(h=>{f(new Tg(h))})},a)},Bt=function(i,e,t){return Math.max(Math.min(i,t),e)},jr=function(){return performance.now()/1e3},no=i=>{if(i.geometry&&(i.geometry.dispose(),i.geometry=null),i.material&&(i.material.dispose(),i.material=null),i.children)for(let e of i.children)no(e)},Zn=(i,e)=>new Promise(t=>{window.setTimeout(()=>{t(i?i():void 0)},e?1:50)}),mo=(i=0)=>{let e=0;if(i===1)e=9;else if(i===2)e=24;else if(i===3)e=45;else if(i>3)throw new Error("getSphericalHarmonicsComponentCountForDegree() -> Invalid spherical harmonics degree");return e},Ld=()=>{let i,e;return{promise:new Promise((n,s)=>{i=n,e=s}),resolve:i,reject:e}},bu=i=>{let e,t;return i||(i=()=>{}),{promise:new Bs((s,r)=>{e=s,t=r},i),resolve:e,reject:t}};class GT{constructor(e,t,n){this.major=e,this.minor=t,this.patch=n}toString(){return`${this.major}_${this.minor}_${this.patch}`}}function Bd(){const i=navigator.userAgent;return i.indexOf("iPhone")>0||i.indexOf("iPad")>0}function Eg(){if(Bd()){const i=navigator.userAgent.match(/OS (\d+)_(\d+)_?(\d+)?/);return new GT(parseInt(i[1]||0,10),parseInt(i[2]||0,10),parseInt(i[3]||0,10))}else return null}const WT=14;class Ie{static OFFSET={X:0,Y:1,Z:2,SCALE0:3,SCALE1:4,SCALE2:5,ROTATION0:6,ROTATION1:7,ROTATION2:8,ROTATION3:9,FDC0:10,FDC1:11,FDC2:12,OPACITY:13,FRC0:14,FRC1:15,FRC2:16,FRC3:17,FRC4:18,FRC5:19,FRC6:20,FRC7:21,FRC8:22,FRC9:23,FRC10:24,FRC11:25,FRC12:26,FRC13:27,FRC14:28,FRC15:29,FRC16:30,FRC17:31,FRC18:32,FRC19:33,FRC20:34,FRC21:35,FRC22:36,FRC23:37};constructor(e=0){this.sphericalHarmonicsDegree=e,this.sphericalHarmonicsCount=mo(this.sphericalHarmonicsDegree),this.componentCount=this.sphericalHarmonicsCount+WT,this.defaultSphericalHarmonics=new Array(this.sphericalHarmonicsCount).fill(0),this.splats=[],this.splatCount=0}static createSplat(e=0){const t=[0,0,0,1,1,1,1,0,0,0,0,0,0,0];let n=mo(e);for(let s=0;s<n;s++)t.push(0);return t}addSplat(e){this.splats.push(e),this.splatCount++}getSplat(e){return this.splats[e]}addDefaultSplat(){const e=Ie.createSplat(this.sphericalHarmonicsDegree);return this.addSplat(e),e}addSplatFromComonents(e,t,n,s,r,o,a,l,c,u,f,d,h,x,...p){const g=[e,t,n,s,r,o,a,l,c,u,f,d,h,x,...this.defaultSphericalHarmonics];for(let m=0;m<p.length&&m<this.sphericalHarmonicsCount;m++)g[m]=p[m];return this.addSplat(g),g}addSplatFromArray(e,t){const n=e.splats[t],s=Ie.createSplat(this.sphericalHarmonicsDegree);for(let r=0;r<this.componentCount&&r<n.length;r++)s[r]=n[r];this.addSplat(s)}}class bt{static DefaultSplatSortDistanceMapPrecision=16;static MemoryPageSize=65536;static BytesPerFloat=4;static BytesPerInt=4;static MaxScenes=32;static ProgressiveLoadSectionSize=262144;static ProgressiveLoadSectionDelayDuration=15;static SphericalHarmonics8BitCompressionRange=3}const XT=bt.SphericalHarmonics8BitCompressionRange,ws=XT/2,Jt=Fa.toHalfFloat.bind(Fa),Ud=Fa.fromHalfFloat.bind(Fa),Lt=(i,e,t=!1,n,s)=>{if(e===0)return i;if(e===1||e===2&&!t)return Fa.fromHalfFloat(i);if(e===2)return Od(i,n,s)},ha=(i,e,t)=>{i=Bt(i,e,t);const n=t-e;return Bt(Math.floor((i-e)/n*255),0,255)},Od=(i,e,t)=>{const n=t-e;return i/255*n+e},wg=(i,e,t)=>ha(Ud(i,e,t)),qT=(i,e,t)=>Jt(Od(i,e,t)),xt=(i,e,t,n=!1)=>t===0?i.getFloat32(e*4,!0):t===1||t===2&&!n?i.getUint16(e*2,!0):i.getUint8(e,!0),YT=(function(){const i=e=>e;return function(e,t,n,s=!1){if(t===n)return e;let r=i;return t===2&&s?n===1?r=qT:n==0&&(r=Od):t===2||t===1?n===0?r=Ud:n==2&&(s?r=wg:r=i):t===0&&(n===1?r=Jt:n==2&&(s?r=ha:r=Jt)),r(e)}})(),$r=(i,e,t,n,s=0)=>{const r=new Uint8Array(i,e),o=new Uint8Array(t,n);for(let a=0;a<s;a++)o[a]=r[a]};class ${static CurrentMajorVersion=0;static CurrentMinorVersion=1;static CenterComponentCount=3;static ScaleComponentCount=3;static RotationComponentCount=4;static ColorComponentCount=4;static CovarianceComponentCount=6;static SplatScaleOffsetFloat=3;static SplatRotationOffsetFloat=6;static CompressionLevels={0:{BytesPerCenter:12,BytesPerScale:12,BytesPerRotation:16,BytesPerColor:4,ScaleOffsetBytes:12,RotationffsetBytes:24,ColorOffsetBytes:40,SphericalHarmonicsOffsetBytes:44,ScaleRange:1,BytesPerSphericalHarmonicsComponent:4,SphericalHarmonicsOffsetFloat:11,SphericalHarmonicsDegrees:{0:{BytesPerSplat:44},1:{BytesPerSplat:80},2:{BytesPerSplat:140}}},1:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:2,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:42},2:{BytesPerSplat:72}}},2:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:1,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:33},2:{BytesPerSplat:48}}}};static CovarianceSizeFloats=6;static HeaderSizeBytes=4096;static SectionHeaderSizeBytes=1024;static BucketStorageSizeBytes=12;static BucketStorageSizeFloats=3;static BucketBlockSize=5;static BucketSize=256;constructor(e,t=!0){this.constructFromBuffer(e,t)}getSplatCount(){return this.splatCount}getMaxSplatCount(){return this.maxSplatCount}getMinSphericalHarmonicsDegree(){let e=0;for(let t=0;t<this.sections.length;t++){const n=this.sections[t];(t===0||n.sphericalHarmonicsDegree<e)&&(e=n.sphericalHarmonicsDegree)}return e}getBucketIndex(e,t){let n;const s=e.fullBucketCount*e.bucketSize;if(t<s)n=Math.floor(t/e.bucketSize);else{let r=s;n=e.fullBucketCount;let o=0;for(;r<e.splatCount;){let a=e.partiallyFilledBucketLengths[o];if(t>=r&&t<r+a)break;r+=a,n++,o++}}return n}getSplatCenter(e,t,n){const s=this.globalSplatIndexToSectionMap[e],r=this.sections[s],o=e-r.splatCountOffset,a=r.bytesPerSplat*o,l=new DataView(this.bufferData,r.dataBase+a),c=xt(l,0,this.compressionLevel),u=xt(l,1,this.compressionLevel),f=xt(l,2,this.compressionLevel);if(this.compressionLevel>=1){const h=this.getBucketIndex(r,o)*$.BucketStorageSizeFloats,x=r.compressionScaleFactor,p=r.compressionScaleRange;t.x=(c-p)*x+r.bucketArray[h],t.y=(u-p)*x+r.bucketArray[h+1],t.z=(f-p)*x+r.bucketArray[h+2]}else t.x=c,t.y=u,t.z=f;n&&t.applyMatrix4(n)}getSplatScaleAndRotation=(function(){const e=new $e,t=new $e,n=new $e,s=new U,r=new U,o=new Ft;return function(a,l,c,u,f){const d=this.globalSplatIndexToSectionMap[a],h=this.sections[d],x=a-h.splatCountOffset,p=h.bytesPerSplat*x+$.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,g=new DataView(this.bufferData,h.dataBase+p);r.set(Lt(xt(g,0,this.compressionLevel),this.compressionLevel),Lt(xt(g,1,this.compressionLevel),this.compressionLevel),Lt(xt(g,2,this.compressionLevel),this.compressionLevel)),f&&(f.x!==void 0&&(r.x=f.x),f.y!==void 0&&(r.y=f.y),f.z!==void 0&&(r.z=f.z)),o.set(Lt(xt(g,4,this.compressionLevel),this.compressionLevel),Lt(xt(g,5,this.compressionLevel),this.compressionLevel),Lt(xt(g,6,this.compressionLevel),this.compressionLevel),Lt(xt(g,3,this.compressionLevel),this.compressionLevel)),u?(e.makeScale(r.x,r.y,r.z),t.makeRotationFromQuaternion(o),n.copy(e).multiply(t).multiply(u),n.decompose(s,c,l)):(l.copy(r),c.copy(o))}})();getSplatColor(e,t){const n=this.globalSplatIndexToSectionMap[e],s=this.sections[n],r=e-s.splatCountOffset,o=s.bytesPerSplat*r+$.CompressionLevels[this.compressionLevel].ColorOffsetBytes,a=new Uint8Array(this.bufferData,s.dataBase+o,4);t.set(a[0],a[1],a[2],a[3])}fillSplatCenterArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);const a=new U;for(let l=n;l<=s;l++){const c=this.globalSplatIndexToSectionMap[l],u=this.sections[c],f=l-u.splatCountOffset,d=(l-n+r)*$.CenterComponentCount,h=u.bytesPerSplat*f,x=new DataView(this.bufferData,u.dataBase+h),p=xt(x,0,this.compressionLevel),g=xt(x,1,this.compressionLevel),m=xt(x,2,this.compressionLevel);if(this.compressionLevel>=1){const v=this.getBucketIndex(u,f)*$.BucketStorageSizeFloats,A=u.compressionScaleFactor,S=u.compressionScaleRange;a.x=(p-S)*A+u.bucketArray[v],a.y=(g-S)*A+u.bucketArray[v+1],a.z=(m-S)*A+u.bucketArray[v+2]}else a.x=p,a.y=g,a.z=m;t&&a.applyMatrix4(t),e[d]=a.x,e[d+1]=a.y,e[d+2]=a.z}}fillSplatScaleRotationArray=(function(){const e=new $e,t=new $e,n=new $e,s=new U,r=new Ft,o=new U,a=l=>{const c=l.w<0?-1:1;l.x*=c,l.y*=c,l.z*=c,l.w*=c};return function(l,c,u,f,d,h,x,p){const g=this.splatCount;f=f||0,d=d||g-1,h===void 0&&(h=f);const m=(_,v)=>YT(_,v,x);for(let _=f;_<=d;_++){const v=this.globalSplatIndexToSectionMap[_],A=this.sections[v],S=_-A.splatCountOffset,b=A.bytesPerSplat*S+$.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,M=(_-f+h)*$.ScaleComponentCount,R=(_-f+h)*$.RotationComponentCount,y=new DataView(this.bufferData,A.dataBase+b),C=p&&p.x!==void 0?p.x:xt(y,0,this.compressionLevel),F=p&&p.y!==void 0?p.y:xt(y,1,this.compressionLevel),I=p&&p.z!==void 0?p.z:xt(y,2,this.compressionLevel),O=xt(y,3,this.compressionLevel),z=xt(y,4,this.compressionLevel),H=xt(y,5,this.compressionLevel),V=xt(y,6,this.compressionLevel);s.set(Lt(C,this.compressionLevel),Lt(F,this.compressionLevel),Lt(I,this.compressionLevel)),r.set(Lt(z,this.compressionLevel),Lt(H,this.compressionLevel),Lt(V,this.compressionLevel),Lt(O,this.compressionLevel)).normalize(),u&&(o.set(0,0,0),e.makeScale(s.x,s.y,s.z),t.makeRotationFromQuaternion(r),n.identity().premultiply(e).premultiply(t),n.premultiply(u),n.decompose(o,r,s),r.normalize()),a(r),l&&(l[M]=m(s.x,0),l[M+1]=m(s.y,0),l[M+2]=m(s.z,0)),c&&(c[R]=m(r.x,0),c[R+1]=m(r.y,0),c[R+2]=m(r.z,0),c[R+3]=m(r.w,0))}}})();static computeCovariance=(function(){const e=new $e,t=new Ze,n=new Ze,s=new Ze,r=new Ze,o=new Ze,a=new Ze;return function(l,c,u,f,d=0,h){e.makeScale(l.x,l.y,l.z),t.setFromMatrix4(e),e.makeRotationFromQuaternion(c),n.setFromMatrix4(e),s.copy(n).multiply(t),r.copy(s).transpose().premultiply(s),u&&(o.setFromMatrix4(u),a.copy(o).transpose(),r.multiply(a),r.premultiply(o)),h>=1?(f[d]=Jt(r.elements[0]),f[d+1]=Jt(r.elements[3]),f[d+2]=Jt(r.elements[6]),f[d+3]=Jt(r.elements[4]),f[d+4]=Jt(r.elements[7]),f[d+5]=Jt(r.elements[8])):(f[d]=r.elements[0],f[d+1]=r.elements[3],f[d+2]=r.elements[6],f[d+3]=r.elements[4],f[d+4]=r.elements[7],f[d+5]=r.elements[8])}})();fillSplatCovarianceArray(e,t,n,s,r,o){const a=this.splatCount,l=new U,c=new Ft;n=n||0,s=s||a-1,r===void 0&&(r=n);for(let u=n;u<=s;u++){const f=this.globalSplatIndexToSectionMap[u],d=this.sections[f],h=u-d.splatCountOffset,x=(u-n+r)*$.CovarianceComponentCount,p=d.bytesPerSplat*h+$.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,g=new DataView(this.bufferData,d.dataBase+p);l.set(Lt(xt(g,0,this.compressionLevel),this.compressionLevel),Lt(xt(g,1,this.compressionLevel),this.compressionLevel),Lt(xt(g,2,this.compressionLevel),this.compressionLevel)),c.set(Lt(xt(g,4,this.compressionLevel),this.compressionLevel),Lt(xt(g,5,this.compressionLevel),this.compressionLevel),Lt(xt(g,6,this.compressionLevel),this.compressionLevel),Lt(xt(g,3,this.compressionLevel),this.compressionLevel)),$.computeCovariance(l,c,t,e,x,o)}}fillSplatColorArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);for(let a=n;a<=s;a++){const l=this.globalSplatIndexToSectionMap[a],c=this.sections[l],u=a-c.splatCountOffset,f=(a-n+r)*$.ColorComponentCount,d=c.bytesPerSplat*u+$.CompressionLevels[this.compressionLevel].ColorOffsetBytes,h=new Uint8Array(this.bufferData,c.dataBase+d);let x=h[3];x=x>=t?x:0,e[f]=h[0],e[f+1]=h[1],e[f+2]=h[2],e[f+3]=x}}fillSphericalHarmonicsArray=(function(){for(let z=0;z<15;z++)new U;const e=new Ze,t=new $e,n=new U,s=new U,r=new Ft,o=[],a=[],l=[],c=[],u=[],f=[],d=[],h=[],x=[],p=[],g=[],m=[],_=[],v=[],A=[],S=[],b=[],M=[],R=z=>z,y=(z,H,V,q)=>{z[0]=H,z[1]=V,z[2]=q},C=(z,H,V,q,G)=>{z[0]=xt(H,q,G,!0),z[1]=xt(H,q+V,G,!0),z[2]=xt(H,q+V+V,G,!0)},F=(z,H)=>{H[0]=z[0],H[1]=z[1],H[2]=z[2]},I=(z,H,V,q)=>{H[V]=q(z[0]),H[V+1]=q(z[1]),H[V+2]=q(z[2])},O=(z,H,V,q,G)=>(H[0]=Lt(z[0],V,!0,q,G),H[1]=Lt(z[1],V,!0,q,G),H[2]=Lt(z[2],V,!0,q,G),H);return function(z,H,V,q,G,Y,ae){const Ae=this.splatCount;q=q||0,G=G||Ae-1,Y===void 0&&(Y=q),V&&H>=1&&(t.copy(V),t.decompose(n,r,s),r.normalize(),t.makeRotationFromQuaternion(r),e.setFromMatrix4(t),y(o,e.elements[4],-e.elements[7],e.elements[1]),y(a,-e.elements[5],e.elements[8],-e.elements[2]),y(l,e.elements[3],-e.elements[6],e.elements[0]));const Pe=We=>wg(We,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff),Ne=We=>ha(We,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff);for(let We=q;We<=G;We++){const He=this.globalSplatIndexToSectionMap[We],J=this.sections[He];H=Math.min(H,J.sphericalHarmonicsDegree);const le=mo(H),be=We-J.splatCountOffset,ze=J.bytesPerSplat*be+$.CompressionLevels[this.compressionLevel].SphericalHarmonicsOffsetBytes,Ce=new DataView(this.bufferData,J.dataBase+ze),Qe=(We-q+Y)*le;let B=V?0:this.compressionLevel,N=R;B!==ae&&(B===1?ae===0?N=Ud:ae==2&&(N=Pe):B===0&&(ae===1?N=Jt:ae==2&&(N=Ne)));const j=this.minSphericalHarmonicsCoeff,P=this.maxSphericalHarmonicsCoeff;H>=1&&(C(x,Ce,3,0,this.compressionLevel),C(p,Ce,3,1,this.compressionLevel),C(g,Ce,3,2,this.compressionLevel),V?(O(x,x,this.compressionLevel,j,P),O(p,p,this.compressionLevel,j,P),O(g,g,this.compressionLevel,j,P),$.rotateSphericalHarmonics3(x,p,g,o,a,l,v,A,S)):(F(x,v),F(p,A),F(g,S)),I(v,z,Qe,N),I(A,z,Qe+3,N),I(S,z,Qe+6,N),H>=2&&(C(x,Ce,5,9,this.compressionLevel),C(p,Ce,5,10,this.compressionLevel),C(g,Ce,5,11,this.compressionLevel),C(m,Ce,5,12,this.compressionLevel),C(_,Ce,5,13,this.compressionLevel),V?(O(x,x,this.compressionLevel,j,P),O(p,p,this.compressionLevel,j,P),O(g,g,this.compressionLevel,j,P),O(m,m,this.compressionLevel,j,P),O(_,_,this.compressionLevel,j,P),$.rotateSphericalHarmonics5(x,p,g,m,_,o,a,l,c,u,f,d,h,v,A,S,b,M)):(F(x,v),F(p,A),F(g,S),F(m,b),F(_,M)),I(v,z,Qe+9,N),I(A,z,Qe+12,N),I(S,z,Qe+15,N),I(b,z,Qe+18,N),I(M,z,Qe+21,N)))}}})();static dot3=(e,t,n,s,r)=>{r[0]=r[1]=r[2]=0;const o=s[0],a=s[1],l=s[2];$.addInto3(e[0]*o,e[1]*o,e[2]*o,r),$.addInto3(t[0]*a,t[1]*a,t[2]*a,r),$.addInto3(n[0]*l,n[1]*l,n[2]*l,r)};static addInto3=(e,t,n,s)=>{s[0]=s[0]+e,s[1]=s[1]+t,s[2]=s[2]+n};static dot5=(e,t,n,s,r,o,a)=>{a[0]=a[1]=a[2]=0;const l=o[0],c=o[1],u=o[2],f=o[3],d=o[4];$.addInto3(e[0]*l,e[1]*l,e[2]*l,a),$.addInto3(t[0]*c,t[1]*c,t[2]*c,a),$.addInto3(n[0]*u,n[1]*u,n[2]*u,a),$.addInto3(s[0]*f,s[1]*f,s[2]*f,a),$.addInto3(r[0]*d,r[1]*d,r[2]*d,a)};static rotateSphericalHarmonics3=(e,t,n,s,r,o,a,l,c)=>{$.dot3(e,t,n,s,a),$.dot3(e,t,n,r,l),$.dot3(e,t,n,o,c)};static rotateSphericalHarmonics5=(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g,m,_)=>{const v=Math.sqrt(.25),A=Math.sqrt(3/4),S=Math.sqrt(1/3),b=Math.sqrt(4/3),M=Math.sqrt(1/12);c[0]=v*(l[2]*o[0]+l[0]*o[2]+(o[2]*l[0]+o[0]*l[2])),c[1]=l[1]*o[0]+o[1]*l[0],c[2]=A*(l[1]*o[1]+o[1]*l[1]),c[3]=l[1]*o[2]+o[1]*l[2],c[4]=v*(l[2]*o[2]-l[0]*o[0]+(o[2]*l[2]-o[0]*l[0])),$.dot5(e,t,n,s,r,c,x),u[0]=v*(a[2]*o[0]+a[0]*o[2]+(o[2]*a[0]+o[0]*a[2])),u[1]=a[1]*o[0]+o[1]*a[0],u[2]=A*(a[1]*o[1]+o[1]*a[1]),u[3]=a[1]*o[2]+o[1]*a[2],u[4]=v*(a[2]*o[2]-a[0]*o[0]+(o[2]*a[2]-o[0]*a[0])),$.dot5(e,t,n,s,r,u,p),f[0]=S*(a[2]*a[0]+a[0]*a[2])+-M*(l[2]*l[0]+l[0]*l[2]+(o[2]*o[0]+o[0]*o[2])),f[1]=b*a[1]*a[0]+-S*(l[1]*l[0]+o[1]*o[0]),f[2]=a[1]*a[1]+-v*(l[1]*l[1]+o[1]*o[1]),f[3]=b*a[1]*a[2]+-S*(l[1]*l[2]+o[1]*o[2]),f[4]=S*(a[2]*a[2]-a[0]*a[0])+-M*(l[2]*l[2]-l[0]*l[0]+(o[2]*o[2]-o[0]*o[0])),$.dot5(e,t,n,s,r,f,g),d[0]=v*(a[2]*l[0]+a[0]*l[2]+(l[2]*a[0]+l[0]*a[2])),d[1]=a[1]*l[0]+l[1]*a[0],d[2]=A*(a[1]*l[1]+l[1]*a[1]),d[3]=a[1]*l[2]+l[1]*a[2],d[4]=v*(a[2]*l[2]-a[0]*l[0]+(l[2]*a[2]-l[0]*a[0])),$.dot5(e,t,n,s,r,d,m),h[0]=v*(l[2]*l[0]+l[0]*l[2]-(o[2]*o[0]+o[0]*o[2])),h[1]=l[1]*l[0]-o[1]*o[0],h[2]=A*(l[1]*l[1]-o[1]*o[1]),h[3]=l[1]*l[2]-o[1]*o[2],h[4]=v*(l[2]*l[2]-l[0]*l[0]-(o[2]*o[2]-o[0]*o[0])),$.dot5(e,t,n,s,r,h,_)};static parseHeader(e){const t=new Uint8Array(e,0,$.HeaderSizeBytes),n=new Uint16Array(e,0,$.HeaderSizeBytes/2),s=new Uint32Array(e,0,$.HeaderSizeBytes/4),r=new Float32Array(e,0,$.HeaderSizeBytes/4),o=t[0],a=t[1],l=s[1],c=s[2],u=s[3],f=s[4],d=n[10],h=new U(r[6],r[7],r[8]),x=r[9]||-ws,p=r[10]||ws;return{versionMajor:o,versionMinor:a,maxSectionCount:l,sectionCount:c,maxSplatCount:u,splatCount:f,compressionLevel:d,sceneCenter:h,minSphericalHarmonicsCoeff:x,maxSphericalHarmonicsCoeff:p}}static writeHeaderCountsToBuffer(e,t,n){const s=new Uint32Array(n,0,$.HeaderSizeBytes/4);s[2]=e,s[4]=t}static writeHeaderToBuffer(e,t){const n=new Uint8Array(t,0,$.HeaderSizeBytes),s=new Uint16Array(t,0,$.HeaderSizeBytes/2),r=new Uint32Array(t,0,$.HeaderSizeBytes/4),o=new Float32Array(t,0,$.HeaderSizeBytes/4);n[0]=e.versionMajor,n[1]=e.versionMinor,n[2]=0,n[3]=0,r[1]=e.maxSectionCount,r[2]=e.sectionCount,r[3]=e.maxSplatCount,r[4]=e.splatCount,s[10]=e.compressionLevel,o[6]=e.sceneCenter.x,o[7]=e.sceneCenter.y,o[8]=e.sceneCenter.z,o[9]=e.minSphericalHarmonicsCoeff||-ws,o[10]=e.maxSphericalHarmonicsCoeff||ws}static parseSectionHeaders(e,t,n=0,s){const r=e.compressionLevel,o=e.maxSectionCount,a=new Uint16Array(t,n,o*$.SectionHeaderSizeBytes/2),l=new Uint32Array(t,n,o*$.SectionHeaderSizeBytes/4),c=new Float32Array(t,n,o*$.SectionHeaderSizeBytes/4),u=[];let f=0,d=f/2,h=f/4,x=$.HeaderSizeBytes+e.maxSectionCount*$.SectionHeaderSizeBytes,p=0;for(let g=0;g<o;g++){const m=l[h+1],_=l[h+2],v=l[h+3],A=c[h+4],S=A/2,b=a[d+10],M=l[h+6]||$.CompressionLevels[r].ScaleRange,R=l[h+8],y=l[h+9],C=y*4,F=b*v+C,I=a[d+20],{bytesPerSplat:O}=$.calculateComponentStorage(r,I),z=O*m,H=z+F,V={bytesPerSplat:O,splatCountOffset:p,splatCount:s?m:0,maxSplatCount:m,bucketSize:_,bucketCount:v,bucketBlockSize:A,halfBucketBlockSize:S,bucketStorageSizeBytes:b,bucketsStorageSizeBytes:F,splatDataStorageSizeBytes:z,storageSizeBytes:H,compressionScaleRange:M,compressionScaleFactor:S/M,base:x,bucketsBase:x+C,dataBase:x+F,fullBucketCount:R,partiallyFilledBucketCount:y,sphericalHarmonicsDegree:I};u[g]=V,x+=H,f+=$.SectionHeaderSizeBytes,d=f/2,h=f/4,p+=m}return u}static writeSectionHeaderToBuffer(e,t,n,s=0){const r=new Uint16Array(n,s,$.SectionHeaderSizeBytes/2),o=new Uint32Array(n,s,$.SectionHeaderSizeBytes/4),a=new Float32Array(n,s,$.SectionHeaderSizeBytes/4);o[0]=e.splatCount,o[1]=e.maxSplatCount,o[2]=t>=1?e.bucketSize:0,o[3]=t>=1?e.bucketCount:0,a[4]=t>=1?e.bucketBlockSize:0,r[10]=t>=1?$.BucketStorageSizeBytes:0,o[6]=t>=1?e.compressionScaleRange:0,o[7]=e.storageSizeBytes,o[8]=t>=1?e.fullBucketCount:0,o[9]=t>=1?e.partiallyFilledBucketCount:0,r[20]=e.sphericalHarmonicsDegree}static writeSectionHeaderSplatCountToBuffer(e,t,n=0){const s=new Uint32Array(t,n,$.SectionHeaderSizeBytes/4);s[0]=e}constructFromBuffer(e,t){this.bufferData=e,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSectionMap=[];const n=$.parseHeader(this.bufferData);this.versionMajor=n.versionMajor,this.versionMinor=n.versionMinor,this.maxSectionCount=n.maxSectionCount,this.sectionCount=t?n.maxSectionCount:0,this.maxSplatCount=n.maxSplatCount,this.splatCount=t?n.maxSplatCount:0,this.compressionLevel=n.compressionLevel,this.sceneCenter=new U().copy(n.sceneCenter),this.minSphericalHarmonicsCoeff=n.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff=n.maxSphericalHarmonicsCoeff,this.sections=$.parseSectionHeaders(n,this.bufferData,$.HeaderSizeBytes,t),this.linkBufferArrays(),this.buildMaps()}static calculateComponentStorage(e,t){const n=$.CompressionLevels[e].BytesPerCenter,s=$.CompressionLevels[e].BytesPerScale,r=$.CompressionLevels[e].BytesPerRotation,o=$.CompressionLevels[e].BytesPerColor,a=mo(t),l=$.CompressionLevels[e].BytesPerSphericalHarmonicsComponent*a,c=n+s+r+o+l;return{bytesPerCenter:n,bytesPerScale:s,bytesPerRotation:r,bytesPerColor:o,sphericalHarmonicsComponentsPerSplat:a,sphericalHarmonicsBytesPerSplat:l,bytesPerSplat:c}}linkBufferArrays(){for(let e=0;e<this.maxSectionCount;e++){const t=this.sections[e];t.bucketArray=new Float32Array(this.bufferData,t.bucketsBase,t.bucketCount*$.BucketStorageSizeFloats),t.partiallyFilledBucketCount>0&&(t.partiallyFilledBucketLengths=new Uint32Array(this.bufferData,t.base,t.partiallyFilledBucketCount))}}buildMaps(){let e=0;for(let t=0;t<this.maxSectionCount;t++){const n=this.sections[t];for(let s=0;s<n.maxSplatCount;s++){const r=e+s;this.globalSplatIndexToLocalSplatIndexMap[r]=s,this.globalSplatIndexToSectionMap[r]=t}e+=n.maxSplatCount}}updateLoadedCounts(e,t){$.writeHeaderCountsToBuffer(e,t,this.bufferData),this.sectionCount=e,this.splatCount=t}updateSectionLoadedCounts(e,t){const n=$.HeaderSizeBytes+$.SectionHeaderSizeBytes*e;$.writeSectionHeaderSplatCountToBuffer(t,this.bufferData,n),this.sections[e].splatCount=t}static writeSplatDataToSectionBuffer=(function(){const e=new ArrayBuffer(12),t=new ArrayBuffer(12),n=new ArrayBuffer(16),s=new ArrayBuffer(4),r=new ArrayBuffer(256),o=new Ft,a=new U,l=new U,{X:c,Y:u,Z:f,SCALE0:d,SCALE1:h,SCALE2:x,ROTATION0:p,ROTATION1:g,ROTATION2:m,ROTATION3:_,FDC0:v,FDC1:A,FDC2:S,OPACITY:b,FRC0:M,FRC9:R}=Ie.OFFSET,y=(C,F,I)=>{const O=I*2+1;return C=Math.round(C*F)+I,Bt(C,0,O)};return function(C,F,I,O,z,H,V,q,G=-ws,Y=ws){const ae=mo(z),Ae=$.CompressionLevels[O].BytesPerCenter,Pe=$.CompressionLevels[O].BytesPerScale,Ne=$.CompressionLevels[O].BytesPerRotation,We=$.CompressionLevels[O].BytesPerColor,He=I,J=He+Ae,le=J+Pe,be=le+Ne,ze=be+We;if(C[p]!==void 0?(o.set(C[p],C[g],C[m],C[_]),o.normalize()):o.set(1,0,0,0),C[d]!==void 0?a.set(C[d]||0,C[h]||0,C[x]||0):a.set(0,0,0),O===0){const Qe=new Float32Array(F,He,$.CenterComponentCount),B=new Float32Array(F,le,$.RotationComponentCount),N=new Float32Array(F,J,$.ScaleComponentCount);if(B.set([o.x,o.y,o.z,o.w]),N.set([a.x,a.y,a.z]),Qe.set([C[c],C[u],C[f]]),z>0){const j=new Float32Array(F,ze,ae);if(z>=1){for(let P=0;P<9;P++)j[P]=C[M+P]||0;if(z>=2)for(let P=0;P<15;P++)j[P+9]=C[R+P]||0}}}else{const Qe=new Uint16Array(e,0,$.CenterComponentCount),B=new Uint16Array(n,0,$.RotationComponentCount),N=new Uint16Array(t,0,$.ScaleComponentCount);if(B.set([Jt(o.x),Jt(o.y),Jt(o.z),Jt(o.w)]),N.set([Jt(a.x),Jt(a.y),Jt(a.z)]),l.set(C[c],C[u],C[f]).sub(H),l.x=y(l.x,V,q),l.y=y(l.y,V,q),l.z=y(l.z,V,q),Qe.set([l.x,l.y,l.z]),z>0){const j=O===1?Uint16Array:Uint8Array,P=O===1?2:1,ie=new j(r,0,ae);if(z>=1){for(let fe=0;fe<9;fe++){const ne=C[M+fe]||0;ie[fe]=O===1?Jt(ne):ha(ne,G,Y)}const re=9*P;if($r(ie.buffer,0,F,ze,re),z>=2){for(let fe=0;fe<15;fe++){const ne=C[R+fe]||0;ie[fe+9]=O===1?Jt(ne):ha(ne,G,Y)}$r(ie.buffer,re,F,ze+re,15*P)}}}$r(Qe.buffer,0,F,He,6),$r(N.buffer,0,F,J,6),$r(B.buffer,0,F,le,8)}const Ce=new Uint8ClampedArray(s,0,4);Ce.set([C[v]||0,C[A]||0,C[S]||0]),Ce[3]=C[b]||0,$r(Ce.buffer,0,F,be,4)}})();static generateFromUncompressedSplatArrays(e,t,n,s,r,o,a=[]){let l=0;for(let S=0;S<e.length;S++){const b=e[S];l=Math.max(b.sphericalHarmonicsDegree,l)}let c,u;for(let S=0;S<e.length;S++){const b=e[S];for(let M=0;M<b.splats.length;M++){const R=b.splats[M];for(let y=Ie.OFFSET.FRC0;y<Ie.OFFSET.FRC23&&y<R.length;y++)(!c||R[y]<c)&&(c=R[y]),(!u||R[y]>u)&&(u=R[y])}}c=c||-ws,u=u||ws;const{bytesPerSplat:f}=$.calculateComponentStorage(n,l),d=$.CompressionLevels[n].ScaleRange,h=[],x=[];let p=0;for(let S=0;S<e.length;S++){const b=e[S],M=new Ie(l);for(let He=0;He<b.splatCount;He++){const J=b.splats[He];(J[Ie.OFFSET.OPACITY]||0)>=t&&M.addSplat(J)}const R=a[S]||{},y=(R.blockSizeFactor||1)*(r||$.BucketBlockSize),C=Math.ceil((R.bucketSizeFactor||1)*(o||$.BucketSize)),F=$.computeBucketsForUncompressedSplatArray(M,y,C),I=F.fullBuckets.length,O=F.partiallyFullBuckets.map(He=>He.splats.length),z=O.length,H=[...F.fullBuckets,...F.partiallyFullBuckets],V=M.splats.length*f,q=z*4,G=n>=1?H.length*$.BucketStorageSizeBytes+q:0,Y=V+G,ae=new ArrayBuffer(Y),Ae=d/(y*.5),Pe=new U;let Ne=0;for(let He=0;He<H.length;He++){const J=H[He];Pe.fromArray(J.center);for(let le=0;le<J.splats.length;le++){let be=J.splats[le];const ze=M.splats[be],Ce=G+Ne*f;$.writeSplatDataToSectionBuffer(ze,ae,Ce,n,l,Pe,Ae,d,c,u),Ne++}}if(p+=Ne,n>=1){const He=new Uint32Array(ae,0,O.length*4);for(let le=0;le<O.length;le++)He[le]=O[le];const J=new Float32Array(ae,q,H.length*$.BucketStorageSizeFloats);for(let le=0;le<H.length;le++){const be=H[le],ze=le*3;J[ze]=be.center[0],J[ze+1]=be.center[1],J[ze+2]=be.center[2]}}h.push(ae);const We=new ArrayBuffer($.SectionHeaderSizeBytes);$.writeSectionHeaderToBuffer({maxSplatCount:Ne,splatCount:Ne,bucketSize:C,bucketCount:H.length,bucketBlockSize:y,compressionScaleRange:d,storageSizeBytes:Y,fullBucketCount:I,partiallyFilledBucketCount:z,sphericalHarmonicsDegree:l},n,We,0),x.push(We)}let g=0;for(let S of h)g+=S.byteLength;const m=$.HeaderSizeBytes+$.SectionHeaderSizeBytes*h.length+g,_=new ArrayBuffer(m);$.writeHeaderToBuffer({versionMajor:0,versionMinor:1,maxSectionCount:h.length,sectionCount:h.length,maxSplatCount:p,splatCount:p,compressionLevel:n,sceneCenter:s,minSphericalHarmonicsCoeff:c,maxSphericalHarmonicsCoeff:u},_);let v=$.HeaderSizeBytes;for(let S of x)new Uint8Array(_,v,$.SectionHeaderSizeBytes).set(new Uint8Array(S)),v+=$.SectionHeaderSizeBytes;for(let S of h)new Uint8Array(_,v,S.byteLength).set(new Uint8Array(S)),v+=S.byteLength;return new $(_)}static computeBucketsForUncompressedSplatArray(e,t,n){let s=e.splatCount;const r=t/2,o=new U,a=new U;for(let p=0;p<s;p++){const g=e.splats[p],m=[g[Ie.OFFSET.X],g[Ie.OFFSET.Y],g[Ie.OFFSET.Z]];(p===0||m[0]<o.x)&&(o.x=m[0]),(p===0||m[0]>a.x)&&(a.x=m[0]),(p===0||m[1]<o.y)&&(o.y=m[1]),(p===0||m[1]>a.y)&&(a.y=m[1]),(p===0||m[2]<o.z)&&(o.z=m[2]),(p===0||m[2]>a.z)&&(a.z=m[2])}const l=new U().copy(a).sub(o),c=Math.ceil(l.y/t),u=Math.ceil(l.z/t),f=new U,d=[],h={};for(let p=0;p<s;p++){const g=e.splats[p],m=[g[Ie.OFFSET.X],g[Ie.OFFSET.Y],g[Ie.OFFSET.Z]],_=Math.floor((m[0]-o.x)/t),v=Math.floor((m[1]-o.y)/t),A=Math.floor((m[2]-o.z)/t);f.x=_*t+o.x+r,f.y=v*t+o.y+r,f.z=A*t+o.z+r;const S=_*(c*u)+v*u+A;let b=h[S];b||(h[S]=b={splats:[],center:f.toArray()}),b.splats.push(p),b.splats.length>=n&&(d.push(b),h[S]=null)}const x=[];for(let p in h)if(h.hasOwnProperty(p)){const g=h[p];g&&x.push(g)}return{fullBuckets:d,partiallyFullBuckets:x}}static preallocateUncompressed(e,t){const n=$.CompressionLevels[0].SphericalHarmonicsDegrees[t],s=$.HeaderSizeBytes+$.SectionHeaderSizeBytes,r=s+n.BytesPerSplat*e,o=new ArrayBuffer(r);return $.writeHeaderToBuffer({versionMajor:$.CurrentMajorVersion,versionMinor:$.CurrentMinorVersion,maxSectionCount:1,sectionCount:1,maxSplatCount:e,splatCount:e,compressionLevel:0,sceneCenter:new U},o),$.writeSectionHeaderToBuffer({maxSplatCount:e,splatCount:e,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:t},0,o,$.HeaderSizeBytes),{splatBuffer:new $(o,!0),splatBufferDataOffsetBytes:s}}}const Yp=new Uint8Array([112,108,121,10]),Qp=new Uint8Array([10,101,110,100,95,104,101,97,100,101,114,10]),Mu="end_header",Cu=new Map([["char",Int8Array],["uchar",Uint8Array],["short",Int16Array],["ushort",Uint16Array],["int",Int32Array],["uint",Uint32Array],["float",Float32Array],["double",Float64Array]]),Ui=(i,e)=>{const t=(1<<e)-1;return(i&t)/t},Kp=(i,e)=>{i.x=Ui(e>>>21,11),i.y=Ui(e>>>11,10),i.z=Ui(e,11)},QT=(i,e)=>{i.x=Ui(e>>>24,8),i.y=Ui(e>>>16,8),i.z=Ui(e>>>8,8),i.w=Ui(e,8)},KT=(i,e)=>{const t=1/(Math.sqrt(2)*.5),n=(Ui(e>>>20,10)-.5)*t,s=(Ui(e>>>10,10)-.5)*t,r=(Ui(e,10)-.5)*t,o=Math.sqrt(1-(n*n+s*s+r*r));switch(e>>>30){case 0:i.set(o,n,s,r);break;case 1:i.set(n,o,s,r);break;case 2:i.set(n,s,o,r);break;case 3:i.set(n,s,r,o);break}},es=(i,e,t)=>i*(1-t)+e*t,Nt=(i,e)=>i.properties.find(t=>t.name===e&&t.storage)?.storage;class ft{static decodeHeaderText(e){let t,n,s,r;const o=e.split(`
`).filter(f=>!f.startsWith("comment "));let a=0,l=!1;for(let f=1;f<o.length;++f){const d=o[f].split(" ");switch(d[0]){case"format":if(d[1]!=="binary_little_endian")throw new Error("Unsupported ply format");break;case"element":t={name:d[1],count:parseInt(d[2],10),properties:[],storageSizeBytes:0},t.name==="chunk"?n=t:t.name==="vertex"?s=t:t.name==="sh"&&(r=t);break;case"property":{if(!Cu.has(d[1]))throw new Error(`Unrecognized property data type '${d[1]}' in ply header`);const h=Cu.get(d[1]),x=h.BYTES_PER_ELEMENT*t.count;t.name==="vertex"&&(a+=h.BYTES_PER_ELEMENT),t.properties.push({type:d[1],name:d[2],storage:null,byteSize:h.BYTES_PER_ELEMENT,storageSizeByes:x}),t.storageSizeBytes+=x;break}case Mu:l=!0;break;default:throw new Error(`Unrecognized header value '${d[0]}' in ply header`)}if(l)break}let c=0,u=0;return r&&(u=r.properties.length,r.properties.length>=45?c=3:r.properties.length>=24?c=2:r.properties.length>=9&&(c=1)),{chunkElement:n,vertexElement:s,shElement:r,bytesPerSplat:a,headerSizeBytes:e.indexOf(Mu)+Mu.length+1,sphericalHarmonicsDegree:c,sphericalHarmonicsPerSplat:u}}static decodeHeader(e){const t=(h,x)=>{const p=h.length-x.length;let g,m;for(g=0;g<=p;++g){for(m=0;m<x.length&&h[g+m]===x[m];++m);if(m===x.length)return g}return-1},n=(h,x)=>{if(h.length<x.length)return!1;for(let p=0;p<x.length;++p)if(h[p]!==x[p])return!1;return!0};let s=new Uint8Array(e),r;if(s.length>=Yp.length&&!n(s,Yp))throw new Error("Invalid PLY header");if(r=t(s,Qp),r===-1)throw new Error("End of PLY header not found");const o=new TextDecoder("ascii").decode(s.slice(0,r)),{chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f,bytesPerSplat:d}=ft.decodeHeaderText(o);return{headerSizeBytes:r+Qp.length,bytesPerSplat:d,chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f}}static readElementData(e,t,n,s,r,o=null){let a=t instanceof DataView?t:new DataView(t);s=s||0,r=r||e.count-1;for(let l=s;l<=r;++l)for(let c=0;c<e.properties.length;++c){const u=e.properties[c],f=Cu.get(u.type),d=f.BYTES_PER_ELEMENT*e.count;if((!u.storage||u.storage.byteLength<d)&&(!o||o(u.name))&&(u.storage=new f(e.count)),u.storage)switch(u.type){case"char":u.storage[l]=a.getInt8(n);break;case"uchar":u.storage[l]=a.getUint8(n);break;case"short":u.storage[l]=a.getInt16(n,!0);break;case"ushort":u.storage[l]=a.getUint16(n,!0);break;case"int":u.storage[l]=a.getInt32(n,!0);break;case"uint":u.storage[l]=a.getUint32(n,!0);break;case"float":u.storage[l]=a.getFloat32(n,!0);break;case"double":u.storage[l]=a.getFloat64(n,!0);break}n+=u.byteSize}return n}static readPly(e,t=null){const n=ft.decodeHeader(e);let s=ft.readElementData(n.chunkElement,e,n.headerSizeBytes,null,null,t);return s=ft.readElementData(n.vertexElement,e,s,null,null,t),ft.readElementData(n.shElement,e,s,null,null,t),{chunkElement:n.chunkElement,vertexElement:n.vertexElement,shElement:n.shElement,sphericalHarmonicsDegree:n.sphericalHarmonicsDegree,sphericalHarmonicsPerSplat:n.sphericalHarmonicsPerSplat}}static getElementStorageArrays(e,t,n){const s={};if(t){const r=Nt(e,"min_r"),o=Nt(e,"min_g"),a=Nt(e,"min_b"),l=Nt(e,"max_r"),c=Nt(e,"max_g"),u=Nt(e,"max_b"),f=Nt(e,"min_x"),d=Nt(e,"min_y"),h=Nt(e,"min_z"),x=Nt(e,"max_x"),p=Nt(e,"max_y"),g=Nt(e,"max_z"),m=Nt(e,"min_scale_x"),_=Nt(e,"min_scale_y"),v=Nt(e,"min_scale_z"),A=Nt(e,"max_scale_x"),S=Nt(e,"max_scale_y"),b=Nt(e,"max_scale_z"),M=Nt(t,"packed_position"),R=Nt(t,"packed_rotation"),y=Nt(t,"packed_scale"),C=Nt(t,"packed_color");s.colorExtremes={minR:r,maxR:l,minG:o,maxG:c,minB:a,maxB:u},s.positionExtremes={minX:f,maxX:x,minY:d,maxY:p,minZ:h,maxZ:g},s.scaleExtremes={minScaleX:m,maxScaleX:A,minScaleY:_,maxScaleY:S,minScaleZ:v,maxScaleZ:b},s.position=M,s.rotation=R,s.scale=y,s.color=C}if(n){const r={};for(let o=0;o<45;o++){const a=`f_rest_${o}`,l=Nt(n,a);if(l)r[a]=l;else break}s.sh=r}return s}static decompressBaseSplat=(function(){const e=new U,t=new Ft,n=new U,s=new Ot,r=Ie.OFFSET;return function(o,a,l,c,u,f,d,h,x,p){p=p||Ie.createSplat();const g=Math.floor((a+o)/256);return Kp(e,l[o]),KT(t,d[o]),Kp(n,u[o]),QT(s,x[o]),p[r.X]=es(c.minX[g],c.maxX[g],e.x),p[r.Y]=es(c.minY[g],c.maxY[g],e.y),p[r.Z]=es(c.minZ[g],c.maxZ[g],e.z),p[r.ROTATION0]=t.x,p[r.ROTATION1]=t.y,p[r.ROTATION2]=t.z,p[r.ROTATION3]=t.w,p[r.SCALE0]=Math.exp(es(f.minScaleX[g],f.maxScaleX[g],n.x)),p[r.SCALE1]=Math.exp(es(f.minScaleY[g],f.maxScaleY[g],n.y)),p[r.SCALE2]=Math.exp(es(f.minScaleZ[g],f.maxScaleZ[g],n.z)),h.minR&&h.maxR?p[r.FDC0]=Bt(Math.round(es(h.minR[g],h.maxR[g],s.x)*255),0,255):p[r.FDC0]=Bt(Math.floor(s.x*255),0,255),h.minG&&h.maxG?p[r.FDC1]=Bt(Math.round(es(h.minG[g],h.maxG[g],s.y)*255),0,255):p[r.FDC1]=Bt(Math.floor(s.y*255),0,255),h.minB&&h.maxB?p[r.FDC2]=Bt(Math.round(es(h.minB[g],h.maxB[g],s.z)*255),0,255):p[r.FDC2]=Bt(Math.floor(s.z*255),0,255),p[r.OPACITY]=Bt(Math.floor(s.w*255),0,255),p}})();static decompressSphericalHarmonics=(function(){const e=[0,3,8,15],t=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(n,s,r,o,a){a=a||Ie.createSplat();let l=e[r],c=e[o];for(let u=0;u<3;++u)for(let f=0;f<15;++f){const d=t[u*15+f];f<l&&f<c&&(a[Ie.OFFSET.FRC0+d]=s[u*c+f][n]*(8/255)-4)}return a}})();static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l,c=null){ft.readElementData(t,o,0,n,s,c);const u=$.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,{positionExtremes:f,scaleExtremes:d,colorExtremes:h,position:x,rotation:p,scale:g,color:m}=ft.getElementStorageArrays(e,t),_=Ie.createSplat();for(let v=n;v<=s;++v){ft.decompressBaseSplat(v,r,x,f,g,d,p,h,m,_);const A=v*u+l;$.writeSplatDataToSectionBuffer(_,a,A,0,0)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a,l=null){ft.readElementData(t,o,0,n,s,l);const{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:d,rotation:h,scale:x,color:p}=ft.getElementStorageArrays(e,t);for(let g=n;g<=s;++g){const m=Ie.createSplat();ft.decompressBaseSplat(g,r,d,c,x,u,h,f,p,m),a.addSplat(m)}}static parseSphericalHarmonicsToUncompressedSplatArraySection(e,t,n,s,r,o,a,l,c,u=null){ft.readElementData(t,r,o,n,s,u);const{sh:f}=ft.getElementStorageArrays(e,void 0,t),d=Object.values(f);for(let h=n;h<=s;++h)ft.decompressSphericalHarmonics(h,d,a,l,c.splats[h])}static parseToUncompressedSplatArray(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=ft.readPly(e);t=Math.min(t,o);const a=new Ie(t),{positionExtremes:l,scaleExtremes:c,colorExtremes:u,position:f,rotation:d,scale:h,color:x}=ft.getElementStorageArrays(n,s);let p;if(t>0){const{sh:g}=ft.getElementStorageArrays(n,void 0,r);p=Object.values(g)}for(let g=0;g<s.count;++g){a.addDefaultSplat();const m=a.getSplat(a.splatCount-1);ft.decompressBaseSplat(g,0,f,l,h,c,d,u,x,m),t>0&&ft.decompressSphericalHarmonics(g,p,t,o,m)}return a}static parseToUncompressedSplatBuffer(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=ft.readPly(e);t=Math.min(t,o);const{splatBuffer:a,splatBufferDataOffsetBytes:l}=$.preallocateUncompressed(s.count,t),{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:d,rotation:h,scale:x,color:p}=ft.getElementStorageArrays(n,s);let g;if(t>0){const{sh:v}=ft.getElementStorageArrays(n,void 0,r);g=Object.values(v)}const m=$.CompressionLevels[0].SphericalHarmonicsDegrees[t].BytesPerSplat,_=Ie.createSplat(t);for(let v=0;v<s.count;++v){ft.decompressBaseSplat(v,0,d,c,x,u,h,f,p,_),t>0&&ft.decompressSphericalHarmonics(v,g,t,o,_);const A=v*m+l;$.writeSplatDataToSectionBuffer(_,a.bufferData,A,0,t)}return a}}const Rn={INRIAV1:0,INRIAV2:1,PlayCanvasCompressed:2},[Rg,Nd,zd,kd,Hd,Vd,Gd]=[0,1,2,3,4,5,6],jp={double:Rg,int:Nd,uint:zd,float:kd,short:Hd,ushort:Vd,uchar:Gd},jT={[Rg]:8,[Nd]:4,[zd]:4,[kd]:4,[Hd]:2,[Vd]:2,[Gd]:1};class gt{static HeaderEndToken="end_header";static decodeSectionHeader(e,t,n=0){const s=[];let r=!1,o=-1,a=0,l=!1,c=null;const u=[],f=[],d=[],h={};for(let m=n;m<e.length;m++){const _=e[m].trim();if(_.startsWith("element"))if(r){o--;break}else{r=!0,n=m,o=m;const v=_.split(" ");let A=0;for(let S of v){const b=S.trim();b.length>0&&(A++,A===2?c=b:A===3&&(a=parseInt(b)))}}else if(_.startsWith("property")){const v=_.match(/(\w+)\s+(\w+)\s+(\w+)/);if(v){const A=v[2],S=v[3];d.push(S);const b=t[S];h[S]=A;const M=jp[A];b!==void 0&&(u.push(b),f[b]=M)}}if(_===gt.HeaderEndToken){l=!0;break}r&&(s.push(_),o++)}const x=[];let p=0;for(let m of d){const _=h[m];if(h.hasOwnProperty(m)){const v=t[m];v!==void 0&&(x[v]=p)}p+=jT[jp[_]]}const g=gt.decodeSphericalHarmonicsFromSectionHeader(d,t);return{headerLines:s,headerStartLine:n,headerEndLine:o,fieldTypes:f,fieldIds:u,fieldOffsets:x,bytesPerVertex:p,vertexCount:a,dataSizeBytes:p*a,endOfHeader:l,sectionName:c,sphericalHarmonicsDegree:g.degree,sphericalHarmonicsCoefficientsPerChannel:g.coefficientsPerChannel,sphericalHarmonicsDegree1Fields:g.degree1Fields,sphericalHarmonicsDegree2Fields:g.degree2Fields}}static decodeSphericalHarmonicsFromSectionHeader(e,t){let n=0,s=0;for(let l of e)l.startsWith("f_rest")&&n++;s=n/3;let r=0;s>=3&&(r=1),s>=8&&(r=2);let o=[],a=[];for(let l=0;l<3;l++){if(r>=1)for(let c=0;c<3;c++)o.push(t["f_rest_"+(c+s*l)]);if(r>=2)for(let c=0;c<5;c++)a.push(t["f_rest_"+(c+s*l+3)])}return{degree:r,coefficientsPerChannel:s,degree1Fields:o,degree2Fields:a}}static getHeaderSectionNames(e){const t=[];for(let n of e)if(n.startsWith("element")){const s=n.split(" ");let r=0;for(let o of s){const a=o.trim();a.length>0&&(r++,r===2&&t.push(a))}}return t}static checkTextForEndHeader(e){return!!e.includes(gt.HeaderEndToken)}static checkBufferForEndHeader(e,t,n,s){const r=new Uint8Array(e,Math.max(0,t-n),n),o=s.decode(r);return gt.checkTextForEndHeader(o)}static extractHeaderFromBufferToText(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,gt.checkBufferForEndHeader(e,n,r*2,t))break}return s}static readHeaderFromBuffer(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,gt.checkBufferForEndHeader(e,n,r*2,t))break}return s}static convertHeaderTextToLines(e){const t=e.split(`
`),n=[];for(let s=0;s<t.length;s++){const r=t[s].trim();if(n.push(r),r===gt.HeaderEndToken)break}return n}static determineHeaderFormatFromHeaderText(e){const t=gt.convertHeaderTextToLines(e);let n=Rn.INRIAV1;for(let s=0;s<t.length;s++){const r=t[s].trim();if(r.startsWith("element chunk")||r.match(/[A-Za-z]*packed_[A-Za-z]*/))n=Rn.PlayCanvasCompressed;else if(r.startsWith("element codebook_centers"))n=Rn.INRIAV2;else if(r===gt.HeaderEndToken)break}return n}static determineHeaderFormatFromPlyBuffer(e){const t=gt.extractHeaderFromBufferToText(e);return gt.determineHeaderFormatFromHeaderText(t)}static readVertex(e,t,n,s,r,o,a=!0){const l=n*t.bytesPerVertex+s,c=t.fieldOffsets,u=t.fieldTypes;for(let f of r){const d=u[f];d===kd?o[f]=e.getFloat32(l+c[f],!0):d===Hd?o[f]=e.getInt16(l+c[f],!0):d===Vd?o[f]=e.getUint16(l+c[f],!0):d===Nd?o[f]=e.getInt32(l+c[f],!0):d===zd?o[f]=e.getUint32(l+c[f],!0):d===Gd&&(a?o[f]=e.getUint8(l+c[f])/255:o[f]=e.getUint8(l+c[f]))}}}const Ig=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0"],$T=Ig.map((i,e)=>e),[$p,ZT,JT,eE,tE,nE,iE,sE,rE,oE,Zp,aE,lE,Jp,em,cE,uE,fE]=$T;class cn{static decodeHeaderLines(e){let t=0;e.forEach(u=>{u.includes("f_rest_")&&t++});let n=0;t>=45?n=45:t>=24?n=24:t>=9&&(n=9);let r=Array.from(Array(Math.max(n-1,0))).map((u,f)=>`f_rest_${f+1}`);const o=[...Ig,...r],a=o.map((u,f)=>f),l=a.reduce((u,f)=>(u[o[f]]=f,u),{}),c=gt.decodeSectionHeader(e,l,0);return c.splatCount=c.vertexCount,c.bytesPerSplat=c.bytesPerVertex,c.fieldsToReadIndexes=a,c}static decodeHeaderText(e){const t=gt.convertHeaderTextToLines(e),n=cn.decodeHeaderLines(t);return n.headerText=e,n.headerSizeBytes=e.indexOf(gt.HeaderEndToken)+gt.HeaderEndToken.length+1,n}static decodeHeaderFromBuffer(e){const t=gt.readHeaderFromBuffer(e);return cn.decodeHeaderText(t)}static findSplatData(e,t){return new DataView(e,t.headerSizeBytes)}static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l=0){l=Math.min(l,e.sphericalHarmonicsDegree);const c=$.CompressionLevels[0].SphericalHarmonicsDegrees[l].BytesPerSplat;for(let u=t;u<=n;u++){const f=cn.parseToUncompressedSplat(s,u,e,r,l),d=u*c+a;$.writeSplatDataToSectionBuffer(f,o,d,0,l)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a=0){a=Math.min(a,e.sphericalHarmonicsDegree);for(let l=t;l<=n;l++){const c=cn.parseToUncompressedSplat(s,l,e,r,a);o.addSplat(c)}}static decodeSectionSplatData(e,t,n,s,r=!0){if(s=Math.min(s,n.sphericalHarmonicsDegree),r){const o=new Ie(s);for(let a=0;a<t;a++){const l=cn.parseToUncompressedSplat(e,a,n,0,s);o.addSplat(l)}return o}else{const{splatBuffer:o,splatBufferDataOffsetBytes:a}=$.preallocateUncompressed(t,s);return cn.parseToUncompressedSplatBufferSection(n,0,t-1,e,0,o.bufferData,a,s),o}}static parseToUncompressedSplat=(function(){let e=[];const t=new Ft,n=Ie.OFFSET.X,s=Ie.OFFSET.Y,r=Ie.OFFSET.Z,o=Ie.OFFSET.SCALE0,a=Ie.OFFSET.SCALE1,l=Ie.OFFSET.SCALE2,c=Ie.OFFSET.ROTATION0,u=Ie.OFFSET.ROTATION1,f=Ie.OFFSET.ROTATION2,d=Ie.OFFSET.ROTATION3,h=Ie.OFFSET.FDC0,x=Ie.OFFSET.FDC1,p=Ie.OFFSET.FDC2,g=Ie.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=Ie.OFFSET.FRC0+_;return function(_,v,A,S=0,b=0){b=Math.min(b,A.sphericalHarmonicsDegree),cn.readSplat(_,A,v,S,e);const M=Ie.createSplat(b);if(e[$p]!==void 0?(M[o]=Math.exp(e[$p]),M[a]=Math.exp(e[ZT]),M[l]=Math.exp(e[JT])):(M[o]=.01,M[a]=.01,M[l]=.01),e[Zp]!==void 0){const R=.28209479177387814;M[h]=(.5+R*e[Zp])*255,M[x]=(.5+R*e[aE])*255,M[p]=(.5+R*e[lE])*255}else e[em]!==void 0?(M[h]=e[em]*255,M[x]=e[cE]*255,M[p]=e[uE]*255):(M[h]=0,M[x]=0,M[p]=0);if(e[Jp]!==void 0&&(M[g]=1/(1+Math.exp(-e[Jp]))*255),M[h]=Bt(Math.floor(M[h]),0,255),M[x]=Bt(Math.floor(M[x]),0,255),M[p]=Bt(Math.floor(M[p]),0,255),M[g]=Bt(Math.floor(M[g]),0,255),b>=1&&e[fE]!==void 0){for(let R=0;R<9;R++)M[m[R]]=e[A.sphericalHarmonicsDegree1Fields[R]];if(b>=2)for(let R=0;R<15;R++)M[m[9+R]]=e[A.sphericalHarmonicsDegree2Fields[R]]}return t.set(e[eE],e[tE],e[nE],e[iE]),t.normalize(),M[c]=t.x,M[u]=t.y,M[f]=t.z,M[d]=t.w,M[n]=e[sE],M[s]=e[rE],M[r]=e[oE],M}})();static readSplat(e,t,n,s,r){return gt.readVertex(e,t,n,s,t.fieldsToReadIndexes,r,!0)}static parseToUncompressedSplatArray(e,t=0){const{header:n,splatCount:s,splatData:r}=tm(e);return cn.decodeSectionSplatData(r,s,n,t,!0)}static parseToUncompressedSplatBuffer(e,t=0){const{header:n,splatCount:s,splatData:r}=tm(e);return cn.decodeSectionSplatData(r,s,n,t,!1)}}function tm(i){const e=cn.decodeHeaderFromBuffer(i),t=e.splatCount,n=cn.findSplatData(i,e);return{header:e,splatCount:t,splatData:n}}const Dg=["features_dc","features_rest_0","features_rest_1","features_rest_2","features_rest_3","features_rest_4","features_rest_5","features_rest_6","features_rest_7","features_rest_8","features_rest_9","features_rest_10","features_rest_11","features_rest_12","features_rest_13","features_rest_14","opacity","scaling","rotation_re","rotation_im"],Cl=Dg.map((i,e)=>e),[Tl,dE,hE,nm,El,pE,Tu]=[0,1,4,16,17,18,19],Pg=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0","f_rest_1","f_rest_2","f_rest_3","f_rest_4","f_rest_5","f_rest_6","f_rest_7","f_rest_8","f_rest_9","f_rest_10","f_rest_11","f_rest_12","f_rest_13","f_rest_14","f_rest_15","f_rest_16","f_rest_17","f_rest_18","f_rest_19","f_rest_20","f_rest_21","f_rest_22","f_rest_23","f_rest_24","f_rest_25","f_rest_26","f_rest_27","f_rest_28","f_rest_29","f_rest_30","f_rest_31","f_rest_32","f_rest_33","f_rest_34","f_rest_35","f_rest_36","f_rest_37","f_rest_38","f_rest_39","f_rest_40","f_rest_41","f_rest_42","f_rest_43","f_rest_44","f_rest_45"],kf=Pg.map((i,e)=>e),[im,mE,gE,xE,_E,vE,AE,SE,yE,bE,Hf,Fg,Lg,sm]=kf,rm=Hf,ME=Fg,CE=Lg,wl=i=>{const e=(31744&i)>>10,t=1023&i;return(i>>15?-1:1)*(e?e===31?t?NaN:1/0:Math.pow(2,e-15)*(1+t/1024):t/1024*6103515625e-14)};class Qn{static decodeSectionHeadersFromHeaderLines(e){const t=kf.reduce((u,f)=>(u[Pg[f]]=f,u),{}),n=Cl.reduce((u,f)=>(u[Dg[f]]=f,u),{}),s=gt.getHeaderSectionNames(e);let r;for(let u=0;u<s.length;u++)s[u]==="codebook_centers"&&(r=u);let o=0,a=!1;const l=[];let c=0;for(;!a;){let u;c===r?u=gt.decodeSectionHeader(e,n,o):u=gt.decodeSectionHeader(e,t,o),a=u.endOfHeader,o=u.headerEndLine+1,a||(u.splatCount=u.vertexCount,u.bytesPerSplat=u.bytesPerVertex),l.push(u),c++}return l}static decodeSectionHeadersFromHeaderText(e){const t=gt.convertHeaderTextToLines(e);return Qn.decodeSectionHeadersFromHeaderLines(t)}static getSplatCountFromSectionHeaders(e){let t=0;for(let n of e)n.sectionName!=="codebook_centers"&&(t+=n.vertexCount);return t}static decodeHeaderFromHeaderText(e){const t=e.indexOf(gt.HeaderEndToken)+gt.HeaderEndToken.length+1,n=Qn.decodeSectionHeadersFromHeaderText(e),s=Qn.getSplatCountFromSectionHeaders(n);return{headerSizeBytes:t,sectionHeaders:n,splatCount:s}}static decodeHeaderFromBuffer(e){const t=gt.readHeaderFromBuffer(e);return Qn.decodeHeaderFromHeaderText(t)}static findVertexData(e,t,n){let s=t.headerSizeBytes;for(let r=0;r<n&&r<t.sectionHeaders.length;r++){const o=t.sectionHeaders[r];s+=o.dataSizeBytes}return new DataView(e,s,t.sectionHeaders[n].dataSizeBytes)}static decodeCodeBook(e,t){const n=[],s=[];for(let r=0;r<t.vertexCount;r++){gt.readVertex(e,t,r,0,Cl,n);for(let o of Cl){const a=Cl[o];let l=s[a];l||(s[a]=l=[]),l.push(n[o])}}for(let r=0;r<s.length;r++){const o=s[r],a=.28209479177387814;for(let l=0;l<o.length;l++){const c=wl(o[l]);r===nm?o[l]=Math.round(1/(1+Math.exp(-c))*255):r===Tl?o[l]=Math.round((.5+a*c)*255):r===El?o[l]=Math.exp(c):o[l]=c}}return s}static decodeSectionSplatData(e,t,n,s,r){r=Math.min(r,n.sphericalHarmonicsDegree);const o=new Ie(r);for(let a=0;a<t;a++){const l=Qn.parseToUncompressedSplat(e,a,n,s,0,r);o.addSplat(l)}return o}static parseToUncompressedSplat=(function(){let e=[];const t=new Ft,n=Ie.OFFSET.X,s=Ie.OFFSET.Y,r=Ie.OFFSET.Z,o=Ie.OFFSET.SCALE0,a=Ie.OFFSET.SCALE1,l=Ie.OFFSET.SCALE2,c=Ie.OFFSET.ROTATION0,u=Ie.OFFSET.ROTATION1,f=Ie.OFFSET.ROTATION2,d=Ie.OFFSET.ROTATION3,h=Ie.OFFSET.FDC0,x=Ie.OFFSET.FDC1,p=Ie.OFFSET.FDC2,g=Ie.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=Ie.OFFSET.FRC0+_;return function(_,v,A,S,b=0,M=0){M=Math.min(M,A.sphericalHarmonicsDegree),Qn.readSplat(_,A,v,b,e);const R=Ie.createSplat(M);if(e[im]!==void 0?(R[o]=S[El][e[im]],R[a]=S[El][e[mE]],R[l]=S[El][e[gE]]):(R[o]=.01,R[a]=.01,R[l]=.01),e[Hf]!==void 0?(R[h]=S[Tl][e[Hf]],R[x]=S[Tl][e[Fg]],R[p]=S[Tl][e[Lg]]):e[rm]!==void 0?(R[h]=e[rm]*255,R[x]=e[ME]*255,R[p]=e[CE]*255):(R[h]=0,R[x]=0,R[p]=0),e[sm]!==void 0&&(R[g]=S[nm][e[sm]]),R[h]=Bt(Math.floor(R[h]),0,255),R[x]=Bt(Math.floor(R[x]),0,255),R[p]=Bt(Math.floor(R[p]),0,255),R[g]=Bt(Math.floor(R[g]),0,255),M>=1&&A.sphericalHarmonicsDegree>=1){for(let O=0;O<9;O++){const z=S[dE+O%3];R[m[O]]=z[e[A.sphericalHarmonicsDegree1Fields[O]]]}if(M>=2&&A.sphericalHarmonicsDegree>=2)for(let O=0;O<15;O++){const z=S[hE+O%5];R[m[9+O]]=z[e[A.sphericalHarmonicsDegree2Fields[O]]]}}const y=S[pE][e[xE]],C=S[Tu][e[_E]],F=S[Tu][e[vE]],I=S[Tu][e[AE]];return t.set(y,C,F,I),t.normalize(),R[c]=t.x,R[u]=t.y,R[f]=t.z,R[d]=t.w,R[n]=wl(e[SE]),R[s]=wl(e[yE]),R[r]=wl(e[bE]),R}})();static readSplat(e,t,n,s,r){return gt.readVertex(e,t,n,s,kf,r,!1)}static parseToUncompressedSplatArray(e,t=0){const n=[],s=Qn.decodeHeaderFromBuffer(e,t);let r;for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName==="codebook_centers"){const c=Qn.findVertexData(e,s,a);r=Qn.decodeCodeBook(c,l)}}for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName!=="codebook_centers"){const c=l.vertexCount,u=Qn.findVertexData(e,s,a),f=Qn.decodeSectionSplatData(u,c,l,r,t);n.push(f)}}const o=new Ie(t);for(let a of n)for(let l of a.splats)o.addSplat(l);return o}}class om{static parseToUncompressedSplatArray(e,t=0){const n=gt.determineHeaderFormatFromPlyBuffer(e);if(n===Rn.PlayCanvasCompressed)return ft.parseToUncompressedSplatArray(e,t);if(n===Rn.INRIAV1)return cn.parseToUncompressedSplatArray(e,t);if(n===Rn.INRIAV2)return Qn.parseToUncompressedSplatArray(e,t)}static parseToUncompressedSplatBuffer(e,t=0){const n=gt.determineHeaderFormatFromPlyBuffer(e);if(n===Rn.PlayCanvasCompressed)return ft.parseToUncompressedSplatBuffer(e,t);if(n===Rn.INRIAV1)return cn.parseToUncompressedSplatBuffer(e,t);if(n===Rn.INRIAV2)throw new Error("parseToUncompressedSplatBuffer() is not implemented for INRIA V2 PLY files")}}class Wd{constructor(e,t,n,s){this.sectionCount=e,this.sectionFilters=t,this.groupingParameters=n,this.partitionGenerator=s}partitionUncompressedSplatArray(e){let t,n,s;if(this.partitionGenerator){const o=this.partitionGenerator(e);t=o.groupingParameters,n=o.sectionCount,s=o.sectionFilters}else t=this.groupingParameters,n=this.sectionCount,s=this.sectionFilters;const r=[];for(let o=0;o<n;o++){const a=new Ie(e.sphericalHarmonicsDegree),l=s[o];for(let c=0;c<e.splatCount;c++)l(c)&&a.addSplat(e.splats[c]);r.push(a)}return{splatArrays:r,parameters:t}}static getStandardPartitioner(e=0,t=new U,n=$.BucketBlockSize,s=$.BucketSize){const r=o=>{const a=Ie.OFFSET.X,l=Ie.OFFSET.Y,c=Ie.OFFSET.Z;e<=0&&(e=o.splatCount);const u=new U,f=.5,d=m=>{m.x=Math.floor(m.x/f)*f,m.y=Math.floor(m.y/f)*f,m.z=Math.floor(m.z/f)*f};o.splats.forEach(m=>{u.set(m[a],m[l],m[c]).sub(t),d(u),m.centerDist=u.lengthSq()}),o.splats.sort((m,_)=>{let v=m.centerDist,A=_.centerDist;return v>A?1:-1});const h=[],x=[];e=Math.min(o.splatCount,e);const p=Math.ceil(o.splatCount/e);let g=0;for(let m=0;m<p;m++){let _=g;h.push(v=>v>=_&&v<_+e),x.push({blocksSize:n,bucketSize:s}),g+=e}return{sectionCount:h.length,sectionFilters:h,groupingParameters:x}};return new Wd(void 0,void 0,void 0,r)}}class Ya{constructor(e,t,n,s,r,o,a){this.splatPartitioner=e,this.alphaRemovalThreshold=t,this.compressionLevel=n,this.sectionSize=s,this.sceneCenter=r?new U().copy(r):void 0,this.blockSize=o,this.bucketSize=a}generateFromUncompressedSplatArray(e){const t=this.splatPartitioner.partitionUncompressedSplatArray(e);return $.generateFromUncompressedSplatArrays(t.splatArrays,this.alphaRemovalThreshold,this.compressionLevel,this.sceneCenter,this.blockSize,this.bucketSize,t.parameters)}static getStandardGenerator(e=1,t=1,n=0,s=new U,r=$.BucketBlockSize,o=$.BucketSize){const a=Wd.getStandardPartitioner(n,s,r,o);return new Ya(a,e,t,n,s,r,o)}}const Wt={Downloading:0,Processing:1,Done:2};class ic extends Error{constructor(e){super(e)}}const Pt={ProgressiveToSplatBuffer:0,ProgressiveToSplatArray:1,DownloadBeforeProcessing:2};function am(i,e){let t=0;for(let s of i)t+=s.sizeBytes;(!e||e.byteLength<t)&&(e=new ArrayBuffer(t));let n=0;for(let s of i)new Uint8Array(e,n,s.sizeBytes).set(s.data),n+=s.sizeBytes;return e}function lm(i,e,t,n,s,r,o,a){return e?Ya.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):$.generateFromUncompressedSplatArrays([i],t,0,new U)}class Xd{static loadFromURL(e,t,n,s,r,o,a=!0,l=0,c,u,f,d,h){let x;!n&&!a?x=Pt.DownloadBeforeProcessing:a?x=Pt.ProgressiveToSplatArray:x=Pt.ProgressiveToSplatBuffer;const p=bt.ProgressiveLoadSectionSize,g=$.HeaderSizeBytes+$.SectionHeaderSizeBytes,m=1;let _,v,A,S,b,M=0,R=0,y=0,C=!1,F=!1,I=!1;const O=Ld();let z=0,H=0,V=0,q=0,G="",Y=null,ae=[],Ae;const Pe=new TextDecoder,Ne=(We,He,J)=>{const le=We>=100;if(J&&(ae.push({data:J,sizeBytes:J.byteLength,startBytes:V,endBytes:V+J.byteLength}),V+=J.byteLength),x===Pt.DownloadBeforeProcessing)le&&O.resolve(ae);else{if(C){if(_===Rn.PlayCanvasCompressed&&!F){const be=Y.headerSizeBytes+Y.chunkElement.storageSizeBytes;b=am(ae,b),b.byteLength>=be&&(ft.readElementData(Y.chunkElement,b,Y.headerSizeBytes),z=be,H=be,F=!0)}}else if(G+=Pe.decode(J),gt.checkTextForEndHeader(G)){if(_=gt.determineHeaderFormatFromHeaderText(G),_===Rn.INRIAV1)Y=cn.decodeHeaderText(G),l=Math.min(l,Y.sphericalHarmonicsDegree),M=Y.splatCount,F=!0,q=Y.headerSizeBytes+Y.bytesPerSplat*M;else if(_===Rn.PlayCanvasCompressed){if(Y=ft.decodeHeaderText(G),l=Math.min(l,Y.sphericalHarmonicsDegree),x===Pt.ProgressiveToSplatBuffer&&l>0)throw new ic("PlyLoader.loadFromURL() -> Selected PLY format has spherical harmonics data that cannot be progressively loaded.");M=Y.vertexElement.count,q=Y.headerSizeBytes+Y.bytesPerSplat*M+Y.chunkElement.storageSizeBytes}else{if(x===Pt.ProgressiveToSplatBuffer)throw new ic("PlyLoader.loadFromURL() -> Selected PLY format cannot be progressively loaded.");x=Pt.DownloadBeforeProcessing;return}if(x===Pt.ProgressiveToSplatBuffer){const be=$.CompressionLevels[0].SphericalHarmonicsDegrees[l],ze=g+be.BytesPerSplat*M;A=new ArrayBuffer(ze),$.writeHeaderToBuffer({versionMajor:$.CurrentMajorVersion,versionMinor:$.CurrentMinorVersion,maxSectionCount:m,sectionCount:m,maxSplatCount:M,splatCount:0,compressionLevel:0,sceneCenter:new U},A)}else Ae=new Ie(l);z=Y.headerSizeBytes,H=Y.headerSizeBytes,C=!0}if(C&&F&&ae.length>0&&(v=am(ae,v),V-z>p||V>=q&&!I||le)){const ze=I?Y.sphericalHarmonicsPerSplat:Y.bytesPerSplat,Qe=(I?V:Math.min(q,V))-H,B=Math.floor(Qe/ze),N=B*ze,j=V-H-N,P=H-ae[0].startBytes,ie=new DataView(v,P,N);if(I)_===Rn.PlayCanvasCompressed&&x===Pt.ProgressiveToSplatArray&&(ft.parseSphericalHarmonicsToUncompressedSplatArraySection(Y.chunkElement,Y.shElement,y,y+B-1,ie,0,l,Y.sphericalHarmonicsDegree,Ae),y+=B);else{if(x===Pt.ProgressiveToSplatBuffer){const re=$.CompressionLevels[0].SphericalHarmonicsDegrees[l],fe=R*re.BytesPerSplat+g;_===Rn.PlayCanvasCompressed?ft.parseToUncompressedSplatBufferSection(Y.chunkElement,Y.vertexElement,0,B-1,R,ie,A,fe):cn.parseToUncompressedSplatBufferSection(Y,0,B-1,ie,0,A,fe,l)}else _===Rn.PlayCanvasCompressed?ft.parseToUncompressedSplatArraySection(Y.chunkElement,Y.vertexElement,0,B-1,R,ie,Ae):cn.parseToUncompressedSplatArraySection(Y,0,B-1,ie,0,Ae,l);R+=B,x===Pt.ProgressiveToSplatBuffer&&(S||($.writeSectionHeaderToBuffer({maxSplatCount:M,splatCount:R,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:l},0,A,$.HeaderSizeBytes),S=new $(A,!1)),S.updateLoadedCounts(1,R)),V>=q&&(I=!0)}if(j===0)ae=[];else{let re=[],fe=0;for(let ne=ae.length-1;ne>=0;ne--){const he=ae[ne];if(fe+=he.sizeBytes,re.unshift(he),fe>=j)break}ae=re}z+=p,H+=N}s&&S&&s(S,le),le&&(x===Pt.ProgressiveToSplatBuffer?O.resolve(S):O.resolve(Ae))}t&&t(We,He,Wt.Downloading)};return t&&t(0,"0%",Wt.Downloading),wc(e,Ne,!1,c).then(()=>(t&&t(0,"0%",Wt.Processing),O.promise.then(We=>{if(t&&t(100,"100%",Wt.Done),x===Pt.DownloadBeforeProcessing){const He=ae.map(J=>J.data);return new Blob(He).arrayBuffer().then(J=>Xd.loadFromFileData(J,r,o,a,l,u,f,d,h))}else return x===Pt.ProgressiveToSplatBuffer?We:Zn(()=>lm(We,a,r,o,u,f,d,h))})))}static loadFromFileData(e,t,n,s,r=0,o,a,l,c){return s?Zn(()=>om.parseToUncompressedSplatArray(e,r)).then(u=>lm(u,s,t,n,o,a,l,c)):Zn(()=>om.parseToUncompressedSplatBuffer(e,r))}}const TE=i=>new ReadableStream({async start(e){e.enqueue(i),e.close()}});async function EE(i){try{const e=TE(i);if(!e)throw new Error("Failed to create stream from data");return await wE(e)}catch(e){throw console.error("Error decompressing gzipped data:",e),e}}async function wE(i){const e=i.pipeThrough(new DecompressionStream("gzip")),n=await new Response(e).arrayBuffer();return new Uint8Array(n)}const RE=1347635022,IE=1,DE=.15;function PE(i){const e=i>>15&1,t=i>>10&31,n=i&1023,s=e===1?-1:1;return t===0?s*Math.pow(2,-14)*n/1024:t===31?n!==0?NaN:s*(1/0):s*Math.pow(2,t-15)*(1+n/1024)}function FE(i){return(i-128)/128}function vr(i){switch(i){case 0:return 0;case 1:return 3;case 2:return 8;case 3:return 15;default:return console.error(`[SPZ: ERROR] Unsupported SH degree: ${i}`),0}}const LE=(function(){let i=[];const e=new Ft,t=Ie.OFFSET.X,n=Ie.OFFSET.Y,s=Ie.OFFSET.Z,r=Ie.OFFSET.SCALE0,o=Ie.OFFSET.SCALE1,a=Ie.OFFSET.SCALE2,l=Ie.OFFSET.ROTATION0,c=Ie.OFFSET.ROTATION1,u=Ie.OFFSET.ROTATION2,f=Ie.OFFSET.ROTATION3,d=Ie.OFFSET.FDC0,h=Ie.OFFSET.FDC1,x=Ie.OFFSET.FDC2,p=Ie.OFFSET.OPACITY,g=[vr(0),vr(1),vr(2),vr(3)],m=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(_,v,A){A=Math.min(v,A);const S=Ie.createSplat(A);_.scale[0]!==void 0?(S[r]=_.scale[0],S[o]=_.scale[1],S[a]=_.scale[2]):(S[r]=.01,S[o]=.01,S[a]=.01),_.color[0]!==void 0?(S[d]=_.color[0],S[h]=_.color[1],S[x]=_.color[2]):i[RED]!==void 0?(S[d]=i[RED]*255,S[h]=i[GREEN]*255,S[x]=i[BLUE]*255):(S[d]=0,S[h]=0,S[x]=0),_.alpha!==void 0&&(S[p]=_.alpha),S[d]=Bt(Math.floor(S[d]),0,255),S[h]=Bt(Math.floor(S[h]),0,255),S[x]=Bt(Math.floor(S[x]),0,255),S[p]=Bt(Math.floor(S[p]),0,255);let b=g[A],M=g[v];for(let R=0;R<3;++R)for(let y=0;y<15;++y){const C=m[R*15+y];y<b&&y<M&&(S[Ie.OFFSET.FRC0+C]=_.sh[R*M+y])}return e.set(_.rotation[3],_.rotation[0],_.rotation[1],_.rotation[2]),e.normalize(),S[l]=e.x,S[c]=e.y,S[u]=e.z,S[f]=e.w,S[t]=_.position[0],S[n]=_.position[1],S[s]=_.position[2],S}})();function BE(i,e,t,n){return!(i.positions.length!==e*3*(n?2:3)||i.scales.length!==e*3||i.rotations.length!==e*3||i.alphas.length!==e||i.colors.length!==e*3||i.sh.length!==e*t*3)}function cm(i,e,t,n,s){e=Math.min(e,i.shDegree);const r=i.numPoints,o=vr(i.shDegree),a=i.positions.length===r*3*2;if(!BE(i,r,o,a))return null;const l={position:[],scale:[],rotation:[],alpha:void 0,color:[],sh:[]};let c;a&&(c=new Uint16Array(i.positions.buffer,i.positions.byteOffset,r*3));const u=1/(1<<i.fractionalBits),f=vr(i.shDegree),d=.28209479177387814;for(let h=0;h<r;h++){if(a)for(let _=0;_<3;_++)l.position[_]=PE(c[h*3+_]);else for(let _=0;_<3;_++){const v=h*9+_*3;let A=i.positions[v];A|=i.positions[v+1]<<8,A|=i.positions[v+2]<<16,A|=A&8388608?4278190080:0,l.position[_]=A*u}for(let _=0;_<3;_++)l.scale[_]=Math.exp(i.scales[h*3+_]/16-10);const x=i.rotations.subarray(h*3,h*3+3),p=[x[0]/127.5-1,x[1]/127.5-1,x[2]/127.5-1];l.rotation[0]=p[0],l.rotation[1]=p[1],l.rotation[2]=p[2];const g=p[0]*p[0]+p[1]*p[1]+p[2]*p[2];l.rotation[3]=Math.sqrt(Math.max(0,1-g)),l.alpha=Math.floor(i.alphas[h]);for(let _=0;_<3;_++)l.color[_]=Math.floor(((i.colors[h*3+_]/255-.5)/DE*d+.5)*255);for(let _=0;_<3;_++)for(let v=0;v<f;v++)l.sh[_*f+v]=FE(i.sh[f*3*h+v*3+_]);const m=LE(l,i.shDegree,e);if(t){const _=$.CompressionLevels[0].SphericalHarmonicsDegrees[e].BytesPerSplat,v=h*_+s;$.writeSplatDataToSectionBuffer(m,n,v,0,e)}else n.addSplat(m)}}const UE=16,OE=1e7;function NE(i){const e=new DataView(i);let t=0;const n={magic:e.getUint32(t,!0),version:e.getUint32(t+4,!0),numPoints:e.getUint32(t+8,!0),shDegree:e.getUint8(t+12),fractionalBits:e.getUint8(t+13),flags:e.getUint8(t+14),reserved:e.getUint8(t+15)};if(t+=UE,n.magic!==RE)return console.error("[SPZ ERROR] deserializePackedGaussians: header not found"),null;if(n.version<1||n.version>2)return console.error(`[SPZ ERROR] deserializePackedGaussians: version not supported: ${n.version}`),null;if(n.numPoints>OE)return console.error(`[SPZ ERROR] deserializePackedGaussians: Too many points: ${n.numPoints}`),null;if(n.shDegree>3)return console.error(`[SPZ ERROR] deserializePackedGaussians: Unsupported SH degree: ${n.shDegree}`),null;const s=n.numPoints,r=vr(n.shDegree),o=n.version===1,a={numPoints:s,shDegree:n.shDegree,fractionalBits:n.fractionalBits,antialiased:(n.flags&IE)!==0,positions:new Uint8Array(s*3*(o?2:3)),scales:new Uint8Array(s*3),rotations:new Uint8Array(s*3),alphas:new Uint8Array(s),colors:new Uint8Array(s*3),sh:new Uint8Array(s*r*3)};try{const l=new Uint8Array(i);let c=a.positions.length,u=t;if(a.positions.set(l.slice(u,u+c)),u+=c,a.alphas.set(l.slice(u,u+a.alphas.length)),u+=a.alphas.length,a.colors.set(l.slice(u,u+a.colors.length)),u+=a.colors.length,a.scales.set(l.slice(u,u+a.scales.length)),u+=a.scales.length,a.rotations.set(l.slice(u,u+a.rotations.length)),u+=a.rotations.length,a.sh.set(l.slice(u,u+a.sh.length)),u+a.sh.length!==i.byteLength)return console.error("[SPZ ERROR] deserializePackedGaussians: incorrect buffer size"),null}catch(l){return console.error("[SPZ ERROR] deserializePackedGaussians: read error",l),null}return a}async function zE(i){try{const e=await EE(i);return NE(e.buffer)}catch(e){return console.error("[SPZ ERROR] loadSpzPacked: decompression error",e),null}}class qd{static loadFromURL(e,t,n,s,r=!0,o=0,a,l,c,u,f){return t&&t(0,"0%",Wt.Downloading),wc(e,t,!0,a).then(d=>(t&&t(0,"0%",Wt.Processing),qd.loadFromFileData(d,n,s,r,o,l,c,u,f)))}static async loadFromFileData(e,t,n,s,r=0,o,a,l,c){await Zn();const u=await zE(e);r=Math.min(u.shDegree,r);const f=new Ie(r);if(s)return cm(u,r,!1,f,0),Ya.getStandardGenerator(t,n,o,a,l,c).generateFromUncompressedSplatArray(f);{const{splatBuffer:d,splatBufferDataOffsetBytes:h}=$.preallocateUncompressed(u.numPoints,r);return cm(u,r,!0,d.bufferData,h),d}}}class St{static RowSizeBytes=32;static CenterSizeBytes=12;static ScaleSizeBytes=12;static RotationSizeBytes=4;static ColorSizeBytes=4;static parseToUncompressedSplatBufferSection(e,t,n,s,r,o){const a=$.CompressionLevels[0].BytesPerCenter,l=$.CompressionLevels[0].BytesPerScale,c=$.CompressionLevels[0].BytesPerRotation,u=$.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat;for(let f=e;f<=t;f++){const d=f*St.RowSizeBytes+s,h=new Float32Array(n,d,3),x=new Float32Array(n,d+St.CenterSizeBytes,3),p=new Uint8Array(n,d+St.CenterSizeBytes+St.ScaleSizeBytes,4),g=new Uint8Array(n,d+St.CenterSizeBytes+St.ScaleSizeBytes+St.RotationSizeBytes,4),m=new Ft((g[1]-128)/128,(g[2]-128)/128,(g[3]-128)/128,(g[0]-128)/128);m.normalize();const _=f*u+o,v=new Float32Array(r,_,3),A=new Float32Array(r,_+a,3),S=new Float32Array(r,_+a+l,4),b=new Uint8Array(r,_+a+l+c,4);v[0]=h[0],v[1]=h[1],v[2]=h[2],A[0]=x[0],A[1]=x[1],A[2]=x[2],S[0]=m.w,S[1]=m.x,S[2]=m.y,S[3]=m.z,b[0]=p[0],b[1]=p[1],b[2]=p[2],b[3]=p[3]}}static parseToUncompressedSplatArraySection(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*St.RowSizeBytes+s,l=new Float32Array(n,a,3),c=new Float32Array(n,a+St.CenterSizeBytes,3),u=new Uint8Array(n,a+St.CenterSizeBytes+St.ScaleSizeBytes,4),f=new Uint8Array(n,a+St.CenterSizeBytes+St.ScaleSizeBytes+St.RotationSizeBytes,4),d=new Ft((f[1]-128)/128,(f[2]-128)/128,(f[3]-128)/128,(f[0]-128)/128);d.normalize(),r.addSplatFromComonents(l[0],l[1],l[2],c[0],c[1],c[2],d.w,d.x,d.y,d.z,u[0],u[1],u[2],u[3])}}static parseStandardSplatToUncompressedSplatArray(e){const t=e.byteLength/St.RowSizeBytes,n=new Ie;for(let s=0;s<t;s++){const r=s*St.RowSizeBytes,o=new Float32Array(e,r,3),a=new Float32Array(e,r+St.CenterSizeBytes,3),l=new Uint8Array(e,r+St.CenterSizeBytes+St.ScaleSizeBytes,4),c=new Uint8Array(e,r+St.CenterSizeBytes+St.ScaleSizeBytes+St.ColorSizeBytes,4),u=new Ft((c[1]-128)/128,(c[2]-128)/128,(c[3]-128)/128,(c[0]-128)/128);u.normalize(),n.addSplatFromComonents(o[0],o[1],o[2],a[0],a[1],a[2],u.w,u.x,u.y,u.z,l[0],l[1],l[2],l[3])}return n}}function um(i,e,t,n,s,r,o,a){return e?Ya.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):$.generateFromUncompressedSplatArrays([i],t,0,new U)}class Yd{static loadFromURL(e,t,n,s,r,o,a=!0,l,c,u,f,d){let h=n?Pt.ProgressiveToSplatBuffer:Pt.ProgressiveToSplatArray;a&&(h=Pt.ProgressiveToSplatArray);const x=$.HeaderSizeBytes+$.SectionHeaderSizeBytes,p=bt.ProgressiveLoadSectionSize,g=1;let m,_,v,A=0,S=0,b;const M=Ld();let R=0,y=0,C=[];const F=(I,O,z,H)=>{const V=I>=100;if(z&&C.push(z),h===Pt.DownloadBeforeProcessing){V&&M.resolve(C);return}if(!H){if(n)throw new ic("Cannon directly load .splat because no file size info is available.");h=Pt.DownloadBeforeProcessing;return}if(!m){A=H/St.RowSizeBytes,m=new ArrayBuffer(H);const q=$.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,G=x+q*A;h===Pt.ProgressiveToSplatBuffer?(_=new ArrayBuffer(G),$.writeHeaderToBuffer({versionMajor:$.CurrentMajorVersion,versionMinor:$.CurrentMinorVersion,maxSectionCount:g,sectionCount:g,maxSplatCount:A,splatCount:S,compressionLevel:0,sceneCenter:new U},_)):b=new Ie(0)}if(z){new Uint8Array(m,y,z.byteLength).set(new Uint8Array(z)),y+=z.byteLength;const q=y-R;if(q>p||V){const Y=(V?q:p)/St.RowSizeBytes,ae=S+Y;h===Pt.ProgressiveToSplatBuffer?St.parseToUncompressedSplatBufferSection(S,ae-1,m,0,_,x):St.parseToUncompressedSplatArraySection(S,ae-1,m,0,b),S=ae,h===Pt.ProgressiveToSplatBuffer&&(v||($.writeSectionHeaderToBuffer({maxSplatCount:A,splatCount:S,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0},0,_,$.HeaderSizeBytes),v=new $(_,!1)),v.updateLoadedCounts(1,S),s&&s(v,V)),R+=p}}V&&(h===Pt.ProgressiveToSplatBuffer?M.resolve(v):M.resolve(b)),t&&t(I,O,Wt.Downloading)};return t&&t(0,"0%",Wt.Downloading),wc(e,F,!1,l).then(()=>(t&&t(0,"0%",Wt.Processing),M.promise.then(I=>(t&&t(100,"100%",Wt.Done),h===Pt.DownloadBeforeProcessing?new Blob(C).arrayBuffer().then(O=>Yd.loadFromFileData(O,r,o,a,c,u,f,d)):h===Pt.ProgressiveToSplatBuffer?I:Zn(()=>um(I,a,r,o,c,u,f,d))))))}static loadFromFileData(e,t,n,s,r,o,a,l){return Zn(()=>{const c=St.parseStandardSplatToUncompressedSplatArray(e);return um(c,s,t,n,r,o,a,l)})}}class pa{static checkVersion(e){const t=$.CurrentMajorVersion,n=$.CurrentMinorVersion,s=$.parseHeader(e);if(s.versionMajor===t&&s.versionMinor>=n||s.versionMajor>t)return!0;throw new Error(`KSplat version not supported: v${s.versionMajor}.${s.versionMinor}. Minimum required: v${t}.${n}`)}static loadFromURL(e,t,n,s,r){let o,a,l,c,u=!1,f=!1,d,h=[],x=!1,p=!1,g=0,m=0,_=0,v=!1,A=!1,S=!1,b=[];const M=Ld(),R=()=>{!u&&!f&&g>=$.HeaderSizeBytes&&(f=!0,new Blob(b).arrayBuffer().then(H=>{l=new ArrayBuffer($.HeaderSizeBytes),new Uint8Array(l).set(new Uint8Array(H,0,$.HeaderSizeBytes)),pa.checkVersion(l),f=!1,u=!0,c=$.parseHeader(l),window.setTimeout(()=>{F()},1)}))};let y=0;const C=()=>{y===0&&(y++,window.setTimeout(()=>{y--,I()},1))},F=()=>{const z=()=>{p=!0,new Blob(b).arrayBuffer().then(V=>{p=!1,x=!0,d=new ArrayBuffer(c.maxSectionCount*$.SectionHeaderSizeBytes),new Uint8Array(d).set(new Uint8Array(V,$.HeaderSizeBytes,c.maxSectionCount*$.SectionHeaderSizeBytes)),h=$.parseSectionHeaders(c,d,0,!1);let q=0;for(let Y=0;Y<c.maxSectionCount;Y++)q+=h[Y].storageSizeBytes;const G=$.HeaderSizeBytes+c.maxSectionCount*$.SectionHeaderSizeBytes+q;if(!o){o=new ArrayBuffer(G);let Y=0;for(let ae=0;ae<b.length;ae++){const Ae=b[ae];new Uint8Array(o,Y,Ae.byteLength).set(new Uint8Array(Ae)),Y+=Ae.byteLength}}_=$.HeaderSizeBytes+$.SectionHeaderSizeBytes*c.maxSectionCount;for(let Y=0;Y<=h.length&&Y<c.maxSectionCount;Y++)_+=h[Y].storageSizeBytes;C()})};!p&&!x&&u&&g>=$.HeaderSizeBytes+$.SectionHeaderSizeBytes*c.maxSectionCount&&z()},I=()=>{if(S)return;S=!0;const z=()=>{if(S=!1,x){if(A)return;if(v=g>=_,g-m>bt.ProgressiveLoadSectionSize||v){m+=bt.ProgressiveLoadSectionSize,A=m>=_,a||(a=new $(o,!1));const V=$.HeaderSizeBytes+$.SectionHeaderSizeBytes*c.maxSectionCount;let q=0,G=0,Y=0;for(let Pe=0;Pe<c.maxSectionCount;Pe++){const Ne=h[Pe],We=q+Ne.partiallyFilledBucketCount*4+Ne.bucketStorageSizeBytes*Ne.bucketCount,He=V+We;if(m>=He){G++;const J=m-He,ze=$.CompressionLevels[c.compressionLevel].SphericalHarmonicsDegrees[Ne.sphericalHarmonicsDegree].BytesPerSplat;let Ce=Math.floor(J/ze);Ce=Math.min(Ce,Ne.maxSplatCount),Y+=Ce,a.updateLoadedCounts(G,Y),a.updateSectionLoadedCounts(Pe,Ce)}else break;q+=Ne.storageSizeBytes}s(a,A);const ae=m/_*100,Ae=ae.toFixed(2)+"%";t&&t(ae,Ae,Wt.Downloading),A?M.resolve(a):I()}}};window.setTimeout(z,bt.ProgressiveLoadSectionDelayDuration)};return wc(e,(z,H,V)=>{V&&(b.push(V),o&&new Uint8Array(o,g,V.byteLength).set(new Uint8Array(V)),g+=V.byteLength),n?(R(),F(),I()):t&&t(z,H,Wt.Downloading)},!n,r).then(z=>(t&&t(0,"0%",Wt.Processing),(n?M.promise:pa.loadFromFileData(z)).then(V=>(t&&t(100,"100%",Wt.Done),V))))}static loadFromFileData(e){return Zn(()=>(pa.checkVersion(e),new $(e)))}static downloadFile=(function(){let e;return function(t,n){const s=new Blob([t.bufferData],{type:"application/octet-stream"});e||(e=document.createElement("a"),document.body.appendChild(e)),e.download=n,e.href=URL.createObjectURL(s),e.click()}})()}const zn={Splat:0,KSplat:1,Ply:2,Spz:3},fm=i=>i.endsWith(".ply")?zn.Ply:i.endsWith(".splat")?zn.Splat:i.endsWith(".ksplat")?zn.KSplat:i.endsWith(".spz")?zn.Spz:null,dm={type:"change"},Eu={type:"start"},hm={type:"end"},Rl=new Rd,pm=new Ps,kE=Math.cos(70*Ds.DEG2RAD);class Il extends Rr{constructor(e,t){super(),this.object=e,this.domElement=t,this.domElement.style.touchAction="none",this.enabled=!0,this.target=new U,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"KeyA",UP:"KeyW",RIGHT:"KeyD",BOTTOM:"KeyS"},this.mouseButtons={LEFT:Lr.ROTATE,MIDDLE:Lr.DOLLY,RIGHT:Lr.PAN},this.touches={ONE:Br.ROTATE,TWO:Br.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this.getPolarAngle=function(){return a.phi},this.getAzimuthalAngle=function(){return a.theta},this.getDistance=function(){return this.object.position.distanceTo(this.target)},this.listenToKeyEvents=function(k){k.addEventListener("keydown",T),this._domElementKeyEvents=k},this.stopListenToKeyEvents=function(){this._domElementKeyEvents.removeEventListener("keydown",T),this._domElementKeyEvents=null},this.saveState=function(){n.target0.copy(n.target),n.position0.copy(n.object.position),n.zoom0=n.object.zoom},this.reset=function(){n.target.copy(n.target0),n.object.position.copy(n.position0),n.object.zoom=n.zoom0,this.clearDampedRotation(),this.clearDampedPan(),n.object.updateProjectionMatrix(),n.dispatchEvent(dm),n.update(),r=s.NONE},this.clearDampedRotation=function(){l.theta=0,l.phi=0},this.clearDampedPan=function(){u.set(0,0,0)},this.update=(function(){const k=new U,ee=new Ft().setFromUnitVectors(e.up,new U(0,1,0)),me=ee.clone().invert(),Te=new U,Fe=new Ft,Re=new U,Ye=2*Math.PI;return function(){ee.setFromUnitVectors(e.up,new U(0,1,0)),me.copy(ee).invert();const De=n.object.position;k.copy(De).sub(n.target),k.applyQuaternion(ee),a.setFromVector3(k),n.autoRotate&&r===s.NONE&&F(y()),n.enableDamping?(a.theta+=l.theta*n.dampingFactor,a.phi+=l.phi*n.dampingFactor):(a.theta+=l.theta,a.phi+=l.phi);let ye=n.minAzimuthAngle,Me=n.maxAzimuthAngle;isFinite(ye)&&isFinite(Me)&&(ye<-Math.PI?ye+=Ye:ye>Math.PI&&(ye-=Ye),Me<-Math.PI?Me+=Ye:Me>Math.PI&&(Me-=Ye),ye<=Me?a.theta=Math.max(ye,Math.min(Me,a.theta)):a.theta=a.theta>(ye+Me)/2?Math.max(ye,a.theta):Math.min(Me,a.theta)),a.phi=Math.max(n.minPolarAngle,Math.min(n.maxPolarAngle,a.phi)),a.makeSafe(),n.enableDamping===!0?n.target.addScaledVector(u,n.dampingFactor):n.target.add(u),n.zoomToCursor&&b||n.object.isOrthographicCamera?a.radius=Y(a.radius):a.radius=Y(a.radius*c),k.setFromSpherical(a),k.applyQuaternion(me),De.copy(n.target).add(k),n.object.lookAt(n.target),n.enableDamping===!0?(l.theta*=1-n.dampingFactor,l.phi*=1-n.dampingFactor,u.multiplyScalar(1-n.dampingFactor)):(l.set(0,0,0),u.set(0,0,0));let ve=!1;if(n.zoomToCursor&&b){let de=null;if(n.object.isPerspectiveCamera){const ke=k.length();de=Y(ke*c);const je=ke-de;n.object.position.addScaledVector(A,je),n.object.updateMatrixWorld()}else if(n.object.isOrthographicCamera){const ke=new U(S.x,S.y,0);ke.unproject(n.object),n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),ve=!0;const je=new U(S.x,S.y,0);je.unproject(n.object),n.object.position.sub(je).add(ke),n.object.updateMatrixWorld(),de=k.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),n.zoomToCursor=!1;de!==null&&(this.screenSpacePanning?n.target.set(0,0,-1).transformDirection(n.object.matrix).multiplyScalar(de).add(n.object.position):(Rl.origin.copy(n.object.position),Rl.direction.set(0,0,-1).transformDirection(n.object.matrix),Math.abs(n.object.up.dot(Rl.direction))<kE?e.lookAt(n.target):(pm.setFromNormalAndCoplanarPoint(n.object.up,n.target),Rl.intersectPlane(pm,n.target))))}else n.object.isOrthographicCamera&&(n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),ve=!0);return c=1,b=!1,ve||Te.distanceToSquared(n.object.position)>o||8*(1-Fe.dot(n.object.quaternion))>o||Re.distanceToSquared(n.target)>0?(n.dispatchEvent(dm),Te.copy(n.object.position),Fe.copy(n.object.quaternion),Re.copy(n.target),ve=!1,!0):!1}})(),this.dispose=function(){n.domElement.removeEventListener("contextmenu",ce),n.domElement.removeEventListener("pointerdown",fe),n.domElement.removeEventListener("pointercancel",he),n.domElement.removeEventListener("wheel",D),n.domElement.removeEventListener("pointermove",ne),n.domElement.removeEventListener("pointerup",he),n._domElementKeyEvents!==null&&(n._domElementKeyEvents.removeEventListener("keydown",T),n._domElementKeyEvents=null)};const n=this,s={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6};let r=s.NONE;const o=1e-6,a=new yp,l=new yp;let c=1;const u=new U,f=new Xe,d=new Xe,h=new Xe,x=new Xe,p=new Xe,g=new Xe,m=new Xe,_=new Xe,v=new Xe,A=new U,S=new Xe;let b=!1;const M=[],R={};function y(){return 2*Math.PI/60/60*n.autoRotateSpeed}function C(){return Math.pow(.95,n.zoomSpeed)}function F(k){l.theta-=k}function I(k){l.phi-=k}const O=(function(){const k=new U;return function(me,Te){k.setFromMatrixColumn(Te,0),k.multiplyScalar(-me),u.add(k)}})(),z=(function(){const k=new U;return function(me,Te){n.screenSpacePanning===!0?k.setFromMatrixColumn(Te,1):(k.setFromMatrixColumn(Te,0),k.crossVectors(n.object.up,k)),k.multiplyScalar(me),u.add(k)}})(),H=(function(){const k=new U;return function(me,Te){const Fe=n.domElement;if(n.object.isPerspectiveCamera){const Re=n.object.position;k.copy(Re).sub(n.target);let Ye=k.length();Ye*=Math.tan(n.object.fov/2*Math.PI/180),O(2*me*Ye/Fe.clientHeight,n.object.matrix),z(2*Te*Ye/Fe.clientHeight,n.object.matrix)}else n.object.isOrthographicCamera?(O(me*(n.object.right-n.object.left)/n.object.zoom/Fe.clientWidth,n.object.matrix),z(Te*(n.object.top-n.object.bottom)/n.object.zoom/Fe.clientHeight,n.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),n.enablePan=!1)}})();function V(k){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c/=k:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function q(k){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c*=k:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function G(k){if(!n.zoomToCursor)return;b=!0;const ee=n.domElement.getBoundingClientRect(),me=k.clientX-ee.left,Te=k.clientY-ee.top,Fe=ee.width,Re=ee.height;S.x=me/Fe*2-1,S.y=-(Te/Re)*2+1,A.set(S.x,S.y,1).unproject(e).sub(e.position).normalize()}function Y(k){return Math.max(n.minDistance,Math.min(n.maxDistance,k))}function ae(k){f.set(k.clientX,k.clientY)}function Ae(k){G(k),m.set(k.clientX,k.clientY)}function Pe(k){x.set(k.clientX,k.clientY)}function Ne(k){d.set(k.clientX,k.clientY),h.subVectors(d,f).multiplyScalar(n.rotateSpeed);const ee=n.domElement;F(2*Math.PI*h.x/ee.clientHeight),I(2*Math.PI*h.y/ee.clientHeight),f.copy(d),n.update()}function We(k){_.set(k.clientX,k.clientY),v.subVectors(_,m),v.y>0?V(C()):v.y<0&&q(C()),m.copy(_),n.update()}function He(k){p.set(k.clientX,k.clientY),g.subVectors(p,x).multiplyScalar(n.panSpeed),H(g.x,g.y),x.copy(p),n.update()}function J(k){G(k),k.deltaY<0?q(C()):k.deltaY>0&&V(C()),n.update()}function le(k){let ee=!1;switch(k.code){case n.keys.UP:k.ctrlKey||k.metaKey||k.shiftKey?I(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):H(0,n.keyPanSpeed),ee=!0;break;case n.keys.BOTTOM:k.ctrlKey||k.metaKey||k.shiftKey?I(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):H(0,-n.keyPanSpeed),ee=!0;break;case n.keys.LEFT:k.ctrlKey||k.metaKey||k.shiftKey?F(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):H(n.keyPanSpeed,0),ee=!0;break;case n.keys.RIGHT:k.ctrlKey||k.metaKey||k.shiftKey?F(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):H(-n.keyPanSpeed,0),ee=!0;break}ee&&(k.preventDefault(),n.update())}function be(){if(M.length===1)f.set(M[0].pageX,M[0].pageY);else{const k=.5*(M[0].pageX+M[1].pageX),ee=.5*(M[0].pageY+M[1].pageY);f.set(k,ee)}}function ze(){if(M.length===1)x.set(M[0].pageX,M[0].pageY);else{const k=.5*(M[0].pageX+M[1].pageX),ee=.5*(M[0].pageY+M[1].pageY);x.set(k,ee)}}function Ce(){const k=M[0].pageX-M[1].pageX,ee=M[0].pageY-M[1].pageY,me=Math.sqrt(k*k+ee*ee);m.set(0,me)}function Qe(){n.enableZoom&&Ce(),n.enablePan&&ze()}function B(){n.enableZoom&&Ce(),n.enableRotate&&be()}function N(k){if(M.length==1)d.set(k.pageX,k.pageY);else{const me=Oe(k),Te=.5*(k.pageX+me.x),Fe=.5*(k.pageY+me.y);d.set(Te,Fe)}h.subVectors(d,f).multiplyScalar(n.rotateSpeed);const ee=n.domElement;F(2*Math.PI*h.x/ee.clientHeight),I(2*Math.PI*h.y/ee.clientHeight),f.copy(d)}function j(k){if(M.length===1)p.set(k.pageX,k.pageY);else{const ee=Oe(k),me=.5*(k.pageX+ee.x),Te=.5*(k.pageY+ee.y);p.set(me,Te)}g.subVectors(p,x).multiplyScalar(n.panSpeed),H(g.x,g.y),x.copy(p)}function P(k){const ee=Oe(k),me=k.pageX-ee.x,Te=k.pageY-ee.y,Fe=Math.sqrt(me*me+Te*Te);_.set(0,Fe),v.set(0,Math.pow(_.y/m.y,n.zoomSpeed)),V(v.y),m.copy(_)}function ie(k){n.enableZoom&&P(k),n.enablePan&&j(k)}function re(k){n.enableZoom&&P(k),n.enableRotate&&N(k)}function fe(k){n.enabled!==!1&&(M.length===0&&(n.domElement.setPointerCapture(k.pointerId),n.domElement.addEventListener("pointermove",ne),n.domElement.addEventListener("pointerup",he)),Z(k),k.pointerType==="touch"?X(k):te(k))}function ne(k){n.enabled!==!1&&(k.pointerType==="touch"?se(k):ge(k))}function he(k){Be(k),M.length===0&&(n.domElement.releasePointerCapture(k.pointerId),n.domElement.removeEventListener("pointermove",ne),n.domElement.removeEventListener("pointerup",he)),n.dispatchEvent(hm),r=s.NONE}function te(k){let ee;switch(k.button){case 0:ee=n.mouseButtons.LEFT;break;case 1:ee=n.mouseButtons.MIDDLE;break;case 2:ee=n.mouseButtons.RIGHT;break;default:ee=-1}switch(ee){case Lr.DOLLY:if(n.enableZoom===!1)return;Ae(k),r=s.DOLLY;break;case Lr.ROTATE:if(k.ctrlKey||k.metaKey||k.shiftKey){if(n.enablePan===!1)return;Pe(k),r=s.PAN}else{if(n.enableRotate===!1)return;ae(k),r=s.ROTATE}break;case Lr.PAN:if(k.ctrlKey||k.metaKey||k.shiftKey){if(n.enableRotate===!1)return;ae(k),r=s.ROTATE}else{if(n.enablePan===!1)return;Pe(k),r=s.PAN}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(Eu)}function ge(k){switch(r){case s.ROTATE:if(n.enableRotate===!1)return;Ne(k);break;case s.DOLLY:if(n.enableZoom===!1)return;We(k);break;case s.PAN:if(n.enablePan===!1)return;He(k);break}}function D(k){n.enabled===!1||n.enableZoom===!1||r!==s.NONE||(k.preventDefault(),n.dispatchEvent(Eu),J(k),n.dispatchEvent(hm))}function T(k){n.enabled===!1||n.enablePan===!1||le(k)}function X(k){switch(Se(k),M.length){case 1:switch(n.touches.ONE){case Br.ROTATE:if(n.enableRotate===!1)return;be(),r=s.TOUCH_ROTATE;break;case Br.PAN:if(n.enablePan===!1)return;ze(),r=s.TOUCH_PAN;break;default:r=s.NONE}break;case 2:switch(n.touches.TWO){case Br.DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;Qe(),r=s.TOUCH_DOLLY_PAN;break;case Br.DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;B(),r=s.TOUCH_DOLLY_ROTATE;break;default:r=s.NONE}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(Eu)}function se(k){switch(Se(k),r){case s.TOUCH_ROTATE:if(n.enableRotate===!1)return;N(k),n.update();break;case s.TOUCH_PAN:if(n.enablePan===!1)return;j(k),n.update();break;case s.TOUCH_DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;ie(k),n.update();break;case s.TOUCH_DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;re(k),n.update();break;default:r=s.NONE}}function ce(k){n.enabled!==!1&&k.preventDefault()}function Z(k){M.push(k)}function Be(k){delete R[k.pointerId];for(let ee=0;ee<M.length;ee++)if(M[ee].pointerId==k.pointerId){M.splice(ee,1);return}}function Se(k){let ee=R[k.pointerId];ee===void 0&&(ee=new Xe,R[k.pointerId]=ee),ee.set(k.pageX,k.pageY)}function Oe(k){const ee=k.pointerId===M[0].pointerId?M[1]:M[0];return R[ee.pointerId]}n.domElement.addEventListener("contextmenu",ce),n.domElement.addEventListener("pointerdown",fe),n.domElement.addEventListener("pointercancel",he),n.domElement.addEventListener("wheel",D,{passive:!1}),this.update()}}const HE=(i,e,t,n,s)=>{const r=performance.now();let o=i.style.display==="none"?0:parseFloat(i.style.opacity);isNaN(o)&&(o=1);const a=window.setInterval(()=>{const c=performance.now()-r;let u=Math.min(c/n,1);u>.999&&(u=1);let f;e?(f=(1-u)*o,f<1e-4&&(f=0)):f=(1-o)*u+o,f>0?(i.style.display=t,i.style.opacity=f):i.style.display="none",u>=1&&(s&&s(),window.clearInterval(a))},16);return a},VE=500;class Qd{static elementIDGen=0;constructor(e,t){this.taskIDGen=0,this.elementID=Qd.elementIDGen++,this.tasks=[],this.message=e||"Loading...",this.container=t||document.body,this.spinnerContainerOuter=document.createElement("div"),this.spinnerContainerOuter.className=`spinnerOuterContainer${this.elementID}`,this.spinnerContainerOuter.style.display="none",this.spinnerContainerPrimary=document.createElement("div"),this.spinnerContainerPrimary.className=`spinnerContainerPrimary${this.elementID}`,this.spinnerPrimary=document.createElement("div"),this.spinnerPrimary.classList.add(`spinner${this.elementID}`,`spinnerPrimary${this.elementID}`),this.messageContainerPrimary=document.createElement("div"),this.messageContainerPrimary.classList.add(`messageContainer${this.elementID}`,`messageContainerPrimary${this.elementID}`),this.messageContainerPrimary.innerHTML=this.message,this.spinnerContainerMin=document.createElement("div"),this.spinnerContainerMin.className=`spinnerContainerMin${this.elementID}`,this.spinnerMin=document.createElement("div"),this.spinnerMin.classList.add(`spinner${this.elementID}`,`spinnerMin${this.elementID}`),this.messageContainerMin=document.createElement("div"),this.messageContainerMin.classList.add(`messageContainer${this.elementID}`,`messageContainerMin${this.elementID}`),this.messageContainerMin.innerHTML=this.message,this.spinnerContainerPrimary.appendChild(this.spinnerPrimary),this.spinnerContainerPrimary.appendChild(this.messageContainerPrimary),this.spinnerContainerOuter.appendChild(this.spinnerContainerPrimary),this.spinnerContainerMin.appendChild(this.spinnerMin),this.spinnerContainerMin.appendChild(this.messageContainerMin),this.spinnerContainerOuter.appendChild(this.spinnerContainerMin);const n=document.createElement("style");n.innerHTML=`

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

        `,this.spinnerContainerOuter.appendChild(n),this.container.appendChild(this.spinnerContainerOuter),this.setMinimized(!1,!0),this.fadeTransitions=[]}addTask(e){const t={message:e,id:this.taskIDGen++};return this.tasks.push(t),this.update(),t.id}removeTask(e){let t=0;for(let n of this.tasks){if(n.id===e){this.tasks.splice(t,1);break}t++}this.update()}removeAllTasks(){this.tasks=[],this.update()}setMessageForTask(e,t){for(let n of this.tasks)if(n.id===e){n.message=t;break}this.update()}update(){this.tasks.length>0?(this.show(),this.setMessage(this.tasks[this.tasks.length-1].message)):this.hide()}show(){this.spinnerContainerOuter.style.display="block",this.visible=!0}hide(){this.spinnerContainerOuter.style.display="none",this.visible=!1}setContainer(e){this.container&&this.spinnerContainerOuter.parentElement===this.container&&this.container.removeChild(this.spinnerContainerOuter),e&&(this.container=e,this.container.appendChild(this.spinnerContainerOuter),this.spinnerContainerOuter.style.zIndex=this.container.style.zIndex+1)}setMinimized(e,t){const n=(s,r,o,a,l)=>{o?s.style.display=r?a:"none":this.fadeTransitions[l]=HE(s,!r,a,VE,()=>{this.fadeTransitions[l]=null})};n(this.spinnerContainerPrimary,!e,t,"block",0),n(this.spinnerContainerMin,e,t,"flex",1),this.minimized=e}setMessage(e){this.messageContainerPrimary.innerHTML=e,this.messageContainerMin.innerHTML=e}}class GE{constructor(e){this.idGen=0,this.tasks=[],this.container=e||document.body,this.progressBarContainerOuter=document.createElement("div"),this.progressBarContainerOuter.className="progressBarOuterContainer",this.progressBarContainerOuter.style.display="none",this.progressBarBox=document.createElement("div"),this.progressBarBox.className="progressBarBox",this.progressBarBackground=document.createElement("div"),this.progressBarBackground.className="progressBarBackground",this.progressBar=document.createElement("div"),this.progressBar.className="progressBar",this.progressBarBackground.appendChild(this.progressBar),this.progressBarBox.appendChild(this.progressBarBackground),this.progressBarContainerOuter.appendChild(this.progressBarBox);const t=document.createElement("style");t.innerHTML=`

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

        `,this.progressBarContainerOuter.appendChild(t),this.container.appendChild(this.progressBarContainerOuter)}show(){this.progressBarContainerOuter.style.display="block"}hide(){this.progressBarContainerOuter.style.display="none"}setProgress(e){this.progressBar.style.width=e+"%"}setContainer(e){this.container&&this.progressBarContainerOuter.parentElement===this.container&&this.container.removeChild(this.progressBarContainerOuter),e&&(this.container=e,this.container.appendChild(this.progressBarContainerOuter),this.progressBarContainerOuter.style.zIndex=this.container.style.zIndex+1)}}class WE{constructor(e){this.container=e||document.body,this.infoCells={};const t=[["Camera position","cameraPosition"],["Camera look-at","cameraLookAt"],["Camera up","cameraUp"],["Camera mode","orthographicCamera"],["Cursor position","cursorPosition"],["FPS","fps"],["Rendering:","renderSplatCount"],["Sort time","sortTime"],["Render window","renderWindow"],["Focal adjustment","focalAdjustment"],["Splat scale","splatScale"],["Point cloud mode","pointCloudMode"]];this.infoPanelContainer=document.createElement("div");const n=document.createElement("style");n.innerHTML=`

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

        `,this.infoPanelContainer.append(n),this.infoPanel=document.createElement("div"),this.infoPanel.className="infoPanel";const s=document.createElement("div");s.style.display="table";for(let r of t){const o=document.createElement("div");o.style.display="table-row",o.className="info-panel-row";const a=document.createElement("div");a.style.display="table-cell",a.innerHTML=`${r[0]}: `,a.classList.add("info-panel-cell","label-cell");const l=document.createElement("div");l.style.display="table-cell",l.style.width="10px",l.innerHTML=" ",l.className="info-panel-cell";const c=document.createElement("div");c.style.display="table-cell",c.innerHTML="",c.className="info-panel-cell",this.infoCells[r[1]]=c,o.appendChild(a),o.appendChild(l),o.appendChild(c),s.appendChild(o)}this.infoPanel.appendChild(s),this.infoPanelContainer.append(this.infoPanel),this.infoPanelContainer.style.display="none",this.container.appendChild(this.infoPanelContainer),this.visible=!1}update=function(e,t,n,s,r,o,a,l,c,u,f,d,h,x){const p=`${t.x.toFixed(5)}, ${t.y.toFixed(5)}, ${t.z.toFixed(5)}`;if(this.infoCells.cameraPosition.innerHTML!==p&&(this.infoCells.cameraPosition.innerHTML=p),n){const m=n,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cameraLookAt.innerHTML!==_&&(this.infoCells.cameraLookAt.innerHTML=_)}const g=`${s.x.toFixed(5)}, ${s.y.toFixed(5)}, ${s.z.toFixed(5)}`;if(this.infoCells.cameraUp.innerHTML!==g&&(this.infoCells.cameraUp.innerHTML=g),this.infoCells.orthographicCamera.innerHTML=r?"Orthographic":"Perspective",o){const m=o,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cursorPosition.innerHTML=_}else this.infoCells.cursorPosition.innerHTML="N/A";this.infoCells.fps.innerHTML=a,this.infoCells.renderWindow.innerHTML=`${e.x} x ${e.y}`,this.infoCells.renderSplatCount.innerHTML=`${c} splats out of ${l} (${u.toFixed(2)}%)`,this.infoCells.sortTime.innerHTML=`${f.toFixed(3)} ms`,this.infoCells.focalAdjustment.innerHTML=`${d.toFixed(3)}`,this.infoCells.splatScale.innerHTML=`${h.toFixed(3)}`,this.infoCells.pointCloudMode.innerHTML=`${x}`};setContainer(e){this.container&&this.infoPanelContainer.parentElement===this.container&&this.container.removeChild(this.infoPanelContainer),e&&(this.container=e,this.container.appendChild(this.infoPanelContainer),this.infoPanelContainer.style.zIndex=this.container.style.zIndex+1)}show(){this.infoPanelContainer.style.display="block",this.visible=!0}hide(){this.infoPanelContainer.style.display="none",this.visible=!1}}const mm=new U;class XE extends jt{constructor(e=new U(0,0,1),t=new U(0,0,0),n=1,s=.1,r=16776960,o=n*.2,a=o*.2){super(),this.type="ArrowHelper";const l=new La(s,s,n,32);l.translate(0,n/2,0);const c=new La(0,a,o,32);c.translate(0,n,0),this.position.copy(t),this.line=new Qt(l,new Er({color:r,toneMapped:!1})),this.line.matrixAutoUpdate=!1,this.add(this.line),this.cone=new Qt(c,new Er({color:r,toneMapped:!1})),this.cone.matrixAutoUpdate=!1,this.add(this.cone),this.setDirection(e)}setDirection(e){if(e.y>.99999)this.quaternion.set(0,0,0,1);else if(e.y<-.99999)this.quaternion.set(1,0,0,0);else{mm.set(e.z,0,-e.x).normalize();const t=Math.acos(e.y);this.quaternion.setFromAxisAngle(mm,t)}}setColor(e){this.line.material.color.set(e),this.cone.material.color.set(e)}copy(e){return super.copy(e,!1),this.line.copy(e.line),this.cone.copy(e.cone),this}dispose(){this.line.geometry.dispose(),this.line.material.dispose(),this.cone.geometry.dispose(),this.cone.material.dispose()}}class ma{constructor(e){this.threeScene=e,this.splatRenderTarget=null,this.renderTargetCopyQuad=null,this.renderTargetCopyCamera=null,this.meshCursor=null,this.focusMarker=null,this.controlPlane=null,this.debugRoot=null,this.secondaryDebugRoot=null}updateSplatRenderTargetForRenderDimensions(e,t){this.destroySplatRendertarget(),this.splatRenderTarget=new qs(e,t,{format:Pn,stencilBuffer:!1,depthBuffer:!0}),this.splatRenderTarget.depthTexture=new Id(e,t),this.splatRenderTarget.depthTexture.format=To,this.splatRenderTarget.depthTexture.type=hi}destroySplatRendertarget(){this.splatRenderTarget&&(this.splatRenderTarget=null)}setupRenderTargetCopyObjects(){const e={sourceColorTexture:{type:"t",value:null},sourceDepthTexture:{type:"t",value:null}},t=new Ln({vertexShader:`
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
            `,uniforms:e,depthWrite:!1,depthTest:!1,transparent:!0,blending:Z0,blendSrc:Ta,blendSrcAlpha:Ta,blendDst:Ea,blendDstAlpha:Ea});t.extensions.fragDepth=!0,this.renderTargetCopyQuad=new Qt(new Ro(2,2),t),this.renderTargetCopyCamera=new Fd(-1,1,1,-1,0,1)}destroyRenderTargetCopyObjects(){this.renderTargetCopyQuad&&(no(this.renderTargetCopyQuad),this.renderTargetCopyQuad=null)}setupMeshCursor(){if(!this.meshCursor){const e=new Dd(.5,1.5,32),t=new Er({color:16777215}),n=new Qt(e,t);n.rotation.set(0,0,Math.PI),n.position.set(0,1,0);const s=new Qt(e,t);s.position.set(0,-1,0);const r=new Qt(e,t);r.rotation.set(0,0,Math.PI/2),r.position.set(1,0,0);const o=new Qt(e,t);o.rotation.set(0,0,-Math.PI/2),o.position.set(-1,0,0),this.meshCursor=new jt,this.meshCursor.add(n),this.meshCursor.add(s),this.meshCursor.add(r),this.meshCursor.add(o),this.meshCursor.scale.set(.1,.1,.1),this.threeScene.add(this.meshCursor),this.meshCursor.visible=!1}}destroyMeshCursor(){this.meshCursor&&(no(this.meshCursor),this.threeScene.remove(this.meshCursor),this.meshCursor=null)}setMeshCursorVisibility(e){this.meshCursor.visible=e}getMeschCursorVisibility(){return this.meshCursor.visible}setMeshCursorPosition(e){this.meshCursor.position.copy(e)}positionAndOrientMeshCursor(e,t){this.meshCursor.position.copy(e),this.meshCursor.up.copy(t.up),this.meshCursor.lookAt(t.position)}setupFocusMarker(){if(!this.focusMarker){const e=new nc(.5,32,32),t=ma.buildFocusMarkerMaterial();t.depthTest=!1,t.depthWrite=!1,t.transparent=!0,this.focusMarker=new Qt(e,t)}}destroyFocusMarker(){this.focusMarker&&(no(this.focusMarker),this.focusMarker=null)}updateFocusMarker=(function(){const e=new U,t=new $e,n=new U;return function(s,r,o){t.copy(r.matrixWorld).invert(),e.copy(s).applyMatrix4(t),e.normalize().multiplyScalar(10),e.applyMatrix4(r.matrixWorld),n.copy(r.position).sub(s);const a=n.length();this.focusMarker.position.copy(s),this.focusMarker.scale.set(a,a,a),this.focusMarker.material.uniforms.realFocusPosition.value.copy(s),this.focusMarker.material.uniforms.viewport.value.copy(o),this.focusMarker.material.uniformsNeedUpdate=!0}})();setFocusMarkerVisibility(e){this.focusMarker.visible=e}setFocusMarkerOpacity(e){this.focusMarker.material.uniforms.opacity.value=e,this.focusMarker.material.uniformsNeedUpdate=!0}getFocusMarkerOpacity(){return this.focusMarker.material.uniforms.opacity.value}setupControlPlane(){if(!this.controlPlane){const e=new Ro(1,1);e.rotateX(-Math.PI/2);const t=new Er({color:16777215});t.transparent=!0,t.opacity=.6,t.depthTest=!1,t.depthWrite=!1,t.side=ui;const n=new Qt(e,t),s=new U(0,1,0);s.normalize();const r=new U(0,0,0),o=.5,a=.01,l=56576,c=new XE(s,r,o,a,l,.1,.03);this.controlPlane=new jt,this.controlPlane.add(n),this.controlPlane.add(c)}}destroyControlPlane(){this.controlPlane&&(no(this.controlPlane),this.controlPlane=null)}setControlPlaneVisibility(e){this.controlPlane.visible=e}positionAndOrientControlPlane=(function(){const e=new Ft,t=new U(0,1,0);return function(n,s){e.setFromUnitVectors(t,s),this.controlPlane.position.copy(n),this.controlPlane.quaternion.copy(e)}})();addDebugMeshes(){this.debugRoot=this.createDebugMeshes(),this.secondaryDebugRoot=this.createSecondaryDebugMeshes(),this.threeScene.add(this.debugRoot),this.threeScene.add(this.secondaryDebugRoot)}destroyDebugMeshes(){for(let e of[this.debugRoot,this.secondaryDebugRoot])e&&(no(e),this.threeScene.remove(e));this.debugRoot=null,this.secondaryDebugRoot=null}createDebugMeshes(e){const t=new nc(1,32,32),n=new jt,s=(r,o)=>{let a=new Qt(t,ma.buildDebugMaterial(r));a.renderOrder=e,n.add(a),a.position.fromArray(o)};return s(16711680,[-50,0,0]),s(16711680,[50,0,0]),s(65280,[0,0,-50]),s(65280,[0,0,50]),s(16755200,[5,0,5]),n}createSecondaryDebugMeshes(e){const t=new No(3,3,3),n=new jt;let s=12303291;const r=a=>{let l=new Qt(t,ma.buildDebugMaterial(s));l.renderOrder=e,n.add(l),l.position.fromArray(a)};let o=10;return r([-o,0,-o]),r([-o,0,o]),r([o,0,-o]),r([o,0,o]),n}static buildDebugMaterial(e){const t=`
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
        `,s={color:{type:"v3",value:new lt(e)}},r=new Ln({uniforms:s,vertexShader:t,fragmentShader:n,transparent:!1,depthTest:!0,depthWrite:!0,side:Vi});return r.extensions.fragDepth=!0,r}static buildFocusMarkerMaterial(e){const t=`
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
        `,s={color:{type:"v3",value:new lt(e)},realFocusPosition:{type:"v3",value:new U},viewport:{type:"v2",value:new Xe},opacity:{value:0}};return new Ln({uniforms:s,vertexShader:t,fragmentShader:n,transparent:!0,depthTest:!1,depthWrite:!1,side:Vi})}dispose(){this.destroyMeshCursor(),this.destroyFocusMarker(),this.destroyDebugMeshes(),this.destroyControlPlane(),this.destroyRenderTargetCopyObjects(),this.destroySplatRendertarget()}}const qE=new U(1,0,0),YE=new U(0,1,0),QE=new U(0,0,1);class wu{constructor(e=new U,t=new U){this.origin=new U,this.direction=new U,this.setParameters(e,t)}setParameters(e,t){this.origin.copy(e),this.direction.copy(t).normalize()}boxContainsPoint(e,t,n){return!(t.x<e.min.x-n||t.x>e.max.x+n||t.y<e.min.y-n||t.y>e.max.y+n||t.z<e.min.z-n||t.z>e.max.z+n)}intersectBox=(function(){const e=new U,t=[],n=[],s=[];return function(r,o){if(n[0]=this.origin.x,n[1]=this.origin.y,n[2]=this.origin.z,s[0]=this.direction.x,s[1]=this.direction.y,s[2]=this.direction.z,this.boxContainsPoint(r,this.origin,1e-4))return o&&(o.origin.copy(this.origin),o.normal.set(0,0,0),o.distance=-1),!0;for(let a=0;a<3;a++){if(s[a]==0)continue;const l=a==0?qE:a==1?YE:QE,c=s[a]<0?r.max:r.min;let u=-Math.sign(s[a]);t[0]=a==0?c.x:a==1?c.y:c.z;let f=t[0]-n[a];if(f*u<0){const d=(a+1)%3,h=(a+2)%3;if(t[2]=s[d]/s[a]*f+n[d],t[1]=s[h]/s[a]*f+n[h],e.set(t[a],t[h],t[d]),this.boxContainsPoint(r,e,1e-4))return o&&(o.origin.copy(e),o.normal.copy(l).multiplyScalar(u),o.distance=e.sub(this.origin).length()),!0}}return!1}})();intersectSphere=(function(){const e=new U;return function(t,n,s){e.copy(t).sub(this.origin);const r=e.dot(this.direction),o=r*r,l=e.dot(e)-o,c=n*n;if(l>c)return!1;const u=Math.sqrt(c-l),f=r-u,d=r+u;if(d<0)return!1;let h=f<0?d:f;return s&&(s.origin.copy(this.origin).addScaledVector(this.direction,h),s.normal.copy(s.origin).sub(t).normalize(),s.distance=h),!0}})()}class Kd{constructor(){this.origin=new U,this.normal=new U,this.distance=0,this.splatIndex=0}set(e,t,n,s){this.origin.copy(e),this.normal.copy(t),this.distance=n,this.splatIndex=s}clone(){const e=new Kd;return e.origin.copy(this.origin),e.normal.copy(this.normal),e.distance=this.distance,e.splatIndex=this.splatIndex,e}}const os={ThreeD:0,TwoD:1};class KE{constructor(e,t,n=!1){this.ray=new wu(e,t),this.raycastAgainstTrueSplatEllipsoid=n}setFromCameraAndScreenPosition=(function(){const e=new Xe;return function(t,n,s){if(e.x=n.x/s.x*2-1,e.y=(s.y-n.y)/s.y*2-1,t.isPerspectiveCamera)this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(e.x,e.y,.5).unproject(t).sub(this.ray.origin).normalize(),this.camera=t;else if(t.isOrthographicCamera)this.ray.origin.set(e.x,e.y,(t.near+t.far)/(t.near-t.far)).unproject(t),this.ray.direction.set(0,0,-1).transformDirection(t.matrixWorld),this.camera=t;else throw new Error("Raycaster::setFromCameraAndScreenPosition() -> Unsupported camera type")}})();intersectSplatMesh=(function(){const e=new $e,t=new $e,n=new $e,s=new wu,r=new U;return function(o,a=[]){const l=o.getSplatTree();if(l){for(let c=0;c<l.subTrees.length;c++){const u=l.subTrees[c];t.copy(o.matrixWorld),o.dynamicMode&&(o.getSceneTransform(c,n),t.multiply(n)),e.copy(t).invert(),s.origin.copy(this.ray.origin).applyMatrix4(e),s.direction.copy(this.ray.origin).add(this.ray.direction),s.direction.applyMatrix4(e).sub(s.origin).normalize();const f=[];u.rootNode&&this.castRayAtSplatTreeNode(s,l,u.rootNode,f),f.forEach(d=>{d.origin.applyMatrix4(t),d.normal.applyMatrix4(t).normalize(),d.distance=r.copy(d.origin).sub(this.ray.origin).length()}),a.push(...f)}return a.sort((c,u)=>c.distance>u.distance?1:-1),a}}})();castRayAtSplatTreeNode=(function(){const e=new Ot,t=new U,n=new U,s=new Ft,r=new Kd,o=1e-7,a=new U(0,0,0),l=new $e,c=new $e,u=new $e,f=new $e,d=new $e,h=new wu;return function(x,p,g,m=[]){if(x.intersectBox(g.boundingBox)){if(g.data&&g.data.indexes&&g.data.indexes.length>0)for(let _=0;_<g.data.indexes.length;_++){const v=g.data.indexes[_],A=p.splatMesh.getSceneIndexForSplat(v);if(p.splatMesh.getScene(A).visible&&(p.splatMesh.getSplatColor(v,e),p.splatMesh.getSplatCenter(v,t),p.splatMesh.getSplatScaleAndRotation(v,n,s),!(n.x<=o||n.y<=o||p.splatMesh.splatRenderMode===os.ThreeD&&n.z<=o)))if(this.raycastAgainstTrueSplatEllipsoid){c.makeScale(n.x,n.y,n.z),u.makeRotationFromQuaternion(s);const b=Math.log10(e.w)*2;if(l.makeScale(b,b,b),d.copy(l).multiply(u).multiply(c),f.copy(d).invert(),h.origin.copy(x.origin).sub(t).applyMatrix4(f),h.direction.copy(x.origin).add(x.direction).sub(t),h.direction.applyMatrix4(f).sub(h.origin).normalize(),h.intersectSphere(a,1,r)){const M=r.clone();M.splatIndex=v,M.origin.applyMatrix4(d).add(t),m.push(M)}}else{let b=n.x+n.y,M=2;if(p.splatMesh.splatRenderMode===os.ThreeD&&(b+=n.z,M=3),b=b/M,x.intersectSphere(t,b,r)){const R=r.clone();R.splatIndex=v,m.push(R)}}}if(g.children&&g.children.length>0)for(let _ of g.children)this.castRayAtSplatTreeNode(x,p,_,m);return m}}})()}class go{static buildVertexShaderBase(e=!1,t=!1,n=0,s=""){let r=`
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
            uniform float sceneOpacity[${bt.MaxScenes}];
            uniform int sceneVisibility[${bt.MaxScenes}];
        `),e&&(r+=`
            uniform highp mat4 transforms[${bt.MaxScenes}];
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
        uniform float sphericalHarmonics8BitCompressionRangeMin[${bt.MaxScenes}];
        uniform float sphericalHarmonics8BitCompressionRangeMax[${bt.MaxScenes}];

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
        `}static getUniforms(e=!1,t=!1,n=0,s=1,r=!1){const o={sceneCenter:{type:"v3",value:new U},fadeInComplete:{type:"i",value:0},orthographicMode:{type:"i",value:0},visibleRegionFadeStartRadius:{type:"f",value:0},visibleRegionRadius:{type:"f",value:0},currentTime:{type:"f",value:0},firstRenderTime:{type:"f",value:0},centersColorsTexture:{type:"t",value:null},sphericalHarmonicsTexture:{type:"t",value:null},sphericalHarmonicsTextureR:{type:"t",value:null},sphericalHarmonicsTextureG:{type:"t",value:null},sphericalHarmonicsTextureB:{type:"t",value:null},sphericalHarmonics8BitCompressionRangeMin:{type:"f",value:[]},sphericalHarmonics8BitCompressionRangeMax:{type:"f",value:[]},focal:{type:"v2",value:new Xe},orthoZoom:{type:"f",value:1},inverseFocalAdjustment:{type:"f",value:1},viewport:{type:"v2",value:new Xe},basisViewport:{type:"v2",value:new Xe},debugColor:{type:"v3",value:new lt},centersColorsTextureSize:{type:"v2",value:new Xe(1024,1024)},sphericalHarmonicsDegree:{type:"i",value:n},sphericalHarmonicsTextureSize:{type:"v2",value:new Xe(1024,1024)},sphericalHarmonics8BitMode:{type:"i",value:0},sphericalHarmonicsMultiTextureMode:{type:"i",value:0},splatScale:{type:"f",value:s},pointCloudModeEnabled:{type:"i",value:r?1:0},sceneIndexesTexture:{type:"t",value:null},sceneIndexesTextureSize:{type:"v2",value:new Xe(1024,1024)},sceneCount:{type:"i",value:1}};for(let a=0;a<bt.MaxScenes;a++)o.sphericalHarmonics8BitCompressionRangeMin.value.push(-3/2),o.sphericalHarmonics8BitCompressionRangeMax.value.push(bt.SphericalHarmonics8BitCompressionRange/2);if(t){const a=[];for(let c=0;c<bt.MaxScenes;c++)a.push(1);o.sceneOpacity={type:"f",value:a};const l=[];for(let c=0;c<bt.MaxScenes;c++)l.push(1);o.sceneVisibility={type:"i",value:l}}if(e){const a=[];for(let l=0;l<bt.MaxScenes;l++)a.push(new $e);o.transforms={type:"mat4",value:a}}return o}}class sc{static build(e=!1,t=!1,n=!1,s=2048,r=1,o=!1,a=0,l=.3){let u=go.buildVertexShaderBase(e,t,a,`
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
        `);u+=sc.buildVertexShaderProjection(n,t,s,l);const f=sc.buildFragmentShader(),d=go.getUniforms(e,t,a,r,o);return d.covariancesTextureSize={type:"v2",value:new Xe(1024,1024)},d.covariancesTexture={type:"t",value:null},d.covariancesTextureHalfFloat={type:"t",value:null},d.covariancesAreHalfFloat={type:"i",value:0},new Ln({uniforms:d,vertexShader:u,fragmentShader:f,transparent:!0,alphaTest:1,blending:ks,depthTest:!0,depthWrite:!1,side:ui})}static buildVertexShaderProjection(e,t,n,s){let r=`

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
        `,r+=go.getVertexShaderFadeIn(),r+="}",r}static buildFragmentShader(){let e=`
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
        `,e}}class rc{static build(e=!1,t=!1,n=1,s=!1,r=0){let a=go.buildVertexShaderBase(e,t,r,`
            uniform vec2 scaleRotationsTextureSize;
            uniform highp sampler2D scaleRotationsTexture;
            varying mat3 vT;
            varying vec2 vQuadCenter;
            varying vec2 vFragCoord;
        `);a+=rc.buildVertexShaderProjection();const l=rc.buildFragmentShader(),c=go.getUniforms(e,t,r,n,s);return c.scaleRotationsTexture={type:"t",value:null},c.scaleRotationsTextureSize={type:"v2",value:new Xe(1024,1024)},new Ln({uniforms:c,vertexShader:a,fragmentShader:l,transparent:!0,alphaTest:1,blending:ks,depthTest:!0,depthWrite:!1,side:ui})}static buildVertexShaderProjection(){let e=`

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
            `,e+=go.getVertexShaderFadeIn(),e+="}",e}static buildFragmentShader(){return`
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
        `}}class jE{static build(e){const t=new Bn;t.setIndex([0,1,2,0,2,3]);const n=new Float32Array(12),s=new xi(n,3);t.setAttribute("position",s),s.setXYZ(0,-1,-1,0),s.setXYZ(1,-1,1,0),s.setXYZ(2,1,1,0),s.setXYZ(3,1,-1,0),s.needsUpdate=!0;const r=new ey().copy(t),o=new Uint32Array(e),a=new WS(o,1,!1);return a.setUsage($A),r.setAttribute("splatIndex",a),r.instanceCount=0,r}}class $E extends jt{constructor(e,t=new U,n=new Ft,s=new U(1,1,1),r=1,o=1,a=!0){super(),this.splatBuffer=e,this.position.copy(t),this.quaternion.copy(n),this.scale.copy(s),this.transform=new $e,this.minimumAlpha=r,this.opacity=o,this.visible=a}copyTransformData(e){this.position.copy(e.position),this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.transform.copy(e.transform)}updateTransform(e){e?(this.matrixWorldAutoUpdate&&this.updateWorldMatrix(!0,!1),this.transform.copy(this.matrixWorld)):(this.matrixAutoUpdate&&this.updateMatrix(),this.transform.copy(this.matrix))}}class jd{static idGen=0;constructor(e,t,n,s){this.min=new U().copy(e),this.max=new U().copy(t),this.boundingBox=new Bi(this.min,this.max),this.center=new U().copy(this.max).sub(this.min).multiplyScalar(.5).add(this.min),this.depth=n,this.children=[],this.data=null,this.id=s||jd.idGen++}}class ga{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.sceneDimensions=new U,this.sceneMin=new U,this.sceneMax=new U,this.rootNode=null,this.nodesWithIndexes=[],this.splatMesh=null}static convertWorkerSubTreeNode(e){const t=new U().fromArray(e.min),n=new U().fromArray(e.max),s=new jd(t,n,e.depth,e.id);if(e.data.indexes){s.data={indexes:[]};for(let r of e.data.indexes)s.data.indexes.push(r)}if(e.children)for(let r of e.children)s.children.push(ga.convertWorkerSubTreeNode(r));return s}static convertWorkerSubTree(e,t){const n=new ga(e.maxDepth,e.maxCentersPerNode);n.sceneMin=new U().fromArray(e.sceneMin),n.sceneMax=new U().fromArray(e.sceneMax),n.splatMesh=t,n.rootNode=ga.convertWorkerSubTreeNode(e.rootNode);const s=(r,o)=>{r.children.length===0&&o(r);for(let a of r.children)s(a,o)};return n.nodesWithIndexes=[],s(n.rootNode,r=>{r.data&&r.data.indexes&&r.data.indexes.length>0&&n.nodesWithIndexes.push(r)}),n}}function ZE(i){let e=0;class t{constructor(l,c){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]]}containsPoint(l){return l[0]>=this.min[0]&&l[0]<=this.max[0]&&l[1]>=this.min[1]&&l[1]<=this.max[1]&&l[2]>=this.min[2]&&l[2]<=this.max[2]}}class n{constructor(l,c){this.maxDepth=l,this.maxCentersPerNode=c,this.sceneDimensions=[],this.sceneMin=[],this.sceneMax=[],this.rootNode=null,this.addedIndexes={},this.nodesWithIndexes=[],this.splatMesh=null,this.disposed=!1}}class s{constructor(l,c,u,f){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]],this.center=[(c[0]-l[0])*.5+l[0],(c[1]-l[1])*.5+l[1],(c[2]-l[2])*.5+l[2]],this.depth=u,this.children=[],this.data=null,this.id=f||e++}}processSplatTreeNode=function(a,l,c,u){const f=l.data.indexes.length;if(f<a.maxCentersPerNode||l.depth>a.maxDepth){const _=[];for(let v=0;v<l.data.indexes.length;v++)a.addedIndexes[l.data.indexes[v]]||(_.push(l.data.indexes[v]),a.addedIndexes[l.data.indexes[v]]=!0);l.data.indexes=_,l.data.indexes.sort((v,A)=>v>A?1:-1),a.nodesWithIndexes.push(l);return}const d=[l.max[0]-l.min[0],l.max[1]-l.min[1],l.max[2]-l.min[2]],h=[d[0]*.5,d[1]*.5,d[2]*.5],x=[l.min[0]+h[0],l.min[1]+h[1],l.min[2]+h[2]],p=[new t([x[0]-h[0],x[1],x[2]-h[2]],[x[0],x[1]+h[1],x[2]]),new t([x[0],x[1],x[2]-h[2]],[x[0]+h[0],x[1]+h[1],x[2]]),new t([x[0],x[1],x[2]],[x[0]+h[0],x[1]+h[1],x[2]+h[2]]),new t([x[0]-h[0],x[1],x[2]],[x[0],x[1]+h[1],x[2]+h[2]]),new t([x[0]-h[0],x[1]-h[1],x[2]-h[2]],[x[0],x[1],x[2]]),new t([x[0],x[1]-h[1],x[2]-h[2]],[x[0]+h[0],x[1],x[2]]),new t([x[0],x[1]-h[1],x[2]],[x[0]+h[0],x[1],x[2]+h[2]]),new t([x[0]-h[0],x[1]-h[1],x[2]],[x[0],x[1],x[2]+h[2]])],g=[];for(let _=0;_<p.length;_++)g[_]=[];const m=[0,0,0];for(let _=0;_<f;_++){const v=l.data.indexes[_],A=c[v];m[0]=u[A],m[1]=u[A+1],m[2]=u[A+2];for(let S=0;S<p.length;S++)p[S].containsPoint(m)&&g[S].push(v)}for(let _=0;_<p.length;_++){const v=new s(p[_].min,p[_].max,l.depth+1);v.data={indexes:g[_]},l.children.push(v)}l.data={};for(let _ of l.children)processSplatTreeNode(a,_,c,u)};const r=(a,l,c)=>{const u=[0,0,0],f=[0,0,0],d=[],h=Math.floor(a.length/4);for(let p=0;p<h;p++){const g=p*4,m=a[g],_=a[g+1],v=a[g+2],A=Math.round(a[g+3]);(p===0||m<u[0])&&(u[0]=m),(p===0||m>f[0])&&(f[0]=m),(p===0||_<u[1])&&(u[1]=_),(p===0||_>f[1])&&(f[1]=_),(p===0||v<u[2])&&(u[2]=v),(p===0||v>f[2])&&(f[2]=v),d.push(A)}const x=new n(l,c);return x.sceneMin=u,x.sceneMax=f,x.rootNode=new s(x.sceneMin,x.sceneMax,0),x.rootNode.data={indexes:d},x};function o(a,l,c){const u=[];for(let d of a){const h=Math.floor(d.length/4);for(let x=0;x<h;x++){const p=x*4,g=Math.round(d[p+3]);u[g]=p}}const f=[];for(let d of a){const h=r(d,l,c);f.push(h),processSplatTreeNode(h,h.rootNode,u,d)}i.postMessage({subTrees:f})}i.onmessage=a=>{a.data.process&&o(a.data.process.centers,a.data.process.maxDepth,a.data.process.maxCentersPerNode)}}function JE(i,e,t,n,s){i.postMessage({process:{centers:e,maxDepth:n,maxCentersPerNode:s}},t)}function e1(){return new Worker(URL.createObjectURL(new Blob(["(",ZE.toString(),")(self)"],{type:"application/javascript"})))}class t1{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.subTrees=[],this.splatMesh=null}dispose(){this.diposeSplatTreeWorker(),this.disposed=!0}diposeSplatTreeWorker(){this.splatTreeWorker&&this.splatTreeWorker.terminate(),this.splatTreeWorker=null}processSplatMesh=function(e,t=()=>!0,n,s){this.splatTreeWorker||(this.splatTreeWorker=e1()),this.splatMesh=e,this.subTrees=[];const r=new U,o=(a,l)=>{const c=new Float32Array(l*4);let u=0;for(let f=0;f<l;f++){const d=f+a;if(t(d)){e.getSplatCenter(d,r);const h=u*4;c[h]=r.x,c[h+1]=r.y,c[h+2]=r.z,c[h+3]=d,u++}}return c};return new Promise(a=>{const l=()=>this.disposed?(this.diposeSplatTreeWorker(),a(),!0):!1;n&&n(!1),Zn(()=>{if(l())return;const c=[];if(e.dynamicMode){let u=0;for(let f=0;f<e.scenes.length;f++){const h=e.getScene(f).splatBuffer.getSplatCount(),x=o(u,h);c.push(x),u+=h}}else{const u=o(0,e.getSplatCount());c.push(u)}this.splatTreeWorker.onmessage=u=>{l()||u.data.subTrees&&(s&&s(!1),Zn(()=>{if(!l()){for(let f of u.data.subTrees){const d=ga.convertWorkerSubTree(f,e);this.subTrees.push(d)}this.diposeSplatTreeWorker(),s&&s(!0),Zn(()=>{a()})}}))},Zn(()=>{if(l())return;n&&n(!0);const u=c.map(f=>f.buffer);JE(this.splatTreeWorker,c,u,this.maxDepth,this.maxCentersPerNode)})})})};countLeaves(){let e=0;return this.visitLeaves(()=>{e++}),e}visitLeaves(e){const t=(n,s)=>{n.children.length===0&&s(n);for(let r of n.children)t(r,s)};for(let n of this.subTrees)t(n.rootNode,e)}}function n1(i){const e={};function t(n){if(e[n]!==void 0)return e[n];let s;switch(n){case"WEBGL_depth_texture":s=i.getExtension("WEBGL_depth_texture")||i.getExtension("MOZ_WEBGL_depth_texture")||i.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":s=i.getExtension("EXT_texture_filter_anisotropic")||i.getExtension("MOZ_EXT_texture_filter_anisotropic")||i.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":s=i.getExtension("WEBGL_compressed_texture_s3tc")||i.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":s=i.getExtension("WEBGL_compressed_texture_pvrtc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:s=i.getExtension(n)}return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(n){n.isWebGL2?(t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance")):(t("WEBGL_depth_texture"),t("OES_texture_float"),t("OES_texture_half_float"),t("OES_texture_half_float_linear"),t("OES_standard_derivatives"),t("OES_element_index_uint"),t("OES_vertex_array_object"),t("ANGLE_instanced_arrays")),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture")},get:function(n){const s=t(n);return s===null&&console.warn("THREE.WebGLRenderer: "+n+" extension not supported."),s}}}function i1(i,e,t){let n;function s(){if(n!==void 0)return n;if(e.has("EXT_texture_filter_anisotropic")===!0){const M=e.get("EXT_texture_filter_anisotropic");n=i.getParameter(M.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else n=0;return n}function r(M){if(M==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";M="mediump"}return M==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}const o=typeof WebGL2RenderingContext<"u"&&i.constructor.name==="WebGL2RenderingContext";let a=t.precision!==void 0?t.precision:"highp";const l=r(a);l!==a&&(console.warn("THREE.WebGLRenderer:",a,"not supported, using",l,"instead."),a=l);const c=o||e.has("WEBGL_draw_buffers"),u=t.logarithmicDepthBuffer===!0,f=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),d=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),h=i.getParameter(i.MAX_TEXTURE_SIZE),x=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),p=i.getParameter(i.MAX_VERTEX_ATTRIBS),g=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),m=i.getParameter(i.MAX_VARYING_VECTORS),_=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),v=d>0,A=o||e.has("OES_texture_float"),S=v&&A,b=o?i.getParameter(i.MAX_SAMPLES):0;return{isWebGL2:o,drawBuffers:c,getMaxAnisotropy:s,getMaxPrecision:r,precision:a,logarithmicDepthBuffer:u,maxTextures:f,maxVertexTextures:d,maxTextureSize:h,maxCubemapSize:x,maxAttributes:p,maxVertexUniforms:g,maxVaryings:m,maxFragmentUniforms:_,vertexTextures:v,floatFragmentTextures:A,floatVertexTextures:S,maxSamples:b}}const xa={Default:0,Instant:2},xo={None:0,Info:3},gm=new Bn,s1=new Er,Dl=6,r1=4,o1=4,a1=4,l1=6,c1=8,Ru=4,Iu=4,xm=1,u1=.012,f1=.003,_m=1,vm=16777216;class ln extends Qt{constructor(e=os.ThreeD,t=!1,n=!1,s=!1,r=1,o=!0,a=!1,l=!1,c=1024,u=xo.None,f=0,d=1,h=.3){super(gm,s1),this.renderer=void 0,this.splatRenderMode=e,this.dynamicMode=t,this.enableOptionalEffects=n,this.halfPrecisionCovariancesOnGPU=s,this.devicePixelRatio=r,this.enableDistancesComputationOnGPU=o,this.integerBasedDistancesComputation=a,this.antialiased=l,this.kernel2DSize=h,this.maxScreenSpaceSplatSize=c,this.logLevel=u,this.sphericalHarmonicsDegree=f,this.minSphericalHarmonicsDegree=0,this.sceneFadeInRateMultiplier=d,this.scenes=[],this.splatTree=null,this.baseSplatTree=null,this.splatDataTextures={},this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new Bi,this.calculatedSceneCenter=new U,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!1,this.lastRenderer=null,this.visible=!1}static buildScenes(e,t,n){const s=[];s.length=t.length;for(let r=0;r<t.length;r++){const o=t[r],a=n[r]||{};let l=a.position||[0,0,0],c=a.rotation||[0,0,0,1],u=a.scale||[1,1,1];const f=new U().fromArray(l),d=new Ft().fromArray(c),h=new U().fromArray(u),x=ln.createScene(o,f,d,h,a.splatAlphaRemovalThreshold||1,a.opacity,a.visible);e.add(x),s[r]=x}return s}static createScene(e,t,n,s,r,o=1,a=!0){return new $E(e,t,n,s,r,o,a)}static buildSplatIndexMaps(e){const t=[],n=[];let s=0;for(let r=0;r<e.length;r++){const a=e[r].getMaxSplatCount();for(let l=0;l<a;l++)t[s]=l,n[s]=r,s++}return{localSplatIndexMap:t,sceneIndexMap:n}}buildSplatTree=function(e=[],t,n){return new Promise(s=>{this.disposeSplatTree(),this.baseSplatTree=new t1(8,1e3);const r=performance.now(),o=new Ot;this.baseSplatTree.processSplatMesh(this,a=>{this.getSplatColor(a,o);const l=this.getSceneIndexForSplat(a),c=e[l]||1;return o.w>=c},t,n).then(()=>{const a=performance.now()-r;if(this.logLevel>=xo.Info&&console.log("SplatTree build: "+a+" ms"),this.disposed)s();else{this.splatTree=this.baseSplatTree,this.baseSplatTree=null;let l=0,c=0,u=0;this.splatTree.visitLeaves(f=>{const d=f.data.indexes.length;d>0&&(c+=d,u++,l++)}),this.logLevel>=xo.Info&&(console.log(`SplatTree leaves: ${this.splatTree.countLeaves()}`),console.log(`SplatTree leaves with splats:${l}`),c=c/u,console.log(`Avg splat count per node: ${c}`),console.log(`Total splat count: ${this.getSplatCount()}`)),s()}})})};build(e,t,n=!0,s=!1,r,o,a=!0){this.sceneOptions=t,this.finalBuild=s;const l=ln.getTotalMaxSplatCountForSplatBuffers(e),c=ln.buildScenes(this,e,t);if(n)for(let p=0;p<this.scenes.length&&p<c.length;p++){const g=c[p],m=this.getScene(p);g.copyTransformData(m)}this.scenes=c;let u=3;for(let p of e){const g=p.getMinSphericalHarmonicsDegree();g<u&&(u=g)}this.minSphericalHarmonicsDegree=Math.min(u,this.sphericalHarmonicsDegree);let f=!1;if(e.length!==this.lastBuildScenes.length)f=!0;else for(let p=0;p<e.length;p++)if(e[p]!==this.lastBuildScenes[p].splatBuffer){f=!0;break}let d=!0;if((this.scenes.length!==1||this.lastBuildSceneCount!==this.scenes.length||this.lastBuildMaxSplatCount!==l||f)&&(d=!1),!d){this.boundingBox=new Bi,a||(this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.firstRenderTime=-1),this.lastBuildScenes=[],this.lastBuildSplatCount=0,this.lastBuildMaxSplatCount=0,this.disposeMeshData(),this.geometry=jE.build(l),this.splatRenderMode===os.ThreeD?this.material=sc.build(this.dynamicMode,this.enableOptionalEffects,this.antialiased,this.maxScreenSpaceSplatSize,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree,this.kernel2DSize):this.material=rc.build(this.dynamicMode,this.enableOptionalEffects,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree);const p=ln.buildSplatIndexMaps(e);this.globalSplatIndexToLocalSplatIndexMap=p.localSplatIndexMap,this.globalSplatIndexToSceneIndexMap=p.sceneIndexMap}const h=this.getSplatCount(!0);this.enableDistancesComputationOnGPU&&this.setupDistancesComputationTransformFeedback();const x=this.refreshGPUDataFromSplatBuffers(d);for(let p=0;p<this.scenes.length;p++)this.lastBuildScenes[p]=this.scenes[p];return this.lastBuildSplatCount=h,this.lastBuildMaxSplatCount=this.getMaxSplatCount(),this.lastBuildSceneCount=this.scenes.length,s&&this.scenes.length>0&&this.buildSplatTree(t.map(p=>p.splatAlphaRemovalThreshold||1),r,o).then(()=>{this.onSplatTreeReadyCallback&&this.onSplatTreeReadyCallback(this.splatTree),this.onSplatTreeReadyCallback=null}),this.visible=this.scenes.length>0,x}freeIntermediateSplatData(){const e=t=>{delete t.source.data,delete t.image,t.onUpdate=null};delete this.splatDataTextures.baseData.covariances,delete this.splatDataTextures.baseData.centers,delete this.splatDataTextures.baseData.colors,delete this.splatDataTextures.baseData.sphericalHarmonics,delete this.splatDataTextures.centerColors.data,delete this.splatDataTextures.covariances.data,this.splatDataTextures.sphericalHarmonics&&delete this.splatDataTextures.sphericalHarmonics.data,this.splatDataTextures.sceneIndexes&&delete this.splatDataTextures.sceneIndexes.data,this.splatDataTextures.centerColors.texture.needsUpdate=!0,this.splatDataTextures.centerColors.texture.onUpdate=()=>{e(this.splatDataTextures.centerColors.texture)},this.splatDataTextures.covariances.texture.needsUpdate=!0,this.splatDataTextures.covariances.texture.onUpdate=()=>{e(this.splatDataTextures.covariances.texture)},this.splatDataTextures.sphericalHarmonics&&(this.splatDataTextures.sphericalHarmonics.texture?(this.splatDataTextures.sphericalHarmonics.texture.needsUpdate=!0,this.splatDataTextures.sphericalHarmonics.texture.onUpdate=()=>{e(this.splatDataTextures.sphericalHarmonics.texture)}):this.splatDataTextures.sphericalHarmonics.textures.forEach(t=>{t.needsUpdate=!0,t.onUpdate=()=>{e(t)}})),this.splatDataTextures.sceneIndexes&&(this.splatDataTextures.sceneIndexes.texture.needsUpdate=!0,this.splatDataTextures.sceneIndexes.texture.onUpdate=()=>{e(this.splatDataTextures.sceneIndexes.texture)})}dispose(){this.disposeMeshData(),this.disposeTextures(),this.disposeSplatTree(),this.enableDistancesComputationOnGPU&&(this.computeDistancesOnGPUSyncTimeout&&(clearTimeout(this.computeDistancesOnGPUSyncTimeout),this.computeDistancesOnGPUSyncTimeout=null),this.disposeDistancesComputationGPUResources()),this.scenes=[],this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.renderer=null,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new Bi,this.calculatedSceneCenter=new U,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!0,this.lastRenderer=null,this.visible=!1}disposeMeshData(){this.geometry&&this.geometry!==gm&&(this.geometry.dispose(),this.geometry=null),this.material&&(this.material.dispose(),this.material=null)}disposeTextures(){for(let e in this.splatDataTextures)if(this.splatDataTextures.hasOwnProperty(e)){const t=this.splatDataTextures[e];t.texture&&(t.texture.dispose(),t.texture=null)}this.splatDataTextures=null}disposeSplatTree(){this.splatTree&&(this.splatTree.dispose(),this.splatTree=null),this.baseSplatTree&&(this.baseSplatTree.dispose(),this.baseSplatTree=null)}getSplatTree(){return this.splatTree}onSplatTreeReady(e){this.onSplatTreeReadyCallback=e}getDataForDistancesComputation(e,t){const n=this.integerBasedDistancesComputation?this.getIntegerCenters(e,t,!0):this.getFloatCenters(e,t,!0),s=this.getSceneIndexes(e,t);return{centers:n,sceneIndexes:s}}refreshGPUDataFromSplatBuffers(e){const t=this.getSplatCount(!0);this.refreshDataTexturesFromSplatBuffers(e);const n=e?this.lastBuildSplatCount:0,{centers:s,sceneIndexes:r}=this.getDataForDistancesComputation(n,t-1);return this.enableDistancesComputationOnGPU&&this.refreshGPUBuffersForDistancesComputation(s,r,e),{from:n,to:t-1,count:t-n,centers:s,sceneIndexes:r}}refreshGPUBuffersForDistancesComputation(e,t,n=!1){const s=n?this.lastBuildSplatCount:0;this.updateGPUCentersBufferForDistancesComputation(n,e,s),this.updateGPUTransformIndexesBufferForDistancesComputation(n,t,s)}refreshDataTexturesFromSplatBuffers(e){const t=this.getSplatCount(!0),n=this.lastBuildSplatCount,s=t-1;e?this.updateBaseDataFromSplatBuffers(n,s):(this.setupDataTextures(),this.updateBaseDataFromSplatBuffers()),this.updateDataTexturesFromBaseData(n,s),this.updateVisibleRegion(e)}setupDataTextures(){const e=this.getMaxSplatCount(),t=this.getSplatCount(!0);this.disposeTextures();const n=(M,R)=>{const y=new Xe(4096,1024);for(;y.x*y.y*M<e*R;)y.y*=2;return y},s=M=>M>=1?l1:o1,r=M=>{const R=s(M),y=n(R,6);return{elementsPerTexelStored:R,texSize:y}};let o=this.getTargetCovarianceCompressionLevel();const a=0,l=this.getTargetSphericalHarmonicsCompressionLevel();let c,u,f;if(this.splatRenderMode===os.ThreeD){const M=r(o);M.texSize.x*M.texSize.y>vm&&o===0&&(o=1),c=new Float32Array(e*Dl)}else u=new Float32Array(e*3),f=new Float32Array(e*4);const d=new Float32Array(e*3),h=new Uint8Array(e*4);let x=Float32Array;l===1?x=Uint16Array:l===2&&(x=Uint8Array);const p=mo(this.minSphericalHarmonicsDegree),g=this.minSphericalHarmonicsDegree?new x(e*p):void 0,m=n(Iu,4),_=new Uint32Array(m.x*m.y*Iu);ln.updateCenterColorsPaddedData(0,t-1,d,h,_);const v=new ns(_,m.x,m.y,ho,hi);if(v.internalFormat="RGBA32UI",v.needsUpdate=!0,this.material.uniforms.centersColorsTexture.value=v,this.material.uniforms.centersColorsTextureSize.value.copy(m),this.material.uniformsNeedUpdate=!0,this.splatDataTextures={baseData:{covariances:c,scales:u,rotations:f,centers:d,colors:h,sphericalHarmonics:g},centerColors:{data:_,texture:v,size:m}},this.splatRenderMode===os.ThreeD){const M=r(o),R=M.elementsPerTexelStored,y=M.texSize;let C=o>=1?Uint32Array:Float32Array;const F=o>=1?c1:a1,I=new C(y.x*y.y*F);o===0?I.set(c):ln.updatePaddedCompressedCovariancesTextureData(c,I,0,0,c.length);let O;if(o>=1)O=new ns(I,y.x,y.y,ho,hi),O.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=O;else{O=new ns(I,y.x,y.y,Pn,bi),this.material.uniforms.covariancesTexture.value=O;const z=new ns(new Uint32Array(32),2,2,ho,hi);z.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=z,z.needsUpdate=!0}O.needsUpdate=!0,this.material.uniforms.covariancesAreHalfFloat.value=o>=1?1:0,this.material.uniforms.covariancesTextureSize.value.copy(y),this.splatDataTextures.covariances={data:I,texture:O,size:y,compressionLevel:o,elementsPerTexelStored:R,elementsPerTexelAllocated:F}}else{const R=n(Ru,6);let y=Float32Array,C=bi;const F=new y(R.x*R.y*Ru);ln.updateScaleRotationsPaddedData(0,t-1,u,f,F);const I=new ns(F,R.x,R.y,Pn,C);I.needsUpdate=!0,this.material.uniforms.scaleRotationsTexture.value=I,this.material.uniforms.scaleRotationsTextureSize.value.copy(R),this.splatDataTextures.scaleRotations={data:F,texture:I,size:R,compressionLevel:a}}if(g){const M=l===2?Gi:wr;let R=p;R%2!==0&&R++;const y=4,C=Pn;let F=n(y,R);if(F.x*F.y<=vm){const I=F.x*F.y*y,O=new x(I);for(let H=0;H<t;H++){const V=p*H,q=R*H;for(let G=0;G<p;G++)O[q+G]=g[V+G]}const z=new ns(O,F.x,F.y,C,M);z.needsUpdate=!0,this.material.uniforms.sphericalHarmonicsTexture.value=z,this.splatDataTextures.sphericalHarmonics={componentCount:p,paddedComponentCount:R,data:O,textureCount:1,texture:z,size:F,compressionLevel:l,elementsPerTexel:y}}else{const I=p/3;R=I,R%2!==0&&R++,F=n(y,R);const O=F.x*F.y*y,z=[this.material.uniforms.sphericalHarmonicsTextureR,this.material.uniforms.sphericalHarmonicsTextureG,this.material.uniforms.sphericalHarmonicsTextureB],H=[],V=[];for(let q=0;q<3;q++){const G=new x(O);H.push(G);for(let ae=0;ae<t;ae++){const Ae=p*ae,Pe=R*ae;if(I>=3){for(let Ne=0;Ne<3;Ne++)G[Pe+Ne]=g[Ae+q*3+Ne];if(I>=8)for(let Ne=0;Ne<5;Ne++)G[Pe+3+Ne]=g[Ae+9+q*5+Ne]}}const Y=new ns(G,F.x,F.y,C,M);V.push(Y),Y.needsUpdate=!0,z[q].value=Y}this.material.uniforms.sphericalHarmonicsMultiTextureMode.value=1,this.splatDataTextures.sphericalHarmonics={componentCount:p,componentCountPerChannel:I,paddedComponentCount:R,data:H,textureCount:3,textures:V,size:F,compressionLevel:l,elementsPerTexel:y}}this.material.uniforms.sphericalHarmonicsTextureSize.value.copy(F),this.material.uniforms.sphericalHarmonics8BitMode.value=l===2?1:0;for(let I=0;I<this.scenes.length;I++){const O=this.scenes[I].splatBuffer;this.material.uniforms.sphericalHarmonics8BitCompressionRangeMin.value[I]=O.minSphericalHarmonicsCoeff,this.material.uniforms.sphericalHarmonics8BitCompressionRangeMax.value[I]=O.maxSphericalHarmonicsCoeff}this.material.uniformsNeedUpdate=!0}const A=n(xm,4),S=new Uint32Array(A.x*A.y*xm);for(let M=0;M<t;M++)S[M]=this.globalSplatIndexToSceneIndexMap[M];const b=new ns(S,A.x,A.y,Mc,hi);b.internalFormat="R32UI",b.needsUpdate=!0,this.material.uniforms.sceneIndexesTexture.value=b,this.material.uniforms.sceneIndexesTextureSize.value.copy(A),this.material.uniformsNeedUpdate=!0,this.splatDataTextures.sceneIndexes={data:S,texture:b,size:A},this.material.uniforms.sceneCount.value=this.scenes.length}updateBaseDataFromSplatBuffers(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0;this.fillSplatDataArrays(this.splatDataTextures.baseData.covariances,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,this.splatDataTextures.baseData.sphericalHarmonics,void 0,s,o,l,e,t,e)}updateDataTexturesFromBaseData(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0,c=this.splatDataTextures.centerColors,u=c.data,f=c.texture;ln.updateCenterColorsPaddedData(e,t,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,u);const d=this.renderer?this.renderer.properties.get(f):null;if(!d||!d.__webglTexture?f.needsUpdate=!0:this.updateDataTexture(u,c.texture,c.size,d,Iu,r1,4,e,t),n){const _=n.texture,v=e*Dl,A=t*Dl;if(s===0)for(let b=v;b<=A;b++){const M=this.splatDataTextures.baseData.covariances[b];n.data[b]=M}else ln.updatePaddedCompressedCovariancesTextureData(this.splatDataTextures.baseData.covariances,n.data,e*n.elementsPerTexelAllocated,v,A);const S=this.renderer?this.renderer.properties.get(_):null;!S||!S.__webglTexture?_.needsUpdate=!0:s===0?this.updateDataTexture(n.data,n.texture,n.size,S,n.elementsPerTexelStored,Dl,4,e,t):this.updateDataTexture(n.data,n.texture,n.size,S,n.elementsPerTexelAllocated,n.elementsPerTexelAllocated,2,e,t)}if(r){const _=r.data,v=r.texture,A=6,S=o===0?4:2;ln.updateScaleRotationsPaddedData(e,t,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,_);const b=this.renderer?this.renderer.properties.get(v):null;!b||!b.__webglTexture?v.needsUpdate=!0:this.updateDataTexture(_,r.texture,r.size,b,Ru,A,S,e,t)}const h=this.splatDataTextures.baseData.sphericalHarmonics;if(h){let _=4;l===1?_=2:l===2&&(_=1);const v=(b,M,R,y,C)=>{const F=this.renderer?this.renderer.properties.get(b):null;!F||!F.__webglTexture?b.needsUpdate=!0:this.updateDataTexture(y,b,M,F,R,C,_,e,t)},A=a.componentCount,S=a.paddedComponentCount;if(a.textureCount===1){const b=a.data;for(let M=e;M<=t;M++){const R=A*M,y=S*M;for(let C=0;C<A;C++)b[y+C]=h[R+C]}v(a.texture,a.size,a.elementsPerTexel,b,S)}else{const b=a.componentCountPerChannel;for(let M=0;M<3;M++){const R=a.data[M];for(let y=e;y<=t;y++){const C=A*y,F=S*y;if(b>=3){for(let I=0;I<3;I++)R[F+I]=h[C+M*3+I];if(b>=8)for(let I=0;I<5;I++)R[F+3+I]=h[C+9+M*5+I]}}v(a.textures[M],a.size,a.elementsPerTexel,R,S)}}}const x=this.splatDataTextures.sceneIndexes,p=x.data;for(let _=this.lastBuildSplatCount;_<=t;_++)p[_]=this.globalSplatIndexToSceneIndexMap[_];const g=x.texture,m=this.renderer?this.renderer.properties.get(g):null;!m||!m.__webglTexture?g.needsUpdate=!0:this.updateDataTexture(p,x.texture,x.size,m,1,1,1,this.lastBuildSplatCount,t)}getTargetCovarianceCompressionLevel(){return this.halfPrecisionCovariancesOnGPU?1:0}getTargetSphericalHarmonicsCompressionLevel(){return Math.max(1,this.getMaximumSplatBufferCompressionLevel())}getMaximumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel>e)&&(e=s.compressionLevel)}return e}getMinimumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel<e)&&(e=s.compressionLevel)}return e}static computeTextureUpdateRegion(e,t,n,s,r){const o=r/s,a=e*o,l=Math.floor(a/n),c=l*n*s,u=t*o,f=Math.floor(u/n),d=f*n*s+n*s;return{dataStart:c,dataEnd:d,startRow:l,endRow:f}}updateDataTexture(e,t,n,s,r,o,a,l,c){const u=this.renderer.getContext(),f=ln.computeTextureUpdateRegion(l,c,n.x,r,o),d=f.dataEnd-f.dataStart,h=new e.constructor(e.buffer,f.dataStart*a,d),x=f.endRow-f.startRow+1,p=this.webGLUtils.convert(t.type),g=this.webGLUtils.convert(t.format,t.colorSpace),m=u.getParameter(u.TEXTURE_BINDING_2D);u.bindTexture(u.TEXTURE_2D,s.__webglTexture),u.texSubImage2D(u.TEXTURE_2D,0,0,f.startRow,n.x,x,g,p,h),u.bindTexture(u.TEXTURE_2D,m)}static updatePaddedCompressedCovariancesTextureData(e,t,n,s,r){let o=new DataView(t.buffer),a=n,l=0;for(let c=s;c<=r;c+=2)o.setUint16(a*2,e[c],!0),o.setUint16(a*2+2,e[c+1],!0),a+=2,l++,l>=3&&(a+=2,l=0)}static updateCenterColorsPaddedData(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*4,l=o*3,c=o*4;r[c]=VT(s,a),r[c+1]=yu(n[l]),r[c+2]=yu(n[l+1]),r[c+3]=yu(n[l+2])}}static updateScaleRotationsPaddedData(e,t,n,s,r){for(let a=e;a<=t;a++){const l=a*3,c=a*4,u=a*6;r[u]=n[l],r[u+1]=n[l+1],r[u+2]=n[l+2],r[u+3]=s[c],r[u+4]=s[c+1],r[u+5]=s[c+2]}}updateVisibleRegion(e){const t=this.getSplatCount(!0),n=new U;if(!e){const r=new U;this.scenes.forEach(o=>{r.add(o.splatBuffer.sceneCenter)}),r.multiplyScalar(1/this.scenes.length),this.calculatedSceneCenter.copy(r),this.material.uniforms.sceneCenter.value.copy(this.calculatedSceneCenter),this.material.uniformsNeedUpdate=!0}const s=e?this.lastBuildSplatCount:0;for(let r=s;r<t;r++){this.getSplatCenter(r,n,!0);const o=n.sub(this.calculatedSceneCenter).length();o>this.maxSplatDistanceFromSceneCenter&&(this.maxSplatDistanceFromSceneCenter=o)}this.maxSplatDistanceFromSceneCenter-this.visibleRegionBufferRadius>_m&&(this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter,this.visibleRegionRadius=Math.max(this.visibleRegionBufferRadius-_m,0)),this.finalBuild&&(this.visibleRegionRadius=this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter),this.updateVisibleRegionFadeDistance()}updateVisibleRegionFadeDistance(e=xa.Default){const t=u1*this.sceneFadeInRateMultiplier,n=f1*this.sceneFadeInRateMultiplier,s=this.finalBuild?t:n,r=e===xa.Default?s:n;this.visibleRegionFadeStartRadius=(this.visibleRegionRadius-this.visibleRegionFadeStartRadius)*r+this.visibleRegionFadeStartRadius;const a=(this.visibleRegionBufferRadius>0?this.visibleRegionFadeStartRadius/this.visibleRegionBufferRadius:0)>.99,l=a||e===xa.Instant?1:0;this.material.uniforms.visibleRegionFadeStartRadius.value=this.visibleRegionFadeStartRadius,this.material.uniforms.visibleRegionRadius.value=this.visibleRegionRadius,this.material.uniforms.firstRenderTime.value=this.firstRenderTime,this.material.uniforms.currentTime.value=performance.now(),this.material.uniforms.fadeInComplete.value=l,this.material.uniformsNeedUpdate=!0,this.visibleRegionChanging=!a}updateRenderIndexes(e,t){const n=this.geometry;n.attributes.splatIndex.set(e),n.attributes.splatIndex.needsUpdate=!0,t>0&&this.firstRenderTime===-1&&(this.firstRenderTime=performance.now()),n.instanceCount=t,n.setDrawRange(0,t)}updateTransforms(){for(let e=0;e<this.scenes.length;e++)this.getScene(e).updateTransform(this.dynamicMode)}updateUniforms=(function(){const e=new Xe;return function(t,n,s,r,o,a){if(this.getSplatCount()>0){if(e.set(t.x*this.devicePixelRatio,t.y*this.devicePixelRatio),this.material.uniforms.viewport.value.copy(e),this.material.uniforms.basisViewport.value.set(1/e.x,1/e.y),this.material.uniforms.focal.value.set(n,s),this.material.uniforms.orthographicMode.value=r?1:0,this.material.uniforms.orthoZoom.value=o,this.material.uniforms.inverseFocalAdjustment.value=a,this.dynamicMode)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.transforms.value[c].copy(this.getScene(c).transform);if(this.enableOptionalEffects)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.sceneOpacity.value[c]=Bt(this.getScene(c).opacity,0,1),this.material.uniforms.sceneVisibility.value[c]=this.getScene(c).visible?1:0,this.material.uniformsNeedUpdate=!0;this.material.uniformsNeedUpdate=!0}}})();setSplatScale(e=1){this.splatScale=e,this.material.uniforms.splatScale.value=e,this.material.uniformsNeedUpdate=!0}getSplatScale(){return this.splatScale}setPointCloudModeEnabled(e){this.pointCloudModeEnabled=e,this.material.uniforms.pointCloudModeEnabled.value=e?1:0,this.material.uniformsNeedUpdate=!0}getPointCloudModeEnabled(){return this.pointCloudModeEnabled}getSplatDataTextures(){return this.splatDataTextures}getSplatCount(e=!1){return e?ln.getTotalSplatCountForScenes(this.scenes):this.lastBuildSplatCount}static getTotalSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getSplatCount());return t}static getTotalSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getSplatCount();return t}getMaxSplatCount(){return ln.getTotalMaxSplatCountForScenes(this.scenes)}static getTotalMaxSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getMaxSplatCount());return t}static getTotalMaxSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getMaxSplatCount();return t}disposeDistancesComputationGPUResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.vao&&(e.deleteVertexArray(this.distancesTransformFeedback.vao),this.distancesTransformFeedback.vao=null),this.distancesTransformFeedback.program&&(e.deleteProgram(this.distancesTransformFeedback.program),e.deleteShader(this.distancesTransformFeedback.vertexShader),e.deleteShader(this.distancesTransformFeedback.fragmentShader),this.distancesTransformFeedback.program=null,this.distancesTransformFeedback.vertexShader=null,this.distancesTransformFeedback.fragmentShader=null),this.disposeDistancesComputationGPUBufferResources(),this.distancesTransformFeedback.id&&(e.deleteTransformFeedback(this.distancesTransformFeedback.id),this.distancesTransformFeedback.id=null)}disposeDistancesComputationGPUBufferResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.centersBuffer&&(this.distancesTransformFeedback.centersBuffer=null,e.deleteBuffer(this.distancesTransformFeedback.centersBuffer)),this.distancesTransformFeedback.outDistancesBuffer&&(e.deleteBuffer(this.distancesTransformFeedback.outDistancesBuffer),this.distancesTransformFeedback.outDistancesBuffer=null)}setRenderer(e){if(e!==this.renderer){this.renderer=e;const t=this.renderer.getContext(),n=new n1(t),s=new i1(t,n,{});if(n.init(s),this.webGLUtils=new Cg(t,n),this.enableDistancesComputationOnGPU&&this.getSplatCount()>0){this.setupDistancesComputationTransformFeedback();const{centers:r,sceneIndexes:o}=this.getDataForDistancesComputation(0,this.getSplatCount()-1);this.refreshGPUBuffersForDistancesComputation(r,o)}}}setupDistancesComputationTransformFeedback=(function(){let e;return function(){const t=this.getMaxSplatCount();if(!this.renderer)return;const n=this.lastRenderer!==this.renderer,s=e!==t;if(!n&&!s)return;n?this.disposeDistancesComputationGPUResources():s&&this.disposeDistancesComputationGPUBufferResources();const r=this.renderer.getContext(),o=(d,h,x)=>{const p=d.createShader(h);if(!p)return console.error("Fatal error: gl could not create a shader object."),null;if(d.shaderSource(p,x),d.compileShader(p),!d.getShaderParameter(p,d.COMPILE_STATUS)){let m="unknown";h===d.VERTEX_SHADER?m="vertex shader":h===d.FRAGMENT_SHADER&&(m="fragement shader");const _=d.getShaderInfoLog(p);return console.error("Failed to compile "+m+" with these errors:"+_),d.deleteShader(p),null}return p};let a;this.integerBasedDistancesComputation?(a=`#version 300 es
                in ivec4 center;
                flat out int distance;`,this.dynamicMode?a+=`
                        in uint sceneIndex;
                        uniform ivec4 transforms[${bt.MaxScenes}];
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
                        uniform mat4 transforms[${bt.MaxScenes}];
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
            `,c=r.getParameter(r.VERTEX_ARRAY_BINDING),u=r.getParameter(r.CURRENT_PROGRAM),f=u?r.getProgramParameter(u,r.DELETE_STATUS):!1;if(n&&(this.distancesTransformFeedback.vao=r.createVertexArray()),r.bindVertexArray(this.distancesTransformFeedback.vao),n){const d=r.createProgram(),h=o(r,r.VERTEX_SHADER,a),x=o(r,r.FRAGMENT_SHADER,l);if(!h||!x)throw new Error("Could not compile shaders for distances computation on GPU.");if(r.attachShader(d,h),r.attachShader(d,x),r.transformFeedbackVaryings(d,["distance"],r.SEPARATE_ATTRIBS),r.linkProgram(d),!r.getProgramParameter(d,r.LINK_STATUS)){const g=r.getProgramInfoLog(d);throw console.error("Fatal error: Failed to link program: "+g),r.deleteProgram(d),r.deleteShader(x),r.deleteShader(h),new Error("Could not link shaders for distances computation on GPU.")}this.distancesTransformFeedback.program=d,this.distancesTransformFeedback.vertexShader=h,this.distancesTransformFeedback.vertexShader=x}if(r.useProgram(this.distancesTransformFeedback.program),this.distancesTransformFeedback.centersLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"center"),this.dynamicMode){this.distancesTransformFeedback.sceneIndexesLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"sceneIndex");for(let d=0;d<this.scenes.length;d++)this.distancesTransformFeedback.transformsLocs[d]=r.getUniformLocation(this.distancesTransformFeedback.program,`transforms[${d}]`)}else this.distancesTransformFeedback.modelViewProjLoc=r.getUniformLocation(this.distancesTransformFeedback.program,"modelViewProj");(n||s)&&(this.distancesTransformFeedback.centersBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?r.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,r.INT,0,0):r.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,r.FLOAT,!1,0,0),this.dynamicMode&&(this.distancesTransformFeedback.sceneIndexesBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),r.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,r.UNSIGNED_INT,0,0))),(n||s)&&(this.distancesTransformFeedback.outDistancesBuffer=r.createBuffer()),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),r.bufferData(r.ARRAY_BUFFER,t*4,r.STATIC_READ),n&&(this.distancesTransformFeedback.id=r.createTransformFeedback()),r.bindTransformFeedback(r.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),r.bindBufferBase(r.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),u&&f!==!0&&r.useProgram(u),c&&r.bindVertexArray(c),this.lastRenderer=this.renderer,e=t}})();updateGPUCentersBufferForDistancesComputation(e,t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=this.integerBasedDistancesComputation?Uint32Array:Float32Array,a=16,l=n*a;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,l,t);else{const c=new o(this.getMaxSplatCount()*a);c.set(t),s.bufferData(s.ARRAY_BUFFER,c,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}updateGPUTransformIndexesBufferForDistancesComputation(e,t,n){if(!this.renderer||!this.dynamicMode)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=n*4;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,o,t);else{const a=new Uint32Array(this.getMaxSplatCount()*4);a.set(t),s.bufferData(s.ARRAY_BUFFER,a,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}getSceneIndexes(e,t){let n;const s=t-e+1;n=new Uint32Array(s);for(let r=e;r<=t;r++)n[r]=this.globalSplatIndexToSceneIndexMap[r];return n}fillTransformsArray=(function(){const e=[];return function(t){e.length!==t.length&&(e.length=t.length);for(let n=0;n<this.scenes.length;n++){const r=this.getScene(n).transform.elements;for(let o=0;o<16;o++)e[n*16+o]=r[o]}t.set(e)}})();computeDistancesOnGPU=(function(){const e=new $e;return function(t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING),o=s.getParameter(s.CURRENT_PROGRAM),a=o?s.getProgramParameter(o,s.DELETE_STATUS):!1;if(s.bindVertexArray(this.distancesTransformFeedback.vao),s.useProgram(this.distancesTransformFeedback.program),s.enable(s.RASTERIZER_DISCARD),this.dynamicMode)for(let u=0;u<this.scenes.length;u++)if(e.copy(this.getScene(u).transform),e.premultiply(t),this.integerBasedDistancesComputation){const f=ln.getIntegerMatrixArray(e),d=[f[2],f[6],f[10],f[14]];s.uniform4i(this.distancesTransformFeedback.transformsLocs[u],d[0],d[1],d[2],d[3])}else s.uniformMatrix4fv(this.distancesTransformFeedback.transformsLocs[u],!1,e.elements);else if(this.integerBasedDistancesComputation){const u=ln.getIntegerMatrixArray(t),f=[u[2],u[6],u[10]];s.uniform3i(this.distancesTransformFeedback.modelViewProjLoc,f[0],f[1],f[2])}else{const u=[t.elements[2],t.elements[6],t.elements[10]];s.uniform3f(this.distancesTransformFeedback.modelViewProjLoc,u[0],u[1],u[2])}s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?s.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,s.INT,0,0):s.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,s.FLOAT,!1,0,0),this.dynamicMode&&(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),s.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,s.UNSIGNED_INT,0,0)),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),s.beginTransformFeedback(s.POINTS),s.drawArrays(s.POINTS,0,this.getSplatCount()),s.endTransformFeedback(),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,null),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,null),s.disable(s.RASTERIZER_DISCARD);const l=s.fenceSync(s.SYNC_GPU_COMMANDS_COMPLETE,0);s.flush();const c=new Promise(u=>{const f=()=>{if(this.disposed)u();else switch(s.clientWaitSync(l,0,0)){case s.TIMEOUT_EXPIRED:return this.computeDistancesOnGPUSyncTimeout=setTimeout(f),this.computeDistancesOnGPUSyncTimeout;case s.WAIT_FAILED:throw new Error("should never get here");default:this.computeDistancesOnGPUSyncTimeout=null,s.deleteSync(l);const p=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao),s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),s.getBufferSubData(s.ARRAY_BUFFER,0,n),s.bindBuffer(s.ARRAY_BUFFER,null),p&&s.bindVertexArray(p),u()}};this.computeDistancesOnGPUSyncTimeout=setTimeout(f)});return o&&a!==!0&&s.useProgram(o),r&&s.bindVertexArray(r),c}})();getLocalSplatParameters(e,t,n){n==null&&(n=!this.dynamicMode),t.splatBuffer=this.getSplatBufferForSplat(e),t.localIndex=this.getSplatLocalIndex(e),t.sceneTransform=n?this.getSceneTransformForSplat(e):null}fillSplatDataArrays(e,t,n,s,r,o,a,l=0,c=0,u=1,f,d,h=0,x){const p=new U;p.x=void 0,p.y=void 0,this.splatRenderMode===os.ThreeD?p.z=void 0:p.z=1;const g=new $e;let m=0,_=this.scenes.length-1;x!=null&&x>=0&&x<=this.scenes.length&&(m=x,_=x);for(let v=m;v<=_;v++){a==null&&(a=!this.dynamicMode);const A=this.getScene(v),S=A.splatBuffer;let b;if(a&&(this.getSceneTransform(v,g),b=g),e&&S.fillSplatCovarianceArray(e,b,f,d,h,l),t||n){if(!t||!n)throw new Error('SplatMesh::fillSplatDataArrays() -> "scales" and "rotations" must both be valid.');S.fillSplatScaleRotationArray(t,n,b,f,d,h,c,p)}s&&S.fillSplatCenterArray(s,b,f,d,h),r&&S.fillSplatColorArray(r,A.minimumAlpha,f,d,h),o&&S.fillSphericalHarmonicsArray(o,this.minSphericalHarmonicsDegree,b,f,d,h,u),h+=S.getSplatCount()}}getIntegerCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e);let o,a=n?4:3;o=new Int32Array(s*a);for(let l=0;l<s;l++){for(let c=0;c<3;c++)o[l*a+c]=Math.round(r[l*3+c]*1e3);n&&(o[l*a+3]=1e3)}return o}getFloatCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);if(this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e),!n)return r;let o=new Float32Array(s*4);for(let a=0;a<s;a++){for(let l=0;l<3;l++)o[a*4+l]=r[a*3+l];o[a*4+3]=1}return o}getSplatCenter=(function(){const e={};return function(t,n,s){this.getLocalSplatParameters(t,e,s),e.splatBuffer.getSplatCenter(e.localIndex,n,e.sceneTransform)}})();getSplatScaleAndRotation=(function(){const e={},t=new U;return function(n,s,r,o){this.getLocalSplatParameters(n,e,o),t.x=void 0,t.y=void 0,t.z=void 0,this.splatRenderMode===os.TwoD&&(t.z=0),e.splatBuffer.getSplatScaleAndRotation(e.localIndex,s,r,e.sceneTransform,t)}})();getSplatColor=(function(){const e={};return function(t,n){this.getLocalSplatParameters(t,e),e.splatBuffer.getSplatColor(e.localIndex,n)}})();getSceneTransform(e,t){const n=this.getScene(e);n.updateTransform(this.dynamicMode),t.copy(n.transform)}getScene(e){if(e<0||e>=this.scenes.length)throw new Error("SplatMesh::getScene() -> Invalid scene index.");return this.scenes[e]}getSceneCount(){return this.scenes.length}getSplatBufferForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).splatBuffer}getSceneIndexForSplat(e){return this.globalSplatIndexToSceneIndexMap[e]}getSceneTransformForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).transform}getSplatLocalIndex(e){return this.globalSplatIndexToLocalSplatIndexMap[e]}static getIntegerMatrixArray(e){const t=e.elements,n=[];for(let s=0;s<16;s++)n[s]=Math.round(t[s]*1e3);return n}computeBoundingBox(e=!1,t){let n=this.getSplatCount();if(t!=null){if(t<0||t>=this.scenes.length)throw new Error("SplatMesh::computeBoundingBox() -> Invalid scene index.");n=this.scenes[t].splatBuffer.getSplatCount()}const s=new Float32Array(n*3);this.fillSplatDataArrays(null,null,null,s,null,null,e,void 0,void 0,void 0,void 0,t);const r=new U,o=new U;for(let a=0;a<n;a++){const l=a*3,c=s[l],u=s[l+1],f=s[l+2];(a===0||c<r.x)&&(r.x=c),(a===0||u<r.y)&&(r.y=u),(a===0||f<r.z)&&(r.z=f),(a===0||c>o.x)&&(o.x=c),(a===0||u>o.y)&&(o.y=u),(a===0||f>o.z)&&(o.z=f)}return new Bi(r,o)}}var d1="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEbA2AAAGAQf39/f39/f39/f39/f39/fwBgAAF/AhIBA2VudgZtZW1vcnkCAwCAgAQDBAMAAQIHVAQRX193YXNtX2NhbGxfY3RvcnMAABhfX3dhc21fYXBwbHlfZGF0YV9yZWxvY3MAAAtzb3J0SW5kZXhlcwABE2Vtc2NyaXB0ZW5fdGxzX2luaXQAAgqWEAMDAAELihAEAXwDewN/A30gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEBA0AgAyABQQJ0IgVqIAIgACAFaigCAEECdGooAgAiBTYCACAFIAogBSAKSBshCiAFIA0gBSANShshDSABQQFqIgEgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiFWooAgAiFkECdGooAgAiFEcEQAJ/IAX9CQI4IAggFEEGdGoiDv0JAgwgDioCHP0gASAOKgIs/SACIA4qAjz9IAP95gEgBf0JAiggDv0JAgggDioCGP0gASAOKgIo/SACIA4qAjj9IAP95gEgBf0JAgggDv0JAgAgDioCEP0gASAOKgIg/SACIA4qAjD9IAP95gEgBf0JAhggDv0JAgQgDioCFP0gASAOKgIk/SACIA4qAjT9IAP95gH95AH95AH95AEiEf1f/QwAAAAAAECPQAAAAAAAQI9AIhL98gEiE/0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBP9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/REgDv0cAQJ/IBEgEf0NCAkKCwwNDg8AAAAAAAAAAP1fIBL98gEiEf0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9HAICfyAR/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAyESIBQhDwsgAyAVaiABIBZBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmogEf0bA2oiDjYCACAOIAogCiAOShshCiAOIA0gDSAOSBshDSACQQFqIgIgC0cNAAsMAwsCfyAFKgIIu/0UIAUqAhi7/SIB/QwAAAAAAECPQAAAAAAAQI9A/fIBIhH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIQ4CfyAR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyECAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQIgAv0RIA79HAEgBf0cAiESIAwhBQNAIAMgBUECdCICaiABIAAgAmooAgBBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmoiAjYCACACIAogAiAKSBshCiACIA0gAiANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEBA0AgAyABQQJ0IgVqAn8gAiAAIAVqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAFBAWoiASALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIRcgBSoCGCEYIAUqAgghGUH4////ByEKQYiAgIB4IQ0gDCEFA0ACfyAXIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCAZIAIqAgCUIBggAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIUaigCAEECdCIVaigCACIORwRAIAX9CQI4IAggDkEGdGoiD/0JAgwgDyoCHP0gASAPKgIs/SACIA8qAjz9IAP95gEgBf0JAiggD/0JAgggDyoCGP0gASAPKgIo/SACIA8qAjj9IAP95gEgBf0JAgggD/0JAgAgDyoCEP0gASAPKgIg/SACIA8qAjD9IAP95gEgBf0JAhggD/0JAgQgDyoCFP0gASAPKgIk/SACIA8qAjT9IAP95gH95AH95AH95AEhESAOIQ8LIAMgFGoCfyAR/R8DIAEgFUECdCIOQQxyaioCAJQgEf0fAiABIA5BCHJqKgIAlCAR/R8AIAEgDmoqAgCUIBH9HwEgASAOQQRyaioCAJSSkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSACQQFqIgIgC0cNAAsMAQtBiICAgHghDUH4////ByEKCyALIAxLBEAgCUEBa7MgDbIgCrKTlSEXIAwhDQNAAn8gFyADIA1BAnRqIgEoAgAgCmuylCIYi0MAAABPXQRAIBioDAELQYCAgIB4CyEOIAEgDjYCACAEIA5BAnRqIgEgASgCAEEBajYCACANQQFqIg0gC0cNAAsLIAlBAk8EQCAEKAIAIQ1BASEKA0AgBCAKQQJ0aiIBIAEoAgAgDWoiDTYCACAKQQFqIgogCUcNAAsLIAxBAEoEQCAMIQoDQCAGIApBAWsiAUECdCICaiAAIAJqKAIANgIAIApBAUshAiABIQogAg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCwsEAEEACw==",Am="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACEgEDZW52Bm1lbW9yeQIDAICABAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=",h1="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQrrDwICAAvlDwQBfAN7B30DfyALIAprIQwCQAJAIA4EQCANBEBB+P///wchCkGIgICAeCENIAsgDE0NAyAMIQUDQCADIAVBAnQiAWogAiAAIAFqKAIAQQJ0aigCACIBNgIAIAEgCiABIApIGyEKIAEgDSABIA1KGyENIAVBAWoiBSALRw0ACwwDCyAPBEAgCyAMTQ0CQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIcaigCACIdQQJ0aigCACIbRwRAAn8gBf0JAjggCCAbQQZ0aiIO/QkCDCAOKgIc/SABIA4qAiz9IAIgDioCPP0gA/3mASAF/QkCKCAO/QkCCCAOKgIY/SABIA4qAij9IAIgDioCOP0gA/3mASAF/QkCCCAO/QkCACAOKgIQ/SABIA4qAiD9IAIgDioCMP0gA/3mASAF/QkCGCAO/QkCBCAOKgIU/SABIA4qAiT9IAIgDioCNP0gA/3mAf3kAf3kAf3kASIR/V/9DAAAAAAAQI9AAAAAAABAj0AiEv3yASIT/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOAn8gE/0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9ESAO/RwBAn8gESAR/Q0ICQoLDA0ODwABAgMAAQID/V8gEv3yASIR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAgJ/IBH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/RwDIRIgGyEPCyADIBxqIAEgHUEEdGr9AAAAIBL9tQEiEf0bACAR/RsBaiAR/RsCaiAR/RsDaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAgi7/RQgBSoCGLv9IgH9DAAAAAAAQI9AAAAAAABAj0D98gEiEf0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBH9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQL9ESAO/RwBIAX9HAIhEiAMIQUDQCADIAVBAnQiAmogASAAIAJqKAIAQQR0av0AAAAgEv21ASIR/RsAIBH9GwFqIBH9GwJqIgI2AgAgAiAKIAIgCkgbIQogAiANIAIgDUobIQ0gBUEBaiIFIAtHDQALDAILIA0EQEH4////ByEKQYiAgIB4IQ0gCyAMTQ0CIAwhBQNAIAMgBUECdCIBagJ/IAIgACABaigCAEECdGoqAgC7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgD0UEQCALIAxNDQEgBSoCKCEUIAUqAhghFSAFKgIIIRZB+P///wchCkGIgICAeCENIAwhBQNAAn8gFCABIAAgBUECdCIHaigCAEEEdGoiAioCCJQgFiACKgIAlCAVIAIqAgSUkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDiADIAdqIA42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gBUEBaiIFIAtHDQALDAILIAsgDE0NAEF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiG2ooAgBBAnQiHGooAgAiDkcEQCAFKgI4IhQgCCAOQQZ0aiIPKgI8lCAFKgIoIhUgDyoCOJQgBSoCCCIWIA8qAjCUIAUqAhgiFyAPKgI0lJKSkiEYIBQgDyoCLJQgFSAPKgIolCAWIA8qAiCUIBcgDyoCJJSSkpIhGSAUIA8qAhyUIBUgDyoCGJQgFiAPKgIQlCAXIA8qAhSUkpKSIRogFCAPKgIMlCAVIA8qAgiUIBYgDyoCAJQgFyAPKgIElJKSkiEUIA4hDwsgAyAbagJ/IBggASAcQQJ0aiIOKgIMlCAZIA4qAgiUIBQgDioCAJQgGiAOKgIElJKSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAJBAWoiAiALRw0ACwwBC0GIgICAeCENQfj///8HIQoLIAsgDEsEQCAJQQFrsyANsiAKspOVIRQgDCENA0ACfyAUIAMgDUECdGoiASgCACAKa7KUIhWLQwAAAE9dBEAgFagMAQtBgICAgHgLIQ4gASAONgIAIAQgDkECdGoiASABKAIAQQFqNgIAIA1BAWoiDSALRw0ACwsgCUECTwRAIAQoAgAhDUEBIQoDQCAEIApBAnRqIgEgASgCACANaiINNgIAIApBAWoiCiAJRw0ACwsgDEEASgRAIAwhCgNAIAYgCkEBayIBQQJ0IgJqIAAgAmooAgA2AgAgCkEBSyABIQoNAAsLIAsgDEoEQCALIQoDQCAGIAsgBCADIApBAWsiCkECdCIBaigCAEECdGoiAigCACIFa0ECdGogACABaigCADYCACACIAVBAWs2AgAgCiAMSg0ACwsL",p1="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=";function m1(i){let e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g,m,_,v,A;function S(b,M,R,y,C,F,I){const O=performance.now();if(!n&&(new Uint32Array(t,a,C.byteLength/A.BytesPerInt).set(C),new Float32Array(t,u,I.byteLength/A.BytesPerFloat).set(I),y)){let G;s?G=new Int32Array(t,f,F.byteLength/A.BytesPerInt):G=new Float32Array(t,f,F.byteLength/A.BytesPerFloat),G.set(F)}g||(g=new Uint32Array(_)),new Float32Array(t,p,16).set(R),new Uint32Array(t,h,_).set(g),e.exports.sortIndexes(a,x,f,d,h,p,l,c,u,_,b,M,o,y,s,r);const z={sortDone:!0,splatSortCount:b,splatRenderCount:M,sortTime:0};if(!n){const V=new Uint32Array(t,l,M);(!m||m.length<M)&&(m=new Uint32Array(M)),m.set(V),z.sortedIndexes=m}const H=performance.now();z.sortTime=H-O,i.postMessage(z)}i.onmessage=b=>{if(b.data.centers)centers=b.data.centers,sceneIndexes=b.data.sceneIndexes,s?new Int32Array(t,x+b.data.range.from*A.BytesPerInt*4,b.data.range.count*4).set(new Int32Array(centers)):new Float32Array(t,x+b.data.range.from*A.BytesPerFloat*4,b.data.range.count*4).set(new Float32Array(centers)),r&&new Uint32Array(t,c+b.data.range.from*4,b.data.range.count).set(new Uint32Array(sceneIndexes)),v=b.data.range.from+b.data.range.count;else if(b.data.sort){const M=Math.min(b.data.sort.splatRenderCount||0,v),R=Math.min(b.data.sort.splatSortCount||0,v),y=b.data.sort.usePrecomputedDistances;let C,F,I;n||(C=b.data.sort.indexesToSort,I=b.data.sort.transforms,y&&(F=b.data.sort.precomputedDistances)),S(R,M,b.data.sort.modelViewProj,y,C,F,I)}else if(b.data.init){A=b.data.init.Constants,o=b.data.init.splatCount,n=b.data.init.useSharedMemory,s=b.data.init.integerBasedSort,r=b.data.init.dynamicMode,_=b.data.init.distanceMapRange,v=0;const M=s?A.BytesPerInt*4:A.BytesPerFloat*4,R=new Uint8Array(b.data.init.sorterWasmBytes),y=16*A.BytesPerFloat,C=o*A.BytesPerInt,F=o*M,I=y,O=s?o*A.BytesPerInt:o*A.BytesPerFloat,z=o*A.BytesPerInt,H=o*A.BytesPerInt,V=s?_*A.BytesPerInt*2:_*A.BytesPerFloat*2,q=r?o*A.BytesPerInt:0,G=r?A.MaxScenes*y:0,Y=A.MemoryPageSize*32,ae=C+F+I+O+z+V+H+q+G+Y,Ae=Math.floor(ae/A.MemoryPageSize)+1,Pe={module:{},env:{memory:new WebAssembly.Memory({initial:Ae,maximum:Ae,shared:!0})}};WebAssembly.compile(R).then(Ne=>WebAssembly.instantiate(Ne,Pe)).then(Ne=>{e=Ne,a=0,x=a+C,p=x+F,f=p+I,d=f+O,h=d+z,l=h+V,c=l+H,u=c+q,t=Pe.env.memory.buffer,n?i.postMessage({sortSetupPhase1Complete:!0,indexesToSortBuffer:t,indexesToSortOffset:a,sortedIndexesBuffer:t,sortedIndexesOffset:l,precomputedDistancesBuffer:t,precomputedDistancesOffset:f,transformsBuffer:t,transformsOffset:u}):i.postMessage({sortSetupPhase1Complete:!0})})}}}function g1(i,e,t,n,s,r=bt.DefaultSplatSortDistanceMapPrecision){const o=new Worker(URL.createObjectURL(new Blob(["(",m1.toString(),")(self)"],{type:"application/javascript"})));let a=d1;const l=Bd()?Eg():null;!t&&!e?(a=Am,l&&l.major<=16&&l.minor<4&&(a=p1)):t?e||l&&l.major<=16&&l.minor<4&&(a=h1):a=Am;const c=atob(a),u=new Uint8Array(c.length);for(let f=0;f<c.length;f++)u[f]=c.charCodeAt(f);return o.postMessage({init:{sorterWasmBytes:u.buffer,splatCount:i,useSharedMemory:e,integerBasedSort:n,dynamicMode:s,distanceMapRange:1<<r,Constants:{BytesPerFloat:bt.BytesPerFloat,BytesPerInt:bt.BytesPerInt,MemoryPageSize:bt.MemoryPageSize,MaxScenes:bt.MaxScenes}}}),o}const dr={None:0,VR:1,AR:2};class Io{static createButton(e,t={}){const n=document.createElement("button");function s(){let c=null;async function u(h){h.addEventListener("end",f),await e.xr.setSession(h),n.textContent="EXIT VR",c=h}function f(){c.removeEventListener("end",f),n.textContent="ENTER VR",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="ENTER VR";const d={...t,optionalFeatures:["local-floor","bounded-floor","layers",...t.optionalFeatures||[]]};n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-vr",d).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",d).then(u).catch(h=>{console.warn(h)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",d).then(u).catch(h=>{console.warn(h)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="VR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="VR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="VRButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-vr").then(function(c){c?s():o(),c&&Io.xrSessionIsGranted&&n.click()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}static registerSessionGrantedListener(){if(typeof navigator<"u"&&"xr"in navigator){if(/WebXRViewer\//i.test(navigator.userAgent))return;navigator.xr.addEventListener("sessiongranted",()=>{Io.xrSessionIsGranted=!0})}}}Io.xrSessionIsGranted=!1;Io.registerSessionGrantedListener();class x1{static createButton(e,t={}){const n=document.createElement("button");function s(){if(t.domOverlay===void 0){const d=document.createElement("div");d.style.display="none",document.body.appendChild(d);const h=document.createElementNS("http://www.w3.org/2000/svg","svg");h.setAttribute("width",38),h.setAttribute("height",38),h.style.position="absolute",h.style.right="20px",h.style.top="20px",h.addEventListener("click",function(){c.end()}),d.appendChild(h);const x=document.createElementNS("http://www.w3.org/2000/svg","path");x.setAttribute("d","M 12,12 L 28,28 M 28,12 12,28"),x.setAttribute("stroke","#fff"),x.setAttribute("stroke-width",2),h.appendChild(x),t.optionalFeatures===void 0&&(t.optionalFeatures=[]),t.optionalFeatures.push("dom-overlay"),t.domOverlay={root:d}}let c=null;async function u(d){d.addEventListener("end",f),e.xr.setReferenceSpaceType("local"),await e.xr.setSession(d),n.textContent="STOP AR",t.domOverlay.root.style.display="",c=d}function f(){c.removeEventListener("end",f),n.textContent="START AR",t.domOverlay.root.style.display="none",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="START AR",n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-ar",t).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(d=>{console.warn(d)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(d=>{console.warn(d)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="AR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="AR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="ARButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-ar").then(function(c){c?s():o()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}}const Du={Always:0,Never:2},_1=50,v1=.75,A1=15e5,S1=10,y1=2.5,b1=60;class io{constructor(e={}){if(e.cameraUp||(e.cameraUp=[0,1,0]),this.cameraUp=new U().fromArray(e.cameraUp),e.initialCameraPosition||(e.initialCameraPosition=[0,10,15]),this.initialCameraPosition=new U().fromArray(e.initialCameraPosition),e.initialCameraLookAt||(e.initialCameraLookAt=[0,0,0]),this.initialCameraLookAt=new U().fromArray(e.initialCameraLookAt),this.dropInMode=e.dropInMode||!1,(e.selfDrivenMode===void 0||e.selfDrivenMode===null)&&(e.selfDrivenMode=!0),this.selfDrivenMode=e.selfDrivenMode&&!this.dropInMode,this.selfDrivenUpdateFunc=this.selfDrivenUpdate.bind(this),e.useBuiltInControls===void 0&&(e.useBuiltInControls=!0),this.useBuiltInControls=e.useBuiltInControls,this.rootElement=e.rootElement,this.ignoreDevicePixelRatio=e.ignoreDevicePixelRatio||!1,this.devicePixelRatio=this.ignoreDevicePixelRatio?1:window.devicePixelRatio||1,this.halfPrecisionCovariancesOnGPU=e.halfPrecisionCovariancesOnGPU||!1,this.threeScene=e.threeScene,this.renderer=e.renderer,this.camera=e.camera,this.gpuAcceleratedSort=e.gpuAcceleratedSort||!1,(e.integerBasedSort===void 0||e.integerBasedSort===null)&&(e.integerBasedSort=!0),this.integerBasedSort=e.integerBasedSort,(e.sharedMemoryForWorkers===void 0||e.sharedMemoryForWorkers===null)&&(e.sharedMemoryForWorkers=!0),this.sharedMemoryForWorkers=e.sharedMemoryForWorkers,this.dynamicScene=!!e.dynamicScene,this.antialiased=e.antialiased||!1,this.kernel2DSize=e.kernel2DSize===void 0?.3:e.kernel2DSize,this.webXRMode=e.webXRMode||dr.None,this.webXRMode!==dr.None&&(this.gpuAcceleratedSort=!1),this.webXRActive=!1,this.webXRSessionInit=e.webXRSessionInit||{},this.renderMode=e.renderMode||Du.Always,this.sceneRevealMode=e.sceneRevealMode||xa.Default,this.focalAdjustment=e.focalAdjustment||1,this.maxScreenSpaceSplatSize=e.maxScreenSpaceSplatSize||1024,this.logLevel=e.logLevel||xo.None,this.sphericalHarmonicsDegree=e.sphericalHarmonicsDegree||0,this.enableOptionalEffects=e.enableOptionalEffects||!1,(e.enableSIMDInSort===void 0||e.enableSIMDInSort===null)&&(e.enableSIMDInSort=!0),this.enableSIMDInSort=e.enableSIMDInSort,(e.inMemoryCompressionLevel===void 0||e.inMemoryCompressionLevel===null)&&(e.inMemoryCompressionLevel=0),this.inMemoryCompressionLevel=e.inMemoryCompressionLevel,(e.optimizeSplatData===void 0||e.optimizeSplatData===null)&&(e.optimizeSplatData=!0),this.optimizeSplatData=e.optimizeSplatData,(e.freeIntermediateSplatData===void 0||e.freeIntermediateSplatData===null)&&(e.freeIntermediateSplatData=!1),this.freeIntermediateSplatData=e.freeIntermediateSplatData,Bd()){const n=Eg();n.major<17&&(this.enableSIMDInSort=!1),n.major<16&&(this.sharedMemoryForWorkers=!1)}(e.splatRenderMode===void 0||e.splatRenderMode===null)&&(e.splatRenderMode=os.ThreeD),this.splatRenderMode=e.splatRenderMode,this.sceneFadeInRateMultiplier=e.sceneFadeInRateMultiplier||1,this.splatSortDistanceMapPrecision=e.splatSortDistanceMapPrecision||bt.DefaultSplatSortDistanceMapPrecision;const t=this.integerBasedSort?20:24;this.splatSortDistanceMapPrecision=Bt(this.splatSortDistanceMapPrecision,10,t),this.onSplatMeshChangedCallback=null,this.createSplatMesh(),this.controls=null,this.perspectiveControls=null,this.orthographicControls=null,this.orthographicCamera=null,this.perspectiveCamera=null,this.showMeshCursor=!1,this.showControlPlane=!1,this.showInfo=!1,this.sceneHelper=null,this.sortWorker=null,this.sortRunning=!1,this.splatRenderCount=0,this.splatSortCount=0,this.lastSplatSortCount=0,this.sortWorkerIndexesToSort=null,this.sortWorkerSortedIndexes=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.preSortMessages=[],this.runAfterNextSort=[],this.selfDrivenModeRunning=!1,this.splatRenderReady=!1,this.raycaster=new KE,this.infoPanel=null,this.startInOrthographicMode=!1,this.currentFPS=0,this.lastSortTime=0,this.consecutiveRenderFrames=0,this.previousCameraTarget=new U,this.nextCameraTarget=new U,this.mousePosition=new Xe,this.mouseDownPosition=new Xe,this.mouseDownTime=null,this.resizeObserver=null,this.mouseMoveListener=null,this.mouseDownListener=null,this.mouseUpListener=null,this.keyDownListener=null,this.sortPromise=null,this.sortPromiseResolver=null,this.splatSceneDownloadPromises={},this.splatSceneDownloadAndBuildPromise=null,this.splatSceneRemovalPromise=null,this.loadingSpinner=new Qd(null,this.rootElement||document.body),this.loadingSpinner.hide(),this.loadingProgressBar=new GE(this.rootElement||document.body),this.loadingProgressBar.hide(),this.infoPanel=new WE(this.rootElement||document.body),this.infoPanel.hide(),this.usingExternalCamera=!!(this.dropInMode||this.camera),this.usingExternalRenderer=!!(this.dropInMode||this.renderer),this.initialized=!1,this.disposing=!1,this.disposed=!1,this.disposePromise=null,this.dropInMode||this.init()}createSplatMesh(){this.splatMesh=new ln(this.splatRenderMode,this.dynamicScene,this.enableOptionalEffects,this.halfPrecisionCovariancesOnGPU,this.devicePixelRatio,this.gpuAcceleratedSort,this.integerBasedSort,this.antialiased,this.maxScreenSpaceSplatSize,this.logLevel,this.sphericalHarmonicsDegree,this.sceneFadeInRateMultiplier,this.kernel2DSize),this.splatMesh.frustumCulled=!1,this.onSplatMeshChangedCallback&&this.onSplatMeshChangedCallback()}init(){this.initialized||(this.rootElement||(this.usingExternalRenderer?this.rootElement=this.renderer.domElement||document.body:(this.rootElement=document.createElement("div"),this.rootElement.style.width="100%",this.rootElement.style.height="100%",this.rootElement.style.position="absolute",document.body.appendChild(this.rootElement))),this.setupCamera(),this.setupRenderer(),this.setupWebXR(this.webXRSessionInit),this.setupControls(),this.setupEventHandlers(),this.threeScene=this.threeScene||new GS,this.sceneHelper=new ma(this.threeScene),this.sceneHelper.setupMeshCursor(),this.sceneHelper.setupFocusMarker(),this.sceneHelper.setupControlPlane(),this.loadingProgressBar.setContainer(this.rootElement),this.loadingSpinner.setContainer(this.rootElement),this.infoPanel.setContainer(this.rootElement),this.initialized=!0)}setupCamera(){if(!this.usingExternalCamera){const e=new Xe;this.getRenderDimensions(e),this.perspectiveCamera=new ci(_1,e.x/e.y,.1,1e3),this.orthographicCamera=new Fd(e.x/-2,e.x/2,e.y/2,e.y/-2,.1,1e3),this.camera=this.startInOrthographicMode?this.orthographicCamera:this.perspectiveCamera,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt)}}setupRenderer(){if(!this.usingExternalRenderer){const e=new Xe;this.getRenderDimensions(e),this.renderer=new HT({antialias:!1,precision:"highp"}),this.renderer.setPixelRatio(this.devicePixelRatio),this.renderer.autoClear=!0,this.renderer.setClearColor(new lt(0),0),this.renderer.setSize(e.x,e.y),this.resizeObserver=new ResizeObserver(()=>{this.getRenderDimensions(e),this.renderer.setSize(e.x,e.y),this.forceRenderNextFrame()}),this.resizeObserver.observe(this.rootElement),this.rootElement.appendChild(this.renderer.domElement)}}setupWebXR(e){this.webXRMode&&(this.webXRMode===dr.VR?this.rootElement.appendChild(Io.createButton(this.renderer,e)):this.webXRMode===dr.AR&&this.rootElement.appendChild(x1.createButton(this.renderer,e)),this.renderer.xr.addEventListener("sessionstart",t=>{this.webXRActive=!0}),this.renderer.xr.addEventListener("sessionend",t=>{this.webXRActive=!1}),this.renderer.xr.enabled=!0,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt))}setupControls(){if(this.useBuiltInControls&&this.webXRMode===dr.None){this.usingExternalCamera?this.camera.isOrthographicCamera?this.orthographicControls=new Il(this.camera,this.renderer.domElement):this.perspectiveControls=new Il(this.camera,this.renderer.domElement):(this.perspectiveControls=new Il(this.perspectiveCamera,this.renderer.domElement),this.orthographicControls=new Il(this.orthographicCamera,this.renderer.domElement));for(let e of[this.orthographicControls,this.perspectiveControls])e&&(e.listenToKeyEvents(window),e.rotateSpeed=.5,e.maxPolarAngle=Math.PI*.75,e.minPolarAngle=.1,e.enableDamping=!0,e.dampingFactor=.05,e.target.copy(this.initialCameraLookAt),e.update());this.controls=this.camera.isOrthographicCamera?this.orthographicControls:this.perspectiveControls,this.controls.update()}}setupEventHandlers(){this.useBuiltInControls&&this.webXRMode===dr.None&&(this.mouseMoveListener=this.onMouseMove.bind(this),this.renderer.domElement.addEventListener("pointermove",this.mouseMoveListener,!1),this.mouseDownListener=this.onMouseDown.bind(this),this.renderer.domElement.addEventListener("pointerdown",this.mouseDownListener,!1),this.mouseUpListener=this.onMouseUp.bind(this),this.renderer.domElement.addEventListener("pointerup",this.mouseUpListener,!1),this.keyDownListener=this.onKeyDown.bind(this),window.addEventListener("keydown",this.keyDownListener,!1))}removeEventHandlers(){this.useBuiltInControls&&(this.renderer.domElement.removeEventListener("pointermove",this.mouseMoveListener),this.mouseMoveListener=null,this.renderer.domElement.removeEventListener("pointerdown",this.mouseDownListener),this.mouseDownListener=null,this.renderer.domElement.removeEventListener("pointerup",this.mouseUpListener),this.mouseUpListener=null,window.removeEventListener("keydown",this.keyDownListener),this.keyDownListener=null)}setRenderMode(e){this.renderMode=e}setActiveSphericalHarmonicsDegrees(e){this.splatMesh.material.uniforms.sphericalHarmonicsDegree.value=e,this.splatMesh.material.uniformsNeedUpdate=!0}onSplatMeshChanged(e){this.onSplatMeshChangedCallback=e}onKeyDown=(function(){const e=new U,t=new $e,n=new $e;return function(s){switch(e.set(0,0,-1),e.transformDirection(this.camera.matrixWorld),t.makeRotationAxis(e,Math.PI/128),n.makeRotationAxis(e,-Math.PI/128),s.code){case"KeyG":this.focalAdjustment+=.02,this.forceRenderNextFrame();break;case"KeyF":this.focalAdjustment-=.02,this.forceRenderNextFrame();break;case"ArrowLeft":this.camera.up.transformDirection(t);break;case"ArrowRight":this.camera.up.transformDirection(n);break;case"KeyC":this.showMeshCursor=!this.showMeshCursor;break;case"KeyU":this.showControlPlane=!this.showControlPlane;break;case"KeyI":this.showInfo=!this.showInfo,this.showInfo?this.infoPanel.show():this.infoPanel.hide();break;case"KeyO":this.usingExternalCamera||this.setOrthographicMode(!this.camera.isOrthographicCamera);break;case"KeyP":this.usingExternalCamera||this.splatMesh.setPointCloudModeEnabled(!this.splatMesh.getPointCloudModeEnabled());break;case"Equal":this.usingExternalCamera||this.splatMesh.setSplatScale(this.splatMesh.getSplatScale()+.05);break;case"Minus":this.usingExternalCamera||this.splatMesh.setSplatScale(Math.max(this.splatMesh.getSplatScale()-.05,0));break}}})();onMouseMove(e){this.mousePosition.set(e.offsetX,e.offsetY)}onMouseDown(){this.mouseDownPosition.copy(this.mousePosition),this.mouseDownTime=jr()}onMouseUp=(function(){const e=new Xe;return function(t){e.copy(this.mousePosition).sub(this.mouseDownPosition),jr()-this.mouseDownTime<.5&&e.length()<2&&this.onMouseClick(t)}})();onMouseClick(e){this.mousePosition.set(e.offsetX,e.offsetY),this.checkForFocalPointChange()}checkForFocalPointChange=(function(){const e=new Xe,t=new U,n=[];return function(){if(!this.transitioningCameraTarget&&(this.getRenderDimensions(e),n.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,e),this.raycaster.intersectSplatMesh(this.splatMesh,n),n.length>0)){const r=n[0].origin;t.copy(r).sub(this.camera.position),t.length()>v1&&(this.previousCameraTarget.copy(this.controls.target),this.nextCameraTarget.copy(r),this.transitioningCameraTarget=!0,this.transitioningCameraTargetStartTime=jr())}}})();getRenderDimensions(e){this.rootElement?(e.x=this.rootElement.offsetWidth,e.y=this.rootElement.offsetHeight):this.renderer.getSize(e)}setOrthographicMode(e){if(e===this.camera.isOrthographicCamera)return;const t=this.camera,n=e?this.orthographicCamera:this.perspectiveCamera;if(n.position.copy(t.position),n.up.copy(t.up),n.rotation.copy(t.rotation),n.quaternion.copy(t.quaternion),n.matrix.copy(t.matrix),this.camera=n,this.controls){const s=a=>{a.saveState(),a.reset()},r=this.controls,o=e?this.orthographicControls:this.perspectiveControls;s(o),s(r),o.target.copy(r.target),e?io.setCameraZoomFromPosition(n,t,r):io.setCameraPositionFromZoom(n,t,o),this.controls=o,this.camera.lookAt(this.controls.target)}}static setCameraPositionFromZoom=(function(){const e=new U;return function(t,n,s){const r=1/(n.zoom*.001);e.copy(s.target).sub(t.position).normalize().multiplyScalar(r).negate(),t.position.copy(s.target).add(e)}})();static setCameraZoomFromPosition=(function(){const e=new U;return function(t,n,s){const r=e.copy(s.target).sub(n.position).length();t.zoom=1/(r*.001)}})();updateSplatMesh=(function(){const e=new Xe;return function(){if(!this.splatMesh)return;if(this.splatMesh.getSplatCount()>0){this.splatMesh.updateVisibleRegionFadeDistance(this.sceneRevealMode),this.splatMesh.updateTransforms(),this.getRenderDimensions(e);const n=this.camera.projectionMatrix.elements[0]*.5*this.devicePixelRatio*e.x,s=this.camera.projectionMatrix.elements[5]*.5*this.devicePixelRatio*e.y,r=this.camera.isOrthographicCamera?1/this.devicePixelRatio:1,o=this.focalAdjustment*r,a=1/o;this.adjustForWebXRStereo(e),this.splatMesh.updateUniforms(e,n*o,s*o,this.camera.isOrthographicCamera,this.camera.zoom||1,a)}}})();adjustForWebXRStereo(e){if(this.camera&&this.webXRActive){const n=this.renderer.xr.getCamera().projectionMatrix.elements[0],s=this.camera.projectionMatrix.elements[0];e.x*=s/n}}isLoadingOrUnloading(){return Object.keys(this.splatSceneDownloadPromises).length>0||this.splatSceneDownloadAndBuildPromise!==null||this.splatSceneRemovalPromise!==null}isDisposingOrDisposed(){return this.disposing||this.disposed}addSplatSceneDownloadPromise(e){this.splatSceneDownloadPromises[e.id]=e}removeSplatSceneDownloadPromise(e){delete this.splatSceneDownloadPromises[e.id]}setSplatSceneDownloadAndBuildPromise(e){this.splatSceneDownloadAndBuildPromise=e}clearSplatSceneDownloadAndBuildPromise(){this.splatSceneDownloadAndBuildPromise=null}addSplatScene(e,t={}){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");t.progressiveLoad&&this.splatMesh.scenes&&this.splatMesh.scenes.length>0&&(console.log('addSplatScene(): "progressiveLoad" option ignore because there are multiple splat scenes'),t.progressiveLoad=!1);const n=t.format!==void 0&&t.format!==null?t.format:fm(e),s=io.isProgressivelyLoadable(n)&&t.progressiveLoad,r=t.showLoadingUI!==void 0&&t.showLoadingUI!==null?t.showLoadingUI:!0;let o=null;r&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=()=>{this.loadingProgressBar.hide(),this.loadingSpinner.removeAllTasks()},l=(p,g,m)=>{if(r)if(m===Wt.Downloading)if(p==100)this.loadingSpinner.setMessageForTask(o,"Download complete!");else if(s)this.loadingSpinner.setMessageForTask(o,"Downloading splats...");else{const _=g?`: ${g}`:"...";this.loadingSpinner.setMessageForTask(o,`Downloading${_}`)}else m===Wt.Processing&&this.loadingSpinner.setMessageForTask(o,"Processing splats...")};let c=!1,u=0;const f=(p,g)=>{r&&((p&&s||g&&!s)&&(this.loadingSpinner.removeTask(o),!g&&!c&&this.loadingProgressBar.show()),s&&(g?(c=!0,this.loadingProgressBar.hide()):this.loadingProgressBar.setProgress(u)))},d=(p,g,m)=>{u=p,l(p,g,m),t.onProgress&&t.onProgress(p,g,m)},h=(p,g,m)=>{!s&&t.onProgress&&t.onProgress(0,"0%",Wt.Processing);const _={rotation:t.rotation||t.orientation,position:t.position,scale:t.scale,splatAlphaRemovalThreshold:t.splatAlphaRemovalThreshold};return this.addSplatBuffers([p],[_],m,g&&r,r,s,s).then(()=>{!s&&t.onProgress&&t.onProgress(100,"100%",Wt.Processing),f(g,m)})};return(s?this.downloadAndBuildSingleSplatSceneProgressiveLoad.bind(this):this.downloadAndBuildSingleSplatSceneStandardLoad.bind(this))(e,n,t.splatAlphaRemovalThreshold,h.bind(this),d,a.bind(this),t.headers)}downloadAndBuildSingleSplatSceneStandardLoad(e,t,n,s,r,o,a){const l=this.downloadSplatSceneToSplatBuffer(e,n,r,!1,void 0,t,a),c=bu(l.abortHandler);return l.then(u=>(this.removeSplatSceneDownloadPromise(l),s(u,!0,!0).then(()=>{c.resolve(),this.clearSplatSceneDownloadAndBuildPromise()}))).catch(u=>{o&&o(),this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(l),c.reject(this.updateError(u,`Viewer::addSplatScene -> Could not load file ${e}`))}),this.addSplatSceneDownloadPromise(l),this.setSplatSceneDownloadAndBuildPromise(c.promise),c.promise}downloadAndBuildSingleSplatSceneProgressiveLoad(e,t,n,s,r,o,a){let l=0,c=!1;const u=[],f=()=>{if(u.length>0&&!c&&!this.isDisposingOrDisposed()){c=!0;const g=u.shift();s(g.splatBuffer,g.firstBuild,g.finalBuild).then(()=>{c=!1,g.firstBuild?x.resolve():g.finalBuild&&(p.resolve(),this.clearSplatSceneDownloadAndBuildPromise()),u.length>0&&Zn(()=>f())})}},d=(g,m)=>{this.isDisposingOrDisposed()||(m||u.length===0||g.getSplatCount()>u[0].splatBuffer.getSplatCount())&&(u.push({splatBuffer:g,firstBuild:l===0,finalBuild:m}),l++,f())},h=this.downloadSplatSceneToSplatBuffer(e,n,r,!0,d,t,a),x=bu(h.abortHandler),p=bu();return this.addSplatSceneDownloadPromise(h),this.setSplatSceneDownloadAndBuildPromise(p.promise),h.then(()=>{this.removeSplatSceneDownloadPromise(h)}).catch(g=>{this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(h);const m=this.updateError(g,"Viewer::addSplatScene -> Could not load one or more scenes");x.reject(m),o&&o(m)}),x.promise}addSplatScenes(e,t=!0,n=void 0){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");const s=e.length,r=[];let o;t&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=(f,d,h,x)=>{r[f]=d;let p=0;for(let g=0;g<s;g++)p+=r[g]||0;p=p/s,h=`${p.toFixed(2)}%`,t&&x===Wt.Downloading&&this.loadingSpinner.setMessageForTask(o,p==100?"Download complete!":`Downloading: ${h}`),n&&n(p,h,x)},l=[],c=[];for(let f=0;f<e.length;f++){const d=e[f],h=d.format!==void 0&&d.format!==null?d.format:fm(d.path),x=this.downloadSplatSceneToSplatBuffer(d.path,d.splatAlphaRemovalThreshold,a.bind(this,f),!1,void 0,h,d.headers);l.push(x),c.push(x.promise)}const u=new Bs((f,d)=>{Promise.all(c).then(h=>{t&&this.loadingSpinner.removeTask(o),n&&n(0,"0%",Wt.Processing),this.addSplatBuffers(h,e,!0,t,t,!1,!1).then(()=>{n&&n(100,"100%",Wt.Processing),this.clearSplatSceneDownloadAndBuildPromise(),f()})}).catch(h=>{t&&this.loadingSpinner.removeTask(o),this.clearSplatSceneDownloadAndBuildPromise(),d(this.updateError(h,"Viewer::addSplatScenes -> Could not load one or more splat scenes."))}).finally(()=>{this.removeSplatSceneDownloadPromise(u)})},f=>{for(let d of l)d.abort(f)});return this.addSplatSceneDownloadPromise(u),this.setSplatSceneDownloadAndBuildPromise(u),u}downloadSplatSceneToSplatBuffer(e,t=1,n=void 0,s=!1,r=void 0,o,a){try{if(o===zn.Splat||o===zn.KSplat||o===zn.Ply){const l=s?!1:this.optimizeSplatData;if(o===zn.Splat)return Yd.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,a);if(o===zn.KSplat)return pa.loadFromURL(e,n,s,r,a);if(o===zn.Ply)return Xd.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,this.sphericalHarmonicsDegree,a)}else if(o===zn.Spz)return qd.loadFromURL(e,n,t,this.inMemoryCompressionLevel,this.optimizeSplatData,this.sphericalHarmonicsDegree,a)}catch(l){throw this.updateError(l,null)}throw new Error(`Viewer::downloadSplatSceneToSplatBuffer -> File format not supported: ${e}`)}static isProgressivelyLoadable(e){return e===zn.Splat||e===zn.KSplat||e===zn.Ply}addSplatBuffers=(function(){return function(e,t=[],n=!0,s=!0,r=!0,o=!1,a=!1,l=!0){if(this.isDisposingOrDisposed())return Promise.resolve();let c=null;const u=()=>{c!==null&&(this.loadingSpinner.removeTask(c),c=null)};return this.splatRenderReady=!1,new Promise(f=>{s&&(c=this.loadingSpinner.addTask("Processing splats...")),Zn(()=>{if(this.isDisposingOrDisposed())f();else{const d=this.addSplatBuffersToMesh(e,t,n,r,o,l),h=this.splatMesh.getMaxSplatCount();this.sortWorker&&this.sortWorker.maxSplatCount!==h&&this.disposeSortWorker(),this.gpuAcceleratedSort||this.preSortMessages.push({centers:d.centers.buffer,sceneIndexes:d.sceneIndexes.buffer,range:{from:d.from,to:d.to,count:d.count}}),(!this.sortWorker&&h>0?this.setupSortWorker(this.splatMesh):Promise.resolve()).then(()=>{this.isDisposingOrDisposed()||this.runSplatSort(!0,!0).then(p=>{!this.sortWorker||!p?(this.splatRenderReady=!0,u(),f()):(a?this.splatRenderReady=!0:this.runAfterNextSort.push(()=>{this.splatRenderReady=!0}),this.runAfterNextSort.push(()=>{u(),f()}))})})}},!0)})}})();addSplatBuffersToMesh=(function(){let e;return function(t,n,s=!0,r=!1,o=!1,a=!0){if(this.isDisposingOrDisposed())return;let l=[],c=[];o||(l=this.splatMesh.scenes.map(h=>h.splatBuffer)||[],c=this.splatMesh.sceneOptions?this.splatMesh.sceneOptions.map(h=>h):[]),l.push(...t),c.push(...n),this.renderer&&this.splatMesh.setRenderer(this.renderer);const u=h=>{if(this.isDisposingOrDisposed())return;const x=this.splatMesh.getSplatCount();r&&x>=A1&&!h&&!e&&(this.loadingSpinner.setMinimized(!0,!0),e=this.loadingSpinner.addTask("Optimizing data structures..."))},f=h=>{this.isDisposingOrDisposed()||h&&e&&(this.loadingSpinner.removeTask(e),e=null)},d=this.splatMesh.build(l,c,!0,s,u,f,a);return s&&this.freeIntermediateSplatData&&this.splatMesh.freeIntermediateSplatData(),d}})();setupSortWorker(e){if(!this.isDisposingOrDisposed())return new Promise(t=>{const n=this.integerBasedSort?Int32Array:Float32Array,s=e.getSplatCount(),r=e.getMaxSplatCount();this.sortWorker=g1(r,this.sharedMemoryForWorkers,this.enableSIMDInSort,this.integerBasedSort,this.splatMesh.dynamicMode,this.splatSortDistanceMapPrecision),this.sortWorker.onmessage=o=>{if(o.data.sortDone){if(this.sortRunning=!1,this.sharedMemoryForWorkers)this.splatMesh.updateRenderIndexes(this.sortWorkerSortedIndexes,o.data.splatRenderCount);else{const a=new Uint32Array(o.data.sortedIndexes.buffer,0,o.data.splatRenderCount);this.splatMesh.updateRenderIndexes(a,o.data.splatRenderCount)}this.lastSplatSortCount=this.splatSortCount,this.lastSortTime=o.data.sortTime,this.sortPromiseResolver(),this.sortPromiseResolver=null,this.forceRenderNextFrame(),this.runAfterNextSort.length>0&&(this.runAfterNextSort.forEach(a=>{a()}),this.runAfterNextSort.length=0)}else if(o.data.sortCanceled)this.sortRunning=!1;else if(o.data.sortSetupPhase1Complete){this.logLevel>=xo.Info&&console.log("Sorting web worker WASM setup complete."),this.sharedMemoryForWorkers?(this.sortWorkerSortedIndexes=new Uint32Array(o.data.sortedIndexesBuffer,o.data.sortedIndexesOffset,r),this.sortWorkerIndexesToSort=new Uint32Array(o.data.indexesToSortBuffer,o.data.indexesToSortOffset,r),this.sortWorkerPrecomputedDistances=new n(o.data.precomputedDistancesBuffer,o.data.precomputedDistancesOffset,r),this.sortWorkerTransforms=new Float32Array(o.data.transformsBuffer,o.data.transformsOffset,bt.MaxScenes*16)):(this.sortWorkerIndexesToSort=new Uint32Array(r),this.sortWorkerPrecomputedDistances=new n(r),this.sortWorkerTransforms=new Float32Array(bt.MaxScenes*16));for(let a=0;a<s;a++)this.sortWorkerIndexesToSort[a]=a;if(this.sortWorker.maxSplatCount=r,this.logLevel>=xo.Info){console.log("Sorting web worker ready.");const a=this.splatMesh.getSplatDataTextures(),l=a.covariances.size,c=a.centerColors.size;console.log("Covariances texture size: "+l.x+" x "+l.y),console.log("Centers/colors texture size: "+c.x+" x "+c.y)}t()}}})}updateError(e,t){return e instanceof Tg?e:e instanceof ic?new Error("File type or server does not support progressive loading."):t?new Error(t):e}disposeSortWorker(){this.sortWorker&&this.sortWorker.terminate(),this.sortWorker=null,this.sortPromise=null,this.sortPromiseResolver&&(this.sortPromiseResolver(),this.sortPromiseResolver=null),this.preSortMessages=[],this.sortRunning=!1}removeSplatScene(e,t=!0){return this.removeSplatScenes([e],t)}removeSplatScenes(e,t=!0){if(this.isLoadingOrUnloading())throw new Error("Cannot remove splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot remove splat scene after dispose() is called.");let n;return this.splatSceneRemovalPromise=new Promise((s,r)=>{let o;t&&(this.loadingSpinner.removeAllTasks(),this.loadingSpinner.show(),o=this.loadingSpinner.addTask("Removing splat scene..."));const a=()=>{t&&(this.loadingSpinner.hide(),this.loadingSpinner.removeTask(o))},l=u=>{a(),this.splatSceneRemovalPromise=null,u?r(u):s()},c=()=>this.isDisposingOrDisposed()?(l(),!0):!1;n=this.sortPromise||Promise.resolve(),n.then(()=>{if(c())return;const u=[],f=[],d=[];for(let h=0;h<this.splatMesh.scenes.length;h++){let x=!1;for(let p of e)if(p===h){x=!0;break}if(!x){const p=this.splatMesh.scenes[h];u.push(p.splatBuffer),f.push(this.splatMesh.sceneOptions[h]),d.push({position:p.position.clone(),quaternion:p.quaternion.clone(),scale:p.scale.clone()})}}this.disposeSortWorker(),this.splatMesh.dispose(),this.sceneRevealMode=xa.Instant,this.createSplatMesh(),this.addSplatBuffers(u,f,!0,!1,!0).then(()=>{c()||(a(),this.splatMesh.scenes.forEach((h,x)=>{h.position.copy(d[x].position),h.quaternion.copy(d[x].quaternion),h.scale.copy(d[x].scale)}),this.splatMesh.updateTransforms(),this.splatRenderReady=!1,this.runSplatSort(!0).then(()=>{if(c()){this.splatRenderReady=!0;return}n=this.sortPromise||Promise.resolve(),n.then(()=>{this.splatRenderReady=!0,l()})}))}).catch(h=>{l(h)})})}),this.splatSceneRemovalPromise}start(){if(this.selfDrivenMode)this.webXRMode?this.renderer.setAnimationLoop(this.selfDrivenUpdateFunc):this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc),this.selfDrivenModeRunning=!0;else throw new Error("Cannot start viewer unless it is in self driven mode.")}stop(){this.selfDrivenMode&&this.selfDrivenModeRunning&&(this.webXRMode?this.renderer.setAnimationLoop(null):cancelAnimationFrame(this.requestFrameId),this.selfDrivenModeRunning=!1)}async dispose(){if(this.isDisposingOrDisposed())return this.disposePromise;let e=[],t=[];for(let n in this.splatSceneDownloadPromises)if(this.splatSceneDownloadPromises.hasOwnProperty(n)){const s=this.splatSceneDownloadPromises[n];t.push(s),e.push(s.promise)}return this.sortPromise&&e.push(this.sortPromise),this.disposing=!0,this.disposePromise=Promise.all(e).finally(()=>{this.stop(),this.orthographicControls&&(this.orthographicControls.dispose(),this.orthographicControls=null),this.perspectiveControls&&(this.perspectiveControls.dispose(),this.perspectiveControls=null),this.controls=null,this.splatMesh&&(this.splatMesh.dispose(),this.splatMesh=null),this.sceneHelper&&(this.sceneHelper.dispose(),this.sceneHelper=null),this.resizeObserver&&(this.resizeObserver.unobserve(this.rootElement),this.resizeObserver=null),this.disposeSortWorker(),this.removeEventHandlers(),this.loadingSpinner.removeAllTasks(),this.loadingSpinner.setContainer(null),this.loadingProgressBar.hide(),this.loadingProgressBar.setContainer(null),this.infoPanel.setContainer(null),this.camera=null,this.threeScene=null,this.splatRenderReady=!1,this.initialized=!1,this.renderer&&(this.usingExternalRenderer||(this.rootElement.removeChild(this.renderer.domElement),this.renderer.dispose()),this.renderer=null),this.usingExternalRenderer||document.body.removeChild(this.rootElement),this.sortWorkerSortedIndexes=null,this.sortWorkerIndexesToSort=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.disposed=!0,this.disposing=!1,this.disposePromise=null}),t.forEach(n=>{n.abort("Scene disposed")}),this.disposePromise}selfDrivenUpdate(){this.selfDrivenMode&&!this.webXRMode&&(this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc)),this.update(),this.shouldRender()?(this.render(),this.consecutiveRenderFrames++):this.consecutiveRenderFrames=0,this.renderNextFrame=!1}forceRenderNextFrame(){this.renderNextFrame=!0}shouldRender=(function(){let e=0;const t=new U,n=new Ft,s=1e-4;return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return!1;let r=!1,o=!1;if(this.camera){const a=this.camera.position,l=this.camera.quaternion;o=Math.abs(a.x-t.x)>s||Math.abs(a.y-t.y)>s||Math.abs(a.z-t.z)>s||Math.abs(l.x-n.x)>s||Math.abs(l.y-n.y)>s||Math.abs(l.z-n.z)>s||Math.abs(l.w-n.w)>s}return r=this.renderMode!==Du.Never&&(e===0||this.splatMesh.visibleRegionChanging||o||this.renderMode===Du.Always||this.dynamicMode===!0||this.renderNextFrame),this.camera&&(t.copy(this.camera.position),n.copy(this.camera.quaternion)),e++,r}})();render=(function(){return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return;const e=n=>{for(let s of n.children)if(s.visible)return!0;return!1},t=this.renderer.autoClear;e(this.threeScene)&&(this.renderer.render(this.threeScene,this.camera),this.renderer.autoClear=!1),this.renderer.render(this.splatMesh,this.camera),this.renderer.autoClear=!1,this.sceneHelper.getFocusMarkerOpacity()>0&&this.renderer.render(this.sceneHelper.focusMarker,this.camera),this.showControlPlane&&this.renderer.render(this.sceneHelper.controlPlane,this.camera),this.renderer.autoClear=t}})();update(e,t){this.dropInMode&&this.updateForDropInMode(e,t),!(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())&&(this.controls&&(this.controls.update(),this.camera.isOrthographicCamera&&!this.usingExternalCamera&&io.setCameraPositionFromZoom(this.camera,this.camera,this.controls)),this.runSplatSort(),this.updateForRendererSizeChanges(),this.updateSplatMesh(),this.updateMeshCursor(),this.updateFPS(),this.timingSensitiveUpdates(),this.updateInfoPanel(),this.updateControlPlane())}updateForDropInMode(e,t){this.renderer=e,this.splatMesh&&this.splatMesh.setRenderer(this.renderer),this.camera=t,this.controls&&(this.controls.object=t),this.init()}updateFPS=(function(){let e=jr(),t=0;return function(){if(this.consecutiveRenderFrames>b1){const n=jr();n-e>=1?(this.currentFPS=t,t=0,e=n):t++}else this.currentFPS=null}})();updateForRendererSizeChanges=(function(){const e=new Xe,t=new Xe;let n;return function(){this.usingExternalCamera||(this.renderer.getSize(t),(n===void 0||n!==this.camera.isOrthographicCamera||t.x!==e.x||t.y!==e.y)&&(this.camera.isOrthographicCamera?(this.camera.left=-t.x/2,this.camera.right=t.x/2,this.camera.top=t.y/2,this.camera.bottom=-t.y/2):this.camera.aspect=t.x/t.y,this.camera.updateProjectionMatrix(),e.copy(t),n=this.camera.isOrthographicCamera))}})();timingSensitiveUpdates=(function(){let e;return function(){const t=jr();e||(e=t);const n=t-e;this.updateCameraTransition(t),this.updateFocusMarker(n),e=t}})();updateCameraTransition=(function(){let e=new U,t=new U,n=new U;return function(s){if(this.transitioningCameraTarget){t.copy(this.previousCameraTarget).sub(this.camera.position).normalize(),n.copy(this.nextCameraTarget).sub(this.camera.position).normalize();const r=Math.acos(t.dot(n)),a=(r/(Math.PI/3)*.65+.3)/r*(s-this.transitioningCameraTargetStartTime);e.copy(this.previousCameraTarget).lerp(this.nextCameraTarget,a),this.camera.lookAt(e),this.controls.target.copy(e),a>=1&&(this.transitioningCameraTarget=!1)}}})();updateFocusMarker=(function(){const e=new Xe;let t=!1;return function(n){if(this.getRenderDimensions(e),this.transitioningCameraTarget){this.sceneHelper.setFocusMarkerVisibility(!0);const s=Math.max(this.sceneHelper.getFocusMarkerOpacity(),0);let r=Math.min(s+S1*n,1);this.sceneHelper.setFocusMarkerOpacity(r),this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e),t=!0,this.forceRenderNextFrame()}else{let s;if(t?s=1:s=Math.min(this.sceneHelper.getFocusMarkerOpacity(),1),s>0){this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e);let r=Math.max(s-y1*n,0);this.sceneHelper.setFocusMarkerOpacity(r),r===0&&this.sceneHelper.setFocusMarkerVisibility(!1)}s>0&&this.forceRenderNextFrame(),t=!1}}})();updateMeshCursor=(function(){const e=[],t=new Xe;return function(){this.showMeshCursor?(this.forceRenderNextFrame(),this.getRenderDimensions(t),e.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,t),this.raycaster.intersectSplatMesh(this.splatMesh,e),e.length>0?(this.sceneHelper.setMeshCursorVisibility(!0),this.sceneHelper.positionAndOrientMeshCursor(e[0].origin,this.camera)):this.sceneHelper.setMeshCursorVisibility(!1)):(this.sceneHelper.getMeschCursorVisibility()&&this.forceRenderNextFrame(),this.sceneHelper.setMeshCursorVisibility(!1))}})();updateInfoPanel=(function(){const e=new Xe;return function(){if(!this.showInfo)return;const t=this.splatMesh.getSplatCount();this.getRenderDimensions(e);const n=this.controls?this.controls.target:null,s=this.showMeshCursor?this.sceneHelper.meshCursor.position:null,r=t>0?this.splatRenderCount/t*100:0;this.infoPanel.update(e,this.camera.position,n,this.camera.up,this.camera.isOrthographicCamera,s,this.currentFPS||"N/A",t,this.splatRenderCount,r,this.lastSortTime,this.focalAdjustment,this.splatMesh.getSplatScale(),this.splatMesh.getPointCloudModeEnabled())}})();updateControlPlane(){this.showControlPlane?(this.sceneHelper.setControlPlaneVisibility(!0),this.sceneHelper.positionAndOrientControlPlane(this.controls.target,this.camera.up)):this.sceneHelper.setControlPlaneVisibility(!1)}runSplatSort=(function(){const e=new $e,t=[],n=new U(0,0,-1),s=new U(0,0,-1),r=new U,o=new U,a=[],l=[{angleThreshold:.55,sortFractions:[.125,.33333,.75]},{angleThreshold:.65,sortFractions:[.33333,.66667]},{angleThreshold:.8,sortFractions:[.5]}];return function(c=!1,u=!1){if(!this.initialized)return Promise.resolve(!1);if(this.sortRunning)return Promise.resolve(!0);if(this.splatMesh.getSplatCount()<=0)return this.splatRenderCount=0,Promise.resolve(!1);let f=0,d=0,h=!1,x=!1;if(s.set(0,0,-1).applyQuaternion(this.camera.quaternion),f=s.dot(n),d=o.copy(this.camera.position).sub(r).length(),!c&&!this.splatMesh.dynamicMode&&a.length===0&&(f<=.99&&(h=!0),d>=1&&(x=!0),!h&&!x))return Promise.resolve(!1);this.sortRunning=!0;let{splatRenderCount:p,shouldSortAll:g}=this.gatherSceneNodesForSort();g=g||u,this.splatRenderCount=p,e.copy(this.camera.matrixWorld).invert();const m=this.perspectiveCamera||this.camera;e.premultiply(m.projectionMatrix),this.splatMesh.dynamicMode||e.multiply(this.splatMesh.matrixWorld);let _=Promise.resolve(!0);return this.gpuAcceleratedSort&&(a.length<=1||a.length%2===0)&&(_=this.splatMesh.computeDistancesOnGPU(e,this.sortWorkerPrecomputedDistances)),_.then(()=>{if(a.length===0)if(this.splatMesh.dynamicMode||g)a.push(this.splatRenderCount);else{for(let S of l)if(f<S.angleThreshold){for(let b of S.sortFractions)a.push(Math.floor(this.splatRenderCount*b));break}a.push(this.splatRenderCount)}let v=Math.min(a.shift(),this.splatRenderCount);this.splatSortCount=v,t[0]=this.camera.position.x,t[1]=this.camera.position.y,t[2]=this.camera.position.z;const A={modelViewProj:e.elements,cameraPosition:t,splatRenderCount:this.splatRenderCount,splatSortCount:v,usePrecomputedDistances:this.gpuAcceleratedSort};return this.splatMesh.dynamicMode&&this.splatMesh.fillTransformsArray(this.sortWorkerTransforms),this.sharedMemoryForWorkers||(A.indexesToSort=this.sortWorkerIndexesToSort,A.transforms=this.sortWorkerTransforms,this.gpuAcceleratedSort&&(A.precomputedDistances=this.sortWorkerPrecomputedDistances)),this.sortPromise=new Promise(S=>{this.sortPromiseResolver=S}),this.preSortMessages.length>0&&(this.preSortMessages.forEach(S=>{this.sortWorker.postMessage(S)}),this.preSortMessages=[]),this.sortWorker.postMessage({sort:A}),a.length===0&&(r.copy(this.camera.position),n.copy(s)),!0}),_}})();gatherSceneNodesForSort=(function(){const e=[];let t=null;const n=new U,s=new U,r=new U,o=new $e,a=new $e,l=new $e,c=new U,u=new U(0,0,-1),f=new U,d=h=>f.copy(h.max).sub(h.min).length();return function(h=!1){this.getRenderDimensions(c);const x=c.y/2/Math.tan(this.camera.fov/2*Ds.DEG2RAD),p=Math.atan(c.x/2/x),g=Math.atan(c.y/2/x),m=Math.cos(p),_=Math.cos(g),v=this.splatMesh.getSplatTree();if(v){a.copy(this.camera.matrixWorld).invert(),this.splatMesh.dynamicMode||a.multiply(this.splatMesh.matrixWorld);let A=0,S=0;for(let M=0;M<v.subTrees.length;M++){const R=v.subTrees[M];o.copy(a),this.splatMesh.dynamicMode&&(this.splatMesh.getSceneTransform(M,l),o.multiply(l));const y=R.nodesWithIndexes.length;for(let C=0;C<y;C++){const F=R.nodesWithIndexes[C];if(!F.data||!F.data.indexes||F.data.indexes.length===0)continue;r.copy(F.center).applyMatrix4(o);const I=r.length();r.normalize(),n.copy(r).setX(0).normalize(),s.copy(r).setY(0).normalize();const O=u.dot(s),z=u.dot(n),H=d(F),V=z<_-.6,q=O<m-.6;!h&&(q||V)&&I>H||(S+=F.data.indexes.length,e[A]=F,F.data.distanceToNode=I,A++)}}e.length=A,e.sort((M,R)=>M.data.distanceToNode<R.data.distanceToNode?-1:1);let b=S*bt.BytesPerInt;for(let M=0;M<A;M++){const R=e[M],y=R.data.indexes.length,C=y*bt.BytesPerInt;new Uint32Array(this.sortWorkerIndexesToSort.buffer,b-C,y).set(R.data.indexes),b-=C}return{splatRenderCount:S,shouldSortAll:!1}}else{const A=this.splatMesh.getSplatCount();if(!t||t.length!==A){t=new Uint32Array(A);for(let S=0;S<A;S++)t[S]=S}return this.sortWorkerIndexesToSort.set(t),{splatRenderCount:A,shouldSortAll:!0}}}})();getSplatMesh(){return this.splatMesh}getSplatScene(e){return this.splatMesh.getScene(e)}getSceneCount(){return this.splatMesh.getSceneCount()}isMobile(){return navigator.userAgent.includes("Mobi")}}function ss(i){if(i===void 0)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return i}function Bg(i,e){i.prototype=Object.create(e.prototype),i.prototype.constructor=i,i.__proto__=e}var ni={autoSleep:120,force3D:"auto",nullTargetWarn:1,units:{lineHeight:""}},Do={duration:.5,overwrite:!1,delay:0},$d,un,Ut,pi=1e8,Tt=1/pi,Vf=Math.PI*2,M1=Vf/4,C1=0,Ug=Math.sqrt,T1=Math.cos,E1=Math.sin,sn=function(e){return typeof e=="string"},Vt=function(e){return typeof e=="function"},xs=function(e){return typeof e=="number"},Zd=function(e){return typeof e>"u"},Wi=function(e){return typeof e=="object"},Hn=function(e){return e!==!1},Jd=function(){return typeof window<"u"},Pl=function(e){return Vt(e)||sn(e)},Og=typeof ArrayBuffer=="function"&&ArrayBuffer.isView||function(){},Sn=Array.isArray,w1=/random\([^)]+\)/g,R1=/,\s*/g,Sm=/(?:-?\.?\d|\.)+/gi,Ng=/[-+=.]*\d+[.e\-+]*\d*[e\-+]*\d*/g,so=/[-+=.]*\d+[.e-]*\d*[a-z%]*/g,Pu=/[-+=.]*\d+\.?\d*(?:e-|e\+)?\d*/gi,zg=/[+-]=-?[.\d]+/,I1=/[^,'"\[\]\s]+/gi,D1=/^[+\-=e\s\d]*\d+[.\d]*([a-z]*|%)\s*$/i,kt,Ri,Gf,eh,ii={},oc={},kg,Hg=function(e){return(oc=Po(e,ii))&&Xn},th=function(e,t){return console.warn("Invalid property",e,"set to",t,"Missing plugin? gsap.registerPlugin()")},Ba=function(e,t){return!t&&console.warn(e)},Vg=function(e,t){return e&&(ii[e]=t)&&oc&&(oc[e]=t)||ii},Ua=function(){return 0},P1={suppressEvents:!0,isStart:!0,kill:!1},Gl={suppressEvents:!0,kill:!1},F1={suppressEvents:!0},nh={},Vs=[],Wf={},Gg,Kn={},Fu={},ym=30,Wl=[],ih="",sh=function(e){var t=e[0],n,s;if(Wi(t)||Vt(t)||(e=[e]),!(n=(t._gsap||{}).harness)){for(s=Wl.length;s--&&!Wl[s].targetTest(t););n=Wl[s]}for(s=e.length;s--;)e[s]&&(e[s]._gsap||(e[s]._gsap=new hx(e[s],n)))||e.splice(s,1);return e},br=function(e){return e._gsap||sh(mi(e))[0]._gsap},Wg=function(e,t,n){return(n=e[t])&&Vt(n)?e[t]():Zd(n)&&e.getAttribute&&e.getAttribute(t)||n},Vn=function(e,t){return(e=e.split(",")).forEach(t)||e},Gt=function(e){return Math.round(e*1e5)/1e5||0},zt=function(e){return Math.round(e*1e7)/1e7||0},_o=function(e,t){var n=t.charAt(0),s=parseFloat(t.substr(2));return e=parseFloat(e),n==="+"?e+s:n==="-"?e-s:n==="*"?e*s:e/s},L1=function(e,t){for(var n=t.length,s=0;e.indexOf(t[s])<0&&++s<n;);return s<n},ac=function(){var e=Vs.length,t=Vs.slice(0),n,s;for(Wf={},Vs.length=0,n=0;n<e;n++)s=t[n],s&&s._lazy&&(s.render(s._lazy[0],s._lazy[1],!0)._lazy=0)},rh=function(e){return!!(e._initted||e._startAt||e.add)},Xg=function(e,t,n,s){Vs.length&&!un&&ac(),e.render(t,n,!!(un&&t<0&&rh(e))),Vs.length&&!un&&ac()},qg=function(e){var t=parseFloat(e);return(t||t===0)&&(e+"").match(I1).length<2?t:sn(e)?e.trim():e},Yg=function(e){return e},si=function(e,t){for(var n in t)n in e||(e[n]=t[n]);return e},B1=function(e){return function(t,n){for(var s in n)s in t||s==="duration"&&e||s==="ease"||(t[s]=n[s])}},Po=function(e,t){for(var n in t)e[n]=t[n];return e},bm=function i(e,t){for(var n in t)n!=="__proto__"&&n!=="constructor"&&n!=="prototype"&&(e[n]=Wi(t[n])?i(e[n]||(e[n]={}),t[n]):t[n]);return e},lc=function(e,t){var n={},s;for(s in e)s in t||(n[s]=e[s]);return n},_a=function(e){var t=e.parent||kt,n=e.keyframes?B1(Sn(e.keyframes)):si;if(Hn(e.inherit))for(;t;)n(e,t.vars.defaults),t=t.parent||t._dp;return e},U1=function(e,t){for(var n=e.length,s=n===t.length;s&&n--&&e[n]===t[n];);return n<0},Qg=function(e,t,n,s,r){var o=e[s],a;if(r)for(a=t[r];o&&o[r]>a;)o=o._prev;return o?(t._next=o._next,o._next=t):(t._next=e[n],e[n]=t),t._next?t._next._prev=t:e[s]=t,t._prev=o,t.parent=t._dp=e,t},Rc=function(e,t,n,s){n===void 0&&(n="_first"),s===void 0&&(s="_last");var r=t._prev,o=t._next;r?r._next=o:e[n]===t&&(e[n]=o),o?o._prev=r:e[s]===t&&(e[s]=r),t._next=t._prev=t.parent=null},Ys=function(e,t){e.parent&&(!t||e.parent.autoRemoveChildren)&&e.parent.remove&&e.parent.remove(e),e._act=0},Mr=function(e,t){if(e&&(!t||t._end>e._dur||t._start<0))for(var n=e;n;)n._dirty=1,n=n.parent;return e},O1=function(e){for(var t=e.parent;t&&t.parent;)t._dirty=1,t.totalDuration(),t=t.parent;return e},Xf=function(e,t,n,s){return e._startAt&&(un?e._startAt.revert(Gl):e.vars.immediateRender&&!e.vars.autoRevert||e._startAt.render(t,!0,s))},N1=function i(e){return!e||e._ts&&i(e.parent)},Mm=function(e){return e._repeat?Fo(e._tTime,e=e.duration()+e._rDelay)*e:0},Fo=function(e,t){var n=Math.floor(e=zt(e/t));return e&&n===e?n-1:n},cc=function(e,t){return(e-t._start)*t._ts+(t._ts>=0?0:t._dirty?t.totalDuration():t._tDur)},Ic=function(e){return e._end=zt(e._start+(e._tDur/Math.abs(e._ts||e._rts||Tt)||0))},Dc=function(e,t){var n=e._dp;return n&&n.smoothChildTiming&&e._ts&&(e._start=zt(n._time-(e._ts>0?t/e._ts:((e._dirty?e.totalDuration():e._tDur)-t)/-e._ts)),Ic(e),n._dirty||Mr(n,e)),e},Kg=function(e,t){var n;if((t._time||!t._dur&&t._initted||t._start<e._time&&(t._dur||!t.add))&&(n=cc(e.rawTime(),t),(!t._dur||Qa(0,t.totalDuration(),n)-t._tTime>Tt)&&t.render(n,!0)),Mr(e,t)._dp&&e._initted&&e._time>=e._dur&&e._ts){if(e._dur<e.duration())for(n=e;n._dp;)n.rawTime()>=0&&n.totalTime(n._tTime),n=n._dp;e._zTime=-Tt}},Fi=function(e,t,n,s){return t.parent&&Ys(t),t._start=zt((xs(n)?n:n||e!==kt?ai(e,n,t):e._time)+t._delay),t._end=zt(t._start+(t.totalDuration()/Math.abs(t.timeScale())||0)),Qg(e,t,"_first","_last",e._sort?"_start":0),qf(t)||(e._recent=t),s||Kg(e,t),e._ts<0&&Dc(e,e._tTime),e},jg=function(e,t){return(ii.ScrollTrigger||th("scrollTrigger",t))&&ii.ScrollTrigger.create(t,e)},$g=function(e,t,n,s,r){if(ah(e,t,r),!e._initted)return 1;if(!n&&e._pt&&!un&&(e._dur&&e.vars.lazy!==!1||!e._dur&&e.vars.lazy)&&Gg!==jn.frame)return Vs.push(e),e._lazy=[r,s],1},z1=function i(e){var t=e.parent;return t&&t._ts&&t._initted&&!t._lock&&(t.rawTime()<0||i(t))},qf=function(e){var t=e.data;return t==="isFromStart"||t==="isStart"},k1=function(e,t,n,s){var r=e.ratio,o=t<0||!t&&(!e._start&&z1(e)&&!(!e._initted&&qf(e))||(e._ts<0||e._dp._ts<0)&&!qf(e))?0:1,a=e._rDelay,l=0,c,u,f;if(a&&e._repeat&&(l=Qa(0,e._tDur,t),u=Fo(l,a),e._yoyo&&u&1&&(o=1-o),u!==Fo(e._tTime,a)&&(r=1-o,e.vars.repeatRefresh&&e._initted&&e.invalidate())),o!==r||un||s||e._zTime===Tt||!t&&e._zTime){if(!e._initted&&$g(e,t,s,n,l))return;for(f=e._zTime,e._zTime=t||(n?Tt:0),n||(n=t&&!f),e.ratio=o,e._from&&(o=1-o),e._time=0,e._tTime=l,c=e._pt;c;)c.r(o,c.d),c=c._next;t<0&&Xf(e,t,n,!0),e._onUpdate&&!n&&Jn(e,"onUpdate"),l&&e._repeat&&!n&&e.parent&&Jn(e,"onRepeat"),(t>=e._tDur||t<0)&&e.ratio===o&&(o&&Ys(e,1),!n&&!un&&(Jn(e,o?"onComplete":"onReverseComplete",!0),e._prom&&e._prom()))}else e._zTime||(e._zTime=t)},H1=function(e,t,n){var s;if(n>t)for(s=e._first;s&&s._start<=n;){if(s.data==="isPause"&&s._start>t)return s;s=s._next}else for(s=e._last;s&&s._start>=n;){if(s.data==="isPause"&&s._start<t)return s;s=s._prev}},Lo=function(e,t,n,s){var r=e._repeat,o=zt(t)||0,a=e._tTime/e._tDur;return a&&!s&&(e._time*=o/e._dur),e._dur=o,e._tDur=r?r<0?1e10:zt(o*(r+1)+e._rDelay*r):o,a>0&&!s&&Dc(e,e._tTime=e._tDur*a),e.parent&&Ic(e),n||Mr(e.parent,e),e},Cm=function(e){return e instanceof Dn?Mr(e):Lo(e,e._dur)},V1={_start:0,endTime:Ua,totalDuration:Ua},ai=function i(e,t,n){var s=e.labels,r=e._recent||V1,o=e.duration()>=pi?r.endTime(!1):e._dur,a,l,c;return sn(t)&&(isNaN(t)||t in s)?(l=t.charAt(0),c=t.substr(-1)==="%",a=t.indexOf("="),l==="<"||l===">"?(a>=0&&(t=t.replace(/=/,"")),(l==="<"?r._start:r.endTime(r._repeat>=0))+(parseFloat(t.substr(1))||0)*(c?(a<0?r:n).totalDuration()/100:1)):a<0?(t in s||(s[t]=o),s[t]):(l=parseFloat(t.charAt(a-1)+t.substr(a+1)),c&&n&&(l=l/100*(Sn(n)?n[0]:n).totalDuration()),a>1?i(e,t.substr(0,a-1),n)+l:o+l)):t==null?o:+t},va=function(e,t,n){var s=xs(t[1]),r=(s?2:1)+(e<2?0:1),o=t[r],a,l;if(s&&(o.duration=t[1]),o.parent=n,e){for(a=o,l=n;l&&!("immediateRender"in a);)a=l.vars.defaults||{},l=Hn(l.vars.inherit)&&l.parent;o.immediateRender=Hn(a.immediateRender),e<2?o.runBackwards=1:o.startAt=t[r-1]}return new Kt(t[0],o,t[r+1])},$s=function(e,t){return e||e===0?t(e):t},Qa=function(e,t,n){return n<e?e:n>t?t:n},_n=function(e,t){return!sn(e)||!(t=D1.exec(e))?"":t[1]},G1=function(e,t,n){return $s(n,function(s){return Qa(e,t,s)})},Yf=[].slice,Zg=function(e,t){return e&&Wi(e)&&"length"in e&&(!t&&!e.length||e.length-1 in e&&Wi(e[0]))&&!e.nodeType&&e!==Ri},W1=function(e,t,n){return n===void 0&&(n=[]),e.forEach(function(s){var r;return sn(s)&&!t||Zg(s,1)?(r=n).push.apply(r,mi(s)):n.push(s)})||n},mi=function(e,t,n){return Ut&&!t&&Ut.selector?Ut.selector(e):sn(e)&&!n&&(Gf||!Bo())?Yf.call((t||eh).querySelectorAll(e),0):Sn(e)?W1(e,n):Zg(e)?Yf.call(e,0):e?[e]:[]},Qf=function(e){return e=mi(e)[0]||Ba("Invalid scope")||{},function(t){var n=e.current||e.nativeElement||e;return mi(t,n.querySelectorAll?n:n===e?Ba("Invalid scope")||eh.createElement("div"):e)}},Jg=function(e){return e.sort(function(){return .5-Math.random()})},ex=function(e){if(Vt(e))return e;var t=Wi(e)?e:{each:e},n=Cr(t.ease),s=t.from||0,r=parseFloat(t.base)||0,o={},a=s>0&&s<1,l=isNaN(s)||a,c=t.axis,u=s,f=s;return sn(s)?u=f={center:.5,edges:.5,end:1}[s]||0:!a&&l&&(u=s[0],f=s[1]),function(d,h,x){var p=(x||t).length,g=o[p],m,_,v,A,S,b,M,R,y;if(!g){if(y=t.grid==="auto"?0:(t.grid||[1,pi])[1],!y){for(M=-pi;M<(M=x[y++].getBoundingClientRect().left)&&y<p;);y<p&&y--}for(g=o[p]=[],m=l?Math.min(y,p)*u-.5:s%y,_=y===pi?0:l?p*f/y-.5:s/y|0,M=0,R=pi,b=0;b<p;b++)v=b%y-m,A=_-(b/y|0),g[b]=S=c?Math.abs(c==="y"?A:v):Ug(v*v+A*A),S>M&&(M=S),S<R&&(R=S);s==="random"&&Jg(g),g.max=M-R,g.min=R,g.v=p=(parseFloat(t.amount)||parseFloat(t.each)*(y>p?p-1:c?c==="y"?p/y:y:Math.max(y,p/y))||0)*(s==="edges"?-1:1),g.b=p<0?r-p:r,g.u=_n(t.amount||t.each)||0,n=n&&p<0?ux(n):n}return p=(g[d]-g.min)/g.max||0,zt(g.b+(n?n(p):p)*g.v)+g.u}},Kf=function(e){var t=Math.pow(10,((e+"").split(".")[1]||"").length);return function(n){var s=zt(Math.round(parseFloat(n)/e)*e*t);return(s-s%1)/t+(xs(n)?0:_n(n))}},tx=function(e,t){var n=Sn(e),s,r;return!n&&Wi(e)&&(s=n=e.radius||pi,e.values?(e=mi(e.values),(r=!xs(e[0]))&&(s*=s)):e=Kf(e.increment)),$s(t,n?Vt(e)?function(o){return r=e(o),Math.abs(r-o)<=s?r:o}:function(o){for(var a=parseFloat(r?o.x:o),l=parseFloat(r?o.y:0),c=pi,u=0,f=e.length,d,h;f--;)r?(d=e[f].x-a,h=e[f].y-l,d=d*d+h*h):d=Math.abs(e[f]-a),d<c&&(c=d,u=f);return u=!s||c<=s?e[u]:o,r||u===o||xs(o)?u:u+_n(o)}:Kf(e))},nx=function(e,t,n,s){return $s(Sn(e)?!t:n===!0?!!(n=0):!s,function(){return Sn(e)?e[~~(Math.random()*e.length)]:(n=n||1e-5)&&(s=n<1?Math.pow(10,(n+"").length-2):1)&&Math.floor(Math.round((e-n/2+Math.random()*(t-e+n*.99))/n)*n*s)/s})},X1=function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(s){return t.reduce(function(r,o){return o(r)},s)}},q1=function(e,t){return function(n){return e(parseFloat(n))+(t||_n(n))}},Y1=function(e,t,n){return sx(e,t,0,1,n)},ix=function(e,t,n){return $s(n,function(s){return e[~~t(s)]})},Q1=function i(e,t,n){var s=t-e;return Sn(e)?ix(e,i(0,e.length),t):$s(n,function(r){return(s+(r-e)%s)%s+e})},K1=function i(e,t,n){var s=t-e,r=s*2;return Sn(e)?ix(e,i(0,e.length-1),t):$s(n,function(o){return o=(r+(o-e)%r)%r||0,e+(o>s?r-o:o)})},Oa=function(e){return e.replace(w1,function(t){var n=t.indexOf("[")+1,s=t.substring(n||7,n?t.indexOf("]"):t.length-1).split(R1);return nx(n?s:+s[0],n?0:+s[1],+s[2]||1e-5)})},sx=function(e,t,n,s,r){var o=t-e,a=s-n;return $s(r,function(l){return n+((l-e)/o*a||0)})},j1=function i(e,t,n,s){var r=isNaN(e+t)?0:function(h){return(1-h)*e+h*t};if(!r){var o=sn(e),a={},l,c,u,f,d;if(n===!0&&(s=1)&&(n=null),o)e={p:e},t={p:t};else if(Sn(e)&&!Sn(t)){for(u=[],f=e.length,d=f-2,c=1;c<f;c++)u.push(i(e[c-1],e[c]));f--,r=function(x){x*=f;var p=Math.min(d,~~x);return u[p](x-p)},n=t}else s||(e=Po(Sn(e)?[]:{},e));if(!u){for(l in t)oh.call(a,e,l,"get",t[l]);r=function(x){return uh(x,a)||(o?e.p:e)}}}return $s(n,r)},Tm=function(e,t,n){var s=e.labels,r=pi,o,a,l;for(o in s)a=s[o]-t,a<0==!!n&&a&&r>(a=Math.abs(a))&&(l=o,r=a);return l},Jn=function(e,t,n){var s=e.vars,r=s[t],o=Ut,a=e._ctx,l,c,u;if(r)return l=s[t+"Params"],c=s.callbackScope||e,n&&Vs.length&&ac(),a&&(Ut=a),u=l?r.apply(c,l):r.call(c),Ut=o,u},na=function(e){return Ys(e),e.scrollTrigger&&e.scrollTrigger.kill(!!un),e.progress()<1&&Jn(e,"onInterrupt"),e},ro,rx=[],ox=function(e){if(e)if(e=!e.name&&e.default||e,Jd()||e.headless){var t=e.name,n=Vt(e),s=t&&!n&&e.init?function(){this._props=[]}:e,r={init:Ua,render:uh,add:oh,kill:dw,modifier:fw,rawVars:0},o={targetTest:0,get:0,getSetter:ch,aliases:{},register:0};if(Bo(),e!==s){if(Kn[t])return;si(s,si(lc(e,r),o)),Po(s.prototype,Po(r,lc(e,o))),Kn[s.prop=t]=s,e.targetTest&&(Wl.push(s),nh[t]=1),t=(t==="css"?"CSS":t.charAt(0).toUpperCase()+t.substr(1))+"Plugin"}Vg(t,s),e.register&&e.register(Xn,s,Gn)}else rx.push(e)},Ct=255,ia={aqua:[0,Ct,Ct],lime:[0,Ct,0],silver:[192,192,192],black:[0,0,0],maroon:[128,0,0],teal:[0,128,128],blue:[0,0,Ct],navy:[0,0,128],white:[Ct,Ct,Ct],olive:[128,128,0],yellow:[Ct,Ct,0],orange:[Ct,165,0],gray:[128,128,128],purple:[128,0,128],green:[0,128,0],red:[Ct,0,0],pink:[Ct,192,203],cyan:[0,Ct,Ct],transparent:[Ct,Ct,Ct,0]},Lu=function(e,t,n){return e+=e<0?1:e>1?-1:0,(e*6<1?t+(n-t)*e*6:e<.5?n:e*3<2?t+(n-t)*(2/3-e)*6:t)*Ct+.5|0},ax=function(e,t,n){var s=e?xs(e)?[e>>16,e>>8&Ct,e&Ct]:0:ia.black,r,o,a,l,c,u,f,d,h,x;if(!s){if(e.substr(-1)===","&&(e=e.substr(0,e.length-1)),ia[e])s=ia[e];else if(e.charAt(0)==="#"){if(e.length<6&&(r=e.charAt(1),o=e.charAt(2),a=e.charAt(3),e="#"+r+r+o+o+a+a+(e.length===5?e.charAt(4)+e.charAt(4):"")),e.length===9)return s=parseInt(e.substr(1,6),16),[s>>16,s>>8&Ct,s&Ct,parseInt(e.substr(7),16)/255];e=parseInt(e.substr(1),16),s=[e>>16,e>>8&Ct,e&Ct]}else if(e.substr(0,3)==="hsl"){if(s=x=e.match(Sm),!t)l=+s[0]%360/360,c=+s[1]/100,u=+s[2]/100,o=u<=.5?u*(c+1):u+c-u*c,r=u*2-o,s.length>3&&(s[3]*=1),s[0]=Lu(l+1/3,r,o),s[1]=Lu(l,r,o),s[2]=Lu(l-1/3,r,o);else if(~e.indexOf("="))return s=e.match(Ng),n&&s.length<4&&(s[3]=1),s}else s=e.match(Sm)||ia.transparent;s=s.map(Number)}return t&&!x&&(r=s[0]/Ct,o=s[1]/Ct,a=s[2]/Ct,f=Math.max(r,o,a),d=Math.min(r,o,a),u=(f+d)/2,f===d?l=c=0:(h=f-d,c=u>.5?h/(2-f-d):h/(f+d),l=f===r?(o-a)/h+(o<a?6:0):f===o?(a-r)/h+2:(r-o)/h+4,l*=60),s[0]=~~(l+.5),s[1]=~~(c*100+.5),s[2]=~~(u*100+.5)),n&&s.length<4&&(s[3]=1),s},lx=function(e){var t=[],n=[],s=-1;return e.split(Gs).forEach(function(r){var o=r.match(so)||[];t.push.apply(t,o),n.push(s+=o.length+1)}),t.c=n,t},Em=function(e,t,n){var s="",r=(e+s).match(Gs),o=t?"hsla(":"rgba(",a=0,l,c,u,f;if(!r)return e;if(r=r.map(function(d){return(d=ax(d,t,1))&&o+(t?d[0]+","+d[1]+"%,"+d[2]+"%,"+d[3]:d.join(","))+")"}),n&&(u=lx(e),l=n.c,l.join(s)!==u.c.join(s)))for(c=e.replace(Gs,"1").split(so),f=c.length-1;a<f;a++)s+=c[a]+(~l.indexOf(a)?r.shift()||o+"0,0,0,0)":(u.length?u:r.length?r:n).shift());if(!c)for(c=e.split(Gs),f=c.length-1;a<f;a++)s+=c[a]+r[a];return s+c[f]},Gs=(function(){var i="(?:\\b(?:(?:rgb|rgba|hsl|hsla)\\(.+?\\))|\\B#(?:[0-9a-f]{3,4}){1,2}\\b",e;for(e in ia)i+="|"+e+"\\b";return new RegExp(i+")","gi")})(),$1=/hsl[a]?\(/,cx=function(e){var t=e.join(" "),n;if(Gs.lastIndex=0,Gs.test(t))return n=$1.test(t),e[1]=Em(e[1],n),e[0]=Em(e[0],n,lx(e[1])),!0},Na,jn=(function(){var i=Date.now,e=500,t=33,n=i(),s=n,r=1e3/240,o=r,a=[],l,c,u,f,d,h,x=function p(g){var m=i()-s,_=g===!0,v,A,S,b;if((m>e||m<0)&&(n+=m-t),s+=m,S=s-n,v=S-o,(v>0||_)&&(b=++f.frame,d=S-f.time*1e3,f.time=S=S/1e3,o+=v+(v>=r?4:r-v),A=1),_||(l=c(p)),A)for(h=0;h<a.length;h++)a[h](S,d,b,g)};return f={time:0,frame:0,tick:function(){x(!0)},deltaRatio:function(g){return d/(1e3/(g||60))},wake:function(){kg&&(!Gf&&Jd()&&(Ri=Gf=window,eh=Ri.document||{},ii.gsap=Xn,(Ri.gsapVersions||(Ri.gsapVersions=[])).push(Xn.version),Hg(oc||Ri.GreenSockGlobals||!Ri.gsap&&Ri||{}),rx.forEach(ox)),u=typeof requestAnimationFrame<"u"&&requestAnimationFrame,l&&f.sleep(),c=u||function(g){return setTimeout(g,o-f.time*1e3+1|0)},Na=1,x(2))},sleep:function(){(u?cancelAnimationFrame:clearTimeout)(l),Na=0,c=Ua},lagSmoothing:function(g,m){e=g||1/0,t=Math.min(m||33,e)},fps:function(g){r=1e3/(g||240),o=f.time*1e3+r},add:function(g,m,_){var v=m?function(A,S,b,M){g(A,S,b,M),f.remove(v)}:g;return f.remove(g),a[_?"unshift":"push"](v),Bo(),v},remove:function(g,m){~(m=a.indexOf(g))&&a.splice(m,1)&&h>=m&&h--},_listeners:a},f})(),Bo=function(){return!Na&&jn.wake()},at={},Z1=/^[\d.\-M][\d.\-,\s]/,J1=/["']/g,ew=function(e){for(var t={},n=e.substr(1,e.length-3).split(":"),s=n[0],r=1,o=n.length,a,l,c;r<o;r++)l=n[r],a=r!==o-1?l.lastIndexOf(","):l.length,c=l.substr(0,a),t[s]=isNaN(c)?c.replace(J1,"").trim():+c,s=l.substr(a+1).trim();return t},tw=function(e){var t=e.indexOf("(")+1,n=e.indexOf(")"),s=e.indexOf("(",t);return e.substring(t,~s&&s<n?e.indexOf(")",n+1):n)},nw=function(e){var t=(e+"").split("("),n=at[t[0]];return n&&t.length>1&&n.config?n.config.apply(null,~e.indexOf("{")?[ew(t[1])]:tw(e).split(",").map(qg)):at._CE&&Z1.test(e)?at._CE("",e):n},ux=function(e){return function(t){return 1-e(1-t)}},fx=function i(e,t){for(var n=e._first,s;n;)n instanceof Dn?i(n,t):n.vars.yoyoEase&&(!n._yoyo||!n._repeat)&&n._yoyo!==t&&(n.timeline?i(n.timeline,t):(s=n._ease,n._ease=n._yEase,n._yEase=s,n._yoyo=t)),n=n._next},Cr=function(e,t){return e&&(Vt(e)?e:at[e]||nw(e))||t},Ir=function(e,t,n,s){n===void 0&&(n=function(l){return 1-t(1-l)}),s===void 0&&(s=function(l){return l<.5?t(l*2)/2:1-t((1-l)*2)/2});var r={easeIn:t,easeOut:n,easeInOut:s},o;return Vn(e,function(a){at[a]=ii[a]=r,at[o=a.toLowerCase()]=n;for(var l in r)at[o+(l==="easeIn"?".in":l==="easeOut"?".out":".inOut")]=at[a+"."+l]=r[l]}),r},dx=function(e){return function(t){return t<.5?(1-e(1-t*2))/2:.5+e((t-.5)*2)/2}},Bu=function i(e,t,n){var s=t>=1?t:1,r=(n||(e?.3:.45))/(t<1?t:1),o=r/Vf*(Math.asin(1/s)||0),a=function(u){return u===1?1:s*Math.pow(2,-10*u)*E1((u-o)*r)+1},l=e==="out"?a:e==="in"?function(c){return 1-a(1-c)}:dx(a);return r=Vf/r,l.config=function(c,u){return i(e,c,u)},l},Uu=function i(e,t){t===void 0&&(t=1.70158);var n=function(o){return o?--o*o*((t+1)*o+t)+1:0},s=e==="out"?n:e==="in"?function(r){return 1-n(1-r)}:dx(n);return s.config=function(r){return i(e,r)},s};Vn("Linear,Quad,Cubic,Quart,Quint,Strong",function(i,e){var t=e<5?e+1:e;Ir(i+",Power"+(t-1),e?function(n){return Math.pow(n,t)}:function(n){return n},function(n){return 1-Math.pow(1-n,t)},function(n){return n<.5?Math.pow(n*2,t)/2:1-Math.pow((1-n)*2,t)/2})});at.Linear.easeNone=at.none=at.Linear.easeIn;Ir("Elastic",Bu("in"),Bu("out"),Bu());(function(i,e){var t=1/e,n=2*t,s=2.5*t,r=function(a){return a<t?i*a*a:a<n?i*Math.pow(a-1.5/e,2)+.75:a<s?i*(a-=2.25/e)*a+.9375:i*Math.pow(a-2.625/e,2)+.984375};Ir("Bounce",function(o){return 1-r(1-o)},r)})(7.5625,2.75);Ir("Expo",function(i){return Math.pow(2,10*(i-1))*i+i*i*i*i*i*i*(1-i)});Ir("Circ",function(i){return-(Ug(1-i*i)-1)});Ir("Sine",function(i){return i===1?1:-T1(i*M1)+1});Ir("Back",Uu("in"),Uu("out"),Uu());at.SteppedEase=at.steps=ii.SteppedEase={config:function(e,t){e===void 0&&(e=1);var n=1/e,s=e+(t?0:1),r=t?1:0,o=1-Tt;return function(a){return((s*Qa(0,o,a)|0)+r)*n}}};Do.ease=at["quad.out"];Vn("onComplete,onUpdate,onStart,onRepeat,onReverseComplete,onInterrupt",function(i){return ih+=i+","+i+"Params,"});var hx=function(e,t){this.id=C1++,e._gsap=this,this.target=e,this.harness=t,this.get=t?t.get:Wg,this.set=t?t.getSetter:ch},za=(function(){function i(t){this.vars=t,this._delay=+t.delay||0,(this._repeat=t.repeat===1/0?-2:t.repeat||0)&&(this._rDelay=t.repeatDelay||0,this._yoyo=!!t.yoyo||!!t.yoyoEase),this._ts=1,Lo(this,+t.duration,1,1),this.data=t.data,Ut&&(this._ctx=Ut,Ut.data.push(this)),Na||jn.wake()}var e=i.prototype;return e.delay=function(n){return n||n===0?(this.parent&&this.parent.smoothChildTiming&&this.startTime(this._start+n-this._delay),this._delay=n,this):this._delay},e.duration=function(n){return arguments.length?this.totalDuration(this._repeat>0?n+(n+this._rDelay)*this._repeat:n):this.totalDuration()&&this._dur},e.totalDuration=function(n){return arguments.length?(this._dirty=0,Lo(this,this._repeat<0?n:(n-this._repeat*this._rDelay)/(this._repeat+1))):this._tDur},e.totalTime=function(n,s){if(Bo(),!arguments.length)return this._tTime;var r=this._dp;if(r&&r.smoothChildTiming&&this._ts){for(Dc(this,n),!r._dp||r.parent||Kg(r,this);r&&r.parent;)r.parent._time!==r._start+(r._ts>=0?r._tTime/r._ts:(r.totalDuration()-r._tTime)/-r._ts)&&r.totalTime(r._tTime,!0),r=r.parent;!this.parent&&this._dp.autoRemoveChildren&&(this._ts>0&&n<this._tDur||this._ts<0&&n>0||!this._tDur&&!n)&&Fi(this._dp,this,this._start-this._delay)}return(this._tTime!==n||!this._dur&&!s||this._initted&&Math.abs(this._zTime)===Tt||!this._initted&&this._dur&&n||!n&&!this._initted&&(this.add||this._ptLookup))&&(this._ts||(this._pTime=n),Xg(this,n,s)),this},e.time=function(n,s){return arguments.length?this.totalTime(Math.min(this.totalDuration(),n+Mm(this))%(this._dur+this._rDelay)||(n?this._dur:0),s):this._time},e.totalProgress=function(n,s){return arguments.length?this.totalTime(this.totalDuration()*n,s):this.totalDuration()?Math.min(1,this._tTime/this._tDur):this.rawTime()>=0&&this._initted?1:0},e.progress=function(n,s){return arguments.length?this.totalTime(this.duration()*(this._yoyo&&!(this.iteration()&1)?1-n:n)+Mm(this),s):this.duration()?Math.min(1,this._time/this._dur):this.rawTime()>0?1:0},e.iteration=function(n,s){var r=this.duration()+this._rDelay;return arguments.length?this.totalTime(this._time+(n-1)*r,s):this._repeat?Fo(this._tTime,r)+1:1},e.timeScale=function(n,s){if(!arguments.length)return this._rts===-Tt?0:this._rts;if(this._rts===n)return this;var r=this.parent&&this._ts?cc(this.parent._time,this):this._tTime;return this._rts=+n||0,this._ts=this._ps||n===-Tt?0:this._rts,this.totalTime(Qa(-Math.abs(this._delay),this.totalDuration(),r),s!==!1),Ic(this),O1(this)},e.paused=function(n){return arguments.length?(this._ps!==n&&(this._ps=n,n?(this._pTime=this._tTime||Math.max(-this._delay,this.rawTime()),this._ts=this._act=0):(Bo(),this._ts=this._rts,this.totalTime(this.parent&&!this.parent.smoothChildTiming?this.rawTime():this._tTime||this._pTime,this.progress()===1&&Math.abs(this._zTime)!==Tt&&(this._tTime-=Tt)))),this):this._ps},e.startTime=function(n){if(arguments.length){this._start=zt(n);var s=this.parent||this._dp;return s&&(s._sort||!this.parent)&&Fi(s,this,this._start-this._delay),this}return this._start},e.endTime=function(n){return this._start+(Hn(n)?this.totalDuration():this.duration())/Math.abs(this._ts||1)},e.rawTime=function(n){var s=this.parent||this._dp;return s?n&&(!this._ts||this._repeat&&this._time&&this.totalProgress()<1)?this._tTime%(this._dur+this._rDelay):this._ts?cc(s.rawTime(n),this):this._tTime:this._tTime},e.revert=function(n){n===void 0&&(n=F1);var s=un;return un=n,rh(this)&&(this.timeline&&this.timeline.revert(n),this.totalTime(-.01,n.suppressEvents)),this.data!=="nested"&&n.kill!==!1&&this.kill(),un=s,this},e.globalTime=function(n){for(var s=this,r=arguments.length?n:s.rawTime();s;)r=s._start+r/(Math.abs(s._ts)||1),s=s._dp;return!this.parent&&this._sat?this._sat.globalTime(n):r},e.repeat=function(n){return arguments.length?(this._repeat=n===1/0?-2:n,Cm(this)):this._repeat===-2?1/0:this._repeat},e.repeatDelay=function(n){if(arguments.length){var s=this._time;return this._rDelay=n,Cm(this),s?this.time(s):this}return this._rDelay},e.yoyo=function(n){return arguments.length?(this._yoyo=n,this):this._yoyo},e.seek=function(n,s){return this.totalTime(ai(this,n),Hn(s))},e.restart=function(n,s){return this.play().totalTime(n?-this._delay:0,Hn(s)),this._dur||(this._zTime=-Tt),this},e.play=function(n,s){return n!=null&&this.seek(n,s),this.reversed(!1).paused(!1)},e.reverse=function(n,s){return n!=null&&this.seek(n||this.totalDuration(),s),this.reversed(!0).paused(!1)},e.pause=function(n,s){return n!=null&&this.seek(n,s),this.paused(!0)},e.resume=function(){return this.paused(!1)},e.reversed=function(n){return arguments.length?(!!n!==this.reversed()&&this.timeScale(-this._rts||(n?-Tt:0)),this):this._rts<0},e.invalidate=function(){return this._initted=this._act=0,this._zTime=-Tt,this},e.isActive=function(){var n=this.parent||this._dp,s=this._start,r;return!!(!n||this._ts&&this._initted&&n.isActive()&&(r=n.rawTime(!0))>=s&&r<this.endTime(!0)-Tt)},e.eventCallback=function(n,s,r){var o=this.vars;return arguments.length>1?(s?(o[n]=s,r&&(o[n+"Params"]=r),n==="onUpdate"&&(this._onUpdate=s)):delete o[n],this):o[n]},e.then=function(n){var s=this,r=s._prom;return new Promise(function(o){var a=Vt(n)?n:Yg,l=function(){var u=s.then;s.then=null,r&&r(),Vt(a)&&(a=a(s))&&(a.then||a===s)&&(s.then=u),o(a),s.then=u};s._initted&&s.totalProgress()===1&&s._ts>=0||!s._tTime&&s._ts<0?l():s._prom=l})},e.kill=function(){na(this)},i})();si(za.prototype,{_time:0,_start:0,_end:0,_tTime:0,_tDur:0,_dirty:0,_repeat:0,_yoyo:!1,parent:null,_initted:!1,_rDelay:0,_ts:1,_dp:0,ratio:0,_zTime:-Tt,_prom:0,_ps:!1,_rts:1});var Dn=(function(i){Bg(e,i);function e(n,s){var r;return n===void 0&&(n={}),r=i.call(this,n)||this,r.labels={},r.smoothChildTiming=!!n.smoothChildTiming,r.autoRemoveChildren=!!n.autoRemoveChildren,r._sort=Hn(n.sortChildren),kt&&Fi(n.parent||kt,ss(r),s),n.reversed&&r.reverse(),n.paused&&r.paused(!0),n.scrollTrigger&&jg(ss(r),n.scrollTrigger),r}var t=e.prototype;return t.to=function(s,r,o){return va(0,arguments,this),this},t.from=function(s,r,o){return va(1,arguments,this),this},t.fromTo=function(s,r,o,a){return va(2,arguments,this),this},t.set=function(s,r,o){return r.duration=0,r.parent=this,_a(r).repeatDelay||(r.repeat=0),r.immediateRender=!!r.immediateRender,new Kt(s,r,ai(this,o),1),this},t.call=function(s,r,o){return Fi(this,Kt.delayedCall(0,s,r),o)},t.staggerTo=function(s,r,o,a,l,c,u){return o.duration=r,o.stagger=o.stagger||a,o.onComplete=c,o.onCompleteParams=u,o.parent=this,new Kt(s,o,ai(this,l)),this},t.staggerFrom=function(s,r,o,a,l,c,u){return o.runBackwards=1,_a(o).immediateRender=Hn(o.immediateRender),this.staggerTo(s,r,o,a,l,c,u)},t.staggerFromTo=function(s,r,o,a,l,c,u,f){return a.startAt=o,_a(a).immediateRender=Hn(a.immediateRender),this.staggerTo(s,r,a,l,c,u,f)},t.render=function(s,r,o){var a=this._time,l=this._dirty?this.totalDuration():this._tDur,c=this._dur,u=s<=0?0:zt(s),f=this._zTime<0!=s<0&&(this._initted||!c),d,h,x,p,g,m,_,v,A,S,b,M;if(this!==kt&&u>l&&s>=0&&(u=l),u!==this._tTime||o||f){if(a!==this._time&&c&&(u+=this._time-a,s+=this._time-a),d=u,A=this._start,v=this._ts,m=!v,f&&(c||(a=this._zTime),(s||!r)&&(this._zTime=s)),this._repeat){if(b=this._yoyo,g=c+this._rDelay,this._repeat<-1&&s<0)return this.totalTime(g*100+s,r,o);if(d=zt(u%g),u===l?(p=this._repeat,d=c):(S=zt(u/g),p=~~S,p&&p===S&&(d=c,p--),d>c&&(d=c)),S=Fo(this._tTime,g),!a&&this._tTime&&S!==p&&this._tTime-S*g-this._dur<=0&&(S=p),b&&p&1&&(d=c-d,M=1),p!==S&&!this._lock){var R=b&&S&1,y=R===(b&&p&1);if(p<S&&(R=!R),a=R?0:u%c?c:u,this._lock=1,this.render(a||(M?0:zt(p*g)),r,!c)._lock=0,this._tTime=u,!r&&this.parent&&Jn(this,"onRepeat"),this.vars.repeatRefresh&&!M&&(this.invalidate()._lock=1,S=p),a&&a!==this._time||m!==!this._ts||this.vars.onRepeat&&!this.parent&&!this._act)return this;if(c=this._dur,l=this._tDur,y&&(this._lock=2,a=R?c:-1e-4,this.render(a,!0),this.vars.repeatRefresh&&!M&&this.invalidate()),this._lock=0,!this._ts&&!m)return this;fx(this,M)}}if(this._hasPause&&!this._forcing&&this._lock<2&&(_=H1(this,zt(a),zt(d)),_&&(u-=d-(d=_._start))),this._tTime=u,this._time=d,this._act=!v,this._initted||(this._onUpdate=this.vars.onUpdate,this._initted=1,this._zTime=s,a=0),!a&&u&&c&&!r&&!S&&(Jn(this,"onStart"),this._tTime!==u))return this;if(d>=a&&s>=0)for(h=this._first;h;){if(x=h._next,(h._act||d>=h._start)&&h._ts&&_!==h){if(h.parent!==this)return this.render(s,r,o);if(h.render(h._ts>0?(d-h._start)*h._ts:(h._dirty?h.totalDuration():h._tDur)+(d-h._start)*h._ts,r,o),d!==this._time||!this._ts&&!m){_=0,x&&(u+=this._zTime=-Tt);break}}h=x}else{h=this._last;for(var C=s<0?s:d;h;){if(x=h._prev,(h._act||C<=h._end)&&h._ts&&_!==h){if(h.parent!==this)return this.render(s,r,o);if(h.render(h._ts>0?(C-h._start)*h._ts:(h._dirty?h.totalDuration():h._tDur)+(C-h._start)*h._ts,r,o||un&&rh(h)),d!==this._time||!this._ts&&!m){_=0,x&&(u+=this._zTime=C?-Tt:Tt);break}}h=x}}if(_&&!r&&(this.pause(),_.render(d>=a?0:-Tt)._zTime=d>=a?1:-1,this._ts))return this._start=A,Ic(this),this.render(s,r,o);this._onUpdate&&!r&&Jn(this,"onUpdate",!0),(u===l&&this._tTime>=this.totalDuration()||!u&&a)&&(A===this._start||Math.abs(v)!==Math.abs(this._ts))&&(this._lock||((s||!c)&&(u===l&&this._ts>0||!u&&this._ts<0)&&Ys(this,1),!r&&!(s<0&&!a)&&(u||a||!l)&&(Jn(this,u===l&&s>=0?"onComplete":"onReverseComplete",!0),this._prom&&!(u<l&&this.timeScale()>0)&&this._prom())))}return this},t.add=function(s,r){var o=this;if(xs(r)||(r=ai(this,r,s)),!(s instanceof za)){if(Sn(s))return s.forEach(function(a){return o.add(a,r)}),this;if(sn(s))return this.addLabel(s,r);if(Vt(s))s=Kt.delayedCall(0,s);else return this}return this!==s?Fi(this,s,r):this},t.getChildren=function(s,r,o,a){s===void 0&&(s=!0),r===void 0&&(r=!0),o===void 0&&(o=!0),a===void 0&&(a=-pi);for(var l=[],c=this._first;c;)c._start>=a&&(c instanceof Kt?r&&l.push(c):(o&&l.push(c),s&&l.push.apply(l,c.getChildren(!0,r,o)))),c=c._next;return l},t.getById=function(s){for(var r=this.getChildren(1,1,1),o=r.length;o--;)if(r[o].vars.id===s)return r[o]},t.remove=function(s){return sn(s)?this.removeLabel(s):Vt(s)?this.killTweensOf(s):(s.parent===this&&Rc(this,s),s===this._recent&&(this._recent=this._last),Mr(this))},t.totalTime=function(s,r){return arguments.length?(this._forcing=1,!this._dp&&this._ts&&(this._start=zt(jn.time-(this._ts>0?s/this._ts:(this.totalDuration()-s)/-this._ts))),i.prototype.totalTime.call(this,s,r),this._forcing=0,this):this._tTime},t.addLabel=function(s,r){return this.labels[s]=ai(this,r),this},t.removeLabel=function(s){return delete this.labels[s],this},t.addPause=function(s,r,o){var a=Kt.delayedCall(0,r||Ua,o);return a.data="isPause",this._hasPause=1,Fi(this,a,ai(this,s))},t.removePause=function(s){var r=this._first;for(s=ai(this,s);r;)r._start===s&&r.data==="isPause"&&Ys(r),r=r._next},t.killTweensOf=function(s,r,o){for(var a=this.getTweensOf(s,o),l=a.length;l--;)Us!==a[l]&&a[l].kill(s,r);return this},t.getTweensOf=function(s,r){for(var o=[],a=mi(s),l=this._first,c=xs(r),u;l;)l instanceof Kt?L1(l._targets,a)&&(c?(!Us||l._initted&&l._ts)&&l.globalTime(0)<=r&&l.globalTime(l.totalDuration())>r:!r||l.isActive())&&o.push(l):(u=l.getTweensOf(a,r)).length&&o.push.apply(o,u),l=l._next;return o},t.tweenTo=function(s,r){r=r||{};var o=this,a=ai(o,s),l=r,c=l.startAt,u=l.onStart,f=l.onStartParams,d=l.immediateRender,h,x=Kt.to(o,si({ease:r.ease||"none",lazy:!1,immediateRender:!1,time:a,overwrite:"auto",duration:r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale())||Tt,onStart:function(){if(o.pause(),!h){var g=r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale());x._dur!==g&&Lo(x,g,0,1).render(x._time,!0,!0),h=1}u&&u.apply(x,f||[])}},r));return d?x.render(0):x},t.tweenFromTo=function(s,r,o){return this.tweenTo(r,si({startAt:{time:ai(this,s)}},o))},t.recent=function(){return this._recent},t.nextLabel=function(s){return s===void 0&&(s=this._time),Tm(this,ai(this,s))},t.previousLabel=function(s){return s===void 0&&(s=this._time),Tm(this,ai(this,s),1)},t.currentLabel=function(s){return arguments.length?this.seek(s,!0):this.previousLabel(this._time+Tt)},t.shiftChildren=function(s,r,o){o===void 0&&(o=0);var a=this._first,l=this.labels,c;for(s=zt(s);a;)a._start>=o&&(a._start+=s,a._end+=s),a=a._next;if(r)for(c in l)l[c]>=o&&(l[c]+=s);return Mr(this)},t.invalidate=function(s){var r=this._first;for(this._lock=0;r;)r.invalidate(s),r=r._next;return i.prototype.invalidate.call(this,s)},t.clear=function(s){s===void 0&&(s=!0);for(var r=this._first,o;r;)o=r._next,this.remove(r),r=o;return this._dp&&(this._time=this._tTime=this._pTime=0),s&&(this.labels={}),Mr(this)},t.totalDuration=function(s){var r=0,o=this,a=o._last,l=pi,c,u,f;if(arguments.length)return o.timeScale((o._repeat<0?o.duration():o.totalDuration())/(o.reversed()?-s:s));if(o._dirty){for(f=o.parent;a;)c=a._prev,a._dirty&&a.totalDuration(),u=a._start,u>l&&o._sort&&a._ts&&!o._lock?(o._lock=1,Fi(o,a,u-a._delay,1)._lock=0):l=u,u<0&&a._ts&&(r-=u,(!f&&!o._dp||f&&f.smoothChildTiming)&&(o._start+=zt(u/o._ts),o._time-=u,o._tTime-=u),o.shiftChildren(-u,!1,-1/0),l=0),a._end>r&&a._ts&&(r=a._end),a=c;Lo(o,o===kt&&o._time>r?o._time:r,1,1),o._dirty=0}return o._tDur},e.updateRoot=function(s){if(kt._ts&&(Xg(kt,cc(s,kt)),Gg=jn.frame),jn.frame>=ym){ym+=ni.autoSleep||120;var r=kt._first;if((!r||!r._ts)&&ni.autoSleep&&jn._listeners.length<2){for(;r&&!r._ts;)r=r._next;r||jn.sleep()}}},e})(za);si(Dn.prototype,{_lock:0,_hasPause:0,_forcing:0});var iw=function(e,t,n,s,r,o,a){var l=new Gn(this._pt,e,t,0,1,vx,null,r),c=0,u=0,f,d,h,x,p,g,m,_;for(l.b=n,l.e=s,n+="",s+="",(m=~s.indexOf("random("))&&(s=Oa(s)),o&&(_=[n,s],o(_,e,t),n=_[0],s=_[1]),d=n.match(Pu)||[];f=Pu.exec(s);)x=f[0],p=s.substring(c,f.index),h?h=(h+1)%5:p.substr(-5)==="rgba("&&(h=1),x!==d[u++]&&(g=parseFloat(d[u-1])||0,l._pt={_next:l._pt,p:p||u===1?p:",",s:g,c:x.charAt(1)==="="?_o(g,x)-g:parseFloat(x)-g,m:h&&h<4?Math.round:0},c=Pu.lastIndex);return l.c=c<s.length?s.substring(c,s.length):"",l.fp=a,(zg.test(s)||m)&&(l.e=0),this._pt=l,l},oh=function(e,t,n,s,r,o,a,l,c,u){Vt(s)&&(s=s(r||0,e,o));var f=e[t],d=n!=="get"?n:Vt(f)?c?e[t.indexOf("set")||!Vt(e["get"+t.substr(3)])?t:"get"+t.substr(3)](c):e[t]():f,h=Vt(f)?c?lw:xx:lh,x;if(sn(s)&&(~s.indexOf("random(")&&(s=Oa(s)),s.charAt(1)==="="&&(x=_o(d,s)+(_n(d)||0),(x||x===0)&&(s=x))),!u||d!==s||jf)return!isNaN(d*s)&&s!==""?(x=new Gn(this._pt,e,t,+d||0,s-(d||0),typeof f=="boolean"?uw:_x,0,h),c&&(x.fp=c),a&&x.modifier(a,this,e),this._pt=x):(!f&&!(t in e)&&th(t,s),iw.call(this,e,t,d,s,h,l||ni.stringFilter,c))},sw=function(e,t,n,s,r){if(Vt(e)&&(e=Aa(e,r,t,n,s)),!Wi(e)||e.style&&e.nodeType||Sn(e)||Og(e))return sn(e)?Aa(e,r,t,n,s):e;var o={},a;for(a in e)o[a]=Aa(e[a],r,t,n,s);return o},px=function(e,t,n,s,r,o){var a,l,c,u;if(Kn[e]&&(a=new Kn[e]).init(r,a.rawVars?t[e]:sw(t[e],s,r,o,n),n,s,o)!==!1&&(n._pt=l=new Gn(n._pt,r,e,0,1,a.render,a,0,a.priority),n!==ro))for(c=n._ptLookup[n._targets.indexOf(r)],u=a._props.length;u--;)c[a._props[u]]=l;return a},Us,jf,ah=function i(e,t,n){var s=e.vars,r=s.ease,o=s.startAt,a=s.immediateRender,l=s.lazy,c=s.onUpdate,u=s.runBackwards,f=s.yoyoEase,d=s.keyframes,h=s.autoRevert,x=e._dur,p=e._startAt,g=e._targets,m=e.parent,_=m&&m.data==="nested"?m.vars.targets:g,v=e._overwrite==="auto"&&!$d,A=e.timeline,S,b,M,R,y,C,F,I,O,z,H,V,q;if(A&&(!d||!r)&&(r="none"),e._ease=Cr(r,Do.ease),e._yEase=f?ux(Cr(f===!0?r:f,Do.ease)):0,f&&e._yoyo&&!e._repeat&&(f=e._yEase,e._yEase=e._ease,e._ease=f),e._from=!A&&!!s.runBackwards,!A||d&&!s.stagger){if(I=g[0]?br(g[0]).harness:0,V=I&&s[I.prop],S=lc(s,nh),p&&(p._zTime<0&&p.progress(1),t<0&&u&&a&&!h?p.render(-1,!0):p.revert(u&&x?Gl:P1),p._lazy=0),o){if(Ys(e._startAt=Kt.set(g,si({data:"isStart",overwrite:!1,parent:m,immediateRender:!0,lazy:!p&&Hn(l),startAt:null,delay:0,onUpdate:c&&function(){return Jn(e,"onUpdate")},stagger:0},o))),e._startAt._dp=0,e._startAt._sat=e,t<0&&(un||!a&&!h)&&e._startAt.revert(Gl),a&&x&&t<=0&&n<=0){t&&(e._zTime=t);return}}else if(u&&x&&!p){if(t&&(a=!1),M=si({overwrite:!1,data:"isFromStart",lazy:a&&!p&&Hn(l),immediateRender:a,stagger:0,parent:m},S),V&&(M[I.prop]=V),Ys(e._startAt=Kt.set(g,M)),e._startAt._dp=0,e._startAt._sat=e,t<0&&(un?e._startAt.revert(Gl):e._startAt.render(-1,!0)),e._zTime=t,!a)i(e._startAt,Tt,Tt);else if(!t)return}for(e._pt=e._ptCache=0,l=x&&Hn(l)||l&&!x,b=0;b<g.length;b++){if(y=g[b],F=y._gsap||sh(g)[b]._gsap,e._ptLookup[b]=z={},Wf[F.id]&&Vs.length&&ac(),H=_===g?b:_.indexOf(y),I&&(O=new I).init(y,V||S,e,H,_)!==!1&&(e._pt=R=new Gn(e._pt,y,O.name,0,1,O.render,O,0,O.priority),O._props.forEach(function(G){z[G]=R}),O.priority&&(C=1)),!I||V)for(M in S)Kn[M]&&(O=px(M,S,e,H,y,_))?O.priority&&(C=1):z[M]=R=oh.call(e,y,M,"get",S[M],H,_,0,s.stringFilter);e._op&&e._op[b]&&e.kill(y,e._op[b]),v&&e._pt&&(Us=e,kt.killTweensOf(y,z,e.globalTime(t)),q=!e.parent,Us=0),e._pt&&l&&(Wf[F.id]=1)}C&&Ax(e),e._onInit&&e._onInit(e)}e._onUpdate=c,e._initted=(!e._op||e._pt)&&!q,d&&t<=0&&A.render(pi,!0,!0)},rw=function(e,t,n,s,r,o,a,l){var c=(e._pt&&e._ptCache||(e._ptCache={}))[t],u,f,d,h;if(!c)for(c=e._ptCache[t]=[],d=e._ptLookup,h=e._targets.length;h--;){if(u=d[h][t],u&&u.d&&u.d._pt)for(u=u.d._pt;u&&u.p!==t&&u.fp!==t;)u=u._next;if(!u)return jf=1,e.vars[t]="+=0",ah(e,a),jf=0,l?Ba(t+" not eligible for reset"):1;c.push(u)}for(h=c.length;h--;)f=c[h],u=f._pt||f,u.s=(s||s===0)&&!r?s:u.s+(s||0)+o*u.c,u.c=n-u.s,f.e&&(f.e=Gt(n)+_n(f.e)),f.b&&(f.b=u.s+_n(f.b))},ow=function(e,t){var n=e[0]?br(e[0]).harness:0,s=n&&n.aliases,r,o,a,l;if(!s)return t;r=Po({},t);for(o in s)if(o in r)for(l=s[o].split(","),a=l.length;a--;)r[l[a]]=r[o];return r},aw=function(e,t,n,s){var r=t.ease||s||"power1.inOut",o,a;if(Sn(t))a=n[e]||(n[e]=[]),t.forEach(function(l,c){return a.push({t:c/(t.length-1)*100,v:l,e:r})});else for(o in t)a=n[o]||(n[o]=[]),o==="ease"||a.push({t:parseFloat(e),v:t[o],e:r})},Aa=function(e,t,n,s,r){return Vt(e)?e.call(t,n,s,r):sn(e)&&~e.indexOf("random(")?Oa(e):e},mx=ih+"repeat,repeatDelay,yoyo,repeatRefresh,yoyoEase,autoRevert",gx={};Vn(mx+",id,stagger,delay,duration,paused,scrollTrigger",function(i){return gx[i]=1});var Kt=(function(i){Bg(e,i);function e(n,s,r,o){var a;typeof s=="number"&&(r.duration=s,s=r,r=null),a=i.call(this,o?s:_a(s))||this;var l=a.vars,c=l.duration,u=l.delay,f=l.immediateRender,d=l.stagger,h=l.overwrite,x=l.keyframes,p=l.defaults,g=l.scrollTrigger,m=l.yoyoEase,_=s.parent||kt,v=(Sn(n)||Og(n)?xs(n[0]):"length"in s)?[n]:mi(n),A,S,b,M,R,y,C,F;if(a._targets=v.length?sh(v):Ba("GSAP target "+n+" not found. https://gsap.com",!ni.nullTargetWarn)||[],a._ptLookup=[],a._overwrite=h,x||d||Pl(c)||Pl(u)){if(s=a.vars,A=a.timeline=new Dn({data:"nested",defaults:p||{},targets:_&&_.data==="nested"?_.vars.targets:v}),A.kill(),A.parent=A._dp=ss(a),A._start=0,d||Pl(c)||Pl(u)){if(M=v.length,C=d&&ex(d),Wi(d))for(R in d)~mx.indexOf(R)&&(F||(F={}),F[R]=d[R]);for(S=0;S<M;S++)b=lc(s,gx),b.stagger=0,m&&(b.yoyoEase=m),F&&Po(b,F),y=v[S],b.duration=+Aa(c,ss(a),S,y,v),b.delay=(+Aa(u,ss(a),S,y,v)||0)-a._delay,!d&&M===1&&b.delay&&(a._delay=u=b.delay,a._start+=u,b.delay=0),A.to(y,b,C?C(S,y,v):0),A._ease=at.none;A.duration()?c=u=0:a.timeline=0}else if(x){_a(si(A.vars.defaults,{ease:"none"})),A._ease=Cr(x.ease||s.ease||"none");var I=0,O,z,H;if(Sn(x))x.forEach(function(V){return A.to(v,V,">")}),A.duration();else{b={};for(R in x)R==="ease"||R==="easeEach"||aw(R,x[R],b,x.easeEach);for(R in b)for(O=b[R].sort(function(V,q){return V.t-q.t}),I=0,S=0;S<O.length;S++)z=O[S],H={ease:z.e,duration:(z.t-(S?O[S-1].t:0))/100*c},H[R]=z.v,A.to(v,H,I),I+=H.duration;A.duration()<c&&A.to({},{duration:c-A.duration()})}}c||a.duration(c=A.duration())}else a.timeline=0;return h===!0&&!$d&&(Us=ss(a),kt.killTweensOf(v),Us=0),Fi(_,ss(a),r),s.reversed&&a.reverse(),s.paused&&a.paused(!0),(f||!c&&!x&&a._start===zt(_._time)&&Hn(f)&&N1(ss(a))&&_.data!=="nested")&&(a._tTime=-Tt,a.render(Math.max(0,-u)||0)),g&&jg(ss(a),g),a}var t=e.prototype;return t.render=function(s,r,o){var a=this._time,l=this._tDur,c=this._dur,u=s<0,f=s>l-Tt&&!u?l:s<Tt?0:s,d,h,x,p,g,m,_,v,A;if(!c)k1(this,s,r,o);else if(f!==this._tTime||!s||o||!this._initted&&this._tTime||this._startAt&&this._zTime<0!==u||this._lazy){if(d=f,v=this.timeline,this._repeat){if(p=c+this._rDelay,this._repeat<-1&&u)return this.totalTime(p*100+s,r,o);if(d=zt(f%p),f===l?(x=this._repeat,d=c):(g=zt(f/p),x=~~g,x&&x===g?(d=c,x--):d>c&&(d=c)),m=this._yoyo&&x&1,m&&(A=this._yEase,d=c-d),g=Fo(this._tTime,p),d===a&&!o&&this._initted&&x===g)return this._tTime=f,this;x!==g&&(v&&this._yEase&&fx(v,m),this.vars.repeatRefresh&&!m&&!this._lock&&d!==p&&this._initted&&(this._lock=o=1,this.render(zt(p*x),!0).invalidate()._lock=0))}if(!this._initted){if($g(this,u?s:d,o,r,f))return this._tTime=0,this;if(a!==this._time&&!(o&&this.vars.repeatRefresh&&x!==g))return this;if(c!==this._dur)return this.render(s,r,o)}if(this._tTime=f,this._time=d,!this._act&&this._ts&&(this._act=1,this._lazy=0),this.ratio=_=(A||this._ease)(d/c),this._from&&(this.ratio=_=1-_),!a&&f&&!r&&!g&&(Jn(this,"onStart"),this._tTime!==f))return this;for(h=this._pt;h;)h.r(_,h.d),h=h._next;v&&v.render(s<0?s:v._dur*v._ease(d/this._dur),r,o)||this._startAt&&(this._zTime=s),this._onUpdate&&!r&&(u&&Xf(this,s,r,o),Jn(this,"onUpdate")),this._repeat&&x!==g&&this.vars.onRepeat&&!r&&this.parent&&Jn(this,"onRepeat"),(f===this._tDur||!f)&&this._tTime===f&&(u&&!this._onUpdate&&Xf(this,s,!0,!0),(s||!c)&&(f===this._tDur&&this._ts>0||!f&&this._ts<0)&&Ys(this,1),!r&&!(u&&!a)&&(f||a||m)&&(Jn(this,f===l?"onComplete":"onReverseComplete",!0),this._prom&&!(f<l&&this.timeScale()>0)&&this._prom()))}return this},t.targets=function(){return this._targets},t.invalidate=function(s){return(!s||!this.vars.runBackwards)&&(this._startAt=0),this._pt=this._op=this._onUpdate=this._lazy=this.ratio=0,this._ptLookup=[],this.timeline&&this.timeline.invalidate(s),i.prototype.invalidate.call(this,s)},t.resetTo=function(s,r,o,a,l){Na||jn.wake(),this._ts||this.play();var c=Math.min(this._dur,(this._dp._time-this._start)*this._ts),u;return this._initted||ah(this,c),u=this._ease(c/this._dur),rw(this,s,r,o,a,u,c,l)?this.resetTo(s,r,o,a,1):(Dc(this,0),this.parent||Qg(this._dp,this,"_first","_last",this._dp._sort?"_start":0),this.render(0))},t.kill=function(s,r){if(r===void 0&&(r="all"),!s&&(!r||r==="all"))return this._lazy=this._pt=0,this.parent?na(this):this.scrollTrigger&&this.scrollTrigger.kill(!!un),this;if(this.timeline){var o=this.timeline.totalDuration();return this.timeline.killTweensOf(s,r,Us&&Us.vars.overwrite!==!0)._first||na(this),this.parent&&o!==this.timeline.totalDuration()&&Lo(this,this._dur*this.timeline._tDur/o,0,1),this}var a=this._targets,l=s?mi(s):a,c=this._ptLookup,u=this._pt,f,d,h,x,p,g,m;if((!r||r==="all")&&U1(a,l))return r==="all"&&(this._pt=0),na(this);for(f=this._op=this._op||[],r!=="all"&&(sn(r)&&(p={},Vn(r,function(_){return p[_]=1}),r=p),r=ow(a,r)),m=a.length;m--;)if(~l.indexOf(a[m])){d=c[m],r==="all"?(f[m]=r,x=d,h={}):(h=f[m]=f[m]||{},x=r);for(p in x)g=d&&d[p],g&&((!("kill"in g.d)||g.d.kill(p)===!0)&&Rc(this,g,"_pt"),delete d[p]),h!=="all"&&(h[p]=1)}return this._initted&&!this._pt&&u&&na(this),this},e.to=function(s,r){return new e(s,r,arguments[2])},e.from=function(s,r){return va(1,arguments)},e.delayedCall=function(s,r,o,a){return new e(r,0,{immediateRender:!1,lazy:!1,overwrite:!1,delay:s,onComplete:r,onReverseComplete:r,onCompleteParams:o,onReverseCompleteParams:o,callbackScope:a})},e.fromTo=function(s,r,o){return va(2,arguments)},e.set=function(s,r){return r.duration=0,r.repeatDelay||(r.repeat=0),new e(s,r)},e.killTweensOf=function(s,r,o){return kt.killTweensOf(s,r,o)},e})(za);si(Kt.prototype,{_targets:[],_lazy:0,_startAt:0,_op:0,_onInit:0});Vn("staggerTo,staggerFrom,staggerFromTo",function(i){Kt[i]=function(){var e=new Dn,t=Yf.call(arguments,0);return t.splice(i==="staggerFromTo"?5:4,0,0),e[i].apply(e,t)}});var lh=function(e,t,n){return e[t]=n},xx=function(e,t,n){return e[t](n)},lw=function(e,t,n,s){return e[t](s.fp,n)},cw=function(e,t,n){return e.setAttribute(t,n)},ch=function(e,t){return Vt(e[t])?xx:Zd(e[t])&&e.setAttribute?cw:lh},_x=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e6)/1e6,t)},uw=function(e,t){return t.set(t.t,t.p,!!(t.s+t.c*e),t)},vx=function(e,t){var n=t._pt,s="";if(!e&&t.b)s=t.b;else if(e===1&&t.e)s=t.e;else{for(;n;)s=n.p+(n.m?n.m(n.s+n.c*e):Math.round((n.s+n.c*e)*1e4)/1e4)+s,n=n._next;s+=t.c}t.set(t.t,t.p,s,t)},uh=function(e,t){for(var n=t._pt;n;)n.r(e,n.d),n=n._next},fw=function(e,t,n,s){for(var r=this._pt,o;r;)o=r._next,r.p===s&&r.modifier(e,t,n),r=o},dw=function(e){for(var t=this._pt,n,s;t;)s=t._next,t.p===e&&!t.op||t.op===e?Rc(this,t,"_pt"):t.dep||(n=1),t=s;return!n},hw=function(e,t,n,s){s.mSet(e,t,s.m.call(s.tween,n,s.mt),s)},Ax=function(e){for(var t=e._pt,n,s,r,o;t;){for(n=t._next,s=r;s&&s.pr>t.pr;)s=s._next;(t._prev=s?s._prev:o)?t._prev._next=t:r=t,(t._next=s)?s._prev=t:o=t,t=n}e._pt=r},Gn=(function(){function i(t,n,s,r,o,a,l,c,u){this.t=n,this.s=r,this.c=o,this.p=s,this.r=a||_x,this.d=l||this,this.set=c||lh,this.pr=u||0,this._next=t,t&&(t._prev=this)}var e=i.prototype;return e.modifier=function(n,s,r){this.mSet=this.mSet||this.set,this.set=hw,this.m=n,this.mt=r,this.tween=s},i})();Vn(ih+"parent,duration,ease,delay,overwrite,runBackwards,startAt,yoyo,immediateRender,repeat,repeatDelay,data,paused,reversed,lazy,callbackScope,stringFilter,id,yoyoEase,stagger,inherit,repeatRefresh,keyframes,autoRevert,scrollTrigger",function(i){return nh[i]=1});ii.TweenMax=ii.TweenLite=Kt;ii.TimelineLite=ii.TimelineMax=Dn;kt=new Dn({sortChildren:!1,defaults:Do,autoRemoveChildren:!0,id:"root",smoothChildTiming:!0});ni.stringFilter=cx;var Tr=[],Xl={},pw=[],wm=0,mw=0,Ou=function(e){return(Xl[e]||pw).map(function(t){return t()})},$f=function(){var e=Date.now(),t=[];e-wm>2&&(Ou("matchMediaInit"),Tr.forEach(function(n){var s=n.queries,r=n.conditions,o,a,l,c;for(a in s)o=Ri.matchMedia(s[a]).matches,o&&(l=1),o!==r[a]&&(r[a]=o,c=1);c&&(n.revert(),l&&t.push(n))}),Ou("matchMediaRevert"),t.forEach(function(n){return n.onMatch(n,function(s){return n.add(null,s)})}),wm=e,Ou("matchMedia"))},Sx=(function(){function i(t,n){this.selector=n&&Qf(n),this.data=[],this._r=[],this.isReverted=!1,this.id=mw++,t&&this.add(t)}var e=i.prototype;return e.add=function(n,s,r){Vt(n)&&(r=s,s=n,n=Vt);var o=this,a=function(){var c=Ut,u=o.selector,f;return c&&c!==o&&c.data.push(o),r&&(o.selector=Qf(r)),Ut=o,f=s.apply(o,arguments),Vt(f)&&o._r.push(f),Ut=c,o.selector=u,o.isReverted=!1,f};return o.last=a,n===Vt?a(o,function(l){return o.add(null,l)}):n?o[n]=a:a},e.ignore=function(n){var s=Ut;Ut=null,n(this),Ut=s},e.getTweens=function(){var n=[];return this.data.forEach(function(s){return s instanceof i?n.push.apply(n,s.getTweens()):s instanceof Kt&&!(s.parent&&s.parent.data==="nested")&&n.push(s)}),n},e.clear=function(){this._r.length=this.data.length=0},e.kill=function(n,s){var r=this;if(n?(function(){for(var a=r.getTweens(),l=r.data.length,c;l--;)c=r.data[l],c.data==="isFlip"&&(c.revert(),c.getChildren(!0,!0,!1).forEach(function(u){return a.splice(a.indexOf(u),1)}));for(a.map(function(u){return{g:u._dur||u._delay||u._sat&&!u._sat.vars.immediateRender?u.globalTime(0):-1/0,t:u}}).sort(function(u,f){return f.g-u.g||-1/0}).forEach(function(u){return u.t.revert(n)}),l=r.data.length;l--;)c=r.data[l],c instanceof Dn?c.data!=="nested"&&(c.scrollTrigger&&c.scrollTrigger.revert(),c.kill()):!(c instanceof Kt)&&c.revert&&c.revert(n);r._r.forEach(function(u){return u(n,r)}),r.isReverted=!0})():this.data.forEach(function(a){return a.kill&&a.kill()}),this.clear(),s)for(var o=Tr.length;o--;)Tr[o].id===this.id&&Tr.splice(o,1)},e.revert=function(n){this.kill(n||{})},i})(),gw=(function(){function i(t){this.contexts=[],this.scope=t,Ut&&Ut.data.push(this)}var e=i.prototype;return e.add=function(n,s,r){Wi(n)||(n={matches:n});var o=new Sx(0,r||this.scope),a=o.conditions={},l,c,u;Ut&&!o.selector&&(o.selector=Ut.selector),this.contexts.push(o),s=o.add("onMatch",s),o.queries=n;for(c in n)c==="all"?u=1:(l=Ri.matchMedia(n[c]),l&&(Tr.indexOf(o)<0&&Tr.push(o),(a[c]=l.matches)&&(u=1),l.addListener?l.addListener($f):l.addEventListener("change",$f)));return u&&s(o,function(f){return o.add(null,f)}),this},e.revert=function(n){this.kill(n||{})},e.kill=function(n){this.contexts.forEach(function(s){return s.kill(n,!0)})},i})(),uc={registerPlugin:function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];t.forEach(function(s){return ox(s)})},timeline:function(e){return new Dn(e)},getTweensOf:function(e,t){return kt.getTweensOf(e,t)},getProperty:function(e,t,n,s){sn(e)&&(e=mi(e)[0]);var r=br(e||{}).get,o=n?Yg:qg;return n==="native"&&(n=""),e&&(t?o((Kn[t]&&Kn[t].get||r)(e,t,n,s)):function(a,l,c){return o((Kn[a]&&Kn[a].get||r)(e,a,l,c))})},quickSetter:function(e,t,n){if(e=mi(e),e.length>1){var s=e.map(function(u){return Xn.quickSetter(u,t,n)}),r=s.length;return function(u){for(var f=r;f--;)s[f](u)}}e=e[0]||{};var o=Kn[t],a=br(e),l=a.harness&&(a.harness.aliases||{})[t]||t,c=o?function(u){var f=new o;ro._pt=0,f.init(e,n?u+n:u,ro,0,[e]),f.render(1,f),ro._pt&&uh(1,ro)}:a.set(e,l);return o?c:function(u){return c(e,l,n?u+n:u,a,1)}},quickTo:function(e,t,n){var s,r=Xn.to(e,si((s={},s[t]="+=0.1",s.paused=!0,s.stagger=0,s),n||{})),o=function(l,c,u){return r.resetTo(t,l,c,u)};return o.tween=r,o},isTweening:function(e){return kt.getTweensOf(e,!0).length>0},defaults:function(e){return e&&e.ease&&(e.ease=Cr(e.ease,Do.ease)),bm(Do,e||{})},config:function(e){return bm(ni,e||{})},registerEffect:function(e){var t=e.name,n=e.effect,s=e.plugins,r=e.defaults,o=e.extendTimeline;(s||"").split(",").forEach(function(a){return a&&!Kn[a]&&!ii[a]&&Ba(t+" effect requires "+a+" plugin.")}),Fu[t]=function(a,l,c){return n(mi(a),si(l||{},r),c)},o&&(Dn.prototype[t]=function(a,l,c){return this.add(Fu[t](a,Wi(l)?l:(c=l)&&{},this),c)})},registerEase:function(e,t){at[e]=Cr(t)},parseEase:function(e,t){return arguments.length?Cr(e,t):at},getById:function(e){return kt.getById(e)},exportRoot:function(e,t){e===void 0&&(e={});var n=new Dn(e),s,r;for(n.smoothChildTiming=Hn(e.smoothChildTiming),kt.remove(n),n._dp=0,n._time=n._tTime=kt._time,s=kt._first;s;)r=s._next,(t||!(!s._dur&&s instanceof Kt&&s.vars.onComplete===s._targets[0]))&&Fi(n,s,s._start-s._delay),s=r;return Fi(kt,n,0),n},context:function(e,t){return e?new Sx(e,t):Ut},matchMedia:function(e){return new gw(e)},matchMediaRefresh:function(){return Tr.forEach(function(e){var t=e.conditions,n,s;for(s in t)t[s]&&(t[s]=!1,n=1);n&&e.revert()})||$f()},addEventListener:function(e,t){var n=Xl[e]||(Xl[e]=[]);~n.indexOf(t)||n.push(t)},removeEventListener:function(e,t){var n=Xl[e],s=n&&n.indexOf(t);s>=0&&n.splice(s,1)},utils:{wrap:Q1,wrapYoyo:K1,distribute:ex,random:nx,snap:tx,normalize:Y1,getUnit:_n,clamp:G1,splitColor:ax,toArray:mi,selector:Qf,mapRange:sx,pipe:X1,unitize:q1,interpolate:j1,shuffle:Jg},install:Hg,effects:Fu,ticker:jn,updateRoot:Dn.updateRoot,plugins:Kn,globalTimeline:kt,core:{PropTween:Gn,globals:Vg,Tween:Kt,Timeline:Dn,Animation:za,getCache:br,_removeLinkedListItem:Rc,reverting:function(){return un},context:function(e){return e&&Ut&&(Ut.data.push(e),e._ctx=Ut),Ut},suppressOverwrites:function(e){return $d=e}}};Vn("to,from,fromTo,delayedCall,set,killTweensOf",function(i){return uc[i]=Kt[i]});jn.add(Dn.updateRoot);ro=uc.to({},{duration:0});var xw=function(e,t){for(var n=e._pt;n&&n.p!==t&&n.op!==t&&n.fp!==t;)n=n._next;return n},_w=function(e,t){var n=e._targets,s,r,o;for(s in t)for(r=n.length;r--;)o=e._ptLookup[r][s],o&&(o=o.d)&&(o._pt&&(o=xw(o,s)),o&&o.modifier&&o.modifier(t[s],e,n[r],s))},Nu=function(e,t){return{name:e,headless:1,rawVars:1,init:function(s,r,o){o._onInit=function(a){var l,c;if(sn(r)&&(l={},Vn(r,function(u){return l[u]=1}),r=l),t){l={};for(c in r)l[c]=t(r[c]);r=l}_w(a,r)}}}},Xn=uc.registerPlugin({name:"attr",init:function(e,t,n,s,r){var o,a,l;this.tween=n;for(o in t)l=e.getAttribute(o)||"",a=this.add(e,"setAttribute",(l||0)+"",t[o],s,r,0,0,o),a.op=o,a.b=l,this._props.push(o)},render:function(e,t){for(var n=t._pt;n;)un?n.set(n.t,n.p,n.b,n):n.r(e,n.d),n=n._next}},{name:"endArray",headless:1,init:function(e,t){for(var n=t.length;n--;)this.add(e,n,e[n]||0,t[n],0,0,0,0,0,1)}},Nu("roundProps",Kf),Nu("modifiers"),Nu("snap",tx))||uc;Kt.version=Dn.version=Xn.version="3.14.2";kg=1;Jd()&&Bo();at.Power0;at.Power1;at.Power2;at.Power3;at.Power4;at.Linear;at.Quad;at.Cubic;at.Quart;at.Quint;at.Strong;at.Elastic;at.Back;at.SteppedEase;at.Bounce;at.Sine;at.Expo;at.Circ;var Rm,Os,vo,fh,Ar,Im,dh,vw=function(){return typeof window<"u"},_s={},hr=180/Math.PI,Ao=Math.PI/180,Zr=Math.atan2,Dm=1e8,hh=/([A-Z])/g,Aw=/(left|right|width|margin|padding|x)/i,Sw=/[\s,\(]\S/,Oi={autoAlpha:"opacity,visibility",scale:"scaleX,scaleY",alpha:"opacity"},Zf=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},yw=function(e,t){return t.set(t.t,t.p,e===1?t.e:Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},bw=function(e,t){return t.set(t.t,t.p,e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},Mw=function(e,t){return t.set(t.t,t.p,e===1?t.e:e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},Cw=function(e,t){var n=t.s+t.c*e;t.set(t.t,t.p,~~(n+(n<0?-.5:.5))+t.u,t)},yx=function(e,t){return t.set(t.t,t.p,e?t.e:t.b,t)},bx=function(e,t){return t.set(t.t,t.p,e!==1?t.b:t.e,t)},Tw=function(e,t,n){return e.style[t]=n},Ew=function(e,t,n){return e.style.setProperty(t,n)},ww=function(e,t,n){return e._gsap[t]=n},Rw=function(e,t,n){return e._gsap.scaleX=e._gsap.scaleY=n},Iw=function(e,t,n,s,r){var o=e._gsap;o.scaleX=o.scaleY=n,o.renderTransform(r,o)},Dw=function(e,t,n,s,r){var o=e._gsap;o[t]=n,o.renderTransform(r,o)},Ht="transform",Wn=Ht+"Origin",Pw=function i(e,t){var n=this,s=this.target,r=s.style,o=s._gsap;if(e in _s&&r){if(this.tfm=this.tfm||{},e!=="transform")e=Oi[e]||e,~e.indexOf(",")?e.split(",").forEach(function(a){return n.tfm[a]=as(s,a)}):this.tfm[e]=o.x?o[e]:as(s,e),e===Wn&&(this.tfm.zOrigin=o.zOrigin);else return Oi.transform.split(",").forEach(function(a){return i.call(n,a,t)});if(this.props.indexOf(Ht)>=0)return;o.svg&&(this.svgo=s.getAttribute("data-svg-origin"),this.props.push(Wn,t,"")),e=Ht}(r||t)&&this.props.push(e,t,r[e])},Mx=function(e){e.translate&&(e.removeProperty("translate"),e.removeProperty("scale"),e.removeProperty("rotate"))},Fw=function(){var e=this.props,t=this.target,n=t.style,s=t._gsap,r,o;for(r=0;r<e.length;r+=3)e[r+1]?e[r+1]===2?t[e[r]](e[r+2]):t[e[r]]=e[r+2]:e[r+2]?n[e[r]]=e[r+2]:n.removeProperty(e[r].substr(0,2)==="--"?e[r]:e[r].replace(hh,"-$1").toLowerCase());if(this.tfm){for(o in this.tfm)s[o]=this.tfm[o];s.svg&&(s.renderTransform(),t.setAttribute("data-svg-origin",this.svgo||"")),r=dh(),(!r||!r.isStart)&&!n[Ht]&&(Mx(n),s.zOrigin&&n[Wn]&&(n[Wn]+=" "+s.zOrigin+"px",s.zOrigin=0,s.renderTransform()),s.uncache=1)}},Cx=function(e,t){var n={target:e,props:[],revert:Fw,save:Pw};return e._gsap||Xn.core.getCache(e),t&&e.style&&e.nodeType&&t.split(",").forEach(function(s){return n.save(s)}),n},Tx,Jf=function(e,t){var n=Os.createElementNS?Os.createElementNS((t||"http://www.w3.org/1999/xhtml").replace(/^https/,"http"),e):Os.createElement(e);return n&&n.style?n:Os.createElement(e)},ei=function i(e,t,n){var s=getComputedStyle(e);return s[t]||s.getPropertyValue(t.replace(hh,"-$1").toLowerCase())||s.getPropertyValue(t)||!n&&i(e,Uo(t)||t,1)||""},Pm="O,Moz,ms,Ms,Webkit".split(","),Uo=function(e,t,n){var s=t||Ar,r=s.style,o=5;if(e in r&&!n)return e;for(e=e.charAt(0).toUpperCase()+e.substr(1);o--&&!(Pm[o]+e in r););return o<0?null:(o===3?"ms":o>=0?Pm[o]:"")+e},ed=function(){vw()&&window.document&&(Rm=window,Os=Rm.document,vo=Os.documentElement,Ar=Jf("div")||{style:{}},Jf("div"),Ht=Uo(Ht),Wn=Ht+"Origin",Ar.style.cssText="border-width:0;line-height:0;position:absolute;padding:0",Tx=!!Uo("perspective"),dh=Xn.core.reverting,fh=1)},Fm=function(e){var t=e.ownerSVGElement,n=Jf("svg",t&&t.getAttribute("xmlns")||"http://www.w3.org/2000/svg"),s=e.cloneNode(!0),r;s.style.display="block",n.appendChild(s),vo.appendChild(n);try{r=s.getBBox()}catch{}return n.removeChild(s),vo.removeChild(n),r},Lm=function(e,t){for(var n=t.length;n--;)if(e.hasAttribute(t[n]))return e.getAttribute(t[n])},Ex=function(e){var t,n;try{t=e.getBBox()}catch{t=Fm(e),n=1}return t&&(t.width||t.height)||n||(t=Fm(e)),t&&!t.width&&!t.x&&!t.y?{x:+Lm(e,["x","cx","x1"])||0,y:+Lm(e,["y","cy","y1"])||0,width:0,height:0}:t},wx=function(e){return!!(e.getCTM&&(!e.parentNode||e.ownerSVGElement)&&Ex(e))},Qs=function(e,t){if(t){var n=e.style,s;t in _s&&t!==Wn&&(t=Ht),n.removeProperty?(s=t.substr(0,2),(s==="ms"||t.substr(0,6)==="webkit")&&(t="-"+t),n.removeProperty(s==="--"?t:t.replace(hh,"-$1").toLowerCase())):n.removeAttribute(t)}},Ns=function(e,t,n,s,r,o){var a=new Gn(e._pt,t,n,0,1,o?bx:yx);return e._pt=a,a.b=s,a.e=r,e._props.push(n),a},Bm={deg:1,rad:1,turn:1},Lw={grid:1,flex:1},Ks=function i(e,t,n,s){var r=parseFloat(n)||0,o=(n+"").trim().substr((r+"").length)||"px",a=Ar.style,l=Aw.test(t),c=e.tagName.toLowerCase()==="svg",u=(c?"client":"offset")+(l?"Width":"Height"),f=100,d=s==="px",h=s==="%",x,p,g,m;if(s===o||!r||Bm[s]||Bm[o])return r;if(o!=="px"&&!d&&(r=i(e,t,n,"px")),m=e.getCTM&&wx(e),(h||o==="%")&&(_s[t]||~t.indexOf("adius")))return x=m?e.getBBox()[l?"width":"height"]:e[u],Gt(h?r/x*f:r/100*x);if(a[l?"width":"height"]=f+(d?o:s),p=s!=="rem"&&~t.indexOf("adius")||s==="em"&&e.appendChild&&!c?e:e.parentNode,m&&(p=(e.ownerSVGElement||{}).parentNode),(!p||p===Os||!p.appendChild)&&(p=Os.body),g=p._gsap,g&&h&&g.width&&l&&g.time===jn.time&&!g.uncache)return Gt(r/g.width*f);if(h&&(t==="height"||t==="width")){var _=e.style[t];e.style[t]=f+s,x=e[u],_?e.style[t]=_:Qs(e,t)}else(h||o==="%")&&!Lw[ei(p,"display")]&&(a.position=ei(e,"position")),p===e&&(a.position="static"),p.appendChild(Ar),x=Ar[u],p.removeChild(Ar),a.position="absolute";return l&&h&&(g=br(p),g.time=jn.time,g.width=p[u]),Gt(d?x*r/f:x&&r?f/x*r:0)},as=function(e,t,n,s){var r;return fh||ed(),t in Oi&&t!=="transform"&&(t=Oi[t],~t.indexOf(",")&&(t=t.split(",")[0])),_s[t]&&t!=="transform"?(r=Ha(e,s),r=t!=="transformOrigin"?r[t]:r.svg?r.origin:dc(ei(e,Wn))+" "+r.zOrigin+"px"):(r=e.style[t],(!r||r==="auto"||s||~(r+"").indexOf("calc("))&&(r=fc[t]&&fc[t](e,t,n)||ei(e,t)||Wg(e,t)||(t==="opacity"?1:0))),n&&!~(r+"").trim().indexOf(" ")?Ks(e,t,r,n)+n:r},Bw=function(e,t,n,s){if(!n||n==="none"){var r=Uo(t,e,1),o=r&&ei(e,r,1);o&&o!==n?(t=r,n=o):t==="borderColor"&&(n=ei(e,"borderTopColor"))}var a=new Gn(this._pt,e.style,t,0,1,vx),l=0,c=0,u,f,d,h,x,p,g,m,_,v,A,S;if(a.b=n,a.e=s,n+="",s+="",s.substring(0,6)==="var(--"&&(s=ei(e,s.substring(4,s.indexOf(")")))),s==="auto"&&(p=e.style[t],e.style[t]=s,s=ei(e,t)||s,p?e.style[t]=p:Qs(e,t)),u=[n,s],cx(u),n=u[0],s=u[1],d=n.match(so)||[],S=s.match(so)||[],S.length){for(;f=so.exec(s);)g=f[0],_=s.substring(l,f.index),x?x=(x+1)%5:(_.substr(-5)==="rgba("||_.substr(-5)==="hsla(")&&(x=1),g!==(p=d[c++]||"")&&(h=parseFloat(p)||0,A=p.substr((h+"").length),g.charAt(1)==="="&&(g=_o(h,g)+A),m=parseFloat(g),v=g.substr((m+"").length),l=so.lastIndex-v.length,v||(v=v||ni.units[t]||A,l===s.length&&(s+=v,a.e+=v)),A!==v&&(h=Ks(e,t,p,v)||0),a._pt={_next:a._pt,p:_||c===1?_:",",s:h,c:m-h,m:x&&x<4||t==="zIndex"?Math.round:0});a.c=l<s.length?s.substring(l,s.length):""}else a.r=t==="display"&&s==="none"?bx:yx;return zg.test(s)&&(a.e=0),this._pt=a,a},Um={top:"0%",bottom:"100%",left:"0%",right:"100%",center:"50%"},Uw=function(e){var t=e.split(" "),n=t[0],s=t[1]||"50%";return(n==="top"||n==="bottom"||s==="left"||s==="right")&&(e=n,n=s,s=e),t[0]=Um[n]||n,t[1]=Um[s]||s,t.join(" ")},Ow=function(e,t){if(t.tween&&t.tween._time===t.tween._dur){var n=t.t,s=n.style,r=t.u,o=n._gsap,a,l,c;if(r==="all"||r===!0)s.cssText="",l=1;else for(r=r.split(","),c=r.length;--c>-1;)a=r[c],_s[a]&&(l=1,a=a==="transformOrigin"?Wn:Ht),Qs(n,a);l&&(Qs(n,Ht),o&&(o.svg&&n.removeAttribute("transform"),s.scale=s.rotate=s.translate="none",Ha(n,1),o.uncache=1,Mx(s)))}},fc={clearProps:function(e,t,n,s,r){if(r.data!=="isFromStart"){var o=e._pt=new Gn(e._pt,t,n,0,0,Ow);return o.u=s,o.pr=-10,o.tween=r,e._props.push(n),1}}},ka=[1,0,0,1,0,0],Rx={},Ix=function(e){return e==="matrix(1, 0, 0, 1, 0, 0)"||e==="none"||!e},Om=function(e){var t=ei(e,Ht);return Ix(t)?ka:t.substr(7).match(Ng).map(Gt)},ph=function(e,t){var n=e._gsap||br(e),s=e.style,r=Om(e),o,a,l,c;return n.svg&&e.getAttribute("transform")?(l=e.transform.baseVal.consolidate().matrix,r=[l.a,l.b,l.c,l.d,l.e,l.f],r.join(",")==="1,0,0,1,0,0"?ka:r):(r===ka&&!e.offsetParent&&e!==vo&&!n.svg&&(l=s.display,s.display="block",o=e.parentNode,(!o||!e.offsetParent&&!e.getBoundingClientRect().width)&&(c=1,a=e.nextElementSibling,vo.appendChild(e)),r=Om(e),l?s.display=l:Qs(e,"display"),c&&(a?o.insertBefore(e,a):o?o.appendChild(e):vo.removeChild(e))),t&&r.length>6?[r[0],r[1],r[4],r[5],r[12],r[13]]:r)},td=function(e,t,n,s,r,o){var a=e._gsap,l=r||ph(e,!0),c=a.xOrigin||0,u=a.yOrigin||0,f=a.xOffset||0,d=a.yOffset||0,h=l[0],x=l[1],p=l[2],g=l[3],m=l[4],_=l[5],v=t.split(" "),A=parseFloat(v[0])||0,S=parseFloat(v[1])||0,b,M,R,y;n?l!==ka&&(M=h*g-x*p)&&(R=A*(g/M)+S*(-p/M)+(p*_-g*m)/M,y=A*(-x/M)+S*(h/M)-(h*_-x*m)/M,A=R,S=y):(b=Ex(e),A=b.x+(~v[0].indexOf("%")?A/100*b.width:A),S=b.y+(~(v[1]||v[0]).indexOf("%")?S/100*b.height:S)),s||s!==!1&&a.smooth?(m=A-c,_=S-u,a.xOffset=f+(m*h+_*p)-m,a.yOffset=d+(m*x+_*g)-_):a.xOffset=a.yOffset=0,a.xOrigin=A,a.yOrigin=S,a.smooth=!!s,a.origin=t,a.originIsAbsolute=!!n,e.style[Wn]="0px 0px",o&&(Ns(o,a,"xOrigin",c,A),Ns(o,a,"yOrigin",u,S),Ns(o,a,"xOffset",f,a.xOffset),Ns(o,a,"yOffset",d,a.yOffset)),e.setAttribute("data-svg-origin",A+" "+S)},Ha=function(e,t){var n=e._gsap||new hx(e);if("x"in n&&!t&&!n.uncache)return n;var s=e.style,r=n.scaleX<0,o="px",a="deg",l=getComputedStyle(e),c=ei(e,Wn)||"0",u,f,d,h,x,p,g,m,_,v,A,S,b,M,R,y,C,F,I,O,z,H,V,q,G,Y,ae,Ae,Pe,Ne,We,He;return u=f=d=p=g=m=_=v=A=0,h=x=1,n.svg=!!(e.getCTM&&wx(e)),l.translate&&((l.translate!=="none"||l.scale!=="none"||l.rotate!=="none")&&(s[Ht]=(l.translate!=="none"?"translate3d("+(l.translate+" 0 0").split(" ").slice(0,3).join(", ")+") ":"")+(l.rotate!=="none"?"rotate("+l.rotate+") ":"")+(l.scale!=="none"?"scale("+l.scale.split(" ").join(",")+") ":"")+(l[Ht]!=="none"?l[Ht]:"")),s.scale=s.rotate=s.translate="none"),M=ph(e,n.svg),n.svg&&(n.uncache?(G=e.getBBox(),c=n.xOrigin-G.x+"px "+(n.yOrigin-G.y)+"px",q=""):q=!t&&e.getAttribute("data-svg-origin"),td(e,q||c,!!q||n.originIsAbsolute,n.smooth!==!1,M)),S=n.xOrigin||0,b=n.yOrigin||0,M!==ka&&(F=M[0],I=M[1],O=M[2],z=M[3],u=H=M[4],f=V=M[5],M.length===6?(h=Math.sqrt(F*F+I*I),x=Math.sqrt(z*z+O*O),p=F||I?Zr(I,F)*hr:0,_=O||z?Zr(O,z)*hr+p:0,_&&(x*=Math.abs(Math.cos(_*Ao))),n.svg&&(u-=S-(S*F+b*O),f-=b-(S*I+b*z))):(He=M[6],Ne=M[7],ae=M[8],Ae=M[9],Pe=M[10],We=M[11],u=M[12],f=M[13],d=M[14],R=Zr(He,Pe),g=R*hr,R&&(y=Math.cos(-R),C=Math.sin(-R),q=H*y+ae*C,G=V*y+Ae*C,Y=He*y+Pe*C,ae=H*-C+ae*y,Ae=V*-C+Ae*y,Pe=He*-C+Pe*y,We=Ne*-C+We*y,H=q,V=G,He=Y),R=Zr(-O,Pe),m=R*hr,R&&(y=Math.cos(-R),C=Math.sin(-R),q=F*y-ae*C,G=I*y-Ae*C,Y=O*y-Pe*C,We=z*C+We*y,F=q,I=G,O=Y),R=Zr(I,F),p=R*hr,R&&(y=Math.cos(R),C=Math.sin(R),q=F*y+I*C,G=H*y+V*C,I=I*y-F*C,V=V*y-H*C,F=q,H=G),g&&Math.abs(g)+Math.abs(p)>359.9&&(g=p=0,m=180-m),h=Gt(Math.sqrt(F*F+I*I+O*O)),x=Gt(Math.sqrt(V*V+He*He)),R=Zr(H,V),_=Math.abs(R)>2e-4?R*hr:0,A=We?1/(We<0?-We:We):0),n.svg&&(q=e.getAttribute("transform"),n.forceCSS=e.setAttribute("transform","")||!Ix(ei(e,Ht)),q&&e.setAttribute("transform",q))),Math.abs(_)>90&&Math.abs(_)<270&&(r?(h*=-1,_+=p<=0?180:-180,p+=p<=0?180:-180):(x*=-1,_+=_<=0?180:-180)),t=t||n.uncache,n.x=u-((n.xPercent=u&&(!t&&n.xPercent||(Math.round(e.offsetWidth/2)===Math.round(-u)?-50:0)))?e.offsetWidth*n.xPercent/100:0)+o,n.y=f-((n.yPercent=f&&(!t&&n.yPercent||(Math.round(e.offsetHeight/2)===Math.round(-f)?-50:0)))?e.offsetHeight*n.yPercent/100:0)+o,n.z=d+o,n.scaleX=Gt(h),n.scaleY=Gt(x),n.rotation=Gt(p)+a,n.rotationX=Gt(g)+a,n.rotationY=Gt(m)+a,n.skewX=_+a,n.skewY=v+a,n.transformPerspective=A+o,(n.zOrigin=parseFloat(c.split(" ")[2])||!t&&n.zOrigin||0)&&(s[Wn]=dc(c)),n.xOffset=n.yOffset=0,n.force3D=ni.force3D,n.renderTransform=n.svg?zw:Tx?Dx:Nw,n.uncache=0,n},dc=function(e){return(e=e.split(" "))[0]+" "+e[1]},zu=function(e,t,n){var s=_n(t);return Gt(parseFloat(t)+parseFloat(Ks(e,"x",n+"px",s)))+s},Nw=function(e,t){t.z="0px",t.rotationY=t.rotationX="0deg",t.force3D=0,Dx(e,t)},lr="0deg",Jo="0px",cr=") ",Dx=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.z,c=n.rotation,u=n.rotationY,f=n.rotationX,d=n.skewX,h=n.skewY,x=n.scaleX,p=n.scaleY,g=n.transformPerspective,m=n.force3D,_=n.target,v=n.zOrigin,A="",S=m==="auto"&&e&&e!==1||m===!0;if(v&&(f!==lr||u!==lr)){var b=parseFloat(u)*Ao,M=Math.sin(b),R=Math.cos(b),y;b=parseFloat(f)*Ao,y=Math.cos(b),o=zu(_,o,M*y*-v),a=zu(_,a,-Math.sin(b)*-v),l=zu(_,l,R*y*-v+v)}g!==Jo&&(A+="perspective("+g+cr),(s||r)&&(A+="translate("+s+"%, "+r+"%) "),(S||o!==Jo||a!==Jo||l!==Jo)&&(A+=l!==Jo||S?"translate3d("+o+", "+a+", "+l+") ":"translate("+o+", "+a+cr),c!==lr&&(A+="rotate("+c+cr),u!==lr&&(A+="rotateY("+u+cr),f!==lr&&(A+="rotateX("+f+cr),(d!==lr||h!==lr)&&(A+="skew("+d+", "+h+cr),(x!==1||p!==1)&&(A+="scale("+x+", "+p+cr),_.style[Ht]=A||"translate(0, 0)"},zw=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.rotation,c=n.skewX,u=n.skewY,f=n.scaleX,d=n.scaleY,h=n.target,x=n.xOrigin,p=n.yOrigin,g=n.xOffset,m=n.yOffset,_=n.forceCSS,v=parseFloat(o),A=parseFloat(a),S,b,M,R,y;l=parseFloat(l),c=parseFloat(c),u=parseFloat(u),u&&(u=parseFloat(u),c+=u,l+=u),l||c?(l*=Ao,c*=Ao,S=Math.cos(l)*f,b=Math.sin(l)*f,M=Math.sin(l-c)*-d,R=Math.cos(l-c)*d,c&&(u*=Ao,y=Math.tan(c-u),y=Math.sqrt(1+y*y),M*=y,R*=y,u&&(y=Math.tan(u),y=Math.sqrt(1+y*y),S*=y,b*=y)),S=Gt(S),b=Gt(b),M=Gt(M),R=Gt(R)):(S=f,R=d,b=M=0),(v&&!~(o+"").indexOf("px")||A&&!~(a+"").indexOf("px"))&&(v=Ks(h,"x",o,"px"),A=Ks(h,"y",a,"px")),(x||p||g||m)&&(v=Gt(v+x-(x*S+p*M)+g),A=Gt(A+p-(x*b+p*R)+m)),(s||r)&&(y=h.getBBox(),v=Gt(v+s/100*y.width),A=Gt(A+r/100*y.height)),y="matrix("+S+","+b+","+M+","+R+","+v+","+A+")",h.setAttribute("transform",y),_&&(h.style[Ht]=y)},kw=function(e,t,n,s,r){var o=360,a=sn(r),l=parseFloat(r)*(a&&~r.indexOf("rad")?hr:1),c=l-s,u=s+c+"deg",f,d;return a&&(f=r.split("_")[1],f==="short"&&(c%=o,c!==c%(o/2)&&(c+=c<0?o:-o)),f==="cw"&&c<0?c=(c+o*Dm)%o-~~(c/o)*o:f==="ccw"&&c>0&&(c=(c-o*Dm)%o-~~(c/o)*o)),e._pt=d=new Gn(e._pt,t,n,s,c,yw),d.e=u,d.u="deg",e._props.push(n),d},Nm=function(e,t){for(var n in t)e[n]=t[n];return e},Hw=function(e,t,n){var s=Nm({},n._gsap),r="perspective,force3D,transformOrigin,svgOrigin",o=n.style,a,l,c,u,f,d,h,x;s.svg?(c=n.getAttribute("transform"),n.setAttribute("transform",""),o[Ht]=t,a=Ha(n,1),Qs(n,Ht),n.setAttribute("transform",c)):(c=getComputedStyle(n)[Ht],o[Ht]=t,a=Ha(n,1),o[Ht]=c);for(l in _s)c=s[l],u=a[l],c!==u&&r.indexOf(l)<0&&(h=_n(c),x=_n(u),f=h!==x?Ks(n,l,c,x):parseFloat(c),d=parseFloat(u),e._pt=new Gn(e._pt,a,l,f,d-f,Zf),e._pt.u=x||0,e._props.push(l));Nm(a,s)};Vn("padding,margin,Width,Radius",function(i,e){var t="Top",n="Right",s="Bottom",r="Left",o=(e<3?[t,n,s,r]:[t+r,t+n,s+n,s+r]).map(function(a){return e<2?i+a:"border"+a+i});fc[e>1?"border"+i:i]=function(a,l,c,u,f){var d,h;if(arguments.length<4)return d=o.map(function(x){return as(a,x,c)}),h=d.join(" "),h.split(d[0]).length===5?d[0]:h;d=(u+"").split(" "),h={},o.forEach(function(x,p){return h[x]=d[p]=d[p]||d[(p-1)/2|0]}),a.init(l,h,f)}});var Px={name:"css",register:ed,targetTest:function(e){return e.style&&e.nodeType},init:function(e,t,n,s,r){var o=this._props,a=e.style,l=n.vars.startAt,c,u,f,d,h,x,p,g,m,_,v,A,S,b,M,R,y;fh||ed(),this.styles=this.styles||Cx(e),R=this.styles.props,this.tween=n;for(p in t)if(p!=="autoRound"&&(u=t[p],!(Kn[p]&&px(p,t,n,s,e,r)))){if(h=typeof u,x=fc[p],h==="function"&&(u=u.call(n,s,e,r),h=typeof u),h==="string"&&~u.indexOf("random(")&&(u=Oa(u)),x)x(this,e,p,u,n)&&(M=1);else if(p.substr(0,2)==="--")c=(getComputedStyle(e).getPropertyValue(p)+"").trim(),u+="",Gs.lastIndex=0,Gs.test(c)||(g=_n(c),m=_n(u),m?g!==m&&(c=Ks(e,p,c,m)+m):g&&(u+=g)),this.add(a,"setProperty",c,u,s,r,0,0,p),o.push(p),R.push(p,0,a[p]);else if(h!=="undefined"){if(l&&p in l?(c=typeof l[p]=="function"?l[p].call(n,s,e,r):l[p],sn(c)&&~c.indexOf("random(")&&(c=Oa(c)),_n(c+"")||c==="auto"||(c+=ni.units[p]||_n(as(e,p))||""),(c+"").charAt(1)==="="&&(c=as(e,p))):c=as(e,p),d=parseFloat(c),_=h==="string"&&u.charAt(1)==="="&&u.substr(0,2),_&&(u=u.substr(2)),f=parseFloat(u),p in Oi&&(p==="autoAlpha"&&(d===1&&as(e,"visibility")==="hidden"&&f&&(d=0),R.push("visibility",0,a.visibility),Ns(this,a,"visibility",d?"inherit":"hidden",f?"inherit":"hidden",!f)),p!=="scale"&&p!=="transform"&&(p=Oi[p],~p.indexOf(",")&&(p=p.split(",")[0]))),v=p in _s,v){if(this.styles.save(p),y=u,h==="string"&&u.substring(0,6)==="var(--"){if(u=ei(e,u.substring(4,u.indexOf(")"))),u.substring(0,5)==="calc("){var C=e.style.perspective;e.style.perspective=u,u=ei(e,"perspective"),C?e.style.perspective=C:Qs(e,"perspective")}f=parseFloat(u)}if(A||(S=e._gsap,S.renderTransform&&!t.parseTransform||Ha(e,t.parseTransform),b=t.smoothOrigin!==!1&&S.smooth,A=this._pt=new Gn(this._pt,a,Ht,0,1,S.renderTransform,S,0,-1),A.dep=1),p==="scale")this._pt=new Gn(this._pt,S,"scaleY",S.scaleY,(_?_o(S.scaleY,_+f):f)-S.scaleY||0,Zf),this._pt.u=0,o.push("scaleY",p),p+="X";else if(p==="transformOrigin"){R.push(Wn,0,a[Wn]),u=Uw(u),S.svg?td(e,u,0,b,0,this):(m=parseFloat(u.split(" ")[2])||0,m!==S.zOrigin&&Ns(this,S,"zOrigin",S.zOrigin,m),Ns(this,a,p,dc(c),dc(u)));continue}else if(p==="svgOrigin"){td(e,u,1,b,0,this);continue}else if(p in Rx){kw(this,S,p,d,_?_o(d,_+u):u);continue}else if(p==="smoothOrigin"){Ns(this,S,"smooth",S.smooth,u);continue}else if(p==="force3D"){S[p]=u;continue}else if(p==="transform"){Hw(this,u,e);continue}}else p in a||(p=Uo(p)||p);if(v||(f||f===0)&&(d||d===0)&&!Sw.test(u)&&p in a)g=(c+"").substr((d+"").length),f||(f=0),m=_n(u)||(p in ni.units?ni.units[p]:g),g!==m&&(d=Ks(e,p,c,m)),this._pt=new Gn(this._pt,v?S:a,p,d,(_?_o(d,_+f):f)-d,!v&&(m==="px"||p==="zIndex")&&t.autoRound!==!1?Cw:Zf),this._pt.u=m||0,v&&y!==u?(this._pt.b=c,this._pt.e=y,this._pt.r=Mw):g!==m&&m!=="%"&&(this._pt.b=c,this._pt.r=bw);else if(p in a)Bw.call(this,e,p,c,_?_+u:u);else if(p in e)this.add(e,p,c||e[p],_?_+u:u,s,r);else if(p!=="parseTransform"){th(p,u);continue}v||(p in a?R.push(p,0,a[p]):typeof e[p]=="function"?R.push(p,2,e[p]()):R.push(p,1,c||e[p])),o.push(p)}}M&&Ax(this)},render:function(e,t){if(t.tween._time||!dh())for(var n=t._pt;n;)n.r(e,n.d),n=n._next;else t.styles.revert()},get:as,aliases:Oi,getSetter:function(e,t,n){var s=Oi[t];return s&&s.indexOf(",")<0&&(t=s),t in _s&&t!==Wn&&(e._gsap.x||as(e,"x"))?n&&Im===n?t==="scale"?Rw:ww:(Im=n||{})&&(t==="scale"?Iw:Dw):e.style&&!Zd(e.style[t])?Tw:~t.indexOf("-")?Ew:ch(e,t)},core:{_removeProperty:Qs,_getMatrix:ph}};Xn.utils.checkPrefix=Uo;Xn.core.getStyleSaver=Cx;(function(i,e,t,n){var s=Vn(i+","+e+","+t,function(r){_s[r]=1});Vn(e,function(r){ni.units[r]="deg",Rx[r]=1}),Oi[s[13]]=i+","+e,Vn(n,function(r){var o=r.split(":");Oi[o[1]]=s[o[0]]})})("x,y,z,scale,scaleX,scaleY,xPercent,yPercent","rotation,rotationX,rotationY,skewX,skewY","transform,transformOrigin,svgOrigin,force3D,smoothOrigin,transformPerspective","0:translateX,1:translateY,2:translateZ,8:rotate,8:rotationZ,8:rotateZ,9:rotateX,10:rotateY");Vn("x,y,z,top,right,bottom,left,width,height,fontSize,padding,margin,perspective",function(i){ni.units[i]="px"});Xn.registerPlugin(Px);var pr=Xn.registerPlugin(Px)||Xn;pr.core.Tween;const Vw=(i,e)=>{const t=i.__vccOpts||i;for(const[n,s]of e)t[n]=s;return t},Gw={class:"top-hud"},Ww={class:"top-actions"},Xw={class:"cinematic-head"},qw={class:"cinematic-loop-toggle"},Yw={class:"cinematic-actions"},Qw=["disabled"],Kw={class:"cinematic-progress-row"},jw=["value"],$w={class:"cinematic-progress-row"},Zw={key:1,class:"fps-counter"},Jw={key:0,class:"loading-overlay"},e3={key:1,class:"error-overlay"},t3={class:"error-card"},n3={class:"error-msg"},i3=["min","max"],s3={class:"focal-row"},r3=["min","max"],o3={class:"focal-row"},a3={class:"focal-row"},l3={class:"camera-track-header"},c3={class:"camera-track-copy"},u3=["onClick"],f3=["src"],d3={key:1,class:"camera-tag-overlay"},h3={class:"camera-tag-text"},p3={key:2},m3=["src"],g3={key:0,class:"ref-info"},x3={class:"info-tag info-tag--accent"},_3={key:1,class:"ref-info"},v3={class:"info-tag"},A3={class:"info-tag"},S3={class:"info-tag"},Jr=380,zm=.065,km=.0022,Fl=.08,y3=1,Hm=.0055,Vm=.0042,b3=1,M3=.35,C3={__name:"GaussianViewer",setup(i){const e=Dt(null),t=Dt(!1),n=Dt(!1),s=Dt(!1),r={FREE:"free",ORBIT:"orbit"},o=Dt(r.FREE),a=Dt([]),l=Dt(""),c=Dt(""),u=Dt(""),f=Dt({}),d=Dt({x:0,y:0,z:0}),h=Dt({x:0,y:0,z:0}),x=Dt(""),p=Dt(0),g=Dt(!1),m=Dt(0),_=Dt(0),v=Dt(null),A=Dt(1),S=Dt(0),b=Dt(!0),M=Dt(!1),R=Dt(!1),y=fr(()=>o.value===r.ORBIT),C=fr(()=>{if(!l.value.trim()){const E=a.value.filter(Q=>Q.tag);return E.length>0?E:a.value.slice(0,60)}const w=l.value.trim().toLowerCase();return a.value.filter(E=>E.tag&&E.tag.toLowerCase().includes(w))}),F=()=>{C.value.length>0?ye(C.value[0]):alert("场景中没有找到符合该描述的视角哦~")};let I,O;const z=new U(0,1,0);let H=null,V=!1,q=!1,G=0;const Y={trajectory:null,startTimeMs:0,elapsedMs:0,lastNearestPoseIndex:-1},ae=Dt({x:0,y:0}),Ae=fr(()=>a.value.length>=2),Pe=fr(()=>M.value?"暂停运镜":R.value?"继续运镜":"开始运镜"),Ne=(w,E)=>!w||!E?null:2*Math.atan(E/2/w)*(180/Math.PI),We=(w,E)=>{if(!w||!E)return null;const Q=w*Math.PI/180/2;return Q<=0?null:E/2/Math.tan(Q)},He=()=>{if(!I||!I.camera)return;const w=f.value.h||e.value?.clientHeight||window.innerHeight;if(m.value=Number(I.camera.fov||0),w&&m.value>0&&m.value<179){const E=We(m.value,w);_.value=E?Number(E.toFixed(1)):0}},J=(w,E={})=>{if(!I||!I.camera)return;const Q=f.value.h||e.value?.clientHeight||window.innerHeight;if(!Q||!w)return;const ue=Ne(w,Q);if(!ue||!Number.isFinite(ue))return;const pe=I.camera,xe=E.duration??0;if(xe>0)pr.to(pe,{fov:ue,duration:xe,ease:E.ease||"power2.out",onUpdate:()=>{pe.updateProjectionMatrix();try{I.update(),I.render()}catch{}He()}});else{pe.fov=ue,pe.updateProjectionMatrix();try{I.update(),I.render()}catch{}He()}},le=w=>Number.isFinite(w)?Math.min(Qe.value,Math.max(Ce.value,w)):null,be=()=>{const w=Number(v.value||_.value||f.value.fl_y||Jr);return le(w)},ze=w=>{if(!I||!I.camera||!Number.isFinite(w)||w<=0)return;const E=be();if(!E)return;const Q=le(E*w);Q&&(v.value=Number(Q.toFixed(1)),J(Q))},Ce=fr(()=>{const w=Number(f.value.fl_y||0);return w>0?Math.max(50,Math.floor(w*.4)):50}),Qe=fr(()=>{const w=Number(f.value.fl_y||0);return w>0?Math.max(500,Math.ceil(w*2.5)):3e3}),B=()=>{g.value=!g.value,g.value&&!v.value&&(v.value=Number((_.value||f.value.fl_y||Jr).toFixed(1)))},N=()=>{const w=Number(v.value);!Number.isFinite(w)||w<=0||J(w)},j=()=>{const w=Number(f.value.fl_y||0);w&&(v.value=Number(w.toFixed(1)),J(w,{duration:.5,ease:"power2.inOut"}))},P=()=>{if(!I||!I.camera)return;const w=new Ti().setFromQuaternion(I.camera.quaternion,"YXZ");d.value={x:(w.x*180/Math.PI).toFixed(1),y:(w.y*180/Math.PI).toFixed(1),z:(w.z*180/Math.PI).toFixed(1)},He()},ie=()=>T.uCenter.value.clone(),re=()=>{G&&(cancelAnimationFrame(G),G=0)},fe=()=>{!I||!I.camera||(pr.killTweensOf(I.camera.position),pr.killTweensOf(I.camera.quaternion),pr.killTweensOf(I.camera))},ne=w=>{c.value=w?.image_url||Be(w),u.value=w?.tag||""},he=(w={})=>{re(),Y.trajectory=null,Y.startTimeMs=0,Y.elapsedMs=0,Y.lastNearestPoseIndex=-1,M.value=!1,R.value=!1,w.resetProgress!==!1&&(S.value=0)},te=()=>{!M.value&&!R.value||he({resetProgress:!1})},ge={FLY_IN:0,DIFFUSION:1,COLORING:2,FINISHED:3},D={isLoaded:!1,lastFrameTime:0,phase:ge.FLY_IN,flyDuration:1.5,diffusionDuration:1,colorDuration:4},T={uTime:{value:0},uCenter:{value:new U(0,0,0)},uGeoRadius:{value:0},uColorRadius:{value:0},uMaxRadius:{value:50},uParticleProgress:{value:0}},X=w=>{if(!I)return;const E=w.getSplatCount();w.updateMatrixWorld();let Q=1/0,ue=1/0,pe=1/0,xe=-1/0,Ee=-1/0,we=-1/0;const _e=new U,Le=Math.max(1,Math.floor(E/1e3));for(let qt=0;qt<E;qt+=Le)w.getSplatCenter(qt,_e),_e.applyMatrix4(w.matrixWorld),_e.x<Q&&(Q=_e.x),_e.x>xe&&(xe=_e.x),_e.y<ue&&(ue=_e.y),_e.y>Ee&&(Ee=_e.y),_e.z<pe&&(pe=_e.z),_e.z>we&&(we=_e.z);const Ve=(Q+xe)/2,tt=(ue+Ee)/2,rt=(pe+we)/2,Ke=Math.max(xe-Q,Ee-ue,we-pe);T.uCenter.value.set(Ve,tt,rt),T.uMaxRadius.value=Ke*.7;let Ge=6e4;E<4e4?Ge=E:E>1e6&&(Ge=4e5);const ut=Math.ceil(E/Ge);let ot=Ke/200*window.devicePixelRatio;ot<.5&&(ot=.5);const Mn=Ke*1;console.log(`[Adaptive] MaxDim: ${Ke.toFixed(2)}, Particles: ~${Math.floor(E/ut)}, Size: ${ot.toFixed(2)}`);const _i=new Bn,fn=[],Ss=[],wt=[];for(let qt=0;qt<E;qt+=ut){w.getSplatCenter(qt,_e),_e.applyMatrix4(w.matrixWorld),Ss.push(_e.x,_e.y,_e.z);const Zt=Mn+Math.random()*(Ke*.5),on=Math.random()*Math.PI*2,Js=Math.acos(2*Math.random()-1),qi=Ve+Zt*Math.sin(Js)*Math.cos(on),Fx=tt+Zt*Math.sin(Js)*Math.sin(on),Lx=rt+Zt*Math.cos(Js);fn.push(qi,Fx,Lx),wt.push(Math.random())}_i.setAttribute("position",new An(fn,3)),_i.setAttribute("aTarget",new An(Ss,3)),_i.setAttribute("aRandom",new An(wt,1));const dn=new Ln({uniforms:{uProgress:T.uParticleProgress,uSize:{value:ot},uColor:{value:new lt(.6,.6,.6)}},vertexShader:`
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
    `,transparent:!0,opacity:1,depthTest:!0,depthWrite:!1});O=new KS(_i,dn),O.frustumCulled=!1,I.threeScene.add(O)},se=w=>{if(!w||!w.material)return;const E=w.material;E.uniforms=E.uniforms||{},E.uniforms.uGeoRadius=T.uGeoRadius,E.uniforms.uColorRadius=T.uColorRadius,E.uniforms.uMaxRadius=T.uMaxRadius,E.uniforms.uCenter=T.uCenter,E.vertexShader=`varying vec3 vWorldPosition;
`+E.vertexShader;const Q=E.vertexShader.lastIndexOf("}");if(Q!==-1){const xe=`vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
`;E.vertexShader=E.vertexShader.substring(0,Q)+xe+"}"}const ue=`
    uniform float uGeoRadius;
    uniform float uColorRadius;
    uniform float uMaxRadius;
    uniform vec3 uCenter;
    varying vec3 vWorldPosition;
  `;E.fragmentShader=ue+E.fragmentShader;const pe=E.fragmentShader.lastIndexOf("}");if(pe!==-1){const xe=E.fragmentShader.substring(0,pe),Ee=`
      float distFromCenter = distance(vWorldPosition, uCenter);
      
      if (distFromCenter > uGeoRadius) {
          discard;
      }
      if (distFromCenter > uColorRadius) {
          if (gl_FragColor.a < 0.8) discard; 
          gl_FragColor.a = 1.0; 
          gl_FragColor.rgb = vec3(0.6, 0.6, 0.6);
      } 
    `;E.fragmentShader=xe+Ee+"}"}E.needsUpdate=!0},ce=w=>{if(!Array.isArray(w))return null;if(w.length===16){const E=w.map(Q=>Number(Q));return E.every(Number.isFinite)?E:null}if(w.length===4&&w.every(E=>Array.isArray(E)&&E.length===4)){const E=w.flat().map(Q=>Number(Q));return E.every(Number.isFinite)?E:null}return null},Z=w=>{if(w==null)return"";let E=String(w).trim();if(!E)return"";try{E=decodeURIComponent(E)}catch{}E=E.replace(/\\/g,"/");const Q=E.split("/");return(Q[Q.length-1]||"").trim().toLowerCase()},Be=w=>{if(!w)return"";const E=w.id||w.image_id||w.imageId;if(E)return Z(E);const Q=w.image_url;if(typeof Q!="string"||Q.length===0)return"";const ue=Q.split("?")[0];return Z(ue)},Se=w=>{if(!w||a.value.length===0)return null;const E=Z(w.imageId);if(E){const xe=a.value.find(Ee=>Be(Ee)===E);if(xe)return xe}const Q=ce(w.matrix);if(!Q)return null;let ue=null,pe=Number.POSITIVE_INFINITY;for(const xe of a.value){const Ee=ce(xe.matrix);if(!Ee)continue;let we=0;for(let _e=0;_e<16;_e+=1){const Le=Math.abs(Ee[_e]-Q[_e]);if(Le>we&&(we=Le),we>=pe)break}we<pe&&(pe=we,ue=xe)}return pe<=1e-4?ue:null},Oe=(w=!1)=>{if(!H||V)return;const E=Se(H);if(E){V=!0,ye(E);return}if(!w||H.imageId&&!q)return;const Q=ce(H.matrix);Q&&(V=!0,ye({matrix:Q,image_url:H.imageId||""}))},k=w=>{if(!I||!I.camera)return null;const E=ce(w?.matrix);if(!E)return null;const Q=I.getSplatMesh(),ue=new $e().fromArray(E),pe=new $e;Q?(Q.updateMatrixWorld(),pe.copy(Q.matrixWorld).multiply(ue)):pe.copy(ue);const xe=new U,Ee=new Ft,we=new U;return pe.decompose(xe,Ee,we),{position:xe,quaternion:Ee,fl_y:Number(w?.fl_y||f.value.fl_y||0),h:Number(w?.h||f.value.h||0)}},ee=()=>{if(!I||a.value.length<2)return null;const w=a.value.map((_e,Le)=>{const Ve=k(_e);return Ve?{index:Le,pose:_e,position:Ve.position,quaternion:Ve.quaternion,fl_y:Ve.fl_y,h:Ve.h}:null}).filter(Boolean);if(w.length<2)return null;const E=[w[0]];for(let _e=1;_e<w.length;_e+=1){const Le=E[E.length-1],Ve=w[_e],tt=Le.position.distanceToSquared(Ve.position)<1e-6,rt=Math.abs(Le.quaternion.dot(Ve.quaternion))>.999999;tt&&rt||E.push(Ve)}if(E.length<2)return null;const Q=E.map(_e=>_e.position.clone()),ue=new $S(Q,!1,"centripetal"),pe=[0];for(let _e=1;_e<E.length;_e+=1){const Le=E[_e-1],Ve=E[_e];pe.push(pe[_e-1]+Le.position.distanceTo(Ve.position))}const xe=pe[pe.length-1],Ee=E.length-1,we=Ds.clamp(Ee*220,6e3,45e3)/A.value;return{keyframes:E,curve:ue,cumulativeDistances:pe,totalDistance:Math.max(xe,1e-5),durationMs:we}},me=(w,E)=>{if(!w)return null;const Q=Ds.clamp(E,0,1),ue=w.totalDistance*Q;let pe=w.keyframes.length-2;for(let rt=0;rt<w.cumulativeDistances.length-1;rt+=1)if(ue<=w.cumulativeDistances[rt+1]){pe=rt;break}const xe=w.cumulativeDistances[pe],Ee=w.cumulativeDistances[pe+1],we=Math.max(Ee-xe,1e-5),_e=Ds.smootherstep((ue-xe)/we,0,1),Le=w.keyframes[pe],Ve=w.keyframes[pe+1],tt=Le.quaternion.clone().slerp(Ve.quaternion,_e);return{position:w.curve.getPointAt(Q),quaternion:tt,fl_y:Le.fl_y&&Ve.fl_y?Ds.lerp(Le.fl_y,Ve.fl_y,_e):Le.fl_y||Ve.fl_y||0,h:Le.h||Ve.h||f.value.h||0,nearestPoseIndex:_e<.5?Le.index:Ve.index}},Te=w=>{if(!w||!I||!I.camera)return;const E=I.camera;if(E.position.copy(w.position),E.quaternion.copy(w.quaternion),w.fl_y&&w.h?(f.value.h=w.h,v.value=Number(w.fl_y.toFixed(1)),J(w.fl_y)):bn(),w.nearestPoseIndex!==Y.lastNearestPoseIndex){Y.lastNearestPoseIndex=w.nearestPoseIndex;const Q=a.value[w.nearestPoseIndex];Q&&ne(Q)}},Fe=w=>{if(!Y.trajectory||!I||!I.camera){he({resetProgress:!1});return}const E=Math.max(Y.trajectory.durationMs,1),Q=Math.max(0,w-Y.startTimeMs);Y.elapsedMs=Q;let ue=Q/E;if(ue>=1&&(b.value?(Y.startTimeMs=w,Y.elapsedMs=0,Y.lastNearestPoseIndex=-1,ue=0):ue=1),S.value=ue,Te(me(Y.trajectory,ue)),!b.value&&ue>=1){he({resetProgress:!1}),S.value=1;return}G=requestAnimationFrame(Fe)},Re=(w={})=>{if(!I||!I.camera)return;const E=ee();E&&(fe(),re(),Y.trajectory=E,Y.elapsedMs=w.resume?Y.elapsedMs:0,Y.startTimeMs=performance.now()-Y.elapsedMs,Y.lastNearestPoseIndex=-1,M.value=!0,R.value=!1,w.resume||(S.value=0,Te(me(E,0))),G=requestAnimationFrame(Fe))},Ye=()=>{M.value&&(re(),Y.elapsedMs=Math.max(0,performance.now()-Y.startTimeMs),M.value=!1,R.value=!0)},W=()=>{if(Ae.value){if(M.value){Ye();return}Re({resume:R.value})}},De=()=>{A.value=Number(Ds.clamp(Number(A.value)||1,.25,3).toFixed(2)),M.value?Re():R.value&&(Y.trajectory=ee())},ye=(w,E={})=>{if(!I||!I.camera)return;const Q=k(w);if(!Q){console.warn("[Viewer] Skip invalid pose matrix:",w);return}E.keepCinematic||te();const ue=I.camera,pe=Q.position,xe=Q.quaternion;ne(w);const Ee=Q.fl_y,we=Q.h;Ee&&we&&(f.value.h=we,v.value=Number(Ee.toFixed(1)),J(Ee,{duration:1.5,ease:"power3.inOut"})),ue.near>.001&&(ue.near=.001,ue.updateProjectionMatrix()),t.value=!1,I.controls&&(I.controls.enabled=!1);const _e=ue.position.clone(),Le=ue.quaternion.clone(),Ve={t:0};fe(),pr.killTweensOf(Ve),pr.to(Ve,{t:1,duration:1.5,ease:"power3.inOut",onUpdate:()=>{ue.position.lerpVectors(_e,pe,Ve.t),ue.quaternion.slerpQuaternions(Le,xe,Ve.t)},onComplete:()=>{const tt=new Ti().setFromQuaternion(ue.quaternion,"YXZ");h.value={x:(tt.x*180/Math.PI).toFixed(1),y:(tt.y*180/Math.PI).toFixed(1),z:(tt.z*180/Math.PI).toFixed(1)},ae.value={x:0,y:0},rn.roll=0,P(),I.controls&&(I.controls.enabled=!0)}})},Me=()=>{const w=/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);return{rootElement:e.value,cameraUp:[0,1,0],initialCameraPosition:[0,0,5],initialCameraLookAt:[0,0,0],useBuiltInControls:!1,gpuAcceleratedSort:!1,webXRMode:dr.None,sharedMemoryForWorkers:!1,antialiased:!w}};let ve="/models/scene_auto_sync.ply",de="/models/webgl_poses_with_tags.json",ke=!1;const je=()=>{const w=new URLSearchParams(window.location.search),E=w.get("payload");if(E)try{const we=JSON.parse(decodeURIComponent(E));return{ply:we.ply||null,poses:we.poses||null,matrix:we.matrix||null,imageId:we.imageId||null}}catch(we){console.warn("[Viewer] 无法解析 payload 查询参数:",we)}const Q=w.get("ply"),ue=w.get("poses"),pe=w.get("matrix"),xe=w.get("imageId");let Ee=null;if(pe)try{Ee=JSON.parse(decodeURIComponent(pe))}catch(we){console.warn("[Viewer] 无法解析 matrix 查询参数:",we)}return Q||ue||Ee?{ply:Q||null,poses:ue||null,matrix:Ee,imageId:xe||null}:null},ct=async(w,E,Q)=>{if(!n.value){n.value=!0,he(),w&&(ve=w),E&&(de=E);try{I&&(I.renderer.setAnimationLoop(null),I.dispose&&await I.dispose(),I=null),e.value&&(e.value.innerHTML=""),D.isLoaded=!1,D.phase=ge.FLY_IN,T.uParticleProgress.value=0,T.uGeoRadius.value=0,T.uColorRadius.value=0,H=null,V=!1,q=!1;const ue=Me();I=new io(ue),window.viewer=I,v.value=Jr,console.log(`[Viewer] 加载模型: ${ve}`),await I.addSplatScene(ve,{showLoadingUI:!0,progressiveLoad:!1,rotation:[0,0,0,1]}),n.value=!1,window.BrainDanceChannel&&window.BrainDanceChannel.postMessage(JSON.stringify({status:"success",msg:"模型加载完成"})),console.log(`[Viewer] 加载位姿: ${de}`),fetch(de).then(Le=>Le.json()).then(Le=>{q=!0,Le.frames?(f.value={w:Le.w,h:Le.h,fl_x:Le.fl_x,fl_y:Le.fl_y},v.value=Number((Le.fl_y||0).toFixed(1)),a.value=Le.frames.map(Ve=>{let tt=Ve.image_url;if(tt&&!tt.startsWith("http")&&de.startsWith("http")){const rt=de.substring(0,de.lastIndexOf("/"));let Ke=tt;const Ge=Ke.indexOf("images/");Ge!==-1?Ke=Ke.substring(Ge):Ke.startsWith("/models/")?Ke=Ke.substring(8):Ke.startsWith("/")&&(Ke=Ke.substring(1)),tt=`${rt}/${Ke}`}return{id:Ve.id,matrix:Ve.matrix,image_url:tt,tag:Ve.tag,fl_x:Ve.fl_x,fl_y:Ve.fl_y,w:Ve.w||Le.w,h:Ve.h||Le.h}}),f.value.fl_y&&f.value.h?J(f.value.fl_y):J(Jr),Oe(!0)):(a.value=Le,J(Jr),Oe(!0))}).catch(Le=>{q=!0,console.error("加载位姿失败:",Le),J(Jr),Oe(!0)});const pe=I.getSplatMesh();pe.visible=!1,setTimeout(()=>{pe&&(X(pe),se(pe),Q&&(Q.matrix||Q.imageId)&&(H={matrix:Q.matrix||null,imageId:Q.imageId||null},Oe(q),setTimeout(()=>{Oe(!1)},50),Q.imageId||setTimeout(()=>{Oe(!0)},800)),D.lastFrameTime=Date.now(),D.startTime=Date.now(),D.isLoaded=!0)},200);let xe=performance.now();const Ee=1e3/120;let we=0,_e=performance.now();I.renderer.setAnimationLoop(()=>{const Le=performance.now(),Ve=Le-xe;if(Ve<Ee||(xe=Le-Ve%Ee,I.update(),I.render(),we++,Le-_e>=1e3&&(p.value=we,we=0,_e=Le),!D.isLoaded||D.phase===ge.FINISHED))return;const tt=Date.now(),rt=(tt-D.lastFrameTime)/1e3||.016;if(D.lastFrameTime=tt,D.phase===ge.FLY_IN){const Ke=1/D.flyDuration;let Ge=T.uParticleProgress.value+rt*Ke;if(Ge>=1.2){Ge=1.2;const ut=I.getSplatMesh();ut&&(ut.visible=!0),D.phase=ge.DIFFUSION,D.diffuseTime=0}T.uParticleProgress.value=Ge}else if(D.phase===ge.DIFFUSION){D.diffuseTime+=rt;const Ke=Math.min(D.diffuseTime/D.diffusionDuration,1),Ge=T.uMaxRadius.value;T.uGeoRadius.value=Ke*(Ge*1.5),O&&O.material&&(O.material.opacity=1-Ke),Ke>=1&&(O&&(O.visible=!1),T.uGeoRadius.value=99999,D.phase=ge.COLORING,D.colorStartTime=tt)}else if(D.phase===ge.COLORING){const Ke=(tt-D.colorStartTime)/1e3,Ge=T.uMaxRadius.value,ut=Ke/D.colorDuration;T.uColorRadius.value=ut*(Ge*1.5),ut>=1&&(D.phase=ge.FINISHED,T.uColorRadius.value=99999)}}),Vo()}catch(ue){console.error("error:",ue),x.value=ue&&(ue.message||String(ue))||"模型加载失败，请检查模型 URL 是否正确可访问"}finally{n.value=!1}}},ht=()=>{!I||!I.controls||(I.controls.dispose(),I.controls=null)},bn=()=>{if(!(!I||!I.camera)){I.camera.updateProjectionMatrix(),He(),P();try{I.update(),I.render()}catch{}}},Un=(w,E)=>{!I||!I.camera||(I.camera.rotateOnWorldAxis(z,-w),I.camera.rotateX(-E),bn())},Pc=w=>{!I||!I.camera||!Number.isFinite(w)||(I.camera.rotateZ(w*b3),bn())},ko=w=>{if(!I||!I.camera||!Number.isFinite(w)||w<=0)return;const E=Math.max(.3,I.camera.position.distanceTo(ie())),Q=Ds.clamp((1-w)*E*M3,-E*.25,E*.25);I.camera.translateZ(Q),bn()},Dr=(w,E)=>Math.atan2(E.clientY-w.clientY,E.clientX-w.clientX),Xi=w=>w>Math.PI?w-Math.PI*2:w<-Math.PI?w+Math.PI*2:w,Ho=()=>{I&&ht()},Ka=()=>{I&&(ht(),rn.roll=0)},Vo=()=>{I&&(y.value?Ka():Ho())},Zs=w=>{w!==r.FREE&&w!==r.ORBIT||o.value!==w&&(o.value=w,Vo(),y.value)},ja=()=>{const w=window.location.hostname==="localhost"||window.location.hostname==="127.0.0.1",E=window.location.protocol==="https:";s.value=w||E},It=Dt(!1),pt={x:0,y:0},mt={active:!1,distance:0},rn={active:!1,angle:0,roll:0},As=(w,E)=>{const Q=w.clientX-E.clientX,ue=w.clientY-E.clientY;return Math.hypot(Q,ue)},Go=w=>{if(te(),y.value){if(w.button!==0)return;It.value=!0,mt.active=!1,rn.active=!1,pt.x=w.clientX,pt.y=w.clientY;return}It.value=!0,mt.active=!1,pt.x=w.clientX,pt.y=w.clientY},Wo=w=>{if(y.value){if(!It.value||!I||!I.camera)return;const pe=w.clientX-pt.x,xe=w.clientY-pt.y;Un(pe*Hm,xe*Vm),pt.x=w.clientX,pt.y=w.clientY;return}if(!It.value||!I||!I.camera)return;const E=w.clientX-pt.x,ue=(w.clientY-pt.y)*zm;I.camera.rotateX(ue*Math.PI/180),I.camera.translateX(-E*km),I.camera.updateProjectionMatrix(),P(),pt.x=w.clientX,pt.y=w.clientY},Pr=()=>{if(y.value){It.value=!1,mt.active=!1,rn.active=!1;return}It.value=!1,mt.active=!1},Fc=w=>{if(!I||!I.camera)return;if(te(),y.value){const Q=w.deltaY<0?1+Fl:1/(1+Fl);ko(Q);return}const E=w.deltaY<0?1+Fl:1/(1+Fl);ze(E)},L=w=>{if(te(),y.value){if(w.touches.length>=2){It.value=!1,mt.active=!0,mt.distance=As(w.touches[0],w.touches[1]),rn.active=!0,rn.angle=Dr(w.touches[0],w.touches[1]);return}mt.active=!1,rn.active=!1,w.touches.length===1&&(It.value=!0,pt.x=w.touches[0].clientX,pt.y=w.touches[0].clientY);return}if(w.touches.length>=2){It.value=!1,mt.active=!0,mt.distance=As(w.touches[0],w.touches[1]);return}mt.active=!1,w.touches.length===1&&(It.value=!0,pt.x=w.touches[0].clientX,pt.y=w.touches[0].clientY)},K=w=>{if(y.value){if(!I||!I.camera||w.touches.length===0)return;if(w.touches.length>=2){const Ee=As(w.touches[0],w.touches[1]),we=Dr(w.touches[0],w.touches[1]);mt.active&&mt.distance>0&&Ee>0&&ko(Ee/mt.distance),rn.active&&Pc(Xi(we-rn.angle)),mt.active=!0,mt.distance=Ee,rn.active=!0,rn.angle=we,It.value=!1;return}if(!It.value)return;const pe=w.touches[0].clientX-pt.x,xe=w.touches[0].clientY-pt.y;Un(pe*Hm,xe*Vm),pt.x=w.touches[0].clientX,pt.y=w.touches[0].clientY;return}if(!I||!I.camera||w.touches.length===0)return;if(w.touches.length>=2){const pe=As(w.touches[0],w.touches[1]);if(mt.active&&mt.distance>0&&pe>0){const xe=pe/mt.distance;ze(1+(xe-1)*y3)}mt.active=!0,mt.distance=pe,It.value=!1;return}if(!It.value)return;const E=w.touches[0].clientX-pt.x,ue=(w.touches[0].clientY-pt.y)*zm;ae.value.x+=ue,I.camera.rotateX(ue*Math.PI/180),I.camera.translateX(-E*km),I.camera.updateProjectionMatrix(),P(),pt.x=w.touches[0].clientX,pt.y=w.touches[0].clientY},oe=w=>{if(y.value){if(w.touches.length>=2){mt.active=!0,mt.distance=As(w.touches[0],w.touches[1]),rn.active=!0,rn.angle=Dr(w.touches[0],w.touches[1]),It.value=!1;return}mt.active=!1,mt.distance=0,rn.active=!1,rn.angle=0,It.value=!1,w.touches.length===1&&(pt.x=w.touches[0].clientX,pt.y=w.touches[0].clientY,It.value=!0);return}if(w.touches.length>=2){mt.active=!0,mt.distance=As(w.touches[0],w.touches[1]),It.value=!1;return}mt.active=!1,mt.distance=0,It.value=!1,w.touches.length===1&&(pt.x=w.touches[0].clientX,pt.y=w.touches[0].clientY,It.value=!0)};return M0(()=>{if(e.value){if(ja(),window.loadModelFromFlutter=w=>{console.log("[Flutter->WebGL] 收到加载请求:",w),typeof w=="string"?ct(w,null,null):typeof w=="object"&&w!==null?ct(w.ply||null,w.poses||null,{matrix:w.matrix||null,imageId:w.imageId||null}):ct(null,null,null)},window.BrainDanceChannel)window.BrainDanceChannel.postMessage(JSON.stringify({status:"ready"}));else{const w=je();w&&!ke?(ke=!0,ct(w.ply,w.poses,{matrix:w.matrix||null,imageId:w.imageId||null})):ct(null,null)}window.addEventListener("mousedown",Go),window.addEventListener("mousemove",Wo),window.addEventListener("mouseup",Pr)}}),C0(async()=>{window.removeEventListener("mousedown",Go),window.removeEventListener("mousemove",Wo),window.removeEventListener("mouseup",Pr),he(),I&&(I.renderer.setAnimationLoop(null),await I.dispose())}),(w,E)=>(an(),gn("div",{class:"app-container",onMousedown:Go,onMousemove:Wo,onMouseup:Pr,onWheel:At(Fc,["prevent"]),onMouseleave:Pr,onTouchstart:L,onTouchmove:At(K,["prevent"]),onTouchend:oe,onTouchcancel:oe},[qe("div",{ref_key:"containerRef",ref:e,class:"viewer-container"},null,512),E[49]||(E[49]=qe("div",{class:"viewer-vignette"},null,-1)),qe("div",Gw,[qe("div",{class:"search-panel archive-card",onMousedown:E[1]||(E[1]=At(()=>{},["stop"])),onTouchstart:E[2]||(E[2]=At(()=>{},["stop"])),onTouchmove:E[3]||(E[3]=At(()=>{},["stop"])),onTouchend:E[4]||(E[4]=At(()=>{},["stop"]))},[qo(qe("input",{type:"text","onUpdate:modelValue":E[0]||(E[0]=Q=>l.value=Q),onKeyup:sA(F,["enter"]),placeholder:"例如：门口、桌面左侧、正面特写",class:"search-input"},null,544),[[el,l.value]]),qe("button",{onClick:F,class:"archive-btn archive-btn--solid search-btn"},"检索视角")],32),qe("div",Ww,[qe("div",{class:"view-mode-switch archive-card",onMousedown:E[7]||(E[7]=At(()=>{},["stop"])),onTouchstart:E[8]||(E[8]=At(()=>{},["stop"])),onTouchmove:E[9]||(E[9]=At(()=>{},["stop"])),onTouchend:E[10]||(E[10]=At(()=>{},["stop"]))},[qe("button",{class:lo(["mode-chip",{active:o.value===r.FREE}]),onClick:E[5]||(E[5]=Q=>Zs(r.FREE))}," 自由模式 ",2),qe("button",{class:lo(["mode-chip",{active:o.value===r.ORBIT}]),onClick:E[6]||(E[6]=Q=>Zs(r.ORBIT))}," Orbit 模式 ",2)],32),qe("button",{class:"archive-btn archive-btn--ghost focal-settings-toggle",onClick:B,onMousedown:E[11]||(E[11]=At(()=>{},["stop"])),onTouchstart:E[12]||(E[12]=At(()=>{},["stop"])),onTouchend:E[13]||(E[13]=At(()=>{},["stop"]))},mn(g.value?"收起焦距":"焦距设置"),33),Ae.value?(an(),gn("div",{key:0,class:"cinematic-panel archive-card",onMousedown:E[17]||(E[17]=At(()=>{},["stop"])),onTouchstart:E[18]||(E[18]=At(()=>{},["stop"])),onTouchmove:E[19]||(E[19]=At(()=>{},["stop"])),onTouchend:E[20]||(E[20]=At(()=>{},["stop"])),onTouchcancel:E[21]||(E[21]=At(()=>{},["stop"]))},[qe("div",Xw,[E[36]||(E[36]=qe("div",null,[qe("div",{class:"eyebrow"},"Camera Move"),qe("div",{class:"cinematic-title"},"自动运镜")],-1)),qe("label",qw,[qo(qe("input",{type:"checkbox","onUpdate:modelValue":E[14]||(E[14]=Q=>b.value=Q)},null,512),[[Jv,b.value]]),E[35]||(E[35]=qe("span",null,"循环",-1))])]),qe("div",Yw,[qe("button",{class:"archive-btn archive-btn--solid cinematic-primary",onClick:W},mn(Pe.value),1),qe("button",{class:"archive-btn archive-btn--ghost cinematic-secondary",onClick:E[15]||(E[15]=Q=>he()),disabled:!M.value&&!R.value&&S.value===0}," 停止 ",8,Qw)]),qe("div",Kw,[E[37]||(E[37]=qe("span",null,"进度",-1)),qe("span",null,mn(Math.round(S.value*100))+"%",1)]),qe("input",{class:"cinematic-progress",type:"range",value:S.value*100,min:"0",max:"100",step:"1",disabled:""},null,8,jw),qe("div",$w,[E[38]||(E[38]=qe("span",null,"速度",-1)),qe("span",null,mn(A.value.toFixed(2))+"x",1)]),qo(qe("input",{class:"cinematic-speed",type:"range","onUpdate:modelValue":E[16]||(E[16]=Q=>A.value=Q),min:"0.25",max:"3",step:"0.05",onInput:De},null,544),[[el,A.value,void 0,{number:!0}]])],32)):ri("",!0),p.value>0?(an(),gn("div",Zw,"FPS "+mn(p.value),1)):ri("",!0)])]),n.value?(an(),gn("div",Jw,[...E[39]||(E[39]=[qe("div",{class:"loading-card"},[qe("div",{class:"loading-dot"}),qe("div",{class:"loading-title"},"场景正在展开"),qe("div",{class:"loading-copy"},"模型与参考镜头正在同步到工作台。")],-1)])])):ri("",!0),x.value?(an(),gn("div",e3,[qe("div",t3,[E[40]||(E[40]=qe("div",{class:"eyebrow"},"Load Failed",-1)),E[41]||(E[41]=qe("div",{class:"error-title"},"模型未能正常打开",-1)),qe("div",n3,mn(x.value),1),qe("button",{class:"archive-btn archive-btn--solid",onClick:E[22]||(E[22]=Q=>ct(Xu(ve),Xu(de),null))}," 重新载入 ")])])):ri("",!0),ri("",!0),g.value?(an(),gn("div",{key:3,class:"focal-settings-panel",onMousedown:E[25]||(E[25]=At(()=>{},["stop"])),onTouchstart:E[26]||(E[26]=At(()=>{},["stop"])),onTouchmove:E[27]||(E[27]=At(()=>{},["stop"])),onTouchend:E[28]||(E[28]=At(()=>{},["stop"])),onTouchcancel:E[29]||(E[29]=At(()=>{},["stop"]))},[E[43]||(E[43]=qe("div",{class:"eyebrow"},"Lens Control",-1)),E[44]||(E[44]=qe("div",{class:"focal-title"},"镜头焦距",-1)),qo(qe("input",{type:"range","onUpdate:modelValue":E[23]||(E[23]=Q=>v.value=Q),min:Ce.value,max:Qe.value,step:"1",onInput:N},null,40,i3),[[el,v.value,void 0,{number:!0}]]),qe("div",s3,[qo(qe("input",{class:"focal-number-input",type:"number","onUpdate:modelValue":E[24]||(E[24]=Q=>v.value=Q),min:Ce.value,max:Qe.value,step:"1",onChange:N},null,40,r3),[[el,v.value,void 0,{number:!0}]]),E[42]||(E[42]=qe("span",null,"px",-1))]),qe("div",o3,[qe("span",null,"当前 FOV: "+mn(m.value.toFixed(1))+"°",1)]),qe("div",a3,[qe("span",null,"当前焦距: "+mn(_.value.toFixed(1))+" px",1)]),qe("button",{class:"archive-btn archive-btn--solid focal-reset-btn",onClick:j},"恢复拍摄焦距")],32)):ri("",!0),!y.value&&C.value.length>0?(an(),gn("div",{key:4,class:"camera-track",onMousedown:E[30]||(E[30]=At(()=>{},["stop"])),onTouchstart:E[31]||(E[31]=At(()=>{},["stop"])),onTouchmove:E[32]||(E[32]=At(()=>{},["stop"])),onTouchend:E[33]||(E[33]=At(()=>{},["stop"]))},[qe("div",l3,[E[45]||(E[45]=qe("div",{class:"eyebrow"},"Shot Strip",-1)),qe("div",c3,mn(l.value?"按当前检索结果排序":"优先显示已打标签镜头"),1)]),(an(!0),gn(Ii,null,W_(C.value,(Q,ue)=>(an(),gn("div",{key:Q.id,class:lo(["camera-btn",{active:c.value===Q.image_url}]),onClick:At(pe=>ye(Q),["stop"])},[Q.image_url?(an(),gn("img",{key:0,src:Q.image_url,class:"btn-thumb"},null,8,f3)):ri("",!0),Q.tag?(an(),gn("div",d3,[qe("div",h3,mn(Q.tag),1)])):Q.image_url?ri("",!0):(an(),gn("span",p3,"未命名视角"))],10,u3))),128))],32)):ri("",!0),c.value?(an(),gn("div",{key:5,class:"reference-overlay",onClick:E[34]||(E[34]=Q=>{c.value="",u.value=""})},[E[46]||(E[46]=qe("div",{class:"eyebrow"},"Reference Still",-1)),E[47]||(E[47]=qe("div",{class:"ref-title"},"参考原图",-1)),qe("img",{src:c.value,class:"ref-img"},null,8,m3),u.value?(an(),gn("div",g3,[qe("span",x3,mn(u.value),1)])):ri("",!0),f.value.fl_y?(an(),gn("div",_3,[qe("span",v3,"焦距: "+mn(f.value.fl_y.toFixed(1))+" px",1),qe("span",A3,"FOV: "+mn((2*Math.atan(f.value.h/(2*f.value.fl_y))*(180/Math.PI)).toFixed(1))+"°",1),qe("span",S3,"分辨率: "+mn(f.value.w)+"x"+mn(f.value.h),1)])):ri("",!0),E[48]||(E[48]=qe("div",{class:"ref-hint"},"点击关闭对比",-1))])):ri("",!0)],32))}},T3=Vw(C3,[["__scopeId","data-v-23691244"]]),E3={__name:"App",setup(i){return(e,t)=>(an(),gn("main",null,[zi(T3)]))}};aA(E3).mount("#app");
