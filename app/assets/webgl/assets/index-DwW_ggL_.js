(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))n(s);new MutationObserver(s=>{for(const r of s)if(r.type==="childList")for(const o of r.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&n(o)}).observe(document,{childList:!0,subtree:!0});function t(s){const r={};return s.integrity&&(r.integrity=s.integrity),s.referrerPolicy&&(r.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?r.credentials="include":s.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function n(s){if(s.ep)return;s.ep=!0;const r=t(s);fetch(s.href,r)}})();function jf(i){const e=Object.create(null);for(const t of i.split(","))e[t]=1;return t=>t in e}const bt={},io=[],Hi=()=>{},Um=()=>!1,ic=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&(i.charCodeAt(2)>122||i.charCodeAt(2)<97),$f=i=>i.startsWith("onUpdate:"),mn=Object.assign,Zf=(i,e)=>{const t=i.indexOf(e);t>-1&&i.splice(t,1)},Ix=Object.prototype.hasOwnProperty,ht=(i,e)=>Ix.call(i,e),$e=Array.isArray,so=i=>Pa(i)==="[object Map]",Om=i=>Pa(i)==="[object Set]",cd=i=>Pa(i)==="[object Date]",Je=i=>typeof i=="function",jt=i=>typeof i=="string",Gi=i=>typeof i=="symbol",_t=i=>i!==null&&typeof i=="object",Nm=i=>(_t(i)||Je(i))&&Je(i.then)&&Je(i.catch),zm=Object.prototype.toString,Pa=i=>zm.call(i),Dx=i=>Pa(i).slice(8,-1),km=i=>Pa(i)==="[object Object]",Jf=i=>jt(i)&&i!=="NaN"&&i[0]!=="-"&&""+parseInt(i,10)===i,$o=jf(",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"),sc=i=>{const e=Object.create(null);return(t=>e[t]||(e[t]=i(t)))},Px=/-\w/g,Vs=sc(i=>i.replace(Px,e=>e.slice(1).toUpperCase())),Fx=/\B([A-Z])/g,Qs=sc(i=>i.replace(Fx,"-$1").toLowerCase()),Hm=sc(i=>i.charAt(0).toUpperCase()+i.slice(1)),Tc=sc(i=>i?`on${Hm(i)}`:""),Os=(i,e)=>!Object.is(i,e),bl=(i,...e)=>{for(let t=0;t<i.length;t++)i[t](...e)},Vm=(i,e,t,n=!1)=>{Object.defineProperty(i,e,{configurable:!0,enumerable:!1,writable:n,value:t})},eh=i=>{const e=parseFloat(i);return isNaN(e)?i:e};let ud;const rc=()=>ud||(ud=typeof globalThis<"u"?globalThis:typeof self<"u"?self:typeof window<"u"?window:typeof global<"u"?global:{});function th(i){if($e(i)){const e={};for(let t=0;t<i.length;t++){const n=i[t],s=jt(n)?Ox(n):th(n);if(s)for(const r in s)e[r]=s[r]}return e}else if(jt(i)||_t(i))return i}const Lx=/;(?![^(]*\))/g,Bx=/:([^]+)/,Ux=/\/\*[^]*?\*\//g;function Ox(i){const e={};return i.replace(Ux,"").split(Lx).forEach(t=>{if(t){const n=t.split(Bx);n.length>1&&(e[n[0].trim()]=n[1].trim())}}),e}function ro(i){let e="";if(jt(i))e=i;else if($e(i))for(let t=0;t<i.length;t++){const n=ro(i[t]);n&&(e+=n+" ")}else if(_t(i))for(const t in i)i[t]&&(e+=t+" ");return e.trim()}const Nx="itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly",zx=jf(Nx);function Gm(i){return!!i||i===""}function kx(i,e){if(i.length!==e.length)return!1;let t=!0;for(let n=0;t&&n<i.length;n++)t=nh(i[n],e[n]);return t}function nh(i,e){if(i===e)return!0;let t=cd(i),n=cd(e);if(t||n)return t&&n?i.getTime()===e.getTime():!1;if(t=Gi(i),n=Gi(e),t||n)return i===e;if(t=$e(i),n=$e(e),t||n)return t&&n?kx(i,e):!1;if(t=_t(i),n=_t(e),t||n){if(!t||!n)return!1;const s=Object.keys(i).length,r=Object.keys(e).length;if(s!==r)return!1;for(const o in i){const a=i.hasOwnProperty(o),l=e.hasOwnProperty(o);if(a&&!l||!a&&l||!nh(i[o],e[o]))return!1}}return String(i)===String(e)}const Wm=i=>!!(i&&i.__v_isRef===!0),Xn=i=>jt(i)?i:i==null?"":$e(i)||_t(i)&&(i.toString===zm||!Je(i.toString))?Wm(i)?Xn(i.value):JSON.stringify(i,Xm,2):String(i),Xm=(i,e)=>Wm(e)?Xm(i,e.value):so(e)?{[`Map(${e.size})`]:[...e.entries()].reduce((t,[n,s],r)=>(t[Cc(n,r)+" =>"]=s,t),{})}:Om(e)?{[`Set(${e.size})`]:[...e.values()].map(t=>Cc(t))}:Gi(e)?Cc(e):_t(e)&&!$e(e)&&!km(e)?String(e):e,Cc=(i,e="")=>{var t;return Gi(i)?`Symbol(${(t=i.description)!=null?t:e})`:i};let Fn;class Hx{constructor(e=!1){this.detached=e,this._active=!0,this._on=0,this.effects=[],this.cleanups=[],this._isPaused=!1,this.__v_skip=!0,this.parent=Fn,!e&&Fn&&(this.index=(Fn.scopes||(Fn.scopes=[])).push(this)-1)}get active(){return this._active}pause(){if(this._active){this._isPaused=!0;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].pause();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].pause()}}resume(){if(this._active&&this._isPaused){this._isPaused=!1;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].resume();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].resume()}}run(e){if(this._active){const t=Fn;try{return Fn=this,e()}finally{Fn=t}}}on(){++this._on===1&&(this.prevScope=Fn,Fn=this)}off(){this._on>0&&--this._on===0&&(Fn=this.prevScope,this.prevScope=void 0)}stop(e){if(this._active){this._active=!1;let t,n;for(t=0,n=this.effects.length;t<n;t++)this.effects[t].stop();for(this.effects.length=0,t=0,n=this.cleanups.length;t<n;t++)this.cleanups[t]();if(this.cleanups.length=0,this.scopes){for(t=0,n=this.scopes.length;t<n;t++)this.scopes[t].stop(!0);this.scopes.length=0}if(!this.detached&&this.parent&&!e){const s=this.parent.scopes.pop();s&&s!==this&&(this.parent.scopes[this.index]=s,s.index=this.index)}this.parent=void 0}}}function Vx(){return Fn}let Tt;const Ec=new WeakSet;class qm{constructor(e){this.fn=e,this.deps=void 0,this.depsTail=void 0,this.flags=5,this.next=void 0,this.cleanup=void 0,this.scheduler=void 0,Fn&&Fn.active&&Fn.effects.push(this)}pause(){this.flags|=64}resume(){this.flags&64&&(this.flags&=-65,Ec.has(this)&&(Ec.delete(this),this.trigger()))}notify(){this.flags&2&&!(this.flags&32)||this.flags&8||Qm(this)}run(){if(!(this.flags&1))return this.fn();this.flags|=2,fd(this),Km(this);const e=Tt,t=Ti;Tt=this,Ti=!0;try{return this.fn()}finally{jm(this),Tt=e,Ti=t,this.flags&=-3}}stop(){if(this.flags&1){for(let e=this.deps;e;e=e.nextDep)rh(e);this.deps=this.depsTail=void 0,fd(this),this.onStop&&this.onStop(),this.flags&=-2}}trigger(){this.flags&64?Ec.add(this):this.scheduler?this.scheduler():this.runIfDirty()}runIfDirty(){Fu(this)&&this.run()}get dirty(){return Fu(this)}}let Ym=0,Zo,Jo;function Qm(i,e=!1){if(i.flags|=8,e){i.next=Jo,Jo=i;return}i.next=Zo,Zo=i}function ih(){Ym++}function sh(){if(--Ym>0)return;if(Jo){let e=Jo;for(Jo=void 0;e;){const t=e.next;e.next=void 0,e.flags&=-9,e=t}}let i;for(;Zo;){let e=Zo;for(Zo=void 0;e;){const t=e.next;if(e.next=void 0,e.flags&=-9,e.flags&1)try{e.trigger()}catch(n){i||(i=n)}e=t}}if(i)throw i}function Km(i){for(let e=i.deps;e;e=e.nextDep)e.version=-1,e.prevActiveLink=e.dep.activeLink,e.dep.activeLink=e}function jm(i){let e,t=i.depsTail,n=t;for(;n;){const s=n.prevDep;n.version===-1?(n===t&&(t=s),rh(n),Gx(n)):e=n,n.dep.activeLink=n.prevActiveLink,n.prevActiveLink=void 0,n=s}i.deps=e,i.depsTail=t}function Fu(i){for(let e=i.deps;e;e=e.nextDep)if(e.dep.version!==e.version||e.dep.computed&&($m(e.dep.computed)||e.dep.version!==e.version))return!0;return!!i._dirty}function $m(i){if(i.flags&4&&!(i.flags&16)||(i.flags&=-17,i.globalVersion===ha)||(i.globalVersion=ha,!i.isSSR&&i.flags&128&&(!i.deps&&!i._dirty||!Fu(i))))return;i.flags|=2;const e=i.dep,t=Tt,n=Ti;Tt=i,Ti=!0;try{Km(i);const s=i.fn(i._value);(e.version===0||Os(s,i._value))&&(i.flags|=128,i._value=s,e.version++)}catch(s){throw e.version++,s}finally{Tt=t,Ti=n,jm(i),i.flags&=-3}}function rh(i,e=!1){const{dep:t,prevSub:n,nextSub:s}=i;if(n&&(n.nextSub=s,i.prevSub=void 0),s&&(s.prevSub=n,i.nextSub=void 0),t.subs===i&&(t.subs=n,!n&&t.computed)){t.computed.flags&=-5;for(let r=t.computed.deps;r;r=r.nextDep)rh(r,!0)}!e&&!--t.sc&&t.map&&t.map.delete(t.key)}function Gx(i){const{prevDep:e,nextDep:t}=i;e&&(e.nextDep=t,i.prevDep=void 0),t&&(t.prevDep=e,i.nextDep=void 0)}let Ti=!0;const Zm=[];function gs(){Zm.push(Ti),Ti=!1}function xs(){const i=Zm.pop();Ti=i===void 0?!0:i}function fd(i){const{cleanup:e}=i;if(i.cleanup=void 0,e){const t=Tt;Tt=void 0;try{e()}finally{Tt=t}}}let ha=0;class Wx{constructor(e,t){this.sub=e,this.dep=t,this.version=t.version,this.nextDep=this.prevDep=this.nextSub=this.prevSub=this.prevActiveLink=void 0}}class oh{constructor(e){this.computed=e,this.version=0,this.activeLink=void 0,this.subs=void 0,this.map=void 0,this.key=void 0,this.sc=0,this.__v_skip=!0}track(e){if(!Tt||!Ti||Tt===this.computed)return;let t=this.activeLink;if(t===void 0||t.sub!==Tt)t=this.activeLink=new Wx(Tt,this),Tt.deps?(t.prevDep=Tt.depsTail,Tt.depsTail.nextDep=t,Tt.depsTail=t):Tt.deps=Tt.depsTail=t,Jm(t);else if(t.version===-1&&(t.version=this.version,t.nextDep)){const n=t.nextDep;n.prevDep=t.prevDep,t.prevDep&&(t.prevDep.nextDep=n),t.prevDep=Tt.depsTail,t.nextDep=void 0,Tt.depsTail.nextDep=t,Tt.depsTail=t,Tt.deps===t&&(Tt.deps=n)}return t}trigger(e){this.version++,ha++,this.notify(e)}notify(e){ih();try{for(let t=this.subs;t;t=t.prevSub)t.sub.notify()&&t.sub.dep.notify()}finally{sh()}}}function Jm(i){if(i.dep.sc++,i.sub.flags&4){const e=i.dep.computed;if(e&&!i.dep.subs){e.flags|=20;for(let n=e.deps;n;n=n.nextDep)Jm(n)}const t=i.dep.subs;t!==i&&(i.prevSub=t,t&&(t.nextSub=i)),i.dep.subs=i}}const Lu=new WeakMap,_r=Symbol(""),Bu=Symbol(""),da=Symbol("");function un(i,e,t){if(Ti&&Tt){let n=Lu.get(i);n||Lu.set(i,n=new Map);let s=n.get(t);s||(n.set(t,s=new oh),s.map=n,s.key=t),s.track()}}function us(i,e,t,n,s,r){const o=Lu.get(i);if(!o){ha++;return}const a=l=>{l&&l.trigger()};if(ih(),e==="clear")o.forEach(a);else{const l=$e(i),c=l&&Jf(t);if(l&&t==="length"){const u=Number(n);o.forEach((f,h)=>{(h==="length"||h===da||!Gi(h)&&h>=u)&&a(f)})}else switch((t!==void 0||o.has(void 0))&&a(o.get(t)),c&&a(o.get(da)),e){case"add":l?c&&a(o.get("length")):(a(o.get(_r)),so(i)&&a(o.get(Bu)));break;case"delete":l||(a(o.get(_r)),so(i)&&a(o.get(Bu)));break;case"set":so(i)&&a(o.get(_r));break}}sh()}function Dr(i){const e=ft(i);return e===i?e:(un(e,"iterate",da),xi(i)?e:e.map(Ci))}function oc(i){return un(i=ft(i),"iterate",da),i}function Rs(i,e){return _s(i)?xo(vr(i)?Ci(e):e):Ci(e)}const Xx={__proto__:null,[Symbol.iterator](){return wc(this,Symbol.iterator,i=>Rs(this,i))},concat(...i){return Dr(this).concat(...i.map(e=>$e(e)?Dr(e):e))},entries(){return wc(this,"entries",i=>(i[1]=Rs(this,i[1]),i))},every(i,e){return Qi(this,"every",i,e,void 0,arguments)},filter(i,e){return Qi(this,"filter",i,e,t=>t.map(n=>Rs(this,n)),arguments)},find(i,e){return Qi(this,"find",i,e,t=>Rs(this,t),arguments)},findIndex(i,e){return Qi(this,"findIndex",i,e,void 0,arguments)},findLast(i,e){return Qi(this,"findLast",i,e,t=>Rs(this,t),arguments)},findLastIndex(i,e){return Qi(this,"findLastIndex",i,e,void 0,arguments)},forEach(i,e){return Qi(this,"forEach",i,e,void 0,arguments)},includes(...i){return Rc(this,"includes",i)},indexOf(...i){return Rc(this,"indexOf",i)},join(i){return Dr(this).join(i)},lastIndexOf(...i){return Rc(this,"lastIndexOf",i)},map(i,e){return Qi(this,"map",i,e,void 0,arguments)},pop(){return Oo(this,"pop")},push(...i){return Oo(this,"push",i)},reduce(i,...e){return hd(this,"reduce",i,e)},reduceRight(i,...e){return hd(this,"reduceRight",i,e)},shift(){return Oo(this,"shift")},some(i,e){return Qi(this,"some",i,e,void 0,arguments)},splice(...i){return Oo(this,"splice",i)},toReversed(){return Dr(this).toReversed()},toSorted(i){return Dr(this).toSorted(i)},toSpliced(...i){return Dr(this).toSpliced(...i)},unshift(...i){return Oo(this,"unshift",i)},values(){return wc(this,"values",i=>Rs(this,i))}};function wc(i,e,t){const n=oc(i),s=n[e]();return n!==i&&!xi(i)&&(s._next=s.next,s.next=()=>{const r=s._next();return r.done||(r.value=t(r.value)),r}),s}const qx=Array.prototype;function Qi(i,e,t,n,s,r){const o=oc(i),a=o!==i&&!xi(i),l=o[e];if(l!==qx[e]){const f=l.apply(i,r);return a?Ci(f):f}let c=t;o!==i&&(a?c=function(f,h){return t.call(this,Rs(i,f),h,i)}:t.length>2&&(c=function(f,h){return t.call(this,f,h,i)}));const u=l.call(o,c,n);return a&&s?s(u):u}function hd(i,e,t,n){const s=oc(i);let r=t;return s!==i&&(xi(i)?t.length>3&&(r=function(o,a,l){return t.call(this,o,a,l,i)}):r=function(o,a,l){return t.call(this,o,Rs(i,a),l,i)}),s[e](r,...n)}function Rc(i,e,t){const n=ft(i);un(n,"iterate",da);const s=n[e](...t);return(s===-1||s===!1)&&uh(t[0])?(t[0]=ft(t[0]),n[e](...t)):s}function Oo(i,e,t=[]){gs(),ih();const n=ft(i)[e].apply(i,t);return sh(),xs(),n}const Yx=jf("__proto__,__v_isRef,__isVue"),e0=new Set(Object.getOwnPropertyNames(Symbol).filter(i=>i!=="arguments"&&i!=="caller").map(i=>Symbol[i]).filter(Gi));function Qx(i){Gi(i)||(i=String(i));const e=ft(this);return un(e,"has",i),e.hasOwnProperty(i)}class t0{constructor(e=!1,t=!1){this._isReadonly=e,this._isShallow=t}get(e,t,n){if(t==="__v_skip")return e.__v_skip;const s=this._isReadonly,r=this._isShallow;if(t==="__v_isReactive")return!s;if(t==="__v_isReadonly")return s;if(t==="__v_isShallow")return r;if(t==="__v_raw")return n===(s?r?s_:r0:r?s0:i0).get(e)||Object.getPrototypeOf(e)===Object.getPrototypeOf(n)?e:void 0;const o=$e(e);if(!s){let l;if(o&&(l=Xx[t]))return l;if(t==="hasOwnProperty")return Qx}const a=Reflect.get(e,t,hn(e)?e:n);if((Gi(t)?e0.has(t):Yx(t))||(s||un(e,"get",t),r))return a;if(hn(a)){const l=o&&Jf(t)?a:a.value;return s&&_t(l)?Ou(l):l}return _t(a)?s?Ou(a):lh(a):a}}class n0 extends t0{constructor(e=!1){super(!1,e)}set(e,t,n,s){let r=e[t];const o=$e(e)&&Jf(t);if(!this._isShallow){const c=_s(r);if(!xi(n)&&!_s(n)&&(r=ft(r),n=ft(n)),!o&&hn(r)&&!hn(n))return c||(r.value=n),!0}const a=o?Number(t)<e.length:ht(e,t),l=Reflect.set(e,t,n,hn(e)?e:s);return e===ft(s)&&(a?Os(n,r)&&us(e,"set",t,n):us(e,"add",t,n)),l}deleteProperty(e,t){const n=ht(e,t);e[t];const s=Reflect.deleteProperty(e,t);return s&&n&&us(e,"delete",t,void 0),s}has(e,t){const n=Reflect.has(e,t);return(!Gi(t)||!e0.has(t))&&un(e,"has",t),n}ownKeys(e){return un(e,"iterate",$e(e)?"length":_r),Reflect.ownKeys(e)}}class Kx extends t0{constructor(e=!1){super(!0,e)}set(e,t){return!0}deleteProperty(e,t){return!0}}const jx=new n0,$x=new Kx,Zx=new n0(!0);const Uu=i=>i,Va=i=>Reflect.getPrototypeOf(i);function Jx(i,e,t){return function(...n){const s=this.__v_raw,r=ft(s),o=so(r),a=i==="entries"||i===Symbol.iterator&&o,l=i==="keys"&&o,c=s[i](...n),u=t?Uu:e?xo:Ci;return!e&&un(r,"iterate",l?Bu:_r),mn(Object.create(c),{next(){const{value:f,done:h}=c.next();return h?{value:f,done:h}:{value:a?[u(f[0]),u(f[1])]:u(f),done:h}}})}}function Ga(i){return function(...e){return i==="delete"?!1:i==="clear"?void 0:this}}function e_(i,e){const t={get(s){const r=this.__v_raw,o=ft(r),a=ft(s);i||(Os(s,a)&&un(o,"get",s),un(o,"get",a));const{has:l}=Va(o),c=e?Uu:i?xo:Ci;if(l.call(o,s))return c(r.get(s));if(l.call(o,a))return c(r.get(a));r!==o&&r.get(s)},get size(){const s=this.__v_raw;return!i&&un(ft(s),"iterate",_r),s.size},has(s){const r=this.__v_raw,o=ft(r),a=ft(s);return i||(Os(s,a)&&un(o,"has",s),un(o,"has",a)),s===a?r.has(s):r.has(s)||r.has(a)},forEach(s,r){const o=this,a=o.__v_raw,l=ft(a),c=e?Uu:i?xo:Ci;return!i&&un(l,"iterate",_r),a.forEach((u,f)=>s.call(r,c(u),c(f),o))}};return mn(t,i?{add:Ga("add"),set:Ga("set"),delete:Ga("delete"),clear:Ga("clear")}:{add(s){!e&&!xi(s)&&!_s(s)&&(s=ft(s));const r=ft(this);return Va(r).has.call(r,s)||(r.add(s),us(r,"add",s,s)),this},set(s,r){!e&&!xi(r)&&!_s(r)&&(r=ft(r));const o=ft(this),{has:a,get:l}=Va(o);let c=a.call(o,s);c||(s=ft(s),c=a.call(o,s));const u=l.call(o,s);return o.set(s,r),c?Os(r,u)&&us(o,"set",s,r):us(o,"add",s,r),this},delete(s){const r=ft(this),{has:o,get:a}=Va(r);let l=o.call(r,s);l||(s=ft(s),l=o.call(r,s)),a&&a.call(r,s);const c=r.delete(s);return l&&us(r,"delete",s,void 0),c},clear(){const s=ft(this),r=s.size!==0,o=s.clear();return r&&us(s,"clear",void 0,void 0),o}}),["keys","values","entries",Symbol.iterator].forEach(s=>{t[s]=Jx(s,i,e)}),t}function ah(i,e){const t=e_(i,e);return(n,s,r)=>s==="__v_isReactive"?!i:s==="__v_isReadonly"?i:s==="__v_raw"?n:Reflect.get(ht(t,s)&&s in n?t:n,s,r)}const t_={get:ah(!1,!1)},n_={get:ah(!1,!0)},i_={get:ah(!0,!1)};const i0=new WeakMap,s0=new WeakMap,r0=new WeakMap,s_=new WeakMap;function r_(i){switch(i){case"Object":case"Array":return 1;case"Map":case"Set":case"WeakMap":case"WeakSet":return 2;default:return 0}}function o_(i){return i.__v_skip||!Object.isExtensible(i)?0:r_(Dx(i))}function lh(i){return _s(i)?i:ch(i,!1,jx,t_,i0)}function a_(i){return ch(i,!1,Zx,n_,s0)}function Ou(i){return ch(i,!0,$x,i_,r0)}function ch(i,e,t,n,s){if(!_t(i)||i.__v_raw&&!(e&&i.__v_isReactive))return i;const r=o_(i);if(r===0)return i;const o=s.get(i);if(o)return o;const a=new Proxy(i,r===2?n:t);return s.set(i,a),a}function vr(i){return _s(i)?vr(i.__v_raw):!!(i&&i.__v_isReactive)}function _s(i){return!!(i&&i.__v_isReadonly)}function xi(i){return!!(i&&i.__v_isShallow)}function uh(i){return i?!!i.__v_raw:!1}function ft(i){const e=i&&i.__v_raw;return e?ft(e):i}function l_(i){return!ht(i,"__v_skip")&&Object.isExtensible(i)&&Vm(i,"__v_skip",!0),i}const Ci=i=>_t(i)?lh(i):i,xo=i=>_t(i)?Ou(i):i;function hn(i){return i?i.__v_isRef===!0:!1}function Xt(i){return c_(i,!1)}function c_(i,e){return hn(i)?i:new u_(i,e)}class u_{constructor(e,t){this.dep=new oh,this.__v_isRef=!0,this.__v_isShallow=!1,this._rawValue=t?e:ft(e),this._value=t?e:Ci(e),this.__v_isShallow=t}get value(){return this.dep.track(),this._value}set value(e){const t=this._rawValue,n=this.__v_isShallow||xi(e)||_s(e);e=n?e:ft(e),Os(e,t)&&(this._rawValue=e,this._value=n?e:Ci(e),this.dep.trigger())}}function Nu(i){return hn(i)?i.value:i}const f_={get:(i,e,t)=>e==="__v_raw"?i:Nu(Reflect.get(i,e,t)),set:(i,e,t,n)=>{const s=i[e];return hn(s)&&!hn(t)?(s.value=t,!0):Reflect.set(i,e,t,n)}};function o0(i){return vr(i)?i:new Proxy(i,f_)}class h_{constructor(e,t,n){this.fn=e,this.setter=t,this._value=void 0,this.dep=new oh(this),this.__v_isRef=!0,this.deps=void 0,this.depsTail=void 0,this.flags=16,this.globalVersion=ha-1,this.next=void 0,this.effect=this,this.__v_isReadonly=!t,this.isSSR=n}notify(){if(this.flags|=16,!(this.flags&8)&&Tt!==this)return Qm(this,!0),!0}get value(){const e=this.dep.track();return $m(this),e&&(e.version=this.dep.version),this._value}set value(e){this.setter&&this.setter(e)}}function d_(i,e,t=!1){let n,s;return Je(i)?n=i:(n=i.get,s=i.set),new h_(n,s,t)}const Wa={},Ul=new WeakMap;let ur;function p_(i,e=!1,t=ur){if(t){let n=Ul.get(t);n||Ul.set(t,n=[]),n.push(i)}}function m_(i,e,t=bt){const{immediate:n,deep:s,once:r,scheduler:o,augmentJob:a,call:l}=t,c=A=>s?A:xi(A)||s===!1||s===0?fs(A,1):fs(A);let u,f,h,d,x=!1,p=!1;if(hn(i)?(f=()=>i.value,x=xi(i)):vr(i)?(f=()=>c(i),x=!0):$e(i)?(p=!0,x=i.some(A=>vr(A)||xi(A)),f=()=>i.map(A=>{if(hn(A))return A.value;if(vr(A))return c(A);if(Je(A))return l?l(A,2):A()})):Je(i)?e?f=l?()=>l(i,2):i:f=()=>{if(h){gs();try{h()}finally{xs()}}const A=ur;ur=u;try{return l?l(i,3,[d]):i(d)}finally{ur=A}}:f=Hi,e&&s){const A=f,y=s===!0?1/0:s;f=()=>fs(A(),y)}const g=Vx(),m=()=>{u.stop(),g&&g.active&&Zf(g.effects,u)};if(r&&e){const A=e;e=(...y)=>{A(...y),m()}}let _=p?new Array(i.length).fill(Wa):Wa;const S=A=>{if(!(!(u.flags&1)||!u.dirty&&!A))if(e){const y=u.run();if(s||x||(p?y.some((b,v)=>Os(b,_[v])):Os(y,_))){h&&h();const b=ur;ur=u;try{const v=[y,_===Wa?void 0:p&&_[0]===Wa?[]:_,d];_=y,l?l(e,3,v):e(...v)}finally{ur=b}}}else u.run()};return a&&a(S),u=new qm(f),u.scheduler=o?()=>o(S,!1):S,d=A=>p_(A,!1,u),h=u.onStop=()=>{const A=Ul.get(u);if(A){if(l)l(A,4);else for(const y of A)y();Ul.delete(u)}},e?n?S(!0):_=u.run():o?o(S.bind(null,!0),!0):u.run(),m.pause=u.pause.bind(u),m.resume=u.resume.bind(u),m.stop=m,m}function fs(i,e=1/0,t){if(e<=0||!_t(i)||i.__v_skip||(t=t||new Map,(t.get(i)||0)>=e))return i;if(t.set(i,e),e--,hn(i))fs(i.value,e,t);else if($e(i))for(let n=0;n<i.length;n++)fs(i[n],e,t);else if(Om(i)||so(i))i.forEach(n=>{fs(n,e,t)});else if(km(i)){for(const n in i)fs(i[n],e,t);for(const n of Object.getOwnPropertySymbols(i))Object.prototype.propertyIsEnumerable.call(i,n)&&fs(i[n],e,t)}return i}function Fa(i,e,t,n){try{return n?i(...n):i()}catch(s){ac(s,e,t)}}function Wi(i,e,t,n){if(Je(i)){const s=Fa(i,e,t,n);return s&&Nm(s)&&s.catch(r=>{ac(r,e,t)}),s}if($e(i)){const s=[];for(let r=0;r<i.length;r++)s.push(Wi(i[r],e,t,n));return s}}function ac(i,e,t,n=!0){const s=e?e.vnode:null,{errorHandler:r,throwUnhandledErrorInProduction:o}=e&&e.appContext.config||bt;if(e){let a=e.parent;const l=e.proxy,c=`https://vuejs.org/error-reference/#runtime-${t}`;for(;a;){const u=a.ec;if(u){for(let f=0;f<u.length;f++)if(u[f](i,l,c)===!1)return}a=a.parent}if(r){gs(),Fa(r,null,10,[i,l,c]),xs();return}}g_(i,t,s,n,o)}function g_(i,e,t,n=!0,s=!1){if(s)throw i;console.error(i)}const Sn=[];let Di=-1;const oo=[];let Is=null,jr=0;const a0=Promise.resolve();let Ol=null;function x_(i){const e=Ol||a0;return i?e.then(this?i.bind(this):i):e}function __(i){let e=Di+1,t=Sn.length;for(;e<t;){const n=e+t>>>1,s=Sn[n],r=pa(s);r<i||r===i&&s.flags&2?e=n+1:t=n}return e}function fh(i){if(!(i.flags&1)){const e=pa(i),t=Sn[Sn.length-1];!t||!(i.flags&2)&&e>=pa(t)?Sn.push(i):Sn.splice(__(e),0,i),i.flags|=1,l0()}}function l0(){Ol||(Ol=a0.then(u0))}function v_(i){$e(i)?oo.push(...i):Is&&i.id===-1?Is.splice(jr+1,0,i):i.flags&1||(oo.push(i),i.flags|=1),l0()}function dd(i,e,t=Di+1){for(;t<Sn.length;t++){const n=Sn[t];if(n&&n.flags&2){if(i&&n.id!==i.uid)continue;Sn.splice(t,1),t--,n.flags&4&&(n.flags&=-2),n(),n.flags&4||(n.flags&=-2)}}}function c0(i){if(oo.length){const e=[...new Set(oo)].sort((t,n)=>pa(t)-pa(n));if(oo.length=0,Is){Is.push(...e);return}for(Is=e,jr=0;jr<Is.length;jr++){const t=Is[jr];t.flags&4&&(t.flags&=-2),t.flags&8||t(),t.flags&=-2}Is=null,jr=0}}const pa=i=>i.id==null?i.flags&2?-1:1/0:i.id;function u0(i){try{for(Di=0;Di<Sn.length;Di++){const e=Sn[Di];e&&!(e.flags&8)&&(e.flags&4&&(e.flags&=-2),Fa(e,e.i,e.i?15:14),e.flags&4||(e.flags&=-2))}}finally{for(;Di<Sn.length;Di++){const e=Sn[Di];e&&(e.flags&=-2)}Di=-1,Sn.length=0,c0(),Ol=null,(Sn.length||oo.length)&&u0()}}let hi=null,f0=null;function Nl(i){const e=hi;return hi=i,f0=i&&i.type.__scopeId||null,e}function S_(i,e=hi,t){if(!e||i._n)return i;const n=(...s)=>{n._d&&Md(-1);const r=Nl(e);let o;try{o=i(...s)}finally{Nl(r),n._d&&Md(1)}return o};return n._n=!0,n._c=!0,n._d=!0,n}function Ic(i,e){if(hi===null)return i;const t=fc(hi),n=i.dirs||(i.dirs=[]);for(let s=0;s<e.length;s++){let[r,o,a,l=bt]=e[s];r&&(Je(r)&&(r={mounted:r,updated:r}),r.deep&&fs(o),n.push({dir:r,instance:t,value:o,oldValue:void 0,arg:a,modifiers:l}))}return i}function er(i,e,t,n){const s=i.dirs,r=e&&e.dirs;for(let o=0;o<s.length;o++){const a=s[o];r&&(a.oldValue=r[o].value);let l=a.dir[n];l&&(gs(),Wi(l,t,8,[i.el,a,i,e]),xs())}}function A_(i,e){if(yn){let t=yn.provides;const n=yn.parent&&yn.parent.provides;n===t&&(t=yn.provides=Object.create(n)),t[i]=e}}function Ml(i,e,t=!1){const n=Av();if(n||ao){let s=ao?ao._context.provides:n?n.parent==null||n.ce?n.vnode.appContext&&n.vnode.appContext.provides:n.parent.provides:void 0;if(s&&i in s)return s[i];if(arguments.length>1)return t&&Je(e)?e.call(n&&n.proxy):e}}const y_=Symbol.for("v-scx"),b_=()=>Ml(y_);function Dc(i,e,t){return h0(i,e,t)}function h0(i,e,t=bt){const{immediate:n,deep:s,flush:r,once:o}=t,a=mn({},t),l=e&&n||!e&&r!=="post";let c;if(ga){if(r==="sync"){const d=b_();c=d.__watcherHandles||(d.__watcherHandles=[])}else if(!l){const d=()=>{};return d.stop=Hi,d.resume=Hi,d.pause=Hi,d}}const u=yn;a.call=(d,x,p)=>Wi(d,u,x,p);let f=!1;r==="post"?a.scheduler=d=>{Pn(d,u&&u.suspense)}:r!=="sync"&&(f=!0,a.scheduler=(d,x)=>{x?d():fh(d)}),a.augmentJob=d=>{e&&(d.flags|=4),f&&(d.flags|=2,u&&(d.id=u.uid,d.i=u))};const h=m_(i,e,a);return ga&&(c?c.push(h):l&&h()),h}function M_(i,e,t){const n=this.proxy,s=jt(i)?i.includes(".")?d0(n,i):()=>n[i]:i.bind(n,n);let r;Je(e)?r=e:(r=e.handler,t=e);const o=La(this),a=h0(s,r.bind(n),t);return o(),a}function d0(i,e){const t=e.split(".");return()=>{let n=i;for(let s=0;s<t.length&&n;s++)n=n[t[s]];return n}}const T_=Symbol("_vte"),C_=i=>i.__isTeleport,E_=Symbol("_leaveCb");function hh(i,e){i.shapeFlag&6&&i.component?(i.transition=e,hh(i.component.subTree,e)):i.shapeFlag&128?(i.ssContent.transition=e.clone(i.ssContent),i.ssFallback.transition=e.clone(i.ssFallback)):i.transition=e}function p0(i){i.ids=[i.ids[0]+i.ids[2]+++"-",0,0]}function pd(i,e){let t;return!!((t=Object.getOwnPropertyDescriptor(i,e))&&!t.configurable)}const zl=new WeakMap;function ea(i,e,t,n,s=!1){if($e(i)){i.forEach((p,g)=>ea(p,e&&($e(e)?e[g]:e),t,n,s));return}if(ta(n)&&!s){n.shapeFlag&512&&n.type.__asyncResolved&&n.component.subTree.component&&ea(i,e,t,n.component.subTree);return}const r=n.shapeFlag&4?fc(n.component):n.el,o=s?null:r,{i:a,r:l}=i,c=e&&e.r,u=a.refs===bt?a.refs={}:a.refs,f=a.setupState,h=ft(f),d=f===bt?Um:p=>pd(u,p)?!1:ht(h,p),x=(p,g)=>!(g&&pd(u,g));if(c!=null&&c!==l){if(md(e),jt(c))u[c]=null,d(c)&&(f[c]=null);else if(hn(c)){const p=e;x(c,p.k)&&(c.value=null),p.k&&(u[p.k]=null)}}if(Je(l))Fa(l,a,12,[o,u]);else{const p=jt(l),g=hn(l);if(p||g){const m=()=>{if(i.f){const _=p?d(l)?f[l]:u[l]:x()||!i.k?l.value:u[i.k];if(s)$e(_)&&Zf(_,r);else if($e(_))_.includes(r)||_.push(r);else if(p)u[l]=[r],d(l)&&(f[l]=u[l]);else{const S=[r];x(l,i.k)&&(l.value=S),i.k&&(u[i.k]=S)}}else p?(u[l]=o,d(l)&&(f[l]=o)):g&&(x(l,i.k)&&(l.value=o),i.k&&(u[i.k]=o))};if(o){const _=()=>{m(),zl.delete(i)};_.id=-1,zl.set(i,_),Pn(_,t)}else md(i),m()}}}function md(i){const e=zl.get(i);e&&(e.flags|=8,zl.delete(i))}rc().requestIdleCallback;rc().cancelIdleCallback;const ta=i=>!!i.type.__asyncLoader,m0=i=>i.type.__isKeepAlive;function w_(i,e){g0(i,"a",e)}function R_(i,e){g0(i,"da",e)}function g0(i,e,t=yn){const n=i.__wdc||(i.__wdc=()=>{let s=t;for(;s;){if(s.isDeactivated)return;s=s.parent}return i()});if(lc(e,n,t),t){let s=t.parent;for(;s&&s.parent;)m0(s.parent.vnode)&&I_(n,e,t,s),s=s.parent}}function I_(i,e,t,n){const s=lc(e,i,n,!0);v0(()=>{Zf(n[e],s)},t)}function lc(i,e,t=yn,n=!1){if(t){const s=t[i]||(t[i]=[]),r=e.__weh||(e.__weh=(...o)=>{gs();const a=La(t),l=Wi(e,t,i,o);return a(),xs(),l});return n?s.unshift(r):s.push(r),r}}const As=i=>(e,t=yn)=>{(!ga||i==="sp")&&lc(i,(...n)=>e(...n),t)},D_=As("bm"),x0=As("m"),P_=As("bu"),F_=As("u"),_0=As("bum"),v0=As("um"),L_=As("sp"),B_=As("rtg"),U_=As("rtc");function O_(i,e=yn){lc("ec",i,e)}const N_=Symbol.for("v-ndc");function z_(i,e,t,n){let s;const r=t,o=$e(i);if(o||jt(i)){const a=o&&vr(i);let l=!1,c=!1;a&&(l=!xi(i),c=_s(i),i=oc(i)),s=new Array(i.length);for(let u=0,f=i.length;u<f;u++)s[u]=e(l?c?xo(Ci(i[u])):Ci(i[u]):i[u],u,void 0,r)}else if(typeof i=="number"){s=new Array(i);for(let a=0;a<i;a++)s[a]=e(a+1,a,void 0,r)}else if(_t(i))if(i[Symbol.iterator])s=Array.from(i,(a,l)=>e(a,l,void 0,r));else{const a=Object.keys(i);s=new Array(a.length);for(let l=0,c=a.length;l<c;l++){const u=a[l];s[l]=e(i[u],u,l,r)}}else s=[];return s}const zu=i=>i?k0(i)?fc(i):zu(i.parent):null,na=mn(Object.create(null),{$:i=>i,$el:i=>i.vnode.el,$data:i=>i.data,$props:i=>i.props,$attrs:i=>i.attrs,$slots:i=>i.slots,$refs:i=>i.refs,$parent:i=>zu(i.parent),$root:i=>zu(i.root),$host:i=>i.ce,$emit:i=>i.emit,$options:i=>A0(i),$forceUpdate:i=>i.f||(i.f=()=>{fh(i.update)}),$nextTick:i=>i.n||(i.n=x_.bind(i.proxy)),$watch:i=>M_.bind(i)}),Pc=(i,e)=>i!==bt&&!i.__isScriptSetup&&ht(i,e),k_={get({_:i},e){if(e==="__v_skip")return!0;const{ctx:t,setupState:n,data:s,props:r,accessCache:o,type:a,appContext:l}=i;if(e[0]!=="$"){const h=o[e];if(h!==void 0)switch(h){case 1:return n[e];case 2:return s[e];case 4:return t[e];case 3:return r[e]}else{if(Pc(n,e))return o[e]=1,n[e];if(s!==bt&&ht(s,e))return o[e]=2,s[e];if(ht(r,e))return o[e]=3,r[e];if(t!==bt&&ht(t,e))return o[e]=4,t[e];ku&&(o[e]=0)}}const c=na[e];let u,f;if(c)return e==="$attrs"&&un(i.attrs,"get",""),c(i);if((u=a.__cssModules)&&(u=u[e]))return u;if(t!==bt&&ht(t,e))return o[e]=4,t[e];if(f=l.config.globalProperties,ht(f,e))return f[e]},set({_:i},e,t){const{data:n,setupState:s,ctx:r}=i;return Pc(s,e)?(s[e]=t,!0):n!==bt&&ht(n,e)?(n[e]=t,!0):ht(i.props,e)||e[0]==="$"&&e.slice(1)in i?!1:(r[e]=t,!0)},has({_:{data:i,setupState:e,accessCache:t,ctx:n,appContext:s,props:r,type:o}},a){let l;return!!(t[a]||i!==bt&&a[0]!=="$"&&ht(i,a)||Pc(e,a)||ht(r,a)||ht(n,a)||ht(na,a)||ht(s.config.globalProperties,a)||(l=o.__cssModules)&&l[a])},defineProperty(i,e,t){return t.get!=null?i._.accessCache[e]=0:ht(t,"value")&&this.set(i,e,t.value,null),Reflect.defineProperty(i,e,t)}};function gd(i){return $e(i)?i.reduce((e,t)=>(e[t]=null,e),{}):i}let ku=!0;function H_(i){const e=A0(i),t=i.proxy,n=i.ctx;ku=!1,e.beforeCreate&&xd(e.beforeCreate,i,"bc");const{data:s,computed:r,methods:o,watch:a,provide:l,inject:c,created:u,beforeMount:f,mounted:h,beforeUpdate:d,updated:x,activated:p,deactivated:g,beforeDestroy:m,beforeUnmount:_,destroyed:S,unmounted:A,render:y,renderTracked:b,renderTriggered:v,errorCaptured:E,serverPrefetch:M,expose:T,inheritAttrs:I,components:P,directives:B,filters:N}=e;if(c&&V_(c,n,null),o)for(const q in o){const X=o[q];Je(X)&&(n[q]=X.bind(t))}if(s){const q=s.call(t,t);_t(q)&&(i.data=lh(q))}if(ku=!0,r)for(const q in r){const X=r[q],ee=Je(X)?X.bind(t,t):Je(X.get)?X.get.bind(t,t):Hi,ce=!Je(X)&&Je(X.set)?X.set.bind(t):Hi,be=Yo({get:ee,set:ce});Object.defineProperty(n,q,{enumerable:!0,configurable:!0,get:()=>be.value,set:Re=>be.value=Re})}if(a)for(const q in a)S0(a[q],n,t,q);if(l){const q=Je(l)?l.call(t):l;Reflect.ownKeys(q).forEach(X=>{A_(X,q[X])})}u&&xd(u,i,"c");function V(q,X){$e(X)?X.forEach(ee=>q(ee.bind(t))):X&&q(X.bind(t))}if(V(D_,f),V(x0,h),V(P_,d),V(F_,x),V(w_,p),V(R_,g),V(O_,E),V(U_,b),V(B_,v),V(_0,_),V(v0,A),V(L_,M),$e(T))if(T.length){const q=i.exposed||(i.exposed={});T.forEach(X=>{Object.defineProperty(q,X,{get:()=>t[X],set:ee=>t[X]=ee,enumerable:!0})})}else i.exposed||(i.exposed={});y&&i.render===Hi&&(i.render=y),I!=null&&(i.inheritAttrs=I),P&&(i.components=P),B&&(i.directives=B),M&&p0(i)}function V_(i,e,t=Hi){$e(i)&&(i=Hu(i));for(const n in i){const s=i[n];let r;_t(s)?"default"in s?r=Ml(s.from||n,s.default,!0):r=Ml(s.from||n):r=Ml(s),hn(r)?Object.defineProperty(e,n,{enumerable:!0,configurable:!0,get:()=>r.value,set:o=>r.value=o}):e[n]=r}}function xd(i,e,t){Wi($e(i)?i.map(n=>n.bind(e.proxy)):i.bind(e.proxy),e,t)}function S0(i,e,t,n){let s=n.includes(".")?d0(t,n):()=>t[n];if(jt(i)){const r=e[i];Je(r)&&Dc(s,r)}else if(Je(i))Dc(s,i.bind(t));else if(_t(i))if($e(i))i.forEach(r=>S0(r,e,t,n));else{const r=Je(i.handler)?i.handler.bind(t):e[i.handler];Je(r)&&Dc(s,r,i)}}function A0(i){const e=i.type,{mixins:t,extends:n}=e,{mixins:s,optionsCache:r,config:{optionMergeStrategies:o}}=i.appContext,a=r.get(e);let l;return a?l=a:!s.length&&!t&&!n?l=e:(l={},s.length&&s.forEach(c=>kl(l,c,o,!0)),kl(l,e,o)),_t(e)&&r.set(e,l),l}function kl(i,e,t,n=!1){const{mixins:s,extends:r}=e;r&&kl(i,r,t,!0),s&&s.forEach(o=>kl(i,o,t,!0));for(const o in e)if(!(n&&o==="expose")){const a=G_[o]||t&&t[o];i[o]=a?a(i[o],e[o]):e[o]}return i}const G_={data:_d,props:vd,emits:vd,methods:qo,computed:qo,beforeCreate:xn,created:xn,beforeMount:xn,mounted:xn,beforeUpdate:xn,updated:xn,beforeDestroy:xn,beforeUnmount:xn,destroyed:xn,unmounted:xn,activated:xn,deactivated:xn,errorCaptured:xn,serverPrefetch:xn,components:qo,directives:qo,watch:X_,provide:_d,inject:W_};function _d(i,e){return e?i?function(){return mn(Je(i)?i.call(this,this):i,Je(e)?e.call(this,this):e)}:e:i}function W_(i,e){return qo(Hu(i),Hu(e))}function Hu(i){if($e(i)){const e={};for(let t=0;t<i.length;t++)e[i[t]]=i[t];return e}return i}function xn(i,e){return i?[...new Set([].concat(i,e))]:e}function qo(i,e){return i?mn(Object.create(null),i,e):e}function vd(i,e){return i?$e(i)&&$e(e)?[...new Set([...i,...e])]:mn(Object.create(null),gd(i),gd(e??{})):e}function X_(i,e){if(!i)return e;if(!e)return i;const t=mn(Object.create(null),i);for(const n in e)t[n]=xn(i[n],e[n]);return t}function y0(){return{app:null,config:{isNativeTag:Um,performance:!1,globalProperties:{},optionMergeStrategies:{},errorHandler:void 0,warnHandler:void 0,compilerOptions:{}},mixins:[],components:{},directives:{},provides:Object.create(null),optionsCache:new WeakMap,propsCache:new WeakMap,emitsCache:new WeakMap}}let q_=0;function Y_(i,e){return function(n,s=null){Je(n)||(n=mn({},n)),s!=null&&!_t(s)&&(s=null);const r=y0(),o=new WeakSet,a=[];let l=!1;const c=r.app={_uid:q_++,_component:n,_props:s,_container:null,_context:r,_instance:null,version:Ev,get config(){return r.config},set config(u){},use(u,...f){return o.has(u)||(u&&Je(u.install)?(o.add(u),u.install(c,...f)):Je(u)&&(o.add(u),u(c,...f))),c},mixin(u){return r.mixins.includes(u)||r.mixins.push(u),c},component(u,f){return f?(r.components[u]=f,c):r.components[u]},directive(u,f){return f?(r.directives[u]=f,c):r.directives[u]},mount(u,f,h){if(!l){const d=c._ceVNode||Vi(n,s);return d.appContext=r,h===!0?h="svg":h===!1&&(h=void 0),i(d,u,h),l=!0,c._container=u,u.__vue_app__=c,fc(d.component)}},onUnmount(u){a.push(u)},unmount(){l&&(Wi(a,c._instance,16),i(null,c._container),delete c._container.__vue_app__)},provide(u,f){return r.provides[u]=f,c},runWithContext(u){const f=ao;ao=c;try{return u()}finally{ao=f}}};return c}}let ao=null;const Q_=(i,e)=>e==="modelValue"||e==="model-value"?i.modelModifiers:i[`${e}Modifiers`]||i[`${Vs(e)}Modifiers`]||i[`${Qs(e)}Modifiers`];function K_(i,e,...t){if(i.isUnmounted)return;const n=i.vnode.props||bt;let s=t;const r=e.startsWith("update:"),o=r&&Q_(n,e.slice(7));o&&(o.trim&&(s=t.map(u=>jt(u)?u.trim():u)),o.number&&(s=t.map(eh)));let a,l=n[a=Tc(e)]||n[a=Tc(Vs(e))];!l&&r&&(l=n[a=Tc(Qs(e))]),l&&Wi(l,i,6,s);const c=n[a+"Once"];if(c){if(!i.emitted)i.emitted={};else if(i.emitted[a])return;i.emitted[a]=!0,Wi(c,i,6,s)}}const j_=new WeakMap;function b0(i,e,t=!1){const n=t?j_:e.emitsCache,s=n.get(i);if(s!==void 0)return s;const r=i.emits;let o={},a=!1;if(!Je(i)){const l=c=>{const u=b0(c,e,!0);u&&(a=!0,mn(o,u))};!t&&e.mixins.length&&e.mixins.forEach(l),i.extends&&l(i.extends),i.mixins&&i.mixins.forEach(l)}return!r&&!a?(_t(i)&&n.set(i,null),null):($e(r)?r.forEach(l=>o[l]=null):mn(o,r),_t(i)&&n.set(i,o),o)}function cc(i,e){return!i||!ic(e)?!1:(e=e.slice(2).replace(/Once$/,""),ht(i,e[0].toLowerCase()+e.slice(1))||ht(i,Qs(e))||ht(i,e))}function Sd(i){const{type:e,vnode:t,proxy:n,withProxy:s,propsOptions:[r],slots:o,attrs:a,emit:l,render:c,renderCache:u,props:f,data:h,setupState:d,ctx:x,inheritAttrs:p}=i,g=Nl(i);let m,_;try{if(t.shapeFlag&4){const A=s||n,y=A;m=Li(c.call(y,A,u,f,d,h,x)),_=a}else{const A=e;m=Li(A.length>1?A(f,{attrs:a,slots:o,emit:l}):A(f,null)),_=e.props?a:$_(a)}}catch(A){ia.length=0,ac(A,i,1),m=Vi(Gs)}let S=m;if(_&&p!==!1){const A=Object.keys(_),{shapeFlag:y}=S;A.length&&y&7&&(r&&A.some($f)&&(_=Z_(_,r)),S=_o(S,_,!1,!0))}return t.dirs&&(S=_o(S,null,!1,!0),S.dirs=S.dirs?S.dirs.concat(t.dirs):t.dirs),t.transition&&hh(S,t.transition),m=S,Nl(g),m}const $_=i=>{let e;for(const t in i)(t==="class"||t==="style"||ic(t))&&((e||(e={}))[t]=i[t]);return e},Z_=(i,e)=>{const t={};for(const n in i)(!$f(n)||!(n.slice(9)in e))&&(t[n]=i[n]);return t};function J_(i,e,t){const{props:n,children:s,component:r}=i,{props:o,children:a,patchFlag:l}=e,c=r.emitsOptions;if(e.dirs||e.transition)return!0;if(t&&l>=0){if(l&1024)return!0;if(l&16)return n?Ad(n,o,c):!!o;if(l&8){const u=e.dynamicProps;for(let f=0;f<u.length;f++){const h=u[f];if(M0(o,n,h)&&!cc(c,h))return!0}}}else return(s||a)&&(!a||!a.$stable)?!0:n===o?!1:n?o?Ad(n,o,c):!0:!!o;return!1}function Ad(i,e,t){const n=Object.keys(e);if(n.length!==Object.keys(i).length)return!0;for(let s=0;s<n.length;s++){const r=n[s];if(M0(e,i,r)&&!cc(t,r))return!0}return!1}function M0(i,e,t){const n=i[t],s=e[t];return t==="style"&&_t(n)&&_t(s)?!nh(n,s):n!==s}function ev({vnode:i,parent:e},t){for(;e;){const n=e.subTree;if(n.suspense&&n.suspense.activeBranch===i&&(n.el=i.el),n===i)(i=e.vnode).el=t,e=e.parent;else break}}const T0={},C0=()=>Object.create(T0),E0=i=>Object.getPrototypeOf(i)===T0;function tv(i,e,t,n=!1){const s={},r=C0();i.propsDefaults=Object.create(null),w0(i,e,s,r);for(const o in i.propsOptions[0])o in s||(s[o]=void 0);t?i.props=n?s:a_(s):i.type.props?i.props=s:i.props=r,i.attrs=r}function nv(i,e,t,n){const{props:s,attrs:r,vnode:{patchFlag:o}}=i,a=ft(s),[l]=i.propsOptions;let c=!1;if((n||o>0)&&!(o&16)){if(o&8){const u=i.vnode.dynamicProps;for(let f=0;f<u.length;f++){let h=u[f];if(cc(i.emitsOptions,h))continue;const d=e[h];if(l)if(ht(r,h))d!==r[h]&&(r[h]=d,c=!0);else{const x=Vs(h);s[x]=Vu(l,a,x,d,i,!1)}else d!==r[h]&&(r[h]=d,c=!0)}}}else{w0(i,e,s,r)&&(c=!0);let u;for(const f in a)(!e||!ht(e,f)&&((u=Qs(f))===f||!ht(e,u)))&&(l?t&&(t[f]!==void 0||t[u]!==void 0)&&(s[f]=Vu(l,a,f,void 0,i,!0)):delete s[f]);if(r!==a)for(const f in r)(!e||!ht(e,f))&&(delete r[f],c=!0)}c&&us(i.attrs,"set","")}function w0(i,e,t,n){const[s,r]=i.propsOptions;let o=!1,a;if(e)for(let l in e){if($o(l))continue;const c=e[l];let u;s&&ht(s,u=Vs(l))?!r||!r.includes(u)?t[u]=c:(a||(a={}))[u]=c:cc(i.emitsOptions,l)||(!(l in n)||c!==n[l])&&(n[l]=c,o=!0)}if(r){const l=ft(t),c=a||bt;for(let u=0;u<r.length;u++){const f=r[u];t[f]=Vu(s,l,f,c[f],i,!ht(c,f))}}return o}function Vu(i,e,t,n,s,r){const o=i[t];if(o!=null){const a=ht(o,"default");if(a&&n===void 0){const l=o.default;if(o.type!==Function&&!o.skipFactory&&Je(l)){const{propsDefaults:c}=s;if(t in c)n=c[t];else{const u=La(s);n=c[t]=l.call(null,e),u()}}else n=l;s.ce&&s.ce._setProp(t,n)}o[0]&&(r&&!a?n=!1:o[1]&&(n===""||n===Qs(t))&&(n=!0))}return n}const iv=new WeakMap;function R0(i,e,t=!1){const n=t?iv:e.propsCache,s=n.get(i);if(s)return s;const r=i.props,o={},a=[];let l=!1;if(!Je(i)){const u=f=>{l=!0;const[h,d]=R0(f,e,!0);mn(o,h),d&&a.push(...d)};!t&&e.mixins.length&&e.mixins.forEach(u),i.extends&&u(i.extends),i.mixins&&i.mixins.forEach(u)}if(!r&&!l)return _t(i)&&n.set(i,io),io;if($e(r))for(let u=0;u<r.length;u++){const f=Vs(r[u]);yd(f)&&(o[f]=bt)}else if(r)for(const u in r){const f=Vs(u);if(yd(f)){const h=r[u],d=o[f]=$e(h)||Je(h)?{type:h}:mn({},h),x=d.type;let p=!1,g=!0;if($e(x))for(let m=0;m<x.length;++m){const _=x[m],S=Je(_)&&_.name;if(S==="Boolean"){p=!0;break}else S==="String"&&(g=!1)}else p=Je(x)&&x.name==="Boolean";d[0]=p,d[1]=g,(p||ht(d,"default"))&&a.push(f)}}const c=[o,a];return _t(i)&&n.set(i,c),c}function yd(i){return i[0]!=="$"&&!$o(i)}const dh=i=>i==="_"||i==="_ctx"||i==="$stable",ph=i=>$e(i)?i.map(Li):[Li(i)],sv=(i,e,t)=>{if(e._n)return e;const n=S_((...s)=>ph(e(...s)),t);return n._c=!1,n},I0=(i,e,t)=>{const n=i._ctx;for(const s in i){if(dh(s))continue;const r=i[s];if(Je(r))e[s]=sv(s,r,n);else if(r!=null){const o=ph(r);e[s]=()=>o}}},D0=(i,e)=>{const t=ph(e);i.slots.default=()=>t},P0=(i,e,t)=>{for(const n in e)(t||!dh(n))&&(i[n]=e[n])},rv=(i,e,t)=>{const n=i.slots=C0();if(i.vnode.shapeFlag&32){const s=e._;s?(P0(n,e,t),t&&Vm(n,"_",s,!0)):I0(e,n)}else e&&D0(i,e)},ov=(i,e,t)=>{const{vnode:n,slots:s}=i;let r=!0,o=bt;if(n.shapeFlag&32){const a=e._;a?t&&a===1?r=!1:P0(s,e,t):(r=!e.$stable,I0(e,s)),o=e}else e&&(D0(i,e),o={default:1});if(r)for(const a in s)!dh(a)&&o[a]==null&&delete s[a]},Pn=fv;function av(i){return lv(i)}function lv(i,e){const t=rc();t.__VUE__=!0;const{insert:n,remove:s,patchProp:r,createElement:o,createText:a,createComment:l,setText:c,setElementText:u,parentNode:f,nextSibling:h,setScopeId:d=Hi,insertStaticContent:x}=i,p=(L,U,Y,w=null,oe=null,re=null,pe=void 0,se=null,me=!!U.dynamicChildren)=>{if(L===U)return;L&&!No(L,U)&&(w=ne(L),Re(L,oe,re,!0),L=null),U.patchFlag===-2&&(me=!1,U.dynamicChildren=null);const{type:ie,ref:Ae,shapeFlag:R}=U;switch(ie){case uc:g(L,U,Y,w);break;case Gs:m(L,U,Y,w);break;case Lc:L==null&&_(U,Y,w,pe);break;case Fi:P(L,U,Y,w,oe,re,pe,se,me);break;default:R&1?y(L,U,Y,w,oe,re,pe,se,me):R&6?B(L,U,Y,w,oe,re,pe,se,me):(R&64||R&128)&&ie.process(L,U,Y,w,oe,re,pe,se,me,Te)}Ae!=null&&oe?ea(Ae,L&&L.ref,re,U||L,!U):Ae==null&&L&&L.ref!=null&&ea(L.ref,null,re,L,!0)},g=(L,U,Y,w)=>{if(L==null)n(U.el=a(U.children),Y,w);else{const oe=U.el=L.el;U.children!==L.children&&c(oe,U.children)}},m=(L,U,Y,w)=>{L==null?n(U.el=l(U.children||""),Y,w):U.el=L.el},_=(L,U,Y,w)=>{[L.el,L.anchor]=x(L.children,U,Y,w,L.el,L.anchor)},S=({el:L,anchor:U},Y,w)=>{let oe;for(;L&&L!==U;)oe=h(L),n(L,Y,w),L=oe;n(U,Y,w)},A=({el:L,anchor:U})=>{let Y;for(;L&&L!==U;)Y=h(L),s(L),L=Y;s(U)},y=(L,U,Y,w,oe,re,pe,se,me)=>{if(U.type==="svg"?pe="svg":U.type==="math"&&(pe="mathml"),L==null)b(U,Y,w,oe,re,pe,se,me);else{const ie=L.el&&L.el._isVueCE?L.el:null;try{ie&&ie._beginPatch(),M(L,U,oe,re,pe,se,me)}finally{ie&&ie._endPatch()}}},b=(L,U,Y,w,oe,re,pe,se)=>{let me,ie;const{props:Ae,shapeFlag:R,transition:C,dirs:W}=L;if(me=L.el=o(L.type,re,Ae&&Ae.is,Ae),R&8?u(me,L.children):R&16&&E(L.children,me,null,w,oe,Fc(L,re),pe,se),W&&er(L,null,w,"created"),v(me,L,L.scopeId,pe,w),Ae){for(const fe in Ae)fe!=="value"&&!$o(fe)&&r(me,fe,null,Ae[fe],re,w);"value"in Ae&&r(me,"value",null,Ae.value,re),(ie=Ae.onVnodeBeforeMount)&&Ii(ie,w,L)}W&&er(L,null,w,"beforeMount");const $=cv(oe,C);$&&C.beforeEnter(me),n(me,U,Y),((ie=Ae&&Ae.onVnodeMounted)||$||W)&&Pn(()=>{ie&&Ii(ie,w,L),$&&C.enter(me),W&&er(L,null,w,"mounted")},oe)},v=(L,U,Y,w,oe)=>{if(Y&&d(L,Y),w)for(let re=0;re<w.length;re++)d(L,w[re]);if(oe){let re=oe.subTree;if(U===re||U0(re.type)&&(re.ssContent===U||re.ssFallback===U)){const pe=oe.vnode;v(L,pe,pe.scopeId,pe.slotScopeIds,oe.parent)}}},E=(L,U,Y,w,oe,re,pe,se,me=0)=>{for(let ie=me;ie<L.length;ie++){const Ae=L[ie]=se?os(L[ie]):Li(L[ie]);p(null,Ae,U,Y,w,oe,re,pe,se)}},M=(L,U,Y,w,oe,re,pe)=>{const se=U.el=L.el;let{patchFlag:me,dynamicChildren:ie,dirs:Ae}=U;me|=L.patchFlag&16;const R=L.props||bt,C=U.props||bt;let W;if(Y&&tr(Y,!1),(W=C.onVnodeBeforeUpdate)&&Ii(W,Y,U,L),Ae&&er(U,L,Y,"beforeUpdate"),Y&&tr(Y,!0),(R.innerHTML&&C.innerHTML==null||R.textContent&&C.textContent==null)&&u(se,""),ie?T(L.dynamicChildren,ie,se,Y,w,Fc(U,oe),re):pe||X(L,U,se,null,Y,w,Fc(U,oe),re,!1),me>0){if(me&16)I(se,R,C,Y,oe);else if(me&2&&R.class!==C.class&&r(se,"class",null,C.class,oe),me&4&&r(se,"style",R.style,C.style,oe),me&8){const $=U.dynamicProps;for(let fe=0;fe<$.length;fe++){const Z=$[fe],Ie=R[Z],ye=C[Z];(ye!==Ie||Z==="value")&&r(se,Z,Ie,ye,oe,Y)}}me&1&&L.children!==U.children&&u(se,U.children)}else!pe&&ie==null&&I(se,R,C,Y,oe);((W=C.onVnodeUpdated)||Ae)&&Pn(()=>{W&&Ii(W,Y,U,L),Ae&&er(U,L,Y,"updated")},w)},T=(L,U,Y,w,oe,re,pe)=>{for(let se=0;se<U.length;se++){const me=L[se],ie=U[se],Ae=me.el&&(me.type===Fi||!No(me,ie)||me.shapeFlag&198)?f(me.el):Y;p(me,ie,Ae,null,w,oe,re,pe,!0)}},I=(L,U,Y,w,oe)=>{if(U!==Y){if(U!==bt)for(const re in U)!$o(re)&&!(re in Y)&&r(L,re,U[re],null,oe,w);for(const re in Y){if($o(re))continue;const pe=Y[re],se=U[re];pe!==se&&re!=="value"&&r(L,re,se,pe,oe,w)}"value"in Y&&r(L,"value",U.value,Y.value,oe)}},P=(L,U,Y,w,oe,re,pe,se,me)=>{const ie=U.el=L?L.el:a(""),Ae=U.anchor=L?L.anchor:a("");let{patchFlag:R,dynamicChildren:C,slotScopeIds:W}=U;W&&(se=se?se.concat(W):W),L==null?(n(ie,Y,w),n(Ae,Y,w),E(U.children||[],Y,Ae,oe,re,pe,se,me)):R>0&&R&64&&C&&L.dynamicChildren&&L.dynamicChildren.length===C.length?(T(L.dynamicChildren,C,Y,oe,re,pe,se),(U.key!=null||oe&&U===oe.subTree)&&F0(L,U,!0)):X(L,U,Y,Ae,oe,re,pe,se,me)},B=(L,U,Y,w,oe,re,pe,se,me)=>{U.slotScopeIds=se,L==null?U.shapeFlag&512?oe.ctx.activate(U,Y,w,pe,me):N(U,Y,w,oe,re,pe,me):G(L,U,me)},N=(L,U,Y,w,oe,re,pe)=>{const se=L.component=Sv(L,w,oe);if(m0(L)&&(se.ctx.renderer=Te),yv(se,!1,pe),se.asyncDep){if(oe&&oe.registerDep(se,V,pe),!L.el){const me=se.subTree=Vi(Gs);m(null,me,U,Y),L.placeholder=me.el}}else V(se,L,U,Y,oe,re,pe)},G=(L,U,Y)=>{const w=U.component=L.component;if(J_(L,U,Y))if(w.asyncDep&&!w.asyncResolved){q(w,U,Y);return}else w.next=U,w.update();else U.el=L.el,w.vnode=U},V=(L,U,Y,w,oe,re,pe)=>{const se=()=>{if(L.isMounted){let{next:R,bu:C,u:W,parent:$,vnode:fe}=L;{const k=L0(L);if(k){R&&(R.el=fe.el,q(L,R,pe)),k.asyncDep.then(()=>{Pn(()=>{L.isUnmounted||ie()},oe)});return}}let Z=R,Ie;tr(L,!1),R?(R.el=fe.el,q(L,R,pe)):R=fe,C&&bl(C),(Ie=R.props&&R.props.onVnodeBeforeUpdate)&&Ii(Ie,$,R,fe),tr(L,!0);const ye=Sd(L),Ue=L.subTree;L.subTree=ye,p(Ue,ye,f(Ue.el),ne(Ue),L,oe,re),R.el=ye.el,Z===null&&ev(L,ye.el),W&&Pn(W,oe),(Ie=R.props&&R.props.onVnodeUpdated)&&Pn(()=>Ii(Ie,$,R,fe),oe)}else{let R;const{el:C,props:W}=U,{bm:$,m:fe,parent:Z,root:Ie,type:ye}=L,Ue=ta(U);tr(L,!1),$&&bl($),!Ue&&(R=W&&W.onVnodeBeforeMount)&&Ii(R,Z,U),tr(L,!0);{Ie.ce&&Ie.ce._hasShadowRoot()&&Ie.ce._injectChildStyle(ye);const k=L.subTree=Sd(L);p(null,k,Y,w,L,oe,re),U.el=k.el}if(fe&&Pn(fe,oe),!Ue&&(R=W&&W.onVnodeMounted)){const k=U;Pn(()=>Ii(R,Z,k),oe)}(U.shapeFlag&256||Z&&ta(Z.vnode)&&Z.vnode.shapeFlag&256)&&L.a&&Pn(L.a,oe),L.isMounted=!0,U=Y=w=null}};L.scope.on();const me=L.effect=new qm(se);L.scope.off();const ie=L.update=me.run.bind(me),Ae=L.job=me.runIfDirty.bind(me);Ae.i=L,Ae.id=L.uid,me.scheduler=()=>fh(Ae),tr(L,!0),ie()},q=(L,U,Y)=>{U.component=L;const w=L.vnode.props;L.vnode=U,L.next=null,nv(L,U.props,w,Y),ov(L,U.children,Y),gs(),dd(L),xs()},X=(L,U,Y,w,oe,re,pe,se,me=!1)=>{const ie=L&&L.children,Ae=L?L.shapeFlag:0,R=U.children,{patchFlag:C,shapeFlag:W}=U;if(C>0){if(C&128){ce(ie,R,Y,w,oe,re,pe,se,me);return}else if(C&256){ee(ie,R,Y,w,oe,re,pe,se,me);return}}W&8?(Ae&16&&J(ie,oe,re),R!==ie&&u(Y,R)):Ae&16?W&16?ce(ie,R,Y,w,oe,re,pe,se,me):J(ie,oe,re,!0):(Ae&8&&u(Y,""),W&16&&E(R,Y,w,oe,re,pe,se,me))},ee=(L,U,Y,w,oe,re,pe,se,me)=>{L=L||io,U=U||io;const ie=L.length,Ae=U.length,R=Math.min(ie,Ae);let C;for(C=0;C<R;C++){const W=U[C]=me?os(U[C]):Li(U[C]);p(L[C],W,Y,null,oe,re,pe,se,me)}ie>Ae?J(L,oe,re,!0,!1,R):E(U,Y,w,oe,re,pe,se,me,R)},ce=(L,U,Y,w,oe,re,pe,se,me)=>{let ie=0;const Ae=U.length;let R=L.length-1,C=Ae-1;for(;ie<=R&&ie<=C;){const W=L[ie],$=U[ie]=me?os(U[ie]):Li(U[ie]);if(No(W,$))p(W,$,Y,null,oe,re,pe,se,me);else break;ie++}for(;ie<=R&&ie<=C;){const W=L[R],$=U[C]=me?os(U[C]):Li(U[C]);if(No(W,$))p(W,$,Y,null,oe,re,pe,se,me);else break;R--,C--}if(ie>R){if(ie<=C){const W=C+1,$=W<Ae?U[W].el:w;for(;ie<=C;)p(null,U[ie]=me?os(U[ie]):Li(U[ie]),Y,$,oe,re,pe,se,me),ie++}}else if(ie>C)for(;ie<=R;)Re(L[ie],oe,re,!0),ie++;else{const W=ie,$=ie,fe=new Map;for(ie=$;ie<=C;ie++){const H=U[ie]=me?os(U[ie]):Li(U[ie]);H.key!=null&&fe.set(H.key,ie)}let Z,Ie=0;const ye=C-$+1;let Ue=!1,k=0;const te=new Array(ye);for(ie=0;ie<ye;ie++)te[ie]=0;for(ie=W;ie<=R;ie++){const H=L[ie];if(Ie>=ye){Re(H,oe,re,!0);continue}let z;if(H.key!=null)z=fe.get(H.key);else for(Z=$;Z<=C;Z++)if(te[Z-$]===0&&No(H,U[Z])){z=Z;break}z===void 0?Re(H,oe,re,!0):(te[z-$]=ie+1,z>=k?k=z:Ue=!0,p(H,U[z],Y,null,oe,re,pe,se,me),Ie++)}const _e=Ue?uv(te):io;for(Z=_e.length-1,ie=ye-1;ie>=0;ie--){const H=$+ie,z=U[H],he=U[H+1],Me=H+1<Ae?he.el||B0(he):w;te[ie]===0?p(null,z,Y,Me,oe,re,pe,se,me):Ue&&(Z<0||ie!==_e[Z]?be(z,Y,Me,2):Z--)}}},be=(L,U,Y,w,oe=null)=>{const{el:re,type:pe,transition:se,children:me,shapeFlag:ie}=L;if(ie&6){be(L.component.subTree,U,Y,w);return}if(ie&128){L.suspense.move(U,Y,w);return}if(ie&64){pe.move(L,U,Y,Te);return}if(pe===Fi){n(re,U,Y);for(let R=0;R<me.length;R++)be(me[R],U,Y,w);n(L.anchor,U,Y);return}if(pe===Lc){S(L,U,Y);return}if(w!==2&&ie&1&&se)if(w===0)se.beforeEnter(re),n(re,U,Y),Pn(()=>se.enter(re),oe);else{const{leave:R,delayLeave:C,afterLeave:W}=se,$=()=>{L.ctx.isUnmounted?s(re):n(re,U,Y)},fe=()=>{re._isLeaving&&re[E_](!0),R(re,()=>{$(),W&&W()})};C?C(re,$,fe):fe()}else n(re,U,Y)},Re=(L,U,Y,w=!1,oe=!1)=>{const{type:re,props:pe,ref:se,children:me,dynamicChildren:ie,shapeFlag:Ae,patchFlag:R,dirs:C,cacheIndex:W}=L;if(R===-2&&(oe=!1),se!=null&&(gs(),ea(se,null,Y,L,!0),xs()),W!=null&&(U.renderCache[W]=void 0),Ae&256){U.ctx.deactivate(L);return}const $=Ae&1&&C,fe=!ta(L);let Z;if(fe&&(Z=pe&&pe.onVnodeBeforeUnmount)&&Ii(Z,U,L),Ae&6)Ne(L.component,Y,w);else{if(Ae&128){L.suspense.unmount(Y,w);return}$&&er(L,null,U,"beforeUnmount"),Ae&64?L.type.remove(L,U,Y,Te,w):ie&&!ie.hasOnce&&(re!==Fi||R>0&&R&64)?J(ie,U,Y,!1,!0):(re===Fi&&R&384||!oe&&Ae&16)&&J(me,U,Y),w&&Fe(L)}(fe&&(Z=pe&&pe.onVnodeUnmounted)||$)&&Pn(()=>{Z&&Ii(Z,U,L),$&&er(L,null,U,"unmounted")},Y)},Fe=L=>{const{type:U,el:Y,anchor:w,transition:oe}=L;if(U===Fi){Oe(Y,w);return}if(U===Lc){A(L);return}const re=()=>{s(Y),oe&&!oe.persisted&&oe.afterLeave&&oe.afterLeave()};if(L.shapeFlag&1&&oe&&!oe.persisted){const{leave:pe,delayLeave:se}=oe,me=()=>pe(Y,re);se?se(L.el,re,me):me()}else re()},Oe=(L,U)=>{let Y;for(;L!==U;)Y=h(L),s(L),L=Y;s(U)},Ne=(L,U,Y)=>{const{bum:w,scope:oe,job:re,subTree:pe,um:se,m:me,a:ie}=L;bd(me),bd(ie),w&&bl(w),oe.stop(),re&&(re.flags|=8,Re(pe,L,U,Y)),se&&Pn(se,U),Pn(()=>{L.isUnmounted=!0},U)},J=(L,U,Y,w=!1,oe=!1,re=0)=>{for(let pe=re;pe<L.length;pe++)Re(L[pe],U,Y,w,oe)},ne=L=>{if(L.shapeFlag&6)return ne(L.component.subTree);if(L.shapeFlag&128)return L.suspense.next();const U=h(L.anchor||L.el),Y=U&&U[T_];return Y?h(Y):U};let xe=!1;const Be=(L,U,Y)=>{let w;L==null?U._vnode&&(Re(U._vnode,null,null,!0),w=U._vnode.component):p(U._vnode||null,L,U,null,null,null,Y),U._vnode=L,xe||(xe=!0,dd(w),c0(),xe=!1)},Te={p,um:Re,m:be,r:Fe,mt:N,mc:E,pc:X,pbc:T,n:ne,o:i};return{render:Be,hydrate:void 0,createApp:Y_(Be)}}function Fc({type:i,props:e},t){return t==="svg"&&i==="foreignObject"||t==="mathml"&&i==="annotation-xml"&&e&&e.encoding&&e.encoding.includes("html")?void 0:t}function tr({effect:i,job:e},t){t?(i.flags|=32,e.flags|=4):(i.flags&=-33,e.flags&=-5)}function cv(i,e){return(!i||i&&!i.pendingBranch)&&e&&!e.persisted}function F0(i,e,t=!1){const n=i.children,s=e.children;if($e(n)&&$e(s))for(let r=0;r<n.length;r++){const o=n[r];let a=s[r];a.shapeFlag&1&&!a.dynamicChildren&&((a.patchFlag<=0||a.patchFlag===32)&&(a=s[r]=os(s[r]),a.el=o.el),!t&&a.patchFlag!==-2&&F0(o,a)),a.type===uc&&(a.patchFlag===-1&&(a=s[r]=os(a)),a.el=o.el),a.type===Gs&&!a.el&&(a.el=o.el)}}function uv(i){const e=i.slice(),t=[0];let n,s,r,o,a;const l=i.length;for(n=0;n<l;n++){const c=i[n];if(c!==0){if(s=t[t.length-1],i[s]<c){e[n]=s,t.push(n);continue}for(r=0,o=t.length-1;r<o;)a=r+o>>1,i[t[a]]<c?r=a+1:o=a;c<i[t[r]]&&(r>0&&(e[n]=t[r-1]),t[r]=n)}}for(r=t.length,o=t[r-1];r-- >0;)t[r]=o,o=e[o];return t}function L0(i){const e=i.subTree.component;if(e)return e.asyncDep&&!e.asyncResolved?e:L0(e)}function bd(i){if(i)for(let e=0;e<i.length;e++)i[e].flags|=8}function B0(i){if(i.placeholder)return i.placeholder;const e=i.component;return e?B0(e.subTree):null}const U0=i=>i.__isSuspense;function fv(i,e){e&&e.pendingBranch?$e(i)?e.effects.push(...i):e.effects.push(i):v_(i)}const Fi=Symbol.for("v-fgt"),uc=Symbol.for("v-txt"),Gs=Symbol.for("v-cmt"),Lc=Symbol.for("v-stc"),ia=[];let Kn=null;function cn(i=!1){ia.push(Kn=i?null:[])}function hv(){ia.pop(),Kn=ia[ia.length-1]||null}let ma=1;function Md(i,e=!1){ma+=i,i<0&&Kn&&e&&(Kn.hasOnce=!0)}function O0(i){return i.dynamicChildren=ma>0?Kn||io:null,hv(),ma>0&&Kn&&Kn.push(i),i}function _n(i,e,t,n,s,r){return O0(je(i,e,t,n,s,r,!0))}function dv(i,e,t,n,s){return O0(Vi(i,e,t,n,s,!0))}function N0(i){return i?i.__v_isVNode===!0:!1}function No(i,e){return i.type===e.type&&i.key===e.key}const z0=({key:i})=>i??null,Tl=({ref:i,ref_key:e,ref_for:t})=>(typeof i=="number"&&(i=""+i),i!=null?jt(i)||hn(i)||Je(i)?{i:hi,r:i,k:e,f:!!t}:i:null);function je(i,e=null,t=null,n=0,s=null,r=i===Fi?0:1,o=!1,a=!1){const l={__v_isVNode:!0,__v_skip:!0,type:i,props:e,key:e&&z0(e),ref:e&&Tl(e),scopeId:f0,slotScopeIds:null,children:t,component:null,suspense:null,ssContent:null,ssFallback:null,dirs:null,transition:null,el:null,anchor:null,target:null,targetStart:null,targetAnchor:null,staticCount:0,shapeFlag:r,patchFlag:n,dynamicProps:s,dynamicChildren:null,appContext:null,ctx:hi};return a?(mh(l,t),r&128&&i.normalize(l)):t&&(l.shapeFlag|=jt(t)?8:16),ma>0&&!o&&Kn&&(l.patchFlag>0||r&6)&&l.patchFlag!==32&&Kn.push(l),l}const Vi=pv;function pv(i,e=null,t=null,n=0,s=null,r=!1){if((!i||i===N_)&&(i=Gs),N0(i)){const a=_o(i,e,!0);return t&&mh(a,t),ma>0&&!r&&Kn&&(a.shapeFlag&6?Kn[Kn.indexOf(i)]=a:Kn.push(a)),a.patchFlag=-2,a}if(Cv(i)&&(i=i.__vccOpts),e){e=mv(e);let{class:a,style:l}=e;a&&!jt(a)&&(e.class=ro(a)),_t(l)&&(uh(l)&&!$e(l)&&(l=mn({},l)),e.style=th(l))}const o=jt(i)?1:U0(i)?128:C_(i)?64:_t(i)?4:Je(i)?2:0;return je(i,e,t,n,s,o,r,!0)}function mv(i){return i?uh(i)||E0(i)?mn({},i):i:null}function _o(i,e,t=!1,n=!1){const{props:s,ref:r,patchFlag:o,children:a,transition:l}=i,c=e?xv(s||{},e):s,u={__v_isVNode:!0,__v_skip:!0,type:i.type,props:c,key:c&&z0(c),ref:e&&e.ref?t&&r?$e(r)?r.concat(Tl(e)):[r,Tl(e)]:Tl(e):r,scopeId:i.scopeId,slotScopeIds:i.slotScopeIds,children:a,target:i.target,targetStart:i.targetStart,targetAnchor:i.targetAnchor,staticCount:i.staticCount,shapeFlag:i.shapeFlag,patchFlag:e&&i.type!==Fi?o===-1?16:o|16:o,dynamicProps:i.dynamicProps,dynamicChildren:i.dynamicChildren,appContext:i.appContext,dirs:i.dirs,transition:l,component:i.component,suspense:i.suspense,ssContent:i.ssContent&&_o(i.ssContent),ssFallback:i.ssFallback&&_o(i.ssFallback),placeholder:i.placeholder,el:i.el,anchor:i.anchor,ctx:i.ctx,ce:i.ce};return l&&n&&hh(u,l.clone(u)),u}function gv(i=" ",e=0){return Vi(uc,null,i,e)}function vi(i="",e=!1){return e?(cn(),dv(Gs,null,i)):Vi(Gs,null,i)}function Li(i){return i==null||typeof i=="boolean"?Vi(Gs):$e(i)?Vi(Fi,null,i.slice()):N0(i)?os(i):Vi(uc,null,String(i))}function os(i){return i.el===null&&i.patchFlag!==-1||i.memo?i:_o(i)}function mh(i,e){let t=0;const{shapeFlag:n}=i;if(e==null)e=null;else if($e(e))t=16;else if(typeof e=="object")if(n&65){const s=e.default;s&&(s._c&&(s._d=!1),mh(i,s()),s._c&&(s._d=!0));return}else{t=32;const s=e._;!s&&!E0(e)?e._ctx=hi:s===3&&hi&&(hi.slots._===1?e._=1:(e._=2,i.patchFlag|=1024))}else Je(e)?(e={default:e,_ctx:hi},t=32):(e=String(e),n&64?(t=16,e=[gv(e)]):t=8);i.children=e,i.shapeFlag|=t}function xv(...i){const e={};for(let t=0;t<i.length;t++){const n=i[t];for(const s in n)if(s==="class")e.class!==n.class&&(e.class=ro([e.class,n.class]));else if(s==="style")e.style=th([e.style,n.style]);else if(ic(s)){const r=e[s],o=n[s];o&&r!==o&&!($e(r)&&r.includes(o))&&(e[s]=r?[].concat(r,o):o)}else s!==""&&(e[s]=n[s])}return e}function Ii(i,e,t,n=null){Wi(i,e,7,[t,n])}const _v=y0();let vv=0;function Sv(i,e,t){const n=i.type,s=(e?e.appContext:i.appContext)||_v,r={uid:vv++,vnode:i,type:n,parent:e,appContext:s,root:null,next:null,subTree:null,effect:null,update:null,job:null,scope:new Hx(!0),render:null,proxy:null,exposed:null,exposeProxy:null,withProxy:null,provides:e?e.provides:Object.create(s.provides),ids:e?e.ids:["",0,0],accessCache:null,renderCache:[],components:null,directives:null,propsOptions:R0(n,s),emitsOptions:b0(n,s),emit:null,emitted:null,propsDefaults:bt,inheritAttrs:n.inheritAttrs,ctx:bt,data:bt,props:bt,attrs:bt,slots:bt,refs:bt,setupState:bt,setupContext:null,suspense:t,suspenseId:t?t.pendingId:0,asyncDep:null,asyncResolved:!1,isMounted:!1,isUnmounted:!1,isDeactivated:!1,bc:null,c:null,bm:null,m:null,bu:null,u:null,um:null,bum:null,da:null,a:null,rtg:null,rtc:null,ec:null,sp:null};return r.ctx={_:r},r.root=e?e.root:r,r.emit=K_.bind(null,r),i.ce&&i.ce(r),r}let yn=null;const Av=()=>yn||hi;let Hl,Gu;{const i=rc(),e=(t,n)=>{let s;return(s=i[t])||(s=i[t]=[]),s.push(n),r=>{s.length>1?s.forEach(o=>o(r)):s[0](r)}};Hl=e("__VUE_INSTANCE_SETTERS__",t=>yn=t),Gu=e("__VUE_SSR_SETTERS__",t=>ga=t)}const La=i=>{const e=yn;return Hl(i),i.scope.on(),()=>{i.scope.off(),Hl(e)}},Td=()=>{yn&&yn.scope.off(),Hl(null)};function k0(i){return i.vnode.shapeFlag&4}let ga=!1;function yv(i,e=!1,t=!1){e&&Gu(e);const{props:n,children:s}=i.vnode,r=k0(i);tv(i,n,r,e),rv(i,s,t||e);const o=r?bv(i,e):void 0;return e&&Gu(!1),o}function bv(i,e){const t=i.type;i.accessCache=Object.create(null),i.proxy=new Proxy(i.ctx,k_);const{setup:n}=t;if(n){gs();const s=i.setupContext=n.length>1?Tv(i):null,r=La(i),o=Fa(n,i,0,[i.props,s]),a=Nm(o);if(xs(),r(),(a||i.sp)&&!ta(i)&&p0(i),a){if(o.then(Td,Td),e)return o.then(l=>{Cd(i,l)}).catch(l=>{ac(l,i,0)});i.asyncDep=o}else Cd(i,o)}else H0(i)}function Cd(i,e,t){Je(e)?i.type.__ssrInlineRender?i.ssrRender=e:i.render=e:_t(e)&&(i.setupState=o0(e)),H0(i)}function H0(i,e,t){const n=i.type;i.render||(i.render=n.render||Hi);{const s=La(i);gs();try{H_(i)}finally{xs(),s()}}}const Mv={get(i,e){return un(i,"get",""),i[e]}};function Tv(i){const e=t=>{i.exposed=t||{}};return{attrs:new Proxy(i.attrs,Mv),slots:i.slots,emit:i.emit,expose:e}}function fc(i){return i.exposed?i.exposeProxy||(i.exposeProxy=new Proxy(o0(l_(i.exposed)),{get(e,t){if(t in e)return e[t];if(t in na)return na[t](i)},has(e,t){return t in e||t in na}})):i.proxy}function Cv(i){return Je(i)&&"__vccOpts"in i}const Yo=(i,e)=>d_(i,e,ga),Ev="3.5.28";let Wu;const Ed=typeof window<"u"&&window.trustedTypes;if(Ed)try{Wu=Ed.createPolicy("vue",{createHTML:i=>i})}catch{}const V0=Wu?i=>Wu.createHTML(i):i=>i,wv="http://www.w3.org/2000/svg",Rv="http://www.w3.org/1998/Math/MathML",ss=typeof document<"u"?document:null,wd=ss&&ss.createElement("template"),Iv={insert:(i,e,t)=>{e.insertBefore(i,t||null)},remove:i=>{const e=i.parentNode;e&&e.removeChild(i)},createElement:(i,e,t,n)=>{const s=e==="svg"?ss.createElementNS(wv,i):e==="mathml"?ss.createElementNS(Rv,i):t?ss.createElement(i,{is:t}):ss.createElement(i);return i==="select"&&n&&n.multiple!=null&&s.setAttribute("multiple",n.multiple),s},createText:i=>ss.createTextNode(i),createComment:i=>ss.createComment(i),setText:(i,e)=>{i.nodeValue=e},setElementText:(i,e)=>{i.textContent=e},parentNode:i=>i.parentNode,nextSibling:i=>i.nextSibling,querySelector:i=>ss.querySelector(i),setScopeId(i,e){i.setAttribute(e,"")},insertStaticContent(i,e,t,n,s,r){const o=t?t.previousSibling:e.lastChild;if(s&&(s===r||s.nextSibling))for(;e.insertBefore(s.cloneNode(!0),t),!(s===r||!(s=s.nextSibling)););else{wd.innerHTML=V0(n==="svg"?`<svg>${i}</svg>`:n==="mathml"?`<math>${i}</math>`:i);const a=wd.content;if(n==="svg"||n==="mathml"){const l=a.firstChild;for(;l.firstChild;)a.appendChild(l.firstChild);a.removeChild(l)}e.insertBefore(a,t)}return[o?o.nextSibling:e.firstChild,t?t.previousSibling:e.lastChild]}},Dv=Symbol("_vtc");function Pv(i,e,t){const n=i[Dv];n&&(e=(e?[e,...n]:[...n]).join(" ")),e==null?i.removeAttribute("class"):t?i.setAttribute("class",e):i.className=e}const Rd=Symbol("_vod"),Fv=Symbol("_vsh"),Lv=Symbol(""),Bv=/(?:^|;)\s*display\s*:/;function Uv(i,e,t){const n=i.style,s=jt(t);let r=!1;if(t&&!s){if(e)if(jt(e))for(const o of e.split(";")){const a=o.slice(0,o.indexOf(":")).trim();t[a]==null&&Cl(n,a,"")}else for(const o in e)t[o]==null&&Cl(n,o,"");for(const o in t)o==="display"&&(r=!0),Cl(n,o,t[o])}else if(s){if(e!==t){const o=n[Lv];o&&(t+=";"+o),n.cssText=t,r=Bv.test(t)}}else e&&i.removeAttribute("style");Rd in i&&(i[Rd]=r?n.display:"",i[Fv]&&(n.display="none"))}const Id=/\s*!important$/;function Cl(i,e,t){if($e(t))t.forEach(n=>Cl(i,e,n));else if(t==null&&(t=""),e.startsWith("--"))i.setProperty(e,t);else{const n=Ov(i,e);Id.test(t)?i.setProperty(Qs(n),t.replace(Id,""),"important"):i[n]=t}}const Dd=["Webkit","Moz","ms"],Bc={};function Ov(i,e){const t=Bc[e];if(t)return t;let n=Vs(e);if(n!=="filter"&&n in i)return Bc[e]=n;n=Hm(n);for(let s=0;s<Dd.length;s++){const r=Dd[s]+n;if(r in i)return Bc[e]=r}return e}const Pd="http://www.w3.org/1999/xlink";function Fd(i,e,t,n,s,r=zx(e)){n&&e.startsWith("xlink:")?t==null?i.removeAttributeNS(Pd,e.slice(6,e.length)):i.setAttributeNS(Pd,e,t):t==null||r&&!Gm(t)?i.removeAttribute(e):i.setAttribute(e,r?"":Gi(t)?String(t):t)}function Ld(i,e,t,n,s){if(e==="innerHTML"||e==="textContent"){t!=null&&(i[e]=e==="innerHTML"?V0(t):t);return}const r=i.tagName;if(e==="value"&&r!=="PROGRESS"&&!r.includes("-")){const a=r==="OPTION"?i.getAttribute("value")||"":i.value,l=t==null?i.type==="checkbox"?"on":"":String(t);(a!==l||!("_value"in i))&&(i.value=l),t==null&&i.removeAttribute(e),i._value=t;return}let o=!1;if(t===""||t==null){const a=typeof i[e];a==="boolean"?t=Gm(t):t==null&&a==="string"?(t="",o=!0):a==="number"&&(t=0,o=!0)}try{i[e]=t}catch{}o&&i.removeAttribute(s||e)}function $r(i,e,t,n){i.addEventListener(e,t,n)}function Nv(i,e,t,n){i.removeEventListener(e,t,n)}const Bd=Symbol("_vei");function zv(i,e,t,n,s=null){const r=i[Bd]||(i[Bd]={}),o=r[e];if(n&&o)o.value=n;else{const[a,l]=kv(e);if(n){const c=r[e]=Gv(n,s);$r(i,a,c,l)}else o&&(Nv(i,a,o,l),r[e]=void 0)}}const Ud=/(?:Once|Passive|Capture)$/;function kv(i){let e;if(Ud.test(i)){e={};let n;for(;n=i.match(Ud);)i=i.slice(0,i.length-n[0].length),e[n[0].toLowerCase()]=!0}return[i[2]===":"?i.slice(3):Qs(i.slice(2)),e]}let Uc=0;const Hv=Promise.resolve(),Vv=()=>Uc||(Hv.then(()=>Uc=0),Uc=Date.now());function Gv(i,e){const t=n=>{if(!n._vts)n._vts=Date.now();else if(n._vts<=t.attached)return;Wi(Wv(n,t.value),e,5,[n])};return t.value=i,t.attached=Vv(),t}function Wv(i,e){if($e(e)){const t=i.stopImmediatePropagation;return i.stopImmediatePropagation=()=>{t.call(i),i._stopped=!0},e.map(n=>s=>!s._stopped&&n&&n(s))}else return e}const Od=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&i.charCodeAt(2)>96&&i.charCodeAt(2)<123,Xv=(i,e,t,n,s,r)=>{const o=s==="svg";e==="class"?Pv(i,n,o):e==="style"?Uv(i,t,n):ic(e)?$f(e)||zv(i,e,t,n,r):(e[0]==="."?(e=e.slice(1),!0):e[0]==="^"?(e=e.slice(1),!1):qv(i,e,n,o))?(Ld(i,e,n),!i.tagName.includes("-")&&(e==="value"||e==="checked"||e==="selected")&&Fd(i,e,n,o,r,e!=="value")):i._isVueCE&&(/[A-Z]/.test(e)||!jt(n))?Ld(i,Vs(e),n,r,e):(e==="true-value"?i._trueValue=n:e==="false-value"&&(i._falseValue=n),Fd(i,e,n,o))};function qv(i,e,t,n){if(n)return!!(e==="innerHTML"||e==="textContent"||e in i&&Od(e)&&Je(t));if(e==="spellcheck"||e==="draggable"||e==="translate"||e==="autocorrect"||e==="sandbox"&&i.tagName==="IFRAME"||e==="form"||e==="list"&&i.tagName==="INPUT"||e==="type"&&i.tagName==="TEXTAREA")return!1;if(e==="width"||e==="height"){const s=i.tagName;if(s==="IMG"||s==="VIDEO"||s==="CANVAS"||s==="SOURCE")return!1}return Od(e)&&jt(t)?!1:e in i}const Nd=i=>{const e=i.props["onUpdate:modelValue"]||!1;return $e(e)?t=>bl(e,t):e};function Yv(i){i.target.composing=!0}function zd(i){const e=i.target;e.composing&&(e.composing=!1,e.dispatchEvent(new Event("input")))}const Oc=Symbol("_assign");function kd(i,e,t){return e&&(i=i.trim()),t&&(i=eh(i)),i}const Nc={created(i,{modifiers:{lazy:e,trim:t,number:n}},s){i[Oc]=Nd(s);const r=n||s.props&&s.props.type==="number";$r(i,e?"change":"input",o=>{o.target.composing||i[Oc](kd(i.value,t,r))}),(t||r)&&$r(i,"change",()=>{i.value=kd(i.value,t,r)}),e||($r(i,"compositionstart",Yv),$r(i,"compositionend",zd),$r(i,"change",zd))},mounted(i,{value:e}){i.value=e??""},beforeUpdate(i,{value:e,oldValue:t,modifiers:{lazy:n,trim:s,number:r}},o){if(i[Oc]=Nd(o),i.composing)return;const a=(r||i.type==="number")&&!/^0\d/.test(i.value)?eh(i.value):i.value,l=e??"";a!==l&&(document.activeElement===i&&i.type!=="range"&&(n&&e===t||s&&i.value.trim()===l)||(i.value=l))}},Qv=["ctrl","shift","alt","meta"],Kv={stop:i=>i.stopPropagation(),prevent:i=>i.preventDefault(),self:i=>i.target!==i.currentTarget,ctrl:i=>!i.ctrlKey,shift:i=>!i.shiftKey,alt:i=>!i.altKey,meta:i=>!i.metaKey,left:i=>"button"in i&&i.button!==0,middle:i=>"button"in i&&i.button!==1,right:i=>"button"in i&&i.button!==2,exact:(i,e)=>Qv.some(t=>i[`${t}Key`]&&!e.includes(t))},Lt=(i,e)=>{if(!i)return i;const t=i._withMods||(i._withMods={}),n=e.join(".");return t[n]||(t[n]=((s,...r)=>{for(let o=0;o<e.length;o++){const a=Kv[e[o]];if(a&&a(s,e))return}return i(s,...r)}))},jv={esc:"escape",space:" ",up:"arrow-up",left:"arrow-left",right:"arrow-right",down:"arrow-down",delete:"backspace"},$v=(i,e)=>{const t=i._withKeys||(i._withKeys={}),n=e.join(".");return t[n]||(t[n]=(s=>{if(!("key"in s))return;const r=Qs(s.key);if(e.some(o=>o===r||jv[o]===r))return i(s)}))},Zv=mn({patchProp:Xv},Iv);let Hd;function Jv(){return Hd||(Hd=av(Zv))}const eS=((...i)=>{const e=Jv().createApp(...i),{mount:t}=e;return e.mount=n=>{const s=nS(n);if(!s)return;const r=e._component;!Je(r)&&!r.render&&!r.template&&(r.template=s.innerHTML),s.nodeType===1&&(s.textContent="");const o=t(s,!1,tS(s));return s instanceof Element&&(s.removeAttribute("v-cloak"),s.setAttribute("data-v-app","")),o},e});function tS(i){if(i instanceof SVGElement)return"svg";if(typeof MathMLElement=="function"&&i instanceof MathMLElement)return"mathml"}function nS(i){return jt(i)?document.querySelector(i):i}const gh="181",li={ROTATE:0,DOLLY:1,PAN:2},ci={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},iS=0,Vd=1,sS=2,G0=1,rS=2,ns=3,Xi=0,Bn=1,fi=2,ps=0,Ns=1,Gd=2,Wd=3,Xd=4,W0=5,dr=100,oS=101,aS=102,lS=103,cS=104,uS=200,fS=201,hS=202,dS=203,xa=204,_a=205,pS=206,mS=207,gS=208,xS=209,_S=210,vS=211,SS=212,AS=213,yS=214,Xu=0,qu=1,Yu=2,vo=3,Qu=4,Ku=5,ju=6,$u=7,X0=0,bS=1,MS=2,zs=0,TS=1,CS=2,ES=3,wS=4,RS=5,IS=6,DS=7,q0=300,So=301,Ao=302,Zu=303,Ju=304,hc=306,ef=1e3,ds=1001,tf=1002,Jn=1003,PS=1004,Xa=1005,di=1006,zc=1007,mr=1008,qi=1009,Y0=1010,Q0=1011,va=1012,xh=1013,pi=1014,Mi=1015,Tr=1016,_h=1017,vh=1018,Sa=1020,K0=35902,j0=35899,$0=1021,Z0=1022,Mn=1023,yo=1026,Aa=1027,J0=1028,dc=1029,Sh=1030,Ah=1031,lo=1033,El=33776,wl=33777,Rl=33778,Il=33779,nf=35840,sf=35841,rf=35842,of=35843,af=36196,lf=37492,cf=37496,uf=37808,ff=37809,hf=37810,df=37811,pf=37812,mf=37813,gf=37814,xf=37815,_f=37816,vf=37817,Sf=37818,Af=37819,yf=37820,bf=37821,Mf=36492,Tf=36494,Cf=36495,Ef=36283,wf=36284,Rf=36285,If=36286,FS=3200,LS=3201,BS=0,US=1,Ds="",ai="srgb",bo="srgb-linear",Vl="linear",mt="srgb",Pr=7680,qd=519,OS=512,NS=513,zS=514,eg=515,kS=516,HS=517,VS=518,GS=519,Yd=35044,WS=35048,Qd="300 es",Oi=2e3,Gl=2001;function tg(i){for(let e=i.length-1;e>=0;--e)if(i[e]>=65535)return!0;return!1}function Wl(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function XS(){const i=Wl("canvas");return i.style.display="block",i}const Kd={};function jd(...i){const e="THREE."+i.shift();console.log(e,...i)}function Ze(...i){const e="THREE."+i.shift();console.warn(e,...i)}function Wt(...i){const e="THREE."+i.shift();console.error(e,...i)}function ya(...i){const e=i.join(" ");e in Kd||(Kd[e]=!0,Ze(...i))}function qS(i,e,t){return new Promise(function(n,s){function r(){switch(i.clientWaitSync(e,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:s();break;case i.TIMEOUT_EXPIRED:setTimeout(r,t);break;default:n()}}setTimeout(r,t)})}class Ks{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[e]===void 0&&(n[e]=[]),n[e].indexOf(t)===-1&&n[e].push(t)}hasEventListener(e,t){const n=this._listeners;return n===void 0?!1:n[e]!==void 0&&n[e].indexOf(t)!==-1}removeEventListener(e,t){const n=this._listeners;if(n===void 0)return;const s=n[e];if(s!==void 0){const r=s.indexOf(t);r!==-1&&s.splice(r,1)}}dispatchEvent(e){const t=this._listeners;if(t===void 0)return;const n=t[e.type];if(n!==void 0){e.target=this;const s=n.slice(0);for(let r=0,o=s.length;r<o;r++)s[r].call(this,e);e.target=null}}}const an=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],Dl=Math.PI/180,Df=180/Math.PI;function Ba(){const i=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(an[i&255]+an[i>>8&255]+an[i>>16&255]+an[i>>24&255]+"-"+an[e&255]+an[e>>8&255]+"-"+an[e>>16&15|64]+an[e>>24&255]+"-"+an[t&63|128]+an[t>>8&255]+"-"+an[t>>16&255]+an[t>>24&255]+an[n&255]+an[n>>8&255]+an[n>>16&255]+an[n>>24&255]).toLowerCase()}function tt(i,e,t){return Math.max(e,Math.min(t,i))}function YS(i,e){return(i%e+e)%e}function kc(i,e,t){return(1-t)*i+t*e}function zo(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return i/4294967295;case Uint16Array:return i/65535;case Uint8Array:return i/255;case Int32Array:return Math.max(i/2147483647,-1);case Int16Array:return Math.max(i/32767,-1);case Int8Array:return Math.max(i/127,-1);default:throw new Error("Invalid component type.")}}function In(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return Math.round(i*4294967295);case Uint16Array:return Math.round(i*65535);case Uint8Array:return Math.round(i*255);case Int32Array:return Math.round(i*2147483647);case Int16Array:return Math.round(i*32767);case Int8Array:return Math.round(i*127);default:throw new Error("Invalid component type.")}}const yh={DEG2RAD:Dl};class Pe{constructor(e=0,t=0){Pe.prototype.isVector2=!0,this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const t=this.x,n=this.y,s=e.elements;return this.x=s[0]*t+s[3]*n+s[6],this.y=s[1]*t+s[4]*n+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=tt(this.x,e.x,t.x),this.y=tt(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=tt(this.x,e,t),this.y=tt(this.y,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(tt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(tt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y;return t*t+n*n}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){const n=Math.cos(t),s=Math.sin(t),r=this.x-e.x,o=this.y-e.y;return this.x=r*n-o*s+e.x,this.y=r*s+o*n+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class Mt{constructor(e=0,t=0,n=0,s=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=n,this._w=s}static slerpFlat(e,t,n,s,r,o,a){let l=n[s+0],c=n[s+1],u=n[s+2],f=n[s+3],h=r[o+0],d=r[o+1],x=r[o+2],p=r[o+3];if(a<=0){e[t+0]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f;return}if(a>=1){e[t+0]=h,e[t+1]=d,e[t+2]=x,e[t+3]=p;return}if(f!==p||l!==h||c!==d||u!==x){let g=l*h+c*d+u*x+f*p;g<0&&(h=-h,d=-d,x=-x,p=-p,g=-g);let m=1-a;if(g<.9995){const _=Math.acos(g),S=Math.sin(_);m=Math.sin(m*_)/S,a=Math.sin(a*_)/S,l=l*m+h*a,c=c*m+d*a,u=u*m+x*a,f=f*m+p*a}else{l=l*m+h*a,c=c*m+d*a,u=u*m+x*a,f=f*m+p*a;const _=1/Math.sqrt(l*l+c*c+u*u+f*f);l*=_,c*=_,u*=_,f*=_}}e[t]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f}static multiplyQuaternionsFlat(e,t,n,s,r,o){const a=n[s],l=n[s+1],c=n[s+2],u=n[s+3],f=r[o],h=r[o+1],d=r[o+2],x=r[o+3];return e[t]=a*x+u*f+l*d-c*h,e[t+1]=l*x+u*h+c*f-a*d,e[t+2]=c*x+u*d+a*h-l*f,e[t+3]=u*x-a*f-l*h-c*d,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,n,s){return this._x=e,this._y=t,this._z=n,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){const n=e._x,s=e._y,r=e._z,o=e._order,a=Math.cos,l=Math.sin,c=a(n/2),u=a(s/2),f=a(r/2),h=l(n/2),d=l(s/2),x=l(r/2);switch(o){case"XYZ":this._x=h*u*f+c*d*x,this._y=c*d*f-h*u*x,this._z=c*u*x+h*d*f,this._w=c*u*f-h*d*x;break;case"YXZ":this._x=h*u*f+c*d*x,this._y=c*d*f-h*u*x,this._z=c*u*x-h*d*f,this._w=c*u*f+h*d*x;break;case"ZXY":this._x=h*u*f-c*d*x,this._y=c*d*f+h*u*x,this._z=c*u*x+h*d*f,this._w=c*u*f-h*d*x;break;case"ZYX":this._x=h*u*f-c*d*x,this._y=c*d*f+h*u*x,this._z=c*u*x-h*d*f,this._w=c*u*f+h*d*x;break;case"YZX":this._x=h*u*f+c*d*x,this._y=c*d*f+h*u*x,this._z=c*u*x-h*d*f,this._w=c*u*f-h*d*x;break;case"XZY":this._x=h*u*f-c*d*x,this._y=c*d*f-h*u*x,this._z=c*u*x+h*d*f,this._w=c*u*f+h*d*x;break;default:Ze("Quaternion: .setFromEuler() encountered an unknown order: "+o)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){const n=t/2,s=Math.sin(n);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(e){const t=e.elements,n=t[0],s=t[4],r=t[8],o=t[1],a=t[5],l=t[9],c=t[2],u=t[6],f=t[10],h=n+a+f;if(h>0){const d=.5/Math.sqrt(h+1);this._w=.25/d,this._x=(u-l)*d,this._y=(r-c)*d,this._z=(o-s)*d}else if(n>a&&n>f){const d=2*Math.sqrt(1+n-a-f);this._w=(u-l)/d,this._x=.25*d,this._y=(s+o)/d,this._z=(r+c)/d}else if(a>f){const d=2*Math.sqrt(1+a-n-f);this._w=(r-c)/d,this._x=(s+o)/d,this._y=.25*d,this._z=(l+u)/d}else{const d=2*Math.sqrt(1+f-n-a);this._w=(o-s)/d,this._x=(r+c)/d,this._y=(l+u)/d,this._z=.25*d}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let n=e.dot(t)+1;return n<1e-8?(n=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=n):(this._x=0,this._y=-e.z,this._z=e.y,this._w=n)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=n),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(tt(this.dot(e),-1,1)))}rotateTowards(e,t){const n=this.angleTo(e);if(n===0)return this;const s=Math.min(1,t/n);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){const n=e._x,s=e._y,r=e._z,o=e._w,a=t._x,l=t._y,c=t._z,u=t._w;return this._x=n*u+o*a+s*c-r*l,this._y=s*u+o*l+r*a-n*c,this._z=r*u+o*c+n*l-s*a,this._w=o*u-n*a-s*l-r*c,this._onChangeCallback(),this}slerp(e,t){if(t<=0)return this;if(t>=1)return this.copy(e);let n=e._x,s=e._y,r=e._z,o=e._w,a=this.dot(e);a<0&&(n=-n,s=-s,r=-r,o=-o,a=-a);let l=1-t;if(a<.9995){const c=Math.acos(a),u=Math.sin(c);l=Math.sin(l*c)/u,t=Math.sin(t*c)/u,this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this._onChangeCallback()}else this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this.normalize();return this}slerpQuaternions(e,t,n){return this.copy(e).slerp(t,n)}random(){const e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),n=Math.random(),s=Math.sqrt(1-n),r=Math.sqrt(n);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(t),r*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class F{constructor(e=0,t=0,n=0){F.prototype.isVector3=!0,this.x=e,this.y=t,this.z=n}set(e,t,n){return n===void 0&&(n=this.z),this.x=e,this.y=t,this.z=n,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion($d.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion($d.setFromAxisAngle(e,t))}applyMatrix3(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[3]*n+r[6]*s,this.y=r[1]*t+r[4]*n+r[7]*s,this.z=r[2]*t+r[5]*n+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=e.elements,o=1/(r[3]*t+r[7]*n+r[11]*s+r[15]);return this.x=(r[0]*t+r[4]*n+r[8]*s+r[12])*o,this.y=(r[1]*t+r[5]*n+r[9]*s+r[13])*o,this.z=(r[2]*t+r[6]*n+r[10]*s+r[14])*o,this}applyQuaternion(e){const t=this.x,n=this.y,s=this.z,r=e.x,o=e.y,a=e.z,l=e.w,c=2*(o*s-a*n),u=2*(a*t-r*s),f=2*(r*n-o*t);return this.x=t+l*c+o*f-a*u,this.y=n+l*u+a*c-r*f,this.z=s+l*f+r*u-o*c,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[4]*n+r[8]*s,this.y=r[1]*t+r[5]*n+r[9]*s,this.z=r[2]*t+r[6]*n+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=tt(this.x,e.x,t.x),this.y=tt(this.y,e.y,t.y),this.z=tt(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=tt(this.x,e,t),this.y=tt(this.y,e,t),this.z=tt(this.z,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(tt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){const n=e.x,s=e.y,r=e.z,o=t.x,a=t.y,l=t.z;return this.x=s*l-r*a,this.y=r*o-n*l,this.z=n*a-s*o,this}projectOnVector(e){const t=e.lengthSq();if(t===0)return this.set(0,0,0);const n=e.dot(this)/t;return this.copy(e).multiplyScalar(n)}projectOnPlane(e){return Hc.copy(this).projectOnVector(e),this.sub(Hc)}reflect(e){return this.sub(Hc.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(tt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y,s=this.z-e.z;return t*t+n*n+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,n){const s=Math.sin(t)*e;return this.x=s*Math.sin(n),this.y=Math.cos(t)*e,this.z=s*Math.cos(n),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,n){return this.x=e*Math.sin(t),this.y=n,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){const t=this.setFromMatrixColumn(e,0).length(),n=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=n,this.z=s,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,t=Math.random()*2-1,n=Math.sqrt(1-t*t);return this.x=n*Math.cos(e),this.y=t,this.z=n*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const Hc=new F,$d=new Mt;class Qe{constructor(e,t,n,s,r,o,a,l,c){Qe.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c)}set(e,t,n,s,r,o,a,l,c){const u=this.elements;return u[0]=e,u[1]=s,u[2]=a,u[3]=t,u[4]=r,u[5]=l,u[6]=n,u[7]=o,u[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],this}extractBasis(e,t,n){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[3],l=n[6],c=n[1],u=n[4],f=n[7],h=n[2],d=n[5],x=n[8],p=s[0],g=s[3],m=s[6],_=s[1],S=s[4],A=s[7],y=s[2],b=s[5],v=s[8];return r[0]=o*p+a*_+l*y,r[3]=o*g+a*S+l*b,r[6]=o*m+a*A+l*v,r[1]=c*p+u*_+f*y,r[4]=c*g+u*S+f*b,r[7]=c*m+u*A+f*v,r[2]=h*p+d*_+x*y,r[5]=h*g+d*S+x*b,r[8]=h*m+d*A+x*v,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8];return t*o*u-t*a*c-n*r*u+n*a*l+s*r*c-s*o*l}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=u*o-a*c,h=a*l-u*r,d=c*r-o*l,x=t*f+n*h+s*d;if(x===0)return this.set(0,0,0,0,0,0,0,0,0);const p=1/x;return e[0]=f*p,e[1]=(s*c-u*n)*p,e[2]=(a*n-s*o)*p,e[3]=h*p,e[4]=(u*t-s*l)*p,e[5]=(s*r-a*t)*p,e[6]=d*p,e[7]=(n*l-c*t)*p,e[8]=(o*t-n*r)*p,this}transpose(){let e;const t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,n,s,r,o,a){const l=Math.cos(r),c=Math.sin(r);return this.set(n*l,n*c,-n*(l*o+c*a)+o+e,-s*c,s*l,-s*(-c*o+l*a)+a+t,0,0,1),this}scale(e,t){return this.premultiply(Vc.makeScale(e,t)),this}rotate(e){return this.premultiply(Vc.makeRotation(-e)),this}translate(e,t){return this.premultiply(Vc.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,n,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<9;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<9;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e}clone(){return new this.constructor().fromArray(this.elements)}}const Vc=new Qe,Zd=new Qe().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Jd=new Qe().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function QS(){const i={enabled:!0,workingColorSpace:bo,spaces:{},convert:function(s,r,o){return this.enabled===!1||r===o||!r||!o||(this.spaces[r].transfer===mt&&(s.r=ms(s.r),s.g=ms(s.g),s.b=ms(s.b)),this.spaces[r].primaries!==this.spaces[o].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[o].fromXYZ)),this.spaces[o].transfer===mt&&(s.r=co(s.r),s.g=co(s.g),s.b=co(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===Ds?Vl:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,o){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[o].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return ya("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),i.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return ya("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),i.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],n=[.3127,.329];return i.define({[bo]:{primaries:e,whitePoint:n,transfer:Vl,toXYZ:Zd,fromXYZ:Jd,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:ai},outputColorSpaceConfig:{drawingBufferColorSpace:ai}},[ai]:{primaries:e,whitePoint:n,transfer:mt,toXYZ:Zd,fromXYZ:Jd,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:ai}}}),i}const lt=QS();function ms(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function co(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}let Fr;class KS{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let n;if(e instanceof HTMLCanvasElement)n=e;else{Fr===void 0&&(Fr=Wl("canvas")),Fr.width=e.width,Fr.height=e.height;const s=Fr.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),n=Fr}return n.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const t=Wl("canvas");t.width=e.width,t.height=e.height;const n=t.getContext("2d");n.drawImage(e,0,0,e.width,e.height);const s=n.getImageData(0,0,e.width,e.height),r=s.data;for(let o=0;o<r.length;o++)r[o]=ms(r[o]/255)*255;return n.putImageData(s,0,0),t}else if(e.data){const t=e.data.slice(0);for(let n=0;n<t.length;n++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[n]=Math.floor(ms(t[n]/255)*255):t[n]=ms(t[n]);return{data:t,width:e.width,height:e.height}}else return Ze("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let jS=0;class bh{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:jS++}),this.uuid=Ba(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):t instanceof VideoFrame?e.set(t.displayHeight,t.displayWidth,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const n={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let o=0,a=s.length;o<a;o++)s[o].isDataTexture?r.push(Gc(s[o].image)):r.push(Gc(s[o]))}else r=Gc(s);n.url=r}return t||(e.images[this.uuid]=n),n}}function Gc(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?KS.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(Ze("Texture: Unable to serialize Texture."),{})}let $S=0;const Wc=new F;class Tn extends Ks{constructor(e=Tn.DEFAULT_IMAGE,t=Tn.DEFAULT_MAPPING,n=ds,s=ds,r=di,o=mr,a=Mn,l=qi,c=Tn.DEFAULT_ANISOTROPY,u=Ds){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:$S++}),this.uuid=Ba(),this.name="",this.source=new bh(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=n,this.wrapT=s,this.magFilter=r,this.minFilter=o,this.anisotropy=c,this.format=a,this.internalFormat=null,this.type=l,this.offset=new Pe(0,0),this.repeat=new Pe(1,1),this.center=new Pe(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Qe,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=u,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0}get width(){return this.source.getSize(Wc).x}get height(){return this.source.getSize(Wc).y}get depth(){return this.source.getSize(Wc).z}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const t in e){const n=e[t];if(n===void 0){Ze(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){Ze(`Texture.setValues(): property '${t}' does not exist.`);continue}s&&n&&s.isVector2&&n.isVector2||s&&n&&s.isVector3&&n.isVector3||s&&n&&s.isMatrix3&&n.isMatrix3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const n={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),t||(e.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==q0)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case ef:e.x=e.x-Math.floor(e.x);break;case ds:e.x=e.x<0?0:1;break;case tf:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case ef:e.y=e.y-Math.floor(e.y);break;case ds:e.y=e.y<0?0:1;break;case tf:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}Tn.DEFAULT_IMAGE=null;Tn.DEFAULT_MAPPING=q0;Tn.DEFAULT_ANISOTROPY=1;class Dt{constructor(e=0,t=0,n=0,s=1){Dt.prototype.isVector4=!0,this.x=e,this.y=t,this.z=n,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,n,s){return this.x=e,this.y=t,this.z=n,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=this.w,o=e.elements;return this.x=o[0]*t+o[4]*n+o[8]*s+o[12]*r,this.y=o[1]*t+o[5]*n+o[9]*s+o[13]*r,this.z=o[2]*t+o[6]*n+o[10]*s+o[14]*r,this.w=o[3]*t+o[7]*n+o[11]*s+o[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,n,s,r;const l=e.elements,c=l[0],u=l[4],f=l[8],h=l[1],d=l[5],x=l[9],p=l[2],g=l[6],m=l[10];if(Math.abs(u-h)<.01&&Math.abs(f-p)<.01&&Math.abs(x-g)<.01){if(Math.abs(u+h)<.1&&Math.abs(f+p)<.1&&Math.abs(x+g)<.1&&Math.abs(c+d+m-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;const S=(c+1)/2,A=(d+1)/2,y=(m+1)/2,b=(u+h)/4,v=(f+p)/4,E=(x+g)/4;return S>A&&S>y?S<.01?(n=0,s=.707106781,r=.707106781):(n=Math.sqrt(S),s=b/n,r=v/n):A>y?A<.01?(n=.707106781,s=0,r=.707106781):(s=Math.sqrt(A),n=b/s,r=E/s):y<.01?(n=.707106781,s=.707106781,r=0):(r=Math.sqrt(y),n=v/r,s=E/r),this.set(n,s,r,t),this}let _=Math.sqrt((g-x)*(g-x)+(f-p)*(f-p)+(h-u)*(h-u));return Math.abs(_)<.001&&(_=1),this.x=(g-x)/_,this.y=(f-p)/_,this.z=(h-u)/_,this.w=Math.acos((c+d+m-1)/2),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=tt(this.x,e.x,t.x),this.y=tt(this.y,e.y,t.y),this.z=tt(this.z,e.z,t.z),this.w=tt(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=tt(this.x,e,t),this.y=tt(this.y,e,t),this.z=tt(this.z,e,t),this.w=tt(this.w,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(tt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this.w=e.w+(t.w-e.w)*n,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class ZS extends Ks{constructor(e=1,t=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:di,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},n),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=n.depth,this.scissor=new Dt(0,0,e,t),this.scissorTest=!1,this.viewport=new Dt(0,0,e,t);const s={width:e,height:t,depth:n.depth},r=new Tn(s);this.textures=[];const o=n.count;for(let a=0;a<o;a++)this.textures[a]=r.clone(),this.textures[a].isRenderTargetTexture=!0,this.textures[a].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview}_setTextureOptions(e={}){const t={minFilter:di,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let n=0;n<this.textures.length;n++)this.textures[n].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,n=1){if(this.width!==e||this.height!==t||this.depth!==n){this.width=e,this.height=t,this.depth=n;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=t,this.textures[s].image.depth=n,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,n=e.textures.length;t<n;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;const s=Object.assign({},e.textures[t].image);this.textures[t].source=new bh(s)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class Ws extends ZS{constructor(e=1,t=1,n={}){super(e,t,n),this.isWebGLRenderTarget=!0}}class ng extends Tn{constructor(e=null,t=1,n=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=Jn,this.minFilter=Jn,this.wrapR=ds,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class JS extends Tn{constructor(e=null,t=1,n=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=Jn,this.minFilter=Jn,this.wrapR=ds,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class Ni{constructor(e=new F(1/0,1/0,1/0),t=new F(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t+=3)this.expandByPoint(Si.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,n=e.count;t<n;t++)this.expandByPoint(Si.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){const n=Si.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(n),this.max.copy(e).add(n),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);const n=e.geometry;if(n!==void 0){const r=n.getAttribute("position");if(t===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let o=0,a=r.count;o<a;o++)e.isMesh===!0?e.getVertexPosition(o,Si):Si.fromBufferAttribute(r,o),Si.applyMatrix4(e.matrixWorld),this.expandByPoint(Si);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),qa.copy(e.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),qa.copy(n.boundingBox)),qa.applyMatrix4(e.matrixWorld),this.union(qa)}const s=e.children;for(let r=0,o=s.length;r<o;r++)this.expandByObject(s[r],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,Si),Si.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,n;return e.normal.x>0?(t=e.normal.x*this.min.x,n=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,n=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,n+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,n+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,n+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,n+=e.normal.z*this.min.z),t<=-e.constant&&n>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(ko),Ya.subVectors(this.max,ko),Lr.subVectors(e.a,ko),Br.subVectors(e.b,ko),Ur.subVectors(e.c,ko),ys.subVectors(Br,Lr),bs.subVectors(Ur,Br),nr.subVectors(Lr,Ur);let t=[0,-ys.z,ys.y,0,-bs.z,bs.y,0,-nr.z,nr.y,ys.z,0,-ys.x,bs.z,0,-bs.x,nr.z,0,-nr.x,-ys.y,ys.x,0,-bs.y,bs.x,0,-nr.y,nr.x,0];return!Xc(t,Lr,Br,Ur,Ya)||(t=[1,0,0,0,1,0,0,0,1],!Xc(t,Lr,Br,Ur,Ya))?!1:(Qa.crossVectors(ys,bs),t=[Qa.x,Qa.y,Qa.z],Xc(t,Lr,Br,Ur,Ya))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,Si).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(Si).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Ki[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Ki[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Ki[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Ki[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Ki[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Ki[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Ki[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Ki[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Ki),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const Ki=[new F,new F,new F,new F,new F,new F,new F,new F],Si=new F,qa=new Ni,Lr=new F,Br=new F,Ur=new F,ys=new F,bs=new F,nr=new F,ko=new F,Ya=new F,Qa=new F,ir=new F;function Xc(i,e,t,n,s){for(let r=0,o=i.length-3;r<=o;r+=3){ir.fromArray(i,r);const a=s.x*Math.abs(ir.x)+s.y*Math.abs(ir.y)+s.z*Math.abs(ir.z),l=e.dot(ir),c=t.dot(ir),u=n.dot(ir);if(Math.max(-Math.max(l,c,u),Math.min(l,c,u))>a)return!1}return!0}const eA=new Ni,Ho=new F,qc=new F;class pc{constructor(e=new F,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){const n=this.center;t!==void 0?n.copy(t):eA.setFromPoints(e).getCenter(n);let s=0;for(let r=0,o=e.length;r<o;r++)s=Math.max(s,n.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){const n=this.center.distanceToSquared(e);return t.copy(e),n>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;Ho.subVectors(e,this.center);const t=Ho.lengthSq();if(t>this.radius*this.radius){const n=Math.sqrt(t),s=(n-this.radius)*.5;this.center.addScaledVector(Ho,s/n),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(qc.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(Ho.copy(e.center).add(qc)),this.expandByPoint(Ho.copy(e.center).sub(qc))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}const ji=new F,Yc=new F,Ka=new F,Ms=new F,Qc=new F,ja=new F,Kc=new F;let mc=class{constructor(e=new F,t=new F(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,ji)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);const n=t.dot(this.direction);return n<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const t=ji.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(ji.copy(this.origin).addScaledVector(this.direction,t),ji.distanceToSquared(e))}distanceSqToSegment(e,t,n,s){Yc.copy(e).add(t).multiplyScalar(.5),Ka.copy(t).sub(e).normalize(),Ms.copy(this.origin).sub(Yc);const r=e.distanceTo(t)*.5,o=-this.direction.dot(Ka),a=Ms.dot(this.direction),l=-Ms.dot(Ka),c=Ms.lengthSq(),u=Math.abs(1-o*o);let f,h,d,x;if(u>0)if(f=o*l-a,h=o*a-l,x=r*u,f>=0)if(h>=-x)if(h<=x){const p=1/u;f*=p,h*=p,d=f*(f+o*h+2*a)+h*(o*f+h+2*l)+c}else h=r,f=Math.max(0,-(o*h+a)),d=-f*f+h*(h+2*l)+c;else h=-r,f=Math.max(0,-(o*h+a)),d=-f*f+h*(h+2*l)+c;else h<=-x?(f=Math.max(0,-(-o*r+a)),h=f>0?-r:Math.min(Math.max(-r,-l),r),d=-f*f+h*(h+2*l)+c):h<=x?(f=0,h=Math.min(Math.max(-r,-l),r),d=h*(h+2*l)+c):(f=Math.max(0,-(o*r+a)),h=f>0?r:Math.min(Math.max(-r,-l),r),d=-f*f+h*(h+2*l)+c);else h=o>0?-r:r,f=Math.max(0,-(o*h+a)),d=-f*f+h*(h+2*l)+c;return n&&n.copy(this.origin).addScaledVector(this.direction,f),s&&s.copy(Yc).addScaledVector(Ka,h),d}intersectSphere(e,t){ji.subVectors(e.center,this.origin);const n=ji.dot(this.direction),s=ji.dot(ji)-n*n,r=e.radius*e.radius;if(s>r)return null;const o=Math.sqrt(r-s),a=n-o,l=n+o;return l<0?null:a<0?this.at(l,t):this.at(a,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(e.normal)+e.constant)/t;return n>=0?n:null}intersectPlane(e,t){const n=this.distanceToPlane(e);return n===null?null:this.at(n,t)}intersectsPlane(e){const t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let n,s,r,o,a,l;const c=1/this.direction.x,u=1/this.direction.y,f=1/this.direction.z,h=this.origin;return c>=0?(n=(e.min.x-h.x)*c,s=(e.max.x-h.x)*c):(n=(e.max.x-h.x)*c,s=(e.min.x-h.x)*c),u>=0?(r=(e.min.y-h.y)*u,o=(e.max.y-h.y)*u):(r=(e.max.y-h.y)*u,o=(e.min.y-h.y)*u),n>o||r>s||((r>n||isNaN(n))&&(n=r),(o<s||isNaN(s))&&(s=o),f>=0?(a=(e.min.z-h.z)*f,l=(e.max.z-h.z)*f):(a=(e.max.z-h.z)*f,l=(e.min.z-h.z)*f),n>l||a>s)||((a>n||n!==n)&&(n=a),(l<s||s!==s)&&(s=l),s<0)?null:this.at(n>=0?n:s,t)}intersectsBox(e){return this.intersectBox(e,ji)!==null}intersectTriangle(e,t,n,s,r){Qc.subVectors(t,e),ja.subVectors(n,e),Kc.crossVectors(Qc,ja);let o=this.direction.dot(Kc),a;if(o>0){if(s)return null;a=1}else if(o<0)a=-1,o=-o;else return null;Ms.subVectors(this.origin,e);const l=a*this.direction.dot(ja.crossVectors(Ms,ja));if(l<0)return null;const c=a*this.direction.dot(Qc.cross(Ms));if(c<0||l+c>o)return null;const u=-a*Ms.dot(Kc);return u<0?null:this.at(u/o,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}};class Ye{constructor(e,t,n,s,r,o,a,l,c,u,f,h,d,x,p,g){Ye.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c,u,f,h,d,x,p,g)}set(e,t,n,s,r,o,a,l,c,u,f,h,d,x,p,g){const m=this.elements;return m[0]=e,m[4]=t,m[8]=n,m[12]=s,m[1]=r,m[5]=o,m[9]=a,m[13]=l,m[2]=c,m[6]=u,m[10]=f,m[14]=h,m[3]=d,m[7]=x,m[11]=p,m[15]=g,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new Ye().fromArray(this.elements)}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],this}copyPosition(e){const t=this.elements,n=e.elements;return t[12]=n[12],t[13]=n[13],t[14]=n[14],this}setFromMatrix3(e){const t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,n){return e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this}makeBasis(e,t,n){return this.set(e.x,t.x,n.x,0,e.y,t.y,n.y,0,e.z,t.z,n.z,0,0,0,0,1),this}extractRotation(e){const t=this.elements,n=e.elements,s=1/Or.setFromMatrixColumn(e,0).length(),r=1/Or.setFromMatrixColumn(e,1).length(),o=1/Or.setFromMatrixColumn(e,2).length();return t[0]=n[0]*s,t[1]=n[1]*s,t[2]=n[2]*s,t[3]=0,t[4]=n[4]*r,t[5]=n[5]*r,t[6]=n[6]*r,t[7]=0,t[8]=n[8]*o,t[9]=n[9]*o,t[10]=n[10]*o,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){const t=this.elements,n=e.x,s=e.y,r=e.z,o=Math.cos(n),a=Math.sin(n),l=Math.cos(s),c=Math.sin(s),u=Math.cos(r),f=Math.sin(r);if(e.order==="XYZ"){const h=o*u,d=o*f,x=a*u,p=a*f;t[0]=l*u,t[4]=-l*f,t[8]=c,t[1]=d+x*c,t[5]=h-p*c,t[9]=-a*l,t[2]=p-h*c,t[6]=x+d*c,t[10]=o*l}else if(e.order==="YXZ"){const h=l*u,d=l*f,x=c*u,p=c*f;t[0]=h+p*a,t[4]=x*a-d,t[8]=o*c,t[1]=o*f,t[5]=o*u,t[9]=-a,t[2]=d*a-x,t[6]=p+h*a,t[10]=o*l}else if(e.order==="ZXY"){const h=l*u,d=l*f,x=c*u,p=c*f;t[0]=h-p*a,t[4]=-o*f,t[8]=x+d*a,t[1]=d+x*a,t[5]=o*u,t[9]=p-h*a,t[2]=-o*c,t[6]=a,t[10]=o*l}else if(e.order==="ZYX"){const h=o*u,d=o*f,x=a*u,p=a*f;t[0]=l*u,t[4]=x*c-d,t[8]=h*c+p,t[1]=l*f,t[5]=p*c+h,t[9]=d*c-x,t[2]=-c,t[6]=a*l,t[10]=o*l}else if(e.order==="YZX"){const h=o*l,d=o*c,x=a*l,p=a*c;t[0]=l*u,t[4]=p-h*f,t[8]=x*f+d,t[1]=f,t[5]=o*u,t[9]=-a*u,t[2]=-c*u,t[6]=d*f+x,t[10]=h-p*f}else if(e.order==="XZY"){const h=o*l,d=o*c,x=a*l,p=a*c;t[0]=l*u,t[4]=-f,t[8]=c*u,t[1]=h*f+p,t[5]=o*u,t[9]=d*f-x,t[2]=x*f-d,t[6]=a*u,t[10]=p*f+h}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(tA,e,nA)}lookAt(e,t,n){const s=this.elements;return Gn.subVectors(e,t),Gn.lengthSq()===0&&(Gn.z=1),Gn.normalize(),Ts.crossVectors(n,Gn),Ts.lengthSq()===0&&(Math.abs(n.z)===1?Gn.x+=1e-4:Gn.z+=1e-4,Gn.normalize(),Ts.crossVectors(n,Gn)),Ts.normalize(),$a.crossVectors(Gn,Ts),s[0]=Ts.x,s[4]=$a.x,s[8]=Gn.x,s[1]=Ts.y,s[5]=$a.y,s[9]=Gn.y,s[2]=Ts.z,s[6]=$a.z,s[10]=Gn.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[4],l=n[8],c=n[12],u=n[1],f=n[5],h=n[9],d=n[13],x=n[2],p=n[6],g=n[10],m=n[14],_=n[3],S=n[7],A=n[11],y=n[15],b=s[0],v=s[4],E=s[8],M=s[12],T=s[1],I=s[5],P=s[9],B=s[13],N=s[2],G=s[6],V=s[10],q=s[14],X=s[3],ee=s[7],ce=s[11],be=s[15];return r[0]=o*b+a*T+l*N+c*X,r[4]=o*v+a*I+l*G+c*ee,r[8]=o*E+a*P+l*V+c*ce,r[12]=o*M+a*B+l*q+c*be,r[1]=u*b+f*T+h*N+d*X,r[5]=u*v+f*I+h*G+d*ee,r[9]=u*E+f*P+h*V+d*ce,r[13]=u*M+f*B+h*q+d*be,r[2]=x*b+p*T+g*N+m*X,r[6]=x*v+p*I+g*G+m*ee,r[10]=x*E+p*P+g*V+m*ce,r[14]=x*M+p*B+g*q+m*be,r[3]=_*b+S*T+A*N+y*X,r[7]=_*v+S*I+A*G+y*ee,r[11]=_*E+S*P+A*V+y*ce,r[15]=_*M+S*B+A*q+y*be,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[4],s=e[8],r=e[12],o=e[1],a=e[5],l=e[9],c=e[13],u=e[2],f=e[6],h=e[10],d=e[14],x=e[3],p=e[7],g=e[11],m=e[15];return x*(+r*l*f-s*c*f-r*a*h+n*c*h+s*a*d-n*l*d)+p*(+t*l*d-t*c*h+r*o*h-s*o*d+s*c*u-r*l*u)+g*(+t*c*f-t*a*d-r*o*f+n*o*d+r*a*u-n*c*u)+m*(-s*a*u-t*l*f+t*a*h+s*o*f-n*o*h+n*l*u)}transpose(){const e=this.elements;let t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,n){const s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=t,s[14]=n),this}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=e[9],h=e[10],d=e[11],x=e[12],p=e[13],g=e[14],m=e[15],_=f*g*c-p*h*c+p*l*d-a*g*d-f*l*m+a*h*m,S=x*h*c-u*g*c-x*l*d+o*g*d+u*l*m-o*h*m,A=u*p*c-x*f*c+x*a*d-o*p*d-u*a*m+o*f*m,y=x*f*l-u*p*l-x*a*h+o*p*h+u*a*g-o*f*g,b=t*_+n*S+s*A+r*y;if(b===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const v=1/b;return e[0]=_*v,e[1]=(p*h*r-f*g*r-p*s*d+n*g*d+f*s*m-n*h*m)*v,e[2]=(a*g*r-p*l*r+p*s*c-n*g*c-a*s*m+n*l*m)*v,e[3]=(f*l*r-a*h*r-f*s*c+n*h*c+a*s*d-n*l*d)*v,e[4]=S*v,e[5]=(u*g*r-x*h*r+x*s*d-t*g*d-u*s*m+t*h*m)*v,e[6]=(x*l*r-o*g*r-x*s*c+t*g*c+o*s*m-t*l*m)*v,e[7]=(o*h*r-u*l*r+u*s*c-t*h*c-o*s*d+t*l*d)*v,e[8]=A*v,e[9]=(x*f*r-u*p*r-x*n*d+t*p*d+u*n*m-t*f*m)*v,e[10]=(o*p*r-x*a*r+x*n*c-t*p*c-o*n*m+t*a*m)*v,e[11]=(u*a*r-o*f*r-u*n*c+t*f*c+o*n*d-t*a*d)*v,e[12]=y*v,e[13]=(u*p*s-x*f*s+x*n*h-t*p*h-u*n*g+t*f*g)*v,e[14]=(x*a*s-o*p*s-x*n*l+t*p*l+o*n*g-t*a*g)*v,e[15]=(o*f*s-u*a*s+u*n*l-t*f*l-o*n*h+t*a*h)*v,this}scale(e){const t=this.elements,n=e.x,s=e.y,r=e.z;return t[0]*=n,t[4]*=s,t[8]*=r,t[1]*=n,t[5]*=s,t[9]*=r,t[2]*=n,t[6]*=s,t[10]*=r,t[3]*=n,t[7]*=s,t[11]*=r,this}getMaxScaleOnAxis(){const e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],n=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,n,s))}makeTranslation(e,t,n){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,n,0,0,0,1),this}makeRotationX(e){const t=Math.cos(e),n=Math.sin(e);return this.set(1,0,0,0,0,t,-n,0,0,n,t,0,0,0,0,1),this}makeRotationY(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,0,n,0,0,1,0,0,-n,0,t,0,0,0,0,1),this}makeRotationZ(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,0,n,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){const n=Math.cos(t),s=Math.sin(t),r=1-n,o=e.x,a=e.y,l=e.z,c=r*o,u=r*a;return this.set(c*o+n,c*a-s*l,c*l+s*a,0,c*a+s*l,u*a+n,u*l-s*o,0,c*l-s*a,u*l+s*o,r*l*l+n,0,0,0,0,1),this}makeScale(e,t,n){return this.set(e,0,0,0,0,t,0,0,0,0,n,0,0,0,0,1),this}makeShear(e,t,n,s,r,o){return this.set(1,n,r,0,e,1,o,0,t,s,1,0,0,0,0,1),this}compose(e,t,n){const s=this.elements,r=t._x,o=t._y,a=t._z,l=t._w,c=r+r,u=o+o,f=a+a,h=r*c,d=r*u,x=r*f,p=o*u,g=o*f,m=a*f,_=l*c,S=l*u,A=l*f,y=n.x,b=n.y,v=n.z;return s[0]=(1-(p+m))*y,s[1]=(d+A)*y,s[2]=(x-S)*y,s[3]=0,s[4]=(d-A)*b,s[5]=(1-(h+m))*b,s[6]=(g+_)*b,s[7]=0,s[8]=(x+S)*v,s[9]=(g-_)*v,s[10]=(1-(h+p))*v,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,t,n){const s=this.elements;let r=Or.set(s[0],s[1],s[2]).length();const o=Or.set(s[4],s[5],s[6]).length(),a=Or.set(s[8],s[9],s[10]).length();this.determinant()<0&&(r=-r),e.x=s[12],e.y=s[13],e.z=s[14],Ai.copy(this);const c=1/r,u=1/o,f=1/a;return Ai.elements[0]*=c,Ai.elements[1]*=c,Ai.elements[2]*=c,Ai.elements[4]*=u,Ai.elements[5]*=u,Ai.elements[6]*=u,Ai.elements[8]*=f,Ai.elements[9]*=f,Ai.elements[10]*=f,t.setFromRotationMatrix(Ai),n.x=r,n.y=o,n.z=a,this}makePerspective(e,t,n,s,r,o,a=Oi,l=!1){const c=this.elements,u=2*r/(t-e),f=2*r/(n-s),h=(t+e)/(t-e),d=(n+s)/(n-s);let x,p;if(l)x=r/(o-r),p=o*r/(o-r);else if(a===Oi)x=-(o+r)/(o-r),p=-2*o*r/(o-r);else if(a===Gl)x=-o/(o-r),p=-o*r/(o-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=h,c[12]=0,c[1]=0,c[5]=f,c[9]=d,c[13]=0,c[2]=0,c[6]=0,c[10]=x,c[14]=p,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(e,t,n,s,r,o,a=Oi,l=!1){const c=this.elements,u=2/(t-e),f=2/(n-s),h=-(t+e)/(t-e),d=-(n+s)/(n-s);let x,p;if(l)x=1/(o-r),p=o/(o-r);else if(a===Oi)x=-2/(o-r),p=-(o+r)/(o-r);else if(a===Gl)x=-1/(o-r),p=-r/(o-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=0,c[12]=h,c[1]=0,c[5]=f,c[9]=0,c[13]=d,c[2]=0,c[6]=0,c[10]=x,c[14]=p,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<16;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<16;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e[t+9]=n[9],e[t+10]=n[10],e[t+11]=n[11],e[t+12]=n[12],e[t+13]=n[13],e[t+14]=n[14],e[t+15]=n[15],e}}const Or=new F,Ai=new Ye,tA=new F(0,0,0),nA=new F(1,1,1),Ts=new F,$a=new F,Gn=new F,ep=new Ye,tp=new Mt;class Ei{constructor(e=0,t=0,n=0,s=Ei.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=n,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,n,s=this._order){return this._x=e,this._y=t,this._z=n,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,n=!0){const s=e.elements,r=s[0],o=s[4],a=s[8],l=s[1],c=s[5],u=s[9],f=s[2],h=s[6],d=s[10];switch(t){case"XYZ":this._y=Math.asin(tt(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-u,d),this._z=Math.atan2(-o,r)):(this._x=Math.atan2(h,c),this._z=0);break;case"YXZ":this._x=Math.asin(-tt(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(a,d),this._z=Math.atan2(l,c)):(this._y=Math.atan2(-f,r),this._z=0);break;case"ZXY":this._x=Math.asin(tt(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(-f,d),this._z=Math.atan2(-o,c)):(this._y=0,this._z=Math.atan2(l,r));break;case"ZYX":this._y=Math.asin(-tt(f,-1,1)),Math.abs(f)<.9999999?(this._x=Math.atan2(h,d),this._z=Math.atan2(l,r)):(this._x=0,this._z=Math.atan2(-o,c));break;case"YZX":this._z=Math.asin(tt(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(-u,c),this._y=Math.atan2(-f,r)):(this._x=0,this._y=Math.atan2(a,d));break;case"XZY":this._z=Math.asin(-tt(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(h,c),this._y=Math.atan2(a,r)):(this._x=Math.atan2(-u,d),this._y=0);break;default:Ze("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,n===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,n){return ep.makeRotationFromQuaternion(e),this.setFromRotationMatrix(ep,t,n)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return tp.setFromEuler(this),this.setFromQuaternion(tp,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}Ei.DEFAULT_ORDER="XYZ";class ig{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let iA=0;const np=new F,Nr=new Mt,$i=new Ye,Za=new F,Vo=new F,sA=new F,rA=new Mt,ip=new F(1,0,0),sp=new F(0,1,0),rp=new F(0,0,1),op={type:"added"},oA={type:"removed"},zr={type:"childadded",child:null},jc={type:"childremoved",child:null};class Kt extends Ks{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:iA++}),this.uuid=Ba(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=Kt.DEFAULT_UP.clone();const e=new F,t=new Ei,n=new Mt,s=new F(1,1,1);function r(){n.setFromEuler(t,!1)}function o(){t.setFromQuaternion(n,void 0,!1)}t._onChange(r),n._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new Ye},normalMatrix:{value:new Qe}}),this.matrix=new Ye,this.matrixWorld=new Ye,this.matrixAutoUpdate=Kt.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=Kt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new ig,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return Nr.setFromAxisAngle(e,t),this.quaternion.multiply(Nr),this}rotateOnWorldAxis(e,t){return Nr.setFromAxisAngle(e,t),this.quaternion.premultiply(Nr),this}rotateX(e){return this.rotateOnAxis(ip,e)}rotateY(e){return this.rotateOnAxis(sp,e)}rotateZ(e){return this.rotateOnAxis(rp,e)}translateOnAxis(e,t){return np.copy(e).applyQuaternion(this.quaternion),this.position.add(np.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(ip,e)}translateY(e){return this.translateOnAxis(sp,e)}translateZ(e){return this.translateOnAxis(rp,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4($i.copy(this.matrixWorld).invert())}lookAt(e,t,n){e.isVector3?Za.copy(e):Za.set(e,t,n);const s=this.parent;this.updateWorldMatrix(!0,!1),Vo.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?$i.lookAt(Vo,Za,this.up):$i.lookAt(Za,Vo,this.up),this.quaternion.setFromRotationMatrix($i),s&&($i.extractRotation(s.matrixWorld),Nr.setFromRotationMatrix($i),this.quaternion.premultiply(Nr.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(Wt("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(op),zr.child=e,this.dispatchEvent(zr),zr.child=null):Wt("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(oA),jc.child=e,this.dispatchEvent(jc),jc.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),$i.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),$i.multiply(e.parent.matrixWorld)),e.applyMatrix4($i),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(op),zr.child=e,this.dispatchEvent(zr),zr.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let n=0,s=this.children.length;n<s;n++){const o=this.children[n].getObjectByProperty(e,t);if(o!==void 0)return o}}getObjectsByProperty(e,t,n=[]){this[e]===t&&n.push(this);const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].getObjectsByProperty(e,t,n);return n}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Vo,e,sA),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Vo,rA,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverseVisible(e)}traverseAncestors(e){const t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].updateMatrixWorld(e)}updateWorldMatrix(e,t){const n=this.parent;if(e===!0&&n!==null&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),t===!0){const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].updateWorldMatrix(!1,!0)}}toJSON(e){const t=e===void 0||typeof e=="string",n={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(a=>({...a,boundingBox:a.boundingBox?a.boundingBox.toJSON():void 0,boundingSphere:a.boundingSphere?a.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(a=>({...a})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(a,l){return a[l.uuid]===void 0&&(a[l.uuid]=l.toJSON(e)),l.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);const a=this.geometry.parameters;if(a!==void 0&&a.shapes!==void 0){const l=a.shapes;if(Array.isArray(l))for(let c=0,u=l.length;c<u;c++){const f=l[c];r(e.shapes,f)}else r(e.shapes,l)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const a=[];for(let l=0,c=this.material.length;l<c;l++)a.push(r(e.materials,this.material[l]));s.material=a}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let a=0;a<this.children.length;a++)s.children.push(this.children[a].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let a=0;a<this.animations.length;a++){const l=this.animations[a];s.animations.push(r(e.animations,l))}}if(t){const a=o(e.geometries),l=o(e.materials),c=o(e.textures),u=o(e.images),f=o(e.shapes),h=o(e.skeletons),d=o(e.animations),x=o(e.nodes);a.length>0&&(n.geometries=a),l.length>0&&(n.materials=l),c.length>0&&(n.textures=c),u.length>0&&(n.images=u),f.length>0&&(n.shapes=f),h.length>0&&(n.skeletons=h),d.length>0&&(n.animations=d),x.length>0&&(n.nodes=x)}return n.object=s,n;function o(a){const l=[];for(const c in a){const u=a[c];delete u.metadata,l.push(u)}return l}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let n=0;n<e.children.length;n++){const s=e.children[n];this.add(s.clone())}return this}}Kt.DEFAULT_UP=new F(0,1,0);Kt.DEFAULT_MATRIX_AUTO_UPDATE=!0;Kt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const yi=new F,Zi=new F,$c=new F,Ji=new F,kr=new F,Hr=new F,ap=new F,Zc=new F,Jc=new F,eu=new F,tu=new Dt,nu=new Dt,iu=new Dt;class bi{constructor(e=new F,t=new F,n=new F){this.a=e,this.b=t,this.c=n}static getNormal(e,t,n,s){s.subVectors(n,t),yi.subVectors(e,t),s.cross(yi);const r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,t,n,s,r){yi.subVectors(s,t),Zi.subVectors(n,t),$c.subVectors(e,t);const o=yi.dot(yi),a=yi.dot(Zi),l=yi.dot($c),c=Zi.dot(Zi),u=Zi.dot($c),f=o*c-a*a;if(f===0)return r.set(0,0,0),null;const h=1/f,d=(c*l-a*u)*h,x=(o*u-a*l)*h;return r.set(1-d-x,x,d)}static containsPoint(e,t,n,s){return this.getBarycoord(e,t,n,s,Ji)===null?!1:Ji.x>=0&&Ji.y>=0&&Ji.x+Ji.y<=1}static getInterpolation(e,t,n,s,r,o,a,l){return this.getBarycoord(e,t,n,s,Ji)===null?(l.x=0,l.y=0,"z"in l&&(l.z=0),"w"in l&&(l.w=0),null):(l.setScalar(0),l.addScaledVector(r,Ji.x),l.addScaledVector(o,Ji.y),l.addScaledVector(a,Ji.z),l)}static getInterpolatedAttribute(e,t,n,s,r,o){return tu.setScalar(0),nu.setScalar(0),iu.setScalar(0),tu.fromBufferAttribute(e,t),nu.fromBufferAttribute(e,n),iu.fromBufferAttribute(e,s),o.setScalar(0),o.addScaledVector(tu,r.x),o.addScaledVector(nu,r.y),o.addScaledVector(iu,r.z),o}static isFrontFacing(e,t,n,s){return yi.subVectors(n,t),Zi.subVectors(e,t),yi.cross(Zi).dot(s)<0}set(e,t,n){return this.a.copy(e),this.b.copy(t),this.c.copy(n),this}setFromPointsAndIndices(e,t,n,s){return this.a.copy(e[t]),this.b.copy(e[n]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,t,n,s){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,n),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return yi.subVectors(this.c,this.b),Zi.subVectors(this.a,this.b),yi.cross(Zi).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return bi.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return bi.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,n,s,r){return bi.getInterpolation(e,this.a,this.b,this.c,t,n,s,r)}containsPoint(e){return bi.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return bi.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){const n=this.a,s=this.b,r=this.c;let o,a;kr.subVectors(s,n),Hr.subVectors(r,n),Zc.subVectors(e,n);const l=kr.dot(Zc),c=Hr.dot(Zc);if(l<=0&&c<=0)return t.copy(n);Jc.subVectors(e,s);const u=kr.dot(Jc),f=Hr.dot(Jc);if(u>=0&&f<=u)return t.copy(s);const h=l*f-u*c;if(h<=0&&l>=0&&u<=0)return o=l/(l-u),t.copy(n).addScaledVector(kr,o);eu.subVectors(e,r);const d=kr.dot(eu),x=Hr.dot(eu);if(x>=0&&d<=x)return t.copy(r);const p=d*c-l*x;if(p<=0&&c>=0&&x<=0)return a=c/(c-x),t.copy(n).addScaledVector(Hr,a);const g=u*x-d*f;if(g<=0&&f-u>=0&&d-x>=0)return ap.subVectors(r,s),a=(f-u)/(f-u+(d-x)),t.copy(s).addScaledVector(ap,a);const m=1/(g+p+h);return o=p*m,a=h*m,t.copy(n).addScaledVector(kr,o).addScaledVector(Hr,a)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}const sg={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Cs={h:0,s:0,l:0},Ja={h:0,s:0,l:0};function su(i,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?i+(e-i)*6*t:t<1/2?e:t<2/3?i+(e-i)*6*(2/3-t):i}class rt{constructor(e,t,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,n)}set(e,t,n){if(t===void 0&&n===void 0){const s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,t,n);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=ai){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,lt.colorSpaceToWorking(this,t),this}setRGB(e,t,n,s=lt.workingColorSpace){return this.r=e,this.g=t,this.b=n,lt.colorSpaceToWorking(this,s),this}setHSL(e,t,n,s=lt.workingColorSpace){if(e=YS(e,1),t=tt(t,0,1),n=tt(n,0,1),t===0)this.r=this.g=this.b=n;else{const r=n<=.5?n*(1+t):n+t-n*t,o=2*n-r;this.r=su(o,r,e+1/3),this.g=su(o,r,e),this.b=su(o,r,e-1/3)}return lt.colorSpaceToWorking(this,s),this}setStyle(e,t=ai){function n(r){r!==void 0&&parseFloat(r)<1&&Ze("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r;const o=s[1],a=s[2];switch(o){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,t);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,t);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,t);break;default:Ze("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){const r=s[1],o=r.length;if(o===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,t);if(o===6)return this.setHex(parseInt(r,16),t);Ze("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=ai){const n=sg[e.toLowerCase()];return n!==void 0?this.setHex(n,t):Ze("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=ms(e.r),this.g=ms(e.g),this.b=ms(e.b),this}copyLinearToSRGB(e){return this.r=co(e.r),this.g=co(e.g),this.b=co(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=ai){return lt.workingToColorSpace(ln.copy(this),e),Math.round(tt(ln.r*255,0,255))*65536+Math.round(tt(ln.g*255,0,255))*256+Math.round(tt(ln.b*255,0,255))}getHexString(e=ai){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=lt.workingColorSpace){lt.workingToColorSpace(ln.copy(this),t);const n=ln.r,s=ln.g,r=ln.b,o=Math.max(n,s,r),a=Math.min(n,s,r);let l,c;const u=(a+o)/2;if(a===o)l=0,c=0;else{const f=o-a;switch(c=u<=.5?f/(o+a):f/(2-o-a),o){case n:l=(s-r)/f+(s<r?6:0);break;case s:l=(r-n)/f+2;break;case r:l=(n-s)/f+4;break}l/=6}return e.h=l,e.s=c,e.l=u,e}getRGB(e,t=lt.workingColorSpace){return lt.workingToColorSpace(ln.copy(this),t),e.r=ln.r,e.g=ln.g,e.b=ln.b,e}getStyle(e=ai){lt.workingToColorSpace(ln.copy(this),e);const t=ln.r,n=ln.g,s=ln.b;return e!==ai?`color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(n*255)},${Math.round(s*255)})`}offsetHSL(e,t,n){return this.getHSL(Cs),this.setHSL(Cs.h+e,Cs.s+t,Cs.l+n)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,n){return this.r=e.r+(t.r-e.r)*n,this.g=e.g+(t.g-e.g)*n,this.b=e.b+(t.b-e.b)*n,this}lerpHSL(e,t){this.getHSL(Cs),e.getHSL(Ja);const n=kc(Cs.h,Ja.h,t),s=kc(Cs.s,Ja.s,t),r=kc(Cs.l,Ja.l,t);return this.setHSL(n,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const t=this.r,n=this.g,s=this.b,r=e.elements;return this.r=r[0]*t+r[3]*n+r[6]*s,this.g=r[1]*t+r[4]*n+r[7]*s,this.b=r[2]*t+r[5]*n+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const ln=new rt;rt.NAMES=sg;let aA=0;class Ua extends Ks{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:aA++}),this.uuid=Ba(),this.name="",this.type="Material",this.blending=Ns,this.side=Xi,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=xa,this.blendDst=_a,this.blendEquation=dr,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new rt(0,0,0),this.blendAlpha=0,this.depthFunc=vo,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=qd,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Pr,this.stencilZFail=Pr,this.stencilZPass=Pr,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const t in e){const n=e[t];if(n===void 0){Ze(`Material: parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){Ze(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(n):s&&s.isVector3&&n&&n.isVector3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});const n={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(e).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(e).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(e).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(e).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(e).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==Ns&&(n.blending=this.blending),this.side!==Xi&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==xa&&(n.blendSrc=this.blendSrc),this.blendDst!==_a&&(n.blendDst=this.blendDst),this.blendEquation!==dr&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==vo&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==qd&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Pr&&(n.stencilFail=this.stencilFail),this.stencilZFail!==Pr&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==Pr&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function s(r){const o=[];for(const a in r){const l=r[a];delete l.metadata,o.push(l)}return o}if(t){const r=s(e.textures),o=s(e.images);r.length>0&&(n.textures=r),o.length>0&&(n.images=o)}return n}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const t=e.clippingPlanes;let n=null;if(t!==null){const s=t.length;n=new Array(s);for(let r=0;r!==s;++r)n[r]=t[r].clone()}return this.clippingPlanes=n,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}class Mr extends Ua{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new rt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Ei,this.combine=X0,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const hs=lA();function lA(){const i=new ArrayBuffer(4),e=new Float32Array(i),t=new Uint32Array(i),n=new Uint32Array(512),s=new Uint32Array(512);for(let l=0;l<256;++l){const c=l-127;c<-27?(n[l]=0,n[l|256]=32768,s[l]=24,s[l|256]=24):c<-14?(n[l]=1024>>-c-14,n[l|256]=1024>>-c-14|32768,s[l]=-c-1,s[l|256]=-c-1):c<=15?(n[l]=c+15<<10,n[l|256]=c+15<<10|32768,s[l]=13,s[l|256]=13):c<128?(n[l]=31744,n[l|256]=64512,s[l]=24,s[l|256]=24):(n[l]=31744,n[l|256]=64512,s[l]=13,s[l|256]=13)}const r=new Uint32Array(2048),o=new Uint32Array(64),a=new Uint32Array(64);for(let l=1;l<1024;++l){let c=l<<13,u=0;for(;(c&8388608)===0;)c<<=1,u-=8388608;c&=-8388609,u+=947912704,r[l]=c|u}for(let l=1024;l<2048;++l)r[l]=939524096+(l-1024<<13);for(let l=1;l<31;++l)o[l]=l<<23;o[31]=1199570944,o[32]=2147483648;for(let l=33;l<63;++l)o[l]=2147483648+(l-32<<23);o[63]=3347054592;for(let l=1;l<64;++l)l!==32&&(a[l]=1024);return{floatView:e,uint32View:t,baseTable:n,shiftTable:s,mantissaTable:r,exponentTable:o,offsetTable:a}}function cA(i){Math.abs(i)>65504&&Ze("DataUtils.toHalfFloat(): Value out of range."),i=tt(i,-65504,65504),hs.floatView[0]=i;const e=hs.uint32View[0],t=e>>23&511;return hs.baseTable[t]+((e&8388607)>>hs.shiftTable[t])}function uA(i){const e=i>>10;return hs.uint32View[0]=hs.mantissaTable[hs.offsetTable[e]+(i&1023)]+hs.exponentTable[e],hs.floatView[0]}class ba{static toHalfFloat(e){return cA(e)}static fromHalfFloat(e){return uA(e)}}const qt=new F,el=new Pe;let fA=0;class _i{constructor(e,t,n=!1){if(Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:fA++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=n,this.usage=Yd,this.updateRanges=[],this.gpuType=Mi,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,n){e*=this.itemSize,n*=t.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=t.array[n+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,n=this.count;t<n;t++)el.fromBufferAttribute(this,t),el.applyMatrix3(e),this.setXY(t,el.x,el.y);else if(this.itemSize===3)for(let t=0,n=this.count;t<n;t++)qt.fromBufferAttribute(this,t),qt.applyMatrix3(e),this.setXYZ(t,qt.x,qt.y,qt.z);return this}applyMatrix4(e){for(let t=0,n=this.count;t<n;t++)qt.fromBufferAttribute(this,t),qt.applyMatrix4(e),this.setXYZ(t,qt.x,qt.y,qt.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)qt.fromBufferAttribute(this,t),qt.applyNormalMatrix(e),this.setXYZ(t,qt.x,qt.y,qt.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)qt.fromBufferAttribute(this,t),qt.transformDirection(e),this.setXYZ(t,qt.x,qt.y,qt.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let n=this.array[e*this.itemSize+t];return this.normalized&&(n=zo(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=In(n,this.array)),this.array[e*this.itemSize+t]=n,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=zo(t,this.array)),t}setX(e,t){return this.normalized&&(t=In(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=zo(t,this.array)),t}setY(e,t){return this.normalized&&(t=In(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=zo(t,this.array)),t}setZ(e,t){return this.normalized&&(t=In(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=zo(t,this.array)),t}setW(e,t){return this.normalized&&(t=In(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,n){return e*=this.itemSize,this.normalized&&(t=In(t,this.array),n=In(n,this.array)),this.array[e+0]=t,this.array[e+1]=n,this}setXYZ(e,t,n,s){return e*=this.itemSize,this.normalized&&(t=In(t,this.array),n=In(n,this.array),s=In(s,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this}setXYZW(e,t,n,s,r){return e*=this.itemSize,this.normalized&&(t=In(t,this.array),n=In(n,this.array),s=In(s,this.array),r=In(r,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==Yd&&(e.usage=this.usage),e}}class rg extends _i{constructor(e,t,n){super(new Uint16Array(e),t,n)}}class og extends _i{constructor(e,t,n){super(new Uint32Array(e),t,n)}}class dn extends _i{constructor(e,t,n){super(new Float32Array(e),t,n)}}let hA=0;const ri=new Ye,ru=new Kt,Vr=new F,Wn=new Ni,Go=new Ni,tn=new F;class En extends Ks{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:hA++}),this.uuid=Ba(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(tg(e)?og:rg)(e,1):this.index=e,this}setIndirect(e){return this.indirect=e,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,n=0){this.groups.push({start:e,count:t,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){const t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const r=new Qe().getNormalMatrix(e);n.applyNormalMatrix(r),n.needsUpdate=!0}const s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return ri.makeRotationFromQuaternion(e),this.applyMatrix4(ri),this}rotateX(e){return ri.makeRotationX(e),this.applyMatrix4(ri),this}rotateY(e){return ri.makeRotationY(e),this.applyMatrix4(ri),this}rotateZ(e){return ri.makeRotationZ(e),this.applyMatrix4(ri),this}translate(e,t,n){return ri.makeTranslation(e,t,n),this.applyMatrix4(ri),this}scale(e,t,n){return ri.makeScale(e,t,n),this.applyMatrix4(ri),this}lookAt(e){return ru.lookAt(e),ru.updateMatrix(),this.applyMatrix4(ru.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Vr).negate(),this.translate(Vr.x,Vr.y,Vr.z),this}setFromPoints(e){const t=this.getAttribute("position");if(t===void 0){const n=[];for(let s=0,r=e.length;s<r;s++){const o=e[s];n.push(o.x,o.y,o.z||0)}this.setAttribute("position",new dn(n,3))}else{const n=Math.min(e.length,t.count);for(let s=0;s<n;s++){const r=e[s];t.setXYZ(s,r.x,r.y,r.z||0)}e.length>t.count&&Ze("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new Ni);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Wt("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new F(-1/0,-1/0,-1/0),new F(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let n=0,s=t.length;n<s;n++){const r=t[n];Wn.setFromBufferAttribute(r),this.morphTargetsRelative?(tn.addVectors(this.boundingBox.min,Wn.min),this.boundingBox.expandByPoint(tn),tn.addVectors(this.boundingBox.max,Wn.max),this.boundingBox.expandByPoint(tn)):(this.boundingBox.expandByPoint(Wn.min),this.boundingBox.expandByPoint(Wn.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&Wt('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new pc);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Wt("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new F,1/0);return}if(e){const n=this.boundingSphere.center;if(Wn.setFromBufferAttribute(e),t)for(let r=0,o=t.length;r<o;r++){const a=t[r];Go.setFromBufferAttribute(a),this.morphTargetsRelative?(tn.addVectors(Wn.min,Go.min),Wn.expandByPoint(tn),tn.addVectors(Wn.max,Go.max),Wn.expandByPoint(tn)):(Wn.expandByPoint(Go.min),Wn.expandByPoint(Go.max))}Wn.getCenter(n);let s=0;for(let r=0,o=e.count;r<o;r++)tn.fromBufferAttribute(e,r),s=Math.max(s,n.distanceToSquared(tn));if(t)for(let r=0,o=t.length;r<o;r++){const a=t[r],l=this.morphTargetsRelative;for(let c=0,u=a.count;c<u;c++)tn.fromBufferAttribute(a,c),l&&(Vr.fromBufferAttribute(e,c),tn.add(Vr)),s=Math.max(s,n.distanceToSquared(tn))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&Wt('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){Wt("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=t.position,s=t.normal,r=t.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new _i(new Float32Array(4*n.count),4));const o=this.getAttribute("tangent"),a=[],l=[];for(let E=0;E<n.count;E++)a[E]=new F,l[E]=new F;const c=new F,u=new F,f=new F,h=new Pe,d=new Pe,x=new Pe,p=new F,g=new F;function m(E,M,T){c.fromBufferAttribute(n,E),u.fromBufferAttribute(n,M),f.fromBufferAttribute(n,T),h.fromBufferAttribute(r,E),d.fromBufferAttribute(r,M),x.fromBufferAttribute(r,T),u.sub(c),f.sub(c),d.sub(h),x.sub(h);const I=1/(d.x*x.y-x.x*d.y);isFinite(I)&&(p.copy(u).multiplyScalar(x.y).addScaledVector(f,-d.y).multiplyScalar(I),g.copy(f).multiplyScalar(d.x).addScaledVector(u,-x.x).multiplyScalar(I),a[E].add(p),a[M].add(p),a[T].add(p),l[E].add(g),l[M].add(g),l[T].add(g))}let _=this.groups;_.length===0&&(_=[{start:0,count:e.count}]);for(let E=0,M=_.length;E<M;++E){const T=_[E],I=T.start,P=T.count;for(let B=I,N=I+P;B<N;B+=3)m(e.getX(B+0),e.getX(B+1),e.getX(B+2))}const S=new F,A=new F,y=new F,b=new F;function v(E){y.fromBufferAttribute(s,E),b.copy(y);const M=a[E];S.copy(M),S.sub(y.multiplyScalar(y.dot(M))).normalize(),A.crossVectors(b,M);const I=A.dot(l[E])<0?-1:1;o.setXYZW(E,S.x,S.y,S.z,I)}for(let E=0,M=_.length;E<M;++E){const T=_[E],I=T.start,P=T.count;for(let B=I,N=I+P;B<N;B+=3)v(e.getX(B+0)),v(e.getX(B+1)),v(e.getX(B+2))}}computeVertexNormals(){const e=this.index,t=this.getAttribute("position");if(t!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new _i(new Float32Array(t.count*3),3),this.setAttribute("normal",n);else for(let h=0,d=n.count;h<d;h++)n.setXYZ(h,0,0,0);const s=new F,r=new F,o=new F,a=new F,l=new F,c=new F,u=new F,f=new F;if(e)for(let h=0,d=e.count;h<d;h+=3){const x=e.getX(h+0),p=e.getX(h+1),g=e.getX(h+2);s.fromBufferAttribute(t,x),r.fromBufferAttribute(t,p),o.fromBufferAttribute(t,g),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),a.fromBufferAttribute(n,x),l.fromBufferAttribute(n,p),c.fromBufferAttribute(n,g),a.add(u),l.add(u),c.add(u),n.setXYZ(x,a.x,a.y,a.z),n.setXYZ(p,l.x,l.y,l.z),n.setXYZ(g,c.x,c.y,c.z)}else for(let h=0,d=t.count;h<d;h+=3)s.fromBufferAttribute(t,h+0),r.fromBufferAttribute(t,h+1),o.fromBufferAttribute(t,h+2),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),n.setXYZ(h+0,u.x,u.y,u.z),n.setXYZ(h+1,u.x,u.y,u.z),n.setXYZ(h+2,u.x,u.y,u.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let t=0,n=e.count;t<n;t++)tn.fromBufferAttribute(e,t),tn.normalize(),e.setXYZ(t,tn.x,tn.y,tn.z)}toNonIndexed(){function e(a,l){const c=a.array,u=a.itemSize,f=a.normalized,h=new c.constructor(l.length*u);let d=0,x=0;for(let p=0,g=l.length;p<g;p++){a.isInterleavedBufferAttribute?d=l[p]*a.data.stride+a.offset:d=l[p]*u;for(let m=0;m<u;m++)h[x++]=c[d++]}return new _i(h,u,f)}if(this.index===null)return Ze("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const t=new En,n=this.index.array,s=this.attributes;for(const a in s){const l=s[a],c=e(l,n);t.setAttribute(a,c)}const r=this.morphAttributes;for(const a in r){const l=[],c=r[a];for(let u=0,f=c.length;u<f;u++){const h=c[u],d=e(h,n);l.push(d)}t.morphAttributes[a]=l}t.morphTargetsRelative=this.morphTargetsRelative;const o=this.groups;for(let a=0,l=o.length;a<l;a++){const c=o[a];t.addGroup(c.start,c.count,c.materialIndex)}return t}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const l=this.parameters;for(const c in l)l[c]!==void 0&&(e[c]=l[c]);return e}e.data={attributes:{}};const t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});const n=this.attributes;for(const l in n){const c=n[l];e.data.attributes[l]=c.toJSON(e.data)}const s={};let r=!1;for(const l in this.morphAttributes){const c=this.morphAttributes[l],u=[];for(let f=0,h=c.length;f<h;f++){const d=c[f];u.push(d.toJSON(e.data))}u.length>0&&(s[l]=u,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);const o=this.groups;o.length>0&&(e.data.groups=JSON.parse(JSON.stringify(o)));const a=this.boundingSphere;return a!==null&&(e.data.boundingSphere=a.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const t={};this.name=e.name;const n=e.index;n!==null&&this.setIndex(n.clone());const s=e.attributes;for(const c in s){const u=s[c];this.setAttribute(c,u.clone(t))}const r=e.morphAttributes;for(const c in r){const u=[],f=r[c];for(let h=0,d=f.length;h<d;h++)u.push(f[h].clone(t));this.morphAttributes[c]=u}this.morphTargetsRelative=e.morphTargetsRelative;const o=e.groups;for(let c=0,u=o.length;c<u;c++){const f=o[c];this.addGroup(f.start,f.count,f.materialIndex)}const a=e.boundingBox;a!==null&&(this.boundingBox=a.clone());const l=e.boundingSphere;return l!==null&&(this.boundingSphere=l.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const lp=new Ye,sr=new mc,tl=new pc,cp=new F,nl=new F,il=new F,sl=new F,ou=new F,rl=new F,up=new F,ol=new F;class Yt extends Kt{constructor(e=new En,t=new Mr){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}getVertexPosition(e,t){const n=this.geometry,s=n.attributes.position,r=n.morphAttributes.position,o=n.morphTargetsRelative;t.fromBufferAttribute(s,e);const a=this.morphTargetInfluences;if(r&&a){rl.set(0,0,0);for(let l=0,c=r.length;l<c;l++){const u=a[l],f=r[l];u!==0&&(ou.fromBufferAttribute(f,e),o?rl.addScaledVector(ou,u):rl.addScaledVector(ou.sub(t),u))}t.add(rl)}return t}raycast(e,t){const n=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),tl.copy(n.boundingSphere),tl.applyMatrix4(r),sr.copy(e.ray).recast(e.near),!(tl.containsPoint(sr.origin)===!1&&(sr.intersectSphere(tl,cp)===null||sr.origin.distanceToSquared(cp)>(e.far-e.near)**2))&&(lp.copy(r).invert(),sr.copy(e.ray).applyMatrix4(lp),!(n.boundingBox!==null&&sr.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(e,t,sr)))}_computeIntersections(e,t,n){let s;const r=this.geometry,o=this.material,a=r.index,l=r.attributes.position,c=r.attributes.uv,u=r.attributes.uv1,f=r.attributes.normal,h=r.groups,d=r.drawRange;if(a!==null)if(Array.isArray(o))for(let x=0,p=h.length;x<p;x++){const g=h[x],m=o[g.materialIndex],_=Math.max(g.start,d.start),S=Math.min(a.count,Math.min(g.start+g.count,d.start+d.count));for(let A=_,y=S;A<y;A+=3){const b=a.getX(A),v=a.getX(A+1),E=a.getX(A+2);s=al(this,m,e,n,c,u,f,b,v,E),s&&(s.faceIndex=Math.floor(A/3),s.face.materialIndex=g.materialIndex,t.push(s))}}else{const x=Math.max(0,d.start),p=Math.min(a.count,d.start+d.count);for(let g=x,m=p;g<m;g+=3){const _=a.getX(g),S=a.getX(g+1),A=a.getX(g+2);s=al(this,o,e,n,c,u,f,_,S,A),s&&(s.faceIndex=Math.floor(g/3),t.push(s))}}else if(l!==void 0)if(Array.isArray(o))for(let x=0,p=h.length;x<p;x++){const g=h[x],m=o[g.materialIndex],_=Math.max(g.start,d.start),S=Math.min(l.count,Math.min(g.start+g.count,d.start+d.count));for(let A=_,y=S;A<y;A+=3){const b=A,v=A+1,E=A+2;s=al(this,m,e,n,c,u,f,b,v,E),s&&(s.faceIndex=Math.floor(A/3),s.face.materialIndex=g.materialIndex,t.push(s))}}else{const x=Math.max(0,d.start),p=Math.min(l.count,d.start+d.count);for(let g=x,m=p;g<m;g+=3){const _=g,S=g+1,A=g+2;s=al(this,o,e,n,c,u,f,_,S,A),s&&(s.faceIndex=Math.floor(g/3),t.push(s))}}}}function dA(i,e,t,n,s,r,o,a){let l;if(e.side===Bn?l=n.intersectTriangle(o,r,s,!0,a):l=n.intersectTriangle(s,r,o,e.side===Xi,a),l===null)return null;ol.copy(a),ol.applyMatrix4(i.matrixWorld);const c=t.ray.origin.distanceTo(ol);return c<t.near||c>t.far?null:{distance:c,point:ol.clone(),object:i}}function al(i,e,t,n,s,r,o,a,l,c){i.getVertexPosition(a,nl),i.getVertexPosition(l,il),i.getVertexPosition(c,sl);const u=dA(i,e,t,n,nl,il,sl,up);if(u){const f=new F;bi.getBarycoord(up,nl,il,sl,f),s&&(u.uv=bi.getInterpolatedAttribute(s,a,l,c,f,new Pe)),r&&(u.uv1=bi.getInterpolatedAttribute(r,a,l,c,f,new Pe)),o&&(u.normal=bi.getInterpolatedAttribute(o,a,l,c,f,new F),u.normal.dot(n.direction)>0&&u.normal.multiplyScalar(-1));const h={a,b:l,c,normal:new F,materialIndex:0};bi.getNormal(nl,il,sl,h.normal),u.face=h,u.barycoord=f}return u}class Fo extends En{constructor(e=1,t=1,n=1,s=1,r=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:n,widthSegments:s,heightSegments:r,depthSegments:o};const a=this;s=Math.floor(s),r=Math.floor(r),o=Math.floor(o);const l=[],c=[],u=[],f=[];let h=0,d=0;x("z","y","x",-1,-1,n,t,e,o,r,0),x("z","y","x",1,-1,n,t,-e,o,r,1),x("x","z","y",1,1,e,n,t,s,o,2),x("x","z","y",1,-1,e,n,-t,s,o,3),x("x","y","z",1,-1,e,t,n,s,r,4),x("x","y","z",-1,-1,e,t,-n,s,r,5),this.setIndex(l),this.setAttribute("position",new dn(c,3)),this.setAttribute("normal",new dn(u,3)),this.setAttribute("uv",new dn(f,2));function x(p,g,m,_,S,A,y,b,v,E,M){const T=A/v,I=y/E,P=A/2,B=y/2,N=b/2,G=v+1,V=E+1;let q=0,X=0;const ee=new F;for(let ce=0;ce<V;ce++){const be=ce*I-B;for(let Re=0;Re<G;Re++){const Fe=Re*T-P;ee[p]=Fe*_,ee[g]=be*S,ee[m]=N,c.push(ee.x,ee.y,ee.z),ee[p]=0,ee[g]=0,ee[m]=b>0?1:-1,u.push(ee.x,ee.y,ee.z),f.push(Re/v),f.push(1-ce/E),q+=1}}for(let ce=0;ce<E;ce++)for(let be=0;be<v;be++){const Re=h+be+G*ce,Fe=h+be+G*(ce+1),Oe=h+(be+1)+G*(ce+1),Ne=h+(be+1)+G*ce;l.push(Re,Fe,Ne),l.push(Fe,Oe,Ne),X+=6}a.addGroup(d,X,M),d+=X,h+=q}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Fo(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}function Mo(i){const e={};for(const t in i){e[t]={};for(const n in i[t]){const s=i[t][n];s&&(s.isColor||s.isMatrix3||s.isMatrix4||s.isVector2||s.isVector3||s.isVector4||s.isTexture||s.isQuaternion)?s.isRenderTargetTexture?(Ze("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][n]=null):e[t][n]=s.clone():Array.isArray(s)?e[t][n]=s.slice():e[t][n]=s}}return e}function vn(i){const e={};for(let t=0;t<i.length;t++){const n=Mo(i[t]);for(const s in n)e[s]=n[s]}return e}function pA(i){const e=[];for(let t=0;t<i.length;t++)e.push(i[t].clone());return e}function ag(i){const e=i.getRenderTarget();return e===null?i.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:lt.workingColorSpace}const mA={clone:Mo,merge:vn};var gA=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,xA=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class Cn extends Ua{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=gA,this.fragmentShader=xA,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=Mo(e.uniforms),this.uniformsGroups=pA(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this}toJSON(e){const t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(const s in this.uniforms){const o=this.uniforms[s].value;o&&o.isTexture?t.uniforms[s]={type:"t",value:o.toJSON(e).uuid}:o&&o.isColor?t.uniforms[s]={type:"c",value:o.getHex()}:o&&o.isVector2?t.uniforms[s]={type:"v2",value:o.toArray()}:o&&o.isVector3?t.uniforms[s]={type:"v3",value:o.toArray()}:o&&o.isVector4?t.uniforms[s]={type:"v4",value:o.toArray()}:o&&o.isMatrix3?t.uniforms[s]={type:"m3",value:o.toArray()}:o&&o.isMatrix4?t.uniforms[s]={type:"m4",value:o.toArray()}:t.uniforms[s]={value:o}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;const n={};for(const s in this.extensions)this.extensions[s]===!0&&(n[s]=!0);return Object.keys(n).length>0&&(t.extensions=n),t}}class lg extends Kt{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new Ye,this.projectionMatrix=new Ye,this.projectionMatrixInverse=new Ye,this.coordinateSystem=Oi,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const Es=new F,fp=new Pe,hp=new Pe;class ui extends lg{constructor(e=50,t=1,n=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=n,this.far=s,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const t=.5*this.getFilmHeight()/e;this.fov=Df*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(Dl*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return Df*2*Math.atan(Math.tan(Dl*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,n){Es.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(Es.x,Es.y).multiplyScalar(-e/Es.z),Es.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(Es.x,Es.y).multiplyScalar(-e/Es.z)}getViewSize(e,t){return this.getViewBounds(e,fp,hp),t.subVectors(hp,fp)}setViewOffset(e,t,n,s,r,o){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let t=e*Math.tan(Dl*.5*this.fov)/this.zoom,n=2*t,s=this.aspect*n,r=-.5*s;const o=this.view;if(this.view!==null&&this.view.enabled){const l=o.fullWidth,c=o.fullHeight;r+=o.offsetX*s/l,t-=o.offsetY*n/c,s*=o.width/l,n*=o.height/c}const a=this.filmOffset;a!==0&&(r+=e*a/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,t,t-n,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}}const Gr=-90,Wr=1;class _A extends Kt{constructor(e,t,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const s=new ui(Gr,Wr,e,t);s.layers=this.layers,this.add(s);const r=new ui(Gr,Wr,e,t);r.layers=this.layers,this.add(r);const o=new ui(Gr,Wr,e,t);o.layers=this.layers,this.add(o);const a=new ui(Gr,Wr,e,t);a.layers=this.layers,this.add(a);const l=new ui(Gr,Wr,e,t);l.layers=this.layers,this.add(l);const c=new ui(Gr,Wr,e,t);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){const e=this.coordinateSystem,t=this.children.concat(),[n,s,r,o,a,l]=t;for(const c of t)this.remove(c);if(e===Oi)n.up.set(0,1,0),n.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),o.up.set(0,0,1),o.lookAt(0,-1,0),a.up.set(0,1,0),a.lookAt(0,0,1),l.up.set(0,1,0),l.lookAt(0,0,-1);else if(e===Gl)n.up.set(0,-1,0),n.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),o.up.set(0,0,-1),o.lookAt(0,-1,0),a.up.set(0,-1,0),a.lookAt(0,0,1),l.up.set(0,-1,0),l.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const c of t)this.add(c),c.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[r,o,a,l,c,u]=this.children,f=e.getRenderTarget(),h=e.getActiveCubeFace(),d=e.getActiveMipmapLevel(),x=e.xr.enabled;e.xr.enabled=!1;const p=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,e.setRenderTarget(n,0,s),e.render(t,r),e.setRenderTarget(n,1,s),e.render(t,o),e.setRenderTarget(n,2,s),e.render(t,a),e.setRenderTarget(n,3,s),e.render(t,l),e.setRenderTarget(n,4,s),e.render(t,c),n.texture.generateMipmaps=p,e.setRenderTarget(n,5,s),e.render(t,u),e.setRenderTarget(f,h,d),e.xr.enabled=x,n.texture.needsPMREMUpdate=!0}}class cg extends Tn{constructor(e=[],t=So,n,s,r,o,a,l,c,u){super(e,t,n,s,r,o,a,l,c,u),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class vA extends Ws{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;const n={width:e,height:e,depth:1},s=[n,n,n,n,n,n];this.texture=new cg(s),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

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
			`},s=new Fo(5,5,5),r=new Cn({name:"CubemapFromEquirect",uniforms:Mo(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:Bn,blending:ps});r.uniforms.tEquirect.value=t;const o=new Yt(s,r),a=t.minFilter;return t.minFilter===mr&&(t.minFilter=di),new _A(1,10,this).update(e,o),t.minFilter=a,o.geometry.dispose(),o.material.dispose(),this}clear(e,t=!0,n=!0,s=!0){const r=e.getRenderTarget();for(let o=0;o<6;o++)e.setRenderTarget(this,o),e.clear(t,n,s);e.setRenderTarget(r)}}class ll extends Kt{constructor(){super(),this.isGroup=!0,this.type="Group"}}const SA={type:"move"};class au{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new ll,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new ll,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new F,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new F),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new ll,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new F,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new F),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const t=this._hand;if(t)for(const n of e.hand.values())this._getHandJoint(t,n)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,n){let s=null,r=null,o=null;const a=this._targetRay,l=this._grip,c=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(c&&e.hand){o=!0;for(const p of e.hand.values()){const g=t.getJointPose(p,n),m=this._getHandJoint(c,p);g!==null&&(m.matrix.fromArray(g.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,m.jointRadius=g.radius),m.visible=g!==null}const u=c.joints["index-finger-tip"],f=c.joints["thumb-tip"],h=u.position.distanceTo(f.position),d=.02,x=.005;c.inputState.pinching&&h>d+x?(c.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!c.inputState.pinching&&h<=d-x&&(c.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else l!==null&&e.gripSpace&&(r=t.getPose(e.gripSpace,n),r!==null&&(l.matrix.fromArray(r.transform.matrix),l.matrix.decompose(l.position,l.rotation,l.scale),l.matrixWorldNeedsUpdate=!0,r.linearVelocity?(l.hasLinearVelocity=!0,l.linearVelocity.copy(r.linearVelocity)):l.hasLinearVelocity=!1,r.angularVelocity?(l.hasAngularVelocity=!0,l.angularVelocity.copy(r.angularVelocity)):l.hasAngularVelocity=!1));a!==null&&(s=t.getPose(e.targetRaySpace,n),s===null&&r!==null&&(s=r),s!==null&&(a.matrix.fromArray(s.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),a.matrixWorldNeedsUpdate=!0,s.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(s.linearVelocity)):a.hasLinearVelocity=!1,s.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(s.angularVelocity)):a.hasAngularVelocity=!1,this.dispatchEvent(SA)))}return a!==null&&(a.visible=s!==null),l!==null&&(l.visible=r!==null),c!==null&&(c.visible=o!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){const n=new ll;n.matrixAutoUpdate=!1,n.visible=!1,e.joints[t.jointName]=n,e.add(n)}return e.joints[t.jointName]}}class AA extends Kt{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Ei,this.environmentIntensity=1,this.environmentRotation=new Ei,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}}class is extends Tn{constructor(e=null,t=1,n=1,s,r,o,a,l,c=Jn,u=Jn,f,h){super(null,o,a,l,c,u,s,r,f,h),this.isDataTexture=!0,this.image={data:e,width:t,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class yA extends _i{constructor(e,t,n,s=1){super(e,t,n),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=s}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}toJSON(){const e=super.toJSON();return e.meshPerAttribute=this.meshPerAttribute,e.isInstancedBufferAttribute=!0,e}}const lu=new F,bA=new F,MA=new Qe;class as{constructor(e=new F(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,n,s){return this.normal.set(e,t,n),this.constant=s,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,n){const s=lu.subVectors(n,t).cross(bA.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t){const n=e.delta(lu),s=this.normal.dot(n);if(s===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;const r=-(e.start.dot(this.normal)+this.constant)/s;return r<0||r>1?null:t.copy(e.start).addScaledVector(n,r)}intersectsLine(e){const t=this.distanceToPoint(e.start),n=this.distanceToPoint(e.end);return t<0&&n>0||n<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){const n=t||MA.getNormalMatrix(e),s=this.coplanarPoint(lu).applyMatrix4(e),r=this.normal.applyMatrix3(n).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const rr=new pc,TA=new Pe(.5,.5),cl=new F;class ug{constructor(e=new as,t=new as,n=new as,s=new as,r=new as,o=new as){this.planes=[e,t,n,s,r,o]}set(e,t,n,s,r,o){const a=this.planes;return a[0].copy(e),a[1].copy(t),a[2].copy(n),a[3].copy(s),a[4].copy(r),a[5].copy(o),this}copy(e){const t=this.planes;for(let n=0;n<6;n++)t[n].copy(e.planes[n]);return this}setFromProjectionMatrix(e,t=Oi,n=!1){const s=this.planes,r=e.elements,o=r[0],a=r[1],l=r[2],c=r[3],u=r[4],f=r[5],h=r[6],d=r[7],x=r[8],p=r[9],g=r[10],m=r[11],_=r[12],S=r[13],A=r[14],y=r[15];if(s[0].setComponents(c-o,d-u,m-x,y-_).normalize(),s[1].setComponents(c+o,d+u,m+x,y+_).normalize(),s[2].setComponents(c+a,d+f,m+p,y+S).normalize(),s[3].setComponents(c-a,d-f,m-p,y-S).normalize(),n)s[4].setComponents(l,h,g,A).normalize(),s[5].setComponents(c-l,d-h,m-g,y-A).normalize();else if(s[4].setComponents(c-l,d-h,m-g,y-A).normalize(),t===Oi)s[5].setComponents(c+l,d+h,m+g,y+A).normalize();else if(t===Gl)s[5].setComponents(l,h,g,A).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),rr.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),rr.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(rr)}intersectsSprite(e){rr.center.set(0,0,0);const t=TA.distanceTo(e.center);return rr.radius=.7071067811865476+t,rr.applyMatrix4(e.matrixWorld),this.intersectsSphere(rr)}intersectsSphere(e){const t=this.planes,n=e.center,s=-e.radius;for(let r=0;r<6;r++)if(t[r].distanceToPoint(n)<s)return!1;return!0}intersectsBox(e){const t=this.planes;for(let n=0;n<6;n++){const s=t[n];if(cl.x=s.normal.x>0?e.max.x:e.min.x,cl.y=s.normal.y>0?e.max.y:e.min.y,cl.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(cl)<0)return!1}return!0}containsPoint(e){const t=this.planes;for(let n=0;n<6;n++)if(t[n].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class CA extends Ua{constructor(e){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new rt(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.size=e.size,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}}const dp=new Ye,Pf=new mc,ul=new pc,fl=new F;class EA extends Kt{constructor(e=new En,t=new CA){super(),this.isPoints=!0,this.type="Points",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}raycast(e,t){const n=this.geometry,s=this.matrixWorld,r=e.params.Points.threshold,o=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),ul.copy(n.boundingSphere),ul.applyMatrix4(s),ul.radius+=r,e.ray.intersectsSphere(ul)===!1)return;dp.copy(s).invert(),Pf.copy(e.ray).applyMatrix4(dp);const a=r/((this.scale.x+this.scale.y+this.scale.z)/3),l=a*a,c=n.index,f=n.attributes.position;if(c!==null){const h=Math.max(0,o.start),d=Math.min(c.count,o.start+o.count);for(let x=h,p=d;x<p;x++){const g=c.getX(x);fl.fromBufferAttribute(f,g),pp(fl,g,l,s,e,t,this)}}else{const h=Math.max(0,o.start),d=Math.min(f.count,o.start+o.count);for(let x=h,p=d;x<p;x++)fl.fromBufferAttribute(f,x),pp(fl,x,l,s,e,t,this)}}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}}function pp(i,e,t,n,s,r,o){const a=Pf.distanceSqToPoint(i);if(a<t){const l=new F;Pf.closestPointToPoint(i,l),l.applyMatrix4(n);const c=s.ray.origin.distanceTo(l);if(c<s.near||c>s.far)return;r.push({distance:c,distanceToRay:Math.sqrt(a),point:l,index:e,face:null,faceIndex:null,barycoord:null,object:o})}}class Mh extends Tn{constructor(e,t,n=pi,s,r,o,a=Jn,l=Jn,c,u=yo,f=1){if(u!==yo&&u!==Aa)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const h={width:e,height:t,depth:f};super(h,s,r,o,a,l,u,n,c),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new bh(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}}class fg extends Tn{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class Ma extends En{constructor(e=1,t=1,n=1,s=32,r=1,o=!1,a=0,l=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:n,radialSegments:s,heightSegments:r,openEnded:o,thetaStart:a,thetaLength:l};const c=this;s=Math.floor(s),r=Math.floor(r);const u=[],f=[],h=[],d=[];let x=0;const p=[],g=n/2;let m=0;_(),o===!1&&(e>0&&S(!0),t>0&&S(!1)),this.setIndex(u),this.setAttribute("position",new dn(f,3)),this.setAttribute("normal",new dn(h,3)),this.setAttribute("uv",new dn(d,2));function _(){const A=new F,y=new F;let b=0;const v=(t-e)/n;for(let E=0;E<=r;E++){const M=[],T=E/r,I=T*(t-e)+e;for(let P=0;P<=s;P++){const B=P/s,N=B*l+a,G=Math.sin(N),V=Math.cos(N);y.x=I*G,y.y=-T*n+g,y.z=I*V,f.push(y.x,y.y,y.z),A.set(G,v,V).normalize(),h.push(A.x,A.y,A.z),d.push(B,1-T),M.push(x++)}p.push(M)}for(let E=0;E<s;E++)for(let M=0;M<r;M++){const T=p[M][E],I=p[M+1][E],P=p[M+1][E+1],B=p[M][E+1];(e>0||M!==0)&&(u.push(T,I,B),b+=3),(t>0||M!==r-1)&&(u.push(I,P,B),b+=3)}c.addGroup(m,b,0),m+=b}function S(A){const y=x,b=new Pe,v=new F;let E=0;const M=A===!0?e:t,T=A===!0?1:-1;for(let P=1;P<=s;P++)f.push(0,g*T,0),h.push(0,T,0),d.push(.5,.5),x++;const I=x;for(let P=0;P<=s;P++){const N=P/s*l+a,G=Math.cos(N),V=Math.sin(N);v.x=M*V,v.y=g*T,v.z=M*G,f.push(v.x,v.y,v.z),h.push(0,T,0),b.x=G*.5+.5,b.y=V*.5*T+.5,d.push(b.x,b.y),x++}for(let P=0;P<s;P++){const B=y+P,N=I+P;A===!0?u.push(N,N+1,B):u.push(N+1,N,B),E+=3}c.addGroup(m,E,A===!0?1:2),m+=E}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Ma(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class Th extends Ma{constructor(e=1,t=1,n=32,s=1,r=!1,o=0,a=Math.PI*2){super(0,e,t,n,s,r,o,a),this.type="ConeGeometry",this.parameters={radius:e,height:t,radialSegments:n,heightSegments:s,openEnded:r,thetaStart:o,thetaLength:a}}static fromJSON(e){return new Th(e.radius,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class To extends En{constructor(e=1,t=1,n=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:n,heightSegments:s};const r=e/2,o=t/2,a=Math.floor(n),l=Math.floor(s),c=a+1,u=l+1,f=e/a,h=t/l,d=[],x=[],p=[],g=[];for(let m=0;m<u;m++){const _=m*h-o;for(let S=0;S<c;S++){const A=S*f-r;x.push(A,-_,0),p.push(0,0,1),g.push(S/a),g.push(1-m/l)}}for(let m=0;m<l;m++)for(let _=0;_<a;_++){const S=_+c*m,A=_+c*(m+1),y=_+1+c*(m+1),b=_+1+c*m;d.push(S,A,b),d.push(A,y,b)}this.setIndex(d),this.setAttribute("position",new dn(x,3)),this.setAttribute("normal",new dn(p,3)),this.setAttribute("uv",new dn(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new To(e.width,e.height,e.widthSegments,e.heightSegments)}}class Xl extends En{constructor(e=1,t=32,n=16,s=0,r=Math.PI*2,o=0,a=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:e,widthSegments:t,heightSegments:n,phiStart:s,phiLength:r,thetaStart:o,thetaLength:a},t=Math.max(3,Math.floor(t)),n=Math.max(2,Math.floor(n));const l=Math.min(o+a,Math.PI);let c=0;const u=[],f=new F,h=new F,d=[],x=[],p=[],g=[];for(let m=0;m<=n;m++){const _=[],S=m/n;let A=0;m===0&&o===0?A=.5/t:m===n&&l===Math.PI&&(A=-.5/t);for(let y=0;y<=t;y++){const b=y/t;f.x=-e*Math.cos(s+b*r)*Math.sin(o+S*a),f.y=e*Math.cos(o+S*a),f.z=e*Math.sin(s+b*r)*Math.sin(o+S*a),x.push(f.x,f.y,f.z),h.copy(f).normalize(),p.push(h.x,h.y,h.z),g.push(b+A,1-S),_.push(c++)}u.push(_)}for(let m=0;m<n;m++)for(let _=0;_<t;_++){const S=u[m][_+1],A=u[m][_],y=u[m+1][_],b=u[m+1][_+1];(m!==0||o>0)&&d.push(S,A,b),(m!==n-1||l<Math.PI)&&d.push(A,y,b)}this.setIndex(d),this.setAttribute("position",new dn(x,3)),this.setAttribute("normal",new dn(p,3)),this.setAttribute("uv",new dn(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Xl(e.radius,e.widthSegments,e.heightSegments,e.phiStart,e.phiLength,e.thetaStart,e.thetaLength)}}class wA extends Ua{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=FS,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class RA extends Ua{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}class Ch extends lg{constructor(e=-1,t=1,n=1,s=-1,r=.1,o=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=n,this.bottom=s,this.near=r,this.far=o,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,n,s,r,o){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,s=(this.top+this.bottom)/2;let r=n-e,o=n+e,a=s+t,l=s-t;if(this.view!==null&&this.view.enabled){const c=(this.right-this.left)/this.view.fullWidth/this.zoom,u=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=c*this.view.offsetX,o=r+c*this.view.width,a-=u*this.view.offsetY,l=a-u*this.view.height}this.projectionMatrix.makeOrthographic(r,o,a,l,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}}class IA extends En{constructor(){super(),this.isInstancedBufferGeometry=!0,this.type="InstancedBufferGeometry",this.instanceCount=1/0}copy(e){return super.copy(e),this.instanceCount=e.instanceCount,this}toJSON(){const e=super.toJSON();return e.instanceCount=this.instanceCount,e.isInstancedBufferGeometry=!0,e}}class DA extends ui{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}class ql{constructor(e=1,t=0,n=0){this.radius=e,this.phi=t,this.theta=n}set(e,t,n){return this.radius=e,this.phi=t,this.theta=n,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=tt(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,n){return this.radius=Math.sqrt(e*e+t*t+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,n),this.phi=Math.acos(tt(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}class PA extends Ks{constructor(e,t=null){super(),this.object=e,this.domElement=t,this.enabled=!0,this.state=-1,this.keys={},this.mouseButtons={LEFT:null,MIDDLE:null,RIGHT:null},this.touches={ONE:null,TWO:null}}connect(e){if(e===void 0){Ze("Controls: connect() now requires an element.");return}this.domElement!==null&&this.disconnect(),this.domElement=e}disconnect(){}dispose(){}update(){}}function mp(i,e,t,n){const s=FA(n);switch(t){case $0:return i*e;case J0:return i*e/s.components*s.byteLength;case dc:return i*e/s.components*s.byteLength;case Sh:return i*e*2/s.components*s.byteLength;case Ah:return i*e*2/s.components*s.byteLength;case Z0:return i*e*3/s.components*s.byteLength;case Mn:return i*e*4/s.components*s.byteLength;case lo:return i*e*4/s.components*s.byteLength;case El:case wl:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case Rl:case Il:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case sf:case of:return Math.max(i,16)*Math.max(e,8)/4;case nf:case rf:return Math.max(i,8)*Math.max(e,8)/2;case af:case lf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case cf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case uf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case ff:return Math.floor((i+4)/5)*Math.floor((e+3)/4)*16;case hf:return Math.floor((i+4)/5)*Math.floor((e+4)/5)*16;case df:return Math.floor((i+5)/6)*Math.floor((e+4)/5)*16;case pf:return Math.floor((i+5)/6)*Math.floor((e+5)/6)*16;case mf:return Math.floor((i+7)/8)*Math.floor((e+4)/5)*16;case gf:return Math.floor((i+7)/8)*Math.floor((e+5)/6)*16;case xf:return Math.floor((i+7)/8)*Math.floor((e+7)/8)*16;case _f:return Math.floor((i+9)/10)*Math.floor((e+4)/5)*16;case vf:return Math.floor((i+9)/10)*Math.floor((e+5)/6)*16;case Sf:return Math.floor((i+9)/10)*Math.floor((e+7)/8)*16;case Af:return Math.floor((i+9)/10)*Math.floor((e+9)/10)*16;case yf:return Math.floor((i+11)/12)*Math.floor((e+9)/10)*16;case bf:return Math.floor((i+11)/12)*Math.floor((e+11)/12)*16;case Mf:case Tf:case Cf:return Math.ceil(i/4)*Math.ceil(e/4)*16;case Ef:case wf:return Math.ceil(i/4)*Math.ceil(e/4)*8;case Rf:case If:return Math.ceil(i/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function FA(i){switch(i){case qi:case Y0:return{byteLength:1,components:1};case va:case Q0:case Tr:return{byteLength:2,components:1};case _h:case vh:return{byteLength:2,components:4};case pi:case xh:case Mi:return{byteLength:4,components:1};case K0:case j0:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${i}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:gh}}));typeof window<"u"&&(window.__THREE__?Ze("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=gh);function hg(){let i=null,e=!1,t=null,n=null;function s(r,o){t(r,o),n=i.requestAnimationFrame(s)}return{start:function(){e!==!0&&t!==null&&(n=i.requestAnimationFrame(s),e=!0)},stop:function(){i.cancelAnimationFrame(n),e=!1},setAnimationLoop:function(r){t=r},setContext:function(r){i=r}}}function LA(i){const e=new WeakMap;function t(a,l){const c=a.array,u=a.usage,f=c.byteLength,h=i.createBuffer();i.bindBuffer(l,h),i.bufferData(l,c,u),a.onUploadCallback();let d;if(c instanceof Float32Array)d=i.FLOAT;else if(typeof Float16Array<"u"&&c instanceof Float16Array)d=i.HALF_FLOAT;else if(c instanceof Uint16Array)a.isFloat16BufferAttribute?d=i.HALF_FLOAT:d=i.UNSIGNED_SHORT;else if(c instanceof Int16Array)d=i.SHORT;else if(c instanceof Uint32Array)d=i.UNSIGNED_INT;else if(c instanceof Int32Array)d=i.INT;else if(c instanceof Int8Array)d=i.BYTE;else if(c instanceof Uint8Array)d=i.UNSIGNED_BYTE;else if(c instanceof Uint8ClampedArray)d=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+c);return{buffer:h,type:d,bytesPerElement:c.BYTES_PER_ELEMENT,version:a.version,size:f}}function n(a,l,c){const u=l.array,f=l.updateRanges;if(i.bindBuffer(c,a),f.length===0)i.bufferSubData(c,0,u);else{f.sort((d,x)=>d.start-x.start);let h=0;for(let d=1;d<f.length;d++){const x=f[h],p=f[d];p.start<=x.start+x.count+1?x.count=Math.max(x.count,p.start+p.count-x.start):(++h,f[h]=p)}f.length=h+1;for(let d=0,x=f.length;d<x;d++){const p=f[d];i.bufferSubData(c,p.start*u.BYTES_PER_ELEMENT,u,p.start,p.count)}l.clearUpdateRanges()}l.onUploadCallback()}function s(a){return a.isInterleavedBufferAttribute&&(a=a.data),e.get(a)}function r(a){a.isInterleavedBufferAttribute&&(a=a.data);const l=e.get(a);l&&(i.deleteBuffer(l.buffer),e.delete(a))}function o(a,l){if(a.isInterleavedBufferAttribute&&(a=a.data),a.isGLBufferAttribute){const u=e.get(a);(!u||u.version<a.version)&&e.set(a,{buffer:a.buffer,type:a.type,bytesPerElement:a.elementSize,version:a.version});return}const c=e.get(a);if(c===void 0)e.set(a,t(a,l));else if(c.version<a.version){if(c.size!==a.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(c.buffer,a,l),c.version=a.version}}return{get:s,remove:r,update:o}}var BA=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,UA=`#ifdef USE_ALPHAHASH
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
#endif`,OA=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,NA=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,zA=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,kA=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,HA=`#ifdef USE_AOMAP
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
#endif`,VA=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,GA=`#ifdef USE_BATCHING
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
#endif`,WA=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,XA=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,qA=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,YA=`float G_BlinnPhong_Implicit( ) {
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
} // validated`,QA=`#ifdef USE_IRIDESCENCE
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
#endif`,KA=`#ifdef USE_BUMPMAP
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
#endif`,jA=`#if NUM_CLIPPING_PLANES > 0
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
#endif`,$A=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,ZA=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,JA=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,ey=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,ty=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,ny=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,iy=`#if defined( USE_COLOR_ALPHA )
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
#endif`,sy=`#define PI 3.141592653589793
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
} // validated`,ry=`#ifdef ENVMAP_TYPE_CUBE_UV
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
#endif`,oy=`vec3 transformedNormal = objectNormal;
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
#endif`,ay=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,ly=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,cy=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,uy=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,fy="gl_FragColor = linearToOutputTexel( gl_FragColor );",hy=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,dy=`#ifdef USE_ENVMAP
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
#endif`,py=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,my=`#ifdef USE_ENVMAP
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
#endif`,gy=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,xy=`#ifdef USE_ENVMAP
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
#endif`,_y=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,vy=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,Sy=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,Ay=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,yy=`#ifdef USE_GRADIENTMAP
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
}`,by=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,My=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,Ty=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,Cy=`uniform bool receiveShadow;
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
#endif`,Ey=`#ifdef USE_ENVMAP
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
#endif`,wy=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,Ry=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,Iy=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,Dy=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,Py=`PhysicalMaterial material;
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
#endif`,Fy=`uniform sampler2D dfgLUT;
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
}`,Ly=`
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
#endif`,By=`#if defined( RE_IndirectDiffuse )
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
#endif`,Uy=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,Oy=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,Ny=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,zy=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,ky=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,Hy=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Vy=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,Gy=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
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
#endif`,Wy=`#if defined( USE_POINTS_UV )
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
#endif`,Xy=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,qy=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,Yy=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,Qy=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Ky=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,jy=`#ifdef USE_MORPHTARGETS
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
#endif`,$y=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Zy=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
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
vec3 nonPerturbedNormal = normal;`,Jy=`#ifdef USE_NORMALMAP_OBJECTSPACE
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
#endif`,eb=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,tb=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,nb=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,ib=`#ifdef USE_NORMALMAP
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
#endif`,sb=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,rb=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,ob=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,ab=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,lb=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,cb=`vec3 packNormalToRGB( const in vec3 normal ) {
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
}`,ub=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,fb=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,hb=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,db=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,pb=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,mb=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,gb=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,xb=`#if NUM_SPOT_LIGHT_COORDS > 0
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
#endif`,_b=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
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
#endif`,vb=`float getShadowMask() {
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
}`,Sb=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,Ab=`#ifdef USE_SKINNING
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
#endif`,yb=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,bb=`#ifdef USE_SKINNING
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
#endif`,Mb=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,Tb=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,Cb=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,Eb=`#ifndef saturate
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
vec3 CustomToneMapping( vec3 color ) { return color; }`,wb=`#ifdef USE_TRANSMISSION
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
#endif`,Rb=`#ifdef USE_TRANSMISSION
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
#endif`,Ib=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,Db=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,Pb=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#endif`,Fb=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const Lb=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,Bb=`uniform sampler2D t2D;
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
}`,Ub=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Ob=`#ifdef ENVMAP_TYPE_CUBE
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
}`,Nb=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,zb=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,kb=`#include <common>
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
}`,Hb=`#if DEPTH_PACKING == 3200
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
}`,Vb=`#define DISTANCE
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
}`,Gb=`#define DISTANCE
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
}`,Wb=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Xb=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,qb=`uniform float scale;
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
}`,Yb=`uniform vec3 diffuse;
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
}`,Qb=`#include <common>
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
}`,Kb=`uniform vec3 diffuse;
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
}`,jb=`#define LAMBERT
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
}`,$b=`#define LAMBERT
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
}`,Zb=`#define MATCAP
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
}`,Jb=`#define MATCAP
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
}`,eM=`#define NORMAL
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
}`,tM=`#define NORMAL
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
}`,nM=`#define PHONG
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
}`,iM=`#define PHONG
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
}`,sM=`#define STANDARD
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
}`,rM=`#define STANDARD
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
}`,oM=`#define TOON
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
}`,aM=`#define TOON
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
}`,lM=`uniform float size;
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
}`,cM=`uniform vec3 diffuse;
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
}`,uM=`#include <common>
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
}`,fM=`uniform vec3 color;
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
}`,hM=`uniform float rotation;
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
}`,dM=`uniform vec3 diffuse;
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
}`,et={alphahash_fragment:BA,alphahash_pars_fragment:UA,alphamap_fragment:OA,alphamap_pars_fragment:NA,alphatest_fragment:zA,alphatest_pars_fragment:kA,aomap_fragment:HA,aomap_pars_fragment:VA,batching_pars_vertex:GA,batching_vertex:WA,begin_vertex:XA,beginnormal_vertex:qA,bsdfs:YA,iridescence_fragment:QA,bumpmap_pars_fragment:KA,clipping_planes_fragment:jA,clipping_planes_pars_fragment:$A,clipping_planes_pars_vertex:ZA,clipping_planes_vertex:JA,color_fragment:ey,color_pars_fragment:ty,color_pars_vertex:ny,color_vertex:iy,common:sy,cube_uv_reflection_fragment:ry,defaultnormal_vertex:oy,displacementmap_pars_vertex:ay,displacementmap_vertex:ly,emissivemap_fragment:cy,emissivemap_pars_fragment:uy,colorspace_fragment:fy,colorspace_pars_fragment:hy,envmap_fragment:dy,envmap_common_pars_fragment:py,envmap_pars_fragment:my,envmap_pars_vertex:gy,envmap_physical_pars_fragment:Ey,envmap_vertex:xy,fog_vertex:_y,fog_pars_vertex:vy,fog_fragment:Sy,fog_pars_fragment:Ay,gradientmap_pars_fragment:yy,lightmap_pars_fragment:by,lights_lambert_fragment:My,lights_lambert_pars_fragment:Ty,lights_pars_begin:Cy,lights_toon_fragment:wy,lights_toon_pars_fragment:Ry,lights_phong_fragment:Iy,lights_phong_pars_fragment:Dy,lights_physical_fragment:Py,lights_physical_pars_fragment:Fy,lights_fragment_begin:Ly,lights_fragment_maps:By,lights_fragment_end:Uy,logdepthbuf_fragment:Oy,logdepthbuf_pars_fragment:Ny,logdepthbuf_pars_vertex:zy,logdepthbuf_vertex:ky,map_fragment:Hy,map_pars_fragment:Vy,map_particle_fragment:Gy,map_particle_pars_fragment:Wy,metalnessmap_fragment:Xy,metalnessmap_pars_fragment:qy,morphinstance_vertex:Yy,morphcolor_vertex:Qy,morphnormal_vertex:Ky,morphtarget_pars_vertex:jy,morphtarget_vertex:$y,normal_fragment_begin:Zy,normal_fragment_maps:Jy,normal_pars_fragment:eb,normal_pars_vertex:tb,normal_vertex:nb,normalmap_pars_fragment:ib,clearcoat_normal_fragment_begin:sb,clearcoat_normal_fragment_maps:rb,clearcoat_pars_fragment:ob,iridescence_pars_fragment:ab,opaque_fragment:lb,packing:cb,premultiplied_alpha_fragment:ub,project_vertex:fb,dithering_fragment:hb,dithering_pars_fragment:db,roughnessmap_fragment:pb,roughnessmap_pars_fragment:mb,shadowmap_pars_fragment:gb,shadowmap_pars_vertex:xb,shadowmap_vertex:_b,shadowmask_pars_fragment:vb,skinbase_vertex:Sb,skinning_pars_vertex:Ab,skinning_vertex:yb,skinnormal_vertex:bb,specularmap_fragment:Mb,specularmap_pars_fragment:Tb,tonemapping_fragment:Cb,tonemapping_pars_fragment:Eb,transmission_fragment:wb,transmission_pars_fragment:Rb,uv_pars_fragment:Ib,uv_pars_vertex:Db,uv_vertex:Pb,worldpos_vertex:Fb,background_vert:Lb,background_frag:Bb,backgroundCube_vert:Ub,backgroundCube_frag:Ob,cube_vert:Nb,cube_frag:zb,depth_vert:kb,depth_frag:Hb,distanceRGBA_vert:Vb,distanceRGBA_frag:Gb,equirect_vert:Wb,equirect_frag:Xb,linedashed_vert:qb,linedashed_frag:Yb,meshbasic_vert:Qb,meshbasic_frag:Kb,meshlambert_vert:jb,meshlambert_frag:$b,meshmatcap_vert:Zb,meshmatcap_frag:Jb,meshnormal_vert:eM,meshnormal_frag:tM,meshphong_vert:nM,meshphong_frag:iM,meshphysical_vert:sM,meshphysical_frag:rM,meshtoon_vert:oM,meshtoon_frag:aM,points_vert:lM,points_frag:cM,shadow_vert:uM,shadow_frag:fM,sprite_vert:hM,sprite_frag:dM},De={common:{diffuse:{value:new rt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Qe},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Qe}},envmap:{envMap:{value:null},envMapRotation:{value:new Qe},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Qe}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Qe}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Qe},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Qe},normalScale:{value:new Pe(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Qe},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Qe}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Qe}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Qe}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new rt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new rt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0},uvTransform:{value:new Qe}},sprite:{diffuse:{value:new rt(16777215)},opacity:{value:1},center:{value:new Pe(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Qe},alphaMap:{value:null},alphaMapTransform:{value:new Qe},alphaTest:{value:0}}},Bi={basic:{uniforms:vn([De.common,De.specularmap,De.envmap,De.aomap,De.lightmap,De.fog]),vertexShader:et.meshbasic_vert,fragmentShader:et.meshbasic_frag},lambert:{uniforms:vn([De.common,De.specularmap,De.envmap,De.aomap,De.lightmap,De.emissivemap,De.bumpmap,De.normalmap,De.displacementmap,De.fog,De.lights,{emissive:{value:new rt(0)}}]),vertexShader:et.meshlambert_vert,fragmentShader:et.meshlambert_frag},phong:{uniforms:vn([De.common,De.specularmap,De.envmap,De.aomap,De.lightmap,De.emissivemap,De.bumpmap,De.normalmap,De.displacementmap,De.fog,De.lights,{emissive:{value:new rt(0)},specular:{value:new rt(1118481)},shininess:{value:30}}]),vertexShader:et.meshphong_vert,fragmentShader:et.meshphong_frag},standard:{uniforms:vn([De.common,De.envmap,De.aomap,De.lightmap,De.emissivemap,De.bumpmap,De.normalmap,De.displacementmap,De.roughnessmap,De.metalnessmap,De.fog,De.lights,{emissive:{value:new rt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:et.meshphysical_vert,fragmentShader:et.meshphysical_frag},toon:{uniforms:vn([De.common,De.aomap,De.lightmap,De.emissivemap,De.bumpmap,De.normalmap,De.displacementmap,De.gradientmap,De.fog,De.lights,{emissive:{value:new rt(0)}}]),vertexShader:et.meshtoon_vert,fragmentShader:et.meshtoon_frag},matcap:{uniforms:vn([De.common,De.bumpmap,De.normalmap,De.displacementmap,De.fog,{matcap:{value:null}}]),vertexShader:et.meshmatcap_vert,fragmentShader:et.meshmatcap_frag},points:{uniforms:vn([De.points,De.fog]),vertexShader:et.points_vert,fragmentShader:et.points_frag},dashed:{uniforms:vn([De.common,De.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:et.linedashed_vert,fragmentShader:et.linedashed_frag},depth:{uniforms:vn([De.common,De.displacementmap]),vertexShader:et.depth_vert,fragmentShader:et.depth_frag},normal:{uniforms:vn([De.common,De.bumpmap,De.normalmap,De.displacementmap,{opacity:{value:1}}]),vertexShader:et.meshnormal_vert,fragmentShader:et.meshnormal_frag},sprite:{uniforms:vn([De.sprite,De.fog]),vertexShader:et.sprite_vert,fragmentShader:et.sprite_frag},background:{uniforms:{uvTransform:{value:new Qe},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:et.background_vert,fragmentShader:et.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Qe}},vertexShader:et.backgroundCube_vert,fragmentShader:et.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:et.cube_vert,fragmentShader:et.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:et.equirect_vert,fragmentShader:et.equirect_frag},distanceRGBA:{uniforms:vn([De.common,De.displacementmap,{referencePosition:{value:new F},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:et.distanceRGBA_vert,fragmentShader:et.distanceRGBA_frag},shadow:{uniforms:vn([De.lights,De.fog,{color:{value:new rt(0)},opacity:{value:1}}]),vertexShader:et.shadow_vert,fragmentShader:et.shadow_frag}};Bi.physical={uniforms:vn([Bi.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Qe},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Qe},clearcoatNormalScale:{value:new Pe(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Qe},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Qe},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Qe},sheen:{value:0},sheenColor:{value:new rt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Qe},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Qe},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Qe},transmissionSamplerSize:{value:new Pe},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Qe},attenuationDistance:{value:0},attenuationColor:{value:new rt(0)},specularColor:{value:new rt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Qe},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Qe},anisotropyVector:{value:new Pe},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Qe}}]),vertexShader:et.meshphysical_vert,fragmentShader:et.meshphysical_frag};const hl={r:0,b:0,g:0},or=new Ei,pM=new Ye;function mM(i,e,t,n,s,r,o){const a=new rt(0);let l=r===!0?0:1,c,u,f=null,h=0,d=null;function x(S){let A=S.isScene===!0?S.background:null;return A&&A.isTexture&&(A=(S.backgroundBlurriness>0?t:e).get(A)),A}function p(S){let A=!1;const y=x(S);y===null?m(a,l):y&&y.isColor&&(m(y,1),A=!0);const b=i.xr.getEnvironmentBlendMode();b==="additive"?n.buffers.color.setClear(0,0,0,1,o):b==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,o),(i.autoClear||A)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function g(S,A){const y=x(A);y&&(y.isCubeTexture||y.mapping===hc)?(u===void 0&&(u=new Yt(new Fo(1,1,1),new Cn({name:"BackgroundCubeMaterial",uniforms:Mo(Bi.backgroundCube.uniforms),vertexShader:Bi.backgroundCube.vertexShader,fragmentShader:Bi.backgroundCube.fragmentShader,side:Bn,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),u.geometry.deleteAttribute("normal"),u.geometry.deleteAttribute("uv"),u.onBeforeRender=function(b,v,E){this.matrixWorld.copyPosition(E.matrixWorld)},Object.defineProperty(u.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),s.update(u)),or.copy(A.backgroundRotation),or.x*=-1,or.y*=-1,or.z*=-1,y.isCubeTexture&&y.isRenderTargetTexture===!1&&(or.y*=-1,or.z*=-1),u.material.uniforms.envMap.value=y,u.material.uniforms.flipEnvMap.value=y.isCubeTexture&&y.isRenderTargetTexture===!1?-1:1,u.material.uniforms.backgroundBlurriness.value=A.backgroundBlurriness,u.material.uniforms.backgroundIntensity.value=A.backgroundIntensity,u.material.uniforms.backgroundRotation.value.setFromMatrix4(pM.makeRotationFromEuler(or)),u.material.toneMapped=lt.getTransfer(y.colorSpace)!==mt,(f!==y||h!==y.version||d!==i.toneMapping)&&(u.material.needsUpdate=!0,f=y,h=y.version,d=i.toneMapping),u.layers.enableAll(),S.unshift(u,u.geometry,u.material,0,0,null)):y&&y.isTexture&&(c===void 0&&(c=new Yt(new To(2,2),new Cn({name:"BackgroundMaterial",uniforms:Mo(Bi.background.uniforms),vertexShader:Bi.background.vertexShader,fragmentShader:Bi.background.fragmentShader,side:Xi,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),s.update(c)),c.material.uniforms.t2D.value=y,c.material.uniforms.backgroundIntensity.value=A.backgroundIntensity,c.material.toneMapped=lt.getTransfer(y.colorSpace)!==mt,y.matrixAutoUpdate===!0&&y.updateMatrix(),c.material.uniforms.uvTransform.value.copy(y.matrix),(f!==y||h!==y.version||d!==i.toneMapping)&&(c.material.needsUpdate=!0,f=y,h=y.version,d=i.toneMapping),c.layers.enableAll(),S.unshift(c,c.geometry,c.material,0,0,null))}function m(S,A){S.getRGB(hl,ag(i)),n.buffers.color.setClear(hl.r,hl.g,hl.b,A,o)}function _(){u!==void 0&&(u.geometry.dispose(),u.material.dispose(),u=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return a},setClearColor:function(S,A=1){a.set(S),l=A,m(a,l)},getClearAlpha:function(){return l},setClearAlpha:function(S){l=S,m(a,l)},render:p,addToRenderList:g,dispose:_}}function gM(i,e){const t=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},s=h(null);let r=s,o=!1;function a(T,I,P,B,N){let G=!1;const V=f(B,P,I);r!==V&&(r=V,c(r.object)),G=d(T,B,P,N),G&&x(T,B,P,N),N!==null&&e.update(N,i.ELEMENT_ARRAY_BUFFER),(G||o)&&(o=!1,A(T,I,P,B),N!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,e.get(N).buffer))}function l(){return i.createVertexArray()}function c(T){return i.bindVertexArray(T)}function u(T){return i.deleteVertexArray(T)}function f(T,I,P){const B=P.wireframe===!0;let N=n[T.id];N===void 0&&(N={},n[T.id]=N);let G=N[I.id];G===void 0&&(G={},N[I.id]=G);let V=G[B];return V===void 0&&(V=h(l()),G[B]=V),V}function h(T){const I=[],P=[],B=[];for(let N=0;N<t;N++)I[N]=0,P[N]=0,B[N]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:I,enabledAttributes:P,attributeDivisors:B,object:T,attributes:{},index:null}}function d(T,I,P,B){const N=r.attributes,G=I.attributes;let V=0;const q=P.getAttributes();for(const X in q)if(q[X].location>=0){const ce=N[X];let be=G[X];if(be===void 0&&(X==="instanceMatrix"&&T.instanceMatrix&&(be=T.instanceMatrix),X==="instanceColor"&&T.instanceColor&&(be=T.instanceColor)),ce===void 0||ce.attribute!==be||be&&ce.data!==be.data)return!0;V++}return r.attributesNum!==V||r.index!==B}function x(T,I,P,B){const N={},G=I.attributes;let V=0;const q=P.getAttributes();for(const X in q)if(q[X].location>=0){let ce=G[X];ce===void 0&&(X==="instanceMatrix"&&T.instanceMatrix&&(ce=T.instanceMatrix),X==="instanceColor"&&T.instanceColor&&(ce=T.instanceColor));const be={};be.attribute=ce,ce&&ce.data&&(be.data=ce.data),N[X]=be,V++}r.attributes=N,r.attributesNum=V,r.index=B}function p(){const T=r.newAttributes;for(let I=0,P=T.length;I<P;I++)T[I]=0}function g(T){m(T,0)}function m(T,I){const P=r.newAttributes,B=r.enabledAttributes,N=r.attributeDivisors;P[T]=1,B[T]===0&&(i.enableVertexAttribArray(T),B[T]=1),N[T]!==I&&(i.vertexAttribDivisor(T,I),N[T]=I)}function _(){const T=r.newAttributes,I=r.enabledAttributes;for(let P=0,B=I.length;P<B;P++)I[P]!==T[P]&&(i.disableVertexAttribArray(P),I[P]=0)}function S(T,I,P,B,N,G,V){V===!0?i.vertexAttribIPointer(T,I,P,N,G):i.vertexAttribPointer(T,I,P,B,N,G)}function A(T,I,P,B){p();const N=B.attributes,G=P.getAttributes(),V=I.defaultAttributeValues;for(const q in G){const X=G[q];if(X.location>=0){let ee=N[q];if(ee===void 0&&(q==="instanceMatrix"&&T.instanceMatrix&&(ee=T.instanceMatrix),q==="instanceColor"&&T.instanceColor&&(ee=T.instanceColor)),ee!==void 0){const ce=ee.normalized,be=ee.itemSize,Re=e.get(ee);if(Re===void 0)continue;const Fe=Re.buffer,Oe=Re.type,Ne=Re.bytesPerElement,J=Oe===i.INT||Oe===i.UNSIGNED_INT||ee.gpuType===xh;if(ee.isInterleavedBufferAttribute){const ne=ee.data,xe=ne.stride,Be=ee.offset;if(ne.isInstancedInterleavedBuffer){for(let Te=0;Te<X.locationSize;Te++)m(X.location+Te,ne.meshPerAttribute);T.isInstancedMesh!==!0&&B._maxInstanceCount===void 0&&(B._maxInstanceCount=ne.meshPerAttribute*ne.count)}else for(let Te=0;Te<X.locationSize;Te++)g(X.location+Te);i.bindBuffer(i.ARRAY_BUFFER,Fe);for(let Te=0;Te<X.locationSize;Te++)S(X.location+Te,be/X.locationSize,Oe,ce,xe*Ne,(Be+be/X.locationSize*Te)*Ne,J)}else{if(ee.isInstancedBufferAttribute){for(let ne=0;ne<X.locationSize;ne++)m(X.location+ne,ee.meshPerAttribute);T.isInstancedMesh!==!0&&B._maxInstanceCount===void 0&&(B._maxInstanceCount=ee.meshPerAttribute*ee.count)}else for(let ne=0;ne<X.locationSize;ne++)g(X.location+ne);i.bindBuffer(i.ARRAY_BUFFER,Fe);for(let ne=0;ne<X.locationSize;ne++)S(X.location+ne,be/X.locationSize,Oe,ce,be*Ne,be/X.locationSize*ne*Ne,J)}}else if(V!==void 0){const ce=V[q];if(ce!==void 0)switch(ce.length){case 2:i.vertexAttrib2fv(X.location,ce);break;case 3:i.vertexAttrib3fv(X.location,ce);break;case 4:i.vertexAttrib4fv(X.location,ce);break;default:i.vertexAttrib1fv(X.location,ce)}}}}_()}function y(){E();for(const T in n){const I=n[T];for(const P in I){const B=I[P];for(const N in B)u(B[N].object),delete B[N];delete I[P]}delete n[T]}}function b(T){if(n[T.id]===void 0)return;const I=n[T.id];for(const P in I){const B=I[P];for(const N in B)u(B[N].object),delete B[N];delete I[P]}delete n[T.id]}function v(T){for(const I in n){const P=n[I];if(P[T.id]===void 0)continue;const B=P[T.id];for(const N in B)u(B[N].object),delete B[N];delete P[T.id]}}function E(){M(),o=!0,r!==s&&(r=s,c(r.object))}function M(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:a,reset:E,resetDefaultState:M,dispose:y,releaseStatesOfGeometry:b,releaseStatesOfProgram:v,initAttributes:p,enableAttribute:g,disableUnusedAttributes:_}}function xM(i,e,t){let n;function s(c){n=c}function r(c,u){i.drawArrays(n,c,u),t.update(u,n,1)}function o(c,u,f){f!==0&&(i.drawArraysInstanced(n,c,u,f),t.update(u,n,f))}function a(c,u,f){if(f===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,c,0,u,0,f);let d=0;for(let x=0;x<f;x++)d+=u[x];t.update(d,n,1)}function l(c,u,f,h){if(f===0)return;const d=e.get("WEBGL_multi_draw");if(d===null)for(let x=0;x<c.length;x++)o(c[x],u[x],h[x]);else{d.multiDrawArraysInstancedWEBGL(n,c,0,u,0,h,0,f);let x=0;for(let p=0;p<f;p++)x+=u[p]*h[p];t.update(x,n,1)}}this.setMode=s,this.render=r,this.renderInstances=o,this.renderMultiDraw=a,this.renderMultiDrawInstances=l}function _M(i,e,t,n){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){const v=e.get("EXT_texture_filter_anisotropic");s=i.getParameter(v.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function o(v){return!(v!==Mn&&n.convert(v)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function a(v){const E=v===Tr&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(v!==qi&&n.convert(v)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE)&&v!==Mi&&!E)}function l(v){if(v==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";v="mediump"}return v==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let c=t.precision!==void 0?t.precision:"highp";const u=l(c);u!==c&&(Ze("WebGLRenderer:",c,"not supported, using",u,"instead."),c=u);const f=t.logarithmicDepthBuffer===!0,h=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),d=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),x=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),p=i.getParameter(i.MAX_TEXTURE_SIZE),g=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),m=i.getParameter(i.MAX_VERTEX_ATTRIBS),_=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),S=i.getParameter(i.MAX_VARYING_VECTORS),A=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),y=x>0,b=i.getParameter(i.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:l,textureFormatReadable:o,textureTypeReadable:a,precision:c,logarithmicDepthBuffer:f,reversedDepthBuffer:h,maxTextures:d,maxVertexTextures:x,maxTextureSize:p,maxCubemapSize:g,maxAttributes:m,maxVertexUniforms:_,maxVaryings:S,maxFragmentUniforms:A,vertexTextures:y,maxSamples:b}}function vM(i){const e=this;let t=null,n=0,s=!1,r=!1;const o=new as,a=new Qe,l={value:null,needsUpdate:!1};this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(f,h){const d=f.length!==0||h||n!==0||s;return s=h,n=f.length,d},this.beginShadows=function(){r=!0,u(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(f,h){t=u(f,h,0)},this.setState=function(f,h,d){const x=f.clippingPlanes,p=f.clipIntersection,g=f.clipShadows,m=i.get(f);if(!s||x===null||x.length===0||r&&!g)r?u(null):c();else{const _=r?0:n,S=_*4;let A=m.clippingState||null;l.value=A,A=u(x,h,S,d);for(let y=0;y!==S;++y)A[y]=t[y];m.clippingState=A,this.numIntersection=p?this.numPlanes:0,this.numPlanes+=_}};function c(){l.value!==t&&(l.value=t,l.needsUpdate=n>0),e.numPlanes=n,e.numIntersection=0}function u(f,h,d,x){const p=f!==null?f.length:0;let g=null;if(p!==0){if(g=l.value,x!==!0||g===null){const m=d+p*4,_=h.matrixWorldInverse;a.getNormalMatrix(_),(g===null||g.length<m)&&(g=new Float32Array(m));for(let S=0,A=d;S!==p;++S,A+=4)o.copy(f[S]).applyMatrix4(_,a),o.normal.toArray(g,A),g[A+3]=o.constant}l.value=g,l.needsUpdate=!0}return e.numPlanes=p,e.numIntersection=0,g}}function SM(i){let e=new WeakMap;function t(o,a){return a===Zu?o.mapping=So:a===Ju&&(o.mapping=Ao),o}function n(o){if(o&&o.isTexture){const a=o.mapping;if(a===Zu||a===Ju)if(e.has(o)){const l=e.get(o).texture;return t(l,o.mapping)}else{const l=o.image;if(l&&l.height>0){const c=new vA(l.height);return c.fromEquirectangularTexture(i,o),e.set(o,c),o.addEventListener("dispose",s),t(c.texture,o.mapping)}else return null}}return o}function s(o){const a=o.target;a.removeEventListener("dispose",s);const l=e.get(a);l!==void 0&&(e.delete(a),l.dispose())}function r(){e=new WeakMap}return{get:n,dispose:r}}const Ps=4,gp=[.125,.215,.35,.446,.526,.582],pr=20,AM=256,Wo=new Ch,xp=new rt;let cu=null,uu=0,fu=0,hu=!1;const yM=new F;class _p{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,n=.1,s=100,r={}){const{size:o=256,position:a=yM}=r;cu=this._renderer.getRenderTarget(),uu=this._renderer.getActiveCubeFace(),fu=this._renderer.getActiveMipmapLevel(),hu=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(o);const l=this._allocateTargets();return l.depthBuffer=!0,this._sceneToCubeUV(e,n,s,l,a),t>0&&this._blur(l,0,0,t),this._applyPMREM(l),this._cleanup(l),l}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Ap(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=Sp(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(cu,uu,fu),this._renderer.xr.enabled=hu,e.scissorTest=!1,Xr(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===So||e.mapping===Ao?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),cu=this._renderer.getRenderTarget(),uu=this._renderer.getActiveCubeFace(),fu=this._renderer.getActiveMipmapLevel(),hu=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const n=t||this._allocateTargets();return this._textureToCubeUV(e,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,n={magFilter:di,minFilter:di,generateMipmaps:!1,type:Tr,format:Mn,colorSpace:bo,depthBuffer:!1},s=vp(e,t,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=vp(e,t,n);const{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=bM(r)),this._blurMaterial=TM(r,e,t),this._ggxMaterial=MM(r,e,t)}return s}_compileMaterial(e){const t=new Yt(new En,e);this._renderer.compile(t,Wo)}_sceneToCubeUV(e,t,n,s,r){const l=new ui(90,1,t,n),c=[1,-1,1,1,1,1],u=[1,1,1,-1,-1,-1],f=this._renderer,h=f.autoClear,d=f.toneMapping;f.getClearColor(xp),f.toneMapping=zs,f.autoClear=!1,f.state.buffers.depth.getReversed()&&(f.setRenderTarget(s),f.clearDepth(),f.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new Yt(new Fo,new Mr({name:"PMREM.Background",side:Bn,depthWrite:!1,depthTest:!1})));const p=this._backgroundBox,g=p.material;let m=!1;const _=e.background;_?_.isColor&&(g.color.copy(_),e.background=null,m=!0):(g.color.copy(xp),m=!0);for(let S=0;S<6;S++){const A=S%3;A===0?(l.up.set(0,c[S],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x+u[S],r.y,r.z)):A===1?(l.up.set(0,0,c[S]),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y+u[S],r.z)):(l.up.set(0,c[S],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y,r.z+u[S]));const y=this._cubeSize;Xr(s,A*y,S>2?y:0,y,y),f.setRenderTarget(s),m&&f.render(p,l),f.render(e,l)}f.toneMapping=d,f.autoClear=h,e.background=_}_textureToCubeUV(e,t){const n=this._renderer,s=e.mapping===So||e.mapping===Ao;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=Ap()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=Sp());const r=s?this._cubemapMaterial:this._equirectMaterial,o=this._lodMeshes[0];o.material=r;const a=r.uniforms;a.envMap.value=e;const l=this._cubeSize;Xr(t,0,0,3*l,2*l),n.setRenderTarget(t),n.render(o,Wo)}_applyPMREM(e){const t=this._renderer,n=t.autoClear;t.autoClear=!1;const s=this._lodMeshes.length;for(let r=1;r<s;r++)this._applyGGXFilter(e,r-1,r);t.autoClear=n}_applyGGXFilter(e,t,n){const s=this._renderer,r=this._pingPongRenderTarget,o=this._ggxMaterial,a=this._lodMeshes[n];a.material=o;const l=o.uniforms,c=n/(this._lodMeshes.length-1),u=t/(this._lodMeshes.length-1),f=Math.sqrt(c*c-u*u),h=.05+c*.95,d=f*h,{_lodMax:x}=this,p=this._sizeLods[n],g=3*p*(n>x-Ps?n-x+Ps:0),m=4*(this._cubeSize-p);l.envMap.value=e.texture,l.roughness.value=d,l.mipInt.value=x-t,Xr(r,g,m,3*p,2*p),s.setRenderTarget(r),s.render(a,Wo),l.envMap.value=r.texture,l.roughness.value=0,l.mipInt.value=x-n,Xr(e,g,m,3*p,2*p),s.setRenderTarget(e),s.render(a,Wo)}_blur(e,t,n,s,r){const o=this._pingPongRenderTarget;this._halfBlur(e,o,t,n,s,"latitudinal",r),this._halfBlur(o,e,n,n,s,"longitudinal",r)}_halfBlur(e,t,n,s,r,o,a){const l=this._renderer,c=this._blurMaterial;o!=="latitudinal"&&o!=="longitudinal"&&Wt("blur direction must be either latitudinal or longitudinal!");const u=3,f=this._lodMeshes[s];f.material=c;const h=c.uniforms,d=this._sizeLods[n]-1,x=isFinite(r)?Math.PI/(2*d):2*Math.PI/(2*pr-1),p=r/x,g=isFinite(r)?1+Math.floor(u*p):pr;g>pr&&Ze(`sigmaRadians, ${r}, is too large and will clip, as it requested ${g} samples when the maximum is set to ${pr}`);const m=[];let _=0;for(let v=0;v<pr;++v){const E=v/p,M=Math.exp(-E*E/2);m.push(M),v===0?_+=M:v<g&&(_+=2*M)}for(let v=0;v<m.length;v++)m[v]=m[v]/_;h.envMap.value=e.texture,h.samples.value=g,h.weights.value=m,h.latitudinal.value=o==="latitudinal",a&&(h.poleAxis.value=a);const{_lodMax:S}=this;h.dTheta.value=x,h.mipInt.value=S-n;const A=this._sizeLods[s],y=3*A*(s>S-Ps?s-S+Ps:0),b=4*(this._cubeSize-A);Xr(t,y,b,3*A,2*A),l.setRenderTarget(t),l.render(f,Wo)}}function bM(i){const e=[],t=[],n=[];let s=i;const r=i-Ps+1+gp.length;for(let o=0;o<r;o++){const a=Math.pow(2,s);e.push(a);let l=1/a;o>i-Ps?l=gp[o-i+Ps-1]:o===0&&(l=0),t.push(l);const c=1/(a-2),u=-c,f=1+c,h=[u,u,f,u,f,f,u,u,f,f,u,f],d=6,x=6,p=3,g=2,m=1,_=new Float32Array(p*x*d),S=new Float32Array(g*x*d),A=new Float32Array(m*x*d);for(let b=0;b<d;b++){const v=b%3*2/3-1,E=b>2?0:-1,M=[v,E,0,v+2/3,E,0,v+2/3,E+1,0,v,E,0,v+2/3,E+1,0,v,E+1,0];_.set(M,p*x*b),S.set(h,g*x*b);const T=[b,b,b,b,b,b];A.set(T,m*x*b)}const y=new En;y.setAttribute("position",new _i(_,p)),y.setAttribute("uv",new _i(S,g)),y.setAttribute("faceIndex",new _i(A,m)),n.push(new Yt(y,null)),s>Ps&&s--}return{lodMeshes:n,sizeLods:e,sigmas:t}}function vp(i,e,t){const n=new Ws(i,e,t);return n.texture.mapping=hc,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function Xr(i,e,t,n,s){i.viewport.set(e,t,n,s),i.scissor.set(e,t,n,s)}function MM(i,e,t){return new Cn({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:AM,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:gc(),fragmentShader:`

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
		`,blending:ps,depthTest:!1,depthWrite:!1})}function TM(i,e,t){const n=new Float32Array(pr),s=new F(0,1,0);return new Cn({name:"SphericalGaussianBlur",defines:{n:pr,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:gc(),fragmentShader:`

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
		`,blending:ps,depthTest:!1,depthWrite:!1})}function Sp(){return new Cn({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:gc(),fragmentShader:`

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
		`,blending:ps,depthTest:!1,depthWrite:!1})}function Ap(){return new Cn({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:gc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:ps,depthTest:!1,depthWrite:!1})}function gc(){return`

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
	`}function CM(i){let e=new WeakMap,t=null;function n(a){if(a&&a.isTexture){const l=a.mapping,c=l===Zu||l===Ju,u=l===So||l===Ao;if(c||u){let f=e.get(a);const h=f!==void 0?f.texture.pmremVersion:0;if(a.isRenderTargetTexture&&a.pmremVersion!==h)return t===null&&(t=new _p(i)),f=c?t.fromEquirectangular(a,f):t.fromCubemap(a,f),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),f.texture;if(f!==void 0)return f.texture;{const d=a.image;return c&&d&&d.height>0||u&&d&&s(d)?(t===null&&(t=new _p(i)),f=c?t.fromEquirectangular(a):t.fromCubemap(a),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),a.addEventListener("dispose",r),f.texture):null}}}return a}function s(a){let l=0;const c=6;for(let u=0;u<c;u++)a[u]!==void 0&&l++;return l===c}function r(a){const l=a.target;l.removeEventListener("dispose",r);const c=e.get(l);c!==void 0&&(e.delete(l),c.dispose())}function o(){e=new WeakMap,t!==null&&(t.dispose(),t=null)}return{get:n,dispose:o}}function EM(i){const e={};function t(n){if(e[n]!==void 0)return e[n];const s=i.getExtension(n);return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(n){const s=t(n);return s===null&&ya("WebGLRenderer: "+n+" extension not supported."),s}}}function wM(i,e,t,n){const s={},r=new WeakMap;function o(f){const h=f.target;h.index!==null&&e.remove(h.index);for(const x in h.attributes)e.remove(h.attributes[x]);h.removeEventListener("dispose",o),delete s[h.id];const d=r.get(h);d&&(e.remove(d),r.delete(h)),n.releaseStatesOfGeometry(h),h.isInstancedBufferGeometry===!0&&delete h._maxInstanceCount,t.memory.geometries--}function a(f,h){return s[h.id]===!0||(h.addEventListener("dispose",o),s[h.id]=!0,t.memory.geometries++),h}function l(f){const h=f.attributes;for(const d in h)e.update(h[d],i.ARRAY_BUFFER)}function c(f){const h=[],d=f.index,x=f.attributes.position;let p=0;if(d!==null){const _=d.array;p=d.version;for(let S=0,A=_.length;S<A;S+=3){const y=_[S+0],b=_[S+1],v=_[S+2];h.push(y,b,b,v,v,y)}}else if(x!==void 0){const _=x.array;p=x.version;for(let S=0,A=_.length/3-1;S<A;S+=3){const y=S+0,b=S+1,v=S+2;h.push(y,b,b,v,v,y)}}else return;const g=new(tg(h)?og:rg)(h,1);g.version=p;const m=r.get(f);m&&e.remove(m),r.set(f,g)}function u(f){const h=r.get(f);if(h){const d=f.index;d!==null&&h.version<d.version&&c(f)}else c(f);return r.get(f)}return{get:a,update:l,getWireframeAttribute:u}}function RM(i,e,t){let n;function s(h){n=h}let r,o;function a(h){r=h.type,o=h.bytesPerElement}function l(h,d){i.drawElements(n,d,r,h*o),t.update(d,n,1)}function c(h,d,x){x!==0&&(i.drawElementsInstanced(n,d,r,h*o,x),t.update(d,n,x))}function u(h,d,x){if(x===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,d,0,r,h,0,x);let g=0;for(let m=0;m<x;m++)g+=d[m];t.update(g,n,1)}function f(h,d,x,p){if(x===0)return;const g=e.get("WEBGL_multi_draw");if(g===null)for(let m=0;m<h.length;m++)c(h[m]/o,d[m],p[m]);else{g.multiDrawElementsInstancedWEBGL(n,d,0,r,h,0,p,0,x);let m=0;for(let _=0;_<x;_++)m+=d[_]*p[_];t.update(m,n,1)}}this.setMode=s,this.setIndex=a,this.render=l,this.renderInstances=c,this.renderMultiDraw=u,this.renderMultiDrawInstances=f}function IM(i){const e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function n(r,o,a){switch(t.calls++,o){case i.TRIANGLES:t.triangles+=a*(r/3);break;case i.LINES:t.lines+=a*(r/2);break;case i.LINE_STRIP:t.lines+=a*(r-1);break;case i.LINE_LOOP:t.lines+=a*r;break;case i.POINTS:t.points+=a*r;break;default:Wt("WebGLInfo: Unknown draw mode:",o);break}}function s(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:s,update:n}}function DM(i,e,t){const n=new WeakMap,s=new Dt;function r(o,a,l){const c=o.morphTargetInfluences,u=a.morphAttributes.position||a.morphAttributes.normal||a.morphAttributes.color,f=u!==void 0?u.length:0;let h=n.get(a);if(h===void 0||h.count!==f){let T=function(){E.dispose(),n.delete(a),a.removeEventListener("dispose",T)};var d=T;h!==void 0&&h.texture.dispose();const x=a.morphAttributes.position!==void 0,p=a.morphAttributes.normal!==void 0,g=a.morphAttributes.color!==void 0,m=a.morphAttributes.position||[],_=a.morphAttributes.normal||[],S=a.morphAttributes.color||[];let A=0;x===!0&&(A=1),p===!0&&(A=2),g===!0&&(A=3);let y=a.attributes.position.count*A,b=1;y>e.maxTextureSize&&(b=Math.ceil(y/e.maxTextureSize),y=e.maxTextureSize);const v=new Float32Array(y*b*4*f),E=new ng(v,y,b,f);E.type=Mi,E.needsUpdate=!0;const M=A*4;for(let I=0;I<f;I++){const P=m[I],B=_[I],N=S[I],G=y*b*4*I;for(let V=0;V<P.count;V++){const q=V*M;x===!0&&(s.fromBufferAttribute(P,V),v[G+q+0]=s.x,v[G+q+1]=s.y,v[G+q+2]=s.z,v[G+q+3]=0),p===!0&&(s.fromBufferAttribute(B,V),v[G+q+4]=s.x,v[G+q+5]=s.y,v[G+q+6]=s.z,v[G+q+7]=0),g===!0&&(s.fromBufferAttribute(N,V),v[G+q+8]=s.x,v[G+q+9]=s.y,v[G+q+10]=s.z,v[G+q+11]=N.itemSize===4?s.w:1)}}h={count:f,texture:E,size:new Pe(y,b)},n.set(a,h),a.addEventListener("dispose",T)}if(o.isInstancedMesh===!0&&o.morphTexture!==null)l.getUniforms().setValue(i,"morphTexture",o.morphTexture,t);else{let x=0;for(let g=0;g<c.length;g++)x+=c[g];const p=a.morphTargetsRelative?1:1-x;l.getUniforms().setValue(i,"morphTargetBaseInfluence",p),l.getUniforms().setValue(i,"morphTargetInfluences",c)}l.getUniforms().setValue(i,"morphTargetsTexture",h.texture,t),l.getUniforms().setValue(i,"morphTargetsTextureSize",h.size)}return{update:r}}function PM(i,e,t,n){let s=new WeakMap;function r(l){const c=n.render.frame,u=l.geometry,f=e.get(l,u);if(s.get(f)!==c&&(e.update(f),s.set(f,c)),l.isInstancedMesh&&(l.hasEventListener("dispose",a)===!1&&l.addEventListener("dispose",a),s.get(l)!==c&&(t.update(l.instanceMatrix,i.ARRAY_BUFFER),l.instanceColor!==null&&t.update(l.instanceColor,i.ARRAY_BUFFER),s.set(l,c))),l.isSkinnedMesh){const h=l.skeleton;s.get(h)!==c&&(h.update(),s.set(h,c))}return f}function o(){s=new WeakMap}function a(l){const c=l.target;c.removeEventListener("dispose",a),t.remove(c.instanceMatrix),c.instanceColor!==null&&t.remove(c.instanceColor)}return{update:r,dispose:o}}const dg=new Tn,yp=new Mh(1,1),pg=new ng,mg=new JS,gg=new cg,bp=[],Mp=[],Tp=new Float32Array(16),Cp=new Float32Array(9),Ep=new Float32Array(4);function Lo(i,e,t){const n=i[0];if(n<=0||n>0)return i;const s=e*t;let r=bp[s];if(r===void 0&&(r=new Float32Array(s),bp[s]=r),e!==0){n.toArray(r,0);for(let o=1,a=0;o!==e;++o)a+=t,i[o].toArray(r,a)}return r}function Jt(i,e){if(i.length!==e.length)return!1;for(let t=0,n=i.length;t<n;t++)if(i[t]!==e[t])return!1;return!0}function en(i,e){for(let t=0,n=e.length;t<n;t++)i[t]=e[t]}function xc(i,e){let t=Mp[e];t===void 0&&(t=new Int32Array(e),Mp[e]=t);for(let n=0;n!==e;++n)t[n]=i.allocateTextureUnit();return t}function FM(i,e){const t=this.cache;t[0]!==e&&(i.uniform1f(this.addr,e),t[0]=e)}function LM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Jt(t,e))return;i.uniform2fv(this.addr,e),en(t,e)}}function BM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(i.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(Jt(t,e))return;i.uniform3fv(this.addr,e),en(t,e)}}function UM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Jt(t,e))return;i.uniform4fv(this.addr,e),en(t,e)}}function OM(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Jt(t,e))return;i.uniformMatrix2fv(this.addr,!1,e),en(t,e)}else{if(Jt(t,n))return;Ep.set(n),i.uniformMatrix2fv(this.addr,!1,Ep),en(t,n)}}function NM(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Jt(t,e))return;i.uniformMatrix3fv(this.addr,!1,e),en(t,e)}else{if(Jt(t,n))return;Cp.set(n),i.uniformMatrix3fv(this.addr,!1,Cp),en(t,n)}}function zM(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Jt(t,e))return;i.uniformMatrix4fv(this.addr,!1,e),en(t,e)}else{if(Jt(t,n))return;Tp.set(n),i.uniformMatrix4fv(this.addr,!1,Tp),en(t,n)}}function kM(i,e){const t=this.cache;t[0]!==e&&(i.uniform1i(this.addr,e),t[0]=e)}function HM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Jt(t,e))return;i.uniform2iv(this.addr,e),en(t,e)}}function VM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Jt(t,e))return;i.uniform3iv(this.addr,e),en(t,e)}}function GM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Jt(t,e))return;i.uniform4iv(this.addr,e),en(t,e)}}function WM(i,e){const t=this.cache;t[0]!==e&&(i.uniform1ui(this.addr,e),t[0]=e)}function XM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Jt(t,e))return;i.uniform2uiv(this.addr,e),en(t,e)}}function qM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Jt(t,e))return;i.uniform3uiv(this.addr,e),en(t,e)}}function YM(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Jt(t,e))return;i.uniform4uiv(this.addr,e),en(t,e)}}function QM(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s);let r;this.type===i.SAMPLER_2D_SHADOW?(yp.compareFunction=eg,r=yp):r=dg,t.setTexture2D(e||r,s)}function KM(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture3D(e||mg,s)}function jM(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTextureCube(e||gg,s)}function $M(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture2DArray(e||pg,s)}function ZM(i){switch(i){case 5126:return FM;case 35664:return LM;case 35665:return BM;case 35666:return UM;case 35674:return OM;case 35675:return NM;case 35676:return zM;case 5124:case 35670:return kM;case 35667:case 35671:return HM;case 35668:case 35672:return VM;case 35669:case 35673:return GM;case 5125:return WM;case 36294:return XM;case 36295:return qM;case 36296:return YM;case 35678:case 36198:case 36298:case 36306:case 35682:return QM;case 35679:case 36299:case 36307:return KM;case 35680:case 36300:case 36308:case 36293:return jM;case 36289:case 36303:case 36311:case 36292:return $M}}function JM(i,e){i.uniform1fv(this.addr,e)}function eT(i,e){const t=Lo(e,this.size,2);i.uniform2fv(this.addr,t)}function tT(i,e){const t=Lo(e,this.size,3);i.uniform3fv(this.addr,t)}function nT(i,e){const t=Lo(e,this.size,4);i.uniform4fv(this.addr,t)}function iT(i,e){const t=Lo(e,this.size,4);i.uniformMatrix2fv(this.addr,!1,t)}function sT(i,e){const t=Lo(e,this.size,9);i.uniformMatrix3fv(this.addr,!1,t)}function rT(i,e){const t=Lo(e,this.size,16);i.uniformMatrix4fv(this.addr,!1,t)}function oT(i,e){i.uniform1iv(this.addr,e)}function aT(i,e){i.uniform2iv(this.addr,e)}function lT(i,e){i.uniform3iv(this.addr,e)}function cT(i,e){i.uniform4iv(this.addr,e)}function uT(i,e){i.uniform1uiv(this.addr,e)}function fT(i,e){i.uniform2uiv(this.addr,e)}function hT(i,e){i.uniform3uiv(this.addr,e)}function dT(i,e){i.uniform4uiv(this.addr,e)}function pT(i,e,t){const n=this.cache,s=e.length,r=xc(t,s);Jt(n,r)||(i.uniform1iv(this.addr,r),en(n,r));for(let o=0;o!==s;++o)t.setTexture2D(e[o]||dg,r[o])}function mT(i,e,t){const n=this.cache,s=e.length,r=xc(t,s);Jt(n,r)||(i.uniform1iv(this.addr,r),en(n,r));for(let o=0;o!==s;++o)t.setTexture3D(e[o]||mg,r[o])}function gT(i,e,t){const n=this.cache,s=e.length,r=xc(t,s);Jt(n,r)||(i.uniform1iv(this.addr,r),en(n,r));for(let o=0;o!==s;++o)t.setTextureCube(e[o]||gg,r[o])}function xT(i,e,t){const n=this.cache,s=e.length,r=xc(t,s);Jt(n,r)||(i.uniform1iv(this.addr,r),en(n,r));for(let o=0;o!==s;++o)t.setTexture2DArray(e[o]||pg,r[o])}function _T(i){switch(i){case 5126:return JM;case 35664:return eT;case 35665:return tT;case 35666:return nT;case 35674:return iT;case 35675:return sT;case 35676:return rT;case 5124:case 35670:return oT;case 35667:case 35671:return aT;case 35668:case 35672:return lT;case 35669:case 35673:return cT;case 5125:return uT;case 36294:return fT;case 36295:return hT;case 36296:return dT;case 35678:case 36198:case 36298:case 36306:case 35682:return pT;case 35679:case 36299:case 36307:return mT;case 35680:case 36300:case 36308:case 36293:return gT;case 36289:case 36303:case 36311:case 36292:return xT}}class vT{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.setValue=ZM(t.type)}}class ST{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=_T(t.type)}}class AT{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,n){const s=this.seq;for(let r=0,o=s.length;r!==o;++r){const a=s[r];a.setValue(e,t[a.id],n)}}}const du=/(\w+)(\])?(\[|\.)?/g;function wp(i,e){i.seq.push(e),i.map[e.id]=e}function yT(i,e,t){const n=i.name,s=n.length;for(du.lastIndex=0;;){const r=du.exec(n),o=du.lastIndex;let a=r[1];const l=r[2]==="]",c=r[3];if(l&&(a=a|0),c===void 0||c==="["&&o+2===s){wp(t,c===void 0?new vT(a,i,e):new ST(a,i,e));break}else{let f=t.map[a];f===void 0&&(f=new AT(a),wp(t,f)),t=f}}}class Pl{constructor(e,t){this.seq=[],this.map={};const n=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let s=0;s<n;++s){const r=e.getActiveUniform(t,s),o=e.getUniformLocation(t,r.name);yT(r,o,this)}}setValue(e,t,n,s){const r=this.map[t];r!==void 0&&r.setValue(e,n,s)}setOptional(e,t,n){const s=t[n];s!==void 0&&this.setValue(e,n,s)}static upload(e,t,n,s){for(let r=0,o=t.length;r!==o;++r){const a=t[r],l=n[a.id];l.needsUpdate!==!1&&a.setValue(e,l.value,s)}}static seqWithValue(e,t){const n=[];for(let s=0,r=e.length;s!==r;++s){const o=e[s];o.id in t&&n.push(o)}return n}}function Rp(i,e,t){const n=i.createShader(e);return i.shaderSource(n,t),i.compileShader(n),n}const bT=37297;let MT=0;function TT(i,e){const t=i.split(`
`),n=[],s=Math.max(e-6,0),r=Math.min(e+6,t.length);for(let o=s;o<r;o++){const a=o+1;n.push(`${a===e?">":" "} ${a}: ${t[o]}`)}return n.join(`
`)}const Ip=new Qe;function CT(i){lt._getMatrix(Ip,lt.workingColorSpace,i);const e=`mat3( ${Ip.elements.map(t=>t.toFixed(4))} )`;switch(lt.getTransfer(i)){case Vl:return[e,"LinearTransferOETF"];case mt:return[e,"sRGBTransferOETF"];default:return Ze("WebGLProgram: Unsupported color space: ",i),[e,"LinearTransferOETF"]}}function Dp(i,e,t){const n=i.getShaderParameter(e,i.COMPILE_STATUS),r=(i.getShaderInfoLog(e)||"").trim();if(n&&r==="")return"";const o=/ERROR: 0:(\d+)/.exec(r);if(o){const a=parseInt(o[1]);return t.toUpperCase()+`

`+r+`

`+TT(i.getShaderSource(e),a)}else return r}function ET(i,e){const t=CT(e);return[`vec4 ${i}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}function wT(i,e){let t;switch(e){case TS:t="Linear";break;case CS:t="Reinhard";break;case ES:t="Cineon";break;case wS:t="ACESFilmic";break;case IS:t="AgX";break;case DS:t="Neutral";break;case RS:t="Custom";break;default:Ze("WebGLProgram: Unsupported toneMapping:",e),t="Linear"}return"vec3 "+i+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}const dl=new F;function RT(){lt.getLuminanceCoefficients(dl);const i=dl.x.toFixed(4),e=dl.y.toFixed(4),t=dl.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function IT(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(Qo).join(`
`)}function DT(i){const e=[];for(const t in i){const n=i[t];n!==!1&&e.push("#define "+t+" "+n)}return e.join(`
`)}function PT(i,e){const t={},n=i.getProgramParameter(e,i.ACTIVE_ATTRIBUTES);for(let s=0;s<n;s++){const r=i.getActiveAttrib(e,s),o=r.name;let a=1;r.type===i.FLOAT_MAT2&&(a=2),r.type===i.FLOAT_MAT3&&(a=3),r.type===i.FLOAT_MAT4&&(a=4),t[o]={type:r.type,location:i.getAttribLocation(e,o),locationSize:a}}return t}function Qo(i){return i!==""}function Pp(i,e){const t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return i.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function Fp(i,e){return i.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const FT=/^[ \t]*#include +<([\w\d./]+)>/gm;function Ff(i){return i.replace(FT,BT)}const LT=new Map;function BT(i,e){let t=et[e];if(t===void 0){const n=LT.get(e);if(n!==void 0)t=et[n],Ze('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,n);else throw new Error("Can not resolve #include <"+e+">")}return Ff(t)}const UT=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function Lp(i){return i.replace(UT,OT)}function OT(i,e,t,n){let s="";for(let r=parseInt(e);r<parseInt(t);r++)s+=n.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function Bp(i){let e=`precision ${i.precision} float;
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
#define LOW_PRECISION`),e}function NT(i){let e="SHADOWMAP_TYPE_BASIC";return i.shadowMapType===G0?e="SHADOWMAP_TYPE_PCF":i.shadowMapType===rS?e="SHADOWMAP_TYPE_PCF_SOFT":i.shadowMapType===ns&&(e="SHADOWMAP_TYPE_VSM"),e}function zT(i){let e="ENVMAP_TYPE_CUBE";if(i.envMap)switch(i.envMapMode){case So:case Ao:e="ENVMAP_TYPE_CUBE";break;case hc:e="ENVMAP_TYPE_CUBE_UV";break}return e}function kT(i){let e="ENVMAP_MODE_REFLECTION";return i.envMap&&i.envMapMode===Ao&&(e="ENVMAP_MODE_REFRACTION"),e}function HT(i){let e="ENVMAP_BLENDING_NONE";if(i.envMap)switch(i.combine){case X0:e="ENVMAP_BLENDING_MULTIPLY";break;case bS:e="ENVMAP_BLENDING_MIX";break;case MS:e="ENVMAP_BLENDING_ADD";break}return e}function VT(i){const e=i.envMapCubeUVHeight;if(e===null)return null;const t=Math.log2(e)-2,n=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:n,maxMip:t}}function GT(i,e,t,n){const s=i.getContext(),r=t.defines;let o=t.vertexShader,a=t.fragmentShader;const l=NT(t),c=zT(t),u=kT(t),f=HT(t),h=VT(t),d=IT(t),x=DT(r),p=s.createProgram();let g,m,_=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(g=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x].filter(Qo).join(`
`),g.length>0&&(g+=`
`),m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x].filter(Qo).join(`
`),m.length>0&&(m+=`
`)):(g=[Bp(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+u:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(Qo).join(`
`),m=[Bp(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+c:"",t.envMap?"#define "+u:"",t.envMap?"#define "+f:"",h?"#define CUBEUV_TEXEL_WIDTH "+h.texelWidth:"",h?"#define CUBEUV_TEXEL_HEIGHT "+h.texelHeight:"",h?"#define CUBEUV_MAX_MIP "+h.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor||t.batchingColor?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==zs?"#define TONE_MAPPING":"",t.toneMapping!==zs?et.tonemapping_pars_fragment:"",t.toneMapping!==zs?wT("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",et.colorspace_pars_fragment,ET("linearToOutputTexel",t.outputColorSpace),RT(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(Qo).join(`
`)),o=Ff(o),o=Pp(o,t),o=Fp(o,t),a=Ff(a),a=Pp(a,t),a=Fp(a,t),o=Lp(o),a=Lp(a),t.isRawShaderMaterial!==!0&&(_=`#version 300 es
`,g=[d,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+g,m=["#define varying in",t.glslVersion===Qd?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===Qd?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+m);const S=_+g+o,A=_+m+a,y=Rp(s,s.VERTEX_SHADER,S),b=Rp(s,s.FRAGMENT_SHADER,A);s.attachShader(p,y),s.attachShader(p,b),t.index0AttributeName!==void 0?s.bindAttribLocation(p,0,t.index0AttributeName):t.morphTargets===!0&&s.bindAttribLocation(p,0,"position"),s.linkProgram(p);function v(I){if(i.debug.checkShaderErrors){const P=s.getProgramInfoLog(p)||"",B=s.getShaderInfoLog(y)||"",N=s.getShaderInfoLog(b)||"",G=P.trim(),V=B.trim(),q=N.trim();let X=!0,ee=!0;if(s.getProgramParameter(p,s.LINK_STATUS)===!1)if(X=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(s,p,y,b);else{const ce=Dp(s,y,"vertex"),be=Dp(s,b,"fragment");Wt("THREE.WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(p,s.VALIDATE_STATUS)+`

Material Name: `+I.name+`
Material Type: `+I.type+`

Program Info Log: `+G+`
`+ce+`
`+be)}else G!==""?Ze("WebGLProgram: Program Info Log:",G):(V===""||q==="")&&(ee=!1);ee&&(I.diagnostics={runnable:X,programLog:G,vertexShader:{log:V,prefix:g},fragmentShader:{log:q,prefix:m}})}s.deleteShader(y),s.deleteShader(b),E=new Pl(s,p),M=PT(s,p)}let E;this.getUniforms=function(){return E===void 0&&v(this),E};let M;this.getAttributes=function(){return M===void 0&&v(this),M};let T=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return T===!1&&(T=s.getProgramParameter(p,bT)),T},this.destroy=function(){n.releaseStatesOfProgram(this),s.deleteProgram(p),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=MT++,this.cacheKey=e,this.usedTimes=1,this.program=p,this.vertexShader=y,this.fragmentShader=b,this}let WT=0;class XT{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const t=e.vertexShader,n=e.fragmentShader,s=this._getShaderStage(t),r=this._getShaderStage(n),o=this._getShaderCacheForMaterial(e);return o.has(s)===!1&&(o.add(s),s.usedTimes++),o.has(r)===!1&&(o.add(r),r.usedTimes++),this}remove(e){const t=this.materialCache.get(e);for(const n of t)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const t=this.materialCache;let n=t.get(e);return n===void 0&&(n=new Set,t.set(e,n)),n}_getShaderStage(e){const t=this.shaderCache;let n=t.get(e);return n===void 0&&(n=new qT(e),t.set(e,n)),n}}class qT{constructor(e){this.id=WT++,this.code=e,this.usedTimes=0}}function YT(i,e,t,n,s,r,o){const a=new ig,l=new XT,c=new Set,u=[],f=s.logarithmicDepthBuffer,h=s.vertexTextures;let d=s.precision;const x={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function p(M){return c.add(M),M===0?"uv":`uv${M}`}function g(M,T,I,P,B){const N=P.fog,G=B.geometry,V=M.isMeshStandardMaterial?P.environment:null,q=(M.isMeshStandardMaterial?t:e).get(M.envMap||V),X=q&&q.mapping===hc?q.image.height:null,ee=x[M.type];M.precision!==null&&(d=s.getMaxPrecision(M.precision),d!==M.precision&&Ze("WebGLProgram.getParameters:",M.precision,"not supported, using",d,"instead."));const ce=G.morphAttributes.position||G.morphAttributes.normal||G.morphAttributes.color,be=ce!==void 0?ce.length:0;let Re=0;G.morphAttributes.position!==void 0&&(Re=1),G.morphAttributes.normal!==void 0&&(Re=2),G.morphAttributes.color!==void 0&&(Re=3);let Fe,Oe,Ne,J;if(ee){const Ge=Bi[ee];Fe=Ge.vertexShader,Oe=Ge.fragmentShader}else Fe=M.vertexShader,Oe=M.fragmentShader,l.update(M),Ne=l.getVertexShaderID(M),J=l.getFragmentShaderID(M);const ne=i.getRenderTarget(),xe=i.state.buffers.depth.getReversed(),Be=B.isInstancedMesh===!0,Te=B.isBatchedMesh===!0,Ve=!!M.map,L=!!M.matcap,U=!!q,Y=!!M.aoMap,w=!!M.lightMap,oe=!!M.bumpMap,re=!!M.normalMap,pe=!!M.displacementMap,se=!!M.emissiveMap,me=!!M.metalnessMap,ie=!!M.roughnessMap,Ae=M.anisotropy>0,R=M.clearcoat>0,C=M.dispersion>0,W=M.iridescence>0,$=M.sheen>0,fe=M.transmission>0,Z=Ae&&!!M.anisotropyMap,Ie=R&&!!M.clearcoatMap,ye=R&&!!M.clearcoatNormalMap,Ue=R&&!!M.clearcoatRoughnessMap,k=W&&!!M.iridescenceMap,te=W&&!!M.iridescenceThicknessMap,_e=$&&!!M.sheenColorMap,H=$&&!!M.sheenRoughnessMap,z=!!M.specularMap,he=!!M.specularColorMap,Me=!!M.specularIntensityMap,O=fe&&!!M.transmissionMap,ve=fe&&!!M.thicknessMap,ge=!!M.gradientMap,Se=!!M.alphaMap,de=M.alphaTest>0,le=!!M.alphaHash,Ce=!!M.extensions;let ze=zs;M.toneMapped&&(ne===null||ne.isXRRenderTarget===!0)&&(ze=i.toneMapping);const it={shaderID:ee,shaderType:M.type,shaderName:M.name,vertexShader:Fe,fragmentShader:Oe,defines:M.defines,customVertexShaderID:Ne,customFragmentShaderID:J,isRawShaderMaterial:M.isRawShaderMaterial===!0,glslVersion:M.glslVersion,precision:d,batching:Te,batchingColor:Te&&B._colorsTexture!==null,instancing:Be,instancingColor:Be&&B.instanceColor!==null,instancingMorph:Be&&B.morphTexture!==null,supportsVertexTextures:h,outputColorSpace:ne===null?i.outputColorSpace:ne.isXRRenderTarget===!0?ne.texture.colorSpace:bo,alphaToCoverage:!!M.alphaToCoverage,map:Ve,matcap:L,envMap:U,envMapMode:U&&q.mapping,envMapCubeUVHeight:X,aoMap:Y,lightMap:w,bumpMap:oe,normalMap:re,displacementMap:h&&pe,emissiveMap:se,normalMapObjectSpace:re&&M.normalMapType===US,normalMapTangentSpace:re&&M.normalMapType===BS,metalnessMap:me,roughnessMap:ie,anisotropy:Ae,anisotropyMap:Z,clearcoat:R,clearcoatMap:Ie,clearcoatNormalMap:ye,clearcoatRoughnessMap:Ue,dispersion:C,iridescence:W,iridescenceMap:k,iridescenceThicknessMap:te,sheen:$,sheenColorMap:_e,sheenRoughnessMap:H,specularMap:z,specularColorMap:he,specularIntensityMap:Me,transmission:fe,transmissionMap:O,thicknessMap:ve,gradientMap:ge,opaque:M.transparent===!1&&M.blending===Ns&&M.alphaToCoverage===!1,alphaMap:Se,alphaTest:de,alphaHash:le,combine:M.combine,mapUv:Ve&&p(M.map.channel),aoMapUv:Y&&p(M.aoMap.channel),lightMapUv:w&&p(M.lightMap.channel),bumpMapUv:oe&&p(M.bumpMap.channel),normalMapUv:re&&p(M.normalMap.channel),displacementMapUv:pe&&p(M.displacementMap.channel),emissiveMapUv:se&&p(M.emissiveMap.channel),metalnessMapUv:me&&p(M.metalnessMap.channel),roughnessMapUv:ie&&p(M.roughnessMap.channel),anisotropyMapUv:Z&&p(M.anisotropyMap.channel),clearcoatMapUv:Ie&&p(M.clearcoatMap.channel),clearcoatNormalMapUv:ye&&p(M.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Ue&&p(M.clearcoatRoughnessMap.channel),iridescenceMapUv:k&&p(M.iridescenceMap.channel),iridescenceThicknessMapUv:te&&p(M.iridescenceThicknessMap.channel),sheenColorMapUv:_e&&p(M.sheenColorMap.channel),sheenRoughnessMapUv:H&&p(M.sheenRoughnessMap.channel),specularMapUv:z&&p(M.specularMap.channel),specularColorMapUv:he&&p(M.specularColorMap.channel),specularIntensityMapUv:Me&&p(M.specularIntensityMap.channel),transmissionMapUv:O&&p(M.transmissionMap.channel),thicknessMapUv:ve&&p(M.thicknessMap.channel),alphaMapUv:Se&&p(M.alphaMap.channel),vertexTangents:!!G.attributes.tangent&&(re||Ae),vertexColors:M.vertexColors,vertexAlphas:M.vertexColors===!0&&!!G.attributes.color&&G.attributes.color.itemSize===4,pointsUvs:B.isPoints===!0&&!!G.attributes.uv&&(Ve||Se),fog:!!N,useFog:M.fog===!0,fogExp2:!!N&&N.isFogExp2,flatShading:M.flatShading===!0&&M.wireframe===!1,sizeAttenuation:M.sizeAttenuation===!0,logarithmicDepthBuffer:f,reversedDepthBuffer:xe,skinning:B.isSkinnedMesh===!0,morphTargets:G.morphAttributes.position!==void 0,morphNormals:G.morphAttributes.normal!==void 0,morphColors:G.morphAttributes.color!==void 0,morphTargetsCount:be,morphTextureStride:Re,numDirLights:T.directional.length,numPointLights:T.point.length,numSpotLights:T.spot.length,numSpotLightMaps:T.spotLightMap.length,numRectAreaLights:T.rectArea.length,numHemiLights:T.hemi.length,numDirLightShadows:T.directionalShadowMap.length,numPointLightShadows:T.pointShadowMap.length,numSpotLightShadows:T.spotShadowMap.length,numSpotLightShadowsWithMaps:T.numSpotLightShadowsWithMaps,numLightProbes:T.numLightProbes,numClippingPlanes:o.numPlanes,numClipIntersection:o.numIntersection,dithering:M.dithering,shadowMapEnabled:i.shadowMap.enabled&&I.length>0,shadowMapType:i.shadowMap.type,toneMapping:ze,decodeVideoTexture:Ve&&M.map.isVideoTexture===!0&&lt.getTransfer(M.map.colorSpace)===mt,decodeVideoTextureEmissive:se&&M.emissiveMap.isVideoTexture===!0&&lt.getTransfer(M.emissiveMap.colorSpace)===mt,premultipliedAlpha:M.premultipliedAlpha,doubleSided:M.side===fi,flipSided:M.side===Bn,useDepthPacking:M.depthPacking>=0,depthPacking:M.depthPacking||0,index0AttributeName:M.index0AttributeName,extensionClipCullDistance:Ce&&M.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Ce&&M.extensions.multiDraw===!0||Te)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:M.customProgramCacheKey()};return it.vertexUv1s=c.has(1),it.vertexUv2s=c.has(2),it.vertexUv3s=c.has(3),c.clear(),it}function m(M){const T=[];if(M.shaderID?T.push(M.shaderID):(T.push(M.customVertexShaderID),T.push(M.customFragmentShaderID)),M.defines!==void 0)for(const I in M.defines)T.push(I),T.push(M.defines[I]);return M.isRawShaderMaterial===!1&&(_(T,M),S(T,M),T.push(i.outputColorSpace)),T.push(M.customProgramCacheKey),T.join()}function _(M,T){M.push(T.precision),M.push(T.outputColorSpace),M.push(T.envMapMode),M.push(T.envMapCubeUVHeight),M.push(T.mapUv),M.push(T.alphaMapUv),M.push(T.lightMapUv),M.push(T.aoMapUv),M.push(T.bumpMapUv),M.push(T.normalMapUv),M.push(T.displacementMapUv),M.push(T.emissiveMapUv),M.push(T.metalnessMapUv),M.push(T.roughnessMapUv),M.push(T.anisotropyMapUv),M.push(T.clearcoatMapUv),M.push(T.clearcoatNormalMapUv),M.push(T.clearcoatRoughnessMapUv),M.push(T.iridescenceMapUv),M.push(T.iridescenceThicknessMapUv),M.push(T.sheenColorMapUv),M.push(T.sheenRoughnessMapUv),M.push(T.specularMapUv),M.push(T.specularColorMapUv),M.push(T.specularIntensityMapUv),M.push(T.transmissionMapUv),M.push(T.thicknessMapUv),M.push(T.combine),M.push(T.fogExp2),M.push(T.sizeAttenuation),M.push(T.morphTargetsCount),M.push(T.morphAttributeCount),M.push(T.numDirLights),M.push(T.numPointLights),M.push(T.numSpotLights),M.push(T.numSpotLightMaps),M.push(T.numHemiLights),M.push(T.numRectAreaLights),M.push(T.numDirLightShadows),M.push(T.numPointLightShadows),M.push(T.numSpotLightShadows),M.push(T.numSpotLightShadowsWithMaps),M.push(T.numLightProbes),M.push(T.shadowMapType),M.push(T.toneMapping),M.push(T.numClippingPlanes),M.push(T.numClipIntersection),M.push(T.depthPacking)}function S(M,T){a.disableAll(),T.supportsVertexTextures&&a.enable(0),T.instancing&&a.enable(1),T.instancingColor&&a.enable(2),T.instancingMorph&&a.enable(3),T.matcap&&a.enable(4),T.envMap&&a.enable(5),T.normalMapObjectSpace&&a.enable(6),T.normalMapTangentSpace&&a.enable(7),T.clearcoat&&a.enable(8),T.iridescence&&a.enable(9),T.alphaTest&&a.enable(10),T.vertexColors&&a.enable(11),T.vertexAlphas&&a.enable(12),T.vertexUv1s&&a.enable(13),T.vertexUv2s&&a.enable(14),T.vertexUv3s&&a.enable(15),T.vertexTangents&&a.enable(16),T.anisotropy&&a.enable(17),T.alphaHash&&a.enable(18),T.batching&&a.enable(19),T.dispersion&&a.enable(20),T.batchingColor&&a.enable(21),T.gradientMap&&a.enable(22),M.push(a.mask),a.disableAll(),T.fog&&a.enable(0),T.useFog&&a.enable(1),T.flatShading&&a.enable(2),T.logarithmicDepthBuffer&&a.enable(3),T.reversedDepthBuffer&&a.enable(4),T.skinning&&a.enable(5),T.morphTargets&&a.enable(6),T.morphNormals&&a.enable(7),T.morphColors&&a.enable(8),T.premultipliedAlpha&&a.enable(9),T.shadowMapEnabled&&a.enable(10),T.doubleSided&&a.enable(11),T.flipSided&&a.enable(12),T.useDepthPacking&&a.enable(13),T.dithering&&a.enable(14),T.transmission&&a.enable(15),T.sheen&&a.enable(16),T.opaque&&a.enable(17),T.pointsUvs&&a.enable(18),T.decodeVideoTexture&&a.enable(19),T.decodeVideoTextureEmissive&&a.enable(20),T.alphaToCoverage&&a.enable(21),M.push(a.mask)}function A(M){const T=x[M.type];let I;if(T){const P=Bi[T];I=mA.clone(P.uniforms)}else I=M.uniforms;return I}function y(M,T){let I;for(let P=0,B=u.length;P<B;P++){const N=u[P];if(N.cacheKey===T){I=N,++I.usedTimes;break}}return I===void 0&&(I=new GT(i,T,M,r),u.push(I)),I}function b(M){if(--M.usedTimes===0){const T=u.indexOf(M);u[T]=u[u.length-1],u.pop(),M.destroy()}}function v(M){l.remove(M)}function E(){l.dispose()}return{getParameters:g,getProgramCacheKey:m,getUniforms:A,acquireProgram:y,releaseProgram:b,releaseShaderCache:v,programs:u,dispose:E}}function QT(){let i=new WeakMap;function e(o){return i.has(o)}function t(o){let a=i.get(o);return a===void 0&&(a={},i.set(o,a)),a}function n(o){i.delete(o)}function s(o,a,l){i.get(o)[a]=l}function r(){i=new WeakMap}return{has:e,get:t,remove:n,update:s,dispose:r}}function KT(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.material.id!==e.material.id?i.material.id-e.material.id:i.z!==e.z?i.z-e.z:i.id-e.id}function Up(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.z!==e.z?e.z-i.z:i.id-e.id}function Op(){const i=[];let e=0;const t=[],n=[],s=[];function r(){e=0,t.length=0,n.length=0,s.length=0}function o(f,h,d,x,p,g){let m=i[e];return m===void 0?(m={id:f.id,object:f,geometry:h,material:d,groupOrder:x,renderOrder:f.renderOrder,z:p,group:g},i[e]=m):(m.id=f.id,m.object=f,m.geometry=h,m.material=d,m.groupOrder=x,m.renderOrder=f.renderOrder,m.z=p,m.group=g),e++,m}function a(f,h,d,x,p,g){const m=o(f,h,d,x,p,g);d.transmission>0?n.push(m):d.transparent===!0?s.push(m):t.push(m)}function l(f,h,d,x,p,g){const m=o(f,h,d,x,p,g);d.transmission>0?n.unshift(m):d.transparent===!0?s.unshift(m):t.unshift(m)}function c(f,h){t.length>1&&t.sort(f||KT),n.length>1&&n.sort(h||Up),s.length>1&&s.sort(h||Up)}function u(){for(let f=e,h=i.length;f<h;f++){const d=i[f];if(d.id===null)break;d.id=null,d.object=null,d.geometry=null,d.material=null,d.group=null}}return{opaque:t,transmissive:n,transparent:s,init:r,push:a,unshift:l,finish:u,sort:c}}function jT(){let i=new WeakMap;function e(n,s){const r=i.get(n);let o;return r===void 0?(o=new Op,i.set(n,[o])):s>=r.length?(o=new Op,r.push(o)):o=r[s],o}function t(){i=new WeakMap}return{get:e,dispose:t}}function $T(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new F,color:new rt};break;case"SpotLight":t={position:new F,direction:new F,color:new rt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new F,color:new rt,distance:0,decay:0};break;case"HemisphereLight":t={direction:new F,skyColor:new rt,groundColor:new rt};break;case"RectAreaLight":t={color:new rt,position:new F,halfWidth:new F,halfHeight:new F};break}return i[e.id]=t,t}}}function ZT(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Pe};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Pe};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Pe,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[e.id]=t,t}}}let JT=0;function eC(i,e){return(e.castShadow?2:0)-(i.castShadow?2:0)+(e.map?1:0)-(i.map?1:0)}function tC(i){const e=new $T,t=ZT(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)n.probe.push(new F);const s=new F,r=new Ye,o=new Ye;function a(c){let u=0,f=0,h=0;for(let M=0;M<9;M++)n.probe[M].set(0,0,0);let d=0,x=0,p=0,g=0,m=0,_=0,S=0,A=0,y=0,b=0,v=0;c.sort(eC);for(let M=0,T=c.length;M<T;M++){const I=c[M],P=I.color,B=I.intensity,N=I.distance,G=I.shadow&&I.shadow.map?I.shadow.map.texture:null;if(I.isAmbientLight)u+=P.r*B,f+=P.g*B,h+=P.b*B;else if(I.isLightProbe){for(let V=0;V<9;V++)n.probe[V].addScaledVector(I.sh.coefficients[V],B);v++}else if(I.isDirectionalLight){const V=e.get(I);if(V.color.copy(I.color).multiplyScalar(I.intensity),I.castShadow){const q=I.shadow,X=t.get(I);X.shadowIntensity=q.intensity,X.shadowBias=q.bias,X.shadowNormalBias=q.normalBias,X.shadowRadius=q.radius,X.shadowMapSize=q.mapSize,n.directionalShadow[d]=X,n.directionalShadowMap[d]=G,n.directionalShadowMatrix[d]=I.shadow.matrix,_++}n.directional[d]=V,d++}else if(I.isSpotLight){const V=e.get(I);V.position.setFromMatrixPosition(I.matrixWorld),V.color.copy(P).multiplyScalar(B),V.distance=N,V.coneCos=Math.cos(I.angle),V.penumbraCos=Math.cos(I.angle*(1-I.penumbra)),V.decay=I.decay,n.spot[p]=V;const q=I.shadow;if(I.map&&(n.spotLightMap[y]=I.map,y++,q.updateMatrices(I),I.castShadow&&b++),n.spotLightMatrix[p]=q.matrix,I.castShadow){const X=t.get(I);X.shadowIntensity=q.intensity,X.shadowBias=q.bias,X.shadowNormalBias=q.normalBias,X.shadowRadius=q.radius,X.shadowMapSize=q.mapSize,n.spotShadow[p]=X,n.spotShadowMap[p]=G,A++}p++}else if(I.isRectAreaLight){const V=e.get(I);V.color.copy(P).multiplyScalar(B),V.halfWidth.set(I.width*.5,0,0),V.halfHeight.set(0,I.height*.5,0),n.rectArea[g]=V,g++}else if(I.isPointLight){const V=e.get(I);if(V.color.copy(I.color).multiplyScalar(I.intensity),V.distance=I.distance,V.decay=I.decay,I.castShadow){const q=I.shadow,X=t.get(I);X.shadowIntensity=q.intensity,X.shadowBias=q.bias,X.shadowNormalBias=q.normalBias,X.shadowRadius=q.radius,X.shadowMapSize=q.mapSize,X.shadowCameraNear=q.camera.near,X.shadowCameraFar=q.camera.far,n.pointShadow[x]=X,n.pointShadowMap[x]=G,n.pointShadowMatrix[x]=I.shadow.matrix,S++}n.point[x]=V,x++}else if(I.isHemisphereLight){const V=e.get(I);V.skyColor.copy(I.color).multiplyScalar(B),V.groundColor.copy(I.groundColor).multiplyScalar(B),n.hemi[m]=V,m++}}g>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=De.LTC_FLOAT_1,n.rectAreaLTC2=De.LTC_FLOAT_2):(n.rectAreaLTC1=De.LTC_HALF_1,n.rectAreaLTC2=De.LTC_HALF_2)),n.ambient[0]=u,n.ambient[1]=f,n.ambient[2]=h;const E=n.hash;(E.directionalLength!==d||E.pointLength!==x||E.spotLength!==p||E.rectAreaLength!==g||E.hemiLength!==m||E.numDirectionalShadows!==_||E.numPointShadows!==S||E.numSpotShadows!==A||E.numSpotMaps!==y||E.numLightProbes!==v)&&(n.directional.length=d,n.spot.length=p,n.rectArea.length=g,n.point.length=x,n.hemi.length=m,n.directionalShadow.length=_,n.directionalShadowMap.length=_,n.pointShadow.length=S,n.pointShadowMap.length=S,n.spotShadow.length=A,n.spotShadowMap.length=A,n.directionalShadowMatrix.length=_,n.pointShadowMatrix.length=S,n.spotLightMatrix.length=A+y-b,n.spotLightMap.length=y,n.numSpotLightShadowsWithMaps=b,n.numLightProbes=v,E.directionalLength=d,E.pointLength=x,E.spotLength=p,E.rectAreaLength=g,E.hemiLength=m,E.numDirectionalShadows=_,E.numPointShadows=S,E.numSpotShadows=A,E.numSpotMaps=y,E.numLightProbes=v,n.version=JT++)}function l(c,u){let f=0,h=0,d=0,x=0,p=0;const g=u.matrixWorldInverse;for(let m=0,_=c.length;m<_;m++){const S=c[m];if(S.isDirectionalLight){const A=n.directional[f];A.direction.setFromMatrixPosition(S.matrixWorld),s.setFromMatrixPosition(S.target.matrixWorld),A.direction.sub(s),A.direction.transformDirection(g),f++}else if(S.isSpotLight){const A=n.spot[d];A.position.setFromMatrixPosition(S.matrixWorld),A.position.applyMatrix4(g),A.direction.setFromMatrixPosition(S.matrixWorld),s.setFromMatrixPosition(S.target.matrixWorld),A.direction.sub(s),A.direction.transformDirection(g),d++}else if(S.isRectAreaLight){const A=n.rectArea[x];A.position.setFromMatrixPosition(S.matrixWorld),A.position.applyMatrix4(g),o.identity(),r.copy(S.matrixWorld),r.premultiply(g),o.extractRotation(r),A.halfWidth.set(S.width*.5,0,0),A.halfHeight.set(0,S.height*.5,0),A.halfWidth.applyMatrix4(o),A.halfHeight.applyMatrix4(o),x++}else if(S.isPointLight){const A=n.point[h];A.position.setFromMatrixPosition(S.matrixWorld),A.position.applyMatrix4(g),h++}else if(S.isHemisphereLight){const A=n.hemi[p];A.direction.setFromMatrixPosition(S.matrixWorld),A.direction.transformDirection(g),p++}}}return{setup:a,setupView:l,state:n}}function Np(i){const e=new tC(i),t=[],n=[];function s(u){c.camera=u,t.length=0,n.length=0}function r(u){t.push(u)}function o(u){n.push(u)}function a(){e.setup(t)}function l(u){e.setupView(t,u)}const c={lightsArray:t,shadowsArray:n,camera:null,lights:e,transmissionRenderTarget:{}};return{init:s,state:c,setupLights:a,setupLightsView:l,pushLight:r,pushShadow:o}}function nC(i){let e=new WeakMap;function t(s,r=0){const o=e.get(s);let a;return o===void 0?(a=new Np(i),e.set(s,[a])):r>=o.length?(a=new Np(i),o.push(a)):a=o[r],a}function n(){e=new WeakMap}return{get:t,dispose:n}}const iC=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,sC=`uniform sampler2D shadow_pass;
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
}`;function rC(i,e,t){let n=new ug;const s=new Pe,r=new Pe,o=new Dt,a=new wA({depthPacking:LS}),l=new RA,c={},u=t.maxTextureSize,f={[Xi]:Bn,[Bn]:Xi,[fi]:fi},h=new Cn({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Pe},radius:{value:4}},vertexShader:iC,fragmentShader:sC}),d=h.clone();d.defines.HORIZONTAL_PASS=1;const x=new En;x.setAttribute("position",new _i(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const p=new Yt(x,h),g=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=G0;let m=this.type;this.render=function(b,v,E){if(g.enabled===!1||g.autoUpdate===!1&&g.needsUpdate===!1||b.length===0)return;const M=i.getRenderTarget(),T=i.getActiveCubeFace(),I=i.getActiveMipmapLevel(),P=i.state;P.setBlending(ps),P.buffers.depth.getReversed()===!0?P.buffers.color.setClear(0,0,0,0):P.buffers.color.setClear(1,1,1,1),P.buffers.depth.setTest(!0),P.setScissorTest(!1);const B=m!==ns&&this.type===ns,N=m===ns&&this.type!==ns;for(let G=0,V=b.length;G<V;G++){const q=b[G],X=q.shadow;if(X===void 0){Ze("WebGLShadowMap:",q,"has no shadow.");continue}if(X.autoUpdate===!1&&X.needsUpdate===!1)continue;s.copy(X.mapSize);const ee=X.getFrameExtents();if(s.multiply(ee),r.copy(X.mapSize),(s.x>u||s.y>u)&&(s.x>u&&(r.x=Math.floor(u/ee.x),s.x=r.x*ee.x,X.mapSize.x=r.x),s.y>u&&(r.y=Math.floor(u/ee.y),s.y=r.y*ee.y,X.mapSize.y=r.y)),X.map===null||B===!0||N===!0){const be=this.type!==ns?{minFilter:Jn,magFilter:Jn}:{};X.map!==null&&X.map.dispose(),X.map=new Ws(s.x,s.y,be),X.map.texture.name=q.name+".shadowMap",X.camera.updateProjectionMatrix()}i.setRenderTarget(X.map),i.clear();const ce=X.getViewportCount();for(let be=0;be<ce;be++){const Re=X.getViewport(be);o.set(r.x*Re.x,r.y*Re.y,r.x*Re.z,r.y*Re.w),P.viewport(o),X.updateMatrices(q,be),n=X.getFrustum(),A(v,E,X.camera,q,this.type)}X.isPointLightShadow!==!0&&this.type===ns&&_(X,E),X.needsUpdate=!1}m=this.type,g.needsUpdate=!1,i.setRenderTarget(M,T,I)};function _(b,v){const E=e.update(p);h.defines.VSM_SAMPLES!==b.blurSamples&&(h.defines.VSM_SAMPLES=b.blurSamples,d.defines.VSM_SAMPLES=b.blurSamples,h.needsUpdate=!0,d.needsUpdate=!0),b.mapPass===null&&(b.mapPass=new Ws(s.x,s.y)),h.uniforms.shadow_pass.value=b.map.texture,h.uniforms.resolution.value=b.mapSize,h.uniforms.radius.value=b.radius,i.setRenderTarget(b.mapPass),i.clear(),i.renderBufferDirect(v,null,E,h,p,null),d.uniforms.shadow_pass.value=b.mapPass.texture,d.uniforms.resolution.value=b.mapSize,d.uniforms.radius.value=b.radius,i.setRenderTarget(b.map),i.clear(),i.renderBufferDirect(v,null,E,d,p,null)}function S(b,v,E,M){let T=null;const I=E.isPointLight===!0?b.customDistanceMaterial:b.customDepthMaterial;if(I!==void 0)T=I;else if(T=E.isPointLight===!0?l:a,i.localClippingEnabled&&v.clipShadows===!0&&Array.isArray(v.clippingPlanes)&&v.clippingPlanes.length!==0||v.displacementMap&&v.displacementScale!==0||v.alphaMap&&v.alphaTest>0||v.map&&v.alphaTest>0||v.alphaToCoverage===!0){const P=T.uuid,B=v.uuid;let N=c[P];N===void 0&&(N={},c[P]=N);let G=N[B];G===void 0&&(G=T.clone(),N[B]=G,v.addEventListener("dispose",y)),T=G}if(T.visible=v.visible,T.wireframe=v.wireframe,M===ns?T.side=v.shadowSide!==null?v.shadowSide:v.side:T.side=v.shadowSide!==null?v.shadowSide:f[v.side],T.alphaMap=v.alphaMap,T.alphaTest=v.alphaToCoverage===!0?.5:v.alphaTest,T.map=v.map,T.clipShadows=v.clipShadows,T.clippingPlanes=v.clippingPlanes,T.clipIntersection=v.clipIntersection,T.displacementMap=v.displacementMap,T.displacementScale=v.displacementScale,T.displacementBias=v.displacementBias,T.wireframeLinewidth=v.wireframeLinewidth,T.linewidth=v.linewidth,E.isPointLight===!0&&T.isMeshDistanceMaterial===!0){const P=i.properties.get(T);P.light=E}return T}function A(b,v,E,M,T){if(b.visible===!1)return;if(b.layers.test(v.layers)&&(b.isMesh||b.isLine||b.isPoints)&&(b.castShadow||b.receiveShadow&&T===ns)&&(!b.frustumCulled||n.intersectsObject(b))){b.modelViewMatrix.multiplyMatrices(E.matrixWorldInverse,b.matrixWorld);const B=e.update(b),N=b.material;if(Array.isArray(N)){const G=B.groups;for(let V=0,q=G.length;V<q;V++){const X=G[V],ee=N[X.materialIndex];if(ee&&ee.visible){const ce=S(b,ee,M,T);b.onBeforeShadow(i,b,v,E,B,ce,X),i.renderBufferDirect(E,null,B,ce,b,X),b.onAfterShadow(i,b,v,E,B,ce,X)}}}else if(N.visible){const G=S(b,N,M,T);b.onBeforeShadow(i,b,v,E,B,G,null),i.renderBufferDirect(E,null,B,G,b,null),b.onAfterShadow(i,b,v,E,B,G,null)}}const P=b.children;for(let B=0,N=P.length;B<N;B++)A(P[B],v,E,M,T)}function y(b){b.target.removeEventListener("dispose",y);for(const E in c){const M=c[E],T=b.target.uuid;T in M&&(M[T].dispose(),delete M[T])}}}const oC={[Xu]:qu,[Yu]:ju,[Qu]:$u,[vo]:Ku,[qu]:Xu,[ju]:Yu,[$u]:Qu,[Ku]:vo};function aC(i,e){function t(){let O=!1;const ve=new Dt;let ge=null;const Se=new Dt(0,0,0,0);return{setMask:function(de){ge!==de&&!O&&(i.colorMask(de,de,de,de),ge=de)},setLocked:function(de){O=de},setClear:function(de,le,Ce,ze,it){it===!0&&(de*=ze,le*=ze,Ce*=ze),ve.set(de,le,Ce,ze),Se.equals(ve)===!1&&(i.clearColor(de,le,Ce,ze),Se.copy(ve))},reset:function(){O=!1,ge=null,Se.set(-1,0,0,0)}}}function n(){let O=!1,ve=!1,ge=null,Se=null,de=null;return{setReversed:function(le){if(ve!==le){const Ce=e.get("EXT_clip_control");le?Ce.clipControlEXT(Ce.LOWER_LEFT_EXT,Ce.ZERO_TO_ONE_EXT):Ce.clipControlEXT(Ce.LOWER_LEFT_EXT,Ce.NEGATIVE_ONE_TO_ONE_EXT),ve=le;const ze=de;de=null,this.setClear(ze)}},getReversed:function(){return ve},setTest:function(le){le?ne(i.DEPTH_TEST):xe(i.DEPTH_TEST)},setMask:function(le){ge!==le&&!O&&(i.depthMask(le),ge=le)},setFunc:function(le){if(ve&&(le=oC[le]),Se!==le){switch(le){case Xu:i.depthFunc(i.NEVER);break;case qu:i.depthFunc(i.ALWAYS);break;case Yu:i.depthFunc(i.LESS);break;case vo:i.depthFunc(i.LEQUAL);break;case Qu:i.depthFunc(i.EQUAL);break;case Ku:i.depthFunc(i.GEQUAL);break;case ju:i.depthFunc(i.GREATER);break;case $u:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}Se=le}},setLocked:function(le){O=le},setClear:function(le){de!==le&&(ve&&(le=1-le),i.clearDepth(le),de=le)},reset:function(){O=!1,ge=null,Se=null,de=null,ve=!1}}}function s(){let O=!1,ve=null,ge=null,Se=null,de=null,le=null,Ce=null,ze=null,it=null;return{setTest:function(Ge){O||(Ge?ne(i.STENCIL_TEST):xe(i.STENCIL_TEST))},setMask:function(Ge){ve!==Ge&&!O&&(i.stencilMask(Ge),ve=Ge)},setFunc:function(Ge,vt,Ct){(ge!==Ge||Se!==vt||de!==Ct)&&(i.stencilFunc(Ge,vt,Ct),ge=Ge,Se=vt,de=Ct)},setOp:function(Ge,vt,Ct){(le!==Ge||Ce!==vt||ze!==Ct)&&(i.stencilOp(Ge,vt,Ct),le=Ge,Ce=vt,ze=Ct)},setLocked:function(Ge){O=Ge},setClear:function(Ge){it!==Ge&&(i.clearStencil(Ge),it=Ge)},reset:function(){O=!1,ve=null,ge=null,Se=null,de=null,le=null,Ce=null,ze=null,it=null}}}const r=new t,o=new n,a=new s,l=new WeakMap,c=new WeakMap;let u={},f={},h=new WeakMap,d=[],x=null,p=!1,g=null,m=null,_=null,S=null,A=null,y=null,b=null,v=new rt(0,0,0),E=0,M=!1,T=null,I=null,P=null,B=null,N=null;const G=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let V=!1,q=0;const X=i.getParameter(i.VERSION);X.indexOf("WebGL")!==-1?(q=parseFloat(/^WebGL (\d)/.exec(X)[1]),V=q>=1):X.indexOf("OpenGL ES")!==-1&&(q=parseFloat(/^OpenGL ES (\d)/.exec(X)[1]),V=q>=2);let ee=null,ce={};const be=i.getParameter(i.SCISSOR_BOX),Re=i.getParameter(i.VIEWPORT),Fe=new Dt().fromArray(be),Oe=new Dt().fromArray(Re);function Ne(O,ve,ge,Se){const de=new Uint8Array(4),le=i.createTexture();i.bindTexture(O,le),i.texParameteri(O,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(O,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let Ce=0;Ce<ge;Ce++)O===i.TEXTURE_3D||O===i.TEXTURE_2D_ARRAY?i.texImage3D(ve,0,i.RGBA,1,1,Se,0,i.RGBA,i.UNSIGNED_BYTE,de):i.texImage2D(ve+Ce,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,de);return le}const J={};J[i.TEXTURE_2D]=Ne(i.TEXTURE_2D,i.TEXTURE_2D,1),J[i.TEXTURE_CUBE_MAP]=Ne(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),J[i.TEXTURE_2D_ARRAY]=Ne(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),J[i.TEXTURE_3D]=Ne(i.TEXTURE_3D,i.TEXTURE_3D,1,1),r.setClear(0,0,0,1),o.setClear(1),a.setClear(0),ne(i.DEPTH_TEST),o.setFunc(vo),oe(!1),re(Vd),ne(i.CULL_FACE),Y(ps);function ne(O){u[O]!==!0&&(i.enable(O),u[O]=!0)}function xe(O){u[O]!==!1&&(i.disable(O),u[O]=!1)}function Be(O,ve){return f[O]!==ve?(i.bindFramebuffer(O,ve),f[O]=ve,O===i.DRAW_FRAMEBUFFER&&(f[i.FRAMEBUFFER]=ve),O===i.FRAMEBUFFER&&(f[i.DRAW_FRAMEBUFFER]=ve),!0):!1}function Te(O,ve){let ge=d,Se=!1;if(O){ge=h.get(ve),ge===void 0&&(ge=[],h.set(ve,ge));const de=O.textures;if(ge.length!==de.length||ge[0]!==i.COLOR_ATTACHMENT0){for(let le=0,Ce=de.length;le<Ce;le++)ge[le]=i.COLOR_ATTACHMENT0+le;ge.length=de.length,Se=!0}}else ge[0]!==i.BACK&&(ge[0]=i.BACK,Se=!0);Se&&i.drawBuffers(ge)}function Ve(O){return x!==O?(i.useProgram(O),x=O,!0):!1}const L={[dr]:i.FUNC_ADD,[oS]:i.FUNC_SUBTRACT,[aS]:i.FUNC_REVERSE_SUBTRACT};L[lS]=i.MIN,L[cS]=i.MAX;const U={[uS]:i.ZERO,[fS]:i.ONE,[hS]:i.SRC_COLOR,[xa]:i.SRC_ALPHA,[_S]:i.SRC_ALPHA_SATURATE,[gS]:i.DST_COLOR,[pS]:i.DST_ALPHA,[dS]:i.ONE_MINUS_SRC_COLOR,[_a]:i.ONE_MINUS_SRC_ALPHA,[xS]:i.ONE_MINUS_DST_COLOR,[mS]:i.ONE_MINUS_DST_ALPHA,[vS]:i.CONSTANT_COLOR,[SS]:i.ONE_MINUS_CONSTANT_COLOR,[AS]:i.CONSTANT_ALPHA,[yS]:i.ONE_MINUS_CONSTANT_ALPHA};function Y(O,ve,ge,Se,de,le,Ce,ze,it,Ge){if(O===ps){p===!0&&(xe(i.BLEND),p=!1);return}if(p===!1&&(ne(i.BLEND),p=!0),O!==W0){if(O!==g||Ge!==M){if((m!==dr||A!==dr)&&(i.blendEquation(i.FUNC_ADD),m=dr,A=dr),Ge)switch(O){case Ns:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Gd:i.blendFunc(i.ONE,i.ONE);break;case Wd:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case Xd:i.blendFuncSeparate(i.DST_COLOR,i.ONE_MINUS_SRC_ALPHA,i.ZERO,i.ONE);break;default:Wt("WebGLState: Invalid blending: ",O);break}else switch(O){case Ns:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Gd:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE,i.ONE,i.ONE);break;case Wd:Wt("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case Xd:Wt("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:Wt("WebGLState: Invalid blending: ",O);break}_=null,S=null,y=null,b=null,v.set(0,0,0),E=0,g=O,M=Ge}return}de=de||ve,le=le||ge,Ce=Ce||Se,(ve!==m||de!==A)&&(i.blendEquationSeparate(L[ve],L[de]),m=ve,A=de),(ge!==_||Se!==S||le!==y||Ce!==b)&&(i.blendFuncSeparate(U[ge],U[Se],U[le],U[Ce]),_=ge,S=Se,y=le,b=Ce),(ze.equals(v)===!1||it!==E)&&(i.blendColor(ze.r,ze.g,ze.b,it),v.copy(ze),E=it),g=O,M=!1}function w(O,ve){O.side===fi?xe(i.CULL_FACE):ne(i.CULL_FACE);let ge=O.side===Bn;ve&&(ge=!ge),oe(ge),O.blending===Ns&&O.transparent===!1?Y(ps):Y(O.blending,O.blendEquation,O.blendSrc,O.blendDst,O.blendEquationAlpha,O.blendSrcAlpha,O.blendDstAlpha,O.blendColor,O.blendAlpha,O.premultipliedAlpha),o.setFunc(O.depthFunc),o.setTest(O.depthTest),o.setMask(O.depthWrite),r.setMask(O.colorWrite);const Se=O.stencilWrite;a.setTest(Se),Se&&(a.setMask(O.stencilWriteMask),a.setFunc(O.stencilFunc,O.stencilRef,O.stencilFuncMask),a.setOp(O.stencilFail,O.stencilZFail,O.stencilZPass)),se(O.polygonOffset,O.polygonOffsetFactor,O.polygonOffsetUnits),O.alphaToCoverage===!0?ne(i.SAMPLE_ALPHA_TO_COVERAGE):xe(i.SAMPLE_ALPHA_TO_COVERAGE)}function oe(O){T!==O&&(O?i.frontFace(i.CW):i.frontFace(i.CCW),T=O)}function re(O){O!==iS?(ne(i.CULL_FACE),O!==I&&(O===Vd?i.cullFace(i.BACK):O===sS?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):xe(i.CULL_FACE),I=O}function pe(O){O!==P&&(V&&i.lineWidth(O),P=O)}function se(O,ve,ge){O?(ne(i.POLYGON_OFFSET_FILL),(B!==ve||N!==ge)&&(i.polygonOffset(ve,ge),B=ve,N=ge)):xe(i.POLYGON_OFFSET_FILL)}function me(O){O?ne(i.SCISSOR_TEST):xe(i.SCISSOR_TEST)}function ie(O){O===void 0&&(O=i.TEXTURE0+G-1),ee!==O&&(i.activeTexture(O),ee=O)}function Ae(O,ve,ge){ge===void 0&&(ee===null?ge=i.TEXTURE0+G-1:ge=ee);let Se=ce[ge];Se===void 0&&(Se={type:void 0,texture:void 0},ce[ge]=Se),(Se.type!==O||Se.texture!==ve)&&(ee!==ge&&(i.activeTexture(ge),ee=ge),i.bindTexture(O,ve||J[O]),Se.type=O,Se.texture=ve)}function R(){const O=ce[ee];O!==void 0&&O.type!==void 0&&(i.bindTexture(O.type,null),O.type=void 0,O.texture=void 0)}function C(){try{i.compressedTexImage2D(...arguments)}catch(O){O("WebGLState:",O)}}function W(){try{i.compressedTexImage3D(...arguments)}catch(O){O("WebGLState:",O)}}function $(){try{i.texSubImage2D(...arguments)}catch(O){O("WebGLState:",O)}}function fe(){try{i.texSubImage3D(...arguments)}catch(O){O("WebGLState:",O)}}function Z(){try{i.compressedTexSubImage2D(...arguments)}catch(O){O("WebGLState:",O)}}function Ie(){try{i.compressedTexSubImage3D(...arguments)}catch(O){O("WebGLState:",O)}}function ye(){try{i.texStorage2D(...arguments)}catch(O){O("WebGLState:",O)}}function Ue(){try{i.texStorage3D(...arguments)}catch(O){O("WebGLState:",O)}}function k(){try{i.texImage2D(...arguments)}catch(O){O("WebGLState:",O)}}function te(){try{i.texImage3D(...arguments)}catch(O){O("WebGLState:",O)}}function _e(O){Fe.equals(O)===!1&&(i.scissor(O.x,O.y,O.z,O.w),Fe.copy(O))}function H(O){Oe.equals(O)===!1&&(i.viewport(O.x,O.y,O.z,O.w),Oe.copy(O))}function z(O,ve){let ge=c.get(ve);ge===void 0&&(ge=new WeakMap,c.set(ve,ge));let Se=ge.get(O);Se===void 0&&(Se=i.getUniformBlockIndex(ve,O.name),ge.set(O,Se))}function he(O,ve){const Se=c.get(ve).get(O);l.get(ve)!==Se&&(i.uniformBlockBinding(ve,Se,O.__bindingPointIndex),l.set(ve,Se))}function Me(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),o.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),u={},ee=null,ce={},f={},h=new WeakMap,d=[],x=null,p=!1,g=null,m=null,_=null,S=null,A=null,y=null,b=null,v=new rt(0,0,0),E=0,M=!1,T=null,I=null,P=null,B=null,N=null,Fe.set(0,0,i.canvas.width,i.canvas.height),Oe.set(0,0,i.canvas.width,i.canvas.height),r.reset(),o.reset(),a.reset()}return{buffers:{color:r,depth:o,stencil:a},enable:ne,disable:xe,bindFramebuffer:Be,drawBuffers:Te,useProgram:Ve,setBlending:Y,setMaterial:w,setFlipSided:oe,setCullFace:re,setLineWidth:pe,setPolygonOffset:se,setScissorTest:me,activeTexture:ie,bindTexture:Ae,unbindTexture:R,compressedTexImage2D:C,compressedTexImage3D:W,texImage2D:k,texImage3D:te,updateUBOMapping:z,uniformBlockBinding:he,texStorage2D:ye,texStorage3D:Ue,texSubImage2D:$,texSubImage3D:fe,compressedTexSubImage2D:Z,compressedTexSubImage3D:Ie,scissor:_e,viewport:H,reset:Me}}function lC(i,e,t,n,s,r,o){const a=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new Pe,u=new WeakMap;let f;const h=new WeakMap;let d=!1;try{d=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function x(R,C){return d?new OffscreenCanvas(R,C):Wl("canvas")}function p(R,C,W){let $=1;const fe=Ae(R);if((fe.width>W||fe.height>W)&&($=W/Math.max(fe.width,fe.height)),$<1)if(typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&R instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&R instanceof ImageBitmap||typeof VideoFrame<"u"&&R instanceof VideoFrame){const Z=Math.floor($*fe.width),Ie=Math.floor($*fe.height);f===void 0&&(f=x(Z,Ie));const ye=C?x(Z,Ie):f;return ye.width=Z,ye.height=Ie,ye.getContext("2d").drawImage(R,0,0,Z,Ie),Ze("WebGLRenderer: Texture has been resized from ("+fe.width+"x"+fe.height+") to ("+Z+"x"+Ie+")."),ye}else return"data"in R&&Ze("WebGLRenderer: Image in DataTexture is too big ("+fe.width+"x"+fe.height+")."),R;return R}function g(R){return R.generateMipmaps}function m(R){i.generateMipmap(R)}function _(R){return R.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:R.isWebGL3DRenderTarget?i.TEXTURE_3D:R.isWebGLArrayRenderTarget||R.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function S(R,C,W,$,fe=!1){if(R!==null){if(i[R]!==void 0)return i[R];Ze("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+R+"'")}let Z=C;if(C===i.RED&&(W===i.FLOAT&&(Z=i.R32F),W===i.HALF_FLOAT&&(Z=i.R16F),W===i.UNSIGNED_BYTE&&(Z=i.R8)),C===i.RED_INTEGER&&(W===i.UNSIGNED_BYTE&&(Z=i.R8UI),W===i.UNSIGNED_SHORT&&(Z=i.R16UI),W===i.UNSIGNED_INT&&(Z=i.R32UI),W===i.BYTE&&(Z=i.R8I),W===i.SHORT&&(Z=i.R16I),W===i.INT&&(Z=i.R32I)),C===i.RG&&(W===i.FLOAT&&(Z=i.RG32F),W===i.HALF_FLOAT&&(Z=i.RG16F),W===i.UNSIGNED_BYTE&&(Z=i.RG8)),C===i.RG_INTEGER&&(W===i.UNSIGNED_BYTE&&(Z=i.RG8UI),W===i.UNSIGNED_SHORT&&(Z=i.RG16UI),W===i.UNSIGNED_INT&&(Z=i.RG32UI),W===i.BYTE&&(Z=i.RG8I),W===i.SHORT&&(Z=i.RG16I),W===i.INT&&(Z=i.RG32I)),C===i.RGB_INTEGER&&(W===i.UNSIGNED_BYTE&&(Z=i.RGB8UI),W===i.UNSIGNED_SHORT&&(Z=i.RGB16UI),W===i.UNSIGNED_INT&&(Z=i.RGB32UI),W===i.BYTE&&(Z=i.RGB8I),W===i.SHORT&&(Z=i.RGB16I),W===i.INT&&(Z=i.RGB32I)),C===i.RGBA_INTEGER&&(W===i.UNSIGNED_BYTE&&(Z=i.RGBA8UI),W===i.UNSIGNED_SHORT&&(Z=i.RGBA16UI),W===i.UNSIGNED_INT&&(Z=i.RGBA32UI),W===i.BYTE&&(Z=i.RGBA8I),W===i.SHORT&&(Z=i.RGBA16I),W===i.INT&&(Z=i.RGBA32I)),C===i.RGB&&(W===i.UNSIGNED_INT_5_9_9_9_REV&&(Z=i.RGB9_E5),W===i.UNSIGNED_INT_10F_11F_11F_REV&&(Z=i.R11F_G11F_B10F)),C===i.RGBA){const Ie=fe?Vl:lt.getTransfer($);W===i.FLOAT&&(Z=i.RGBA32F),W===i.HALF_FLOAT&&(Z=i.RGBA16F),W===i.UNSIGNED_BYTE&&(Z=Ie===mt?i.SRGB8_ALPHA8:i.RGBA8),W===i.UNSIGNED_SHORT_4_4_4_4&&(Z=i.RGBA4),W===i.UNSIGNED_SHORT_5_5_5_1&&(Z=i.RGB5_A1)}return(Z===i.R16F||Z===i.R32F||Z===i.RG16F||Z===i.RG32F||Z===i.RGBA16F||Z===i.RGBA32F)&&e.get("EXT_color_buffer_float"),Z}function A(R,C){let W;return R?C===null||C===pi||C===Sa?W=i.DEPTH24_STENCIL8:C===Mi?W=i.DEPTH32F_STENCIL8:C===va&&(W=i.DEPTH24_STENCIL8,Ze("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):C===null||C===pi||C===Sa?W=i.DEPTH_COMPONENT24:C===Mi?W=i.DEPTH_COMPONENT32F:C===va&&(W=i.DEPTH_COMPONENT16),W}function y(R,C){return g(R)===!0||R.isFramebufferTexture&&R.minFilter!==Jn&&R.minFilter!==di?Math.log2(Math.max(C.width,C.height))+1:R.mipmaps!==void 0&&R.mipmaps.length>0?R.mipmaps.length:R.isCompressedTexture&&Array.isArray(R.image)?C.mipmaps.length:1}function b(R){const C=R.target;C.removeEventListener("dispose",b),E(C),C.isVideoTexture&&u.delete(C)}function v(R){const C=R.target;C.removeEventListener("dispose",v),T(C)}function E(R){const C=n.get(R);if(C.__webglInit===void 0)return;const W=R.source,$=h.get(W);if($){const fe=$[C.__cacheKey];fe.usedTimes--,fe.usedTimes===0&&M(R),Object.keys($).length===0&&h.delete(W)}n.remove(R)}function M(R){const C=n.get(R);i.deleteTexture(C.__webglTexture);const W=R.source,$=h.get(W);delete $[C.__cacheKey],o.memory.textures--}function T(R){const C=n.get(R);if(R.depthTexture&&(R.depthTexture.dispose(),n.remove(R.depthTexture)),R.isWebGLCubeRenderTarget)for(let $=0;$<6;$++){if(Array.isArray(C.__webglFramebuffer[$]))for(let fe=0;fe<C.__webglFramebuffer[$].length;fe++)i.deleteFramebuffer(C.__webglFramebuffer[$][fe]);else i.deleteFramebuffer(C.__webglFramebuffer[$]);C.__webglDepthbuffer&&i.deleteRenderbuffer(C.__webglDepthbuffer[$])}else{if(Array.isArray(C.__webglFramebuffer))for(let $=0;$<C.__webglFramebuffer.length;$++)i.deleteFramebuffer(C.__webglFramebuffer[$]);else i.deleteFramebuffer(C.__webglFramebuffer);if(C.__webglDepthbuffer&&i.deleteRenderbuffer(C.__webglDepthbuffer),C.__webglMultisampledFramebuffer&&i.deleteFramebuffer(C.__webglMultisampledFramebuffer),C.__webglColorRenderbuffer)for(let $=0;$<C.__webglColorRenderbuffer.length;$++)C.__webglColorRenderbuffer[$]&&i.deleteRenderbuffer(C.__webglColorRenderbuffer[$]);C.__webglDepthRenderbuffer&&i.deleteRenderbuffer(C.__webglDepthRenderbuffer)}const W=R.textures;for(let $=0,fe=W.length;$<fe;$++){const Z=n.get(W[$]);Z.__webglTexture&&(i.deleteTexture(Z.__webglTexture),o.memory.textures--),n.remove(W[$])}n.remove(R)}let I=0;function P(){I=0}function B(){const R=I;return R>=s.maxTextures&&Ze("WebGLTextures: Trying to use "+R+" texture units while this GPU supports only "+s.maxTextures),I+=1,R}function N(R){const C=[];return C.push(R.wrapS),C.push(R.wrapT),C.push(R.wrapR||0),C.push(R.magFilter),C.push(R.minFilter),C.push(R.anisotropy),C.push(R.internalFormat),C.push(R.format),C.push(R.type),C.push(R.generateMipmaps),C.push(R.premultiplyAlpha),C.push(R.flipY),C.push(R.unpackAlignment),C.push(R.colorSpace),C.join()}function G(R,C){const W=n.get(R);if(R.isVideoTexture&&me(R),R.isRenderTargetTexture===!1&&R.isExternalTexture!==!0&&R.version>0&&W.__version!==R.version){const $=R.image;if($===null)Ze("WebGLRenderer: Texture marked for update but no image data found.");else if($.complete===!1)Ze("WebGLRenderer: Texture marked for update but image is incomplete");else{J(W,R,C);return}}else R.isExternalTexture&&(W.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(i.TEXTURE_2D,W.__webglTexture,i.TEXTURE0+C)}function V(R,C){const W=n.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&W.__version!==R.version){J(W,R,C);return}else R.isExternalTexture&&(W.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(i.TEXTURE_2D_ARRAY,W.__webglTexture,i.TEXTURE0+C)}function q(R,C){const W=n.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&W.__version!==R.version){J(W,R,C);return}t.bindTexture(i.TEXTURE_3D,W.__webglTexture,i.TEXTURE0+C)}function X(R,C){const W=n.get(R);if(R.version>0&&W.__version!==R.version){ne(W,R,C);return}t.bindTexture(i.TEXTURE_CUBE_MAP,W.__webglTexture,i.TEXTURE0+C)}const ee={[ef]:i.REPEAT,[ds]:i.CLAMP_TO_EDGE,[tf]:i.MIRRORED_REPEAT},ce={[Jn]:i.NEAREST,[PS]:i.NEAREST_MIPMAP_NEAREST,[Xa]:i.NEAREST_MIPMAP_LINEAR,[di]:i.LINEAR,[zc]:i.LINEAR_MIPMAP_NEAREST,[mr]:i.LINEAR_MIPMAP_LINEAR},be={[OS]:i.NEVER,[GS]:i.ALWAYS,[NS]:i.LESS,[eg]:i.LEQUAL,[zS]:i.EQUAL,[VS]:i.GEQUAL,[kS]:i.GREATER,[HS]:i.NOTEQUAL};function Re(R,C){if(C.type===Mi&&e.has("OES_texture_float_linear")===!1&&(C.magFilter===di||C.magFilter===zc||C.magFilter===Xa||C.magFilter===mr||C.minFilter===di||C.minFilter===zc||C.minFilter===Xa||C.minFilter===mr)&&Ze("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(R,i.TEXTURE_WRAP_S,ee[C.wrapS]),i.texParameteri(R,i.TEXTURE_WRAP_T,ee[C.wrapT]),(R===i.TEXTURE_3D||R===i.TEXTURE_2D_ARRAY)&&i.texParameteri(R,i.TEXTURE_WRAP_R,ee[C.wrapR]),i.texParameteri(R,i.TEXTURE_MAG_FILTER,ce[C.magFilter]),i.texParameteri(R,i.TEXTURE_MIN_FILTER,ce[C.minFilter]),C.compareFunction&&(i.texParameteri(R,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(R,i.TEXTURE_COMPARE_FUNC,be[C.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(C.magFilter===Jn||C.minFilter!==Xa&&C.minFilter!==mr||C.type===Mi&&e.has("OES_texture_float_linear")===!1)return;if(C.anisotropy>1||n.get(C).__currentAnisotropy){const W=e.get("EXT_texture_filter_anisotropic");i.texParameterf(R,W.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(C.anisotropy,s.getMaxAnisotropy())),n.get(C).__currentAnisotropy=C.anisotropy}}}function Fe(R,C){let W=!1;R.__webglInit===void 0&&(R.__webglInit=!0,C.addEventListener("dispose",b));const $=C.source;let fe=h.get($);fe===void 0&&(fe={},h.set($,fe));const Z=N(C);if(Z!==R.__cacheKey){fe[Z]===void 0&&(fe[Z]={texture:i.createTexture(),usedTimes:0},o.memory.textures++,W=!0),fe[Z].usedTimes++;const Ie=fe[R.__cacheKey];Ie!==void 0&&(fe[R.__cacheKey].usedTimes--,Ie.usedTimes===0&&M(C)),R.__cacheKey=Z,R.__webglTexture=fe[Z].texture}return W}function Oe(R,C,W){return Math.floor(Math.floor(R/W)/C)}function Ne(R,C,W,$){const Z=R.updateRanges;if(Z.length===0)t.texSubImage2D(i.TEXTURE_2D,0,0,0,C.width,C.height,W,$,C.data);else{Z.sort((te,_e)=>te.start-_e.start);let Ie=0;for(let te=1;te<Z.length;te++){const _e=Z[Ie],H=Z[te],z=_e.start+_e.count,he=Oe(H.start,C.width,4),Me=Oe(_e.start,C.width,4);H.start<=z+1&&he===Me&&Oe(H.start+H.count-1,C.width,4)===he?_e.count=Math.max(_e.count,H.start+H.count-_e.start):(++Ie,Z[Ie]=H)}Z.length=Ie+1;const ye=i.getParameter(i.UNPACK_ROW_LENGTH),Ue=i.getParameter(i.UNPACK_SKIP_PIXELS),k=i.getParameter(i.UNPACK_SKIP_ROWS);i.pixelStorei(i.UNPACK_ROW_LENGTH,C.width);for(let te=0,_e=Z.length;te<_e;te++){const H=Z[te],z=Math.floor(H.start/4),he=Math.ceil(H.count/4),Me=z%C.width,O=Math.floor(z/C.width),ve=he,ge=1;i.pixelStorei(i.UNPACK_SKIP_PIXELS,Me),i.pixelStorei(i.UNPACK_SKIP_ROWS,O),t.texSubImage2D(i.TEXTURE_2D,0,Me,O,ve,ge,W,$,C.data)}R.clearUpdateRanges(),i.pixelStorei(i.UNPACK_ROW_LENGTH,ye),i.pixelStorei(i.UNPACK_SKIP_PIXELS,Ue),i.pixelStorei(i.UNPACK_SKIP_ROWS,k)}}function J(R,C,W){let $=i.TEXTURE_2D;(C.isDataArrayTexture||C.isCompressedArrayTexture)&&($=i.TEXTURE_2D_ARRAY),C.isData3DTexture&&($=i.TEXTURE_3D);const fe=Fe(R,C),Z=C.source;t.bindTexture($,R.__webglTexture,i.TEXTURE0+W);const Ie=n.get(Z);if(Z.version!==Ie.__version||fe===!0){t.activeTexture(i.TEXTURE0+W);const ye=lt.getPrimaries(lt.workingColorSpace),Ue=C.colorSpace===Ds?null:lt.getPrimaries(C.colorSpace),k=C.colorSpace===Ds||ye===Ue?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,C.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,C.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,C.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,k);let te=p(C.image,!1,s.maxTextureSize);te=ie(C,te);const _e=r.convert(C.format,C.colorSpace),H=r.convert(C.type);let z=S(C.internalFormat,_e,H,C.colorSpace,C.isVideoTexture);Re($,C);let he;const Me=C.mipmaps,O=C.isVideoTexture!==!0,ve=Ie.__version===void 0||fe===!0,ge=Z.dataReady,Se=y(C,te);if(C.isDepthTexture)z=A(C.format===Aa,C.type),ve&&(O?t.texStorage2D(i.TEXTURE_2D,1,z,te.width,te.height):t.texImage2D(i.TEXTURE_2D,0,z,te.width,te.height,0,_e,H,null));else if(C.isDataTexture)if(Me.length>0){O&&ve&&t.texStorage2D(i.TEXTURE_2D,Se,z,Me[0].width,Me[0].height);for(let de=0,le=Me.length;de<le;de++)he=Me[de],O?ge&&t.texSubImage2D(i.TEXTURE_2D,de,0,0,he.width,he.height,_e,H,he.data):t.texImage2D(i.TEXTURE_2D,de,z,he.width,he.height,0,_e,H,he.data);C.generateMipmaps=!1}else O?(ve&&t.texStorage2D(i.TEXTURE_2D,Se,z,te.width,te.height),ge&&Ne(C,te,_e,H)):t.texImage2D(i.TEXTURE_2D,0,z,te.width,te.height,0,_e,H,te.data);else if(C.isCompressedTexture)if(C.isCompressedArrayTexture){O&&ve&&t.texStorage3D(i.TEXTURE_2D_ARRAY,Se,z,Me[0].width,Me[0].height,te.depth);for(let de=0,le=Me.length;de<le;de++)if(he=Me[de],C.format!==Mn)if(_e!==null)if(O){if(ge)if(C.layerUpdates.size>0){const Ce=mp(he.width,he.height,C.format,C.type);for(const ze of C.layerUpdates){const it=he.data.subarray(ze*Ce/he.data.BYTES_PER_ELEMENT,(ze+1)*Ce/he.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,de,0,0,ze,he.width,he.height,1,_e,it)}C.clearLayerUpdates()}else t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,de,0,0,0,he.width,he.height,te.depth,_e,he.data)}else t.compressedTexImage3D(i.TEXTURE_2D_ARRAY,de,z,he.width,he.height,te.depth,0,he.data,0,0);else Ze("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else O?ge&&t.texSubImage3D(i.TEXTURE_2D_ARRAY,de,0,0,0,he.width,he.height,te.depth,_e,H,he.data):t.texImage3D(i.TEXTURE_2D_ARRAY,de,z,he.width,he.height,te.depth,0,_e,H,he.data)}else{O&&ve&&t.texStorage2D(i.TEXTURE_2D,Se,z,Me[0].width,Me[0].height);for(let de=0,le=Me.length;de<le;de++)he=Me[de],C.format!==Mn?_e!==null?O?ge&&t.compressedTexSubImage2D(i.TEXTURE_2D,de,0,0,he.width,he.height,_e,he.data):t.compressedTexImage2D(i.TEXTURE_2D,de,z,he.width,he.height,0,he.data):Ze("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):O?ge&&t.texSubImage2D(i.TEXTURE_2D,de,0,0,he.width,he.height,_e,H,he.data):t.texImage2D(i.TEXTURE_2D,de,z,he.width,he.height,0,_e,H,he.data)}else if(C.isDataArrayTexture)if(O){if(ve&&t.texStorage3D(i.TEXTURE_2D_ARRAY,Se,z,te.width,te.height,te.depth),ge)if(C.layerUpdates.size>0){const de=mp(te.width,te.height,C.format,C.type);for(const le of C.layerUpdates){const Ce=te.data.subarray(le*de/te.data.BYTES_PER_ELEMENT,(le+1)*de/te.data.BYTES_PER_ELEMENT);t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,le,te.width,te.height,1,_e,H,Ce)}C.clearLayerUpdates()}else t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,te.width,te.height,te.depth,_e,H,te.data)}else t.texImage3D(i.TEXTURE_2D_ARRAY,0,z,te.width,te.height,te.depth,0,_e,H,te.data);else if(C.isData3DTexture)O?(ve&&t.texStorage3D(i.TEXTURE_3D,Se,z,te.width,te.height,te.depth),ge&&t.texSubImage3D(i.TEXTURE_3D,0,0,0,0,te.width,te.height,te.depth,_e,H,te.data)):t.texImage3D(i.TEXTURE_3D,0,z,te.width,te.height,te.depth,0,_e,H,te.data);else if(C.isFramebufferTexture){if(ve)if(O)t.texStorage2D(i.TEXTURE_2D,Se,z,te.width,te.height);else{let de=te.width,le=te.height;for(let Ce=0;Ce<Se;Ce++)t.texImage2D(i.TEXTURE_2D,Ce,z,de,le,0,_e,H,null),de>>=1,le>>=1}}else if(Me.length>0){if(O&&ve){const de=Ae(Me[0]);t.texStorage2D(i.TEXTURE_2D,Se,z,de.width,de.height)}for(let de=0,le=Me.length;de<le;de++)he=Me[de],O?ge&&t.texSubImage2D(i.TEXTURE_2D,de,0,0,_e,H,he):t.texImage2D(i.TEXTURE_2D,de,z,_e,H,he);C.generateMipmaps=!1}else if(O){if(ve){const de=Ae(te);t.texStorage2D(i.TEXTURE_2D,Se,z,de.width,de.height)}ge&&t.texSubImage2D(i.TEXTURE_2D,0,0,0,_e,H,te)}else t.texImage2D(i.TEXTURE_2D,0,z,_e,H,te);g(C)&&m($),Ie.__version=Z.version,C.onUpdate&&C.onUpdate(C)}R.__version=C.version}function ne(R,C,W){if(C.image.length!==6)return;const $=Fe(R,C),fe=C.source;t.bindTexture(i.TEXTURE_CUBE_MAP,R.__webglTexture,i.TEXTURE0+W);const Z=n.get(fe);if(fe.version!==Z.__version||$===!0){t.activeTexture(i.TEXTURE0+W);const Ie=lt.getPrimaries(lt.workingColorSpace),ye=C.colorSpace===Ds?null:lt.getPrimaries(C.colorSpace),Ue=C.colorSpace===Ds||Ie===ye?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,C.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,C.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,C.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,Ue);const k=C.isCompressedTexture||C.image[0].isCompressedTexture,te=C.image[0]&&C.image[0].isDataTexture,_e=[];for(let le=0;le<6;le++)!k&&!te?_e[le]=p(C.image[le],!0,s.maxCubemapSize):_e[le]=te?C.image[le].image:C.image[le],_e[le]=ie(C,_e[le]);const H=_e[0],z=r.convert(C.format,C.colorSpace),he=r.convert(C.type),Me=S(C.internalFormat,z,he,C.colorSpace),O=C.isVideoTexture!==!0,ve=Z.__version===void 0||$===!0,ge=fe.dataReady;let Se=y(C,H);Re(i.TEXTURE_CUBE_MAP,C);let de;if(k){O&&ve&&t.texStorage2D(i.TEXTURE_CUBE_MAP,Se,Me,H.width,H.height);for(let le=0;le<6;le++){de=_e[le].mipmaps;for(let Ce=0;Ce<de.length;Ce++){const ze=de[Ce];C.format!==Mn?z!==null?O?ge&&t.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,Ce,0,0,ze.width,ze.height,z,ze.data):t.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,Ce,Me,ze.width,ze.height,0,ze.data):Ze("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):O?ge&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,Ce,0,0,ze.width,ze.height,z,he,ze.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,Ce,Me,ze.width,ze.height,0,z,he,ze.data)}}}else{if(de=C.mipmaps,O&&ve){de.length>0&&Se++;const le=Ae(_e[0]);t.texStorage2D(i.TEXTURE_CUBE_MAP,Se,Me,le.width,le.height)}for(let le=0;le<6;le++)if(te){O?ge&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,0,0,0,_e[le].width,_e[le].height,z,he,_e[le].data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,0,Me,_e[le].width,_e[le].height,0,z,he,_e[le].data);for(let Ce=0;Ce<de.length;Ce++){const it=de[Ce].image[le].image;O?ge&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,Ce+1,0,0,it.width,it.height,z,he,it.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,Ce+1,Me,it.width,it.height,0,z,he,it.data)}}else{O?ge&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,0,0,0,z,he,_e[le]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,0,Me,z,he,_e[le]);for(let Ce=0;Ce<de.length;Ce++){const ze=de[Ce];O?ge&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,Ce+1,0,0,z,he,ze.image[le]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,Ce+1,Me,z,he,ze.image[le])}}}g(C)&&m(i.TEXTURE_CUBE_MAP),Z.__version=fe.version,C.onUpdate&&C.onUpdate(C)}R.__version=C.version}function xe(R,C,W,$,fe,Z){const Ie=r.convert(W.format,W.colorSpace),ye=r.convert(W.type),Ue=S(W.internalFormat,Ie,ye,W.colorSpace),k=n.get(C),te=n.get(W);if(te.__renderTarget=C,!k.__hasExternalTextures){const _e=Math.max(1,C.width>>Z),H=Math.max(1,C.height>>Z);fe===i.TEXTURE_3D||fe===i.TEXTURE_2D_ARRAY?t.texImage3D(fe,Z,Ue,_e,H,C.depth,0,Ie,ye,null):t.texImage2D(fe,Z,Ue,_e,H,0,Ie,ye,null)}t.bindFramebuffer(i.FRAMEBUFFER,R),se(C)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,$,fe,te.__webglTexture,0,pe(C)):(fe===i.TEXTURE_2D||fe>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&fe<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,$,fe,te.__webglTexture,Z),t.bindFramebuffer(i.FRAMEBUFFER,null)}function Be(R,C,W){if(i.bindRenderbuffer(i.RENDERBUFFER,R),C.depthBuffer){const $=C.depthTexture,fe=$&&$.isDepthTexture?$.type:null,Z=A(C.stencilBuffer,fe),Ie=C.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,ye=pe(C);se(C)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,ye,Z,C.width,C.height):W?i.renderbufferStorageMultisample(i.RENDERBUFFER,ye,Z,C.width,C.height):i.renderbufferStorage(i.RENDERBUFFER,Z,C.width,C.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,Ie,i.RENDERBUFFER,R)}else{const $=C.textures;for(let fe=0;fe<$.length;fe++){const Z=$[fe],Ie=r.convert(Z.format,Z.colorSpace),ye=r.convert(Z.type),Ue=S(Z.internalFormat,Ie,ye,Z.colorSpace),k=pe(C);W&&se(C)===!1?i.renderbufferStorageMultisample(i.RENDERBUFFER,k,Ue,C.width,C.height):se(C)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,k,Ue,C.width,C.height):i.renderbufferStorage(i.RENDERBUFFER,Ue,C.width,C.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function Te(R,C){if(C&&C.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(t.bindFramebuffer(i.FRAMEBUFFER,R),!(C.depthTexture&&C.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const $=n.get(C.depthTexture);$.__renderTarget=C,(!$.__webglTexture||C.depthTexture.image.width!==C.width||C.depthTexture.image.height!==C.height)&&(C.depthTexture.image.width=C.width,C.depthTexture.image.height=C.height,C.depthTexture.needsUpdate=!0),G(C.depthTexture,0);const fe=$.__webglTexture,Z=pe(C);if(C.depthTexture.format===yo)se(C)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,fe,0,Z):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,fe,0);else if(C.depthTexture.format===Aa)se(C)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,fe,0,Z):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,fe,0);else throw new Error("Unknown depthTexture format")}function Ve(R){const C=n.get(R),W=R.isWebGLCubeRenderTarget===!0;if(C.__boundDepthTexture!==R.depthTexture){const $=R.depthTexture;if(C.__depthDisposeCallback&&C.__depthDisposeCallback(),$){const fe=()=>{delete C.__boundDepthTexture,delete C.__depthDisposeCallback,$.removeEventListener("dispose",fe)};$.addEventListener("dispose",fe),C.__depthDisposeCallback=fe}C.__boundDepthTexture=$}if(R.depthTexture&&!C.__autoAllocateDepthBuffer){if(W)throw new Error("target.depthTexture not supported in Cube render targets");const $=R.texture.mipmaps;$&&$.length>0?Te(C.__webglFramebuffer[0],R):Te(C.__webglFramebuffer,R)}else if(W){C.__webglDepthbuffer=[];for(let $=0;$<6;$++)if(t.bindFramebuffer(i.FRAMEBUFFER,C.__webglFramebuffer[$]),C.__webglDepthbuffer[$]===void 0)C.__webglDepthbuffer[$]=i.createRenderbuffer(),Be(C.__webglDepthbuffer[$],R,!1);else{const fe=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Z=C.__webglDepthbuffer[$];i.bindRenderbuffer(i.RENDERBUFFER,Z),i.framebufferRenderbuffer(i.FRAMEBUFFER,fe,i.RENDERBUFFER,Z)}}else{const $=R.texture.mipmaps;if($&&$.length>0?t.bindFramebuffer(i.FRAMEBUFFER,C.__webglFramebuffer[0]):t.bindFramebuffer(i.FRAMEBUFFER,C.__webglFramebuffer),C.__webglDepthbuffer===void 0)C.__webglDepthbuffer=i.createRenderbuffer(),Be(C.__webglDepthbuffer,R,!1);else{const fe=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Z=C.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,Z),i.framebufferRenderbuffer(i.FRAMEBUFFER,fe,i.RENDERBUFFER,Z)}}t.bindFramebuffer(i.FRAMEBUFFER,null)}function L(R,C,W){const $=n.get(R);C!==void 0&&xe($.__webglFramebuffer,R,R.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),W!==void 0&&Ve(R)}function U(R){const C=R.texture,W=n.get(R),$=n.get(C);R.addEventListener("dispose",v);const fe=R.textures,Z=R.isWebGLCubeRenderTarget===!0,Ie=fe.length>1;if(Ie||($.__webglTexture===void 0&&($.__webglTexture=i.createTexture()),$.__version=C.version,o.memory.textures++),Z){W.__webglFramebuffer=[];for(let ye=0;ye<6;ye++)if(C.mipmaps&&C.mipmaps.length>0){W.__webglFramebuffer[ye]=[];for(let Ue=0;Ue<C.mipmaps.length;Ue++)W.__webglFramebuffer[ye][Ue]=i.createFramebuffer()}else W.__webglFramebuffer[ye]=i.createFramebuffer()}else{if(C.mipmaps&&C.mipmaps.length>0){W.__webglFramebuffer=[];for(let ye=0;ye<C.mipmaps.length;ye++)W.__webglFramebuffer[ye]=i.createFramebuffer()}else W.__webglFramebuffer=i.createFramebuffer();if(Ie)for(let ye=0,Ue=fe.length;ye<Ue;ye++){const k=n.get(fe[ye]);k.__webglTexture===void 0&&(k.__webglTexture=i.createTexture(),o.memory.textures++)}if(R.samples>0&&se(R)===!1){W.__webglMultisampledFramebuffer=i.createFramebuffer(),W.__webglColorRenderbuffer=[],t.bindFramebuffer(i.FRAMEBUFFER,W.__webglMultisampledFramebuffer);for(let ye=0;ye<fe.length;ye++){const Ue=fe[ye];W.__webglColorRenderbuffer[ye]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,W.__webglColorRenderbuffer[ye]);const k=r.convert(Ue.format,Ue.colorSpace),te=r.convert(Ue.type),_e=S(Ue.internalFormat,k,te,Ue.colorSpace,R.isXRRenderTarget===!0),H=pe(R);i.renderbufferStorageMultisample(i.RENDERBUFFER,H,_e,R.width,R.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+ye,i.RENDERBUFFER,W.__webglColorRenderbuffer[ye])}i.bindRenderbuffer(i.RENDERBUFFER,null),R.depthBuffer&&(W.__webglDepthRenderbuffer=i.createRenderbuffer(),Be(W.__webglDepthRenderbuffer,R,!0)),t.bindFramebuffer(i.FRAMEBUFFER,null)}}if(Z){t.bindTexture(i.TEXTURE_CUBE_MAP,$.__webglTexture),Re(i.TEXTURE_CUBE_MAP,C);for(let ye=0;ye<6;ye++)if(C.mipmaps&&C.mipmaps.length>0)for(let Ue=0;Ue<C.mipmaps.length;Ue++)xe(W.__webglFramebuffer[ye][Ue],R,C,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+ye,Ue);else xe(W.__webglFramebuffer[ye],R,C,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+ye,0);g(C)&&m(i.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(Ie){for(let ye=0,Ue=fe.length;ye<Ue;ye++){const k=fe[ye],te=n.get(k);let _e=i.TEXTURE_2D;(R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(_e=R.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(_e,te.__webglTexture),Re(_e,k),xe(W.__webglFramebuffer,R,k,i.COLOR_ATTACHMENT0+ye,_e,0),g(k)&&m(_e)}t.unbindTexture()}else{let ye=i.TEXTURE_2D;if((R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(ye=R.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(ye,$.__webglTexture),Re(ye,C),C.mipmaps&&C.mipmaps.length>0)for(let Ue=0;Ue<C.mipmaps.length;Ue++)xe(W.__webglFramebuffer[Ue],R,C,i.COLOR_ATTACHMENT0,ye,Ue);else xe(W.__webglFramebuffer,R,C,i.COLOR_ATTACHMENT0,ye,0);g(C)&&m(ye),t.unbindTexture()}R.depthBuffer&&Ve(R)}function Y(R){const C=R.textures;for(let W=0,$=C.length;W<$;W++){const fe=C[W];if(g(fe)){const Z=_(R),Ie=n.get(fe).__webglTexture;t.bindTexture(Z,Ie),m(Z),t.unbindTexture()}}}const w=[],oe=[];function re(R){if(R.samples>0){if(se(R)===!1){const C=R.textures,W=R.width,$=R.height;let fe=i.COLOR_BUFFER_BIT;const Z=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Ie=n.get(R),ye=C.length>1;if(ye)for(let k=0;k<C.length;k++)t.bindFramebuffer(i.FRAMEBUFFER,Ie.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.RENDERBUFFER,null),t.bindFramebuffer(i.FRAMEBUFFER,Ie.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.TEXTURE_2D,null,0);t.bindFramebuffer(i.READ_FRAMEBUFFER,Ie.__webglMultisampledFramebuffer);const Ue=R.texture.mipmaps;Ue&&Ue.length>0?t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Ie.__webglFramebuffer[0]):t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Ie.__webglFramebuffer);for(let k=0;k<C.length;k++){if(R.resolveDepthBuffer&&(R.depthBuffer&&(fe|=i.DEPTH_BUFFER_BIT),R.stencilBuffer&&R.resolveStencilBuffer&&(fe|=i.STENCIL_BUFFER_BIT)),ye){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,Ie.__webglColorRenderbuffer[k]);const te=n.get(C[k]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,te,0)}i.blitFramebuffer(0,0,W,$,0,0,W,$,fe,i.NEAREST),l===!0&&(w.length=0,oe.length=0,w.push(i.COLOR_ATTACHMENT0+k),R.depthBuffer&&R.resolveDepthBuffer===!1&&(w.push(Z),oe.push(Z),i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,oe)),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,w))}if(t.bindFramebuffer(i.READ_FRAMEBUFFER,null),t.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),ye)for(let k=0;k<C.length;k++){t.bindFramebuffer(i.FRAMEBUFFER,Ie.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.RENDERBUFFER,Ie.__webglColorRenderbuffer[k]);const te=n.get(C[k]).__webglTexture;t.bindFramebuffer(i.FRAMEBUFFER,Ie.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+k,i.TEXTURE_2D,te,0)}t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Ie.__webglMultisampledFramebuffer)}else if(R.depthBuffer&&R.resolveDepthBuffer===!1&&l){const C=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[C])}}}function pe(R){return Math.min(s.maxSamples,R.samples)}function se(R){const C=n.get(R);return R.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&C.__useRenderToTexture!==!1}function me(R){const C=o.render.frame;u.get(R)!==C&&(u.set(R,C),R.update())}function ie(R,C){const W=R.colorSpace,$=R.format,fe=R.type;return R.isCompressedTexture===!0||R.isVideoTexture===!0||W!==bo&&W!==Ds&&(lt.getTransfer(W)===mt?($!==Mn||fe!==qi)&&Ze("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):Wt("WebGLTextures: Unsupported texture color space:",W)),C}function Ae(R){return typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement?(c.width=R.naturalWidth||R.width,c.height=R.naturalHeight||R.height):typeof VideoFrame<"u"&&R instanceof VideoFrame?(c.width=R.displayWidth,c.height=R.displayHeight):(c.width=R.width,c.height=R.height),c}this.allocateTextureUnit=B,this.resetTextureUnits=P,this.setTexture2D=G,this.setTexture2DArray=V,this.setTexture3D=q,this.setTextureCube=X,this.rebindTextures=L,this.setupRenderTarget=U,this.updateRenderTargetMipmap=Y,this.updateMultisampleRenderTarget=re,this.setupDepthRenderbuffer=Ve,this.setupFrameBufferTexture=xe,this.useMultisampledRTT=se}function xg(i,e){function t(n,s=Ds){let r;const o=lt.getTransfer(s);if(n===qi)return i.UNSIGNED_BYTE;if(n===_h)return i.UNSIGNED_SHORT_4_4_4_4;if(n===vh)return i.UNSIGNED_SHORT_5_5_5_1;if(n===K0)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===j0)return i.UNSIGNED_INT_10F_11F_11F_REV;if(n===Y0)return i.BYTE;if(n===Q0)return i.SHORT;if(n===va)return i.UNSIGNED_SHORT;if(n===xh)return i.INT;if(n===pi)return i.UNSIGNED_INT;if(n===Mi)return i.FLOAT;if(n===Tr)return i.HALF_FLOAT;if(n===$0)return i.ALPHA;if(n===Z0)return i.RGB;if(n===Mn)return i.RGBA;if(n===yo)return i.DEPTH_COMPONENT;if(n===Aa)return i.DEPTH_STENCIL;if(n===J0)return i.RED;if(n===dc)return i.RED_INTEGER;if(n===Sh)return i.RG;if(n===Ah)return i.RG_INTEGER;if(n===lo)return i.RGBA_INTEGER;if(n===El||n===wl||n===Rl||n===Il)if(o===mt)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(n===El)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===wl)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===Rl)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===Il)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(n===El)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===wl)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===Rl)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===Il)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===nf||n===sf||n===rf||n===of)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(n===nf)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===sf)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===rf)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===of)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===af||n===lf||n===cf)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(n===af||n===lf)return o===mt?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(n===cf)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(n===uf||n===ff||n===hf||n===df||n===pf||n===mf||n===gf||n===xf||n===_f||n===vf||n===Sf||n===Af||n===yf||n===bf)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(n===uf)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===ff)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===hf)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===df)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===pf)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===mf)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===gf)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===xf)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===_f)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===vf)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===Sf)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===Af)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===yf)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===bf)return o===mt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===Mf||n===Tf||n===Cf)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(n===Mf)return o===mt?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===Tf)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===Cf)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===Ef||n===wf||n===Rf||n===If)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(n===Ef)return r.COMPRESSED_RED_RGTC1_EXT;if(n===wf)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===Rf)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===If)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===Sa?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:t}}const cC=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,uC=`
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

}`;class fC{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){const n=new fg(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=n}}getMesh(e){if(this.texture!==null&&this.mesh===null){const t=e.cameras[0].viewport,n=new Cn({vertexShader:cC,fragmentShader:uC,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new Yt(new To(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class hC extends Ks{constructor(e,t){super();const n=this;let s=null,r=1,o=null,a="local-floor",l=1,c=null,u=null,f=null,h=null,d=null,x=null;const p=typeof XRWebGLBinding<"u",g=new fC,m={},_=t.getContextAttributes();let S=null,A=null;const y=[],b=[],v=new Pe;let E=null;const M=new ui;M.viewport=new Dt;const T=new ui;T.viewport=new Dt;const I=[M,T],P=new DA;let B=null,N=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(J){let ne=y[J];return ne===void 0&&(ne=new au,y[J]=ne),ne.getTargetRaySpace()},this.getControllerGrip=function(J){let ne=y[J];return ne===void 0&&(ne=new au,y[J]=ne),ne.getGripSpace()},this.getHand=function(J){let ne=y[J];return ne===void 0&&(ne=new au,y[J]=ne),ne.getHandSpace()};function G(J){const ne=b.indexOf(J.inputSource);if(ne===-1)return;const xe=y[ne];xe!==void 0&&(xe.update(J.inputSource,J.frame,c||o),xe.dispatchEvent({type:J.type,data:J.inputSource}))}function V(){s.removeEventListener("select",G),s.removeEventListener("selectstart",G),s.removeEventListener("selectend",G),s.removeEventListener("squeeze",G),s.removeEventListener("squeezestart",G),s.removeEventListener("squeezeend",G),s.removeEventListener("end",V),s.removeEventListener("inputsourceschange",q);for(let J=0;J<y.length;J++){const ne=b[J];ne!==null&&(b[J]=null,y[J].disconnect(ne))}B=null,N=null,g.reset();for(const J in m)delete m[J];e.setRenderTarget(S),d=null,h=null,f=null,s=null,A=null,Ne.stop(),n.isPresenting=!1,e.setPixelRatio(E),e.setSize(v.width,v.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(J){r=J,n.isPresenting===!0&&Ze("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(J){a=J,n.isPresenting===!0&&Ze("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return c||o},this.setReferenceSpace=function(J){c=J},this.getBaseLayer=function(){return h!==null?h:d},this.getBinding=function(){return f===null&&p&&(f=new XRWebGLBinding(s,t)),f},this.getFrame=function(){return x},this.getSession=function(){return s},this.setSession=async function(J){if(s=J,s!==null){if(S=e.getRenderTarget(),s.addEventListener("select",G),s.addEventListener("selectstart",G),s.addEventListener("selectend",G),s.addEventListener("squeeze",G),s.addEventListener("squeezestart",G),s.addEventListener("squeezeend",G),s.addEventListener("end",V),s.addEventListener("inputsourceschange",q),_.xrCompatible!==!0&&await t.makeXRCompatible(),E=e.getPixelRatio(),e.getSize(v),p&&"createProjectionLayer"in XRWebGLBinding.prototype){let xe=null,Be=null,Te=null;_.depth&&(Te=_.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,xe=_.stencil?Aa:yo,Be=_.stencil?Sa:pi);const Ve={colorFormat:t.RGBA8,depthFormat:Te,scaleFactor:r};f=this.getBinding(),h=f.createProjectionLayer(Ve),s.updateRenderState({layers:[h]}),e.setPixelRatio(1),e.setSize(h.textureWidth,h.textureHeight,!1),A=new Ws(h.textureWidth,h.textureHeight,{format:Mn,type:qi,depthTexture:new Mh(h.textureWidth,h.textureHeight,Be,void 0,void 0,void 0,void 0,void 0,void 0,xe),stencilBuffer:_.stencil,colorSpace:e.outputColorSpace,samples:_.antialias?4:0,resolveDepthBuffer:h.ignoreDepthValues===!1,resolveStencilBuffer:h.ignoreDepthValues===!1})}else{const xe={antialias:_.antialias,alpha:!0,depth:_.depth,stencil:_.stencil,framebufferScaleFactor:r};d=new XRWebGLLayer(s,t,xe),s.updateRenderState({baseLayer:d}),e.setPixelRatio(1),e.setSize(d.framebufferWidth,d.framebufferHeight,!1),A=new Ws(d.framebufferWidth,d.framebufferHeight,{format:Mn,type:qi,colorSpace:e.outputColorSpace,stencilBuffer:_.stencil,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1})}A.isXRRenderTarget=!0,this.setFoveation(l),c=null,o=await s.requestReferenceSpace(a),Ne.setContext(s),Ne.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return g.getDepthTexture()};function q(J){for(let ne=0;ne<J.removed.length;ne++){const xe=J.removed[ne],Be=b.indexOf(xe);Be>=0&&(b[Be]=null,y[Be].disconnect(xe))}for(let ne=0;ne<J.added.length;ne++){const xe=J.added[ne];let Be=b.indexOf(xe);if(Be===-1){for(let Ve=0;Ve<y.length;Ve++)if(Ve>=b.length){b.push(xe),Be=Ve;break}else if(b[Ve]===null){b[Ve]=xe,Be=Ve;break}if(Be===-1)break}const Te=y[Be];Te&&Te.connect(xe)}}const X=new F,ee=new F;function ce(J,ne,xe){X.setFromMatrixPosition(ne.matrixWorld),ee.setFromMatrixPosition(xe.matrixWorld);const Be=X.distanceTo(ee),Te=ne.projectionMatrix.elements,Ve=xe.projectionMatrix.elements,L=Te[14]/(Te[10]-1),U=Te[14]/(Te[10]+1),Y=(Te[9]+1)/Te[5],w=(Te[9]-1)/Te[5],oe=(Te[8]-1)/Te[0],re=(Ve[8]+1)/Ve[0],pe=L*oe,se=L*re,me=Be/(-oe+re),ie=me*-oe;if(ne.matrixWorld.decompose(J.position,J.quaternion,J.scale),J.translateX(ie),J.translateZ(me),J.matrixWorld.compose(J.position,J.quaternion,J.scale),J.matrixWorldInverse.copy(J.matrixWorld).invert(),Te[10]===-1)J.projectionMatrix.copy(ne.projectionMatrix),J.projectionMatrixInverse.copy(ne.projectionMatrixInverse);else{const Ae=L+me,R=U+me,C=pe-ie,W=se+(Be-ie),$=Y*U/R*Ae,fe=w*U/R*Ae;J.projectionMatrix.makePerspective(C,W,$,fe,Ae,R),J.projectionMatrixInverse.copy(J.projectionMatrix).invert()}}function be(J,ne){ne===null?J.matrixWorld.copy(J.matrix):J.matrixWorld.multiplyMatrices(ne.matrixWorld,J.matrix),J.matrixWorldInverse.copy(J.matrixWorld).invert()}this.updateCamera=function(J){if(s===null)return;let ne=J.near,xe=J.far;g.texture!==null&&(g.depthNear>0&&(ne=g.depthNear),g.depthFar>0&&(xe=g.depthFar)),P.near=T.near=M.near=ne,P.far=T.far=M.far=xe,(B!==P.near||N!==P.far)&&(s.updateRenderState({depthNear:P.near,depthFar:P.far}),B=P.near,N=P.far),P.layers.mask=J.layers.mask|6,M.layers.mask=P.layers.mask&3,T.layers.mask=P.layers.mask&5;const Be=J.parent,Te=P.cameras;be(P,Be);for(let Ve=0;Ve<Te.length;Ve++)be(Te[Ve],Be);Te.length===2?ce(P,M,T):P.projectionMatrix.copy(M.projectionMatrix),Re(J,P,Be)};function Re(J,ne,xe){xe===null?J.matrix.copy(ne.matrixWorld):(J.matrix.copy(xe.matrixWorld),J.matrix.invert(),J.matrix.multiply(ne.matrixWorld)),J.matrix.decompose(J.position,J.quaternion,J.scale),J.updateMatrixWorld(!0),J.projectionMatrix.copy(ne.projectionMatrix),J.projectionMatrixInverse.copy(ne.projectionMatrixInverse),J.isPerspectiveCamera&&(J.fov=Df*2*Math.atan(1/J.projectionMatrix.elements[5]),J.zoom=1)}this.getCamera=function(){return P},this.getFoveation=function(){if(!(h===null&&d===null))return l},this.setFoveation=function(J){l=J,h!==null&&(h.fixedFoveation=J),d!==null&&d.fixedFoveation!==void 0&&(d.fixedFoveation=J)},this.hasDepthSensing=function(){return g.texture!==null},this.getDepthSensingMesh=function(){return g.getMesh(P)},this.getCameraTexture=function(J){return m[J]};let Fe=null;function Oe(J,ne){if(u=ne.getViewerPose(c||o),x=ne,u!==null){const xe=u.views;d!==null&&(e.setRenderTargetFramebuffer(A,d.framebuffer),e.setRenderTarget(A));let Be=!1;xe.length!==P.cameras.length&&(P.cameras.length=0,Be=!0);for(let U=0;U<xe.length;U++){const Y=xe[U];let w=null;if(d!==null)w=d.getViewport(Y);else{const re=f.getViewSubImage(h,Y);w=re.viewport,U===0&&(e.setRenderTargetTextures(A,re.colorTexture,re.depthStencilTexture),e.setRenderTarget(A))}let oe=I[U];oe===void 0&&(oe=new ui,oe.layers.enable(U),oe.viewport=new Dt,I[U]=oe),oe.matrix.fromArray(Y.transform.matrix),oe.matrix.decompose(oe.position,oe.quaternion,oe.scale),oe.projectionMatrix.fromArray(Y.projectionMatrix),oe.projectionMatrixInverse.copy(oe.projectionMatrix).invert(),oe.viewport.set(w.x,w.y,w.width,w.height),U===0&&(P.matrix.copy(oe.matrix),P.matrix.decompose(P.position,P.quaternion,P.scale)),Be===!0&&P.cameras.push(oe)}const Te=s.enabledFeatures;if(Te&&Te.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&p){f=n.getBinding();const U=f.getDepthInformation(xe[0]);U&&U.isValid&&U.texture&&g.init(U,s.renderState)}if(Te&&Te.includes("camera-access")&&p){e.state.unbindTexture(),f=n.getBinding();for(let U=0;U<xe.length;U++){const Y=xe[U].camera;if(Y){let w=m[Y];w||(w=new fg,m[Y]=w);const oe=f.getCameraImage(Y);w.sourceTexture=oe}}}}for(let xe=0;xe<y.length;xe++){const Be=b[xe],Te=y[xe];Be!==null&&Te!==void 0&&Te.update(Be,ne,c||o)}Fe&&Fe(J,ne),ne.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:ne}),x=null}const Ne=new hg;Ne.setAnimationLoop(Oe),this.setAnimationLoop=function(J){Fe=J},this.dispose=function(){}}}const ar=new Ei,dC=new Ye;function pC(i,e){function t(g,m){g.matrixAutoUpdate===!0&&g.updateMatrix(),m.value.copy(g.matrix)}function n(g,m){m.color.getRGB(g.fogColor.value,ag(i)),m.isFog?(g.fogNear.value=m.near,g.fogFar.value=m.far):m.isFogExp2&&(g.fogDensity.value=m.density)}function s(g,m,_,S,A){m.isMeshBasicMaterial||m.isMeshLambertMaterial?r(g,m):m.isMeshToonMaterial?(r(g,m),f(g,m)):m.isMeshPhongMaterial?(r(g,m),u(g,m)):m.isMeshStandardMaterial?(r(g,m),h(g,m),m.isMeshPhysicalMaterial&&d(g,m,A)):m.isMeshMatcapMaterial?(r(g,m),x(g,m)):m.isMeshDepthMaterial?r(g,m):m.isMeshDistanceMaterial?(r(g,m),p(g,m)):m.isMeshNormalMaterial?r(g,m):m.isLineBasicMaterial?(o(g,m),m.isLineDashedMaterial&&a(g,m)):m.isPointsMaterial?l(g,m,_,S):m.isSpriteMaterial?c(g,m):m.isShadowMaterial?(g.color.value.copy(m.color),g.opacity.value=m.opacity):m.isShaderMaterial&&(m.uniformsNeedUpdate=!1)}function r(g,m){g.opacity.value=m.opacity,m.color&&g.diffuse.value.copy(m.color),m.emissive&&g.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity),m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.bumpMap&&(g.bumpMap.value=m.bumpMap,t(m.bumpMap,g.bumpMapTransform),g.bumpScale.value=m.bumpScale,m.side===Bn&&(g.bumpScale.value*=-1)),m.normalMap&&(g.normalMap.value=m.normalMap,t(m.normalMap,g.normalMapTransform),g.normalScale.value.copy(m.normalScale),m.side===Bn&&g.normalScale.value.negate()),m.displacementMap&&(g.displacementMap.value=m.displacementMap,t(m.displacementMap,g.displacementMapTransform),g.displacementScale.value=m.displacementScale,g.displacementBias.value=m.displacementBias),m.emissiveMap&&(g.emissiveMap.value=m.emissiveMap,t(m.emissiveMap,g.emissiveMapTransform)),m.specularMap&&(g.specularMap.value=m.specularMap,t(m.specularMap,g.specularMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest);const _=e.get(m),S=_.envMap,A=_.envMapRotation;S&&(g.envMap.value=S,ar.copy(A),ar.x*=-1,ar.y*=-1,ar.z*=-1,S.isCubeTexture&&S.isRenderTargetTexture===!1&&(ar.y*=-1,ar.z*=-1),g.envMapRotation.value.setFromMatrix4(dC.makeRotationFromEuler(ar)),g.flipEnvMap.value=S.isCubeTexture&&S.isRenderTargetTexture===!1?-1:1,g.reflectivity.value=m.reflectivity,g.ior.value=m.ior,g.refractionRatio.value=m.refractionRatio),m.lightMap&&(g.lightMap.value=m.lightMap,g.lightMapIntensity.value=m.lightMapIntensity,t(m.lightMap,g.lightMapTransform)),m.aoMap&&(g.aoMap.value=m.aoMap,g.aoMapIntensity.value=m.aoMapIntensity,t(m.aoMap,g.aoMapTransform))}function o(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform))}function a(g,m){g.dashSize.value=m.dashSize,g.totalSize.value=m.dashSize+m.gapSize,g.scale.value=m.scale}function l(g,m,_,S){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.size.value=m.size*_,g.scale.value=S*.5,m.map&&(g.map.value=m.map,t(m.map,g.uvTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function c(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.rotation.value=m.rotation,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function u(g,m){g.specular.value.copy(m.specular),g.shininess.value=Math.max(m.shininess,1e-4)}function f(g,m){m.gradientMap&&(g.gradientMap.value=m.gradientMap)}function h(g,m){g.metalness.value=m.metalness,m.metalnessMap&&(g.metalnessMap.value=m.metalnessMap,t(m.metalnessMap,g.metalnessMapTransform)),g.roughness.value=m.roughness,m.roughnessMap&&(g.roughnessMap.value=m.roughnessMap,t(m.roughnessMap,g.roughnessMapTransform)),m.envMap&&(g.envMapIntensity.value=m.envMapIntensity)}function d(g,m,_){g.ior.value=m.ior,m.sheen>0&&(g.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen),g.sheenRoughness.value=m.sheenRoughness,m.sheenColorMap&&(g.sheenColorMap.value=m.sheenColorMap,t(m.sheenColorMap,g.sheenColorMapTransform)),m.sheenRoughnessMap&&(g.sheenRoughnessMap.value=m.sheenRoughnessMap,t(m.sheenRoughnessMap,g.sheenRoughnessMapTransform))),m.clearcoat>0&&(g.clearcoat.value=m.clearcoat,g.clearcoatRoughness.value=m.clearcoatRoughness,m.clearcoatMap&&(g.clearcoatMap.value=m.clearcoatMap,t(m.clearcoatMap,g.clearcoatMapTransform)),m.clearcoatRoughnessMap&&(g.clearcoatRoughnessMap.value=m.clearcoatRoughnessMap,t(m.clearcoatRoughnessMap,g.clearcoatRoughnessMapTransform)),m.clearcoatNormalMap&&(g.clearcoatNormalMap.value=m.clearcoatNormalMap,t(m.clearcoatNormalMap,g.clearcoatNormalMapTransform),g.clearcoatNormalScale.value.copy(m.clearcoatNormalScale),m.side===Bn&&g.clearcoatNormalScale.value.negate())),m.dispersion>0&&(g.dispersion.value=m.dispersion),m.iridescence>0&&(g.iridescence.value=m.iridescence,g.iridescenceIOR.value=m.iridescenceIOR,g.iridescenceThicknessMinimum.value=m.iridescenceThicknessRange[0],g.iridescenceThicknessMaximum.value=m.iridescenceThicknessRange[1],m.iridescenceMap&&(g.iridescenceMap.value=m.iridescenceMap,t(m.iridescenceMap,g.iridescenceMapTransform)),m.iridescenceThicknessMap&&(g.iridescenceThicknessMap.value=m.iridescenceThicknessMap,t(m.iridescenceThicknessMap,g.iridescenceThicknessMapTransform))),m.transmission>0&&(g.transmission.value=m.transmission,g.transmissionSamplerMap.value=_.texture,g.transmissionSamplerSize.value.set(_.width,_.height),m.transmissionMap&&(g.transmissionMap.value=m.transmissionMap,t(m.transmissionMap,g.transmissionMapTransform)),g.thickness.value=m.thickness,m.thicknessMap&&(g.thicknessMap.value=m.thicknessMap,t(m.thicknessMap,g.thicknessMapTransform)),g.attenuationDistance.value=m.attenuationDistance,g.attenuationColor.value.copy(m.attenuationColor)),m.anisotropy>0&&(g.anisotropyVector.value.set(m.anisotropy*Math.cos(m.anisotropyRotation),m.anisotropy*Math.sin(m.anisotropyRotation)),m.anisotropyMap&&(g.anisotropyMap.value=m.anisotropyMap,t(m.anisotropyMap,g.anisotropyMapTransform))),g.specularIntensity.value=m.specularIntensity,g.specularColor.value.copy(m.specularColor),m.specularColorMap&&(g.specularColorMap.value=m.specularColorMap,t(m.specularColorMap,g.specularColorMapTransform)),m.specularIntensityMap&&(g.specularIntensityMap.value=m.specularIntensityMap,t(m.specularIntensityMap,g.specularIntensityMapTransform))}function x(g,m){m.matcap&&(g.matcap.value=m.matcap)}function p(g,m){const _=e.get(m).light;g.referencePosition.value.setFromMatrixPosition(_.matrixWorld),g.nearDistance.value=_.shadow.camera.near,g.farDistance.value=_.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:s}}function mC(i,e,t,n){let s={},r={},o=[];const a=i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);function l(_,S){const A=S.program;n.uniformBlockBinding(_,A)}function c(_,S){let A=s[_.id];A===void 0&&(x(_),A=u(_),s[_.id]=A,_.addEventListener("dispose",g));const y=S.program;n.updateUBOMapping(_,y);const b=e.render.frame;r[_.id]!==b&&(h(_),r[_.id]=b)}function u(_){const S=f();_.__bindingPointIndex=S;const A=i.createBuffer(),y=_.__size,b=_.usage;return i.bindBuffer(i.UNIFORM_BUFFER,A),i.bufferData(i.UNIFORM_BUFFER,y,b),i.bindBuffer(i.UNIFORM_BUFFER,null),i.bindBufferBase(i.UNIFORM_BUFFER,S,A),A}function f(){for(let _=0;_<a;_++)if(o.indexOf(_)===-1)return o.push(_),_;return Wt("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function h(_){const S=s[_.id],A=_.uniforms,y=_.__cache;i.bindBuffer(i.UNIFORM_BUFFER,S);for(let b=0,v=A.length;b<v;b++){const E=Array.isArray(A[b])?A[b]:[A[b]];for(let M=0,T=E.length;M<T;M++){const I=E[M];if(d(I,b,M,y)===!0){const P=I.__offset,B=Array.isArray(I.value)?I.value:[I.value];let N=0;for(let G=0;G<B.length;G++){const V=B[G],q=p(V);typeof V=="number"||typeof V=="boolean"?(I.__data[0]=V,i.bufferSubData(i.UNIFORM_BUFFER,P+N,I.__data)):V.isMatrix3?(I.__data[0]=V.elements[0],I.__data[1]=V.elements[1],I.__data[2]=V.elements[2],I.__data[3]=0,I.__data[4]=V.elements[3],I.__data[5]=V.elements[4],I.__data[6]=V.elements[5],I.__data[7]=0,I.__data[8]=V.elements[6],I.__data[9]=V.elements[7],I.__data[10]=V.elements[8],I.__data[11]=0):(V.toArray(I.__data,N),N+=q.storage/Float32Array.BYTES_PER_ELEMENT)}i.bufferSubData(i.UNIFORM_BUFFER,P,I.__data)}}}i.bindBuffer(i.UNIFORM_BUFFER,null)}function d(_,S,A,y){const b=_.value,v=S+"_"+A;if(y[v]===void 0)return typeof b=="number"||typeof b=="boolean"?y[v]=b:y[v]=b.clone(),!0;{const E=y[v];if(typeof b=="number"||typeof b=="boolean"){if(E!==b)return y[v]=b,!0}else if(E.equals(b)===!1)return E.copy(b),!0}return!1}function x(_){const S=_.uniforms;let A=0;const y=16;for(let v=0,E=S.length;v<E;v++){const M=Array.isArray(S[v])?S[v]:[S[v]];for(let T=0,I=M.length;T<I;T++){const P=M[T],B=Array.isArray(P.value)?P.value:[P.value];for(let N=0,G=B.length;N<G;N++){const V=B[N],q=p(V),X=A%y,ee=X%q.boundary,ce=X+ee;A+=ee,ce!==0&&y-ce<q.storage&&(A+=y-ce),P.__data=new Float32Array(q.storage/Float32Array.BYTES_PER_ELEMENT),P.__offset=A,A+=q.storage}}}const b=A%y;return b>0&&(A+=y-b),_.__size=A,_.__cache={},this}function p(_){const S={boundary:0,storage:0};return typeof _=="number"||typeof _=="boolean"?(S.boundary=4,S.storage=4):_.isVector2?(S.boundary=8,S.storage=8):_.isVector3||_.isColor?(S.boundary=16,S.storage=12):_.isVector4?(S.boundary=16,S.storage=16):_.isMatrix3?(S.boundary=48,S.storage=48):_.isMatrix4?(S.boundary=64,S.storage=64):_.isTexture?Ze("WebGLRenderer: Texture samplers can not be part of an uniforms group."):Ze("WebGLRenderer: Unsupported uniform value type.",_),S}function g(_){const S=_.target;S.removeEventListener("dispose",g);const A=o.indexOf(S.__bindingPointIndex);o.splice(A,1),i.deleteBuffer(s[S.id]),delete s[S.id],delete r[S.id]}function m(){for(const _ in s)i.deleteBuffer(s[_]);o=[],s={},r={}}return{bind:l,update:c,dispose:m}}const gC=new Uint16Array([11481,15204,11534,15171,11808,15015,12385,14843,12894,14716,13396,14600,13693,14483,13976,14366,14237,14171,14405,13961,14511,13770,14605,13598,14687,13444,14760,13305,14822,13066,14876,12857,14923,12675,14963,12517,14997,12379,15025,12230,15049,12023,15070,11843,15086,11687,15100,11551,15111,11433,15120,11330,15127,11217,15132,11060,15135,10922,15138,10801,15139,10695,15139,10600,13012,14923,13020,14917,13064,14886,13176,14800,13349,14666,13513,14526,13724,14398,13960,14230,14200,14020,14383,13827,14488,13651,14583,13491,14667,13348,14740,13132,14803,12908,14856,12713,14901,12542,14938,12394,14968,12241,14992,12017,15010,11822,15024,11654,15034,11507,15041,11380,15044,11269,15044,11081,15042,10913,15037,10764,15031,10635,15023,10520,15014,10419,15003,10330,13657,14676,13658,14673,13670,14660,13698,14622,13750,14547,13834,14442,13956,14317,14112,14093,14291,13889,14407,13704,14499,13538,14586,13389,14664,13201,14733,12966,14792,12758,14842,12577,14882,12418,14915,12272,14940,12033,14959,11826,14972,11646,14980,11490,14983,11355,14983,11212,14979,11008,14971,10830,14961,10675,14950,10540,14936,10420,14923,10315,14909,10204,14894,10041,14089,14460,14090,14459,14096,14452,14112,14431,14141,14388,14186,14305,14252,14130,14341,13941,14399,13756,14467,13585,14539,13430,14610,13272,14677,13026,14737,12808,14790,12617,14833,12449,14869,12303,14896,12065,14916,11845,14929,11655,14937,11490,14939,11347,14936,11184,14930,10970,14921,10783,14912,10621,14900,10480,14885,10356,14867,10247,14848,10062,14827,9894,14805,9745,14400,14208,14400,14206,14402,14198,14406,14174,14415,14122,14427,14035,14444,13913,14469,13767,14504,13613,14548,13463,14598,13324,14651,13082,14704,12858,14752,12658,14795,12483,14831,12330,14860,12106,14881,11875,14895,11675,14903,11501,14905,11351,14903,11178,14900,10953,14892,10757,14880,10589,14865,10442,14847,10313,14827,10162,14805,9965,14782,9792,14757,9642,14731,9507,14562,13883,14562,13883,14563,13877,14566,13862,14570,13830,14576,13773,14584,13689,14595,13582,14613,13461,14637,13336,14668,13120,14704,12897,14741,12695,14776,12516,14808,12358,14835,12150,14856,11910,14870,11701,14878,11519,14882,11361,14884,11187,14880,10951,14871,10748,14858,10572,14842,10418,14823,10286,14801,10099,14777,9897,14751,9722,14725,9567,14696,9430,14666,9309,14702,13604,14702,13604,14702,13600,14703,13591,14705,13570,14707,13533,14709,13477,14712,13400,14718,13305,14727,13106,14743,12907,14762,12716,14784,12539,14807,12380,14827,12190,14844,11943,14855,11727,14863,11539,14870,11376,14871,11204,14868,10960,14858,10748,14845,10565,14829,10406,14809,10269,14786,10058,14761,9852,14734,9671,14705,9512,14674,9374,14641,9253,14608,9076,14821,13366,14821,13365,14821,13364,14821,13358,14821,13344,14821,13320,14819,13252,14817,13145,14815,13011,14814,12858,14817,12698,14823,12539,14832,12389,14841,12214,14850,11968,14856,11750,14861,11558,14866,11390,14867,11226,14862,10972,14853,10754,14840,10565,14823,10401,14803,10259,14780,10032,14754,9820,14725,9635,14694,9473,14661,9333,14627,9203,14593,8988,14557,8798,14923,13014,14922,13014,14922,13012,14922,13004,14920,12987,14919,12957,14915,12907,14909,12834,14902,12738,14894,12623,14888,12498,14883,12370,14880,12203,14878,11970,14875,11759,14873,11569,14874,11401,14872,11243,14865,10986,14855,10762,14842,10568,14825,10401,14804,10255,14781,10017,14754,9799,14725,9611,14692,9445,14658,9301,14623,9139,14587,8920,14548,8729,14509,8562,15008,12672,15008,12672,15008,12671,15007,12667,15005,12656,15001,12637,14997,12605,14989,12556,14978,12490,14966,12407,14953,12313,14940,12136,14927,11934,14914,11742,14903,11563,14896,11401,14889,11247,14879,10992,14866,10767,14851,10570,14833,10400,14812,10252,14789,10007,14761,9784,14731,9592,14698,9424,14663,9279,14627,9088,14588,8868,14548,8676,14508,8508,14467,8360,15080,12386,15080,12386,15079,12385,15078,12383,15076,12378,15072,12367,15066,12347,15057,12315,15045,12253,15030,12138,15012,11998,14993,11845,14972,11685,14951,11530,14935,11383,14920,11228,14904,10981,14887,10762,14870,10567,14850,10397,14827,10248,14803,9997,14774,9771,14743,9578,14710,9407,14674,9259,14637,9048,14596,8826,14555,8632,14514,8464,14471,8317,14427,8182,15139,12008,15139,12008,15138,12008,15137,12007,15135,12003,15130,11990,15124,11969,15115,11929,15102,11872,15086,11794,15064,11693,15041,11581,15013,11459,14987,11336,14966,11170,14944,10944,14921,10738,14898,10552,14875,10387,14850,10239,14824,9983,14794,9758,14762,9563,14728,9392,14692,9244,14653,9014,14611,8791,14569,8597,14526,8427,14481,8281,14436,8110,14391,7885,15188,11617,15188,11617,15187,11617,15186,11618,15183,11617,15179,11612,15173,11601,15163,11581,15150,11546,15133,11495,15110,11427,15083,11346,15051,11246,15024,11057,14996,10868,14967,10687,14938,10517,14911,10362,14882,10206,14853,9956,14821,9737,14787,9543,14752,9375,14715,9228,14675,8980,14632,8760,14589,8565,14544,8395,14498,8248,14451,8049,14404,7824,14357,7630,15228,11298,15228,11298,15227,11299,15226,11301,15223,11303,15219,11302,15213,11299,15204,11290,15191,11271,15174,11217,15150,11129,15119,11015,15087,10886,15057,10744,15024,10599,14990,10455,14957,10318,14924,10143,14891,9911,14856,9701,14820,9516,14782,9352,14744,9200,14703,8946,14659,8725,14615,8533,14568,8366,14521,8220,14472,7992,14423,7770,14374,7578,14315,7408,15260,10819,15260,10819,15259,10822,15258,10826,15256,10832,15251,10836,15246,10841,15237,10838,15225,10821,15207,10788,15183,10734,15151,10660,15120,10571,15087,10469,15049,10359,15012,10249,14974,10041,14937,9837,14900,9647,14860,9475,14820,9320,14779,9147,14736,8902,14691,8688,14646,8499,14598,8335,14549,8189,14499,7940,14448,7720,14397,7529,14347,7363,14256,7218,15285,10410,15285,10411,15285,10413,15284,10418,15282,10425,15278,10434,15272,10442,15264,10449,15252,10445,15235,10433,15210,10403,15179,10358,15149,10301,15113,10218,15073,10059,15033,9894,14991,9726,14951,9565,14909,9413,14865,9273,14822,9073,14777,8845,14730,8641,14682,8459,14633,8300,14583,8129,14531,7883,14479,7670,14426,7482,14373,7321,14305,7176,14201,6939,15305,9939,15305,9940,15305,9945,15304,9955,15302,9967,15298,9989,15293,10010,15286,10033,15274,10044,15258,10045,15233,10022,15205,9975,15174,9903,15136,9808,15095,9697,15053,9578,15009,9451,14965,9327,14918,9198,14871,8973,14825,8766,14775,8579,14725,8408,14675,8259,14622,8058,14569,7821,14515,7615,14460,7435,14405,7276,14350,7108,14256,6866,14149,6653,15321,9444,15321,9445,15321,9448,15320,9458,15317,9470,15314,9490,15310,9515,15302,9540,15292,9562,15276,9579,15251,9577,15226,9559,15195,9519,15156,9463,15116,9389,15071,9304,15025,9208,14978,9023,14927,8838,14878,8661,14827,8496,14774,8344,14722,8206,14667,7973,14612,7749,14556,7555,14499,7382,14443,7229,14385,7025,14322,6791,14210,6588,14100,6409,15333,8920,15333,8921,15332,8927,15332,8943,15329,8965,15326,9002,15322,9048,15316,9106,15307,9162,15291,9204,15267,9221,15244,9221,15212,9196,15175,9134,15133,9043,15088,8930,15040,8801,14990,8665,14938,8526,14886,8391,14830,8261,14775,8087,14719,7866,14661,7664,14603,7482,14544,7322,14485,7178,14426,6936,14367,6713,14281,6517,14166,6348,14054,6198,15341,8360,15341,8361,15341,8366,15341,8379,15339,8399,15336,8431,15332,8473,15326,8527,15318,8585,15302,8632,15281,8670,15258,8690,15227,8690,15191,8664,15149,8612,15104,8543,15055,8456,15001,8360,14948,8259,14892,8122,14834,7923,14776,7734,14716,7558,14656,7397,14595,7250,14534,7070,14472,6835,14410,6628,14350,6443,14243,6283,14125,6135,14010,5889,15348,7715,15348,7717,15348,7725,15347,7745,15345,7780,15343,7836,15339,7905,15334,8e3,15326,8103,15310,8193,15293,8239,15270,8270,15240,8287,15204,8283,15163,8260,15118,8223,15067,8143,15014,8014,14958,7873,14899,7723,14839,7573,14778,7430,14715,7293,14652,7164,14588,6931,14524,6720,14460,6531,14396,6362,14330,6210,14207,6015,14086,5781,13969,5576,15352,7114,15352,7116,15352,7128,15352,7159,15350,7195,15348,7237,15345,7299,15340,7374,15332,7457,15317,7544,15301,7633,15280,7703,15251,7754,15216,7775,15176,7767,15131,7733,15079,7670,15026,7588,14967,7492,14906,7387,14844,7278,14779,7171,14714,6965,14648,6770,14581,6587,14515,6420,14448,6269,14382,6123,14299,5881,14172,5665,14049,5477,13929,5310,15355,6329,15355,6330,15355,6339,15355,6362,15353,6410,15351,6472,15349,6572,15344,6688,15337,6835,15323,6985,15309,7142,15287,7220,15260,7277,15226,7310,15188,7326,15142,7318,15090,7285,15036,7239,14976,7177,14914,7045,14849,6892,14782,6736,14714,6581,14645,6433,14576,6293,14506,6164,14438,5946,14369,5733,14270,5540,14140,5369,14014,5216,13892,5043,15357,5483,15357,5484,15357,5496,15357,5528,15356,5597,15354,5692,15351,5835,15347,6011,15339,6195,15328,6317,15314,6446,15293,6566,15268,6668,15235,6746,15197,6796,15152,6811,15101,6790,15046,6748,14985,6673,14921,6583,14854,6479,14785,6371,14714,6259,14643,6149,14571,5946,14499,5750,14428,5567,14358,5401,14242,5250,14109,5111,13980,4870,13856,4657,15359,4555,15359,4557,15358,4573,15358,4633,15357,4715,15355,4841,15353,5061,15349,5216,15342,5391,15331,5577,15318,5770,15299,5967,15274,6150,15243,6223,15206,6280,15161,6310,15111,6317,15055,6300,14994,6262,14928,6208,14860,6141,14788,5994,14715,5838,14641,5684,14566,5529,14492,5384,14418,5247,14346,5121,14216,4892,14079,4682,13948,4496,13822,4330,15359,3498,15359,3501,15359,3520,15359,3598,15358,3719,15356,3860,15355,4137,15351,4305,15344,4563,15334,4809,15321,5116,15303,5273,15280,5418,15250,5547,15214,5653,15170,5722,15120,5761,15064,5763,15002,5733,14935,5673,14865,5597,14792,5504,14716,5400,14640,5294,14563,5185,14486,5041,14410,4841,14335,4655,14191,4482,14051,4325,13918,4183,13790,4012,15360,2282,15360,2285,15360,2306,15360,2401,15359,2547,15357,2748,15355,3103,15352,3349,15345,3675,15336,4020,15324,4272,15307,4496,15285,4716,15255,4908,15220,5086,15178,5170,15128,5214,15072,5234,15010,5231,14943,5206,14871,5166,14796,5102,14718,4971,14639,4833,14559,4687,14480,4541,14402,4401,14315,4268,14167,4142,14025,3958,13888,3747,13759,3556,15360,923,15360,925,15360,946,15360,1052,15359,1214,15357,1494,15356,1892,15352,2274,15346,2663,15338,3099,15326,3393,15309,3679,15288,3980,15260,4183,15226,4325,15185,4437,15136,4517,15080,4570,15018,4591,14950,4581,14877,4545,14800,4485,14720,4411,14638,4325,14556,4231,14475,4136,14395,3988,14297,3803,14145,3628,13999,3465,13861,3314,13729,3177,15360,263,15360,264,15360,272,15360,325,15359,407,15358,548,15356,780,15352,1144,15347,1580,15339,2099,15328,2425,15312,2795,15292,3133,15264,3329,15232,3517,15191,3689,15143,3819,15088,3923,15025,3978,14956,3999,14882,3979,14804,3931,14722,3855,14639,3756,14554,3645,14470,3529,14388,3409,14279,3289,14124,3173,13975,3055,13834,2848,13701,2658,15360,49,15360,49,15360,52,15360,75,15359,111,15358,201,15356,283,15353,519,15348,726,15340,1045,15329,1415,15314,1795,15295,2173,15269,2410,15237,2649,15197,2866,15150,3054,15095,3140,15032,3196,14963,3228,14888,3236,14808,3224,14725,3191,14639,3146,14553,3088,14466,2976,14382,2836,14262,2692,14103,2549,13952,2409,13808,2278,13674,2154,15360,4,15360,4,15360,4,15360,13,15359,33,15358,59,15357,112,15353,199,15348,302,15341,456,15331,628,15316,827,15297,1082,15272,1332,15241,1601,15202,1851,15156,2069,15101,2172,15039,2256,14970,2314,14894,2348,14813,2358,14728,2344,14640,2311,14551,2263,14463,2203,14376,2133,14247,2059,14084,1915,13930,1761,13784,1609,13648,1464,15360,0,15360,0,15360,0,15360,3,15359,18,15358,26,15357,53,15354,80,15348,97,15341,165,15332,238,15318,326,15299,427,15275,529,15245,654,15207,771,15161,885,15108,994,15046,1089,14976,1170,14900,1229,14817,1266,14731,1284,14641,1282,14550,1260,14460,1223,14370,1174,14232,1116,14066,1050,13909,981,13761,910,13623,839]);let es=null;function xC(){return es===null&&(es=new is(gC,32,32,Sh,Tr),es.minFilter=di,es.magFilter=di,es.wrapS=ds,es.wrapT=ds,es.generateMipmaps=!1,es.needsUpdate=!0),es}class _C{constructor(e={}){const{canvas:t=XS(),context:n=null,depth:s=!0,stencil:r=!1,alpha:o=!1,antialias:a=!1,premultipliedAlpha:l=!0,preserveDrawingBuffer:c=!1,powerPreference:u="default",failIfMajorPerformanceCaveat:f=!1,reversedDepthBuffer:h=!1}=e;this.isWebGLRenderer=!0;let d;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");d=n.getContextAttributes().alpha}else d=o;const x=new Set([lo,Ah,dc]),p=new Set([qi,pi,va,Sa,_h,vh]),g=new Uint32Array(4),m=new Int32Array(4);let _=null,S=null;const A=[],y=[];this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=zs,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const b=this;let v=!1;this._outputColorSpace=ai;let E=0,M=0,T=null,I=-1,P=null;const B=new Dt,N=new Dt;let G=null;const V=new rt(0);let q=0,X=t.width,ee=t.height,ce=1,be=null,Re=null;const Fe=new Dt(0,0,X,ee),Oe=new Dt(0,0,X,ee);let Ne=!1;const J=new ug;let ne=!1,xe=!1;const Be=new Ye,Te=new F,Ve=new Dt,L={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let U=!1;function Y(){return T===null?ce:1}let w=n;function oe(D,Q){return t.getContext(D,Q)}try{const D={alpha:!0,depth:s,stencil:r,antialias:a,premultipliedAlpha:l,preserveDrawingBuffer:c,powerPreference:u,failIfMajorPerformanceCaveat:f};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${gh}`),t.addEventListener("webglcontextlost",de,!1),t.addEventListener("webglcontextrestored",le,!1),t.addEventListener("webglcontextcreationerror",Ce,!1),w===null){const Q="webgl2";if(w=oe(Q,D),w===null)throw oe(Q)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(D){throw D("WebGLRenderer: "+D.message),D}let re,pe,se,me,ie,Ae,R,C,W,$,fe,Z,Ie,ye,Ue,k,te,_e,H,z,he,Me,O,ve;function ge(){re=new EM(w),re.init(),Me=new xg(w,re),pe=new _M(w,re,e,Me),se=new aC(w,re),pe.reversedDepthBuffer&&h&&se.buffers.depth.setReversed(!0),me=new IM(w),ie=new QT,Ae=new lC(w,re,se,ie,pe,Me,me),R=new SM(b),C=new CM(b),W=new LA(w),O=new gM(w,W),$=new wM(w,W,me,O),fe=new PM(w,$,W,me),H=new DM(w,pe,Ae),k=new vM(ie),Z=new YT(b,R,C,re,pe,O,k),Ie=new pC(b,ie),ye=new jT,Ue=new nC(re),_e=new mM(b,R,C,se,fe,d,l),te=new rC(b,fe,pe),ve=new mC(w,me,pe,se),z=new xM(w,re,me),he=new RM(w,re,me),me.programs=Z.programs,b.capabilities=pe,b.extensions=re,b.properties=ie,b.renderLists=ye,b.shadowMap=te,b.state=se,b.info=me}ge();const Se=new hC(b,w);this.xr=Se,this.getContext=function(){return w},this.getContextAttributes=function(){return w.getContextAttributes()},this.forceContextLoss=function(){const D=re.get("WEBGL_lose_context");D&&D.loseContext()},this.forceContextRestore=function(){const D=re.get("WEBGL_lose_context");D&&D.restoreContext()},this.getPixelRatio=function(){return ce},this.setPixelRatio=function(D){D!==void 0&&(ce=D,this.setSize(X,ee,!1))},this.getSize=function(D){return D.set(X,ee)},this.setSize=function(D,Q,ae=!0){if(Se.isPresenting){Ze("WebGLRenderer: Can't change size while VR device is presenting.");return}X=D,ee=Q,t.width=Math.floor(D*ce),t.height=Math.floor(Q*ce),ae===!0&&(t.style.width=D+"px",t.style.height=Q+"px"),this.setViewport(0,0,D,Q)},this.getDrawingBufferSize=function(D){return D.set(X*ce,ee*ce).floor()},this.setDrawingBufferSize=function(D,Q,ae){X=D,ee=Q,ce=ae,t.width=Math.floor(D*ae),t.height=Math.floor(Q*ae),this.setViewport(0,0,D,Q)},this.getCurrentViewport=function(D){return D.copy(B)},this.getViewport=function(D){return D.copy(Fe)},this.setViewport=function(D,Q,ae,ue){D.isVector4?Fe.set(D.x,D.y,D.z,D.w):Fe.set(D,Q,ae,ue),se.viewport(B.copy(Fe).multiplyScalar(ce).round())},this.getScissor=function(D){return D.copy(Oe)},this.setScissor=function(D,Q,ae,ue){D.isVector4?Oe.set(D.x,D.y,D.z,D.w):Oe.set(D,Q,ae,ue),se.scissor(N.copy(Oe).multiplyScalar(ce).round())},this.getScissorTest=function(){return Ne},this.setScissorTest=function(D){se.setScissorTest(Ne=D)},this.setOpaqueSort=function(D){be=D},this.setTransparentSort=function(D){Re=D},this.getClearColor=function(D){return D.copy(_e.getClearColor())},this.setClearColor=function(){_e.setClearColor(...arguments)},this.getClearAlpha=function(){return _e.getClearAlpha()},this.setClearAlpha=function(){_e.setClearAlpha(...arguments)},this.clear=function(D=!0,Q=!0,ae=!0){let ue=0;if(D){let K=!1;if(T!==null){const Ee=T.texture.format;K=x.has(Ee)}if(K){const Ee=T.texture.type,Le=p.has(Ee),He=_e.getClearColor(),ke=_e.getClearAlpha(),qe=He.r,Ke=He.g,We=He.b;Le?(g[0]=qe,g[1]=Ke,g[2]=We,g[3]=ke,w.clearBufferuiv(w.COLOR,0,g)):(m[0]=qe,m[1]=Ke,m[2]=We,m[3]=ke,w.clearBufferiv(w.COLOR,0,m))}else ue|=w.COLOR_BUFFER_BIT}Q&&(ue|=w.DEPTH_BUFFER_BIT),ae&&(ue|=w.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),w.clear(ue)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){t.removeEventListener("webglcontextlost",de,!1),t.removeEventListener("webglcontextrestored",le,!1),t.removeEventListener("webglcontextcreationerror",Ce,!1),_e.dispose(),ye.dispose(),Ue.dispose(),ie.dispose(),R.dispose(),C.dispose(),fe.dispose(),O.dispose(),ve.dispose(),Z.dispose(),Se.dispose(),Se.removeEventListener("sessionstart",za),Se.removeEventListener("sessionend",$s),Ri.stop()};function de(D){D.preventDefault(),jd("WebGLRenderer: Context Lost."),v=!0}function le(){jd("WebGLRenderer: Context Restored."),v=!1;const D=me.autoReset,Q=te.enabled,ae=te.autoUpdate,ue=te.needsUpdate,K=te.type;ge(),me.autoReset=D,te.enabled=Q,te.autoUpdate=ae,te.needsUpdate=ue,te.type=K}function Ce(D){Wt("WebGLRenderer: A WebGL context could not be created. Reason: ",D.statusMessage)}function ze(D){const Q=D.target;Q.removeEventListener("dispose",ze),it(Q)}function it(D){Ge(D),ie.remove(D)}function Ge(D){const Q=ie.get(D).programs;Q!==void 0&&(Q.forEach(function(ae){Z.releaseProgram(ae)}),D.isShaderMaterial&&Z.releaseShaderCache(D))}this.renderBufferDirect=function(D,Q,ae,ue,K,Ee){Q===null&&(Q=L);const Le=K.isMesh&&K.matrixWorld.determinant()<0,He=yc(D,Q,ae,ue,K);se.setMaterial(ue,Le);let ke=ae.index,qe=1;if(ue.wireframe===!0){if(ke=$.getWireframeAttribute(ae),ke===void 0)return;qe=2}const Ke=ae.drawRange,We=ae.attributes.position;let nt=Ke.start*qe,dt=(Ke.start+Ke.count)*qe;Ee!==null&&(nt=Math.max(nt,Ee.start*qe),dt=Math.min(dt,(Ee.start+Ee.count)*qe)),ke!==null?(nt=Math.max(nt,0),dt=Math.min(dt,ke.count)):We!=null&&(nt=Math.max(nt,0),dt=Math.min(dt,We.count));const zt=dt-nt;if(zt<0||zt===1/0)return;O.setup(K,ue,He,ae,ke);let kt,St=z;if(ke!==null&&(kt=W.get(ke),St=he,St.setIndex(kt)),K.isMesh)ue.wireframe===!0?(se.setLineWidth(ue.wireframeLinewidth*Y()),St.setMode(w.LINES)):St.setMode(w.TRIANGLES);else if(K.isLine){let Xe=ue.linewidth;Xe===void 0&&(Xe=1),se.setLineWidth(Xe*Y()),K.isLineSegments?St.setMode(w.LINES):K.isLineLoop?St.setMode(w.LINE_LOOP):St.setMode(w.LINE_STRIP)}else K.isPoints?St.setMode(w.POINTS):K.isSprite&&St.setMode(w.TRIANGLES);if(K.isBatchedMesh)if(K._multiDrawInstances!==null)ya("WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."),St.renderMultiDrawInstances(K._multiDrawStarts,K._multiDrawCounts,K._multiDrawCount,K._multiDrawInstances);else if(re.get("WEBGL_multi_draw"))St.renderMultiDraw(K._multiDrawStarts,K._multiDrawCounts,K._multiDrawCount);else{const Xe=K._multiDrawStarts,Pt=K._multiDrawCounts,ot=K._multiDrawCount,Hn=ke?W.get(ke).bytesPerElement:1,Ir=ie.get(ue).currentProgram.getUniforms();for(let Vn=0;Vn<ot;Vn++)Ir.setValue(w,"_gl_DrawID",Vn),St.render(Xe[Vn]/Hn,Pt[Vn])}else if(K.isInstancedMesh)St.renderInstances(nt,zt,K.count);else if(ae.isInstancedBufferGeometry){const Xe=ae._maxInstanceCount!==void 0?ae._maxInstanceCount:1/0,Pt=Math.min(ae.instanceCount,Xe);St.renderInstances(nt,zt,Pt)}else St.render(nt,zt)};function vt(D,Q,ae){D.transparent===!0&&D.side===fi&&D.forceSinglePass===!1?(D.side=Bn,D.needsUpdate=!0,Zs(D,Q,ae),D.side=Xi,D.needsUpdate=!0,Zs(D,Q,ae),D.side=fi):Zs(D,Q,ae)}this.compile=function(D,Q,ae=null){ae===null&&(ae=D),S=Ue.get(ae),S.init(Q),y.push(S),ae.traverseVisible(function(K){K.isLight&&K.layers.test(Q.layers)&&(S.pushLight(K),K.castShadow&&S.pushShadow(K))}),D!==ae&&D.traverseVisible(function(K){K.isLight&&K.layers.test(Q.layers)&&(S.pushLight(K),K.castShadow&&S.pushShadow(K))}),S.setupLights();const ue=new Set;return D.traverse(function(K){if(!(K.isMesh||K.isPoints||K.isLine||K.isSprite))return;const Ee=K.material;if(Ee)if(Array.isArray(Ee))for(let Le=0;Le<Ee.length;Le++){const He=Ee[Le];vt(He,ae,K),ue.add(He)}else vt(Ee,ae,K),ue.add(Ee)}),S=y.pop(),ue},this.compileAsync=function(D,Q,ae=null){const ue=this.compile(D,Q,ae);return new Promise(K=>{function Ee(){if(ue.forEach(function(Le){ie.get(Le).currentProgram.isReady()&&ue.delete(Le)}),ue.size===0){K(D);return}setTimeout(Ee,10)}re.get("KHR_parallel_shader_compile")!==null?Ee():setTimeout(Ee,10)})};let Ct=null;function wi(D){Ct&&Ct(D)}function za(){Ri.stop()}function $s(){Ri.start()}const Ri=new hg;Ri.setAnimationLoop(wi),typeof self<"u"&&Ri.setContext(self),this.setAnimationLoop=function(D){Ct=D,Se.setAnimationLoop(D),D===null?Ri.stop():Ri.start()},Se.addEventListener("sessionstart",za),Se.addEventListener("sessionend",$s),this.render=function(D,Q){if(Q!==void 0&&Q.isCamera!==!0){Wt("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(v===!0)return;if(D.matrixWorldAutoUpdate===!0&&D.updateMatrixWorld(),Q.parent===null&&Q.matrixWorldAutoUpdate===!0&&Q.updateMatrixWorld(),Se.enabled===!0&&Se.isPresenting===!0&&(Se.cameraAutoUpdate===!0&&Se.updateCamera(Q),Q=Se.getCamera()),D.isScene===!0&&D.onBeforeRender(b,D,Q,T),S=Ue.get(D,y.length),S.init(Q),y.push(S),Be.multiplyMatrices(Q.projectionMatrix,Q.matrixWorldInverse),J.setFromProjectionMatrix(Be,Oi,Q.reversedDepth),xe=this.localClippingEnabled,ne=k.init(this.clippingPlanes,xe),_=ye.get(D,A.length),_.init(),A.push(_),Se.enabled===!0&&Se.isPresenting===!0){const Ee=b.xr.getDepthSensingMesh();Ee!==null&&Er(Ee,Q,-1/0,b.sortObjects)}Er(D,Q,0,b.sortObjects),_.finish(),b.sortObjects===!0&&_.sort(be,Re),U=Se.enabled===!1||Se.isPresenting===!1||Se.hasDepthSensing()===!1,U&&_e.addToRenderList(_,D),this.info.render.frame++,ne===!0&&k.beginShadows();const ae=S.state.shadowsArray;te.render(ae,D,Q),ne===!0&&k.endShadows(),this.info.autoReset===!0&&this.info.reset();const ue=_.opaque,K=_.transmissive;if(S.setupLights(),Q.isArrayCamera){const Ee=Q.cameras;if(K.length>0)for(let Le=0,He=Ee.length;Le<He;Le++){const ke=Ee[Le];ka(ue,K,D,ke)}U&&_e.render(D);for(let Le=0,He=Ee.length;Le<He;Le++){const ke=Ee[Le];Bo(_,D,ke,ke.viewport)}}else K.length>0&&ka(ue,K,D,Q),U&&_e.render(D),Bo(_,D,Q);T!==null&&M===0&&(Ae.updateMultisampleRenderTarget(T),Ae.updateRenderTargetMipmap(T)),D.isScene===!0&&D.onAfterRender(b,D,Q),O.resetDefaultState(),I=-1,P=null,y.pop(),y.length>0?(S=y[y.length-1],ne===!0&&k.setGlobalState(b.clippingPlanes,S.state.camera)):S=null,A.pop(),A.length>0?_=A[A.length-1]:_=null};function Er(D,Q,ae,ue){if(D.visible===!1)return;if(D.layers.test(Q.layers)){if(D.isGroup)ae=D.renderOrder;else if(D.isLOD)D.autoUpdate===!0&&D.update(Q);else if(D.isLight)S.pushLight(D),D.castShadow&&S.pushShadow(D);else if(D.isSprite){if(!D.frustumCulled||J.intersectsSprite(D)){ue&&Ve.setFromMatrixPosition(D.matrixWorld).applyMatrix4(Be);const Le=fe.update(D),He=D.material;He.visible&&_.push(D,Le,He,ae,Ve.z,null)}}else if((D.isMesh||D.isLine||D.isPoints)&&(!D.frustumCulled||J.intersectsObject(D))){const Le=fe.update(D),He=D.material;if(ue&&(D.boundingSphere!==void 0?(D.boundingSphere===null&&D.computeBoundingSphere(),Ve.copy(D.boundingSphere.center)):(Le.boundingSphere===null&&Le.computeBoundingSphere(),Ve.copy(Le.boundingSphere.center)),Ve.applyMatrix4(D.matrixWorld).applyMatrix4(Be)),Array.isArray(He)){const ke=Le.groups;for(let qe=0,Ke=ke.length;qe<Ke;qe++){const We=ke[qe],nt=He[We.materialIndex];nt&&nt.visible&&_.push(D,Le,nt,ae,Ve.z,We)}}else He.visible&&_.push(D,Le,He,ae,Ve.z,null)}}const Ee=D.children;for(let Le=0,He=Ee.length;Le<He;Le++)Er(Ee[Le],Q,ae,ue)}function Bo(D,Q,ae,ue){const{opaque:K,transmissive:Ee,transparent:Le}=D;S.setupLightsView(ae),ne===!0&&k.setGlobalState(b.clippingPlanes,ae),ue&&se.viewport(B.copy(ue)),K.length>0&&ii(K,Q,ae),Ee.length>0&&ii(Ee,Q,ae),Le.length>0&&ii(Le,Q,ae),se.buffers.depth.setTest(!0),se.buffers.depth.setMask(!0),se.buffers.color.setMask(!0),se.setPolygonOffset(!1)}function ka(D,Q,ae,ue){if((ae.isScene===!0?ae.overrideMaterial:null)!==null)return;S.state.transmissionRenderTarget[ue.id]===void 0&&(S.state.transmissionRenderTarget[ue.id]=new Ws(1,1,{generateMipmaps:!0,type:re.has("EXT_color_buffer_half_float")||re.has("EXT_color_buffer_float")?Tr:qi,minFilter:mr,samples:4,stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:lt.workingColorSpace}));const Ee=S.state.transmissionRenderTarget[ue.id],Le=ue.viewport||B;Ee.setSize(Le.z*b.transmissionResolutionScale,Le.w*b.transmissionResolutionScale);const He=b.getRenderTarget(),ke=b.getActiveCubeFace(),qe=b.getActiveMipmapLevel();b.setRenderTarget(Ee),b.getClearColor(V),q=b.getClearAlpha(),q<1&&b.setClearColor(16777215,.5),b.clear(),U&&_e.render(ae);const Ke=b.toneMapping;b.toneMapping=zs;const We=ue.viewport;if(ue.viewport!==void 0&&(ue.viewport=void 0),S.setupLightsView(ue),ne===!0&&k.setGlobalState(b.clippingPlanes,ue),ii(D,ae,ue),Ae.updateMultisampleRenderTarget(Ee),Ae.updateRenderTargetMipmap(Ee),re.has("WEBGL_multisampled_render_to_texture")===!1){let nt=!1;for(let dt=0,zt=Q.length;dt<zt;dt++){const kt=Q[dt],{object:St,geometry:Xe,material:Pt,group:ot}=kt;if(Pt.side===fi&&St.layers.test(ue.layers)){const Hn=Pt.side;Pt.side=Bn,Pt.needsUpdate=!0,wr(St,ae,ue,Xe,Pt,ot),Pt.side=Hn,Pt.needsUpdate=!0,nt=!0}}nt===!0&&(Ae.updateMultisampleRenderTarget(Ee),Ae.updateRenderTargetMipmap(Ee))}b.setRenderTarget(He,ke,qe),b.setClearColor(V,q),We!==void 0&&(ue.viewport=We),b.toneMapping=Ke}function ii(D,Q,ae){const ue=Q.isScene===!0?Q.overrideMaterial:null;for(let K=0,Ee=D.length;K<Ee;K++){const Le=D[K],{object:He,geometry:ke,group:qe}=Le;let Ke=Le.material;Ke.allowOverride===!0&&ue!==null&&(Ke=ue),He.layers.test(ae.layers)&&wr(He,Q,ae,ke,Ke,qe)}}function wr(D,Q,ae,ue,K,Ee){D.onBeforeRender(b,Q,ae,ue,K,Ee),D.modelViewMatrix.multiplyMatrices(ae.matrixWorldInverse,D.matrixWorld),D.normalMatrix.getNormalMatrix(D.modelViewMatrix),K.onBeforeRender(b,Q,ae,ue,D,Ee),K.transparent===!0&&K.side===fi&&K.forceSinglePass===!1?(K.side=Bn,K.needsUpdate=!0,b.renderBufferDirect(ae,Q,ue,K,D,Ee),K.side=Xi,K.needsUpdate=!0,b.renderBufferDirect(ae,Q,ue,K,D,Ee),K.side=fi):b.renderBufferDirect(ae,Q,ue,K,D,Ee),D.onAfterRender(b,Q,ae,ue,K,Ee)}function Zs(D,Q,ae){Q.isScene!==!0&&(Q=L);const ue=ie.get(D),K=S.state.lights,Ee=S.state.shadowsArray,Le=K.state.version,He=Z.getParameters(D,K.state,Ee,Q,ae),ke=Z.getProgramCacheKey(He);let qe=ue.programs;ue.environment=D.isMeshStandardMaterial?Q.environment:null,ue.fog=Q.fog,ue.envMap=(D.isMeshStandardMaterial?C:R).get(D.envMap||ue.environment),ue.envMapRotation=ue.environment!==null&&D.envMap===null?Q.environmentRotation:D.envMapRotation,qe===void 0&&(D.addEventListener("dispose",ze),qe=new Map,ue.programs=qe);let Ke=qe.get(ke);if(Ke!==void 0){if(ue.currentProgram===Ke&&ue.lightsStateVersion===Le)return Ha(D,He),Ke}else He.uniforms=Z.getUniforms(D),D.onBeforeCompile(He,b),Ke=Z.acquireProgram(He,ke),qe.set(ke,Ke),ue.uniforms=He.uniforms;const We=ue.uniforms;return(!D.isShaderMaterial&&!D.isRawShaderMaterial||D.clipping===!0)&&(We.clippingPlanes=k.uniform),Ha(D,He),ue.needsLights=Cx(D),ue.lightsStateVersion=Le,ue.needsLights&&(We.ambientLightColor.value=K.state.ambient,We.lightProbe.value=K.state.probe,We.directionalLights.value=K.state.directional,We.directionalLightShadows.value=K.state.directionalShadow,We.spotLights.value=K.state.spot,We.spotLightShadows.value=K.state.spotShadow,We.rectAreaLights.value=K.state.rectArea,We.ltc_1.value=K.state.rectAreaLTC1,We.ltc_2.value=K.state.rectAreaLTC2,We.pointLights.value=K.state.point,We.pointLightShadows.value=K.state.pointShadow,We.hemisphereLights.value=K.state.hemi,We.directionalShadowMap.value=K.state.directionalShadowMap,We.directionalShadowMatrix.value=K.state.directionalShadowMatrix,We.spotShadowMap.value=K.state.spotShadowMap,We.spotLightMatrix.value=K.state.spotLightMatrix,We.spotLightMap.value=K.state.spotLightMap,We.pointShadowMap.value=K.state.pointShadowMap,We.pointShadowMatrix.value=K.state.pointShadowMatrix),ue.currentProgram=Ke,ue.uniformsList=null,Ke}function Rr(D){if(D.uniformsList===null){const Q=D.currentProgram.getUniforms();D.uniformsList=Pl.seqWithValue(Q.seq,D.uniforms)}return D.uniformsList}function Ha(D,Q){const ae=ie.get(D);ae.outputColorSpace=Q.outputColorSpace,ae.batching=Q.batching,ae.batchingColor=Q.batchingColor,ae.instancing=Q.instancing,ae.instancingColor=Q.instancingColor,ae.instancingMorph=Q.instancingMorph,ae.skinning=Q.skinning,ae.morphTargets=Q.morphTargets,ae.morphNormals=Q.morphNormals,ae.morphColors=Q.morphColors,ae.morphTargetsCount=Q.morphTargetsCount,ae.numClippingPlanes=Q.numClippingPlanes,ae.numIntersection=Q.numClipIntersection,ae.vertexAlphas=Q.vertexAlphas,ae.vertexTangents=Q.vertexTangents,ae.toneMapping=Q.toneMapping}function yc(D,Q,ae,ue,K){Q.isScene!==!0&&(Q=L),Ae.resetTextureUnits();const Ee=Q.fog,Le=ue.isMeshStandardMaterial?Q.environment:null,He=T===null?b.outputColorSpace:T.isXRRenderTarget===!0?T.texture.colorSpace:bo,ke=(ue.isMeshStandardMaterial?C:R).get(ue.envMap||Le),qe=ue.vertexColors===!0&&!!ae.attributes.color&&ae.attributes.color.itemSize===4,Ke=!!ae.attributes.tangent&&(!!ue.normalMap||ue.anisotropy>0),We=!!ae.morphAttributes.position,nt=!!ae.morphAttributes.normal,dt=!!ae.morphAttributes.color;let zt=zs;ue.toneMapped&&(T===null||T.isXRRenderTarget===!0)&&(zt=b.toneMapping);const kt=ae.morphAttributes.position||ae.morphAttributes.normal||ae.morphAttributes.color,St=kt!==void 0?kt.length:0,Xe=ie.get(ue),Pt=S.state.lights;if(ne===!0&&(xe===!0||D!==P)){const gn=D===P&&ue.id===I;k.setState(ue,D,gn)}let ot=!1;ue.version===Xe.__version?(Xe.needsLights&&Xe.lightsStateVersion!==Pt.state.version||Xe.outputColorSpace!==He||K.isBatchedMesh&&Xe.batching===!1||!K.isBatchedMesh&&Xe.batching===!0||K.isBatchedMesh&&Xe.batchingColor===!0&&K.colorTexture===null||K.isBatchedMesh&&Xe.batchingColor===!1&&K.colorTexture!==null||K.isInstancedMesh&&Xe.instancing===!1||!K.isInstancedMesh&&Xe.instancing===!0||K.isSkinnedMesh&&Xe.skinning===!1||!K.isSkinnedMesh&&Xe.skinning===!0||K.isInstancedMesh&&Xe.instancingColor===!0&&K.instanceColor===null||K.isInstancedMesh&&Xe.instancingColor===!1&&K.instanceColor!==null||K.isInstancedMesh&&Xe.instancingMorph===!0&&K.morphTexture===null||K.isInstancedMesh&&Xe.instancingMorph===!1&&K.morphTexture!==null||Xe.envMap!==ke||ue.fog===!0&&Xe.fog!==Ee||Xe.numClippingPlanes!==void 0&&(Xe.numClippingPlanes!==k.numPlanes||Xe.numIntersection!==k.numIntersection)||Xe.vertexAlphas!==qe||Xe.vertexTangents!==Ke||Xe.morphTargets!==We||Xe.morphNormals!==nt||Xe.morphColors!==dt||Xe.toneMapping!==zt||Xe.morphTargetsCount!==St)&&(ot=!0):(ot=!0,Xe.__version=ue.version);let Hn=Xe.currentProgram;ot===!0&&(Hn=Zs(ue,Q,K));let Ir=!1,Vn=!1,Uo=!1;const Ft=Hn.getUniforms(),wn=Xe.uniforms;if(se.useProgram(Hn.program)&&(Ir=!0,Vn=!0,Uo=!0),ue.id!==I&&(I=ue.id,Vn=!0),Ir||P!==D){se.buffers.depth.getReversed()&&D.reversedDepth!==!0&&(D._reversedDepth=!0,D.updateProjectionMatrix()),Ft.setValue(w,"projectionMatrix",D.projectionMatrix),Ft.setValue(w,"viewMatrix",D.matrixWorldInverse);const Rn=Ft.map.cameraPosition;Rn!==void 0&&Rn.setValue(w,Te.setFromMatrixPosition(D.matrixWorld)),pe.logarithmicDepthBuffer&&Ft.setValue(w,"logDepthBufFC",2/(Math.log(D.far+1)/Math.LN2)),(ue.isMeshPhongMaterial||ue.isMeshToonMaterial||ue.isMeshLambertMaterial||ue.isMeshBasicMaterial||ue.isMeshStandardMaterial||ue.isShaderMaterial)&&Ft.setValue(w,"isOrthographic",D.isOrthographicCamera===!0),P!==D&&(P=D,Vn=!0,Uo=!0)}if(K.isSkinnedMesh){Ft.setOptional(w,K,"bindMatrix"),Ft.setOptional(w,K,"bindMatrixInverse");const gn=K.skeleton;gn&&(gn.boneTexture===null&&gn.computeBoneTexture(),Ft.setValue(w,"boneTexture",gn.boneTexture,Ae))}K.isBatchedMesh&&(Ft.setOptional(w,K,"batchingTexture"),Ft.setValue(w,"batchingTexture",K._matricesTexture,Ae),Ft.setOptional(w,K,"batchingIdTexture"),Ft.setValue(w,"batchingIdTexture",K._indirectTexture,Ae),Ft.setOptional(w,K,"batchingColorTexture"),K._colorsTexture!==null&&Ft.setValue(w,"batchingColorTexture",K._colorsTexture,Ae));const si=ae.morphAttributes;if((si.position!==void 0||si.normal!==void 0||si.color!==void 0)&&H.update(K,ae,Hn),(Vn||Xe.receiveShadow!==K.receiveShadow)&&(Xe.receiveShadow=K.receiveShadow,Ft.setValue(w,"receiveShadow",K.receiveShadow)),ue.isMeshGouraudMaterial&&ue.envMap!==null&&(wn.envMap.value=ke,wn.flipEnvMap.value=ke.isCubeTexture&&ke.isRenderTargetTexture===!1?-1:1),ue.isMeshStandardMaterial&&ue.envMap===null&&Q.environment!==null&&(wn.envMapIntensity.value=Q.environmentIntensity),wn.dfgLUT!==void 0&&(wn.dfgLUT.value=xC()),Vn&&(Ft.setValue(w,"toneMappingExposure",b.toneMappingExposure),Xe.needsLights&&bc(wn,Uo),Ee&&ue.fog===!0&&Ie.refreshFogUniforms(wn,Ee),Ie.refreshMaterialUniforms(wn,ue,ce,ee,S.state.transmissionRenderTarget[D.id]),Pl.upload(w,Rr(Xe),wn,Ae)),ue.isShaderMaterial&&ue.uniformsNeedUpdate===!0&&(Pl.upload(w,Rr(Xe),wn,Ae),ue.uniformsNeedUpdate=!1),ue.isSpriteMaterial&&Ft.setValue(w,"center",K.center),Ft.setValue(w,"modelViewMatrix",K.modelViewMatrix),Ft.setValue(w,"normalMatrix",K.normalMatrix),Ft.setValue(w,"modelMatrix",K.matrixWorld),ue.isShaderMaterial||ue.isRawShaderMaterial){const gn=ue.uniformsGroups;for(let Rn=0,Mc=gn.length;Rn<Mc;Rn++){const Js=gn[Rn];ve.update(Js,Hn),ve.bind(Js,Hn)}}return Hn}function bc(D,Q){D.ambientLightColor.needsUpdate=Q,D.lightProbe.needsUpdate=Q,D.directionalLights.needsUpdate=Q,D.directionalLightShadows.needsUpdate=Q,D.pointLights.needsUpdate=Q,D.pointLightShadows.needsUpdate=Q,D.spotLights.needsUpdate=Q,D.spotLightShadows.needsUpdate=Q,D.rectAreaLights.needsUpdate=Q,D.hemisphereLights.needsUpdate=Q}function Cx(D){return D.isMeshLambertMaterial||D.isMeshToonMaterial||D.isMeshPhongMaterial||D.isMeshStandardMaterial||D.isShadowMaterial||D.isShaderMaterial&&D.lights===!0}this.getActiveCubeFace=function(){return E},this.getActiveMipmapLevel=function(){return M},this.getRenderTarget=function(){return T},this.setRenderTargetTextures=function(D,Q,ae){const ue=ie.get(D);ue.__autoAllocateDepthBuffer=D.resolveDepthBuffer===!1,ue.__autoAllocateDepthBuffer===!1&&(ue.__useRenderToTexture=!1),ie.get(D.texture).__webglTexture=Q,ie.get(D.depthTexture).__webglTexture=ue.__autoAllocateDepthBuffer?void 0:ae,ue.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(D,Q){const ae=ie.get(D);ae.__webglFramebuffer=Q,ae.__useDefaultFramebuffer=Q===void 0};const Ex=w.createFramebuffer();this.setRenderTarget=function(D,Q=0,ae=0){T=D,E=Q,M=ae;let ue=!0,K=null,Ee=!1,Le=!1;if(D){const ke=ie.get(D);if(ke.__useDefaultFramebuffer!==void 0)se.bindFramebuffer(w.FRAMEBUFFER,null),ue=!1;else if(ke.__webglFramebuffer===void 0)Ae.setupRenderTarget(D);else if(ke.__hasExternalTextures)Ae.rebindTextures(D,ie.get(D.texture).__webglTexture,ie.get(D.depthTexture).__webglTexture);else if(D.depthBuffer){const We=D.depthTexture;if(ke.__boundDepthTexture!==We){if(We!==null&&ie.has(We)&&(D.width!==We.image.width||D.height!==We.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");Ae.setupDepthRenderbuffer(D)}}const qe=D.texture;(qe.isData3DTexture||qe.isDataArrayTexture||qe.isCompressedArrayTexture)&&(Le=!0);const Ke=ie.get(D).__webglFramebuffer;D.isWebGLCubeRenderTarget?(Array.isArray(Ke[Q])?K=Ke[Q][ae]:K=Ke[Q],Ee=!0):D.samples>0&&Ae.useMultisampledRTT(D)===!1?K=ie.get(D).__webglMultisampledFramebuffer:Array.isArray(Ke)?K=Ke[ae]:K=Ke,B.copy(D.viewport),N.copy(D.scissor),G=D.scissorTest}else B.copy(Fe).multiplyScalar(ce).floor(),N.copy(Oe).multiplyScalar(ce).floor(),G=Ne;if(ae!==0&&(K=Ex),se.bindFramebuffer(w.FRAMEBUFFER,K)&&ue&&se.drawBuffers(D,K),se.viewport(B),se.scissor(N),se.setScissorTest(G),Ee){const ke=ie.get(D.texture);w.framebufferTexture2D(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_CUBE_MAP_POSITIVE_X+Q,ke.__webglTexture,ae)}else if(Le){const ke=Q;for(let qe=0;qe<D.textures.length;qe++){const Ke=ie.get(D.textures[qe]);w.framebufferTextureLayer(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0+qe,Ke.__webglTexture,ae,ke)}}else if(D!==null&&ae!==0){const ke=ie.get(D.texture);w.framebufferTexture2D(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,ke.__webglTexture,ae)}I=-1},this.readRenderTargetPixels=function(D,Q,ae,ue,K,Ee,Le,He=0){if(!(D&&D.isWebGLRenderTarget)){Wt("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let ke=ie.get(D).__webglFramebuffer;if(D.isWebGLCubeRenderTarget&&Le!==void 0&&(ke=ke[Le]),ke){se.bindFramebuffer(w.FRAMEBUFFER,ke);try{const qe=D.textures[He],Ke=qe.format,We=qe.type;if(!pe.textureFormatReadable(Ke)){Wt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!pe.textureTypeReadable(We)){Wt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}Q>=0&&Q<=D.width-ue&&ae>=0&&ae<=D.height-K&&(D.textures.length>1&&w.readBuffer(w.COLOR_ATTACHMENT0+He),w.readPixels(Q,ae,ue,K,Me.convert(Ke),Me.convert(We),Ee))}finally{const qe=T!==null?ie.get(T).__webglFramebuffer:null;se.bindFramebuffer(w.FRAMEBUFFER,qe)}}},this.readRenderTargetPixelsAsync=async function(D,Q,ae,ue,K,Ee,Le,He=0){if(!(D&&D.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let ke=ie.get(D).__webglFramebuffer;if(D.isWebGLCubeRenderTarget&&Le!==void 0&&(ke=ke[Le]),ke)if(Q>=0&&Q<=D.width-ue&&ae>=0&&ae<=D.height-K){se.bindFramebuffer(w.FRAMEBUFFER,ke);const qe=D.textures[He],Ke=qe.format,We=qe.type;if(!pe.textureFormatReadable(Ke))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!pe.textureTypeReadable(We))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const nt=w.createBuffer();w.bindBuffer(w.PIXEL_PACK_BUFFER,nt),w.bufferData(w.PIXEL_PACK_BUFFER,Ee.byteLength,w.STREAM_READ),D.textures.length>1&&w.readBuffer(w.COLOR_ATTACHMENT0+He),w.readPixels(Q,ae,ue,K,Me.convert(Ke),Me.convert(We),0);const dt=T!==null?ie.get(T).__webglFramebuffer:null;se.bindFramebuffer(w.FRAMEBUFFER,dt);const zt=w.fenceSync(w.SYNC_GPU_COMMANDS_COMPLETE,0);return w.flush(),await qS(w,zt,4),w.bindBuffer(w.PIXEL_PACK_BUFFER,nt),w.getBufferSubData(w.PIXEL_PACK_BUFFER,0,Ee),w.deleteBuffer(nt),w.deleteSync(zt),Ee}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(D,Q=null,ae=0){const ue=Math.pow(2,-ae),K=Math.floor(D.image.width*ue),Ee=Math.floor(D.image.height*ue),Le=Q!==null?Q.x:0,He=Q!==null?Q.y:0;Ae.setTexture2D(D,0),w.copyTexSubImage2D(w.TEXTURE_2D,ae,0,0,Le,He,K,Ee),se.unbindTexture()};const wx=w.createFramebuffer(),Rx=w.createFramebuffer();this.copyTextureToTexture=function(D,Q,ae=null,ue=null,K=0,Ee=null){Ee===null&&(K!==0?(ya("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."),Ee=K,K=0):Ee=0);let Le,He,ke,qe,Ke,We,nt,dt,zt;const kt=D.isCompressedTexture?D.mipmaps[Ee]:D.image;if(ae!==null)Le=ae.max.x-ae.min.x,He=ae.max.y-ae.min.y,ke=ae.isBox3?ae.max.z-ae.min.z:1,qe=ae.min.x,Ke=ae.min.y,We=ae.isBox3?ae.min.z:0;else{const si=Math.pow(2,-K);Le=Math.floor(kt.width*si),He=Math.floor(kt.height*si),D.isDataArrayTexture?ke=kt.depth:D.isData3DTexture?ke=Math.floor(kt.depth*si):ke=1,qe=0,Ke=0,We=0}ue!==null?(nt=ue.x,dt=ue.y,zt=ue.z):(nt=0,dt=0,zt=0);const St=Me.convert(Q.format),Xe=Me.convert(Q.type);let Pt;Q.isData3DTexture?(Ae.setTexture3D(Q,0),Pt=w.TEXTURE_3D):Q.isDataArrayTexture||Q.isCompressedArrayTexture?(Ae.setTexture2DArray(Q,0),Pt=w.TEXTURE_2D_ARRAY):(Ae.setTexture2D(Q,0),Pt=w.TEXTURE_2D),w.pixelStorei(w.UNPACK_FLIP_Y_WEBGL,Q.flipY),w.pixelStorei(w.UNPACK_PREMULTIPLY_ALPHA_WEBGL,Q.premultiplyAlpha),w.pixelStorei(w.UNPACK_ALIGNMENT,Q.unpackAlignment);const ot=w.getParameter(w.UNPACK_ROW_LENGTH),Hn=w.getParameter(w.UNPACK_IMAGE_HEIGHT),Ir=w.getParameter(w.UNPACK_SKIP_PIXELS),Vn=w.getParameter(w.UNPACK_SKIP_ROWS),Uo=w.getParameter(w.UNPACK_SKIP_IMAGES);w.pixelStorei(w.UNPACK_ROW_LENGTH,kt.width),w.pixelStorei(w.UNPACK_IMAGE_HEIGHT,kt.height),w.pixelStorei(w.UNPACK_SKIP_PIXELS,qe),w.pixelStorei(w.UNPACK_SKIP_ROWS,Ke),w.pixelStorei(w.UNPACK_SKIP_IMAGES,We);const Ft=D.isDataArrayTexture||D.isData3DTexture,wn=Q.isDataArrayTexture||Q.isData3DTexture;if(D.isDepthTexture){const si=ie.get(D),gn=ie.get(Q),Rn=ie.get(si.__renderTarget),Mc=ie.get(gn.__renderTarget);se.bindFramebuffer(w.READ_FRAMEBUFFER,Rn.__webglFramebuffer),se.bindFramebuffer(w.DRAW_FRAMEBUFFER,Mc.__webglFramebuffer);for(let Js=0;Js<ke;Js++)Ft&&(w.framebufferTextureLayer(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,ie.get(D).__webglTexture,K,We+Js),w.framebufferTextureLayer(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,ie.get(Q).__webglTexture,Ee,zt+Js)),w.blitFramebuffer(qe,Ke,Le,He,nt,dt,Le,He,w.DEPTH_BUFFER_BIT,w.NEAREST);se.bindFramebuffer(w.READ_FRAMEBUFFER,null),se.bindFramebuffer(w.DRAW_FRAMEBUFFER,null)}else if(K!==0||D.isRenderTargetTexture||ie.has(D)){const si=ie.get(D),gn=ie.get(Q);se.bindFramebuffer(w.READ_FRAMEBUFFER,wx),se.bindFramebuffer(w.DRAW_FRAMEBUFFER,Rx);for(let Rn=0;Rn<ke;Rn++)Ft?w.framebufferTextureLayer(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,si.__webglTexture,K,We+Rn):w.framebufferTexture2D(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,si.__webglTexture,K),wn?w.framebufferTextureLayer(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,gn.__webglTexture,Ee,zt+Rn):w.framebufferTexture2D(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,gn.__webglTexture,Ee),K!==0?w.blitFramebuffer(qe,Ke,Le,He,nt,dt,Le,He,w.COLOR_BUFFER_BIT,w.NEAREST):wn?w.copyTexSubImage3D(Pt,Ee,nt,dt,zt+Rn,qe,Ke,Le,He):w.copyTexSubImage2D(Pt,Ee,nt,dt,qe,Ke,Le,He);se.bindFramebuffer(w.READ_FRAMEBUFFER,null),se.bindFramebuffer(w.DRAW_FRAMEBUFFER,null)}else wn?D.isDataTexture||D.isData3DTexture?w.texSubImage3D(Pt,Ee,nt,dt,zt,Le,He,ke,St,Xe,kt.data):Q.isCompressedArrayTexture?w.compressedTexSubImage3D(Pt,Ee,nt,dt,zt,Le,He,ke,St,kt.data):w.texSubImage3D(Pt,Ee,nt,dt,zt,Le,He,ke,St,Xe,kt):D.isDataTexture?w.texSubImage2D(w.TEXTURE_2D,Ee,nt,dt,Le,He,St,Xe,kt.data):D.isCompressedTexture?w.compressedTexSubImage2D(w.TEXTURE_2D,Ee,nt,dt,kt.width,kt.height,St,kt.data):w.texSubImage2D(w.TEXTURE_2D,Ee,nt,dt,Le,He,St,Xe,kt);w.pixelStorei(w.UNPACK_ROW_LENGTH,ot),w.pixelStorei(w.UNPACK_IMAGE_HEIGHT,Hn),w.pixelStorei(w.UNPACK_SKIP_PIXELS,Ir),w.pixelStorei(w.UNPACK_SKIP_ROWS,Vn),w.pixelStorei(w.UNPACK_SKIP_IMAGES,Uo),Ee===0&&Q.generateMipmaps&&w.generateMipmap(Pt),se.unbindTexture()},this.initRenderTarget=function(D){ie.get(D).__webglFramebuffer===void 0&&Ae.setupRenderTarget(D)},this.initTexture=function(D){D.isCubeTexture?Ae.setTextureCube(D,0):D.isData3DTexture?Ae.setTexture3D(D,0):D.isDataArrayTexture||D.isCompressedArrayTexture?Ae.setTexture2DArray(D,0):Ae.setTexture2D(D,0),se.unbindTexture()},this.resetState=function(){E=0,M=0,T=null,se.reset(),O.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Oi}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const t=this.getContext();t.drawingBufferColorSpace=lt._getDrawingBufferColorSpace(e),t.unpackColorSpace=lt._getUnpackColorSpace()}}class Fs{static idGen=0;constructor(e,t){let n,s;this.promise=new Promise((c,u)=>{n=c,s=u});const r=n.bind(this),o=s.bind(this),a=(...c)=>{r(...c)},l=c=>{o(c)};e(a.bind(this),l.bind(this)),this.abortHandler=t,this.id=Fs.idGen++}then(e){return new Fs((t,n)=>{this.promise=this.promise.then((...s)=>{const r=e(...s);r instanceof Promise||r instanceof Fs?r.then((...o)=>{t(...o)}):t(r)}).catch(s=>{n(s)})},this.abortHandler)}catch(e){return new Fs(t=>{this.promise=this.promise.then((...n)=>{t(...n)}).catch(e)},this.abortHandler)}abort(e){this.abortHandler&&this.abortHandler(e)}}class _g extends Error{constructor(e){super(e)}}(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){i[0]=t;const n=e[0];let s=n>>16&32768,r=n>>12&2047;const o=n>>23&255;return o<103?s:o>142?(s|=31744,s|=(o==255?0:1)&&n&8388607,s):o<113?(r|=2048,s|=(r>>114-o)+(r>>113-o&1),s):(s|=o-112<<10|r>>1,s+=r&1,s)}})();const pu=(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){return i[0]=t,e[0]}})(),vC=function(i,e){return i[e]+(i[e+1]<<8)+(i[e+2]<<16)+(i[e+3]<<24)},_c=function(i,e,t=!0,n){const s=new AbortController,r=s.signal;let o=!1;const a=u=>{s.abort(u),o=!0};let l=!1;const c=(u,f,h,d)=>{e&&!l&&(e(u,f,h,d),u===100&&(l=!0))};return new Fs((u,f)=>{const h={signal:r};n&&(h.headers=n),fetch(i,h).then(async d=>{if(!d.ok){const S=await d.text();f(new Error(`Fetch failed: ${d.status} ${d.statusText} ${S}`));return}const x=d.body.getReader();let p=0,g=d.headers.get("Content-Length"),m=g?parseInt(g):void 0;const _=[];for(;!o;)try{const{value:S,done:A}=await x.read();if(A){if(c(100,"100%",S,m),t){const v=new Blob(_).arrayBuffer();u(v)}else u();break}p+=S.length;let y,b;m!==void 0&&(y=p/m*100,b=`${y.toFixed(2)}%`),t&&_.push(S),c(y,b,S,m)}catch(S){f(S);return}}).catch(d=>{f(new _g(d))})},a)},Rt=function(i,e,t){return Math.max(Math.min(i,t),e)},qr=function(){return performance.now()/1e3},Zr=i=>{if(i.geometry&&(i.geometry.dispose(),i.geometry=null),i.material&&(i.material.dispose(),i.material=null),i.children)for(let e of i.children)Zr(e)},jn=(i,e)=>new Promise(t=>{window.setTimeout(()=>{t(i?i():void 0)},e?1:50)}),uo=(i=0)=>{let e=0;if(i===1)e=9;else if(i===2)e=24;else if(i===3)e=45;else if(i>3)throw new Error("getSphericalHarmonicsComponentCountForDegree() -> Invalid spherical harmonics degree");return e},Eh=()=>{let i,e;return{promise:new Promise((n,s)=>{i=n,e=s}),resolve:i,reject:e}},mu=i=>{let e,t;return i||(i=()=>{}),{promise:new Fs((s,r)=>{e=s,t=r},i),resolve:e,reject:t}};class SC{constructor(e,t,n){this.major=e,this.minor=t,this.patch=n}toString(){return`${this.major}_${this.minor}_${this.patch}`}}function wh(){const i=navigator.userAgent;return i.indexOf("iPhone")>0||i.indexOf("iPad")>0}function vg(){if(wh()){const i=navigator.userAgent.match(/OS (\d+)_(\d+)_?(\d+)?/);return new SC(parseInt(i[1]||0,10),parseInt(i[2]||0,10),parseInt(i[3]||0,10))}else return null}const AC=14;class we{static OFFSET={X:0,Y:1,Z:2,SCALE0:3,SCALE1:4,SCALE2:5,ROTATION0:6,ROTATION1:7,ROTATION2:8,ROTATION3:9,FDC0:10,FDC1:11,FDC2:12,OPACITY:13,FRC0:14,FRC1:15,FRC2:16,FRC3:17,FRC4:18,FRC5:19,FRC6:20,FRC7:21,FRC8:22,FRC9:23,FRC10:24,FRC11:25,FRC12:26,FRC13:27,FRC14:28,FRC15:29,FRC16:30,FRC17:31,FRC18:32,FRC19:33,FRC20:34,FRC21:35,FRC22:36,FRC23:37};constructor(e=0){this.sphericalHarmonicsDegree=e,this.sphericalHarmonicsCount=uo(this.sphericalHarmonicsDegree),this.componentCount=this.sphericalHarmonicsCount+AC,this.defaultSphericalHarmonics=new Array(this.sphericalHarmonicsCount).fill(0),this.splats=[],this.splatCount=0}static createSplat(e=0){const t=[0,0,0,1,1,1,1,0,0,0,0,0,0,0];let n=uo(e);for(let s=0;s<n;s++)t.push(0);return t}addSplat(e){this.splats.push(e),this.splatCount++}getSplat(e){return this.splats[e]}addDefaultSplat(){const e=we.createSplat(this.sphericalHarmonicsDegree);return this.addSplat(e),e}addSplatFromComonents(e,t,n,s,r,o,a,l,c,u,f,h,d,x,...p){const g=[e,t,n,s,r,o,a,l,c,u,f,h,d,x,...this.defaultSphericalHarmonics];for(let m=0;m<p.length&&m<this.sphericalHarmonicsCount;m++)g[m]=p[m];return this.addSplat(g),g}addSplatFromArray(e,t){const n=e.splats[t],s=we.createSplat(this.sphericalHarmonicsDegree);for(let r=0;r<this.componentCount&&r<n.length;r++)s[r]=n[r];this.addSplat(s)}}class gt{static DefaultSplatSortDistanceMapPrecision=16;static MemoryPageSize=65536;static BytesPerFloat=4;static BytesPerInt=4;static MaxScenes=32;static ProgressiveLoadSectionSize=262144;static ProgressiveLoadSectionDelayDuration=15;static SphericalHarmonics8BitCompressionRange=3}const yC=gt.SphericalHarmonics8BitCompressionRange,ws=yC/2,Zt=ba.toHalfFloat.bind(ba),Rh=ba.fromHalfFloat.bind(ba),wt=(i,e,t=!1,n,s)=>{if(e===0)return i;if(e===1||e===2&&!t)return ba.fromHalfFloat(i);if(e===2)return Ih(i,n,s)},sa=(i,e,t)=>{i=Rt(i,e,t);const n=t-e;return Rt(Math.floor((i-e)/n*255),0,255)},Ih=(i,e,t)=>{const n=t-e;return i/255*n+e},Sg=(i,e,t)=>sa(Rh(i,e,t)),bC=(i,e,t)=>Zt(Ih(i,e,t)),ut=(i,e,t,n=!1)=>t===0?i.getFloat32(e*4,!0):t===1||t===2&&!n?i.getUint16(e*2,!0):i.getUint8(e,!0),MC=(function(){const i=e=>e;return function(e,t,n,s=!1){if(t===n)return e;let r=i;return t===2&&s?n===1?r=bC:n==0&&(r=Ih):t===2||t===1?n===0?r=Rh:n==2&&(s?r=Sg:r=i):t===0&&(n===1?r=Zt:n==2&&(s?r=sa:r=Zt)),r(e)}})(),Yr=(i,e,t,n,s=0)=>{const r=new Uint8Array(i,e),o=new Uint8Array(t,n);for(let a=0;a<s;a++)o[a]=r[a]};class j{static CurrentMajorVersion=0;static CurrentMinorVersion=1;static CenterComponentCount=3;static ScaleComponentCount=3;static RotationComponentCount=4;static ColorComponentCount=4;static CovarianceComponentCount=6;static SplatScaleOffsetFloat=3;static SplatRotationOffsetFloat=6;static CompressionLevels={0:{BytesPerCenter:12,BytesPerScale:12,BytesPerRotation:16,BytesPerColor:4,ScaleOffsetBytes:12,RotationffsetBytes:24,ColorOffsetBytes:40,SphericalHarmonicsOffsetBytes:44,ScaleRange:1,BytesPerSphericalHarmonicsComponent:4,SphericalHarmonicsOffsetFloat:11,SphericalHarmonicsDegrees:{0:{BytesPerSplat:44},1:{BytesPerSplat:80},2:{BytesPerSplat:140}}},1:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:2,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:42},2:{BytesPerSplat:72}}},2:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:1,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:33},2:{BytesPerSplat:48}}}};static CovarianceSizeFloats=6;static HeaderSizeBytes=4096;static SectionHeaderSizeBytes=1024;static BucketStorageSizeBytes=12;static BucketStorageSizeFloats=3;static BucketBlockSize=5;static BucketSize=256;constructor(e,t=!0){this.constructFromBuffer(e,t)}getSplatCount(){return this.splatCount}getMaxSplatCount(){return this.maxSplatCount}getMinSphericalHarmonicsDegree(){let e=0;for(let t=0;t<this.sections.length;t++){const n=this.sections[t];(t===0||n.sphericalHarmonicsDegree<e)&&(e=n.sphericalHarmonicsDegree)}return e}getBucketIndex(e,t){let n;const s=e.fullBucketCount*e.bucketSize;if(t<s)n=Math.floor(t/e.bucketSize);else{let r=s;n=e.fullBucketCount;let o=0;for(;r<e.splatCount;){let a=e.partiallyFilledBucketLengths[o];if(t>=r&&t<r+a)break;r+=a,n++,o++}}return n}getSplatCenter(e,t,n){const s=this.globalSplatIndexToSectionMap[e],r=this.sections[s],o=e-r.splatCountOffset,a=r.bytesPerSplat*o,l=new DataView(this.bufferData,r.dataBase+a),c=ut(l,0,this.compressionLevel),u=ut(l,1,this.compressionLevel),f=ut(l,2,this.compressionLevel);if(this.compressionLevel>=1){const d=this.getBucketIndex(r,o)*j.BucketStorageSizeFloats,x=r.compressionScaleFactor,p=r.compressionScaleRange;t.x=(c-p)*x+r.bucketArray[d],t.y=(u-p)*x+r.bucketArray[d+1],t.z=(f-p)*x+r.bucketArray[d+2]}else t.x=c,t.y=u,t.z=f;n&&t.applyMatrix4(n)}getSplatScaleAndRotation=(function(){const e=new Ye,t=new Ye,n=new Ye,s=new F,r=new F,o=new Mt;return function(a,l,c,u,f){const h=this.globalSplatIndexToSectionMap[a],d=this.sections[h],x=a-d.splatCountOffset,p=d.bytesPerSplat*x+j.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,g=new DataView(this.bufferData,d.dataBase+p);r.set(wt(ut(g,0,this.compressionLevel),this.compressionLevel),wt(ut(g,1,this.compressionLevel),this.compressionLevel),wt(ut(g,2,this.compressionLevel),this.compressionLevel)),f&&(f.x!==void 0&&(r.x=f.x),f.y!==void 0&&(r.y=f.y),f.z!==void 0&&(r.z=f.z)),o.set(wt(ut(g,4,this.compressionLevel),this.compressionLevel),wt(ut(g,5,this.compressionLevel),this.compressionLevel),wt(ut(g,6,this.compressionLevel),this.compressionLevel),wt(ut(g,3,this.compressionLevel),this.compressionLevel)),u?(e.makeScale(r.x,r.y,r.z),t.makeRotationFromQuaternion(o),n.copy(e).multiply(t).multiply(u),n.decompose(s,c,l)):(l.copy(r),c.copy(o))}})();getSplatColor(e,t){const n=this.globalSplatIndexToSectionMap[e],s=this.sections[n],r=e-s.splatCountOffset,o=s.bytesPerSplat*r+j.CompressionLevels[this.compressionLevel].ColorOffsetBytes,a=new Uint8Array(this.bufferData,s.dataBase+o,4);t.set(a[0],a[1],a[2],a[3])}fillSplatCenterArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);const a=new F;for(let l=n;l<=s;l++){const c=this.globalSplatIndexToSectionMap[l],u=this.sections[c],f=l-u.splatCountOffset,h=(l-n+r)*j.CenterComponentCount,d=u.bytesPerSplat*f,x=new DataView(this.bufferData,u.dataBase+d),p=ut(x,0,this.compressionLevel),g=ut(x,1,this.compressionLevel),m=ut(x,2,this.compressionLevel);if(this.compressionLevel>=1){const S=this.getBucketIndex(u,f)*j.BucketStorageSizeFloats,A=u.compressionScaleFactor,y=u.compressionScaleRange;a.x=(p-y)*A+u.bucketArray[S],a.y=(g-y)*A+u.bucketArray[S+1],a.z=(m-y)*A+u.bucketArray[S+2]}else a.x=p,a.y=g,a.z=m;t&&a.applyMatrix4(t),e[h]=a.x,e[h+1]=a.y,e[h+2]=a.z}}fillSplatScaleRotationArray=(function(){const e=new Ye,t=new Ye,n=new Ye,s=new F,r=new Mt,o=new F,a=l=>{const c=l.w<0?-1:1;l.x*=c,l.y*=c,l.z*=c,l.w*=c};return function(l,c,u,f,h,d,x,p){const g=this.splatCount;f=f||0,h=h||g-1,d===void 0&&(d=f);const m=(_,S)=>MC(_,S,x);for(let _=f;_<=h;_++){const S=this.globalSplatIndexToSectionMap[_],A=this.sections[S],y=_-A.splatCountOffset,b=A.bytesPerSplat*y+j.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,v=(_-f+d)*j.ScaleComponentCount,E=(_-f+d)*j.RotationComponentCount,M=new DataView(this.bufferData,A.dataBase+b),T=p&&p.x!==void 0?p.x:ut(M,0,this.compressionLevel),I=p&&p.y!==void 0?p.y:ut(M,1,this.compressionLevel),P=p&&p.z!==void 0?p.z:ut(M,2,this.compressionLevel),B=ut(M,3,this.compressionLevel),N=ut(M,4,this.compressionLevel),G=ut(M,5,this.compressionLevel),V=ut(M,6,this.compressionLevel);s.set(wt(T,this.compressionLevel),wt(I,this.compressionLevel),wt(P,this.compressionLevel)),r.set(wt(N,this.compressionLevel),wt(G,this.compressionLevel),wt(V,this.compressionLevel),wt(B,this.compressionLevel)).normalize(),u&&(o.set(0,0,0),e.makeScale(s.x,s.y,s.z),t.makeRotationFromQuaternion(r),n.identity().premultiply(e).premultiply(t),n.premultiply(u),n.decompose(o,r,s),r.normalize()),a(r),l&&(l[v]=m(s.x,0),l[v+1]=m(s.y,0),l[v+2]=m(s.z,0)),c&&(c[E]=m(r.x,0),c[E+1]=m(r.y,0),c[E+2]=m(r.z,0),c[E+3]=m(r.w,0))}}})();static computeCovariance=(function(){const e=new Ye,t=new Qe,n=new Qe,s=new Qe,r=new Qe,o=new Qe,a=new Qe;return function(l,c,u,f,h=0,d){e.makeScale(l.x,l.y,l.z),t.setFromMatrix4(e),e.makeRotationFromQuaternion(c),n.setFromMatrix4(e),s.copy(n).multiply(t),r.copy(s).transpose().premultiply(s),u&&(o.setFromMatrix4(u),a.copy(o).transpose(),r.multiply(a),r.premultiply(o)),d>=1?(f[h]=Zt(r.elements[0]),f[h+1]=Zt(r.elements[3]),f[h+2]=Zt(r.elements[6]),f[h+3]=Zt(r.elements[4]),f[h+4]=Zt(r.elements[7]),f[h+5]=Zt(r.elements[8])):(f[h]=r.elements[0],f[h+1]=r.elements[3],f[h+2]=r.elements[6],f[h+3]=r.elements[4],f[h+4]=r.elements[7],f[h+5]=r.elements[8])}})();fillSplatCovarianceArray(e,t,n,s,r,o){const a=this.splatCount,l=new F,c=new Mt;n=n||0,s=s||a-1,r===void 0&&(r=n);for(let u=n;u<=s;u++){const f=this.globalSplatIndexToSectionMap[u],h=this.sections[f],d=u-h.splatCountOffset,x=(u-n+r)*j.CovarianceComponentCount,p=h.bytesPerSplat*d+j.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,g=new DataView(this.bufferData,h.dataBase+p);l.set(wt(ut(g,0,this.compressionLevel),this.compressionLevel),wt(ut(g,1,this.compressionLevel),this.compressionLevel),wt(ut(g,2,this.compressionLevel),this.compressionLevel)),c.set(wt(ut(g,4,this.compressionLevel),this.compressionLevel),wt(ut(g,5,this.compressionLevel),this.compressionLevel),wt(ut(g,6,this.compressionLevel),this.compressionLevel),wt(ut(g,3,this.compressionLevel),this.compressionLevel)),j.computeCovariance(l,c,t,e,x,o)}}fillSplatColorArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);for(let a=n;a<=s;a++){const l=this.globalSplatIndexToSectionMap[a],c=this.sections[l],u=a-c.splatCountOffset,f=(a-n+r)*j.ColorComponentCount,h=c.bytesPerSplat*u+j.CompressionLevels[this.compressionLevel].ColorOffsetBytes,d=new Uint8Array(this.bufferData,c.dataBase+h);let x=d[3];x=x>=t?x:0,e[f]=d[0],e[f+1]=d[1],e[f+2]=d[2],e[f+3]=x}}fillSphericalHarmonicsArray=(function(){for(let N=0;N<15;N++)new F;const e=new Qe,t=new Ye,n=new F,s=new F,r=new Mt,o=[],a=[],l=[],c=[],u=[],f=[],h=[],d=[],x=[],p=[],g=[],m=[],_=[],S=[],A=[],y=[],b=[],v=[],E=N=>N,M=(N,G,V,q)=>{N[0]=G,N[1]=V,N[2]=q},T=(N,G,V,q,X)=>{N[0]=ut(G,q,X,!0),N[1]=ut(G,q+V,X,!0),N[2]=ut(G,q+V+V,X,!0)},I=(N,G)=>{G[0]=N[0],G[1]=N[1],G[2]=N[2]},P=(N,G,V,q)=>{G[V]=q(N[0]),G[V+1]=q(N[1]),G[V+2]=q(N[2])},B=(N,G,V,q,X)=>(G[0]=wt(N[0],V,!0,q,X),G[1]=wt(N[1],V,!0,q,X),G[2]=wt(N[2],V,!0,q,X),G);return function(N,G,V,q,X,ee,ce){const be=this.splatCount;q=q||0,X=X||be-1,ee===void 0&&(ee=q),V&&G>=1&&(t.copy(V),t.decompose(n,r,s),r.normalize(),t.makeRotationFromQuaternion(r),e.setFromMatrix4(t),M(o,e.elements[4],-e.elements[7],e.elements[1]),M(a,-e.elements[5],e.elements[8],-e.elements[2]),M(l,e.elements[3],-e.elements[6],e.elements[0]));const Re=Oe=>Sg(Oe,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff),Fe=Oe=>sa(Oe,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff);for(let Oe=q;Oe<=X;Oe++){const Ne=this.globalSplatIndexToSectionMap[Oe],J=this.sections[Ne];G=Math.min(G,J.sphericalHarmonicsDegree);const ne=uo(G),xe=Oe-J.splatCountOffset,Be=J.bytesPerSplat*xe+j.CompressionLevels[this.compressionLevel].SphericalHarmonicsOffsetBytes,Te=new DataView(this.bufferData,J.dataBase+Be),Ve=(Oe-q+ee)*ne;let L=V?0:this.compressionLevel,U=E;L!==ce&&(L===1?ce===0?U=Rh:ce==2&&(U=Re):L===0&&(ce===1?U=Zt:ce==2&&(U=Fe)));const Y=this.minSphericalHarmonicsCoeff,w=this.maxSphericalHarmonicsCoeff;G>=1&&(T(x,Te,3,0,this.compressionLevel),T(p,Te,3,1,this.compressionLevel),T(g,Te,3,2,this.compressionLevel),V?(B(x,x,this.compressionLevel,Y,w),B(p,p,this.compressionLevel,Y,w),B(g,g,this.compressionLevel,Y,w),j.rotateSphericalHarmonics3(x,p,g,o,a,l,S,A,y)):(I(x,S),I(p,A),I(g,y)),P(S,N,Ve,U),P(A,N,Ve+3,U),P(y,N,Ve+6,U),G>=2&&(T(x,Te,5,9,this.compressionLevel),T(p,Te,5,10,this.compressionLevel),T(g,Te,5,11,this.compressionLevel),T(m,Te,5,12,this.compressionLevel),T(_,Te,5,13,this.compressionLevel),V?(B(x,x,this.compressionLevel,Y,w),B(p,p,this.compressionLevel,Y,w),B(g,g,this.compressionLevel,Y,w),B(m,m,this.compressionLevel,Y,w),B(_,_,this.compressionLevel,Y,w),j.rotateSphericalHarmonics5(x,p,g,m,_,o,a,l,c,u,f,h,d,S,A,y,b,v)):(I(x,S),I(p,A),I(g,y),I(m,b),I(_,v)),P(S,N,Ve+9,U),P(A,N,Ve+12,U),P(y,N,Ve+15,U),P(b,N,Ve+18,U),P(v,N,Ve+21,U)))}}})();static dot3=(e,t,n,s,r)=>{r[0]=r[1]=r[2]=0;const o=s[0],a=s[1],l=s[2];j.addInto3(e[0]*o,e[1]*o,e[2]*o,r),j.addInto3(t[0]*a,t[1]*a,t[2]*a,r),j.addInto3(n[0]*l,n[1]*l,n[2]*l,r)};static addInto3=(e,t,n,s)=>{s[0]=s[0]+e,s[1]=s[1]+t,s[2]=s[2]+n};static dot5=(e,t,n,s,r,o,a)=>{a[0]=a[1]=a[2]=0;const l=o[0],c=o[1],u=o[2],f=o[3],h=o[4];j.addInto3(e[0]*l,e[1]*l,e[2]*l,a),j.addInto3(t[0]*c,t[1]*c,t[2]*c,a),j.addInto3(n[0]*u,n[1]*u,n[2]*u,a),j.addInto3(s[0]*f,s[1]*f,s[2]*f,a),j.addInto3(r[0]*h,r[1]*h,r[2]*h,a)};static rotateSphericalHarmonics3=(e,t,n,s,r,o,a,l,c)=>{j.dot3(e,t,n,s,a),j.dot3(e,t,n,r,l),j.dot3(e,t,n,o,c)};static rotateSphericalHarmonics5=(e,t,n,s,r,o,a,l,c,u,f,h,d,x,p,g,m,_)=>{const S=Math.sqrt(.25),A=Math.sqrt(3/4),y=Math.sqrt(1/3),b=Math.sqrt(4/3),v=Math.sqrt(1/12);c[0]=S*(l[2]*o[0]+l[0]*o[2]+(o[2]*l[0]+o[0]*l[2])),c[1]=l[1]*o[0]+o[1]*l[0],c[2]=A*(l[1]*o[1]+o[1]*l[1]),c[3]=l[1]*o[2]+o[1]*l[2],c[4]=S*(l[2]*o[2]-l[0]*o[0]+(o[2]*l[2]-o[0]*l[0])),j.dot5(e,t,n,s,r,c,x),u[0]=S*(a[2]*o[0]+a[0]*o[2]+(o[2]*a[0]+o[0]*a[2])),u[1]=a[1]*o[0]+o[1]*a[0],u[2]=A*(a[1]*o[1]+o[1]*a[1]),u[3]=a[1]*o[2]+o[1]*a[2],u[4]=S*(a[2]*o[2]-a[0]*o[0]+(o[2]*a[2]-o[0]*a[0])),j.dot5(e,t,n,s,r,u,p),f[0]=y*(a[2]*a[0]+a[0]*a[2])+-v*(l[2]*l[0]+l[0]*l[2]+(o[2]*o[0]+o[0]*o[2])),f[1]=b*a[1]*a[0]+-y*(l[1]*l[0]+o[1]*o[0]),f[2]=a[1]*a[1]+-S*(l[1]*l[1]+o[1]*o[1]),f[3]=b*a[1]*a[2]+-y*(l[1]*l[2]+o[1]*o[2]),f[4]=y*(a[2]*a[2]-a[0]*a[0])+-v*(l[2]*l[2]-l[0]*l[0]+(o[2]*o[2]-o[0]*o[0])),j.dot5(e,t,n,s,r,f,g),h[0]=S*(a[2]*l[0]+a[0]*l[2]+(l[2]*a[0]+l[0]*a[2])),h[1]=a[1]*l[0]+l[1]*a[0],h[2]=A*(a[1]*l[1]+l[1]*a[1]),h[3]=a[1]*l[2]+l[1]*a[2],h[4]=S*(a[2]*l[2]-a[0]*l[0]+(l[2]*a[2]-l[0]*a[0])),j.dot5(e,t,n,s,r,h,m),d[0]=S*(l[2]*l[0]+l[0]*l[2]-(o[2]*o[0]+o[0]*o[2])),d[1]=l[1]*l[0]-o[1]*o[0],d[2]=A*(l[1]*l[1]-o[1]*o[1]),d[3]=l[1]*l[2]-o[1]*o[2],d[4]=S*(l[2]*l[2]-l[0]*l[0]-(o[2]*o[2]-o[0]*o[0])),j.dot5(e,t,n,s,r,d,_)};static parseHeader(e){const t=new Uint8Array(e,0,j.HeaderSizeBytes),n=new Uint16Array(e,0,j.HeaderSizeBytes/2),s=new Uint32Array(e,0,j.HeaderSizeBytes/4),r=new Float32Array(e,0,j.HeaderSizeBytes/4),o=t[0],a=t[1],l=s[1],c=s[2],u=s[3],f=s[4],h=n[10],d=new F(r[6],r[7],r[8]),x=r[9]||-ws,p=r[10]||ws;return{versionMajor:o,versionMinor:a,maxSectionCount:l,sectionCount:c,maxSplatCount:u,splatCount:f,compressionLevel:h,sceneCenter:d,minSphericalHarmonicsCoeff:x,maxSphericalHarmonicsCoeff:p}}static writeHeaderCountsToBuffer(e,t,n){const s=new Uint32Array(n,0,j.HeaderSizeBytes/4);s[2]=e,s[4]=t}static writeHeaderToBuffer(e,t){const n=new Uint8Array(t,0,j.HeaderSizeBytes),s=new Uint16Array(t,0,j.HeaderSizeBytes/2),r=new Uint32Array(t,0,j.HeaderSizeBytes/4),o=new Float32Array(t,0,j.HeaderSizeBytes/4);n[0]=e.versionMajor,n[1]=e.versionMinor,n[2]=0,n[3]=0,r[1]=e.maxSectionCount,r[2]=e.sectionCount,r[3]=e.maxSplatCount,r[4]=e.splatCount,s[10]=e.compressionLevel,o[6]=e.sceneCenter.x,o[7]=e.sceneCenter.y,o[8]=e.sceneCenter.z,o[9]=e.minSphericalHarmonicsCoeff||-ws,o[10]=e.maxSphericalHarmonicsCoeff||ws}static parseSectionHeaders(e,t,n=0,s){const r=e.compressionLevel,o=e.maxSectionCount,a=new Uint16Array(t,n,o*j.SectionHeaderSizeBytes/2),l=new Uint32Array(t,n,o*j.SectionHeaderSizeBytes/4),c=new Float32Array(t,n,o*j.SectionHeaderSizeBytes/4),u=[];let f=0,h=f/2,d=f/4,x=j.HeaderSizeBytes+e.maxSectionCount*j.SectionHeaderSizeBytes,p=0;for(let g=0;g<o;g++){const m=l[d+1],_=l[d+2],S=l[d+3],A=c[d+4],y=A/2,b=a[h+10],v=l[d+6]||j.CompressionLevels[r].ScaleRange,E=l[d+8],M=l[d+9],T=M*4,I=b*S+T,P=a[h+20],{bytesPerSplat:B}=j.calculateComponentStorage(r,P),N=B*m,G=N+I,V={bytesPerSplat:B,splatCountOffset:p,splatCount:s?m:0,maxSplatCount:m,bucketSize:_,bucketCount:S,bucketBlockSize:A,halfBucketBlockSize:y,bucketStorageSizeBytes:b,bucketsStorageSizeBytes:I,splatDataStorageSizeBytes:N,storageSizeBytes:G,compressionScaleRange:v,compressionScaleFactor:y/v,base:x,bucketsBase:x+T,dataBase:x+I,fullBucketCount:E,partiallyFilledBucketCount:M,sphericalHarmonicsDegree:P};u[g]=V,x+=G,f+=j.SectionHeaderSizeBytes,h=f/2,d=f/4,p+=m}return u}static writeSectionHeaderToBuffer(e,t,n,s=0){const r=new Uint16Array(n,s,j.SectionHeaderSizeBytes/2),o=new Uint32Array(n,s,j.SectionHeaderSizeBytes/4),a=new Float32Array(n,s,j.SectionHeaderSizeBytes/4);o[0]=e.splatCount,o[1]=e.maxSplatCount,o[2]=t>=1?e.bucketSize:0,o[3]=t>=1?e.bucketCount:0,a[4]=t>=1?e.bucketBlockSize:0,r[10]=t>=1?j.BucketStorageSizeBytes:0,o[6]=t>=1?e.compressionScaleRange:0,o[7]=e.storageSizeBytes,o[8]=t>=1?e.fullBucketCount:0,o[9]=t>=1?e.partiallyFilledBucketCount:0,r[20]=e.sphericalHarmonicsDegree}static writeSectionHeaderSplatCountToBuffer(e,t,n=0){const s=new Uint32Array(t,n,j.SectionHeaderSizeBytes/4);s[0]=e}constructFromBuffer(e,t){this.bufferData=e,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSectionMap=[];const n=j.parseHeader(this.bufferData);this.versionMajor=n.versionMajor,this.versionMinor=n.versionMinor,this.maxSectionCount=n.maxSectionCount,this.sectionCount=t?n.maxSectionCount:0,this.maxSplatCount=n.maxSplatCount,this.splatCount=t?n.maxSplatCount:0,this.compressionLevel=n.compressionLevel,this.sceneCenter=new F().copy(n.sceneCenter),this.minSphericalHarmonicsCoeff=n.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff=n.maxSphericalHarmonicsCoeff,this.sections=j.parseSectionHeaders(n,this.bufferData,j.HeaderSizeBytes,t),this.linkBufferArrays(),this.buildMaps()}static calculateComponentStorage(e,t){const n=j.CompressionLevels[e].BytesPerCenter,s=j.CompressionLevels[e].BytesPerScale,r=j.CompressionLevels[e].BytesPerRotation,o=j.CompressionLevels[e].BytesPerColor,a=uo(t),l=j.CompressionLevels[e].BytesPerSphericalHarmonicsComponent*a,c=n+s+r+o+l;return{bytesPerCenter:n,bytesPerScale:s,bytesPerRotation:r,bytesPerColor:o,sphericalHarmonicsComponentsPerSplat:a,sphericalHarmonicsBytesPerSplat:l,bytesPerSplat:c}}linkBufferArrays(){for(let e=0;e<this.maxSectionCount;e++){const t=this.sections[e];t.bucketArray=new Float32Array(this.bufferData,t.bucketsBase,t.bucketCount*j.BucketStorageSizeFloats),t.partiallyFilledBucketCount>0&&(t.partiallyFilledBucketLengths=new Uint32Array(this.bufferData,t.base,t.partiallyFilledBucketCount))}}buildMaps(){let e=0;for(let t=0;t<this.maxSectionCount;t++){const n=this.sections[t];for(let s=0;s<n.maxSplatCount;s++){const r=e+s;this.globalSplatIndexToLocalSplatIndexMap[r]=s,this.globalSplatIndexToSectionMap[r]=t}e+=n.maxSplatCount}}updateLoadedCounts(e,t){j.writeHeaderCountsToBuffer(e,t,this.bufferData),this.sectionCount=e,this.splatCount=t}updateSectionLoadedCounts(e,t){const n=j.HeaderSizeBytes+j.SectionHeaderSizeBytes*e;j.writeSectionHeaderSplatCountToBuffer(t,this.bufferData,n),this.sections[e].splatCount=t}static writeSplatDataToSectionBuffer=(function(){const e=new ArrayBuffer(12),t=new ArrayBuffer(12),n=new ArrayBuffer(16),s=new ArrayBuffer(4),r=new ArrayBuffer(256),o=new Mt,a=new F,l=new F,{X:c,Y:u,Z:f,SCALE0:h,SCALE1:d,SCALE2:x,ROTATION0:p,ROTATION1:g,ROTATION2:m,ROTATION3:_,FDC0:S,FDC1:A,FDC2:y,OPACITY:b,FRC0:v,FRC9:E}=we.OFFSET,M=(T,I,P)=>{const B=P*2+1;return T=Math.round(T*I)+P,Rt(T,0,B)};return function(T,I,P,B,N,G,V,q,X=-ws,ee=ws){const ce=uo(N),be=j.CompressionLevels[B].BytesPerCenter,Re=j.CompressionLevels[B].BytesPerScale,Fe=j.CompressionLevels[B].BytesPerRotation,Oe=j.CompressionLevels[B].BytesPerColor,Ne=P,J=Ne+be,ne=J+Re,xe=ne+Fe,Be=xe+Oe;if(T[p]!==void 0?(o.set(T[p],T[g],T[m],T[_]),o.normalize()):o.set(1,0,0,0),T[h]!==void 0?a.set(T[h]||0,T[d]||0,T[x]||0):a.set(0,0,0),B===0){const Ve=new Float32Array(I,Ne,j.CenterComponentCount),L=new Float32Array(I,ne,j.RotationComponentCount),U=new Float32Array(I,J,j.ScaleComponentCount);if(L.set([o.x,o.y,o.z,o.w]),U.set([a.x,a.y,a.z]),Ve.set([T[c],T[u],T[f]]),N>0){const Y=new Float32Array(I,Be,ce);if(N>=1){for(let w=0;w<9;w++)Y[w]=T[v+w]||0;if(N>=2)for(let w=0;w<15;w++)Y[w+9]=T[E+w]||0}}}else{const Ve=new Uint16Array(e,0,j.CenterComponentCount),L=new Uint16Array(n,0,j.RotationComponentCount),U=new Uint16Array(t,0,j.ScaleComponentCount);if(L.set([Zt(o.x),Zt(o.y),Zt(o.z),Zt(o.w)]),U.set([Zt(a.x),Zt(a.y),Zt(a.z)]),l.set(T[c],T[u],T[f]).sub(G),l.x=M(l.x,V,q),l.y=M(l.y,V,q),l.z=M(l.z,V,q),Ve.set([l.x,l.y,l.z]),N>0){const Y=B===1?Uint16Array:Uint8Array,w=B===1?2:1,oe=new Y(r,0,ce);if(N>=1){for(let pe=0;pe<9;pe++){const se=T[v+pe]||0;oe[pe]=B===1?Zt(se):sa(se,X,ee)}const re=9*w;if(Yr(oe.buffer,0,I,Be,re),N>=2){for(let pe=0;pe<15;pe++){const se=T[E+pe]||0;oe[pe+9]=B===1?Zt(se):sa(se,X,ee)}Yr(oe.buffer,re,I,Be+re,15*w)}}}Yr(Ve.buffer,0,I,Ne,6),Yr(U.buffer,0,I,J,6),Yr(L.buffer,0,I,ne,8)}const Te=new Uint8ClampedArray(s,0,4);Te.set([T[S]||0,T[A]||0,T[y]||0]),Te[3]=T[b]||0,Yr(Te.buffer,0,I,xe,4)}})();static generateFromUncompressedSplatArrays(e,t,n,s,r,o,a=[]){let l=0;for(let y=0;y<e.length;y++){const b=e[y];l=Math.max(b.sphericalHarmonicsDegree,l)}let c,u;for(let y=0;y<e.length;y++){const b=e[y];for(let v=0;v<b.splats.length;v++){const E=b.splats[v];for(let M=we.OFFSET.FRC0;M<we.OFFSET.FRC23&&M<E.length;M++)(!c||E[M]<c)&&(c=E[M]),(!u||E[M]>u)&&(u=E[M])}}c=c||-ws,u=u||ws;const{bytesPerSplat:f}=j.calculateComponentStorage(n,l),h=j.CompressionLevels[n].ScaleRange,d=[],x=[];let p=0;for(let y=0;y<e.length;y++){const b=e[y],v=new we(l);for(let Ne=0;Ne<b.splatCount;Ne++){const J=b.splats[Ne];(J[we.OFFSET.OPACITY]||0)>=t&&v.addSplat(J)}const E=a[y]||{},M=(E.blockSizeFactor||1)*(r||j.BucketBlockSize),T=Math.ceil((E.bucketSizeFactor||1)*(o||j.BucketSize)),I=j.computeBucketsForUncompressedSplatArray(v,M,T),P=I.fullBuckets.length,B=I.partiallyFullBuckets.map(Ne=>Ne.splats.length),N=B.length,G=[...I.fullBuckets,...I.partiallyFullBuckets],V=v.splats.length*f,q=N*4,X=n>=1?G.length*j.BucketStorageSizeBytes+q:0,ee=V+X,ce=new ArrayBuffer(ee),be=h/(M*.5),Re=new F;let Fe=0;for(let Ne=0;Ne<G.length;Ne++){const J=G[Ne];Re.fromArray(J.center);for(let ne=0;ne<J.splats.length;ne++){let xe=J.splats[ne];const Be=v.splats[xe],Te=X+Fe*f;j.writeSplatDataToSectionBuffer(Be,ce,Te,n,l,Re,be,h,c,u),Fe++}}if(p+=Fe,n>=1){const Ne=new Uint32Array(ce,0,B.length*4);for(let ne=0;ne<B.length;ne++)Ne[ne]=B[ne];const J=new Float32Array(ce,q,G.length*j.BucketStorageSizeFloats);for(let ne=0;ne<G.length;ne++){const xe=G[ne],Be=ne*3;J[Be]=xe.center[0],J[Be+1]=xe.center[1],J[Be+2]=xe.center[2]}}d.push(ce);const Oe=new ArrayBuffer(j.SectionHeaderSizeBytes);j.writeSectionHeaderToBuffer({maxSplatCount:Fe,splatCount:Fe,bucketSize:T,bucketCount:G.length,bucketBlockSize:M,compressionScaleRange:h,storageSizeBytes:ee,fullBucketCount:P,partiallyFilledBucketCount:N,sphericalHarmonicsDegree:l},n,Oe,0),x.push(Oe)}let g=0;for(let y of d)g+=y.byteLength;const m=j.HeaderSizeBytes+j.SectionHeaderSizeBytes*d.length+g,_=new ArrayBuffer(m);j.writeHeaderToBuffer({versionMajor:0,versionMinor:1,maxSectionCount:d.length,sectionCount:d.length,maxSplatCount:p,splatCount:p,compressionLevel:n,sceneCenter:s,minSphericalHarmonicsCoeff:c,maxSphericalHarmonicsCoeff:u},_);let S=j.HeaderSizeBytes;for(let y of x)new Uint8Array(_,S,j.SectionHeaderSizeBytes).set(new Uint8Array(y)),S+=j.SectionHeaderSizeBytes;for(let y of d)new Uint8Array(_,S,y.byteLength).set(new Uint8Array(y)),S+=y.byteLength;return new j(_)}static computeBucketsForUncompressedSplatArray(e,t,n){let s=e.splatCount;const r=t/2,o=new F,a=new F;for(let p=0;p<s;p++){const g=e.splats[p],m=[g[we.OFFSET.X],g[we.OFFSET.Y],g[we.OFFSET.Z]];(p===0||m[0]<o.x)&&(o.x=m[0]),(p===0||m[0]>a.x)&&(a.x=m[0]),(p===0||m[1]<o.y)&&(o.y=m[1]),(p===0||m[1]>a.y)&&(a.y=m[1]),(p===0||m[2]<o.z)&&(o.z=m[2]),(p===0||m[2]>a.z)&&(a.z=m[2])}const l=new F().copy(a).sub(o),c=Math.ceil(l.y/t),u=Math.ceil(l.z/t),f=new F,h=[],d={};for(let p=0;p<s;p++){const g=e.splats[p],m=[g[we.OFFSET.X],g[we.OFFSET.Y],g[we.OFFSET.Z]],_=Math.floor((m[0]-o.x)/t),S=Math.floor((m[1]-o.y)/t),A=Math.floor((m[2]-o.z)/t);f.x=_*t+o.x+r,f.y=S*t+o.y+r,f.z=A*t+o.z+r;const y=_*(c*u)+S*u+A;let b=d[y];b||(d[y]=b={splats:[],center:f.toArray()}),b.splats.push(p),b.splats.length>=n&&(h.push(b),d[y]=null)}const x=[];for(let p in d)if(d.hasOwnProperty(p)){const g=d[p];g&&x.push(g)}return{fullBuckets:h,partiallyFullBuckets:x}}static preallocateUncompressed(e,t){const n=j.CompressionLevels[0].SphericalHarmonicsDegrees[t],s=j.HeaderSizeBytes+j.SectionHeaderSizeBytes,r=s+n.BytesPerSplat*e,o=new ArrayBuffer(r);return j.writeHeaderToBuffer({versionMajor:j.CurrentMajorVersion,versionMinor:j.CurrentMinorVersion,maxSectionCount:1,sectionCount:1,maxSplatCount:e,splatCount:e,compressionLevel:0,sceneCenter:new F},o),j.writeSectionHeaderToBuffer({maxSplatCount:e,splatCount:e,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:t},0,o,j.HeaderSizeBytes),{splatBuffer:new j(o,!0),splatBufferDataOffsetBytes:s}}}const zp=new Uint8Array([112,108,121,10]),kp=new Uint8Array([10,101,110,100,95,104,101,97,100,101,114,10]),gu="end_header",xu=new Map([["char",Int8Array],["uchar",Uint8Array],["short",Int16Array],["ushort",Uint16Array],["int",Int32Array],["uint",Uint32Array],["float",Float32Array],["double",Float64Array]]),zi=(i,e)=>{const t=(1<<e)-1;return(i&t)/t},Hp=(i,e)=>{i.x=zi(e>>>21,11),i.y=zi(e>>>11,10),i.z=zi(e,11)},TC=(i,e)=>{i.x=zi(e>>>24,8),i.y=zi(e>>>16,8),i.z=zi(e>>>8,8),i.w=zi(e,8)},CC=(i,e)=>{const t=1/(Math.sqrt(2)*.5),n=(zi(e>>>20,10)-.5)*t,s=(zi(e>>>10,10)-.5)*t,r=(zi(e,10)-.5)*t,o=Math.sqrt(1-(n*n+s*s+r*r));switch(e>>>30){case 0:i.set(o,n,s,r);break;case 1:i.set(n,o,s,r);break;case 2:i.set(n,s,o,r);break;case 3:i.set(n,s,r,o);break}},ts=(i,e,t)=>i*(1-t)+e*t,Bt=(i,e)=>i.properties.find(t=>t.name===e&&t.storage)?.storage;class at{static decodeHeaderText(e){let t,n,s,r;const o=e.split(`
`).filter(f=>!f.startsWith("comment "));let a=0,l=!1;for(let f=1;f<o.length;++f){const h=o[f].split(" ");switch(h[0]){case"format":if(h[1]!=="binary_little_endian")throw new Error("Unsupported ply format");break;case"element":t={name:h[1],count:parseInt(h[2],10),properties:[],storageSizeBytes:0},t.name==="chunk"?n=t:t.name==="vertex"?s=t:t.name==="sh"&&(r=t);break;case"property":{if(!xu.has(h[1]))throw new Error(`Unrecognized property data type '${h[1]}' in ply header`);const d=xu.get(h[1]),x=d.BYTES_PER_ELEMENT*t.count;t.name==="vertex"&&(a+=d.BYTES_PER_ELEMENT),t.properties.push({type:h[1],name:h[2],storage:null,byteSize:d.BYTES_PER_ELEMENT,storageSizeByes:x}),t.storageSizeBytes+=x;break}case gu:l=!0;break;default:throw new Error(`Unrecognized header value '${h[0]}' in ply header`)}if(l)break}let c=0,u=0;return r&&(u=r.properties.length,r.properties.length>=45?c=3:r.properties.length>=24?c=2:r.properties.length>=9&&(c=1)),{chunkElement:n,vertexElement:s,shElement:r,bytesPerSplat:a,headerSizeBytes:e.indexOf(gu)+gu.length+1,sphericalHarmonicsDegree:c,sphericalHarmonicsPerSplat:u}}static decodeHeader(e){const t=(d,x)=>{const p=d.length-x.length;let g,m;for(g=0;g<=p;++g){for(m=0;m<x.length&&d[g+m]===x[m];++m);if(m===x.length)return g}return-1},n=(d,x)=>{if(d.length<x.length)return!1;for(let p=0;p<x.length;++p)if(d[p]!==x[p])return!1;return!0};let s=new Uint8Array(e),r;if(s.length>=zp.length&&!n(s,zp))throw new Error("Invalid PLY header");if(r=t(s,kp),r===-1)throw new Error("End of PLY header not found");const o=new TextDecoder("ascii").decode(s.slice(0,r)),{chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f,bytesPerSplat:h}=at.decodeHeaderText(o);return{headerSizeBytes:r+kp.length,bytesPerSplat:h,chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f}}static readElementData(e,t,n,s,r,o=null){let a=t instanceof DataView?t:new DataView(t);s=s||0,r=r||e.count-1;for(let l=s;l<=r;++l)for(let c=0;c<e.properties.length;++c){const u=e.properties[c],f=xu.get(u.type),h=f.BYTES_PER_ELEMENT*e.count;if((!u.storage||u.storage.byteLength<h)&&(!o||o(u.name))&&(u.storage=new f(e.count)),u.storage)switch(u.type){case"char":u.storage[l]=a.getInt8(n);break;case"uchar":u.storage[l]=a.getUint8(n);break;case"short":u.storage[l]=a.getInt16(n,!0);break;case"ushort":u.storage[l]=a.getUint16(n,!0);break;case"int":u.storage[l]=a.getInt32(n,!0);break;case"uint":u.storage[l]=a.getUint32(n,!0);break;case"float":u.storage[l]=a.getFloat32(n,!0);break;case"double":u.storage[l]=a.getFloat64(n,!0);break}n+=u.byteSize}return n}static readPly(e,t=null){const n=at.decodeHeader(e);let s=at.readElementData(n.chunkElement,e,n.headerSizeBytes,null,null,t);return s=at.readElementData(n.vertexElement,e,s,null,null,t),at.readElementData(n.shElement,e,s,null,null,t),{chunkElement:n.chunkElement,vertexElement:n.vertexElement,shElement:n.shElement,sphericalHarmonicsDegree:n.sphericalHarmonicsDegree,sphericalHarmonicsPerSplat:n.sphericalHarmonicsPerSplat}}static getElementStorageArrays(e,t,n){const s={};if(t){const r=Bt(e,"min_r"),o=Bt(e,"min_g"),a=Bt(e,"min_b"),l=Bt(e,"max_r"),c=Bt(e,"max_g"),u=Bt(e,"max_b"),f=Bt(e,"min_x"),h=Bt(e,"min_y"),d=Bt(e,"min_z"),x=Bt(e,"max_x"),p=Bt(e,"max_y"),g=Bt(e,"max_z"),m=Bt(e,"min_scale_x"),_=Bt(e,"min_scale_y"),S=Bt(e,"min_scale_z"),A=Bt(e,"max_scale_x"),y=Bt(e,"max_scale_y"),b=Bt(e,"max_scale_z"),v=Bt(t,"packed_position"),E=Bt(t,"packed_rotation"),M=Bt(t,"packed_scale"),T=Bt(t,"packed_color");s.colorExtremes={minR:r,maxR:l,minG:o,maxG:c,minB:a,maxB:u},s.positionExtremes={minX:f,maxX:x,minY:h,maxY:p,minZ:d,maxZ:g},s.scaleExtremes={minScaleX:m,maxScaleX:A,minScaleY:_,maxScaleY:y,minScaleZ:S,maxScaleZ:b},s.position=v,s.rotation=E,s.scale=M,s.color=T}if(n){const r={};for(let o=0;o<45;o++){const a=`f_rest_${o}`,l=Bt(n,a);if(l)r[a]=l;else break}s.sh=r}return s}static decompressBaseSplat=(function(){const e=new F,t=new Mt,n=new F,s=new Dt,r=we.OFFSET;return function(o,a,l,c,u,f,h,d,x,p){p=p||we.createSplat();const g=Math.floor((a+o)/256);return Hp(e,l[o]),CC(t,h[o]),Hp(n,u[o]),TC(s,x[o]),p[r.X]=ts(c.minX[g],c.maxX[g],e.x),p[r.Y]=ts(c.minY[g],c.maxY[g],e.y),p[r.Z]=ts(c.minZ[g],c.maxZ[g],e.z),p[r.ROTATION0]=t.x,p[r.ROTATION1]=t.y,p[r.ROTATION2]=t.z,p[r.ROTATION3]=t.w,p[r.SCALE0]=Math.exp(ts(f.minScaleX[g],f.maxScaleX[g],n.x)),p[r.SCALE1]=Math.exp(ts(f.minScaleY[g],f.maxScaleY[g],n.y)),p[r.SCALE2]=Math.exp(ts(f.minScaleZ[g],f.maxScaleZ[g],n.z)),d.minR&&d.maxR?p[r.FDC0]=Rt(Math.round(ts(d.minR[g],d.maxR[g],s.x)*255),0,255):p[r.FDC0]=Rt(Math.floor(s.x*255),0,255),d.minG&&d.maxG?p[r.FDC1]=Rt(Math.round(ts(d.minG[g],d.maxG[g],s.y)*255),0,255):p[r.FDC1]=Rt(Math.floor(s.y*255),0,255),d.minB&&d.maxB?p[r.FDC2]=Rt(Math.round(ts(d.minB[g],d.maxB[g],s.z)*255),0,255):p[r.FDC2]=Rt(Math.floor(s.z*255),0,255),p[r.OPACITY]=Rt(Math.floor(s.w*255),0,255),p}})();static decompressSphericalHarmonics=(function(){const e=[0,3,8,15],t=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(n,s,r,o,a){a=a||we.createSplat();let l=e[r],c=e[o];for(let u=0;u<3;++u)for(let f=0;f<15;++f){const h=t[u*15+f];f<l&&f<c&&(a[we.OFFSET.FRC0+h]=s[u*c+f][n]*(8/255)-4)}return a}})();static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l,c=null){at.readElementData(t,o,0,n,s,c);const u=j.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,{positionExtremes:f,scaleExtremes:h,colorExtremes:d,position:x,rotation:p,scale:g,color:m}=at.getElementStorageArrays(e,t),_=we.createSplat();for(let S=n;S<=s;++S){at.decompressBaseSplat(S,r,x,f,g,h,p,d,m,_);const A=S*u+l;j.writeSplatDataToSectionBuffer(_,a,A,0,0)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a,l=null){at.readElementData(t,o,0,n,s,l);const{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:h,rotation:d,scale:x,color:p}=at.getElementStorageArrays(e,t);for(let g=n;g<=s;++g){const m=we.createSplat();at.decompressBaseSplat(g,r,h,c,x,u,d,f,p,m),a.addSplat(m)}}static parseSphericalHarmonicsToUncompressedSplatArraySection(e,t,n,s,r,o,a,l,c,u=null){at.readElementData(t,r,o,n,s,u);const{sh:f}=at.getElementStorageArrays(e,void 0,t),h=Object.values(f);for(let d=n;d<=s;++d)at.decompressSphericalHarmonics(d,h,a,l,c.splats[d])}static parseToUncompressedSplatArray(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=at.readPly(e);t=Math.min(t,o);const a=new we(t),{positionExtremes:l,scaleExtremes:c,colorExtremes:u,position:f,rotation:h,scale:d,color:x}=at.getElementStorageArrays(n,s);let p;if(t>0){const{sh:g}=at.getElementStorageArrays(n,void 0,r);p=Object.values(g)}for(let g=0;g<s.count;++g){a.addDefaultSplat();const m=a.getSplat(a.splatCount-1);at.decompressBaseSplat(g,0,f,l,d,c,h,u,x,m),t>0&&at.decompressSphericalHarmonics(g,p,t,o,m)}return a}static parseToUncompressedSplatBuffer(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=at.readPly(e);t=Math.min(t,o);const{splatBuffer:a,splatBufferDataOffsetBytes:l}=j.preallocateUncompressed(s.count,t),{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:h,rotation:d,scale:x,color:p}=at.getElementStorageArrays(n,s);let g;if(t>0){const{sh:S}=at.getElementStorageArrays(n,void 0,r);g=Object.values(S)}const m=j.CompressionLevels[0].SphericalHarmonicsDegrees[t].BytesPerSplat,_=we.createSplat(t);for(let S=0;S<s.count;++S){at.decompressBaseSplat(S,0,h,c,x,u,d,f,p,_),t>0&&at.decompressSphericalHarmonics(S,g,t,o,_);const A=S*m+l;j.writeSplatDataToSectionBuffer(_,a.bufferData,A,0,t)}return a}}const An={INRIAV1:0,INRIAV2:1,PlayCanvasCompressed:2},[Ag,Dh,Ph,Fh,Lh,Bh,Uh]=[0,1,2,3,4,5,6],Vp={double:Ag,int:Dh,uint:Ph,float:Fh,short:Lh,ushort:Bh,uchar:Uh},EC={[Ag]:8,[Dh]:4,[Ph]:4,[Fh]:4,[Lh]:2,[Bh]:2,[Uh]:1};class ct{static HeaderEndToken="end_header";static decodeSectionHeader(e,t,n=0){const s=[];let r=!1,o=-1,a=0,l=!1,c=null;const u=[],f=[],h=[],d={};for(let m=n;m<e.length;m++){const _=e[m].trim();if(_.startsWith("element"))if(r){o--;break}else{r=!0,n=m,o=m;const S=_.split(" ");let A=0;for(let y of S){const b=y.trim();b.length>0&&(A++,A===2?c=b:A===3&&(a=parseInt(b)))}}else if(_.startsWith("property")){const S=_.match(/(\w+)\s+(\w+)\s+(\w+)/);if(S){const A=S[2],y=S[3];h.push(y);const b=t[y];d[y]=A;const v=Vp[A];b!==void 0&&(u.push(b),f[b]=v)}}if(_===ct.HeaderEndToken){l=!0;break}r&&(s.push(_),o++)}const x=[];let p=0;for(let m of h){const _=d[m];if(d.hasOwnProperty(m)){const S=t[m];S!==void 0&&(x[S]=p)}p+=EC[Vp[_]]}const g=ct.decodeSphericalHarmonicsFromSectionHeader(h,t);return{headerLines:s,headerStartLine:n,headerEndLine:o,fieldTypes:f,fieldIds:u,fieldOffsets:x,bytesPerVertex:p,vertexCount:a,dataSizeBytes:p*a,endOfHeader:l,sectionName:c,sphericalHarmonicsDegree:g.degree,sphericalHarmonicsCoefficientsPerChannel:g.coefficientsPerChannel,sphericalHarmonicsDegree1Fields:g.degree1Fields,sphericalHarmonicsDegree2Fields:g.degree2Fields}}static decodeSphericalHarmonicsFromSectionHeader(e,t){let n=0,s=0;for(let l of e)l.startsWith("f_rest")&&n++;s=n/3;let r=0;s>=3&&(r=1),s>=8&&(r=2);let o=[],a=[];for(let l=0;l<3;l++){if(r>=1)for(let c=0;c<3;c++)o.push(t["f_rest_"+(c+s*l)]);if(r>=2)for(let c=0;c<5;c++)a.push(t["f_rest_"+(c+s*l+3)])}return{degree:r,coefficientsPerChannel:s,degree1Fields:o,degree2Fields:a}}static getHeaderSectionNames(e){const t=[];for(let n of e)if(n.startsWith("element")){const s=n.split(" ");let r=0;for(let o of s){const a=o.trim();a.length>0&&(r++,r===2&&t.push(a))}}return t}static checkTextForEndHeader(e){return!!e.includes(ct.HeaderEndToken)}static checkBufferForEndHeader(e,t,n,s){const r=new Uint8Array(e,Math.max(0,t-n),n),o=s.decode(r);return ct.checkTextForEndHeader(o)}static extractHeaderFromBufferToText(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,ct.checkBufferForEndHeader(e,n,r*2,t))break}return s}static readHeaderFromBuffer(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,ct.checkBufferForEndHeader(e,n,r*2,t))break}return s}static convertHeaderTextToLines(e){const t=e.split(`
`),n=[];for(let s=0;s<t.length;s++){const r=t[s].trim();if(n.push(r),r===ct.HeaderEndToken)break}return n}static determineHeaderFormatFromHeaderText(e){const t=ct.convertHeaderTextToLines(e);let n=An.INRIAV1;for(let s=0;s<t.length;s++){const r=t[s].trim();if(r.startsWith("element chunk")||r.match(/[A-Za-z]*packed_[A-Za-z]*/))n=An.PlayCanvasCompressed;else if(r.startsWith("element codebook_centers"))n=An.INRIAV2;else if(r===ct.HeaderEndToken)break}return n}static determineHeaderFormatFromPlyBuffer(e){const t=ct.extractHeaderFromBufferToText(e);return ct.determineHeaderFormatFromHeaderText(t)}static readVertex(e,t,n,s,r,o,a=!0){const l=n*t.bytesPerVertex+s,c=t.fieldOffsets,u=t.fieldTypes;for(let f of r){const h=u[f];h===Fh?o[f]=e.getFloat32(l+c[f],!0):h===Lh?o[f]=e.getInt16(l+c[f],!0):h===Bh?o[f]=e.getUint16(l+c[f],!0):h===Dh?o[f]=e.getInt32(l+c[f],!0):h===Ph?o[f]=e.getUint32(l+c[f],!0):h===Uh&&(a?o[f]=e.getUint8(l+c[f])/255:o[f]=e.getUint8(l+c[f]))}}}const yg=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0"],wC=yg.map((i,e)=>e),[Gp,RC,IC,DC,PC,FC,LC,BC,UC,OC,Wp,NC,zC,Xp,qp,kC,HC,VC]=wC;class rn{static decodeHeaderLines(e){let t=0;e.forEach(u=>{u.includes("f_rest_")&&t++});let n=0;t>=45?n=45:t>=24?n=24:t>=9&&(n=9);let r=Array.from(Array(Math.max(n-1,0))).map((u,f)=>`f_rest_${f+1}`);const o=[...yg,...r],a=o.map((u,f)=>f),l=a.reduce((u,f)=>(u[o[f]]=f,u),{}),c=ct.decodeSectionHeader(e,l,0);return c.splatCount=c.vertexCount,c.bytesPerSplat=c.bytesPerVertex,c.fieldsToReadIndexes=a,c}static decodeHeaderText(e){const t=ct.convertHeaderTextToLines(e),n=rn.decodeHeaderLines(t);return n.headerText=e,n.headerSizeBytes=e.indexOf(ct.HeaderEndToken)+ct.HeaderEndToken.length+1,n}static decodeHeaderFromBuffer(e){const t=ct.readHeaderFromBuffer(e);return rn.decodeHeaderText(t)}static findSplatData(e,t){return new DataView(e,t.headerSizeBytes)}static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l=0){l=Math.min(l,e.sphericalHarmonicsDegree);const c=j.CompressionLevels[0].SphericalHarmonicsDegrees[l].BytesPerSplat;for(let u=t;u<=n;u++){const f=rn.parseToUncompressedSplat(s,u,e,r,l),h=u*c+a;j.writeSplatDataToSectionBuffer(f,o,h,0,l)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a=0){a=Math.min(a,e.sphericalHarmonicsDegree);for(let l=t;l<=n;l++){const c=rn.parseToUncompressedSplat(s,l,e,r,a);o.addSplat(c)}}static decodeSectionSplatData(e,t,n,s,r=!0){if(s=Math.min(s,n.sphericalHarmonicsDegree),r){const o=new we(s);for(let a=0;a<t;a++){const l=rn.parseToUncompressedSplat(e,a,n,0,s);o.addSplat(l)}return o}else{const{splatBuffer:o,splatBufferDataOffsetBytes:a}=j.preallocateUncompressed(t,s);return rn.parseToUncompressedSplatBufferSection(n,0,t-1,e,0,o.bufferData,a,s),o}}static parseToUncompressedSplat=(function(){let e=[];const t=new Mt,n=we.OFFSET.X,s=we.OFFSET.Y,r=we.OFFSET.Z,o=we.OFFSET.SCALE0,a=we.OFFSET.SCALE1,l=we.OFFSET.SCALE2,c=we.OFFSET.ROTATION0,u=we.OFFSET.ROTATION1,f=we.OFFSET.ROTATION2,h=we.OFFSET.ROTATION3,d=we.OFFSET.FDC0,x=we.OFFSET.FDC1,p=we.OFFSET.FDC2,g=we.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=we.OFFSET.FRC0+_;return function(_,S,A,y=0,b=0){b=Math.min(b,A.sphericalHarmonicsDegree),rn.readSplat(_,A,S,y,e);const v=we.createSplat(b);if(e[Gp]!==void 0?(v[o]=Math.exp(e[Gp]),v[a]=Math.exp(e[RC]),v[l]=Math.exp(e[IC])):(v[o]=.01,v[a]=.01,v[l]=.01),e[Wp]!==void 0){const E=.28209479177387814;v[d]=(.5+E*e[Wp])*255,v[x]=(.5+E*e[NC])*255,v[p]=(.5+E*e[zC])*255}else e[qp]!==void 0?(v[d]=e[qp]*255,v[x]=e[kC]*255,v[p]=e[HC]*255):(v[d]=0,v[x]=0,v[p]=0);if(e[Xp]!==void 0&&(v[g]=1/(1+Math.exp(-e[Xp]))*255),v[d]=Rt(Math.floor(v[d]),0,255),v[x]=Rt(Math.floor(v[x]),0,255),v[p]=Rt(Math.floor(v[p]),0,255),v[g]=Rt(Math.floor(v[g]),0,255),b>=1&&e[VC]!==void 0){for(let E=0;E<9;E++)v[m[E]]=e[A.sphericalHarmonicsDegree1Fields[E]];if(b>=2)for(let E=0;E<15;E++)v[m[9+E]]=e[A.sphericalHarmonicsDegree2Fields[E]]}return t.set(e[DC],e[PC],e[FC],e[LC]),t.normalize(),v[c]=t.x,v[u]=t.y,v[f]=t.z,v[h]=t.w,v[n]=e[BC],v[s]=e[UC],v[r]=e[OC],v}})();static readSplat(e,t,n,s,r){return ct.readVertex(e,t,n,s,t.fieldsToReadIndexes,r,!0)}static parseToUncompressedSplatArray(e,t=0){const{header:n,splatCount:s,splatData:r}=Yp(e);return rn.decodeSectionSplatData(r,s,n,t,!0)}static parseToUncompressedSplatBuffer(e,t=0){const{header:n,splatCount:s,splatData:r}=Yp(e);return rn.decodeSectionSplatData(r,s,n,t,!1)}}function Yp(i){const e=rn.decodeHeaderFromBuffer(i),t=e.splatCount,n=rn.findSplatData(i,e);return{header:e,splatCount:t,splatData:n}}const bg=["features_dc","features_rest_0","features_rest_1","features_rest_2","features_rest_3","features_rest_4","features_rest_5","features_rest_6","features_rest_7","features_rest_8","features_rest_9","features_rest_10","features_rest_11","features_rest_12","features_rest_13","features_rest_14","opacity","scaling","rotation_re","rotation_im"],pl=bg.map((i,e)=>e),[ml,GC,WC,Qp,gl,XC,_u]=[0,1,4,16,17,18,19],Mg=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0","f_rest_1","f_rest_2","f_rest_3","f_rest_4","f_rest_5","f_rest_6","f_rest_7","f_rest_8","f_rest_9","f_rest_10","f_rest_11","f_rest_12","f_rest_13","f_rest_14","f_rest_15","f_rest_16","f_rest_17","f_rest_18","f_rest_19","f_rest_20","f_rest_21","f_rest_22","f_rest_23","f_rest_24","f_rest_25","f_rest_26","f_rest_27","f_rest_28","f_rest_29","f_rest_30","f_rest_31","f_rest_32","f_rest_33","f_rest_34","f_rest_35","f_rest_36","f_rest_37","f_rest_38","f_rest_39","f_rest_40","f_rest_41","f_rest_42","f_rest_43","f_rest_44","f_rest_45"],Lf=Mg.map((i,e)=>e),[Kp,qC,YC,QC,KC,jC,$C,ZC,JC,eE,Bf,Tg,Cg,jp]=Lf,$p=Bf,tE=Tg,nE=Cg,xl=i=>{const e=(31744&i)>>10,t=1023&i;return(i>>15?-1:1)*(e?e===31?t?NaN:1/0:Math.pow(2,e-15)*(1+t/1024):t/1024*6103515625e-14)};class qn{static decodeSectionHeadersFromHeaderLines(e){const t=Lf.reduce((u,f)=>(u[Mg[f]]=f,u),{}),n=pl.reduce((u,f)=>(u[bg[f]]=f,u),{}),s=ct.getHeaderSectionNames(e);let r;for(let u=0;u<s.length;u++)s[u]==="codebook_centers"&&(r=u);let o=0,a=!1;const l=[];let c=0;for(;!a;){let u;c===r?u=ct.decodeSectionHeader(e,n,o):u=ct.decodeSectionHeader(e,t,o),a=u.endOfHeader,o=u.headerEndLine+1,a||(u.splatCount=u.vertexCount,u.bytesPerSplat=u.bytesPerVertex),l.push(u),c++}return l}static decodeSectionHeadersFromHeaderText(e){const t=ct.convertHeaderTextToLines(e);return qn.decodeSectionHeadersFromHeaderLines(t)}static getSplatCountFromSectionHeaders(e){let t=0;for(let n of e)n.sectionName!=="codebook_centers"&&(t+=n.vertexCount);return t}static decodeHeaderFromHeaderText(e){const t=e.indexOf(ct.HeaderEndToken)+ct.HeaderEndToken.length+1,n=qn.decodeSectionHeadersFromHeaderText(e),s=qn.getSplatCountFromSectionHeaders(n);return{headerSizeBytes:t,sectionHeaders:n,splatCount:s}}static decodeHeaderFromBuffer(e){const t=ct.readHeaderFromBuffer(e);return qn.decodeHeaderFromHeaderText(t)}static findVertexData(e,t,n){let s=t.headerSizeBytes;for(let r=0;r<n&&r<t.sectionHeaders.length;r++){const o=t.sectionHeaders[r];s+=o.dataSizeBytes}return new DataView(e,s,t.sectionHeaders[n].dataSizeBytes)}static decodeCodeBook(e,t){const n=[],s=[];for(let r=0;r<t.vertexCount;r++){ct.readVertex(e,t,r,0,pl,n);for(let o of pl){const a=pl[o];let l=s[a];l||(s[a]=l=[]),l.push(n[o])}}for(let r=0;r<s.length;r++){const o=s[r],a=.28209479177387814;for(let l=0;l<o.length;l++){const c=xl(o[l]);r===Qp?o[l]=Math.round(1/(1+Math.exp(-c))*255):r===ml?o[l]=Math.round((.5+a*c)*255):r===gl?o[l]=Math.exp(c):o[l]=c}}return s}static decodeSectionSplatData(e,t,n,s,r){r=Math.min(r,n.sphericalHarmonicsDegree);const o=new we(r);for(let a=0;a<t;a++){const l=qn.parseToUncompressedSplat(e,a,n,s,0,r);o.addSplat(l)}return o}static parseToUncompressedSplat=(function(){let e=[];const t=new Mt,n=we.OFFSET.X,s=we.OFFSET.Y,r=we.OFFSET.Z,o=we.OFFSET.SCALE0,a=we.OFFSET.SCALE1,l=we.OFFSET.SCALE2,c=we.OFFSET.ROTATION0,u=we.OFFSET.ROTATION1,f=we.OFFSET.ROTATION2,h=we.OFFSET.ROTATION3,d=we.OFFSET.FDC0,x=we.OFFSET.FDC1,p=we.OFFSET.FDC2,g=we.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=we.OFFSET.FRC0+_;return function(_,S,A,y,b=0,v=0){v=Math.min(v,A.sphericalHarmonicsDegree),qn.readSplat(_,A,S,b,e);const E=we.createSplat(v);if(e[Kp]!==void 0?(E[o]=y[gl][e[Kp]],E[a]=y[gl][e[qC]],E[l]=y[gl][e[YC]]):(E[o]=.01,E[a]=.01,E[l]=.01),e[Bf]!==void 0?(E[d]=y[ml][e[Bf]],E[x]=y[ml][e[Tg]],E[p]=y[ml][e[Cg]]):e[$p]!==void 0?(E[d]=e[$p]*255,E[x]=e[tE]*255,E[p]=e[nE]*255):(E[d]=0,E[x]=0,E[p]=0),e[jp]!==void 0&&(E[g]=y[Qp][e[jp]]),E[d]=Rt(Math.floor(E[d]),0,255),E[x]=Rt(Math.floor(E[x]),0,255),E[p]=Rt(Math.floor(E[p]),0,255),E[g]=Rt(Math.floor(E[g]),0,255),v>=1&&A.sphericalHarmonicsDegree>=1){for(let B=0;B<9;B++){const N=y[GC+B%3];E[m[B]]=N[e[A.sphericalHarmonicsDegree1Fields[B]]]}if(v>=2&&A.sphericalHarmonicsDegree>=2)for(let B=0;B<15;B++){const N=y[WC+B%5];E[m[9+B]]=N[e[A.sphericalHarmonicsDegree2Fields[B]]]}}const M=y[XC][e[QC]],T=y[_u][e[KC]],I=y[_u][e[jC]],P=y[_u][e[$C]];return t.set(M,T,I,P),t.normalize(),E[c]=t.x,E[u]=t.y,E[f]=t.z,E[h]=t.w,E[n]=xl(e[ZC]),E[s]=xl(e[JC]),E[r]=xl(e[eE]),E}})();static readSplat(e,t,n,s,r){return ct.readVertex(e,t,n,s,Lf,r,!1)}static parseToUncompressedSplatArray(e,t=0){const n=[],s=qn.decodeHeaderFromBuffer(e,t);let r;for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName==="codebook_centers"){const c=qn.findVertexData(e,s,a);r=qn.decodeCodeBook(c,l)}}for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName!=="codebook_centers"){const c=l.vertexCount,u=qn.findVertexData(e,s,a),f=qn.decodeSectionSplatData(u,c,l,r,t);n.push(f)}}const o=new we(t);for(let a of n)for(let l of a.splats)o.addSplat(l);return o}}class Zp{static parseToUncompressedSplatArray(e,t=0){const n=ct.determineHeaderFormatFromPlyBuffer(e);if(n===An.PlayCanvasCompressed)return at.parseToUncompressedSplatArray(e,t);if(n===An.INRIAV1)return rn.parseToUncompressedSplatArray(e,t);if(n===An.INRIAV2)return qn.parseToUncompressedSplatArray(e,t)}static parseToUncompressedSplatBuffer(e,t=0){const n=ct.determineHeaderFormatFromPlyBuffer(e);if(n===An.PlayCanvasCompressed)return at.parseToUncompressedSplatBuffer(e,t);if(n===An.INRIAV1)return rn.parseToUncompressedSplatBuffer(e,t);if(n===An.INRIAV2)throw new Error("parseToUncompressedSplatBuffer() is not implemented for INRIA V2 PLY files")}}class Oh{constructor(e,t,n,s){this.sectionCount=e,this.sectionFilters=t,this.groupingParameters=n,this.partitionGenerator=s}partitionUncompressedSplatArray(e){let t,n,s;if(this.partitionGenerator){const o=this.partitionGenerator(e);t=o.groupingParameters,n=o.sectionCount,s=o.sectionFilters}else t=this.groupingParameters,n=this.sectionCount,s=this.sectionFilters;const r=[];for(let o=0;o<n;o++){const a=new we(e.sphericalHarmonicsDegree),l=s[o];for(let c=0;c<e.splatCount;c++)l(c)&&a.addSplat(e.splats[c]);r.push(a)}return{splatArrays:r,parameters:t}}static getStandardPartitioner(e=0,t=new F,n=j.BucketBlockSize,s=j.BucketSize){const r=o=>{const a=we.OFFSET.X,l=we.OFFSET.Y,c=we.OFFSET.Z;e<=0&&(e=o.splatCount);const u=new F,f=.5,h=m=>{m.x=Math.floor(m.x/f)*f,m.y=Math.floor(m.y/f)*f,m.z=Math.floor(m.z/f)*f};o.splats.forEach(m=>{u.set(m[a],m[l],m[c]).sub(t),h(u),m.centerDist=u.lengthSq()}),o.splats.sort((m,_)=>{let S=m.centerDist,A=_.centerDist;return S>A?1:-1});const d=[],x=[];e=Math.min(o.splatCount,e);const p=Math.ceil(o.splatCount/e);let g=0;for(let m=0;m<p;m++){let _=g;d.push(S=>S>=_&&S<_+e),x.push({blocksSize:n,bucketSize:s}),g+=e}return{sectionCount:d.length,sectionFilters:d,groupingParameters:x}};return new Oh(void 0,void 0,void 0,r)}}class Oa{constructor(e,t,n,s,r,o,a){this.splatPartitioner=e,this.alphaRemovalThreshold=t,this.compressionLevel=n,this.sectionSize=s,this.sceneCenter=r?new F().copy(r):void 0,this.blockSize=o,this.bucketSize=a}generateFromUncompressedSplatArray(e){const t=this.splatPartitioner.partitionUncompressedSplatArray(e);return j.generateFromUncompressedSplatArrays(t.splatArrays,this.alphaRemovalThreshold,this.compressionLevel,this.sceneCenter,this.blockSize,this.bucketSize,t.parameters)}static getStandardGenerator(e=1,t=1,n=0,s=new F,r=j.BucketBlockSize,o=j.BucketSize){const a=Oh.getStandardPartitioner(n,s,r,o);return new Oa(a,e,t,n,s,r,o)}}const Gt={Downloading:0,Processing:1,Done:2};class Yl extends Error{constructor(e){super(e)}}const Et={ProgressiveToSplatBuffer:0,ProgressiveToSplatArray:1,DownloadBeforeProcessing:2};function Jp(i,e){let t=0;for(let s of i)t+=s.sizeBytes;(!e||e.byteLength<t)&&(e=new ArrayBuffer(t));let n=0;for(let s of i)new Uint8Array(e,n,s.sizeBytes).set(s.data),n+=s.sizeBytes;return e}function em(i,e,t,n,s,r,o,a){return e?Oa.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):j.generateFromUncompressedSplatArrays([i],t,0,new F)}class Nh{static loadFromURL(e,t,n,s,r,o,a=!0,l=0,c,u,f,h,d){let x;!n&&!a?x=Et.DownloadBeforeProcessing:a?x=Et.ProgressiveToSplatArray:x=Et.ProgressiveToSplatBuffer;const p=gt.ProgressiveLoadSectionSize,g=j.HeaderSizeBytes+j.SectionHeaderSizeBytes,m=1;let _,S,A,y,b,v=0,E=0,M=0,T=!1,I=!1,P=!1;const B=Eh();let N=0,G=0,V=0,q=0,X="",ee=null,ce=[],be;const Re=new TextDecoder,Fe=(Oe,Ne,J)=>{const ne=Oe>=100;if(J&&(ce.push({data:J,sizeBytes:J.byteLength,startBytes:V,endBytes:V+J.byteLength}),V+=J.byteLength),x===Et.DownloadBeforeProcessing)ne&&B.resolve(ce);else{if(T){if(_===An.PlayCanvasCompressed&&!I){const xe=ee.headerSizeBytes+ee.chunkElement.storageSizeBytes;b=Jp(ce,b),b.byteLength>=xe&&(at.readElementData(ee.chunkElement,b,ee.headerSizeBytes),N=xe,G=xe,I=!0)}}else if(X+=Re.decode(J),ct.checkTextForEndHeader(X)){if(_=ct.determineHeaderFormatFromHeaderText(X),_===An.INRIAV1)ee=rn.decodeHeaderText(X),l=Math.min(l,ee.sphericalHarmonicsDegree),v=ee.splatCount,I=!0,q=ee.headerSizeBytes+ee.bytesPerSplat*v;else if(_===An.PlayCanvasCompressed){if(ee=at.decodeHeaderText(X),l=Math.min(l,ee.sphericalHarmonicsDegree),x===Et.ProgressiveToSplatBuffer&&l>0)throw new Yl("PlyLoader.loadFromURL() -> Selected PLY format has spherical harmonics data that cannot be progressively loaded.");v=ee.vertexElement.count,q=ee.headerSizeBytes+ee.bytesPerSplat*v+ee.chunkElement.storageSizeBytes}else{if(x===Et.ProgressiveToSplatBuffer)throw new Yl("PlyLoader.loadFromURL() -> Selected PLY format cannot be progressively loaded.");x=Et.DownloadBeforeProcessing;return}if(x===Et.ProgressiveToSplatBuffer){const xe=j.CompressionLevels[0].SphericalHarmonicsDegrees[l],Be=g+xe.BytesPerSplat*v;A=new ArrayBuffer(Be),j.writeHeaderToBuffer({versionMajor:j.CurrentMajorVersion,versionMinor:j.CurrentMinorVersion,maxSectionCount:m,sectionCount:m,maxSplatCount:v,splatCount:0,compressionLevel:0,sceneCenter:new F},A)}else be=new we(l);N=ee.headerSizeBytes,G=ee.headerSizeBytes,T=!0}if(T&&I&&ce.length>0&&(S=Jp(ce,S),V-N>p||V>=q&&!P||ne)){const Be=P?ee.sphericalHarmonicsPerSplat:ee.bytesPerSplat,Ve=(P?V:Math.min(q,V))-G,L=Math.floor(Ve/Be),U=L*Be,Y=V-G-U,w=G-ce[0].startBytes,oe=new DataView(S,w,U);if(P)_===An.PlayCanvasCompressed&&x===Et.ProgressiveToSplatArray&&(at.parseSphericalHarmonicsToUncompressedSplatArraySection(ee.chunkElement,ee.shElement,M,M+L-1,oe,0,l,ee.sphericalHarmonicsDegree,be),M+=L);else{if(x===Et.ProgressiveToSplatBuffer){const re=j.CompressionLevels[0].SphericalHarmonicsDegrees[l],pe=E*re.BytesPerSplat+g;_===An.PlayCanvasCompressed?at.parseToUncompressedSplatBufferSection(ee.chunkElement,ee.vertexElement,0,L-1,E,oe,A,pe):rn.parseToUncompressedSplatBufferSection(ee,0,L-1,oe,0,A,pe,l)}else _===An.PlayCanvasCompressed?at.parseToUncompressedSplatArraySection(ee.chunkElement,ee.vertexElement,0,L-1,E,oe,be):rn.parseToUncompressedSplatArraySection(ee,0,L-1,oe,0,be,l);E+=L,x===Et.ProgressiveToSplatBuffer&&(y||(j.writeSectionHeaderToBuffer({maxSplatCount:v,splatCount:E,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:l},0,A,j.HeaderSizeBytes),y=new j(A,!1)),y.updateLoadedCounts(1,E)),V>=q&&(P=!0)}if(Y===0)ce=[];else{let re=[],pe=0;for(let se=ce.length-1;se>=0;se--){const me=ce[se];if(pe+=me.sizeBytes,re.unshift(me),pe>=Y)break}ce=re}N+=p,G+=U}s&&y&&s(y,ne),ne&&(x===Et.ProgressiveToSplatBuffer?B.resolve(y):B.resolve(be))}t&&t(Oe,Ne,Gt.Downloading)};return t&&t(0,"0%",Gt.Downloading),_c(e,Fe,!1,c).then(()=>(t&&t(0,"0%",Gt.Processing),B.promise.then(Oe=>{if(t&&t(100,"100%",Gt.Done),x===Et.DownloadBeforeProcessing){const Ne=ce.map(J=>J.data);return new Blob(Ne).arrayBuffer().then(J=>Nh.loadFromFileData(J,r,o,a,l,u,f,h,d))}else return x===Et.ProgressiveToSplatBuffer?Oe:jn(()=>em(Oe,a,r,o,u,f,h,d))})))}static loadFromFileData(e,t,n,s,r=0,o,a,l,c){return s?jn(()=>Zp.parseToUncompressedSplatArray(e,r)).then(u=>em(u,s,t,n,o,a,l,c)):jn(()=>Zp.parseToUncompressedSplatBuffer(e,r))}}const iE=i=>new ReadableStream({async start(e){e.enqueue(i),e.close()}});async function sE(i){try{const e=iE(i);if(!e)throw new Error("Failed to create stream from data");return await rE(e)}catch(e){throw console.error("Error decompressing gzipped data:",e),e}}async function rE(i){const e=i.pipeThrough(new DecompressionStream("gzip")),n=await new Response(e).arrayBuffer();return new Uint8Array(n)}const oE=1347635022,aE=1,lE=.15;function cE(i){const e=i>>15&1,t=i>>10&31,n=i&1023,s=e===1?-1:1;return t===0?s*Math.pow(2,-14)*n/1024:t===31?n!==0?NaN:s*(1/0):s*Math.pow(2,t-15)*(1+n/1024)}function uE(i){return(i-128)/128}function gr(i){switch(i){case 0:return 0;case 1:return 3;case 2:return 8;case 3:return 15;default:return console.error(`[SPZ: ERROR] Unsupported SH degree: ${i}`),0}}const fE=(function(){let i=[];const e=new Mt,t=we.OFFSET.X,n=we.OFFSET.Y,s=we.OFFSET.Z,r=we.OFFSET.SCALE0,o=we.OFFSET.SCALE1,a=we.OFFSET.SCALE2,l=we.OFFSET.ROTATION0,c=we.OFFSET.ROTATION1,u=we.OFFSET.ROTATION2,f=we.OFFSET.ROTATION3,h=we.OFFSET.FDC0,d=we.OFFSET.FDC1,x=we.OFFSET.FDC2,p=we.OFFSET.OPACITY,g=[gr(0),gr(1),gr(2),gr(3)],m=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(_,S,A){A=Math.min(S,A);const y=we.createSplat(A);_.scale[0]!==void 0?(y[r]=_.scale[0],y[o]=_.scale[1],y[a]=_.scale[2]):(y[r]=.01,y[o]=.01,y[a]=.01),_.color[0]!==void 0?(y[h]=_.color[0],y[d]=_.color[1],y[x]=_.color[2]):i[RED]!==void 0?(y[h]=i[RED]*255,y[d]=i[GREEN]*255,y[x]=i[BLUE]*255):(y[h]=0,y[d]=0,y[x]=0),_.alpha!==void 0&&(y[p]=_.alpha),y[h]=Rt(Math.floor(y[h]),0,255),y[d]=Rt(Math.floor(y[d]),0,255),y[x]=Rt(Math.floor(y[x]),0,255),y[p]=Rt(Math.floor(y[p]),0,255);let b=g[A],v=g[S];for(let E=0;E<3;++E)for(let M=0;M<15;++M){const T=m[E*15+M];M<b&&M<v&&(y[we.OFFSET.FRC0+T]=_.sh[E*v+M])}return e.set(_.rotation[3],_.rotation[0],_.rotation[1],_.rotation[2]),e.normalize(),y[l]=e.x,y[c]=e.y,y[u]=e.z,y[f]=e.w,y[t]=_.position[0],y[n]=_.position[1],y[s]=_.position[2],y}})();function hE(i,e,t,n){return!(i.positions.length!==e*3*(n?2:3)||i.scales.length!==e*3||i.rotations.length!==e*3||i.alphas.length!==e||i.colors.length!==e*3||i.sh.length!==e*t*3)}function tm(i,e,t,n,s){e=Math.min(e,i.shDegree);const r=i.numPoints,o=gr(i.shDegree),a=i.positions.length===r*3*2;if(!hE(i,r,o,a))return null;const l={position:[],scale:[],rotation:[],alpha:void 0,color:[],sh:[]};let c;a&&(c=new Uint16Array(i.positions.buffer,i.positions.byteOffset,r*3));const u=1/(1<<i.fractionalBits),f=gr(i.shDegree),h=.28209479177387814;for(let d=0;d<r;d++){if(a)for(let _=0;_<3;_++)l.position[_]=cE(c[d*3+_]);else for(let _=0;_<3;_++){const S=d*9+_*3;let A=i.positions[S];A|=i.positions[S+1]<<8,A|=i.positions[S+2]<<16,A|=A&8388608?4278190080:0,l.position[_]=A*u}for(let _=0;_<3;_++)l.scale[_]=Math.exp(i.scales[d*3+_]/16-10);const x=i.rotations.subarray(d*3,d*3+3),p=[x[0]/127.5-1,x[1]/127.5-1,x[2]/127.5-1];l.rotation[0]=p[0],l.rotation[1]=p[1],l.rotation[2]=p[2];const g=p[0]*p[0]+p[1]*p[1]+p[2]*p[2];l.rotation[3]=Math.sqrt(Math.max(0,1-g)),l.alpha=Math.floor(i.alphas[d]);for(let _=0;_<3;_++)l.color[_]=Math.floor(((i.colors[d*3+_]/255-.5)/lE*h+.5)*255);for(let _=0;_<3;_++)for(let S=0;S<f;S++)l.sh[_*f+S]=uE(i.sh[f*3*d+S*3+_]);const m=fE(l,i.shDegree,e);if(t){const _=j.CompressionLevels[0].SphericalHarmonicsDegrees[e].BytesPerSplat,S=d*_+s;j.writeSplatDataToSectionBuffer(m,n,S,0,e)}else n.addSplat(m)}}const dE=16,pE=1e7;function mE(i){const e=new DataView(i);let t=0;const n={magic:e.getUint32(t,!0),version:e.getUint32(t+4,!0),numPoints:e.getUint32(t+8,!0),shDegree:e.getUint8(t+12),fractionalBits:e.getUint8(t+13),flags:e.getUint8(t+14),reserved:e.getUint8(t+15)};if(t+=dE,n.magic!==oE)return console.error("[SPZ ERROR] deserializePackedGaussians: header not found"),null;if(n.version<1||n.version>2)return console.error(`[SPZ ERROR] deserializePackedGaussians: version not supported: ${n.version}`),null;if(n.numPoints>pE)return console.error(`[SPZ ERROR] deserializePackedGaussians: Too many points: ${n.numPoints}`),null;if(n.shDegree>3)return console.error(`[SPZ ERROR] deserializePackedGaussians: Unsupported SH degree: ${n.shDegree}`),null;const s=n.numPoints,r=gr(n.shDegree),o=n.version===1,a={numPoints:s,shDegree:n.shDegree,fractionalBits:n.fractionalBits,antialiased:(n.flags&aE)!==0,positions:new Uint8Array(s*3*(o?2:3)),scales:new Uint8Array(s*3),rotations:new Uint8Array(s*3),alphas:new Uint8Array(s),colors:new Uint8Array(s*3),sh:new Uint8Array(s*r*3)};try{const l=new Uint8Array(i);let c=a.positions.length,u=t;if(a.positions.set(l.slice(u,u+c)),u+=c,a.alphas.set(l.slice(u,u+a.alphas.length)),u+=a.alphas.length,a.colors.set(l.slice(u,u+a.colors.length)),u+=a.colors.length,a.scales.set(l.slice(u,u+a.scales.length)),u+=a.scales.length,a.rotations.set(l.slice(u,u+a.rotations.length)),u+=a.rotations.length,a.sh.set(l.slice(u,u+a.sh.length)),u+a.sh.length!==i.byteLength)return console.error("[SPZ ERROR] deserializePackedGaussians: incorrect buffer size"),null}catch(l){return console.error("[SPZ ERROR] deserializePackedGaussians: read error",l),null}return a}async function gE(i){try{const e=await sE(i);return mE(e.buffer)}catch(e){return console.error("[SPZ ERROR] loadSpzPacked: decompression error",e),null}}class zh{static loadFromURL(e,t,n,s,r=!0,o=0,a,l,c,u,f){return t&&t(0,"0%",Gt.Downloading),_c(e,t,!0,a).then(h=>(t&&t(0,"0%",Gt.Processing),zh.loadFromFileData(h,n,s,r,o,l,c,u,f)))}static async loadFromFileData(e,t,n,s,r=0,o,a,l,c){await jn();const u=await gE(e);r=Math.min(u.shDegree,r);const f=new we(r);if(s)return tm(u,r,!1,f,0),Oa.getStandardGenerator(t,n,o,a,l,c).generateFromUncompressedSplatArray(f);{const{splatBuffer:h,splatBufferDataOffsetBytes:d}=j.preallocateUncompressed(u.numPoints,r);return tm(u,r,!0,h.bufferData,d),h}}}class pt{static RowSizeBytes=32;static CenterSizeBytes=12;static ScaleSizeBytes=12;static RotationSizeBytes=4;static ColorSizeBytes=4;static parseToUncompressedSplatBufferSection(e,t,n,s,r,o){const a=j.CompressionLevels[0].BytesPerCenter,l=j.CompressionLevels[0].BytesPerScale,c=j.CompressionLevels[0].BytesPerRotation,u=j.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat;for(let f=e;f<=t;f++){const h=f*pt.RowSizeBytes+s,d=new Float32Array(n,h,3),x=new Float32Array(n,h+pt.CenterSizeBytes,3),p=new Uint8Array(n,h+pt.CenterSizeBytes+pt.ScaleSizeBytes,4),g=new Uint8Array(n,h+pt.CenterSizeBytes+pt.ScaleSizeBytes+pt.RotationSizeBytes,4),m=new Mt((g[1]-128)/128,(g[2]-128)/128,(g[3]-128)/128,(g[0]-128)/128);m.normalize();const _=f*u+o,S=new Float32Array(r,_,3),A=new Float32Array(r,_+a,3),y=new Float32Array(r,_+a+l,4),b=new Uint8Array(r,_+a+l+c,4);S[0]=d[0],S[1]=d[1],S[2]=d[2],A[0]=x[0],A[1]=x[1],A[2]=x[2],y[0]=m.w,y[1]=m.x,y[2]=m.y,y[3]=m.z,b[0]=p[0],b[1]=p[1],b[2]=p[2],b[3]=p[3]}}static parseToUncompressedSplatArraySection(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*pt.RowSizeBytes+s,l=new Float32Array(n,a,3),c=new Float32Array(n,a+pt.CenterSizeBytes,3),u=new Uint8Array(n,a+pt.CenterSizeBytes+pt.ScaleSizeBytes,4),f=new Uint8Array(n,a+pt.CenterSizeBytes+pt.ScaleSizeBytes+pt.RotationSizeBytes,4),h=new Mt((f[1]-128)/128,(f[2]-128)/128,(f[3]-128)/128,(f[0]-128)/128);h.normalize(),r.addSplatFromComonents(l[0],l[1],l[2],c[0],c[1],c[2],h.w,h.x,h.y,h.z,u[0],u[1],u[2],u[3])}}static parseStandardSplatToUncompressedSplatArray(e){const t=e.byteLength/pt.RowSizeBytes,n=new we;for(let s=0;s<t;s++){const r=s*pt.RowSizeBytes,o=new Float32Array(e,r,3),a=new Float32Array(e,r+pt.CenterSizeBytes,3),l=new Uint8Array(e,r+pt.CenterSizeBytes+pt.ScaleSizeBytes,4),c=new Uint8Array(e,r+pt.CenterSizeBytes+pt.ScaleSizeBytes+pt.ColorSizeBytes,4),u=new Mt((c[1]-128)/128,(c[2]-128)/128,(c[3]-128)/128,(c[0]-128)/128);u.normalize(),n.addSplatFromComonents(o[0],o[1],o[2],a[0],a[1],a[2],u.w,u.x,u.y,u.z,l[0],l[1],l[2],l[3])}return n}}function nm(i,e,t,n,s,r,o,a){return e?Oa.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):j.generateFromUncompressedSplatArrays([i],t,0,new F)}class kh{static loadFromURL(e,t,n,s,r,o,a=!0,l,c,u,f,h){let d=n?Et.ProgressiveToSplatBuffer:Et.ProgressiveToSplatArray;a&&(d=Et.ProgressiveToSplatArray);const x=j.HeaderSizeBytes+j.SectionHeaderSizeBytes,p=gt.ProgressiveLoadSectionSize,g=1;let m,_,S,A=0,y=0,b;const v=Eh();let E=0,M=0,T=[];const I=(P,B,N,G)=>{const V=P>=100;if(N&&T.push(N),d===Et.DownloadBeforeProcessing){V&&v.resolve(T);return}if(!G){if(n)throw new Yl("Cannon directly load .splat because no file size info is available.");d=Et.DownloadBeforeProcessing;return}if(!m){A=G/pt.RowSizeBytes,m=new ArrayBuffer(G);const q=j.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,X=x+q*A;d===Et.ProgressiveToSplatBuffer?(_=new ArrayBuffer(X),j.writeHeaderToBuffer({versionMajor:j.CurrentMajorVersion,versionMinor:j.CurrentMinorVersion,maxSectionCount:g,sectionCount:g,maxSplatCount:A,splatCount:y,compressionLevel:0,sceneCenter:new F},_)):b=new we(0)}if(N){new Uint8Array(m,M,N.byteLength).set(new Uint8Array(N)),M+=N.byteLength;const q=M-E;if(q>p||V){const ee=(V?q:p)/pt.RowSizeBytes,ce=y+ee;d===Et.ProgressiveToSplatBuffer?pt.parseToUncompressedSplatBufferSection(y,ce-1,m,0,_,x):pt.parseToUncompressedSplatArraySection(y,ce-1,m,0,b),y=ce,d===Et.ProgressiveToSplatBuffer&&(S||(j.writeSectionHeaderToBuffer({maxSplatCount:A,splatCount:y,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0},0,_,j.HeaderSizeBytes),S=new j(_,!1)),S.updateLoadedCounts(1,y),s&&s(S,V)),E+=p}}V&&(d===Et.ProgressiveToSplatBuffer?v.resolve(S):v.resolve(b)),t&&t(P,B,Gt.Downloading)};return t&&t(0,"0%",Gt.Downloading),_c(e,I,!1,l).then(()=>(t&&t(0,"0%",Gt.Processing),v.promise.then(P=>(t&&t(100,"100%",Gt.Done),d===Et.DownloadBeforeProcessing?new Blob(T).arrayBuffer().then(B=>kh.loadFromFileData(B,r,o,a,c,u,f,h)):d===Et.ProgressiveToSplatBuffer?P:jn(()=>nm(P,a,r,o,c,u,f,h))))))}static loadFromFileData(e,t,n,s,r,o,a,l){return jn(()=>{const c=pt.parseStandardSplatToUncompressedSplatArray(e);return nm(c,s,t,n,r,o,a,l)})}}class ra{static checkVersion(e){const t=j.CurrentMajorVersion,n=j.CurrentMinorVersion,s=j.parseHeader(e);if(s.versionMajor===t&&s.versionMinor>=n||s.versionMajor>t)return!0;throw new Error(`KSplat version not supported: v${s.versionMajor}.${s.versionMinor}. Minimum required: v${t}.${n}`)}static loadFromURL(e,t,n,s,r){let o,a,l,c,u=!1,f=!1,h,d=[],x=!1,p=!1,g=0,m=0,_=0,S=!1,A=!1,y=!1,b=[];const v=Eh(),E=()=>{!u&&!f&&g>=j.HeaderSizeBytes&&(f=!0,new Blob(b).arrayBuffer().then(G=>{l=new ArrayBuffer(j.HeaderSizeBytes),new Uint8Array(l).set(new Uint8Array(G,0,j.HeaderSizeBytes)),ra.checkVersion(l),f=!1,u=!0,c=j.parseHeader(l),window.setTimeout(()=>{I()},1)}))};let M=0;const T=()=>{M===0&&(M++,window.setTimeout(()=>{M--,P()},1))},I=()=>{const N=()=>{p=!0,new Blob(b).arrayBuffer().then(V=>{p=!1,x=!0,h=new ArrayBuffer(c.maxSectionCount*j.SectionHeaderSizeBytes),new Uint8Array(h).set(new Uint8Array(V,j.HeaderSizeBytes,c.maxSectionCount*j.SectionHeaderSizeBytes)),d=j.parseSectionHeaders(c,h,0,!1);let q=0;for(let ee=0;ee<c.maxSectionCount;ee++)q+=d[ee].storageSizeBytes;const X=j.HeaderSizeBytes+c.maxSectionCount*j.SectionHeaderSizeBytes+q;if(!o){o=new ArrayBuffer(X);let ee=0;for(let ce=0;ce<b.length;ce++){const be=b[ce];new Uint8Array(o,ee,be.byteLength).set(new Uint8Array(be)),ee+=be.byteLength}}_=j.HeaderSizeBytes+j.SectionHeaderSizeBytes*c.maxSectionCount;for(let ee=0;ee<=d.length&&ee<c.maxSectionCount;ee++)_+=d[ee].storageSizeBytes;T()})};!p&&!x&&u&&g>=j.HeaderSizeBytes+j.SectionHeaderSizeBytes*c.maxSectionCount&&N()},P=()=>{if(y)return;y=!0;const N=()=>{if(y=!1,x){if(A)return;if(S=g>=_,g-m>gt.ProgressiveLoadSectionSize||S){m+=gt.ProgressiveLoadSectionSize,A=m>=_,a||(a=new j(o,!1));const V=j.HeaderSizeBytes+j.SectionHeaderSizeBytes*c.maxSectionCount;let q=0,X=0,ee=0;for(let Re=0;Re<c.maxSectionCount;Re++){const Fe=d[Re],Oe=q+Fe.partiallyFilledBucketCount*4+Fe.bucketStorageSizeBytes*Fe.bucketCount,Ne=V+Oe;if(m>=Ne){X++;const J=m-Ne,Be=j.CompressionLevels[c.compressionLevel].SphericalHarmonicsDegrees[Fe.sphericalHarmonicsDegree].BytesPerSplat;let Te=Math.floor(J/Be);Te=Math.min(Te,Fe.maxSplatCount),ee+=Te,a.updateLoadedCounts(X,ee),a.updateSectionLoadedCounts(Re,Te)}else break;q+=Fe.storageSizeBytes}s(a,A);const ce=m/_*100,be=ce.toFixed(2)+"%";t&&t(ce,be,Gt.Downloading),A?v.resolve(a):P()}}};window.setTimeout(N,gt.ProgressiveLoadSectionDelayDuration)};return _c(e,(N,G,V)=>{V&&(b.push(V),o&&new Uint8Array(o,g,V.byteLength).set(new Uint8Array(V)),g+=V.byteLength),n?(E(),I(),P()):t&&t(N,G,Gt.Downloading)},!n,r).then(N=>(t&&t(0,"0%",Gt.Processing),(n?v.promise:ra.loadFromFileData(N)).then(V=>(t&&t(100,"100%",Gt.Done),V))))}static loadFromFileData(e){return jn(()=>(ra.checkVersion(e),new j(e)))}static downloadFile=(function(){let e;return function(t,n){const s=new Blob([t.bufferData],{type:"application/octet-stream"});e||(e=document.createElement("a"),document.body.appendChild(e)),e.download=n,e.href=URL.createObjectURL(s),e.click()}})()}const Ln={Splat:0,KSplat:1,Ply:2,Spz:3},im=i=>i.endsWith(".ply")?Ln.Ply:i.endsWith(".splat")?Ln.Splat:i.endsWith(".ksplat")?Ln.KSplat:i.endsWith(".spz")?Ln.Spz:null,sm={type:"change"},vu={type:"start"},rm={type:"end"},_l=new mc,om=new as,xE=Math.cos(70*yh.DEG2RAD);let vl=class extends Ks{constructor(e,t){super(),this.object=e,this.domElement=t,this.domElement.style.touchAction="none",this.enabled=!0,this.target=new F,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"KeyA",UP:"KeyW",RIGHT:"KeyD",BOTTOM:"KeyS"},this.mouseButtons={LEFT:li.ROTATE,MIDDLE:li.DOLLY,RIGHT:li.PAN},this.touches={ONE:ci.ROTATE,TWO:ci.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this.getPolarAngle=function(){return a.phi},this.getAzimuthalAngle=function(){return a.theta},this.getDistance=function(){return this.object.position.distanceTo(this.target)},this.listenToKeyEvents=function(k){k.addEventListener("keydown",C),this._domElementKeyEvents=k},this.stopListenToKeyEvents=function(){this._domElementKeyEvents.removeEventListener("keydown",C),this._domElementKeyEvents=null},this.saveState=function(){n.target0.copy(n.target),n.position0.copy(n.object.position),n.zoom0=n.object.zoom},this.reset=function(){n.target.copy(n.target0),n.object.position.copy(n.position0),n.object.zoom=n.zoom0,this.clearDampedRotation(),this.clearDampedPan(),n.object.updateProjectionMatrix(),n.dispatchEvent(sm),n.update(),r=s.NONE},this.clearDampedRotation=function(){l.theta=0,l.phi=0},this.clearDampedPan=function(){u.set(0,0,0)},this.update=(function(){const k=new F,te=new Mt().setFromUnitVectors(e.up,new F(0,1,0)),_e=te.clone().invert(),H=new F,z=new Mt,he=new F,Me=2*Math.PI;return function(){te.setFromUnitVectors(e.up,new F(0,1,0)),_e.copy(te).invert();const ve=n.object.position;k.copy(ve).sub(n.target),k.applyQuaternion(te),a.setFromVector3(k),n.autoRotate&&r===s.NONE&&I(M()),n.enableDamping?(a.theta+=l.theta*n.dampingFactor,a.phi+=l.phi*n.dampingFactor):(a.theta+=l.theta,a.phi+=l.phi);let ge=n.minAzimuthAngle,Se=n.maxAzimuthAngle;isFinite(ge)&&isFinite(Se)&&(ge<-Math.PI?ge+=Me:ge>Math.PI&&(ge-=Me),Se<-Math.PI?Se+=Me:Se>Math.PI&&(Se-=Me),ge<=Se?a.theta=Math.max(ge,Math.min(Se,a.theta)):a.theta=a.theta>(ge+Se)/2?Math.max(ge,a.theta):Math.min(Se,a.theta)),a.phi=Math.max(n.minPolarAngle,Math.min(n.maxPolarAngle,a.phi)),a.makeSafe(),n.enableDamping===!0?n.target.addScaledVector(u,n.dampingFactor):n.target.add(u),n.zoomToCursor&&b||n.object.isOrthographicCamera?a.radius=ee(a.radius):a.radius=ee(a.radius*c),k.setFromSpherical(a),k.applyQuaternion(_e),ve.copy(n.target).add(k),n.object.lookAt(n.target),n.enableDamping===!0?(l.theta*=1-n.dampingFactor,l.phi*=1-n.dampingFactor,u.multiplyScalar(1-n.dampingFactor)):(l.set(0,0,0),u.set(0,0,0));let de=!1;if(n.zoomToCursor&&b){let le=null;if(n.object.isPerspectiveCamera){const Ce=k.length();le=ee(Ce*c);const ze=Ce-le;n.object.position.addScaledVector(A,ze),n.object.updateMatrixWorld()}else if(n.object.isOrthographicCamera){const Ce=new F(y.x,y.y,0);Ce.unproject(n.object),n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),de=!0;const ze=new F(y.x,y.y,0);ze.unproject(n.object),n.object.position.sub(ze).add(Ce),n.object.updateMatrixWorld(),le=k.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),n.zoomToCursor=!1;le!==null&&(this.screenSpacePanning?n.target.set(0,0,-1).transformDirection(n.object.matrix).multiplyScalar(le).add(n.object.position):(_l.origin.copy(n.object.position),_l.direction.set(0,0,-1).transformDirection(n.object.matrix),Math.abs(n.object.up.dot(_l.direction))<xE?e.lookAt(n.target):(om.setFromNormalAndCoplanarPoint(n.object.up,n.target),_l.intersectPlane(om,n.target))))}else n.object.isOrthographicCamera&&(n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),de=!0);return c=1,b=!1,de||H.distanceToSquared(n.object.position)>o||8*(1-z.dot(n.object.quaternion))>o||he.distanceToSquared(n.target)>0?(n.dispatchEvent(sm),H.copy(n.object.position),z.copy(n.object.quaternion),he.copy(n.target),de=!1,!0):!1}})(),this.dispose=function(){n.domElement.removeEventListener("contextmenu",fe),n.domElement.removeEventListener("pointerdown",pe),n.domElement.removeEventListener("pointercancel",me),n.domElement.removeEventListener("wheel",R),n.domElement.removeEventListener("pointermove",se),n.domElement.removeEventListener("pointerup",me),n._domElementKeyEvents!==null&&(n._domElementKeyEvents.removeEventListener("keydown",C),n._domElementKeyEvents=null)};const n=this,s={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6};let r=s.NONE;const o=1e-6,a=new ql,l=new ql;let c=1;const u=new F,f=new Pe,h=new Pe,d=new Pe,x=new Pe,p=new Pe,g=new Pe,m=new Pe,_=new Pe,S=new Pe,A=new F,y=new Pe;let b=!1;const v=[],E={};function M(){return 2*Math.PI/60/60*n.autoRotateSpeed}function T(){return Math.pow(.95,n.zoomSpeed)}function I(k){l.theta-=k}function P(k){l.phi-=k}const B=(function(){const k=new F;return function(_e,H){k.setFromMatrixColumn(H,0),k.multiplyScalar(-_e),u.add(k)}})(),N=(function(){const k=new F;return function(_e,H){n.screenSpacePanning===!0?k.setFromMatrixColumn(H,1):(k.setFromMatrixColumn(H,0),k.crossVectors(n.object.up,k)),k.multiplyScalar(_e),u.add(k)}})(),G=(function(){const k=new F;return function(_e,H){const z=n.domElement;if(n.object.isPerspectiveCamera){const he=n.object.position;k.copy(he).sub(n.target);let Me=k.length();Me*=Math.tan(n.object.fov/2*Math.PI/180),B(2*_e*Me/z.clientHeight,n.object.matrix),N(2*H*Me/z.clientHeight,n.object.matrix)}else n.object.isOrthographicCamera?(B(_e*(n.object.right-n.object.left)/n.object.zoom/z.clientWidth,n.object.matrix),N(H*(n.object.top-n.object.bottom)/n.object.zoom/z.clientHeight,n.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),n.enablePan=!1)}})();function V(k){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c/=k:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function q(k){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c*=k:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function X(k){if(!n.zoomToCursor)return;b=!0;const te=n.domElement.getBoundingClientRect(),_e=k.clientX-te.left,H=k.clientY-te.top,z=te.width,he=te.height;y.x=_e/z*2-1,y.y=-(H/he)*2+1,A.set(y.x,y.y,1).unproject(e).sub(e.position).normalize()}function ee(k){return Math.max(n.minDistance,Math.min(n.maxDistance,k))}function ce(k){f.set(k.clientX,k.clientY)}function be(k){X(k),m.set(k.clientX,k.clientY)}function Re(k){x.set(k.clientX,k.clientY)}function Fe(k){h.set(k.clientX,k.clientY),d.subVectors(h,f).multiplyScalar(n.rotateSpeed);const te=n.domElement;I(2*Math.PI*d.x/te.clientHeight),P(2*Math.PI*d.y/te.clientHeight),f.copy(h),n.update()}function Oe(k){_.set(k.clientX,k.clientY),S.subVectors(_,m),S.y>0?V(T()):S.y<0&&q(T()),m.copy(_),n.update()}function Ne(k){p.set(k.clientX,k.clientY),g.subVectors(p,x).multiplyScalar(n.panSpeed),G(g.x,g.y),x.copy(p),n.update()}function J(k){X(k),k.deltaY<0?q(T()):k.deltaY>0&&V(T()),n.update()}function ne(k){let te=!1;switch(k.code){case n.keys.UP:k.ctrlKey||k.metaKey||k.shiftKey?P(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):G(0,n.keyPanSpeed),te=!0;break;case n.keys.BOTTOM:k.ctrlKey||k.metaKey||k.shiftKey?P(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):G(0,-n.keyPanSpeed),te=!0;break;case n.keys.LEFT:k.ctrlKey||k.metaKey||k.shiftKey?I(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):G(n.keyPanSpeed,0),te=!0;break;case n.keys.RIGHT:k.ctrlKey||k.metaKey||k.shiftKey?I(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):G(-n.keyPanSpeed,0),te=!0;break}te&&(k.preventDefault(),n.update())}function xe(){if(v.length===1)f.set(v[0].pageX,v[0].pageY);else{const k=.5*(v[0].pageX+v[1].pageX),te=.5*(v[0].pageY+v[1].pageY);f.set(k,te)}}function Be(){if(v.length===1)x.set(v[0].pageX,v[0].pageY);else{const k=.5*(v[0].pageX+v[1].pageX),te=.5*(v[0].pageY+v[1].pageY);x.set(k,te)}}function Te(){const k=v[0].pageX-v[1].pageX,te=v[0].pageY-v[1].pageY,_e=Math.sqrt(k*k+te*te);m.set(0,_e)}function Ve(){n.enableZoom&&Te(),n.enablePan&&Be()}function L(){n.enableZoom&&Te(),n.enableRotate&&xe()}function U(k){if(v.length==1)h.set(k.pageX,k.pageY);else{const _e=Ue(k),H=.5*(k.pageX+_e.x),z=.5*(k.pageY+_e.y);h.set(H,z)}d.subVectors(h,f).multiplyScalar(n.rotateSpeed);const te=n.domElement;I(2*Math.PI*d.x/te.clientHeight),P(2*Math.PI*d.y/te.clientHeight),f.copy(h)}function Y(k){if(v.length===1)p.set(k.pageX,k.pageY);else{const te=Ue(k),_e=.5*(k.pageX+te.x),H=.5*(k.pageY+te.y);p.set(_e,H)}g.subVectors(p,x).multiplyScalar(n.panSpeed),G(g.x,g.y),x.copy(p)}function w(k){const te=Ue(k),_e=k.pageX-te.x,H=k.pageY-te.y,z=Math.sqrt(_e*_e+H*H);_.set(0,z),S.set(0,Math.pow(_.y/m.y,n.zoomSpeed)),V(S.y),m.copy(_)}function oe(k){n.enableZoom&&w(k),n.enablePan&&Y(k)}function re(k){n.enableZoom&&w(k),n.enableRotate&&U(k)}function pe(k){n.enabled!==!1&&(v.length===0&&(n.domElement.setPointerCapture(k.pointerId),n.domElement.addEventListener("pointermove",se),n.domElement.addEventListener("pointerup",me)),Z(k),k.pointerType==="touch"?W(k):ie(k))}function se(k){n.enabled!==!1&&(k.pointerType==="touch"?$(k):Ae(k))}function me(k){Ie(k),v.length===0&&(n.domElement.releasePointerCapture(k.pointerId),n.domElement.removeEventListener("pointermove",se),n.domElement.removeEventListener("pointerup",me)),n.dispatchEvent(rm),r=s.NONE}function ie(k){let te;switch(k.button){case 0:te=n.mouseButtons.LEFT;break;case 1:te=n.mouseButtons.MIDDLE;break;case 2:te=n.mouseButtons.RIGHT;break;default:te=-1}switch(te){case li.DOLLY:if(n.enableZoom===!1)return;be(k),r=s.DOLLY;break;case li.ROTATE:if(k.ctrlKey||k.metaKey||k.shiftKey){if(n.enablePan===!1)return;Re(k),r=s.PAN}else{if(n.enableRotate===!1)return;ce(k),r=s.ROTATE}break;case li.PAN:if(k.ctrlKey||k.metaKey||k.shiftKey){if(n.enableRotate===!1)return;ce(k),r=s.ROTATE}else{if(n.enablePan===!1)return;Re(k),r=s.PAN}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(vu)}function Ae(k){switch(r){case s.ROTATE:if(n.enableRotate===!1)return;Fe(k);break;case s.DOLLY:if(n.enableZoom===!1)return;Oe(k);break;case s.PAN:if(n.enablePan===!1)return;Ne(k);break}}function R(k){n.enabled===!1||n.enableZoom===!1||r!==s.NONE||(k.preventDefault(),n.dispatchEvent(vu),J(k),n.dispatchEvent(rm))}function C(k){n.enabled===!1||n.enablePan===!1||ne(k)}function W(k){switch(ye(k),v.length){case 1:switch(n.touches.ONE){case ci.ROTATE:if(n.enableRotate===!1)return;xe(),r=s.TOUCH_ROTATE;break;case ci.PAN:if(n.enablePan===!1)return;Be(),r=s.TOUCH_PAN;break;default:r=s.NONE}break;case 2:switch(n.touches.TWO){case ci.DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;Ve(),r=s.TOUCH_DOLLY_PAN;break;case ci.DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;L(),r=s.TOUCH_DOLLY_ROTATE;break;default:r=s.NONE}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(vu)}function $(k){switch(ye(k),r){case s.TOUCH_ROTATE:if(n.enableRotate===!1)return;U(k),n.update();break;case s.TOUCH_PAN:if(n.enablePan===!1)return;Y(k),n.update();break;case s.TOUCH_DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;oe(k),n.update();break;case s.TOUCH_DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;re(k),n.update();break;default:r=s.NONE}}function fe(k){n.enabled!==!1&&k.preventDefault()}function Z(k){v.push(k)}function Ie(k){delete E[k.pointerId];for(let te=0;te<v.length;te++)if(v[te].pointerId==k.pointerId){v.splice(te,1);return}}function ye(k){let te=E[k.pointerId];te===void 0&&(te=new Pe,E[k.pointerId]=te),te.set(k.pageX,k.pageY)}function Ue(k){const te=k.pointerId===v[0].pointerId?v[1]:v[0];return E[te.pointerId]}n.domElement.addEventListener("contextmenu",fe),n.domElement.addEventListener("pointerdown",pe),n.domElement.addEventListener("pointercancel",me),n.domElement.addEventListener("wheel",R,{passive:!1}),this.update()}};const _E=(i,e,t,n,s)=>{const r=performance.now();let o=i.style.display==="none"?0:parseFloat(i.style.opacity);isNaN(o)&&(o=1);const a=window.setInterval(()=>{const c=performance.now()-r;let u=Math.min(c/n,1);u>.999&&(u=1);let f;e?(f=(1-u)*o,f<1e-4&&(f=0)):f=(1-o)*u+o,f>0?(i.style.display=t,i.style.opacity=f):i.style.display="none",u>=1&&(s&&s(),window.clearInterval(a))},16);return a},vE=500;class Hh{static elementIDGen=0;constructor(e,t){this.taskIDGen=0,this.elementID=Hh.elementIDGen++,this.tasks=[],this.message=e||"Loading...",this.container=t||document.body,this.spinnerContainerOuter=document.createElement("div"),this.spinnerContainerOuter.className=`spinnerOuterContainer${this.elementID}`,this.spinnerContainerOuter.style.display="none",this.spinnerContainerPrimary=document.createElement("div"),this.spinnerContainerPrimary.className=`spinnerContainerPrimary${this.elementID}`,this.spinnerPrimary=document.createElement("div"),this.spinnerPrimary.classList.add(`spinner${this.elementID}`,`spinnerPrimary${this.elementID}`),this.messageContainerPrimary=document.createElement("div"),this.messageContainerPrimary.classList.add(`messageContainer${this.elementID}`,`messageContainerPrimary${this.elementID}`),this.messageContainerPrimary.innerHTML=this.message,this.spinnerContainerMin=document.createElement("div"),this.spinnerContainerMin.className=`spinnerContainerMin${this.elementID}`,this.spinnerMin=document.createElement("div"),this.spinnerMin.classList.add(`spinner${this.elementID}`,`spinnerMin${this.elementID}`),this.messageContainerMin=document.createElement("div"),this.messageContainerMin.classList.add(`messageContainer${this.elementID}`,`messageContainerMin${this.elementID}`),this.messageContainerMin.innerHTML=this.message,this.spinnerContainerPrimary.appendChild(this.spinnerPrimary),this.spinnerContainerPrimary.appendChild(this.messageContainerPrimary),this.spinnerContainerOuter.appendChild(this.spinnerContainerPrimary),this.spinnerContainerMin.appendChild(this.spinnerMin),this.spinnerContainerMin.appendChild(this.messageContainerMin),this.spinnerContainerOuter.appendChild(this.spinnerContainerMin);const n=document.createElement("style");n.innerHTML=`

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

        `,this.spinnerContainerOuter.appendChild(n),this.container.appendChild(this.spinnerContainerOuter),this.setMinimized(!1,!0),this.fadeTransitions=[]}addTask(e){const t={message:e,id:this.taskIDGen++};return this.tasks.push(t),this.update(),t.id}removeTask(e){let t=0;for(let n of this.tasks){if(n.id===e){this.tasks.splice(t,1);break}t++}this.update()}removeAllTasks(){this.tasks=[],this.update()}setMessageForTask(e,t){for(let n of this.tasks)if(n.id===e){n.message=t;break}this.update()}update(){this.tasks.length>0?(this.show(),this.setMessage(this.tasks[this.tasks.length-1].message)):this.hide()}show(){this.spinnerContainerOuter.style.display="block",this.visible=!0}hide(){this.spinnerContainerOuter.style.display="none",this.visible=!1}setContainer(e){this.container&&this.spinnerContainerOuter.parentElement===this.container&&this.container.removeChild(this.spinnerContainerOuter),e&&(this.container=e,this.container.appendChild(this.spinnerContainerOuter),this.spinnerContainerOuter.style.zIndex=this.container.style.zIndex+1)}setMinimized(e,t){const n=(s,r,o,a,l)=>{o?s.style.display=r?a:"none":this.fadeTransitions[l]=_E(s,!r,a,vE,()=>{this.fadeTransitions[l]=null})};n(this.spinnerContainerPrimary,!e,t,"block",0),n(this.spinnerContainerMin,e,t,"flex",1),this.minimized=e}setMessage(e){this.messageContainerPrimary.innerHTML=e,this.messageContainerMin.innerHTML=e}}class SE{constructor(e){this.idGen=0,this.tasks=[],this.container=e||document.body,this.progressBarContainerOuter=document.createElement("div"),this.progressBarContainerOuter.className="progressBarOuterContainer",this.progressBarContainerOuter.style.display="none",this.progressBarBox=document.createElement("div"),this.progressBarBox.className="progressBarBox",this.progressBarBackground=document.createElement("div"),this.progressBarBackground.className="progressBarBackground",this.progressBar=document.createElement("div"),this.progressBar.className="progressBar",this.progressBarBackground.appendChild(this.progressBar),this.progressBarBox.appendChild(this.progressBarBackground),this.progressBarContainerOuter.appendChild(this.progressBarBox);const t=document.createElement("style");t.innerHTML=`

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

        `,this.progressBarContainerOuter.appendChild(t),this.container.appendChild(this.progressBarContainerOuter)}show(){this.progressBarContainerOuter.style.display="block"}hide(){this.progressBarContainerOuter.style.display="none"}setProgress(e){this.progressBar.style.width=e+"%"}setContainer(e){this.container&&this.progressBarContainerOuter.parentElement===this.container&&this.container.removeChild(this.progressBarContainerOuter),e&&(this.container=e,this.container.appendChild(this.progressBarContainerOuter),this.progressBarContainerOuter.style.zIndex=this.container.style.zIndex+1)}}class AE{constructor(e){this.container=e||document.body,this.infoCells={};const t=[["Camera position","cameraPosition"],["Camera look-at","cameraLookAt"],["Camera up","cameraUp"],["Camera mode","orthographicCamera"],["Cursor position","cursorPosition"],["FPS","fps"],["Rendering:","renderSplatCount"],["Sort time","sortTime"],["Render window","renderWindow"],["Focal adjustment","focalAdjustment"],["Splat scale","splatScale"],["Point cloud mode","pointCloudMode"]];this.infoPanelContainer=document.createElement("div");const n=document.createElement("style");n.innerHTML=`

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

        `,this.infoPanelContainer.append(n),this.infoPanel=document.createElement("div"),this.infoPanel.className="infoPanel";const s=document.createElement("div");s.style.display="table";for(let r of t){const o=document.createElement("div");o.style.display="table-row",o.className="info-panel-row";const a=document.createElement("div");a.style.display="table-cell",a.innerHTML=`${r[0]}: `,a.classList.add("info-panel-cell","label-cell");const l=document.createElement("div");l.style.display="table-cell",l.style.width="10px",l.innerHTML=" ",l.className="info-panel-cell";const c=document.createElement("div");c.style.display="table-cell",c.innerHTML="",c.className="info-panel-cell",this.infoCells[r[1]]=c,o.appendChild(a),o.appendChild(l),o.appendChild(c),s.appendChild(o)}this.infoPanel.appendChild(s),this.infoPanelContainer.append(this.infoPanel),this.infoPanelContainer.style.display="none",this.container.appendChild(this.infoPanelContainer),this.visible=!1}update=function(e,t,n,s,r,o,a,l,c,u,f,h,d,x){const p=`${t.x.toFixed(5)}, ${t.y.toFixed(5)}, ${t.z.toFixed(5)}`;if(this.infoCells.cameraPosition.innerHTML!==p&&(this.infoCells.cameraPosition.innerHTML=p),n){const m=n,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cameraLookAt.innerHTML!==_&&(this.infoCells.cameraLookAt.innerHTML=_)}const g=`${s.x.toFixed(5)}, ${s.y.toFixed(5)}, ${s.z.toFixed(5)}`;if(this.infoCells.cameraUp.innerHTML!==g&&(this.infoCells.cameraUp.innerHTML=g),this.infoCells.orthographicCamera.innerHTML=r?"Orthographic":"Perspective",o){const m=o,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cursorPosition.innerHTML=_}else this.infoCells.cursorPosition.innerHTML="N/A";this.infoCells.fps.innerHTML=a,this.infoCells.renderWindow.innerHTML=`${e.x} x ${e.y}`,this.infoCells.renderSplatCount.innerHTML=`${c} splats out of ${l} (${u.toFixed(2)}%)`,this.infoCells.sortTime.innerHTML=`${f.toFixed(3)} ms`,this.infoCells.focalAdjustment.innerHTML=`${h.toFixed(3)}`,this.infoCells.splatScale.innerHTML=`${d.toFixed(3)}`,this.infoCells.pointCloudMode.innerHTML=`${x}`};setContainer(e){this.container&&this.infoPanelContainer.parentElement===this.container&&this.container.removeChild(this.infoPanelContainer),e&&(this.container=e,this.container.appendChild(this.infoPanelContainer),this.infoPanelContainer.style.zIndex=this.container.style.zIndex+1)}show(){this.infoPanelContainer.style.display="block",this.visible=!0}hide(){this.infoPanelContainer.style.display="none",this.visible=!1}}const am=new F;class yE extends Kt{constructor(e=new F(0,0,1),t=new F(0,0,0),n=1,s=.1,r=16776960,o=n*.2,a=o*.2){super(),this.type="ArrowHelper";const l=new Ma(s,s,n,32);l.translate(0,n/2,0);const c=new Ma(0,a,o,32);c.translate(0,n,0),this.position.copy(t),this.line=new Yt(l,new Mr({color:r,toneMapped:!1})),this.line.matrixAutoUpdate=!1,this.add(this.line),this.cone=new Yt(c,new Mr({color:r,toneMapped:!1})),this.cone.matrixAutoUpdate=!1,this.add(this.cone),this.setDirection(e)}setDirection(e){if(e.y>.99999)this.quaternion.set(0,0,0,1);else if(e.y<-.99999)this.quaternion.set(1,0,0,0);else{am.set(e.z,0,-e.x).normalize();const t=Math.acos(e.y);this.quaternion.setFromAxisAngle(am,t)}}setColor(e){this.line.material.color.set(e),this.cone.material.color.set(e)}copy(e){return super.copy(e,!1),this.line.copy(e.line),this.cone.copy(e.cone),this}dispose(){this.line.geometry.dispose(),this.line.material.dispose(),this.cone.geometry.dispose(),this.cone.material.dispose()}}class oa{constructor(e){this.threeScene=e,this.splatRenderTarget=null,this.renderTargetCopyQuad=null,this.renderTargetCopyCamera=null,this.meshCursor=null,this.focusMarker=null,this.controlPlane=null,this.debugRoot=null,this.secondaryDebugRoot=null}updateSplatRenderTargetForRenderDimensions(e,t){this.destroySplatRendertarget(),this.splatRenderTarget=new Ws(e,t,{format:Mn,stencilBuffer:!1,depthBuffer:!0}),this.splatRenderTarget.depthTexture=new Mh(e,t),this.splatRenderTarget.depthTexture.format=yo,this.splatRenderTarget.depthTexture.type=pi}destroySplatRendertarget(){this.splatRenderTarget&&(this.splatRenderTarget=null)}setupRenderTargetCopyObjects(){const e={sourceColorTexture:{type:"t",value:null},sourceDepthTexture:{type:"t",value:null}},t=new Cn({vertexShader:`
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
            `,uniforms:e,depthWrite:!1,depthTest:!1,transparent:!0,blending:W0,blendSrc:xa,blendSrcAlpha:xa,blendDst:_a,blendDstAlpha:_a});t.extensions.fragDepth=!0,this.renderTargetCopyQuad=new Yt(new To(2,2),t),this.renderTargetCopyCamera=new Ch(-1,1,1,-1,0,1)}destroyRenderTargetCopyObjects(){this.renderTargetCopyQuad&&(Zr(this.renderTargetCopyQuad),this.renderTargetCopyQuad=null)}setupMeshCursor(){if(!this.meshCursor){const e=new Th(.5,1.5,32),t=new Mr({color:16777215}),n=new Yt(e,t);n.rotation.set(0,0,Math.PI),n.position.set(0,1,0);const s=new Yt(e,t);s.position.set(0,-1,0);const r=new Yt(e,t);r.rotation.set(0,0,Math.PI/2),r.position.set(1,0,0);const o=new Yt(e,t);o.rotation.set(0,0,-Math.PI/2),o.position.set(-1,0,0),this.meshCursor=new Kt,this.meshCursor.add(n),this.meshCursor.add(s),this.meshCursor.add(r),this.meshCursor.add(o),this.meshCursor.scale.set(.1,.1,.1),this.threeScene.add(this.meshCursor),this.meshCursor.visible=!1}}destroyMeshCursor(){this.meshCursor&&(Zr(this.meshCursor),this.threeScene.remove(this.meshCursor),this.meshCursor=null)}setMeshCursorVisibility(e){this.meshCursor.visible=e}getMeschCursorVisibility(){return this.meshCursor.visible}setMeshCursorPosition(e){this.meshCursor.position.copy(e)}positionAndOrientMeshCursor(e,t){this.meshCursor.position.copy(e),this.meshCursor.up.copy(t.up),this.meshCursor.lookAt(t.position)}setupFocusMarker(){if(!this.focusMarker){const e=new Xl(.5,32,32),t=oa.buildFocusMarkerMaterial();t.depthTest=!1,t.depthWrite=!1,t.transparent=!0,this.focusMarker=new Yt(e,t)}}destroyFocusMarker(){this.focusMarker&&(Zr(this.focusMarker),this.focusMarker=null)}updateFocusMarker=(function(){const e=new F,t=new Ye,n=new F;return function(s,r,o){t.copy(r.matrixWorld).invert(),e.copy(s).applyMatrix4(t),e.normalize().multiplyScalar(10),e.applyMatrix4(r.matrixWorld),n.copy(r.position).sub(s);const a=n.length();this.focusMarker.position.copy(s),this.focusMarker.scale.set(a,a,a),this.focusMarker.material.uniforms.realFocusPosition.value.copy(s),this.focusMarker.material.uniforms.viewport.value.copy(o),this.focusMarker.material.uniformsNeedUpdate=!0}})();setFocusMarkerVisibility(e){this.focusMarker.visible=e}setFocusMarkerOpacity(e){this.focusMarker.material.uniforms.opacity.value=e,this.focusMarker.material.uniformsNeedUpdate=!0}getFocusMarkerOpacity(){return this.focusMarker.material.uniforms.opacity.value}setupControlPlane(){if(!this.controlPlane){const e=new To(1,1);e.rotateX(-Math.PI/2);const t=new Mr({color:16777215});t.transparent=!0,t.opacity=.6,t.depthTest=!1,t.depthWrite=!1,t.side=fi;const n=new Yt(e,t),s=new F(0,1,0);s.normalize();const r=new F(0,0,0),o=.5,a=.01,l=56576,c=new yE(s,r,o,a,l,.1,.03);this.controlPlane=new Kt,this.controlPlane.add(n),this.controlPlane.add(c)}}destroyControlPlane(){this.controlPlane&&(Zr(this.controlPlane),this.controlPlane=null)}setControlPlaneVisibility(e){this.controlPlane.visible=e}positionAndOrientControlPlane=(function(){const e=new Mt,t=new F(0,1,0);return function(n,s){e.setFromUnitVectors(t,s),this.controlPlane.position.copy(n),this.controlPlane.quaternion.copy(e)}})();addDebugMeshes(){this.debugRoot=this.createDebugMeshes(),this.secondaryDebugRoot=this.createSecondaryDebugMeshes(),this.threeScene.add(this.debugRoot),this.threeScene.add(this.secondaryDebugRoot)}destroyDebugMeshes(){for(let e of[this.debugRoot,this.secondaryDebugRoot])e&&(Zr(e),this.threeScene.remove(e));this.debugRoot=null,this.secondaryDebugRoot=null}createDebugMeshes(e){const t=new Xl(1,32,32),n=new Kt,s=(r,o)=>{let a=new Yt(t,oa.buildDebugMaterial(r));a.renderOrder=e,n.add(a),a.position.fromArray(o)};return s(16711680,[-50,0,0]),s(16711680,[50,0,0]),s(65280,[0,0,-50]),s(65280,[0,0,50]),s(16755200,[5,0,5]),n}createSecondaryDebugMeshes(e){const t=new Fo(3,3,3),n=new Kt;let s=12303291;const r=a=>{let l=new Yt(t,oa.buildDebugMaterial(s));l.renderOrder=e,n.add(l),l.position.fromArray(a)};let o=10;return r([-o,0,-o]),r([-o,0,o]),r([o,0,-o]),r([o,0,o]),n}static buildDebugMaterial(e){const t=`
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
        `,s={color:{type:"v3",value:new rt(e)}},r=new Cn({uniforms:s,vertexShader:t,fragmentShader:n,transparent:!1,depthTest:!0,depthWrite:!0,side:Xi});return r.extensions.fragDepth=!0,r}static buildFocusMarkerMaterial(e){const t=`
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
        `,s={color:{type:"v3",value:new rt(e)},realFocusPosition:{type:"v3",value:new F},viewport:{type:"v2",value:new Pe},opacity:{value:0}};return new Cn({uniforms:s,vertexShader:t,fragmentShader:n,transparent:!0,depthTest:!1,depthWrite:!1,side:Xi})}dispose(){this.destroyMeshCursor(),this.destroyFocusMarker(),this.destroyDebugMeshes(),this.destroyControlPlane(),this.destroyRenderTargetCopyObjects(),this.destroySplatRendertarget()}}const bE=new F(1,0,0),ME=new F(0,1,0),TE=new F(0,0,1);class Su{constructor(e=new F,t=new F){this.origin=new F,this.direction=new F,this.setParameters(e,t)}setParameters(e,t){this.origin.copy(e),this.direction.copy(t).normalize()}boxContainsPoint(e,t,n){return!(t.x<e.min.x-n||t.x>e.max.x+n||t.y<e.min.y-n||t.y>e.max.y+n||t.z<e.min.z-n||t.z>e.max.z+n)}intersectBox=(function(){const e=new F,t=[],n=[],s=[];return function(r,o){if(n[0]=this.origin.x,n[1]=this.origin.y,n[2]=this.origin.z,s[0]=this.direction.x,s[1]=this.direction.y,s[2]=this.direction.z,this.boxContainsPoint(r,this.origin,1e-4))return o&&(o.origin.copy(this.origin),o.normal.set(0,0,0),o.distance=-1),!0;for(let a=0;a<3;a++){if(s[a]==0)continue;const l=a==0?bE:a==1?ME:TE,c=s[a]<0?r.max:r.min;let u=-Math.sign(s[a]);t[0]=a==0?c.x:a==1?c.y:c.z;let f=t[0]-n[a];if(f*u<0){const h=(a+1)%3,d=(a+2)%3;if(t[2]=s[h]/s[a]*f+n[h],t[1]=s[d]/s[a]*f+n[d],e.set(t[a],t[d],t[h]),this.boxContainsPoint(r,e,1e-4))return o&&(o.origin.copy(e),o.normal.copy(l).multiplyScalar(u),o.distance=e.sub(this.origin).length()),!0}}return!1}})();intersectSphere=(function(){const e=new F;return function(t,n,s){e.copy(t).sub(this.origin);const r=e.dot(this.direction),o=r*r,l=e.dot(e)-o,c=n*n;if(l>c)return!1;const u=Math.sqrt(c-l),f=r-u,h=r+u;if(h<0)return!1;let d=f<0?h:f;return s&&(s.origin.copy(this.origin).addScaledVector(this.direction,d),s.normal.copy(s.origin).sub(t).normalize(),s.distance=d),!0}})()}class Vh{constructor(){this.origin=new F,this.normal=new F,this.distance=0,this.splatIndex=0}set(e,t,n,s){this.origin.copy(e),this.normal.copy(t),this.distance=n,this.splatIndex=s}clone(){const e=new Vh;return e.origin.copy(this.origin),e.normal.copy(this.normal),e.distance=this.distance,e.splatIndex=this.splatIndex,e}}const ls={ThreeD:0,TwoD:1};class CE{constructor(e,t,n=!1){this.ray=new Su(e,t),this.raycastAgainstTrueSplatEllipsoid=n}setFromCameraAndScreenPosition=(function(){const e=new Pe;return function(t,n,s){if(e.x=n.x/s.x*2-1,e.y=(s.y-n.y)/s.y*2-1,t.isPerspectiveCamera)this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(e.x,e.y,.5).unproject(t).sub(this.ray.origin).normalize(),this.camera=t;else if(t.isOrthographicCamera)this.ray.origin.set(e.x,e.y,(t.near+t.far)/(t.near-t.far)).unproject(t),this.ray.direction.set(0,0,-1).transformDirection(t.matrixWorld),this.camera=t;else throw new Error("Raycaster::setFromCameraAndScreenPosition() -> Unsupported camera type")}})();intersectSplatMesh=(function(){const e=new Ye,t=new Ye,n=new Ye,s=new Su,r=new F;return function(o,a=[]){const l=o.getSplatTree();if(l){for(let c=0;c<l.subTrees.length;c++){const u=l.subTrees[c];t.copy(o.matrixWorld),o.dynamicMode&&(o.getSceneTransform(c,n),t.multiply(n)),e.copy(t).invert(),s.origin.copy(this.ray.origin).applyMatrix4(e),s.direction.copy(this.ray.origin).add(this.ray.direction),s.direction.applyMatrix4(e).sub(s.origin).normalize();const f=[];u.rootNode&&this.castRayAtSplatTreeNode(s,l,u.rootNode,f),f.forEach(h=>{h.origin.applyMatrix4(t),h.normal.applyMatrix4(t).normalize(),h.distance=r.copy(h.origin).sub(this.ray.origin).length()}),a.push(...f)}return a.sort((c,u)=>c.distance>u.distance?1:-1),a}}})();castRayAtSplatTreeNode=(function(){const e=new Dt,t=new F,n=new F,s=new Mt,r=new Vh,o=1e-7,a=new F(0,0,0),l=new Ye,c=new Ye,u=new Ye,f=new Ye,h=new Ye,d=new Su;return function(x,p,g,m=[]){if(x.intersectBox(g.boundingBox)){if(g.data&&g.data.indexes&&g.data.indexes.length>0)for(let _=0;_<g.data.indexes.length;_++){const S=g.data.indexes[_],A=p.splatMesh.getSceneIndexForSplat(S);if(p.splatMesh.getScene(A).visible&&(p.splatMesh.getSplatColor(S,e),p.splatMesh.getSplatCenter(S,t),p.splatMesh.getSplatScaleAndRotation(S,n,s),!(n.x<=o||n.y<=o||p.splatMesh.splatRenderMode===ls.ThreeD&&n.z<=o)))if(this.raycastAgainstTrueSplatEllipsoid){c.makeScale(n.x,n.y,n.z),u.makeRotationFromQuaternion(s);const b=Math.log10(e.w)*2;if(l.makeScale(b,b,b),h.copy(l).multiply(u).multiply(c),f.copy(h).invert(),d.origin.copy(x.origin).sub(t).applyMatrix4(f),d.direction.copy(x.origin).add(x.direction).sub(t),d.direction.applyMatrix4(f).sub(d.origin).normalize(),d.intersectSphere(a,1,r)){const v=r.clone();v.splatIndex=S,v.origin.applyMatrix4(h).add(t),m.push(v)}}else{let b=n.x+n.y,v=2;if(p.splatMesh.splatRenderMode===ls.ThreeD&&(b+=n.z,v=3),b=b/v,x.intersectSphere(t,b,r)){const E=r.clone();E.splatIndex=S,m.push(E)}}}if(g.children&&g.children.length>0)for(let _ of g.children)this.castRayAtSplatTreeNode(x,p,_,m);return m}}})()}class fo{static buildVertexShaderBase(e=!1,t=!1,n=0,s=""){let r=`
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
            uniform float sceneOpacity[${gt.MaxScenes}];
            uniform int sceneVisibility[${gt.MaxScenes}];
        `),e&&(r+=`
            uniform highp mat4 transforms[${gt.MaxScenes}];
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
        uniform float sphericalHarmonics8BitCompressionRangeMin[${gt.MaxScenes}];
        uniform float sphericalHarmonics8BitCompressionRangeMax[${gt.MaxScenes}];

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
        `}static getUniforms(e=!1,t=!1,n=0,s=1,r=!1){const o={sceneCenter:{type:"v3",value:new F},fadeInComplete:{type:"i",value:0},orthographicMode:{type:"i",value:0},visibleRegionFadeStartRadius:{type:"f",value:0},visibleRegionRadius:{type:"f",value:0},currentTime:{type:"f",value:0},firstRenderTime:{type:"f",value:0},centersColorsTexture:{type:"t",value:null},sphericalHarmonicsTexture:{type:"t",value:null},sphericalHarmonicsTextureR:{type:"t",value:null},sphericalHarmonicsTextureG:{type:"t",value:null},sphericalHarmonicsTextureB:{type:"t",value:null},sphericalHarmonics8BitCompressionRangeMin:{type:"f",value:[]},sphericalHarmonics8BitCompressionRangeMax:{type:"f",value:[]},focal:{type:"v2",value:new Pe},orthoZoom:{type:"f",value:1},inverseFocalAdjustment:{type:"f",value:1},viewport:{type:"v2",value:new Pe},basisViewport:{type:"v2",value:new Pe},debugColor:{type:"v3",value:new rt},centersColorsTextureSize:{type:"v2",value:new Pe(1024,1024)},sphericalHarmonicsDegree:{type:"i",value:n},sphericalHarmonicsTextureSize:{type:"v2",value:new Pe(1024,1024)},sphericalHarmonics8BitMode:{type:"i",value:0},sphericalHarmonicsMultiTextureMode:{type:"i",value:0},splatScale:{type:"f",value:s},pointCloudModeEnabled:{type:"i",value:r?1:0},sceneIndexesTexture:{type:"t",value:null},sceneIndexesTextureSize:{type:"v2",value:new Pe(1024,1024)},sceneCount:{type:"i",value:1}};for(let a=0;a<gt.MaxScenes;a++)o.sphericalHarmonics8BitCompressionRangeMin.value.push(-3/2),o.sphericalHarmonics8BitCompressionRangeMax.value.push(gt.SphericalHarmonics8BitCompressionRange/2);if(t){const a=[];for(let c=0;c<gt.MaxScenes;c++)a.push(1);o.sceneOpacity={type:"f",value:a};const l=[];for(let c=0;c<gt.MaxScenes;c++)l.push(1);o.sceneVisibility={type:"i",value:l}}if(e){const a=[];for(let l=0;l<gt.MaxScenes;l++)a.push(new Ye);o.transforms={type:"mat4",value:a}}return o}}class Ql{static build(e=!1,t=!1,n=!1,s=2048,r=1,o=!1,a=0,l=.3){let u=fo.buildVertexShaderBase(e,t,a,`
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
        `);u+=Ql.buildVertexShaderProjection(n,t,s,l);const f=Ql.buildFragmentShader(),h=fo.getUniforms(e,t,a,r,o);return h.covariancesTextureSize={type:"v2",value:new Pe(1024,1024)},h.covariancesTexture={type:"t",value:null},h.covariancesTextureHalfFloat={type:"t",value:null},h.covariancesAreHalfFloat={type:"i",value:0},new Cn({uniforms:h,vertexShader:u,fragmentShader:f,transparent:!0,alphaTest:1,blending:Ns,depthTest:!0,depthWrite:!1,side:fi})}static buildVertexShaderProjection(e,t,n,s){let r=`

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
        `,r+=fo.getVertexShaderFadeIn(),r+="}",r}static buildFragmentShader(){let e=`
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
        `,e}}class Kl{static build(e=!1,t=!1,n=1,s=!1,r=0){let a=fo.buildVertexShaderBase(e,t,r,`
            uniform vec2 scaleRotationsTextureSize;
            uniform highp sampler2D scaleRotationsTexture;
            varying mat3 vT;
            varying vec2 vQuadCenter;
            varying vec2 vFragCoord;
        `);a+=Kl.buildVertexShaderProjection();const l=Kl.buildFragmentShader(),c=fo.getUniforms(e,t,r,n,s);return c.scaleRotationsTexture={type:"t",value:null},c.scaleRotationsTextureSize={type:"v2",value:new Pe(1024,1024)},new Cn({uniforms:c,vertexShader:a,fragmentShader:l,transparent:!0,alphaTest:1,blending:Ns,depthTest:!0,depthWrite:!1,side:fi})}static buildVertexShaderProjection(){let e=`

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
            `,e+=fo.getVertexShaderFadeIn(),e+="}",e}static buildFragmentShader(){return`
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
        `}}class EE{static build(e){const t=new En;t.setIndex([0,1,2,0,2,3]);const n=new Float32Array(12),s=new _i(n,3);t.setAttribute("position",s),s.setXYZ(0,-1,-1,0),s.setXYZ(1,-1,1,0),s.setXYZ(2,1,1,0),s.setXYZ(3,1,-1,0),s.needsUpdate=!0;const r=new IA().copy(t),o=new Uint32Array(e),a=new yA(o,1,!1);return a.setUsage(WS),r.setAttribute("splatIndex",a),r.instanceCount=0,r}}class wE extends Kt{constructor(e,t=new F,n=new Mt,s=new F(1,1,1),r=1,o=1,a=!0){super(),this.splatBuffer=e,this.position.copy(t),this.quaternion.copy(n),this.scale.copy(s),this.transform=new Ye,this.minimumAlpha=r,this.opacity=o,this.visible=a}copyTransformData(e){this.position.copy(e.position),this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.transform.copy(e.transform)}updateTransform(e){e?(this.matrixWorldAutoUpdate&&this.updateWorldMatrix(!0,!1),this.transform.copy(this.matrixWorld)):(this.matrixAutoUpdate&&this.updateMatrix(),this.transform.copy(this.matrix))}}class Gh{static idGen=0;constructor(e,t,n,s){this.min=new F().copy(e),this.max=new F().copy(t),this.boundingBox=new Ni(this.min,this.max),this.center=new F().copy(this.max).sub(this.min).multiplyScalar(.5).add(this.min),this.depth=n,this.children=[],this.data=null,this.id=s||Gh.idGen++}}class aa{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.sceneDimensions=new F,this.sceneMin=new F,this.sceneMax=new F,this.rootNode=null,this.nodesWithIndexes=[],this.splatMesh=null}static convertWorkerSubTreeNode(e){const t=new F().fromArray(e.min),n=new F().fromArray(e.max),s=new Gh(t,n,e.depth,e.id);if(e.data.indexes){s.data={indexes:[]};for(let r of e.data.indexes)s.data.indexes.push(r)}if(e.children)for(let r of e.children)s.children.push(aa.convertWorkerSubTreeNode(r));return s}static convertWorkerSubTree(e,t){const n=new aa(e.maxDepth,e.maxCentersPerNode);n.sceneMin=new F().fromArray(e.sceneMin),n.sceneMax=new F().fromArray(e.sceneMax),n.splatMesh=t,n.rootNode=aa.convertWorkerSubTreeNode(e.rootNode);const s=(r,o)=>{r.children.length===0&&o(r);for(let a of r.children)s(a,o)};return n.nodesWithIndexes=[],s(n.rootNode,r=>{r.data&&r.data.indexes&&r.data.indexes.length>0&&n.nodesWithIndexes.push(r)}),n}}function RE(i){let e=0;class t{constructor(l,c){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]]}containsPoint(l){return l[0]>=this.min[0]&&l[0]<=this.max[0]&&l[1]>=this.min[1]&&l[1]<=this.max[1]&&l[2]>=this.min[2]&&l[2]<=this.max[2]}}class n{constructor(l,c){this.maxDepth=l,this.maxCentersPerNode=c,this.sceneDimensions=[],this.sceneMin=[],this.sceneMax=[],this.rootNode=null,this.addedIndexes={},this.nodesWithIndexes=[],this.splatMesh=null,this.disposed=!1}}class s{constructor(l,c,u,f){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]],this.center=[(c[0]-l[0])*.5+l[0],(c[1]-l[1])*.5+l[1],(c[2]-l[2])*.5+l[2]],this.depth=u,this.children=[],this.data=null,this.id=f||e++}}processSplatTreeNode=function(a,l,c,u){const f=l.data.indexes.length;if(f<a.maxCentersPerNode||l.depth>a.maxDepth){const _=[];for(let S=0;S<l.data.indexes.length;S++)a.addedIndexes[l.data.indexes[S]]||(_.push(l.data.indexes[S]),a.addedIndexes[l.data.indexes[S]]=!0);l.data.indexes=_,l.data.indexes.sort((S,A)=>S>A?1:-1),a.nodesWithIndexes.push(l);return}const h=[l.max[0]-l.min[0],l.max[1]-l.min[1],l.max[2]-l.min[2]],d=[h[0]*.5,h[1]*.5,h[2]*.5],x=[l.min[0]+d[0],l.min[1]+d[1],l.min[2]+d[2]],p=[new t([x[0]-d[0],x[1],x[2]-d[2]],[x[0],x[1]+d[1],x[2]]),new t([x[0],x[1],x[2]-d[2]],[x[0]+d[0],x[1]+d[1],x[2]]),new t([x[0],x[1],x[2]],[x[0]+d[0],x[1]+d[1],x[2]+d[2]]),new t([x[0]-d[0],x[1],x[2]],[x[0],x[1]+d[1],x[2]+d[2]]),new t([x[0]-d[0],x[1]-d[1],x[2]-d[2]],[x[0],x[1],x[2]]),new t([x[0],x[1]-d[1],x[2]-d[2]],[x[0]+d[0],x[1],x[2]]),new t([x[0],x[1]-d[1],x[2]],[x[0]+d[0],x[1],x[2]+d[2]]),new t([x[0]-d[0],x[1]-d[1],x[2]],[x[0],x[1],x[2]+d[2]])],g=[];for(let _=0;_<p.length;_++)g[_]=[];const m=[0,0,0];for(let _=0;_<f;_++){const S=l.data.indexes[_],A=c[S];m[0]=u[A],m[1]=u[A+1],m[2]=u[A+2];for(let y=0;y<p.length;y++)p[y].containsPoint(m)&&g[y].push(S)}for(let _=0;_<p.length;_++){const S=new s(p[_].min,p[_].max,l.depth+1);S.data={indexes:g[_]},l.children.push(S)}l.data={};for(let _ of l.children)processSplatTreeNode(a,_,c,u)};const r=(a,l,c)=>{const u=[0,0,0],f=[0,0,0],h=[],d=Math.floor(a.length/4);for(let p=0;p<d;p++){const g=p*4,m=a[g],_=a[g+1],S=a[g+2],A=Math.round(a[g+3]);(p===0||m<u[0])&&(u[0]=m),(p===0||m>f[0])&&(f[0]=m),(p===0||_<u[1])&&(u[1]=_),(p===0||_>f[1])&&(f[1]=_),(p===0||S<u[2])&&(u[2]=S),(p===0||S>f[2])&&(f[2]=S),h.push(A)}const x=new n(l,c);return x.sceneMin=u,x.sceneMax=f,x.rootNode=new s(x.sceneMin,x.sceneMax,0),x.rootNode.data={indexes:h},x};function o(a,l,c){const u=[];for(let h of a){const d=Math.floor(h.length/4);for(let x=0;x<d;x++){const p=x*4,g=Math.round(h[p+3]);u[g]=p}}const f=[];for(let h of a){const d=r(h,l,c);f.push(d),processSplatTreeNode(d,d.rootNode,u,h)}i.postMessage({subTrees:f})}i.onmessage=a=>{a.data.process&&o(a.data.process.centers,a.data.process.maxDepth,a.data.process.maxCentersPerNode)}}function IE(i,e,t,n,s){i.postMessage({process:{centers:e,maxDepth:n,maxCentersPerNode:s}},t)}function DE(){return new Worker(URL.createObjectURL(new Blob(["(",RE.toString(),")(self)"],{type:"application/javascript"})))}class PE{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.subTrees=[],this.splatMesh=null}dispose(){this.diposeSplatTreeWorker(),this.disposed=!0}diposeSplatTreeWorker(){this.splatTreeWorker&&this.splatTreeWorker.terminate(),this.splatTreeWorker=null}processSplatMesh=function(e,t=()=>!0,n,s){this.splatTreeWorker||(this.splatTreeWorker=DE()),this.splatMesh=e,this.subTrees=[];const r=new F,o=(a,l)=>{const c=new Float32Array(l*4);let u=0;for(let f=0;f<l;f++){const h=f+a;if(t(h)){e.getSplatCenter(h,r);const d=u*4;c[d]=r.x,c[d+1]=r.y,c[d+2]=r.z,c[d+3]=h,u++}}return c};return new Promise(a=>{const l=()=>this.disposed?(this.diposeSplatTreeWorker(),a(),!0):!1;n&&n(!1),jn(()=>{if(l())return;const c=[];if(e.dynamicMode){let u=0;for(let f=0;f<e.scenes.length;f++){const d=e.getScene(f).splatBuffer.getSplatCount(),x=o(u,d);c.push(x),u+=d}}else{const u=o(0,e.getSplatCount());c.push(u)}this.splatTreeWorker.onmessage=u=>{l()||u.data.subTrees&&(s&&s(!1),jn(()=>{if(!l()){for(let f of u.data.subTrees){const h=aa.convertWorkerSubTree(f,e);this.subTrees.push(h)}this.diposeSplatTreeWorker(),s&&s(!0),jn(()=>{a()})}}))},jn(()=>{if(l())return;n&&n(!0);const u=c.map(f=>f.buffer);IE(this.splatTreeWorker,c,u,this.maxDepth,this.maxCentersPerNode)})})})};countLeaves(){let e=0;return this.visitLeaves(()=>{e++}),e}visitLeaves(e){const t=(n,s)=>{n.children.length===0&&s(n);for(let r of n.children)t(r,s)};for(let n of this.subTrees)t(n.rootNode,e)}}function FE(i){const e={};function t(n){if(e[n]!==void 0)return e[n];let s;switch(n){case"WEBGL_depth_texture":s=i.getExtension("WEBGL_depth_texture")||i.getExtension("MOZ_WEBGL_depth_texture")||i.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":s=i.getExtension("EXT_texture_filter_anisotropic")||i.getExtension("MOZ_EXT_texture_filter_anisotropic")||i.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":s=i.getExtension("WEBGL_compressed_texture_s3tc")||i.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":s=i.getExtension("WEBGL_compressed_texture_pvrtc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:s=i.getExtension(n)}return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(n){n.isWebGL2?(t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance")):(t("WEBGL_depth_texture"),t("OES_texture_float"),t("OES_texture_half_float"),t("OES_texture_half_float_linear"),t("OES_standard_derivatives"),t("OES_element_index_uint"),t("OES_vertex_array_object"),t("ANGLE_instanced_arrays")),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture")},get:function(n){const s=t(n);return s===null&&console.warn("THREE.WebGLRenderer: "+n+" extension not supported."),s}}}function LE(i,e,t){let n;function s(){if(n!==void 0)return n;if(e.has("EXT_texture_filter_anisotropic")===!0){const v=e.get("EXT_texture_filter_anisotropic");n=i.getParameter(v.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else n=0;return n}function r(v){if(v==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";v="mediump"}return v==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}const o=typeof WebGL2RenderingContext<"u"&&i.constructor.name==="WebGL2RenderingContext";let a=t.precision!==void 0?t.precision:"highp";const l=r(a);l!==a&&(console.warn("THREE.WebGLRenderer:",a,"not supported, using",l,"instead."),a=l);const c=o||e.has("WEBGL_draw_buffers"),u=t.logarithmicDepthBuffer===!0,f=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),h=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),d=i.getParameter(i.MAX_TEXTURE_SIZE),x=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),p=i.getParameter(i.MAX_VERTEX_ATTRIBS),g=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),m=i.getParameter(i.MAX_VARYING_VECTORS),_=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),S=h>0,A=o||e.has("OES_texture_float"),y=S&&A,b=o?i.getParameter(i.MAX_SAMPLES):0;return{isWebGL2:o,drawBuffers:c,getMaxAnisotropy:s,getMaxPrecision:r,precision:a,logarithmicDepthBuffer:u,maxTextures:f,maxVertexTextures:h,maxTextureSize:d,maxCubemapSize:x,maxAttributes:p,maxVertexUniforms:g,maxVaryings:m,maxFragmentUniforms:_,vertexTextures:S,floatFragmentTextures:A,floatVertexTextures:y,maxSamples:b}}const la={Default:0,Instant:2},ho={None:0,Info:3},lm=new En,BE=new Mr,Sl=6,UE=4,OE=4,NE=4,zE=6,kE=8,Au=4,yu=4,cm=1,HE=.012,VE=.003,um=1,fm=16777216;class sn extends Yt{constructor(e=ls.ThreeD,t=!1,n=!1,s=!1,r=1,o=!0,a=!1,l=!1,c=1024,u=ho.None,f=0,h=1,d=.3){super(lm,BE),this.renderer=void 0,this.splatRenderMode=e,this.dynamicMode=t,this.enableOptionalEffects=n,this.halfPrecisionCovariancesOnGPU=s,this.devicePixelRatio=r,this.enableDistancesComputationOnGPU=o,this.integerBasedDistancesComputation=a,this.antialiased=l,this.kernel2DSize=d,this.maxScreenSpaceSplatSize=c,this.logLevel=u,this.sphericalHarmonicsDegree=f,this.minSphericalHarmonicsDegree=0,this.sceneFadeInRateMultiplier=h,this.scenes=[],this.splatTree=null,this.baseSplatTree=null,this.splatDataTextures={},this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new Ni,this.calculatedSceneCenter=new F,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!1,this.lastRenderer=null,this.visible=!1}static buildScenes(e,t,n){const s=[];s.length=t.length;for(let r=0;r<t.length;r++){const o=t[r],a=n[r]||{};let l=a.position||[0,0,0],c=a.rotation||[0,0,0,1],u=a.scale||[1,1,1];const f=new F().fromArray(l),h=new Mt().fromArray(c),d=new F().fromArray(u),x=sn.createScene(o,f,h,d,a.splatAlphaRemovalThreshold||1,a.opacity,a.visible);e.add(x),s[r]=x}return s}static createScene(e,t,n,s,r,o=1,a=!0){return new wE(e,t,n,s,r,o,a)}static buildSplatIndexMaps(e){const t=[],n=[];let s=0;for(let r=0;r<e.length;r++){const a=e[r].getMaxSplatCount();for(let l=0;l<a;l++)t[s]=l,n[s]=r,s++}return{localSplatIndexMap:t,sceneIndexMap:n}}buildSplatTree=function(e=[],t,n){return new Promise(s=>{this.disposeSplatTree(),this.baseSplatTree=new PE(8,1e3);const r=performance.now(),o=new Dt;this.baseSplatTree.processSplatMesh(this,a=>{this.getSplatColor(a,o);const l=this.getSceneIndexForSplat(a),c=e[l]||1;return o.w>=c},t,n).then(()=>{const a=performance.now()-r;if(this.logLevel>=ho.Info&&console.log("SplatTree build: "+a+" ms"),this.disposed)s();else{this.splatTree=this.baseSplatTree,this.baseSplatTree=null;let l=0,c=0,u=0;this.splatTree.visitLeaves(f=>{const h=f.data.indexes.length;h>0&&(c+=h,u++,l++)}),this.logLevel>=ho.Info&&(console.log(`SplatTree leaves: ${this.splatTree.countLeaves()}`),console.log(`SplatTree leaves with splats:${l}`),c=c/u,console.log(`Avg splat count per node: ${c}`),console.log(`Total splat count: ${this.getSplatCount()}`)),s()}})})};build(e,t,n=!0,s=!1,r,o,a=!0){this.sceneOptions=t,this.finalBuild=s;const l=sn.getTotalMaxSplatCountForSplatBuffers(e),c=sn.buildScenes(this,e,t);if(n)for(let p=0;p<this.scenes.length&&p<c.length;p++){const g=c[p],m=this.getScene(p);g.copyTransformData(m)}this.scenes=c;let u=3;for(let p of e){const g=p.getMinSphericalHarmonicsDegree();g<u&&(u=g)}this.minSphericalHarmonicsDegree=Math.min(u,this.sphericalHarmonicsDegree);let f=!1;if(e.length!==this.lastBuildScenes.length)f=!0;else for(let p=0;p<e.length;p++)if(e[p]!==this.lastBuildScenes[p].splatBuffer){f=!0;break}let h=!0;if((this.scenes.length!==1||this.lastBuildSceneCount!==this.scenes.length||this.lastBuildMaxSplatCount!==l||f)&&(h=!1),!h){this.boundingBox=new Ni,a||(this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.firstRenderTime=-1),this.lastBuildScenes=[],this.lastBuildSplatCount=0,this.lastBuildMaxSplatCount=0,this.disposeMeshData(),this.geometry=EE.build(l),this.splatRenderMode===ls.ThreeD?this.material=Ql.build(this.dynamicMode,this.enableOptionalEffects,this.antialiased,this.maxScreenSpaceSplatSize,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree,this.kernel2DSize):this.material=Kl.build(this.dynamicMode,this.enableOptionalEffects,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree);const p=sn.buildSplatIndexMaps(e);this.globalSplatIndexToLocalSplatIndexMap=p.localSplatIndexMap,this.globalSplatIndexToSceneIndexMap=p.sceneIndexMap}const d=this.getSplatCount(!0);this.enableDistancesComputationOnGPU&&this.setupDistancesComputationTransformFeedback();const x=this.refreshGPUDataFromSplatBuffers(h);for(let p=0;p<this.scenes.length;p++)this.lastBuildScenes[p]=this.scenes[p];return this.lastBuildSplatCount=d,this.lastBuildMaxSplatCount=this.getMaxSplatCount(),this.lastBuildSceneCount=this.scenes.length,s&&this.scenes.length>0&&this.buildSplatTree(t.map(p=>p.splatAlphaRemovalThreshold||1),r,o).then(()=>{this.onSplatTreeReadyCallback&&this.onSplatTreeReadyCallback(this.splatTree),this.onSplatTreeReadyCallback=null}),this.visible=this.scenes.length>0,x}freeIntermediateSplatData(){const e=t=>{delete t.source.data,delete t.image,t.onUpdate=null};delete this.splatDataTextures.baseData.covariances,delete this.splatDataTextures.baseData.centers,delete this.splatDataTextures.baseData.colors,delete this.splatDataTextures.baseData.sphericalHarmonics,delete this.splatDataTextures.centerColors.data,delete this.splatDataTextures.covariances.data,this.splatDataTextures.sphericalHarmonics&&delete this.splatDataTextures.sphericalHarmonics.data,this.splatDataTextures.sceneIndexes&&delete this.splatDataTextures.sceneIndexes.data,this.splatDataTextures.centerColors.texture.needsUpdate=!0,this.splatDataTextures.centerColors.texture.onUpdate=()=>{e(this.splatDataTextures.centerColors.texture)},this.splatDataTextures.covariances.texture.needsUpdate=!0,this.splatDataTextures.covariances.texture.onUpdate=()=>{e(this.splatDataTextures.covariances.texture)},this.splatDataTextures.sphericalHarmonics&&(this.splatDataTextures.sphericalHarmonics.texture?(this.splatDataTextures.sphericalHarmonics.texture.needsUpdate=!0,this.splatDataTextures.sphericalHarmonics.texture.onUpdate=()=>{e(this.splatDataTextures.sphericalHarmonics.texture)}):this.splatDataTextures.sphericalHarmonics.textures.forEach(t=>{t.needsUpdate=!0,t.onUpdate=()=>{e(t)}})),this.splatDataTextures.sceneIndexes&&(this.splatDataTextures.sceneIndexes.texture.needsUpdate=!0,this.splatDataTextures.sceneIndexes.texture.onUpdate=()=>{e(this.splatDataTextures.sceneIndexes.texture)})}dispose(){this.disposeMeshData(),this.disposeTextures(),this.disposeSplatTree(),this.enableDistancesComputationOnGPU&&(this.computeDistancesOnGPUSyncTimeout&&(clearTimeout(this.computeDistancesOnGPUSyncTimeout),this.computeDistancesOnGPUSyncTimeout=null),this.disposeDistancesComputationGPUResources()),this.scenes=[],this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.renderer=null,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new Ni,this.calculatedSceneCenter=new F,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!0,this.lastRenderer=null,this.visible=!1}disposeMeshData(){this.geometry&&this.geometry!==lm&&(this.geometry.dispose(),this.geometry=null),this.material&&(this.material.dispose(),this.material=null)}disposeTextures(){for(let e in this.splatDataTextures)if(this.splatDataTextures.hasOwnProperty(e)){const t=this.splatDataTextures[e];t.texture&&(t.texture.dispose(),t.texture=null)}this.splatDataTextures=null}disposeSplatTree(){this.splatTree&&(this.splatTree.dispose(),this.splatTree=null),this.baseSplatTree&&(this.baseSplatTree.dispose(),this.baseSplatTree=null)}getSplatTree(){return this.splatTree}onSplatTreeReady(e){this.onSplatTreeReadyCallback=e}getDataForDistancesComputation(e,t){const n=this.integerBasedDistancesComputation?this.getIntegerCenters(e,t,!0):this.getFloatCenters(e,t,!0),s=this.getSceneIndexes(e,t);return{centers:n,sceneIndexes:s}}refreshGPUDataFromSplatBuffers(e){const t=this.getSplatCount(!0);this.refreshDataTexturesFromSplatBuffers(e);const n=e?this.lastBuildSplatCount:0,{centers:s,sceneIndexes:r}=this.getDataForDistancesComputation(n,t-1);return this.enableDistancesComputationOnGPU&&this.refreshGPUBuffersForDistancesComputation(s,r,e),{from:n,to:t-1,count:t-n,centers:s,sceneIndexes:r}}refreshGPUBuffersForDistancesComputation(e,t,n=!1){const s=n?this.lastBuildSplatCount:0;this.updateGPUCentersBufferForDistancesComputation(n,e,s),this.updateGPUTransformIndexesBufferForDistancesComputation(n,t,s)}refreshDataTexturesFromSplatBuffers(e){const t=this.getSplatCount(!0),n=this.lastBuildSplatCount,s=t-1;e?this.updateBaseDataFromSplatBuffers(n,s):(this.setupDataTextures(),this.updateBaseDataFromSplatBuffers()),this.updateDataTexturesFromBaseData(n,s),this.updateVisibleRegion(e)}setupDataTextures(){const e=this.getMaxSplatCount(),t=this.getSplatCount(!0);this.disposeTextures();const n=(v,E)=>{const M=new Pe(4096,1024);for(;M.x*M.y*v<e*E;)M.y*=2;return M},s=v=>v>=1?zE:OE,r=v=>{const E=s(v),M=n(E,6);return{elementsPerTexelStored:E,texSize:M}};let o=this.getTargetCovarianceCompressionLevel();const a=0,l=this.getTargetSphericalHarmonicsCompressionLevel();let c,u,f;if(this.splatRenderMode===ls.ThreeD){const v=r(o);v.texSize.x*v.texSize.y>fm&&o===0&&(o=1),c=new Float32Array(e*Sl)}else u=new Float32Array(e*3),f=new Float32Array(e*4);const h=new Float32Array(e*3),d=new Uint8Array(e*4);let x=Float32Array;l===1?x=Uint16Array:l===2&&(x=Uint8Array);const p=uo(this.minSphericalHarmonicsDegree),g=this.minSphericalHarmonicsDegree?new x(e*p):void 0,m=n(yu,4),_=new Uint32Array(m.x*m.y*yu);sn.updateCenterColorsPaddedData(0,t-1,h,d,_);const S=new is(_,m.x,m.y,lo,pi);if(S.internalFormat="RGBA32UI",S.needsUpdate=!0,this.material.uniforms.centersColorsTexture.value=S,this.material.uniforms.centersColorsTextureSize.value.copy(m),this.material.uniformsNeedUpdate=!0,this.splatDataTextures={baseData:{covariances:c,scales:u,rotations:f,centers:h,colors:d,sphericalHarmonics:g},centerColors:{data:_,texture:S,size:m}},this.splatRenderMode===ls.ThreeD){const v=r(o),E=v.elementsPerTexelStored,M=v.texSize;let T=o>=1?Uint32Array:Float32Array;const I=o>=1?kE:NE,P=new T(M.x*M.y*I);o===0?P.set(c):sn.updatePaddedCompressedCovariancesTextureData(c,P,0,0,c.length);let B;if(o>=1)B=new is(P,M.x,M.y,lo,pi),B.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=B;else{B=new is(P,M.x,M.y,Mn,Mi),this.material.uniforms.covariancesTexture.value=B;const N=new is(new Uint32Array(32),2,2,lo,pi);N.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=N,N.needsUpdate=!0}B.needsUpdate=!0,this.material.uniforms.covariancesAreHalfFloat.value=o>=1?1:0,this.material.uniforms.covariancesTextureSize.value.copy(M),this.splatDataTextures.covariances={data:P,texture:B,size:M,compressionLevel:o,elementsPerTexelStored:E,elementsPerTexelAllocated:I}}else{const E=n(Au,6);let M=Float32Array,T=Mi;const I=new M(E.x*E.y*Au);sn.updateScaleRotationsPaddedData(0,t-1,u,f,I);const P=new is(I,E.x,E.y,Mn,T);P.needsUpdate=!0,this.material.uniforms.scaleRotationsTexture.value=P,this.material.uniforms.scaleRotationsTextureSize.value.copy(E),this.splatDataTextures.scaleRotations={data:I,texture:P,size:E,compressionLevel:a}}if(g){const v=l===2?qi:Tr;let E=p;E%2!==0&&E++;const M=4,T=Mn;let I=n(M,E);if(I.x*I.y<=fm){const P=I.x*I.y*M,B=new x(P);for(let G=0;G<t;G++){const V=p*G,q=E*G;for(let X=0;X<p;X++)B[q+X]=g[V+X]}const N=new is(B,I.x,I.y,T,v);N.needsUpdate=!0,this.material.uniforms.sphericalHarmonicsTexture.value=N,this.splatDataTextures.sphericalHarmonics={componentCount:p,paddedComponentCount:E,data:B,textureCount:1,texture:N,size:I,compressionLevel:l,elementsPerTexel:M}}else{const P=p/3;E=P,E%2!==0&&E++,I=n(M,E);const B=I.x*I.y*M,N=[this.material.uniforms.sphericalHarmonicsTextureR,this.material.uniforms.sphericalHarmonicsTextureG,this.material.uniforms.sphericalHarmonicsTextureB],G=[],V=[];for(let q=0;q<3;q++){const X=new x(B);G.push(X);for(let ce=0;ce<t;ce++){const be=p*ce,Re=E*ce;if(P>=3){for(let Fe=0;Fe<3;Fe++)X[Re+Fe]=g[be+q*3+Fe];if(P>=8)for(let Fe=0;Fe<5;Fe++)X[Re+3+Fe]=g[be+9+q*5+Fe]}}const ee=new is(X,I.x,I.y,T,v);V.push(ee),ee.needsUpdate=!0,N[q].value=ee}this.material.uniforms.sphericalHarmonicsMultiTextureMode.value=1,this.splatDataTextures.sphericalHarmonics={componentCount:p,componentCountPerChannel:P,paddedComponentCount:E,data:G,textureCount:3,textures:V,size:I,compressionLevel:l,elementsPerTexel:M}}this.material.uniforms.sphericalHarmonicsTextureSize.value.copy(I),this.material.uniforms.sphericalHarmonics8BitMode.value=l===2?1:0;for(let P=0;P<this.scenes.length;P++){const B=this.scenes[P].splatBuffer;this.material.uniforms.sphericalHarmonics8BitCompressionRangeMin.value[P]=B.minSphericalHarmonicsCoeff,this.material.uniforms.sphericalHarmonics8BitCompressionRangeMax.value[P]=B.maxSphericalHarmonicsCoeff}this.material.uniformsNeedUpdate=!0}const A=n(cm,4),y=new Uint32Array(A.x*A.y*cm);for(let v=0;v<t;v++)y[v]=this.globalSplatIndexToSceneIndexMap[v];const b=new is(y,A.x,A.y,dc,pi);b.internalFormat="R32UI",b.needsUpdate=!0,this.material.uniforms.sceneIndexesTexture.value=b,this.material.uniforms.sceneIndexesTextureSize.value.copy(A),this.material.uniformsNeedUpdate=!0,this.splatDataTextures.sceneIndexes={data:y,texture:b,size:A},this.material.uniforms.sceneCount.value=this.scenes.length}updateBaseDataFromSplatBuffers(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0;this.fillSplatDataArrays(this.splatDataTextures.baseData.covariances,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,this.splatDataTextures.baseData.sphericalHarmonics,void 0,s,o,l,e,t,e)}updateDataTexturesFromBaseData(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0,c=this.splatDataTextures.centerColors,u=c.data,f=c.texture;sn.updateCenterColorsPaddedData(e,t,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,u);const h=this.renderer?this.renderer.properties.get(f):null;if(!h||!h.__webglTexture?f.needsUpdate=!0:this.updateDataTexture(u,c.texture,c.size,h,yu,UE,4,e,t),n){const _=n.texture,S=e*Sl,A=t*Sl;if(s===0)for(let b=S;b<=A;b++){const v=this.splatDataTextures.baseData.covariances[b];n.data[b]=v}else sn.updatePaddedCompressedCovariancesTextureData(this.splatDataTextures.baseData.covariances,n.data,e*n.elementsPerTexelAllocated,S,A);const y=this.renderer?this.renderer.properties.get(_):null;!y||!y.__webglTexture?_.needsUpdate=!0:s===0?this.updateDataTexture(n.data,n.texture,n.size,y,n.elementsPerTexelStored,Sl,4,e,t):this.updateDataTexture(n.data,n.texture,n.size,y,n.elementsPerTexelAllocated,n.elementsPerTexelAllocated,2,e,t)}if(r){const _=r.data,S=r.texture,A=6,y=o===0?4:2;sn.updateScaleRotationsPaddedData(e,t,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,_);const b=this.renderer?this.renderer.properties.get(S):null;!b||!b.__webglTexture?S.needsUpdate=!0:this.updateDataTexture(_,r.texture,r.size,b,Au,A,y,e,t)}const d=this.splatDataTextures.baseData.sphericalHarmonics;if(d){let _=4;l===1?_=2:l===2&&(_=1);const S=(b,v,E,M,T)=>{const I=this.renderer?this.renderer.properties.get(b):null;!I||!I.__webglTexture?b.needsUpdate=!0:this.updateDataTexture(M,b,v,I,E,T,_,e,t)},A=a.componentCount,y=a.paddedComponentCount;if(a.textureCount===1){const b=a.data;for(let v=e;v<=t;v++){const E=A*v,M=y*v;for(let T=0;T<A;T++)b[M+T]=d[E+T]}S(a.texture,a.size,a.elementsPerTexel,b,y)}else{const b=a.componentCountPerChannel;for(let v=0;v<3;v++){const E=a.data[v];for(let M=e;M<=t;M++){const T=A*M,I=y*M;if(b>=3){for(let P=0;P<3;P++)E[I+P]=d[T+v*3+P];if(b>=8)for(let P=0;P<5;P++)E[I+3+P]=d[T+9+v*5+P]}}S(a.textures[v],a.size,a.elementsPerTexel,E,y)}}}const x=this.splatDataTextures.sceneIndexes,p=x.data;for(let _=this.lastBuildSplatCount;_<=t;_++)p[_]=this.globalSplatIndexToSceneIndexMap[_];const g=x.texture,m=this.renderer?this.renderer.properties.get(g):null;!m||!m.__webglTexture?g.needsUpdate=!0:this.updateDataTexture(p,x.texture,x.size,m,1,1,1,this.lastBuildSplatCount,t)}getTargetCovarianceCompressionLevel(){return this.halfPrecisionCovariancesOnGPU?1:0}getTargetSphericalHarmonicsCompressionLevel(){return Math.max(1,this.getMaximumSplatBufferCompressionLevel())}getMaximumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel>e)&&(e=s.compressionLevel)}return e}getMinimumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel<e)&&(e=s.compressionLevel)}return e}static computeTextureUpdateRegion(e,t,n,s,r){const o=r/s,a=e*o,l=Math.floor(a/n),c=l*n*s,u=t*o,f=Math.floor(u/n),h=f*n*s+n*s;return{dataStart:c,dataEnd:h,startRow:l,endRow:f}}updateDataTexture(e,t,n,s,r,o,a,l,c){const u=this.renderer.getContext(),f=sn.computeTextureUpdateRegion(l,c,n.x,r,o),h=f.dataEnd-f.dataStart,d=new e.constructor(e.buffer,f.dataStart*a,h),x=f.endRow-f.startRow+1,p=this.webGLUtils.convert(t.type),g=this.webGLUtils.convert(t.format,t.colorSpace),m=u.getParameter(u.TEXTURE_BINDING_2D);u.bindTexture(u.TEXTURE_2D,s.__webglTexture),u.texSubImage2D(u.TEXTURE_2D,0,0,f.startRow,n.x,x,g,p,d),u.bindTexture(u.TEXTURE_2D,m)}static updatePaddedCompressedCovariancesTextureData(e,t,n,s,r){let o=new DataView(t.buffer),a=n,l=0;for(let c=s;c<=r;c+=2)o.setUint16(a*2,e[c],!0),o.setUint16(a*2+2,e[c+1],!0),a+=2,l++,l>=3&&(a+=2,l=0)}static updateCenterColorsPaddedData(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*4,l=o*3,c=o*4;r[c]=vC(s,a),r[c+1]=pu(n[l]),r[c+2]=pu(n[l+1]),r[c+3]=pu(n[l+2])}}static updateScaleRotationsPaddedData(e,t,n,s,r){for(let a=e;a<=t;a++){const l=a*3,c=a*4,u=a*6;r[u]=n[l],r[u+1]=n[l+1],r[u+2]=n[l+2],r[u+3]=s[c],r[u+4]=s[c+1],r[u+5]=s[c+2]}}updateVisibleRegion(e){const t=this.getSplatCount(!0),n=new F;if(!e){const r=new F;this.scenes.forEach(o=>{r.add(o.splatBuffer.sceneCenter)}),r.multiplyScalar(1/this.scenes.length),this.calculatedSceneCenter.copy(r),this.material.uniforms.sceneCenter.value.copy(this.calculatedSceneCenter),this.material.uniformsNeedUpdate=!0}const s=e?this.lastBuildSplatCount:0;for(let r=s;r<t;r++){this.getSplatCenter(r,n,!0);const o=n.sub(this.calculatedSceneCenter).length();o>this.maxSplatDistanceFromSceneCenter&&(this.maxSplatDistanceFromSceneCenter=o)}this.maxSplatDistanceFromSceneCenter-this.visibleRegionBufferRadius>um&&(this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter,this.visibleRegionRadius=Math.max(this.visibleRegionBufferRadius-um,0)),this.finalBuild&&(this.visibleRegionRadius=this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter),this.updateVisibleRegionFadeDistance()}updateVisibleRegionFadeDistance(e=la.Default){const t=HE*this.sceneFadeInRateMultiplier,n=VE*this.sceneFadeInRateMultiplier,s=this.finalBuild?t:n,r=e===la.Default?s:n;this.visibleRegionFadeStartRadius=(this.visibleRegionRadius-this.visibleRegionFadeStartRadius)*r+this.visibleRegionFadeStartRadius;const a=(this.visibleRegionBufferRadius>0?this.visibleRegionFadeStartRadius/this.visibleRegionBufferRadius:0)>.99,l=a||e===la.Instant?1:0;this.material.uniforms.visibleRegionFadeStartRadius.value=this.visibleRegionFadeStartRadius,this.material.uniforms.visibleRegionRadius.value=this.visibleRegionRadius,this.material.uniforms.firstRenderTime.value=this.firstRenderTime,this.material.uniforms.currentTime.value=performance.now(),this.material.uniforms.fadeInComplete.value=l,this.material.uniformsNeedUpdate=!0,this.visibleRegionChanging=!a}updateRenderIndexes(e,t){const n=this.geometry;n.attributes.splatIndex.set(e),n.attributes.splatIndex.needsUpdate=!0,t>0&&this.firstRenderTime===-1&&(this.firstRenderTime=performance.now()),n.instanceCount=t,n.setDrawRange(0,t)}updateTransforms(){for(let e=0;e<this.scenes.length;e++)this.getScene(e).updateTransform(this.dynamicMode)}updateUniforms=(function(){const e=new Pe;return function(t,n,s,r,o,a){if(this.getSplatCount()>0){if(e.set(t.x*this.devicePixelRatio,t.y*this.devicePixelRatio),this.material.uniforms.viewport.value.copy(e),this.material.uniforms.basisViewport.value.set(1/e.x,1/e.y),this.material.uniforms.focal.value.set(n,s),this.material.uniforms.orthographicMode.value=r?1:0,this.material.uniforms.orthoZoom.value=o,this.material.uniforms.inverseFocalAdjustment.value=a,this.dynamicMode)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.transforms.value[c].copy(this.getScene(c).transform);if(this.enableOptionalEffects)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.sceneOpacity.value[c]=Rt(this.getScene(c).opacity,0,1),this.material.uniforms.sceneVisibility.value[c]=this.getScene(c).visible?1:0,this.material.uniformsNeedUpdate=!0;this.material.uniformsNeedUpdate=!0}}})();setSplatScale(e=1){this.splatScale=e,this.material.uniforms.splatScale.value=e,this.material.uniformsNeedUpdate=!0}getSplatScale(){return this.splatScale}setPointCloudModeEnabled(e){this.pointCloudModeEnabled=e,this.material.uniforms.pointCloudModeEnabled.value=e?1:0,this.material.uniformsNeedUpdate=!0}getPointCloudModeEnabled(){return this.pointCloudModeEnabled}getSplatDataTextures(){return this.splatDataTextures}getSplatCount(e=!1){return e?sn.getTotalSplatCountForScenes(this.scenes):this.lastBuildSplatCount}static getTotalSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getSplatCount());return t}static getTotalSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getSplatCount();return t}getMaxSplatCount(){return sn.getTotalMaxSplatCountForScenes(this.scenes)}static getTotalMaxSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getMaxSplatCount());return t}static getTotalMaxSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getMaxSplatCount();return t}disposeDistancesComputationGPUResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.vao&&(e.deleteVertexArray(this.distancesTransformFeedback.vao),this.distancesTransformFeedback.vao=null),this.distancesTransformFeedback.program&&(e.deleteProgram(this.distancesTransformFeedback.program),e.deleteShader(this.distancesTransformFeedback.vertexShader),e.deleteShader(this.distancesTransformFeedback.fragmentShader),this.distancesTransformFeedback.program=null,this.distancesTransformFeedback.vertexShader=null,this.distancesTransformFeedback.fragmentShader=null),this.disposeDistancesComputationGPUBufferResources(),this.distancesTransformFeedback.id&&(e.deleteTransformFeedback(this.distancesTransformFeedback.id),this.distancesTransformFeedback.id=null)}disposeDistancesComputationGPUBufferResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.centersBuffer&&(this.distancesTransformFeedback.centersBuffer=null,e.deleteBuffer(this.distancesTransformFeedback.centersBuffer)),this.distancesTransformFeedback.outDistancesBuffer&&(e.deleteBuffer(this.distancesTransformFeedback.outDistancesBuffer),this.distancesTransformFeedback.outDistancesBuffer=null)}setRenderer(e){if(e!==this.renderer){this.renderer=e;const t=this.renderer.getContext(),n=new FE(t),s=new LE(t,n,{});if(n.init(s),this.webGLUtils=new xg(t,n),this.enableDistancesComputationOnGPU&&this.getSplatCount()>0){this.setupDistancesComputationTransformFeedback();const{centers:r,sceneIndexes:o}=this.getDataForDistancesComputation(0,this.getSplatCount()-1);this.refreshGPUBuffersForDistancesComputation(r,o)}}}setupDistancesComputationTransformFeedback=(function(){let e;return function(){const t=this.getMaxSplatCount();if(!this.renderer)return;const n=this.lastRenderer!==this.renderer,s=e!==t;if(!n&&!s)return;n?this.disposeDistancesComputationGPUResources():s&&this.disposeDistancesComputationGPUBufferResources();const r=this.renderer.getContext(),o=(h,d,x)=>{const p=h.createShader(d);if(!p)return console.error("Fatal error: gl could not create a shader object."),null;if(h.shaderSource(p,x),h.compileShader(p),!h.getShaderParameter(p,h.COMPILE_STATUS)){let m="unknown";d===h.VERTEX_SHADER?m="vertex shader":d===h.FRAGMENT_SHADER&&(m="fragement shader");const _=h.getShaderInfoLog(p);return console.error("Failed to compile "+m+" with these errors:"+_),h.deleteShader(p),null}return p};let a;this.integerBasedDistancesComputation?(a=`#version 300 es
                in ivec4 center;
                flat out int distance;`,this.dynamicMode?a+=`
                        in uint sceneIndex;
                        uniform ivec4 transforms[${gt.MaxScenes}];
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
                        uniform mat4 transforms[${gt.MaxScenes}];
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
            `,c=r.getParameter(r.VERTEX_ARRAY_BINDING),u=r.getParameter(r.CURRENT_PROGRAM),f=u?r.getProgramParameter(u,r.DELETE_STATUS):!1;if(n&&(this.distancesTransformFeedback.vao=r.createVertexArray()),r.bindVertexArray(this.distancesTransformFeedback.vao),n){const h=r.createProgram(),d=o(r,r.VERTEX_SHADER,a),x=o(r,r.FRAGMENT_SHADER,l);if(!d||!x)throw new Error("Could not compile shaders for distances computation on GPU.");if(r.attachShader(h,d),r.attachShader(h,x),r.transformFeedbackVaryings(h,["distance"],r.SEPARATE_ATTRIBS),r.linkProgram(h),!r.getProgramParameter(h,r.LINK_STATUS)){const g=r.getProgramInfoLog(h);throw console.error("Fatal error: Failed to link program: "+g),r.deleteProgram(h),r.deleteShader(x),r.deleteShader(d),new Error("Could not link shaders for distances computation on GPU.")}this.distancesTransformFeedback.program=h,this.distancesTransformFeedback.vertexShader=d,this.distancesTransformFeedback.vertexShader=x}if(r.useProgram(this.distancesTransformFeedback.program),this.distancesTransformFeedback.centersLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"center"),this.dynamicMode){this.distancesTransformFeedback.sceneIndexesLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"sceneIndex");for(let h=0;h<this.scenes.length;h++)this.distancesTransformFeedback.transformsLocs[h]=r.getUniformLocation(this.distancesTransformFeedback.program,`transforms[${h}]`)}else this.distancesTransformFeedback.modelViewProjLoc=r.getUniformLocation(this.distancesTransformFeedback.program,"modelViewProj");(n||s)&&(this.distancesTransformFeedback.centersBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?r.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,r.INT,0,0):r.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,r.FLOAT,!1,0,0),this.dynamicMode&&(this.distancesTransformFeedback.sceneIndexesBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),r.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,r.UNSIGNED_INT,0,0))),(n||s)&&(this.distancesTransformFeedback.outDistancesBuffer=r.createBuffer()),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),r.bufferData(r.ARRAY_BUFFER,t*4,r.STATIC_READ),n&&(this.distancesTransformFeedback.id=r.createTransformFeedback()),r.bindTransformFeedback(r.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),r.bindBufferBase(r.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),u&&f!==!0&&r.useProgram(u),c&&r.bindVertexArray(c),this.lastRenderer=this.renderer,e=t}})();updateGPUCentersBufferForDistancesComputation(e,t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=this.integerBasedDistancesComputation?Uint32Array:Float32Array,a=16,l=n*a;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,l,t);else{const c=new o(this.getMaxSplatCount()*a);c.set(t),s.bufferData(s.ARRAY_BUFFER,c,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}updateGPUTransformIndexesBufferForDistancesComputation(e,t,n){if(!this.renderer||!this.dynamicMode)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=n*4;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,o,t);else{const a=new Uint32Array(this.getMaxSplatCount()*4);a.set(t),s.bufferData(s.ARRAY_BUFFER,a,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}getSceneIndexes(e,t){let n;const s=t-e+1;n=new Uint32Array(s);for(let r=e;r<=t;r++)n[r]=this.globalSplatIndexToSceneIndexMap[r];return n}fillTransformsArray=(function(){const e=[];return function(t){e.length!==t.length&&(e.length=t.length);for(let n=0;n<this.scenes.length;n++){const r=this.getScene(n).transform.elements;for(let o=0;o<16;o++)e[n*16+o]=r[o]}t.set(e)}})();computeDistancesOnGPU=(function(){const e=new Ye;return function(t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING),o=s.getParameter(s.CURRENT_PROGRAM),a=o?s.getProgramParameter(o,s.DELETE_STATUS):!1;if(s.bindVertexArray(this.distancesTransformFeedback.vao),s.useProgram(this.distancesTransformFeedback.program),s.enable(s.RASTERIZER_DISCARD),this.dynamicMode)for(let u=0;u<this.scenes.length;u++)if(e.copy(this.getScene(u).transform),e.premultiply(t),this.integerBasedDistancesComputation){const f=sn.getIntegerMatrixArray(e),h=[f[2],f[6],f[10],f[14]];s.uniform4i(this.distancesTransformFeedback.transformsLocs[u],h[0],h[1],h[2],h[3])}else s.uniformMatrix4fv(this.distancesTransformFeedback.transformsLocs[u],!1,e.elements);else if(this.integerBasedDistancesComputation){const u=sn.getIntegerMatrixArray(t),f=[u[2],u[6],u[10]];s.uniform3i(this.distancesTransformFeedback.modelViewProjLoc,f[0],f[1],f[2])}else{const u=[t.elements[2],t.elements[6],t.elements[10]];s.uniform3f(this.distancesTransformFeedback.modelViewProjLoc,u[0],u[1],u[2])}s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?s.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,s.INT,0,0):s.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,s.FLOAT,!1,0,0),this.dynamicMode&&(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),s.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,s.UNSIGNED_INT,0,0)),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),s.beginTransformFeedback(s.POINTS),s.drawArrays(s.POINTS,0,this.getSplatCount()),s.endTransformFeedback(),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,null),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,null),s.disable(s.RASTERIZER_DISCARD);const l=s.fenceSync(s.SYNC_GPU_COMMANDS_COMPLETE,0);s.flush();const c=new Promise(u=>{const f=()=>{if(this.disposed)u();else switch(s.clientWaitSync(l,0,0)){case s.TIMEOUT_EXPIRED:return this.computeDistancesOnGPUSyncTimeout=setTimeout(f),this.computeDistancesOnGPUSyncTimeout;case s.WAIT_FAILED:throw new Error("should never get here");default:this.computeDistancesOnGPUSyncTimeout=null,s.deleteSync(l);const p=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao),s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),s.getBufferSubData(s.ARRAY_BUFFER,0,n),s.bindBuffer(s.ARRAY_BUFFER,null),p&&s.bindVertexArray(p),u()}};this.computeDistancesOnGPUSyncTimeout=setTimeout(f)});return o&&a!==!0&&s.useProgram(o),r&&s.bindVertexArray(r),c}})();getLocalSplatParameters(e,t,n){n==null&&(n=!this.dynamicMode),t.splatBuffer=this.getSplatBufferForSplat(e),t.localIndex=this.getSplatLocalIndex(e),t.sceneTransform=n?this.getSceneTransformForSplat(e):null}fillSplatDataArrays(e,t,n,s,r,o,a,l=0,c=0,u=1,f,h,d=0,x){const p=new F;p.x=void 0,p.y=void 0,this.splatRenderMode===ls.ThreeD?p.z=void 0:p.z=1;const g=new Ye;let m=0,_=this.scenes.length-1;x!=null&&x>=0&&x<=this.scenes.length&&(m=x,_=x);for(let S=m;S<=_;S++){a==null&&(a=!this.dynamicMode);const A=this.getScene(S),y=A.splatBuffer;let b;if(a&&(this.getSceneTransform(S,g),b=g),e&&y.fillSplatCovarianceArray(e,b,f,h,d,l),t||n){if(!t||!n)throw new Error('SplatMesh::fillSplatDataArrays() -> "scales" and "rotations" must both be valid.');y.fillSplatScaleRotationArray(t,n,b,f,h,d,c,p)}s&&y.fillSplatCenterArray(s,b,f,h,d),r&&y.fillSplatColorArray(r,A.minimumAlpha,f,h,d),o&&y.fillSphericalHarmonicsArray(o,this.minSphericalHarmonicsDegree,b,f,h,d,u),d+=y.getSplatCount()}}getIntegerCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e);let o,a=n?4:3;o=new Int32Array(s*a);for(let l=0;l<s;l++){for(let c=0;c<3;c++)o[l*a+c]=Math.round(r[l*3+c]*1e3);n&&(o[l*a+3]=1e3)}return o}getFloatCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);if(this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e),!n)return r;let o=new Float32Array(s*4);for(let a=0;a<s;a++){for(let l=0;l<3;l++)o[a*4+l]=r[a*3+l];o[a*4+3]=1}return o}getSplatCenter=(function(){const e={};return function(t,n,s){this.getLocalSplatParameters(t,e,s),e.splatBuffer.getSplatCenter(e.localIndex,n,e.sceneTransform)}})();getSplatScaleAndRotation=(function(){const e={},t=new F;return function(n,s,r,o){this.getLocalSplatParameters(n,e,o),t.x=void 0,t.y=void 0,t.z=void 0,this.splatRenderMode===ls.TwoD&&(t.z=0),e.splatBuffer.getSplatScaleAndRotation(e.localIndex,s,r,e.sceneTransform,t)}})();getSplatColor=(function(){const e={};return function(t,n){this.getLocalSplatParameters(t,e),e.splatBuffer.getSplatColor(e.localIndex,n)}})();getSceneTransform(e,t){const n=this.getScene(e);n.updateTransform(this.dynamicMode),t.copy(n.transform)}getScene(e){if(e<0||e>=this.scenes.length)throw new Error("SplatMesh::getScene() -> Invalid scene index.");return this.scenes[e]}getSceneCount(){return this.scenes.length}getSplatBufferForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).splatBuffer}getSceneIndexForSplat(e){return this.globalSplatIndexToSceneIndexMap[e]}getSceneTransformForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).transform}getSplatLocalIndex(e){return this.globalSplatIndexToLocalSplatIndexMap[e]}static getIntegerMatrixArray(e){const t=e.elements,n=[];for(let s=0;s<16;s++)n[s]=Math.round(t[s]*1e3);return n}computeBoundingBox(e=!1,t){let n=this.getSplatCount();if(t!=null){if(t<0||t>=this.scenes.length)throw new Error("SplatMesh::computeBoundingBox() -> Invalid scene index.");n=this.scenes[t].splatBuffer.getSplatCount()}const s=new Float32Array(n*3);this.fillSplatDataArrays(null,null,null,s,null,null,e,void 0,void 0,void 0,void 0,t);const r=new F,o=new F;for(let a=0;a<n;a++){const l=a*3,c=s[l],u=s[l+1],f=s[l+2];(a===0||c<r.x)&&(r.x=c),(a===0||u<r.y)&&(r.y=u),(a===0||f<r.z)&&(r.z=f),(a===0||c>o.x)&&(o.x=c),(a===0||u>o.y)&&(o.y=u),(a===0||f>o.z)&&(o.z=f)}return new Ni(r,o)}}var GE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEbA2AAAGAQf39/f39/f39/f39/f39/fwBgAAF/AhIBA2VudgZtZW1vcnkCAwCAgAQDBAMAAQIHVAQRX193YXNtX2NhbGxfY3RvcnMAABhfX3dhc21fYXBwbHlfZGF0YV9yZWxvY3MAAAtzb3J0SW5kZXhlcwABE2Vtc2NyaXB0ZW5fdGxzX2luaXQAAgqWEAMDAAELihAEAXwDewN/A30gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEBA0AgAyABQQJ0IgVqIAIgACAFaigCAEECdGooAgAiBTYCACAFIAogBSAKSBshCiAFIA0gBSANShshDSABQQFqIgEgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiFWooAgAiFkECdGooAgAiFEcEQAJ/IAX9CQI4IAggFEEGdGoiDv0JAgwgDioCHP0gASAOKgIs/SACIA4qAjz9IAP95gEgBf0JAiggDv0JAgggDioCGP0gASAOKgIo/SACIA4qAjj9IAP95gEgBf0JAgggDv0JAgAgDioCEP0gASAOKgIg/SACIA4qAjD9IAP95gEgBf0JAhggDv0JAgQgDioCFP0gASAOKgIk/SACIA4qAjT9IAP95gH95AH95AH95AEiEf1f/QwAAAAAAECPQAAAAAAAQI9AIhL98gEiE/0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBP9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/REgDv0cAQJ/IBEgEf0NCAkKCwwNDg8AAAAAAAAAAP1fIBL98gEiEf0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9HAICfyAR/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAyESIBQhDwsgAyAVaiABIBZBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmogEf0bA2oiDjYCACAOIAogCiAOShshCiAOIA0gDSAOSBshDSACQQFqIgIgC0cNAAsMAwsCfyAFKgIIu/0UIAUqAhi7/SIB/QwAAAAAAECPQAAAAAAAQI9A/fIBIhH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIQ4CfyAR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyECAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQIgAv0RIA79HAEgBf0cAiESIAwhBQNAIAMgBUECdCICaiABIAAgAmooAgBBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmoiAjYCACACIAogAiAKSBshCiACIA0gAiANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEBA0AgAyABQQJ0IgVqAn8gAiAAIAVqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAFBAWoiASALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIRcgBSoCGCEYIAUqAgghGUH4////ByEKQYiAgIB4IQ0gDCEFA0ACfyAXIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCAZIAIqAgCUIBggAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIUaigCAEECdCIVaigCACIORwRAIAX9CQI4IAggDkEGdGoiD/0JAgwgDyoCHP0gASAPKgIs/SACIA8qAjz9IAP95gEgBf0JAiggD/0JAgggDyoCGP0gASAPKgIo/SACIA8qAjj9IAP95gEgBf0JAgggD/0JAgAgDyoCEP0gASAPKgIg/SACIA8qAjD9IAP95gEgBf0JAhggD/0JAgQgDyoCFP0gASAPKgIk/SACIA8qAjT9IAP95gH95AH95AH95AEhESAOIQ8LIAMgFGoCfyAR/R8DIAEgFUECdCIOQQxyaioCAJQgEf0fAiABIA5BCHJqKgIAlCAR/R8AIAEgDmoqAgCUIBH9HwEgASAOQQRyaioCAJSSkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSACQQFqIgIgC0cNAAsMAQtBiICAgHghDUH4////ByEKCyALIAxLBEAgCUEBa7MgDbIgCrKTlSEXIAwhDQNAAn8gFyADIA1BAnRqIgEoAgAgCmuylCIYi0MAAABPXQRAIBioDAELQYCAgIB4CyEOIAEgDjYCACAEIA5BAnRqIgEgASgCAEEBajYCACANQQFqIg0gC0cNAAsLIAlBAk8EQCAEKAIAIQ1BASEKA0AgBCAKQQJ0aiIBIAEoAgAgDWoiDTYCACAKQQFqIgogCUcNAAsLIAxBAEoEQCAMIQoDQCAGIApBAWsiAUECdCICaiAAIAJqKAIANgIAIApBAUshAiABIQogAg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCwsEAEEACw==",hm="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACEgEDZW52Bm1lbW9yeQIDAICABAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=",WE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQrrDwICAAvlDwQBfAN7B30DfyALIAprIQwCQAJAIA4EQCANBEBB+P///wchCkGIgICAeCENIAsgDE0NAyAMIQUDQCADIAVBAnQiAWogAiAAIAFqKAIAQQJ0aigCACIBNgIAIAEgCiABIApIGyEKIAEgDSABIA1KGyENIAVBAWoiBSALRw0ACwwDCyAPBEAgCyAMTQ0CQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIcaigCACIdQQJ0aigCACIbRwRAAn8gBf0JAjggCCAbQQZ0aiIO/QkCDCAOKgIc/SABIA4qAiz9IAIgDioCPP0gA/3mASAF/QkCKCAO/QkCCCAOKgIY/SABIA4qAij9IAIgDioCOP0gA/3mASAF/QkCCCAO/QkCACAOKgIQ/SABIA4qAiD9IAIgDioCMP0gA/3mASAF/QkCGCAO/QkCBCAOKgIU/SABIA4qAiT9IAIgDioCNP0gA/3mAf3kAf3kAf3kASIR/V/9DAAAAAAAQI9AAAAAAABAj0AiEv3yASIT/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOAn8gE/0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9ESAO/RwBAn8gESAR/Q0ICQoLDA0ODwABAgMAAQID/V8gEv3yASIR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAgJ/IBH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/RwDIRIgGyEPCyADIBxqIAEgHUEEdGr9AAAAIBL9tQEiEf0bACAR/RsBaiAR/RsCaiAR/RsDaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAgi7/RQgBSoCGLv9IgH9DAAAAAAAQI9AAAAAAABAj0D98gEiEf0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBH9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQL9ESAO/RwBIAX9HAIhEiAMIQUDQCADIAVBAnQiAmogASAAIAJqKAIAQQR0av0AAAAgEv21ASIR/RsAIBH9GwFqIBH9GwJqIgI2AgAgAiAKIAIgCkgbIQogAiANIAIgDUobIQ0gBUEBaiIFIAtHDQALDAILIA0EQEH4////ByEKQYiAgIB4IQ0gCyAMTQ0CIAwhBQNAIAMgBUECdCIBagJ/IAIgACABaigCAEECdGoqAgC7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgD0UEQCALIAxNDQEgBSoCKCEUIAUqAhghFSAFKgIIIRZB+P///wchCkGIgICAeCENIAwhBQNAAn8gFCABIAAgBUECdCIHaigCAEEEdGoiAioCCJQgFiACKgIAlCAVIAIqAgSUkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDiADIAdqIA42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gBUEBaiIFIAtHDQALDAILIAsgDE0NAEF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiG2ooAgBBAnQiHGooAgAiDkcEQCAFKgI4IhQgCCAOQQZ0aiIPKgI8lCAFKgIoIhUgDyoCOJQgBSoCCCIWIA8qAjCUIAUqAhgiFyAPKgI0lJKSkiEYIBQgDyoCLJQgFSAPKgIolCAWIA8qAiCUIBcgDyoCJJSSkpIhGSAUIA8qAhyUIBUgDyoCGJQgFiAPKgIQlCAXIA8qAhSUkpKSIRogFCAPKgIMlCAVIA8qAgiUIBYgDyoCAJQgFyAPKgIElJKSkiEUIA4hDwsgAyAbagJ/IBggASAcQQJ0aiIOKgIMlCAZIA4qAgiUIBQgDioCAJQgGiAOKgIElJKSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAJBAWoiAiALRw0ACwwBC0GIgICAeCENQfj///8HIQoLIAsgDEsEQCAJQQFrsyANsiAKspOVIRQgDCENA0ACfyAUIAMgDUECdGoiASgCACAKa7KUIhWLQwAAAE9dBEAgFagMAQtBgICAgHgLIQ4gASAONgIAIAQgDkECdGoiASABKAIAQQFqNgIAIA1BAWoiDSALRw0ACwsgCUECTwRAIAQoAgAhDUEBIQoDQCAEIApBAnRqIgEgASgCACANaiINNgIAIApBAWoiCiAJRw0ACwsgDEEASgRAIAwhCgNAIAYgCkEBayIBQQJ0IgJqIAAgAmooAgA2AgAgCkEBSyABIQoNAAsLIAsgDEoEQCALIQoDQCAGIAsgBCADIApBAWsiCkECdCIBaigCAEECdGoiAigCACIFa0ECdGogACABaigCADYCACACIAVBAWs2AgAgCiAMSg0ACwsL",XE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=";function qE(i){let e,t,n,s,r,o,a,l,c,u,f,h,d,x,p,g,m,_,S,A;function y(b,v,E,M,T,I,P){const B=performance.now();if(!n&&(new Uint32Array(t,a,T.byteLength/A.BytesPerInt).set(T),new Float32Array(t,u,P.byteLength/A.BytesPerFloat).set(P),M)){let X;s?X=new Int32Array(t,f,I.byteLength/A.BytesPerInt):X=new Float32Array(t,f,I.byteLength/A.BytesPerFloat),X.set(I)}g||(g=new Uint32Array(_)),new Float32Array(t,p,16).set(E),new Uint32Array(t,d,_).set(g),e.exports.sortIndexes(a,x,f,h,d,p,l,c,u,_,b,v,o,M,s,r);const N={sortDone:!0,splatSortCount:b,splatRenderCount:v,sortTime:0};if(!n){const V=new Uint32Array(t,l,v);(!m||m.length<v)&&(m=new Uint32Array(v)),m.set(V),N.sortedIndexes=m}const G=performance.now();N.sortTime=G-B,i.postMessage(N)}i.onmessage=b=>{if(b.data.centers)centers=b.data.centers,sceneIndexes=b.data.sceneIndexes,s?new Int32Array(t,x+b.data.range.from*A.BytesPerInt*4,b.data.range.count*4).set(new Int32Array(centers)):new Float32Array(t,x+b.data.range.from*A.BytesPerFloat*4,b.data.range.count*4).set(new Float32Array(centers)),r&&new Uint32Array(t,c+b.data.range.from*4,b.data.range.count).set(new Uint32Array(sceneIndexes)),S=b.data.range.from+b.data.range.count;else if(b.data.sort){const v=Math.min(b.data.sort.splatRenderCount||0,S),E=Math.min(b.data.sort.splatSortCount||0,S),M=b.data.sort.usePrecomputedDistances;let T,I,P;n||(T=b.data.sort.indexesToSort,P=b.data.sort.transforms,M&&(I=b.data.sort.precomputedDistances)),y(E,v,b.data.sort.modelViewProj,M,T,I,P)}else if(b.data.init){A=b.data.init.Constants,o=b.data.init.splatCount,n=b.data.init.useSharedMemory,s=b.data.init.integerBasedSort,r=b.data.init.dynamicMode,_=b.data.init.distanceMapRange,S=0;const v=s?A.BytesPerInt*4:A.BytesPerFloat*4,E=new Uint8Array(b.data.init.sorterWasmBytes),M=16*A.BytesPerFloat,T=o*A.BytesPerInt,I=o*v,P=M,B=s?o*A.BytesPerInt:o*A.BytesPerFloat,N=o*A.BytesPerInt,G=o*A.BytesPerInt,V=s?_*A.BytesPerInt*2:_*A.BytesPerFloat*2,q=r?o*A.BytesPerInt:0,X=r?A.MaxScenes*M:0,ee=A.MemoryPageSize*32,ce=T+I+P+B+N+V+G+q+X+ee,be=Math.floor(ce/A.MemoryPageSize)+1,Re={module:{},env:{memory:new WebAssembly.Memory({initial:be,maximum:be,shared:!0})}};WebAssembly.compile(E).then(Fe=>WebAssembly.instantiate(Fe,Re)).then(Fe=>{e=Fe,a=0,x=a+T,p=x+I,f=p+P,h=f+B,d=h+N,l=d+V,c=l+G,u=c+q,t=Re.env.memory.buffer,n?i.postMessage({sortSetupPhase1Complete:!0,indexesToSortBuffer:t,indexesToSortOffset:a,sortedIndexesBuffer:t,sortedIndexesOffset:l,precomputedDistancesBuffer:t,precomputedDistancesOffset:f,transformsBuffer:t,transformsOffset:u}):i.postMessage({sortSetupPhase1Complete:!0})})}}}function YE(i,e,t,n,s,r=gt.DefaultSplatSortDistanceMapPrecision){const o=new Worker(URL.createObjectURL(new Blob(["(",qE.toString(),")(self)"],{type:"application/javascript"})));let a=GE;const l=wh()?vg():null;!t&&!e?(a=hm,l&&l.major<=16&&l.minor<4&&(a=XE)):t?e||l&&l.major<=16&&l.minor<4&&(a=WE):a=hm;const c=atob(a),u=new Uint8Array(c.length);for(let f=0;f<c.length;f++)u[f]=c.charCodeAt(f);return o.postMessage({init:{sorterWasmBytes:u.buffer,splatCount:i,useSharedMemory:e,integerBasedSort:n,dynamicMode:s,distanceMapRange:1<<r,Constants:{BytesPerFloat:gt.BytesPerFloat,BytesPerInt:gt.BytesPerInt,MemoryPageSize:gt.MemoryPageSize,MaxScenes:gt.MaxScenes}}}),o}const fr={None:0,VR:1,AR:2};class Co{static createButton(e,t={}){const n=document.createElement("button");function s(){let c=null;async function u(d){d.addEventListener("end",f),await e.xr.setSession(d),n.textContent="EXIT VR",c=d}function f(){c.removeEventListener("end",f),n.textContent="ENTER VR",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="ENTER VR";const h={...t,optionalFeatures:["local-floor","bounded-floor","layers",...t.optionalFeatures||[]]};n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-vr",h).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",h).then(u).catch(d=>{console.warn(d)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",h).then(u).catch(d=>{console.warn(d)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="VR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="VR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="VRButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-vr").then(function(c){c?s():o(),c&&Co.xrSessionIsGranted&&n.click()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}static registerSessionGrantedListener(){if(typeof navigator<"u"&&"xr"in navigator){if(/WebXRViewer\//i.test(navigator.userAgent))return;navigator.xr.addEventListener("sessiongranted",()=>{Co.xrSessionIsGranted=!0})}}}Co.xrSessionIsGranted=!1;Co.registerSessionGrantedListener();class QE{static createButton(e,t={}){const n=document.createElement("button");function s(){if(t.domOverlay===void 0){const h=document.createElement("div");h.style.display="none",document.body.appendChild(h);const d=document.createElementNS("http://www.w3.org/2000/svg","svg");d.setAttribute("width",38),d.setAttribute("height",38),d.style.position="absolute",d.style.right="20px",d.style.top="20px",d.addEventListener("click",function(){c.end()}),h.appendChild(d);const x=document.createElementNS("http://www.w3.org/2000/svg","path");x.setAttribute("d","M 12,12 L 28,28 M 28,12 12,28"),x.setAttribute("stroke","#fff"),x.setAttribute("stroke-width",2),d.appendChild(x),t.optionalFeatures===void 0&&(t.optionalFeatures=[]),t.optionalFeatures.push("dom-overlay"),t.domOverlay={root:h}}let c=null;async function u(h){h.addEventListener("end",f),e.xr.setReferenceSpaceType("local"),await e.xr.setSession(h),n.textContent="STOP AR",t.domOverlay.root.style.display="",c=h}function f(){c.removeEventListener("end",f),n.textContent="START AR",t.domOverlay.root.style.display="none",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="START AR",n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-ar",t).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(h=>{console.warn(h)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(h=>{console.warn(h)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="AR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="AR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="ARButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-ar").then(function(c){c?s():o()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}}const bu={Always:0,Never:2},KE=50,jE=.75,$E=15e5,ZE=10,JE=2.5,e1=60;class eo{constructor(e={}){if(e.cameraUp||(e.cameraUp=[0,1,0]),this.cameraUp=new F().fromArray(e.cameraUp),e.initialCameraPosition||(e.initialCameraPosition=[0,10,15]),this.initialCameraPosition=new F().fromArray(e.initialCameraPosition),e.initialCameraLookAt||(e.initialCameraLookAt=[0,0,0]),this.initialCameraLookAt=new F().fromArray(e.initialCameraLookAt),this.dropInMode=e.dropInMode||!1,(e.selfDrivenMode===void 0||e.selfDrivenMode===null)&&(e.selfDrivenMode=!0),this.selfDrivenMode=e.selfDrivenMode&&!this.dropInMode,this.selfDrivenUpdateFunc=this.selfDrivenUpdate.bind(this),e.useBuiltInControls===void 0&&(e.useBuiltInControls=!0),this.useBuiltInControls=e.useBuiltInControls,this.rootElement=e.rootElement,this.ignoreDevicePixelRatio=e.ignoreDevicePixelRatio||!1,this.devicePixelRatio=this.ignoreDevicePixelRatio?1:window.devicePixelRatio||1,this.halfPrecisionCovariancesOnGPU=e.halfPrecisionCovariancesOnGPU||!1,this.threeScene=e.threeScene,this.renderer=e.renderer,this.camera=e.camera,this.gpuAcceleratedSort=e.gpuAcceleratedSort||!1,(e.integerBasedSort===void 0||e.integerBasedSort===null)&&(e.integerBasedSort=!0),this.integerBasedSort=e.integerBasedSort,(e.sharedMemoryForWorkers===void 0||e.sharedMemoryForWorkers===null)&&(e.sharedMemoryForWorkers=!0),this.sharedMemoryForWorkers=e.sharedMemoryForWorkers,this.dynamicScene=!!e.dynamicScene,this.antialiased=e.antialiased||!1,this.kernel2DSize=e.kernel2DSize===void 0?.3:e.kernel2DSize,this.webXRMode=e.webXRMode||fr.None,this.webXRMode!==fr.None&&(this.gpuAcceleratedSort=!1),this.webXRActive=!1,this.webXRSessionInit=e.webXRSessionInit||{},this.renderMode=e.renderMode||bu.Always,this.sceneRevealMode=e.sceneRevealMode||la.Default,this.focalAdjustment=e.focalAdjustment||1,this.maxScreenSpaceSplatSize=e.maxScreenSpaceSplatSize||1024,this.logLevel=e.logLevel||ho.None,this.sphericalHarmonicsDegree=e.sphericalHarmonicsDegree||0,this.enableOptionalEffects=e.enableOptionalEffects||!1,(e.enableSIMDInSort===void 0||e.enableSIMDInSort===null)&&(e.enableSIMDInSort=!0),this.enableSIMDInSort=e.enableSIMDInSort,(e.inMemoryCompressionLevel===void 0||e.inMemoryCompressionLevel===null)&&(e.inMemoryCompressionLevel=0),this.inMemoryCompressionLevel=e.inMemoryCompressionLevel,(e.optimizeSplatData===void 0||e.optimizeSplatData===null)&&(e.optimizeSplatData=!0),this.optimizeSplatData=e.optimizeSplatData,(e.freeIntermediateSplatData===void 0||e.freeIntermediateSplatData===null)&&(e.freeIntermediateSplatData=!1),this.freeIntermediateSplatData=e.freeIntermediateSplatData,wh()){const n=vg();n.major<17&&(this.enableSIMDInSort=!1),n.major<16&&(this.sharedMemoryForWorkers=!1)}(e.splatRenderMode===void 0||e.splatRenderMode===null)&&(e.splatRenderMode=ls.ThreeD),this.splatRenderMode=e.splatRenderMode,this.sceneFadeInRateMultiplier=e.sceneFadeInRateMultiplier||1,this.splatSortDistanceMapPrecision=e.splatSortDistanceMapPrecision||gt.DefaultSplatSortDistanceMapPrecision;const t=this.integerBasedSort?20:24;this.splatSortDistanceMapPrecision=Rt(this.splatSortDistanceMapPrecision,10,t),this.onSplatMeshChangedCallback=null,this.createSplatMesh(),this.controls=null,this.perspectiveControls=null,this.orthographicControls=null,this.orthographicCamera=null,this.perspectiveCamera=null,this.showMeshCursor=!1,this.showControlPlane=!1,this.showInfo=!1,this.sceneHelper=null,this.sortWorker=null,this.sortRunning=!1,this.splatRenderCount=0,this.splatSortCount=0,this.lastSplatSortCount=0,this.sortWorkerIndexesToSort=null,this.sortWorkerSortedIndexes=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.preSortMessages=[],this.runAfterNextSort=[],this.selfDrivenModeRunning=!1,this.splatRenderReady=!1,this.raycaster=new CE,this.infoPanel=null,this.startInOrthographicMode=!1,this.currentFPS=0,this.lastSortTime=0,this.consecutiveRenderFrames=0,this.previousCameraTarget=new F,this.nextCameraTarget=new F,this.mousePosition=new Pe,this.mouseDownPosition=new Pe,this.mouseDownTime=null,this.resizeObserver=null,this.mouseMoveListener=null,this.mouseDownListener=null,this.mouseUpListener=null,this.keyDownListener=null,this.sortPromise=null,this.sortPromiseResolver=null,this.splatSceneDownloadPromises={},this.splatSceneDownloadAndBuildPromise=null,this.splatSceneRemovalPromise=null,this.loadingSpinner=new Hh(null,this.rootElement||document.body),this.loadingSpinner.hide(),this.loadingProgressBar=new SE(this.rootElement||document.body),this.loadingProgressBar.hide(),this.infoPanel=new AE(this.rootElement||document.body),this.infoPanel.hide(),this.usingExternalCamera=!!(this.dropInMode||this.camera),this.usingExternalRenderer=!!(this.dropInMode||this.renderer),this.initialized=!1,this.disposing=!1,this.disposed=!1,this.disposePromise=null,this.dropInMode||this.init()}createSplatMesh(){this.splatMesh=new sn(this.splatRenderMode,this.dynamicScene,this.enableOptionalEffects,this.halfPrecisionCovariancesOnGPU,this.devicePixelRatio,this.gpuAcceleratedSort,this.integerBasedSort,this.antialiased,this.maxScreenSpaceSplatSize,this.logLevel,this.sphericalHarmonicsDegree,this.sceneFadeInRateMultiplier,this.kernel2DSize),this.splatMesh.frustumCulled=!1,this.onSplatMeshChangedCallback&&this.onSplatMeshChangedCallback()}init(){this.initialized||(this.rootElement||(this.usingExternalRenderer?this.rootElement=this.renderer.domElement||document.body:(this.rootElement=document.createElement("div"),this.rootElement.style.width="100%",this.rootElement.style.height="100%",this.rootElement.style.position="absolute",document.body.appendChild(this.rootElement))),this.setupCamera(),this.setupRenderer(),this.setupWebXR(this.webXRSessionInit),this.setupControls(),this.setupEventHandlers(),this.threeScene=this.threeScene||new AA,this.sceneHelper=new oa(this.threeScene),this.sceneHelper.setupMeshCursor(),this.sceneHelper.setupFocusMarker(),this.sceneHelper.setupControlPlane(),this.loadingProgressBar.setContainer(this.rootElement),this.loadingSpinner.setContainer(this.rootElement),this.infoPanel.setContainer(this.rootElement),this.initialized=!0)}setupCamera(){if(!this.usingExternalCamera){const e=new Pe;this.getRenderDimensions(e),this.perspectiveCamera=new ui(KE,e.x/e.y,.1,1e3),this.orthographicCamera=new Ch(e.x/-2,e.x/2,e.y/2,e.y/-2,.1,1e3),this.camera=this.startInOrthographicMode?this.orthographicCamera:this.perspectiveCamera,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt)}}setupRenderer(){if(!this.usingExternalRenderer){const e=new Pe;this.getRenderDimensions(e),this.renderer=new _C({antialias:!1,precision:"highp"}),this.renderer.setPixelRatio(this.devicePixelRatio),this.renderer.autoClear=!0,this.renderer.setClearColor(new rt(0),0),this.renderer.setSize(e.x,e.y),this.resizeObserver=new ResizeObserver(()=>{this.getRenderDimensions(e),this.renderer.setSize(e.x,e.y),this.forceRenderNextFrame()}),this.resizeObserver.observe(this.rootElement),this.rootElement.appendChild(this.renderer.domElement)}}setupWebXR(e){this.webXRMode&&(this.webXRMode===fr.VR?this.rootElement.appendChild(Co.createButton(this.renderer,e)):this.webXRMode===fr.AR&&this.rootElement.appendChild(QE.createButton(this.renderer,e)),this.renderer.xr.addEventListener("sessionstart",t=>{this.webXRActive=!0}),this.renderer.xr.addEventListener("sessionend",t=>{this.webXRActive=!1}),this.renderer.xr.enabled=!0,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt))}setupControls(){if(this.useBuiltInControls&&this.webXRMode===fr.None){this.usingExternalCamera?this.camera.isOrthographicCamera?this.orthographicControls=new vl(this.camera,this.renderer.domElement):this.perspectiveControls=new vl(this.camera,this.renderer.domElement):(this.perspectiveControls=new vl(this.perspectiveCamera,this.renderer.domElement),this.orthographicControls=new vl(this.orthographicCamera,this.renderer.domElement));for(let e of[this.orthographicControls,this.perspectiveControls])e&&(e.listenToKeyEvents(window),e.rotateSpeed=.5,e.maxPolarAngle=Math.PI*.75,e.minPolarAngle=.1,e.enableDamping=!0,e.dampingFactor=.05,e.target.copy(this.initialCameraLookAt),e.update());this.controls=this.camera.isOrthographicCamera?this.orthographicControls:this.perspectiveControls,this.controls.update()}}setupEventHandlers(){this.useBuiltInControls&&this.webXRMode===fr.None&&(this.mouseMoveListener=this.onMouseMove.bind(this),this.renderer.domElement.addEventListener("pointermove",this.mouseMoveListener,!1),this.mouseDownListener=this.onMouseDown.bind(this),this.renderer.domElement.addEventListener("pointerdown",this.mouseDownListener,!1),this.mouseUpListener=this.onMouseUp.bind(this),this.renderer.domElement.addEventListener("pointerup",this.mouseUpListener,!1),this.keyDownListener=this.onKeyDown.bind(this),window.addEventListener("keydown",this.keyDownListener,!1))}removeEventHandlers(){this.useBuiltInControls&&(this.renderer.domElement.removeEventListener("pointermove",this.mouseMoveListener),this.mouseMoveListener=null,this.renderer.domElement.removeEventListener("pointerdown",this.mouseDownListener),this.mouseDownListener=null,this.renderer.domElement.removeEventListener("pointerup",this.mouseUpListener),this.mouseUpListener=null,window.removeEventListener("keydown",this.keyDownListener),this.keyDownListener=null)}setRenderMode(e){this.renderMode=e}setActiveSphericalHarmonicsDegrees(e){this.splatMesh.material.uniforms.sphericalHarmonicsDegree.value=e,this.splatMesh.material.uniformsNeedUpdate=!0}onSplatMeshChanged(e){this.onSplatMeshChangedCallback=e}onKeyDown=(function(){const e=new F,t=new Ye,n=new Ye;return function(s){switch(e.set(0,0,-1),e.transformDirection(this.camera.matrixWorld),t.makeRotationAxis(e,Math.PI/128),n.makeRotationAxis(e,-Math.PI/128),s.code){case"KeyG":this.focalAdjustment+=.02,this.forceRenderNextFrame();break;case"KeyF":this.focalAdjustment-=.02,this.forceRenderNextFrame();break;case"ArrowLeft":this.camera.up.transformDirection(t);break;case"ArrowRight":this.camera.up.transformDirection(n);break;case"KeyC":this.showMeshCursor=!this.showMeshCursor;break;case"KeyU":this.showControlPlane=!this.showControlPlane;break;case"KeyI":this.showInfo=!this.showInfo,this.showInfo?this.infoPanel.show():this.infoPanel.hide();break;case"KeyO":this.usingExternalCamera||this.setOrthographicMode(!this.camera.isOrthographicCamera);break;case"KeyP":this.usingExternalCamera||this.splatMesh.setPointCloudModeEnabled(!this.splatMesh.getPointCloudModeEnabled());break;case"Equal":this.usingExternalCamera||this.splatMesh.setSplatScale(this.splatMesh.getSplatScale()+.05);break;case"Minus":this.usingExternalCamera||this.splatMesh.setSplatScale(Math.max(this.splatMesh.getSplatScale()-.05,0));break}}})();onMouseMove(e){this.mousePosition.set(e.offsetX,e.offsetY)}onMouseDown(){this.mouseDownPosition.copy(this.mousePosition),this.mouseDownTime=qr()}onMouseUp=(function(){const e=new Pe;return function(t){e.copy(this.mousePosition).sub(this.mouseDownPosition),qr()-this.mouseDownTime<.5&&e.length()<2&&this.onMouseClick(t)}})();onMouseClick(e){this.mousePosition.set(e.offsetX,e.offsetY),this.checkForFocalPointChange()}checkForFocalPointChange=(function(){const e=new Pe,t=new F,n=[];return function(){if(!this.transitioningCameraTarget&&(this.getRenderDimensions(e),n.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,e),this.raycaster.intersectSplatMesh(this.splatMesh,n),n.length>0)){const r=n[0].origin;t.copy(r).sub(this.camera.position),t.length()>jE&&(this.previousCameraTarget.copy(this.controls.target),this.nextCameraTarget.copy(r),this.transitioningCameraTarget=!0,this.transitioningCameraTargetStartTime=qr())}}})();getRenderDimensions(e){this.rootElement?(e.x=this.rootElement.offsetWidth,e.y=this.rootElement.offsetHeight):this.renderer.getSize(e)}setOrthographicMode(e){if(e===this.camera.isOrthographicCamera)return;const t=this.camera,n=e?this.orthographicCamera:this.perspectiveCamera;if(n.position.copy(t.position),n.up.copy(t.up),n.rotation.copy(t.rotation),n.quaternion.copy(t.quaternion),n.matrix.copy(t.matrix),this.camera=n,this.controls){const s=a=>{a.saveState(),a.reset()},r=this.controls,o=e?this.orthographicControls:this.perspectiveControls;s(o),s(r),o.target.copy(r.target),e?eo.setCameraZoomFromPosition(n,t,r):eo.setCameraPositionFromZoom(n,t,o),this.controls=o,this.camera.lookAt(this.controls.target)}}static setCameraPositionFromZoom=(function(){const e=new F;return function(t,n,s){const r=1/(n.zoom*.001);e.copy(s.target).sub(t.position).normalize().multiplyScalar(r).negate(),t.position.copy(s.target).add(e)}})();static setCameraZoomFromPosition=(function(){const e=new F;return function(t,n,s){const r=e.copy(s.target).sub(n.position).length();t.zoom=1/(r*.001)}})();updateSplatMesh=(function(){const e=new Pe;return function(){if(!this.splatMesh)return;if(this.splatMesh.getSplatCount()>0){this.splatMesh.updateVisibleRegionFadeDistance(this.sceneRevealMode),this.splatMesh.updateTransforms(),this.getRenderDimensions(e);const n=this.camera.projectionMatrix.elements[0]*.5*this.devicePixelRatio*e.x,s=this.camera.projectionMatrix.elements[5]*.5*this.devicePixelRatio*e.y,r=this.camera.isOrthographicCamera?1/this.devicePixelRatio:1,o=this.focalAdjustment*r,a=1/o;this.adjustForWebXRStereo(e),this.splatMesh.updateUniforms(e,n*o,s*o,this.camera.isOrthographicCamera,this.camera.zoom||1,a)}}})();adjustForWebXRStereo(e){if(this.camera&&this.webXRActive){const n=this.renderer.xr.getCamera().projectionMatrix.elements[0],s=this.camera.projectionMatrix.elements[0];e.x*=s/n}}isLoadingOrUnloading(){return Object.keys(this.splatSceneDownloadPromises).length>0||this.splatSceneDownloadAndBuildPromise!==null||this.splatSceneRemovalPromise!==null}isDisposingOrDisposed(){return this.disposing||this.disposed}addSplatSceneDownloadPromise(e){this.splatSceneDownloadPromises[e.id]=e}removeSplatSceneDownloadPromise(e){delete this.splatSceneDownloadPromises[e.id]}setSplatSceneDownloadAndBuildPromise(e){this.splatSceneDownloadAndBuildPromise=e}clearSplatSceneDownloadAndBuildPromise(){this.splatSceneDownloadAndBuildPromise=null}addSplatScene(e,t={}){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");t.progressiveLoad&&this.splatMesh.scenes&&this.splatMesh.scenes.length>0&&(console.log('addSplatScene(): "progressiveLoad" option ignore because there are multiple splat scenes'),t.progressiveLoad=!1);const n=t.format!==void 0&&t.format!==null?t.format:im(e),s=eo.isProgressivelyLoadable(n)&&t.progressiveLoad,r=t.showLoadingUI!==void 0&&t.showLoadingUI!==null?t.showLoadingUI:!0;let o=null;r&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=()=>{this.loadingProgressBar.hide(),this.loadingSpinner.removeAllTasks()},l=(p,g,m)=>{if(r)if(m===Gt.Downloading)if(p==100)this.loadingSpinner.setMessageForTask(o,"Download complete!");else if(s)this.loadingSpinner.setMessageForTask(o,"Downloading splats...");else{const _=g?`: ${g}`:"...";this.loadingSpinner.setMessageForTask(o,`Downloading${_}`)}else m===Gt.Processing&&this.loadingSpinner.setMessageForTask(o,"Processing splats...")};let c=!1,u=0;const f=(p,g)=>{r&&((p&&s||g&&!s)&&(this.loadingSpinner.removeTask(o),!g&&!c&&this.loadingProgressBar.show()),s&&(g?(c=!0,this.loadingProgressBar.hide()):this.loadingProgressBar.setProgress(u)))},h=(p,g,m)=>{u=p,l(p,g,m),t.onProgress&&t.onProgress(p,g,m)},d=(p,g,m)=>{!s&&t.onProgress&&t.onProgress(0,"0%",Gt.Processing);const _={rotation:t.rotation||t.orientation,position:t.position,scale:t.scale,splatAlphaRemovalThreshold:t.splatAlphaRemovalThreshold};return this.addSplatBuffers([p],[_],m,g&&r,r,s,s).then(()=>{!s&&t.onProgress&&t.onProgress(100,"100%",Gt.Processing),f(g,m)})};return(s?this.downloadAndBuildSingleSplatSceneProgressiveLoad.bind(this):this.downloadAndBuildSingleSplatSceneStandardLoad.bind(this))(e,n,t.splatAlphaRemovalThreshold,d.bind(this),h,a.bind(this),t.headers)}downloadAndBuildSingleSplatSceneStandardLoad(e,t,n,s,r,o,a){const l=this.downloadSplatSceneToSplatBuffer(e,n,r,!1,void 0,t,a),c=mu(l.abortHandler);return l.then(u=>(this.removeSplatSceneDownloadPromise(l),s(u,!0,!0).then(()=>{c.resolve(),this.clearSplatSceneDownloadAndBuildPromise()}))).catch(u=>{o&&o(),this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(l),c.reject(this.updateError(u,`Viewer::addSplatScene -> Could not load file ${e}`))}),this.addSplatSceneDownloadPromise(l),this.setSplatSceneDownloadAndBuildPromise(c.promise),c.promise}downloadAndBuildSingleSplatSceneProgressiveLoad(e,t,n,s,r,o,a){let l=0,c=!1;const u=[],f=()=>{if(u.length>0&&!c&&!this.isDisposingOrDisposed()){c=!0;const g=u.shift();s(g.splatBuffer,g.firstBuild,g.finalBuild).then(()=>{c=!1,g.firstBuild?x.resolve():g.finalBuild&&(p.resolve(),this.clearSplatSceneDownloadAndBuildPromise()),u.length>0&&jn(()=>f())})}},h=(g,m)=>{this.isDisposingOrDisposed()||(m||u.length===0||g.getSplatCount()>u[0].splatBuffer.getSplatCount())&&(u.push({splatBuffer:g,firstBuild:l===0,finalBuild:m}),l++,f())},d=this.downloadSplatSceneToSplatBuffer(e,n,r,!0,h,t,a),x=mu(d.abortHandler),p=mu();return this.addSplatSceneDownloadPromise(d),this.setSplatSceneDownloadAndBuildPromise(p.promise),d.then(()=>{this.removeSplatSceneDownloadPromise(d)}).catch(g=>{this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(d);const m=this.updateError(g,"Viewer::addSplatScene -> Could not load one or more scenes");x.reject(m),o&&o(m)}),x.promise}addSplatScenes(e,t=!0,n=void 0){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");const s=e.length,r=[];let o;t&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=(f,h,d,x)=>{r[f]=h;let p=0;for(let g=0;g<s;g++)p+=r[g]||0;p=p/s,d=`${p.toFixed(2)}%`,t&&x===Gt.Downloading&&this.loadingSpinner.setMessageForTask(o,p==100?"Download complete!":`Downloading: ${d}`),n&&n(p,d,x)},l=[],c=[];for(let f=0;f<e.length;f++){const h=e[f],d=h.format!==void 0&&h.format!==null?h.format:im(h.path),x=this.downloadSplatSceneToSplatBuffer(h.path,h.splatAlphaRemovalThreshold,a.bind(this,f),!1,void 0,d,h.headers);l.push(x),c.push(x.promise)}const u=new Fs((f,h)=>{Promise.all(c).then(d=>{t&&this.loadingSpinner.removeTask(o),n&&n(0,"0%",Gt.Processing),this.addSplatBuffers(d,e,!0,t,t,!1,!1).then(()=>{n&&n(100,"100%",Gt.Processing),this.clearSplatSceneDownloadAndBuildPromise(),f()})}).catch(d=>{t&&this.loadingSpinner.removeTask(o),this.clearSplatSceneDownloadAndBuildPromise(),h(this.updateError(d,"Viewer::addSplatScenes -> Could not load one or more splat scenes."))}).finally(()=>{this.removeSplatSceneDownloadPromise(u)})},f=>{for(let h of l)h.abort(f)});return this.addSplatSceneDownloadPromise(u),this.setSplatSceneDownloadAndBuildPromise(u),u}downloadSplatSceneToSplatBuffer(e,t=1,n=void 0,s=!1,r=void 0,o,a){try{if(o===Ln.Splat||o===Ln.KSplat||o===Ln.Ply){const l=s?!1:this.optimizeSplatData;if(o===Ln.Splat)return kh.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,a);if(o===Ln.KSplat)return ra.loadFromURL(e,n,s,r,a);if(o===Ln.Ply)return Nh.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,this.sphericalHarmonicsDegree,a)}else if(o===Ln.Spz)return zh.loadFromURL(e,n,t,this.inMemoryCompressionLevel,this.optimizeSplatData,this.sphericalHarmonicsDegree,a)}catch(l){throw this.updateError(l,null)}throw new Error(`Viewer::downloadSplatSceneToSplatBuffer -> File format not supported: ${e}`)}static isProgressivelyLoadable(e){return e===Ln.Splat||e===Ln.KSplat||e===Ln.Ply}addSplatBuffers=(function(){return function(e,t=[],n=!0,s=!0,r=!0,o=!1,a=!1,l=!0){if(this.isDisposingOrDisposed())return Promise.resolve();let c=null;const u=()=>{c!==null&&(this.loadingSpinner.removeTask(c),c=null)};return this.splatRenderReady=!1,new Promise(f=>{s&&(c=this.loadingSpinner.addTask("Processing splats...")),jn(()=>{if(this.isDisposingOrDisposed())f();else{const h=this.addSplatBuffersToMesh(e,t,n,r,o,l),d=this.splatMesh.getMaxSplatCount();this.sortWorker&&this.sortWorker.maxSplatCount!==d&&this.disposeSortWorker(),this.gpuAcceleratedSort||this.preSortMessages.push({centers:h.centers.buffer,sceneIndexes:h.sceneIndexes.buffer,range:{from:h.from,to:h.to,count:h.count}}),(!this.sortWorker&&d>0?this.setupSortWorker(this.splatMesh):Promise.resolve()).then(()=>{this.isDisposingOrDisposed()||this.runSplatSort(!0,!0).then(p=>{!this.sortWorker||!p?(this.splatRenderReady=!0,u(),f()):(a?this.splatRenderReady=!0:this.runAfterNextSort.push(()=>{this.splatRenderReady=!0}),this.runAfterNextSort.push(()=>{u(),f()}))})})}},!0)})}})();addSplatBuffersToMesh=(function(){let e;return function(t,n,s=!0,r=!1,o=!1,a=!0){if(this.isDisposingOrDisposed())return;let l=[],c=[];o||(l=this.splatMesh.scenes.map(d=>d.splatBuffer)||[],c=this.splatMesh.sceneOptions?this.splatMesh.sceneOptions.map(d=>d):[]),l.push(...t),c.push(...n),this.renderer&&this.splatMesh.setRenderer(this.renderer);const u=d=>{if(this.isDisposingOrDisposed())return;const x=this.splatMesh.getSplatCount();r&&x>=$E&&!d&&!e&&(this.loadingSpinner.setMinimized(!0,!0),e=this.loadingSpinner.addTask("Optimizing data structures..."))},f=d=>{this.isDisposingOrDisposed()||d&&e&&(this.loadingSpinner.removeTask(e),e=null)},h=this.splatMesh.build(l,c,!0,s,u,f,a);return s&&this.freeIntermediateSplatData&&this.splatMesh.freeIntermediateSplatData(),h}})();setupSortWorker(e){if(!this.isDisposingOrDisposed())return new Promise(t=>{const n=this.integerBasedSort?Int32Array:Float32Array,s=e.getSplatCount(),r=e.getMaxSplatCount();this.sortWorker=YE(r,this.sharedMemoryForWorkers,this.enableSIMDInSort,this.integerBasedSort,this.splatMesh.dynamicMode,this.splatSortDistanceMapPrecision),this.sortWorker.onmessage=o=>{if(o.data.sortDone){if(this.sortRunning=!1,this.sharedMemoryForWorkers)this.splatMesh.updateRenderIndexes(this.sortWorkerSortedIndexes,o.data.splatRenderCount);else{const a=new Uint32Array(o.data.sortedIndexes.buffer,0,o.data.splatRenderCount);this.splatMesh.updateRenderIndexes(a,o.data.splatRenderCount)}this.lastSplatSortCount=this.splatSortCount,this.lastSortTime=o.data.sortTime,this.sortPromiseResolver(),this.sortPromiseResolver=null,this.forceRenderNextFrame(),this.runAfterNextSort.length>0&&(this.runAfterNextSort.forEach(a=>{a()}),this.runAfterNextSort.length=0)}else if(o.data.sortCanceled)this.sortRunning=!1;else if(o.data.sortSetupPhase1Complete){this.logLevel>=ho.Info&&console.log("Sorting web worker WASM setup complete."),this.sharedMemoryForWorkers?(this.sortWorkerSortedIndexes=new Uint32Array(o.data.sortedIndexesBuffer,o.data.sortedIndexesOffset,r),this.sortWorkerIndexesToSort=new Uint32Array(o.data.indexesToSortBuffer,o.data.indexesToSortOffset,r),this.sortWorkerPrecomputedDistances=new n(o.data.precomputedDistancesBuffer,o.data.precomputedDistancesOffset,r),this.sortWorkerTransforms=new Float32Array(o.data.transformsBuffer,o.data.transformsOffset,gt.MaxScenes*16)):(this.sortWorkerIndexesToSort=new Uint32Array(r),this.sortWorkerPrecomputedDistances=new n(r),this.sortWorkerTransforms=new Float32Array(gt.MaxScenes*16));for(let a=0;a<s;a++)this.sortWorkerIndexesToSort[a]=a;if(this.sortWorker.maxSplatCount=r,this.logLevel>=ho.Info){console.log("Sorting web worker ready.");const a=this.splatMesh.getSplatDataTextures(),l=a.covariances.size,c=a.centerColors.size;console.log("Covariances texture size: "+l.x+" x "+l.y),console.log("Centers/colors texture size: "+c.x+" x "+c.y)}t()}}})}updateError(e,t){return e instanceof _g?e:e instanceof Yl?new Error("File type or server does not support progressive loading."):t?new Error(t):e}disposeSortWorker(){this.sortWorker&&this.sortWorker.terminate(),this.sortWorker=null,this.sortPromise=null,this.sortPromiseResolver&&(this.sortPromiseResolver(),this.sortPromiseResolver=null),this.preSortMessages=[],this.sortRunning=!1}removeSplatScene(e,t=!0){return this.removeSplatScenes([e],t)}removeSplatScenes(e,t=!0){if(this.isLoadingOrUnloading())throw new Error("Cannot remove splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot remove splat scene after dispose() is called.");let n;return this.splatSceneRemovalPromise=new Promise((s,r)=>{let o;t&&(this.loadingSpinner.removeAllTasks(),this.loadingSpinner.show(),o=this.loadingSpinner.addTask("Removing splat scene..."));const a=()=>{t&&(this.loadingSpinner.hide(),this.loadingSpinner.removeTask(o))},l=u=>{a(),this.splatSceneRemovalPromise=null,u?r(u):s()},c=()=>this.isDisposingOrDisposed()?(l(),!0):!1;n=this.sortPromise||Promise.resolve(),n.then(()=>{if(c())return;const u=[],f=[],h=[];for(let d=0;d<this.splatMesh.scenes.length;d++){let x=!1;for(let p of e)if(p===d){x=!0;break}if(!x){const p=this.splatMesh.scenes[d];u.push(p.splatBuffer),f.push(this.splatMesh.sceneOptions[d]),h.push({position:p.position.clone(),quaternion:p.quaternion.clone(),scale:p.scale.clone()})}}this.disposeSortWorker(),this.splatMesh.dispose(),this.sceneRevealMode=la.Instant,this.createSplatMesh(),this.addSplatBuffers(u,f,!0,!1,!0).then(()=>{c()||(a(),this.splatMesh.scenes.forEach((d,x)=>{d.position.copy(h[x].position),d.quaternion.copy(h[x].quaternion),d.scale.copy(h[x].scale)}),this.splatMesh.updateTransforms(),this.splatRenderReady=!1,this.runSplatSort(!0).then(()=>{if(c()){this.splatRenderReady=!0;return}n=this.sortPromise||Promise.resolve(),n.then(()=>{this.splatRenderReady=!0,l()})}))}).catch(d=>{l(d)})})}),this.splatSceneRemovalPromise}start(){if(this.selfDrivenMode)this.webXRMode?this.renderer.setAnimationLoop(this.selfDrivenUpdateFunc):this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc),this.selfDrivenModeRunning=!0;else throw new Error("Cannot start viewer unless it is in self driven mode.")}stop(){this.selfDrivenMode&&this.selfDrivenModeRunning&&(this.webXRMode?this.renderer.setAnimationLoop(null):cancelAnimationFrame(this.requestFrameId),this.selfDrivenModeRunning=!1)}async dispose(){if(this.isDisposingOrDisposed())return this.disposePromise;let e=[],t=[];for(let n in this.splatSceneDownloadPromises)if(this.splatSceneDownloadPromises.hasOwnProperty(n)){const s=this.splatSceneDownloadPromises[n];t.push(s),e.push(s.promise)}return this.sortPromise&&e.push(this.sortPromise),this.disposing=!0,this.disposePromise=Promise.all(e).finally(()=>{this.stop(),this.orthographicControls&&(this.orthographicControls.dispose(),this.orthographicControls=null),this.perspectiveControls&&(this.perspectiveControls.dispose(),this.perspectiveControls=null),this.controls=null,this.splatMesh&&(this.splatMesh.dispose(),this.splatMesh=null),this.sceneHelper&&(this.sceneHelper.dispose(),this.sceneHelper=null),this.resizeObserver&&(this.resizeObserver.unobserve(this.rootElement),this.resizeObserver=null),this.disposeSortWorker(),this.removeEventHandlers(),this.loadingSpinner.removeAllTasks(),this.loadingSpinner.setContainer(null),this.loadingProgressBar.hide(),this.loadingProgressBar.setContainer(null),this.infoPanel.setContainer(null),this.camera=null,this.threeScene=null,this.splatRenderReady=!1,this.initialized=!1,this.renderer&&(this.usingExternalRenderer||(this.rootElement.removeChild(this.renderer.domElement),this.renderer.dispose()),this.renderer=null),this.usingExternalRenderer||document.body.removeChild(this.rootElement),this.sortWorkerSortedIndexes=null,this.sortWorkerIndexesToSort=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.disposed=!0,this.disposing=!1,this.disposePromise=null}),t.forEach(n=>{n.abort("Scene disposed")}),this.disposePromise}selfDrivenUpdate(){this.selfDrivenMode&&!this.webXRMode&&(this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc)),this.update(),this.shouldRender()?(this.render(),this.consecutiveRenderFrames++):this.consecutiveRenderFrames=0,this.renderNextFrame=!1}forceRenderNextFrame(){this.renderNextFrame=!0}shouldRender=(function(){let e=0;const t=new F,n=new Mt,s=1e-4;return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return!1;let r=!1,o=!1;if(this.camera){const a=this.camera.position,l=this.camera.quaternion;o=Math.abs(a.x-t.x)>s||Math.abs(a.y-t.y)>s||Math.abs(a.z-t.z)>s||Math.abs(l.x-n.x)>s||Math.abs(l.y-n.y)>s||Math.abs(l.z-n.z)>s||Math.abs(l.w-n.w)>s}return r=this.renderMode!==bu.Never&&(e===0||this.splatMesh.visibleRegionChanging||o||this.renderMode===bu.Always||this.dynamicMode===!0||this.renderNextFrame),this.camera&&(t.copy(this.camera.position),n.copy(this.camera.quaternion)),e++,r}})();render=(function(){return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return;const e=n=>{for(let s of n.children)if(s.visible)return!0;return!1},t=this.renderer.autoClear;e(this.threeScene)&&(this.renderer.render(this.threeScene,this.camera),this.renderer.autoClear=!1),this.renderer.render(this.splatMesh,this.camera),this.renderer.autoClear=!1,this.sceneHelper.getFocusMarkerOpacity()>0&&this.renderer.render(this.sceneHelper.focusMarker,this.camera),this.showControlPlane&&this.renderer.render(this.sceneHelper.controlPlane,this.camera),this.renderer.autoClear=t}})();update(e,t){this.dropInMode&&this.updateForDropInMode(e,t),!(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())&&(this.controls&&(this.controls.update(),this.camera.isOrthographicCamera&&!this.usingExternalCamera&&eo.setCameraPositionFromZoom(this.camera,this.camera,this.controls)),this.runSplatSort(),this.updateForRendererSizeChanges(),this.updateSplatMesh(),this.updateMeshCursor(),this.updateFPS(),this.timingSensitiveUpdates(),this.updateInfoPanel(),this.updateControlPlane())}updateForDropInMode(e,t){this.renderer=e,this.splatMesh&&this.splatMesh.setRenderer(this.renderer),this.camera=t,this.controls&&(this.controls.object=t),this.init()}updateFPS=(function(){let e=qr(),t=0;return function(){if(this.consecutiveRenderFrames>e1){const n=qr();n-e>=1?(this.currentFPS=t,t=0,e=n):t++}else this.currentFPS=null}})();updateForRendererSizeChanges=(function(){const e=new Pe,t=new Pe;let n;return function(){this.usingExternalCamera||(this.renderer.getSize(t),(n===void 0||n!==this.camera.isOrthographicCamera||t.x!==e.x||t.y!==e.y)&&(this.camera.isOrthographicCamera?(this.camera.left=-t.x/2,this.camera.right=t.x/2,this.camera.top=t.y/2,this.camera.bottom=-t.y/2):this.camera.aspect=t.x/t.y,this.camera.updateProjectionMatrix(),e.copy(t),n=this.camera.isOrthographicCamera))}})();timingSensitiveUpdates=(function(){let e;return function(){const t=qr();e||(e=t);const n=t-e;this.updateCameraTransition(t),this.updateFocusMarker(n),e=t}})();updateCameraTransition=(function(){let e=new F,t=new F,n=new F;return function(s){if(this.transitioningCameraTarget){t.copy(this.previousCameraTarget).sub(this.camera.position).normalize(),n.copy(this.nextCameraTarget).sub(this.camera.position).normalize();const r=Math.acos(t.dot(n)),a=(r/(Math.PI/3)*.65+.3)/r*(s-this.transitioningCameraTargetStartTime);e.copy(this.previousCameraTarget).lerp(this.nextCameraTarget,a),this.camera.lookAt(e),this.controls.target.copy(e),a>=1&&(this.transitioningCameraTarget=!1)}}})();updateFocusMarker=(function(){const e=new Pe;let t=!1;return function(n){if(this.getRenderDimensions(e),this.transitioningCameraTarget){this.sceneHelper.setFocusMarkerVisibility(!0);const s=Math.max(this.sceneHelper.getFocusMarkerOpacity(),0);let r=Math.min(s+ZE*n,1);this.sceneHelper.setFocusMarkerOpacity(r),this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e),t=!0,this.forceRenderNextFrame()}else{let s;if(t?s=1:s=Math.min(this.sceneHelper.getFocusMarkerOpacity(),1),s>0){this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e);let r=Math.max(s-JE*n,0);this.sceneHelper.setFocusMarkerOpacity(r),r===0&&this.sceneHelper.setFocusMarkerVisibility(!1)}s>0&&this.forceRenderNextFrame(),t=!1}}})();updateMeshCursor=(function(){const e=[],t=new Pe;return function(){this.showMeshCursor?(this.forceRenderNextFrame(),this.getRenderDimensions(t),e.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,t),this.raycaster.intersectSplatMesh(this.splatMesh,e),e.length>0?(this.sceneHelper.setMeshCursorVisibility(!0),this.sceneHelper.positionAndOrientMeshCursor(e[0].origin,this.camera)):this.sceneHelper.setMeshCursorVisibility(!1)):(this.sceneHelper.getMeschCursorVisibility()&&this.forceRenderNextFrame(),this.sceneHelper.setMeshCursorVisibility(!1))}})();updateInfoPanel=(function(){const e=new Pe;return function(){if(!this.showInfo)return;const t=this.splatMesh.getSplatCount();this.getRenderDimensions(e);const n=this.controls?this.controls.target:null,s=this.showMeshCursor?this.sceneHelper.meshCursor.position:null,r=t>0?this.splatRenderCount/t*100:0;this.infoPanel.update(e,this.camera.position,n,this.camera.up,this.camera.isOrthographicCamera,s,this.currentFPS||"N/A",t,this.splatRenderCount,r,this.lastSortTime,this.focalAdjustment,this.splatMesh.getSplatScale(),this.splatMesh.getPointCloudModeEnabled())}})();updateControlPlane(){this.showControlPlane?(this.sceneHelper.setControlPlaneVisibility(!0),this.sceneHelper.positionAndOrientControlPlane(this.controls.target,this.camera.up)):this.sceneHelper.setControlPlaneVisibility(!1)}runSplatSort=(function(){const e=new Ye,t=[],n=new F(0,0,-1),s=new F(0,0,-1),r=new F,o=new F,a=[],l=[{angleThreshold:.55,sortFractions:[.125,.33333,.75]},{angleThreshold:.65,sortFractions:[.33333,.66667]},{angleThreshold:.8,sortFractions:[.5]}];return function(c=!1,u=!1){if(!this.initialized)return Promise.resolve(!1);if(this.sortRunning)return Promise.resolve(!0);if(this.splatMesh.getSplatCount()<=0)return this.splatRenderCount=0,Promise.resolve(!1);let f=0,h=0,d=!1,x=!1;if(s.set(0,0,-1).applyQuaternion(this.camera.quaternion),f=s.dot(n),h=o.copy(this.camera.position).sub(r).length(),!c&&!this.splatMesh.dynamicMode&&a.length===0&&(f<=.99&&(d=!0),h>=1&&(x=!0),!d&&!x))return Promise.resolve(!1);this.sortRunning=!0;let{splatRenderCount:p,shouldSortAll:g}=this.gatherSceneNodesForSort();g=g||u,this.splatRenderCount=p,e.copy(this.camera.matrixWorld).invert();const m=this.perspectiveCamera||this.camera;e.premultiply(m.projectionMatrix),this.splatMesh.dynamicMode||e.multiply(this.splatMesh.matrixWorld);let _=Promise.resolve(!0);return this.gpuAcceleratedSort&&(a.length<=1||a.length%2===0)&&(_=this.splatMesh.computeDistancesOnGPU(e,this.sortWorkerPrecomputedDistances)),_.then(()=>{if(a.length===0)if(this.splatMesh.dynamicMode||g)a.push(this.splatRenderCount);else{for(let y of l)if(f<y.angleThreshold){for(let b of y.sortFractions)a.push(Math.floor(this.splatRenderCount*b));break}a.push(this.splatRenderCount)}let S=Math.min(a.shift(),this.splatRenderCount);this.splatSortCount=S,t[0]=this.camera.position.x,t[1]=this.camera.position.y,t[2]=this.camera.position.z;const A={modelViewProj:e.elements,cameraPosition:t,splatRenderCount:this.splatRenderCount,splatSortCount:S,usePrecomputedDistances:this.gpuAcceleratedSort};return this.splatMesh.dynamicMode&&this.splatMesh.fillTransformsArray(this.sortWorkerTransforms),this.sharedMemoryForWorkers||(A.indexesToSort=this.sortWorkerIndexesToSort,A.transforms=this.sortWorkerTransforms,this.gpuAcceleratedSort&&(A.precomputedDistances=this.sortWorkerPrecomputedDistances)),this.sortPromise=new Promise(y=>{this.sortPromiseResolver=y}),this.preSortMessages.length>0&&(this.preSortMessages.forEach(y=>{this.sortWorker.postMessage(y)}),this.preSortMessages=[]),this.sortWorker.postMessage({sort:A}),a.length===0&&(r.copy(this.camera.position),n.copy(s)),!0}),_}})();gatherSceneNodesForSort=(function(){const e=[];let t=null;const n=new F,s=new F,r=new F,o=new Ye,a=new Ye,l=new Ye,c=new F,u=new F(0,0,-1),f=new F,h=d=>f.copy(d.max).sub(d.min).length();return function(d=!1){this.getRenderDimensions(c);const x=c.y/2/Math.tan(this.camera.fov/2*yh.DEG2RAD),p=Math.atan(c.x/2/x),g=Math.atan(c.y/2/x),m=Math.cos(p),_=Math.cos(g),S=this.splatMesh.getSplatTree();if(S){a.copy(this.camera.matrixWorld).invert(),this.splatMesh.dynamicMode||a.multiply(this.splatMesh.matrixWorld);let A=0,y=0;for(let v=0;v<S.subTrees.length;v++){const E=S.subTrees[v];o.copy(a),this.splatMesh.dynamicMode&&(this.splatMesh.getSceneTransform(v,l),o.multiply(l));const M=E.nodesWithIndexes.length;for(let T=0;T<M;T++){const I=E.nodesWithIndexes[T];if(!I.data||!I.data.indexes||I.data.indexes.length===0)continue;r.copy(I.center).applyMatrix4(o);const P=r.length();r.normalize(),n.copy(r).setX(0).normalize(),s.copy(r).setY(0).normalize();const B=u.dot(s),N=u.dot(n),G=h(I),V=N<_-.6,q=B<m-.6;!d&&(q||V)&&P>G||(y+=I.data.indexes.length,e[A]=I,I.data.distanceToNode=P,A++)}}e.length=A,e.sort((v,E)=>v.data.distanceToNode<E.data.distanceToNode?-1:1);let b=y*gt.BytesPerInt;for(let v=0;v<A;v++){const E=e[v],M=E.data.indexes.length,T=M*gt.BytesPerInt;new Uint32Array(this.sortWorkerIndexesToSort.buffer,b-T,M).set(E.data.indexes),b-=T}return{splatRenderCount:y,shouldSortAll:!1}}else{const A=this.splatMesh.getSplatCount();if(!t||t.length!==A){t=new Uint32Array(A);for(let y=0;y<A;y++)t[y]=y}return this.sortWorkerIndexesToSort.set(t),{splatRenderCount:A,shouldSortAll:!0}}}})();getSplatMesh(){return this.splatMesh}getSplatScene(e){return this.splatMesh.getScene(e)}getSceneCount(){return this.splatMesh.getSceneCount()}isMobile(){return navigator.userAgent.includes("Mobi")}}const dm={type:"change"},Wh={type:"start"},Eg={type:"end"},Al=new mc,pm=new as,t1=Math.cos(70*yh.DEG2RAD),$t=new F,Dn=2*Math.PI,xt={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6},Mu=1e-6;class n1 extends PA{constructor(e,t=null){super(e,t),this.state=xt.NONE,this.target=new F,this.cursor=new F,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minTargetRadius=0,this.maxTargetRadius=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.keyRotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:li.ROTATE,MIDDLE:li.DOLLY,RIGHT:li.PAN},this.touches={ONE:ci.ROTATE,TWO:ci.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this._lastPosition=new F,this._lastQuaternion=new Mt,this._lastTargetPosition=new F,this._quat=new Mt().setFromUnitVectors(e.up,new F(0,1,0)),this._quatInverse=this._quat.clone().invert(),this._spherical=new ql,this._sphericalDelta=new ql,this._scale=1,this._panOffset=new F,this._rotateStart=new Pe,this._rotateEnd=new Pe,this._rotateDelta=new Pe,this._panStart=new Pe,this._panEnd=new Pe,this._panDelta=new Pe,this._dollyStart=new Pe,this._dollyEnd=new Pe,this._dollyDelta=new Pe,this._dollyDirection=new F,this._mouse=new Pe,this._performCursorZoom=!1,this._pointers=[],this._pointerPositions={},this._controlActive=!1,this._onPointerMove=s1.bind(this),this._onPointerDown=i1.bind(this),this._onPointerUp=r1.bind(this),this._onContextMenu=h1.bind(this),this._onMouseWheel=l1.bind(this),this._onKeyDown=c1.bind(this),this._onTouchStart=u1.bind(this),this._onTouchMove=f1.bind(this),this._onMouseDown=o1.bind(this),this._onMouseMove=a1.bind(this),this._interceptControlDown=d1.bind(this),this._interceptControlUp=p1.bind(this),this.domElement!==null&&this.connect(this.domElement),this.update()}connect(e){super.connect(e),this.domElement.addEventListener("pointerdown",this._onPointerDown),this.domElement.addEventListener("pointercancel",this._onPointerUp),this.domElement.addEventListener("contextmenu",this._onContextMenu),this.domElement.addEventListener("wheel",this._onMouseWheel,{passive:!1}),this.domElement.getRootNode().addEventListener("keydown",this._interceptControlDown,{passive:!0,capture:!0}),this.domElement.style.touchAction="none"}disconnect(){this.domElement.removeEventListener("pointerdown",this._onPointerDown),this.domElement.removeEventListener("pointermove",this._onPointerMove),this.domElement.removeEventListener("pointerup",this._onPointerUp),this.domElement.removeEventListener("pointercancel",this._onPointerUp),this.domElement.removeEventListener("wheel",this._onMouseWheel),this.domElement.removeEventListener("contextmenu",this._onContextMenu),this.stopListenToKeyEvents(),this.domElement.getRootNode().removeEventListener("keydown",this._interceptControlDown,{capture:!0}),this.domElement.style.touchAction="auto"}dispose(){this.disconnect()}getPolarAngle(){return this._spherical.phi}getAzimuthalAngle(){return this._spherical.theta}getDistance(){return this.object.position.distanceTo(this.target)}listenToKeyEvents(e){e.addEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=e}stopListenToKeyEvents(){this._domElementKeyEvents!==null&&(this._domElementKeyEvents.removeEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=null)}saveState(){this.target0.copy(this.target),this.position0.copy(this.object.position),this.zoom0=this.object.zoom}reset(){this.target.copy(this.target0),this.object.position.copy(this.position0),this.object.zoom=this.zoom0,this.object.updateProjectionMatrix(),this.dispatchEvent(dm),this.update(),this.state=xt.NONE}update(e=null){const t=this.object.position;$t.copy(t).sub(this.target),$t.applyQuaternion(this._quat),this._spherical.setFromVector3($t),this.autoRotate&&this.state===xt.NONE&&this._rotateLeft(this._getAutoRotationAngle(e)),this.enableDamping?(this._spherical.theta+=this._sphericalDelta.theta*this.dampingFactor,this._spherical.phi+=this._sphericalDelta.phi*this.dampingFactor):(this._spherical.theta+=this._sphericalDelta.theta,this._spherical.phi+=this._sphericalDelta.phi);let n=this.minAzimuthAngle,s=this.maxAzimuthAngle;isFinite(n)&&isFinite(s)&&(n<-Math.PI?n+=Dn:n>Math.PI&&(n-=Dn),s<-Math.PI?s+=Dn:s>Math.PI&&(s-=Dn),n<=s?this._spherical.theta=Math.max(n,Math.min(s,this._spherical.theta)):this._spherical.theta=this._spherical.theta>(n+s)/2?Math.max(n,this._spherical.theta):Math.min(s,this._spherical.theta)),this._spherical.phi=Math.max(this.minPolarAngle,Math.min(this.maxPolarAngle,this._spherical.phi)),this._spherical.makeSafe(),this.enableDamping===!0?this.target.addScaledVector(this._panOffset,this.dampingFactor):this.target.add(this._panOffset),this.target.sub(this.cursor),this.target.clampLength(this.minTargetRadius,this.maxTargetRadius),this.target.add(this.cursor);let r=!1;if(this.zoomToCursor&&this._performCursorZoom||this.object.isOrthographicCamera)this._spherical.radius=this._clampDistance(this._spherical.radius);else{const o=this._spherical.radius;this._spherical.radius=this._clampDistance(this._spherical.radius*this._scale),r=o!=this._spherical.radius}if($t.setFromSpherical(this._spherical),$t.applyQuaternion(this._quatInverse),t.copy(this.target).add($t),this.object.lookAt(this.target),this.enableDamping===!0?(this._sphericalDelta.theta*=1-this.dampingFactor,this._sphericalDelta.phi*=1-this.dampingFactor,this._panOffset.multiplyScalar(1-this.dampingFactor)):(this._sphericalDelta.set(0,0,0),this._panOffset.set(0,0,0)),this.zoomToCursor&&this._performCursorZoom){let o=null;if(this.object.isPerspectiveCamera){const a=$t.length();o=this._clampDistance(a*this._scale);const l=a-o;this.object.position.addScaledVector(this._dollyDirection,l),this.object.updateMatrixWorld(),r=!!l}else if(this.object.isOrthographicCamera){const a=new F(this._mouse.x,this._mouse.y,0);a.unproject(this.object);const l=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),this.object.updateProjectionMatrix(),r=l!==this.object.zoom;const c=new F(this._mouse.x,this._mouse.y,0);c.unproject(this.object),this.object.position.sub(c).add(a),this.object.updateMatrixWorld(),o=$t.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),this.zoomToCursor=!1;o!==null&&(this.screenSpacePanning?this.target.set(0,0,-1).transformDirection(this.object.matrix).multiplyScalar(o).add(this.object.position):(Al.origin.copy(this.object.position),Al.direction.set(0,0,-1).transformDirection(this.object.matrix),Math.abs(this.object.up.dot(Al.direction))<t1?this.object.lookAt(this.target):(pm.setFromNormalAndCoplanarPoint(this.object.up,this.target),Al.intersectPlane(pm,this.target))))}else if(this.object.isOrthographicCamera){const o=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),o!==this.object.zoom&&(this.object.updateProjectionMatrix(),r=!0)}return this._scale=1,this._performCursorZoom=!1,r||this._lastPosition.distanceToSquared(this.object.position)>Mu||8*(1-this._lastQuaternion.dot(this.object.quaternion))>Mu||this._lastTargetPosition.distanceToSquared(this.target)>Mu?(this.dispatchEvent(dm),this._lastPosition.copy(this.object.position),this._lastQuaternion.copy(this.object.quaternion),this._lastTargetPosition.copy(this.target),!0):!1}_getAutoRotationAngle(e){return e!==null?Dn/60*this.autoRotateSpeed*e:Dn/60/60*this.autoRotateSpeed}_getZoomScale(e){const t=Math.abs(e*.01);return Math.pow(.95,this.zoomSpeed*t)}_rotateLeft(e){this._sphericalDelta.theta-=e}_rotateUp(e){this._sphericalDelta.phi-=e}_panLeft(e,t){$t.setFromMatrixColumn(t,0),$t.multiplyScalar(-e),this._panOffset.add($t)}_panUp(e,t){this.screenSpacePanning===!0?$t.setFromMatrixColumn(t,1):($t.setFromMatrixColumn(t,0),$t.crossVectors(this.object.up,$t)),$t.multiplyScalar(e),this._panOffset.add($t)}_pan(e,t){const n=this.domElement;if(this.object.isPerspectiveCamera){const s=this.object.position;$t.copy(s).sub(this.target);let r=$t.length();r*=Math.tan(this.object.fov/2*Math.PI/180),this._panLeft(2*e*r/n.clientHeight,this.object.matrix),this._panUp(2*t*r/n.clientHeight,this.object.matrix)}else this.object.isOrthographicCamera?(this._panLeft(e*(this.object.right-this.object.left)/this.object.zoom/n.clientWidth,this.object.matrix),this._panUp(t*(this.object.top-this.object.bottom)/this.object.zoom/n.clientHeight,this.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),this.enablePan=!1)}_dollyOut(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale/=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_dollyIn(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale*=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_updateZoomParameters(e,t){if(!this.zoomToCursor)return;this._performCursorZoom=!0;const n=this.domElement.getBoundingClientRect(),s=e-n.left,r=t-n.top,o=n.width,a=n.height;this._mouse.x=s/o*2-1,this._mouse.y=-(r/a)*2+1,this._dollyDirection.set(this._mouse.x,this._mouse.y,1).unproject(this.object).sub(this.object.position).normalize()}_clampDistance(e){return Math.max(this.minDistance,Math.min(this.maxDistance,e))}_handleMouseDownRotate(e){this._rotateStart.set(e.clientX,e.clientY)}_handleMouseDownDolly(e){this._updateZoomParameters(e.clientX,e.clientX),this._dollyStart.set(e.clientX,e.clientY)}_handleMouseDownPan(e){this._panStart.set(e.clientX,e.clientY)}_handleMouseMoveRotate(e){this._rotateEnd.set(e.clientX,e.clientY),this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const t=this.domElement;this._rotateLeft(Dn*this._rotateDelta.x/t.clientHeight),this._rotateUp(Dn*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd),this.update()}_handleMouseMoveDolly(e){this._dollyEnd.set(e.clientX,e.clientY),this._dollyDelta.subVectors(this._dollyEnd,this._dollyStart),this._dollyDelta.y>0?this._dollyOut(this._getZoomScale(this._dollyDelta.y)):this._dollyDelta.y<0&&this._dollyIn(this._getZoomScale(this._dollyDelta.y)),this._dollyStart.copy(this._dollyEnd),this.update()}_handleMouseMovePan(e){this._panEnd.set(e.clientX,e.clientY),this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd),this.update()}_handleMouseWheel(e){this._updateZoomParameters(e.clientX,e.clientY),e.deltaY<0?this._dollyIn(this._getZoomScale(e.deltaY)):e.deltaY>0&&this._dollyOut(this._getZoomScale(e.deltaY)),this.update()}_handleKeyDown(e){let t=!1;switch(e.code){case this.keys.UP:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(Dn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,this.keyPanSpeed),t=!0;break;case this.keys.BOTTOM:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(-Dn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,-this.keyPanSpeed),t=!0;break;case this.keys.LEFT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(Dn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(this.keyPanSpeed,0),t=!0;break;case this.keys.RIGHT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(-Dn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(-this.keyPanSpeed,0),t=!0;break}t&&(e.preventDefault(),this.update())}_handleTouchStartRotate(e){if(this._pointers.length===1)this._rotateStart.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._rotateStart.set(n,s)}}_handleTouchStartPan(e){if(this._pointers.length===1)this._panStart.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._panStart.set(n,s)}}_handleTouchStartDolly(e){const t=this._getSecondPointerPosition(e),n=e.pageX-t.x,s=e.pageY-t.y,r=Math.sqrt(n*n+s*s);this._dollyStart.set(0,r)}_handleTouchStartDollyPan(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enablePan&&this._handleTouchStartPan(e)}_handleTouchStartDollyRotate(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enableRotate&&this._handleTouchStartRotate(e)}_handleTouchMoveRotate(e){if(this._pointers.length==1)this._rotateEnd.set(e.pageX,e.pageY);else{const n=this._getSecondPointerPosition(e),s=.5*(e.pageX+n.x),r=.5*(e.pageY+n.y);this._rotateEnd.set(s,r)}this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const t=this.domElement;this._rotateLeft(Dn*this._rotateDelta.x/t.clientHeight),this._rotateUp(Dn*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd)}_handleTouchMovePan(e){if(this._pointers.length===1)this._panEnd.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._panEnd.set(n,s)}this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd)}_handleTouchMoveDolly(e){const t=this._getSecondPointerPosition(e),n=e.pageX-t.x,s=e.pageY-t.y,r=Math.sqrt(n*n+s*s);this._dollyEnd.set(0,r),this._dollyDelta.set(0,Math.pow(this._dollyEnd.y/this._dollyStart.y,this.zoomSpeed)),this._dollyOut(this._dollyDelta.y),this._dollyStart.copy(this._dollyEnd);const o=(e.pageX+t.x)*.5,a=(e.pageY+t.y)*.5;this._updateZoomParameters(o,a)}_handleTouchMoveDollyPan(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enablePan&&this._handleTouchMovePan(e)}_handleTouchMoveDollyRotate(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enableRotate&&this._handleTouchMoveRotate(e)}_addPointer(e){this._pointers.push(e.pointerId)}_removePointer(e){delete this._pointerPositions[e.pointerId];for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId){this._pointers.splice(t,1);return}}_isTrackingPointer(e){for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId)return!0;return!1}_trackPointer(e){let t=this._pointerPositions[e.pointerId];t===void 0&&(t=new Pe,this._pointerPositions[e.pointerId]=t),t.set(e.pageX,e.pageY)}_getSecondPointerPosition(e){const t=e.pointerId===this._pointers[0]?this._pointers[1]:this._pointers[0];return this._pointerPositions[t]}_customWheelEvent(e){const t=e.deltaMode,n={clientX:e.clientX,clientY:e.clientY,deltaY:e.deltaY};switch(t){case 1:n.deltaY*=16;break;case 2:n.deltaY*=100;break}return e.ctrlKey&&!this._controlActive&&(n.deltaY*=10),n}}function i1(i){this.enabled!==!1&&(this._pointers.length===0&&(this.domElement.setPointerCapture(i.pointerId),this.domElement.addEventListener("pointermove",this._onPointerMove),this.domElement.addEventListener("pointerup",this._onPointerUp)),!this._isTrackingPointer(i)&&(this._addPointer(i),i.pointerType==="touch"?this._onTouchStart(i):this._onMouseDown(i)))}function s1(i){this.enabled!==!1&&(i.pointerType==="touch"?this._onTouchMove(i):this._onMouseMove(i))}function r1(i){switch(this._removePointer(i),this._pointers.length){case 0:this.domElement.releasePointerCapture(i.pointerId),this.domElement.removeEventListener("pointermove",this._onPointerMove),this.domElement.removeEventListener("pointerup",this._onPointerUp),this.dispatchEvent(Eg),this.state=xt.NONE;break;case 1:const e=this._pointers[0],t=this._pointerPositions[e];this._onTouchStart({pointerId:e,pageX:t.x,pageY:t.y});break}}function o1(i){let e;switch(i.button){case 0:e=this.mouseButtons.LEFT;break;case 1:e=this.mouseButtons.MIDDLE;break;case 2:e=this.mouseButtons.RIGHT;break;default:e=-1}switch(e){case li.DOLLY:if(this.enableZoom===!1)return;this._handleMouseDownDolly(i),this.state=xt.DOLLY;break;case li.ROTATE:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=xt.PAN}else{if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=xt.ROTATE}break;case li.PAN:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=xt.ROTATE}else{if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=xt.PAN}break;default:this.state=xt.NONE}this.state!==xt.NONE&&this.dispatchEvent(Wh)}function a1(i){switch(this.state){case xt.ROTATE:if(this.enableRotate===!1)return;this._handleMouseMoveRotate(i);break;case xt.DOLLY:if(this.enableZoom===!1)return;this._handleMouseMoveDolly(i);break;case xt.PAN:if(this.enablePan===!1)return;this._handleMouseMovePan(i);break}}function l1(i){this.enabled===!1||this.enableZoom===!1||this.state!==xt.NONE||(i.preventDefault(),this.dispatchEvent(Wh),this._handleMouseWheel(this._customWheelEvent(i)),this.dispatchEvent(Eg))}function c1(i){this.enabled!==!1&&this._handleKeyDown(i)}function u1(i){switch(this._trackPointer(i),this._pointers.length){case 1:switch(this.touches.ONE){case ci.ROTATE:if(this.enableRotate===!1)return;this._handleTouchStartRotate(i),this.state=xt.TOUCH_ROTATE;break;case ci.PAN:if(this.enablePan===!1)return;this._handleTouchStartPan(i),this.state=xt.TOUCH_PAN;break;default:this.state=xt.NONE}break;case 2:switch(this.touches.TWO){case ci.DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchStartDollyPan(i),this.state=xt.TOUCH_DOLLY_PAN;break;case ci.DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchStartDollyRotate(i),this.state=xt.TOUCH_DOLLY_ROTATE;break;default:this.state=xt.NONE}break;default:this.state=xt.NONE}this.state!==xt.NONE&&this.dispatchEvent(Wh)}function f1(i){switch(this._trackPointer(i),this.state){case xt.TOUCH_ROTATE:if(this.enableRotate===!1)return;this._handleTouchMoveRotate(i),this.update();break;case xt.TOUCH_PAN:if(this.enablePan===!1)return;this._handleTouchMovePan(i),this.update();break;case xt.TOUCH_DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchMoveDollyPan(i),this.update();break;case xt.TOUCH_DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchMoveDollyRotate(i),this.update();break;default:this.state=xt.NONE}}function h1(i){this.enabled!==!1&&i.preventDefault()}function d1(i){i.key==="Control"&&(this._controlActive=!0,this.domElement.getRootNode().addEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function p1(i){i.key==="Control"&&(this._controlActive=!1,this.domElement.getRootNode().removeEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function rs(i){if(i===void 0)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return i}function wg(i,e){i.prototype=Object.create(e.prototype),i.prototype.constructor=i,i.__proto__=e}var ei={autoSleep:120,force3D:"auto",nullTargetWarn:1,units:{lineHeight:""}},Eo={duration:.5,overwrite:!1,delay:0},Xh,on,It,mi=1e8,yt=1/mi,Uf=Math.PI*2,m1=Uf/4,g1=0,Rg=Math.sqrt,x1=Math.cos,_1=Math.sin,nn=function(e){return typeof e=="string"},Ht=function(e){return typeof e=="function"},vs=function(e){return typeof e=="number"},qh=function(e){return typeof e>"u"},Yi=function(e){return typeof e=="object"},Un=function(e){return e!==!1},Yh=function(){return typeof window<"u"},yl=function(e){return Ht(e)||nn(e)},Ig=typeof ArrayBuffer=="function"&&ArrayBuffer.isView||function(){},pn=Array.isArray,v1=/random\([^)]+\)/g,S1=/,\s*/g,mm=/(?:-?\.?\d|\.)+/gi,Dg=/[-+=.]*\d+[.e\-+]*\d*[e\-+]*\d*/g,to=/[-+=.]*\d+[.e-]*\d*[a-z%]*/g,Tu=/[-+=.]*\d+\.?\d*(?:e-|e\+)?\d*/gi,Pg=/[+-]=-?[.\d]+/,A1=/[^,'"\[\]\s]+/gi,y1=/^[+\-=e\s\d]*\d+[.\d]*([a-z]*|%)\s*$/i,Ot,Pi,Of,Qh,ti={},jl={},Fg,Lg=function(e){return(jl=wo(e,ti))&&kn},Kh=function(e,t){return console.warn("Invalid property",e,"set to",t,"Missing plugin? gsap.registerPlugin()")},Ta=function(e,t){return!t&&console.warn(e)},Bg=function(e,t){return e&&(ti[e]=t)&&jl&&(jl[e]=t)||ti},Ca=function(){return 0},b1={suppressEvents:!0,isStart:!0,kill:!1},Fl={suppressEvents:!0,kill:!1},M1={suppressEvents:!0},jh={},ks=[],Nf={},Ug,Yn={},Cu={},gm=30,Ll=[],$h="",Zh=function(e){var t=e[0],n,s;if(Yi(t)||Ht(t)||(e=[e]),!(n=(t._gsap||{}).harness)){for(s=Ll.length;s--&&!Ll[s].targetTest(t););n=Ll[s]}for(s=e.length;s--;)e[s]&&(e[s]._gsap||(e[s]._gsap=new ox(e[s],n)))||e.splice(s,1);return e},Sr=function(e){return e._gsap||Zh(gi(e))[0]._gsap},Og=function(e,t,n){return(n=e[t])&&Ht(n)?e[t]():qh(n)&&e.getAttribute&&e.getAttribute(t)||n},On=function(e,t){return(e=e.split(",")).forEach(t)||e},Vt=function(e){return Math.round(e*1e5)/1e5||0},Ut=function(e){return Math.round(e*1e7)/1e7||0},po=function(e,t){var n=t.charAt(0),s=parseFloat(t.substr(2));return e=parseFloat(e),n==="+"?e+s:n==="-"?e-s:n==="*"?e*s:e/s},T1=function(e,t){for(var n=t.length,s=0;e.indexOf(t[s])<0&&++s<n;);return s<n},$l=function(){var e=ks.length,t=ks.slice(0),n,s;for(Nf={},ks.length=0,n=0;n<e;n++)s=t[n],s&&s._lazy&&(s.render(s._lazy[0],s._lazy[1],!0)._lazy=0)},Jh=function(e){return!!(e._initted||e._startAt||e.add)},Ng=function(e,t,n,s){ks.length&&!on&&$l(),e.render(t,n,!!(on&&t<0&&Jh(e))),ks.length&&!on&&$l()},zg=function(e){var t=parseFloat(e);return(t||t===0)&&(e+"").match(A1).length<2?t:nn(e)?e.trim():e},kg=function(e){return e},ni=function(e,t){for(var n in t)n in e||(e[n]=t[n]);return e},C1=function(e){return function(t,n){for(var s in n)s in t||s==="duration"&&e||s==="ease"||(t[s]=n[s])}},wo=function(e,t){for(var n in t)e[n]=t[n];return e},xm=function i(e,t){for(var n in t)n!=="__proto__"&&n!=="constructor"&&n!=="prototype"&&(e[n]=Yi(t[n])?i(e[n]||(e[n]={}),t[n]):t[n]);return e},Zl=function(e,t){var n={},s;for(s in e)s in t||(n[s]=e[s]);return n},ca=function(e){var t=e.parent||Ot,n=e.keyframes?C1(pn(e.keyframes)):ni;if(Un(e.inherit))for(;t;)n(e,t.vars.defaults),t=t.parent||t._dp;return e},E1=function(e,t){for(var n=e.length,s=n===t.length;s&&n--&&e[n]===t[n];);return n<0},Hg=function(e,t,n,s,r){var o=e[s],a;if(r)for(a=t[r];o&&o[r]>a;)o=o._prev;return o?(t._next=o._next,o._next=t):(t._next=e[n],e[n]=t),t._next?t._next._prev=t:e[s]=t,t._prev=o,t.parent=t._dp=e,t},vc=function(e,t,n,s){n===void 0&&(n="_first"),s===void 0&&(s="_last");var r=t._prev,o=t._next;r?r._next=o:e[n]===t&&(e[n]=o),o?o._prev=r:e[s]===t&&(e[s]=r),t._next=t._prev=t.parent=null},Xs=function(e,t){e.parent&&(!t||e.parent.autoRemoveChildren)&&e.parent.remove&&e.parent.remove(e),e._act=0},Ar=function(e,t){if(e&&(!t||t._end>e._dur||t._start<0))for(var n=e;n;)n._dirty=1,n=n.parent;return e},w1=function(e){for(var t=e.parent;t&&t.parent;)t._dirty=1,t.totalDuration(),t=t.parent;return e},zf=function(e,t,n,s){return e._startAt&&(on?e._startAt.revert(Fl):e.vars.immediateRender&&!e.vars.autoRevert||e._startAt.render(t,!0,s))},R1=function i(e){return!e||e._ts&&i(e.parent)},_m=function(e){return e._repeat?Ro(e._tTime,e=e.duration()+e._rDelay)*e:0},Ro=function(e,t){var n=Math.floor(e=Ut(e/t));return e&&n===e?n-1:n},Jl=function(e,t){return(e-t._start)*t._ts+(t._ts>=0?0:t._dirty?t.totalDuration():t._tDur)},Sc=function(e){return e._end=Ut(e._start+(e._tDur/Math.abs(e._ts||e._rts||yt)||0))},Ac=function(e,t){var n=e._dp;return n&&n.smoothChildTiming&&e._ts&&(e._start=Ut(n._time-(e._ts>0?t/e._ts:((e._dirty?e.totalDuration():e._tDur)-t)/-e._ts)),Sc(e),n._dirty||Ar(n,e)),e},Vg=function(e,t){var n;if((t._time||!t._dur&&t._initted||t._start<e._time&&(t._dur||!t.add))&&(n=Jl(e.rawTime(),t),(!t._dur||Na(0,t.totalDuration(),n)-t._tTime>yt)&&t.render(n,!0)),Ar(e,t)._dp&&e._initted&&e._time>=e._dur&&e._ts){if(e._dur<e.duration())for(n=e;n._dp;)n.rawTime()>=0&&n.totalTime(n._tTime),n=n._dp;e._zTime=-yt}},Ui=function(e,t,n,s){return t.parent&&Xs(t),t._start=Ut((vs(n)?n:n||e!==Ot?oi(e,n,t):e._time)+t._delay),t._end=Ut(t._start+(t.totalDuration()/Math.abs(t.timeScale())||0)),Hg(e,t,"_first","_last",e._sort?"_start":0),kf(t)||(e._recent=t),s||Vg(e,t),e._ts<0&&Ac(e,e._tTime),e},Gg=function(e,t){return(ti.ScrollTrigger||Kh("scrollTrigger",t))&&ti.ScrollTrigger.create(t,e)},Wg=function(e,t,n,s,r){if(td(e,t,r),!e._initted)return 1;if(!n&&e._pt&&!on&&(e._dur&&e.vars.lazy!==!1||!e._dur&&e.vars.lazy)&&Ug!==Qn.frame)return ks.push(e),e._lazy=[r,s],1},I1=function i(e){var t=e.parent;return t&&t._ts&&t._initted&&!t._lock&&(t.rawTime()<0||i(t))},kf=function(e){var t=e.data;return t==="isFromStart"||t==="isStart"},D1=function(e,t,n,s){var r=e.ratio,o=t<0||!t&&(!e._start&&I1(e)&&!(!e._initted&&kf(e))||(e._ts<0||e._dp._ts<0)&&!kf(e))?0:1,a=e._rDelay,l=0,c,u,f;if(a&&e._repeat&&(l=Na(0,e._tDur,t),u=Ro(l,a),e._yoyo&&u&1&&(o=1-o),u!==Ro(e._tTime,a)&&(r=1-o,e.vars.repeatRefresh&&e._initted&&e.invalidate())),o!==r||on||s||e._zTime===yt||!t&&e._zTime){if(!e._initted&&Wg(e,t,s,n,l))return;for(f=e._zTime,e._zTime=t||(n?yt:0),n||(n=t&&!f),e.ratio=o,e._from&&(o=1-o),e._time=0,e._tTime=l,c=e._pt;c;)c.r(o,c.d),c=c._next;t<0&&zf(e,t,n,!0),e._onUpdate&&!n&&$n(e,"onUpdate"),l&&e._repeat&&!n&&e.parent&&$n(e,"onRepeat"),(t>=e._tDur||t<0)&&e.ratio===o&&(o&&Xs(e,1),!n&&!on&&($n(e,o?"onComplete":"onReverseComplete",!0),e._prom&&e._prom()))}else e._zTime||(e._zTime=t)},P1=function(e,t,n){var s;if(n>t)for(s=e._first;s&&s._start<=n;){if(s.data==="isPause"&&s._start>t)return s;s=s._next}else for(s=e._last;s&&s._start>=n;){if(s.data==="isPause"&&s._start<t)return s;s=s._prev}},Io=function(e,t,n,s){var r=e._repeat,o=Ut(t)||0,a=e._tTime/e._tDur;return a&&!s&&(e._time*=o/e._dur),e._dur=o,e._tDur=r?r<0?1e10:Ut(o*(r+1)+e._rDelay*r):o,a>0&&!s&&Ac(e,e._tTime=e._tDur*a),e.parent&&Sc(e),n||Ar(e.parent,e),e},vm=function(e){return e instanceof bn?Ar(e):Io(e,e._dur)},F1={_start:0,endTime:Ca,totalDuration:Ca},oi=function i(e,t,n){var s=e.labels,r=e._recent||F1,o=e.duration()>=mi?r.endTime(!1):e._dur,a,l,c;return nn(t)&&(isNaN(t)||t in s)?(l=t.charAt(0),c=t.substr(-1)==="%",a=t.indexOf("="),l==="<"||l===">"?(a>=0&&(t=t.replace(/=/,"")),(l==="<"?r._start:r.endTime(r._repeat>=0))+(parseFloat(t.substr(1))||0)*(c?(a<0?r:n).totalDuration()/100:1)):a<0?(t in s||(s[t]=o),s[t]):(l=parseFloat(t.charAt(a-1)+t.substr(a+1)),c&&n&&(l=l/100*(pn(n)?n[0]:n).totalDuration()),a>1?i(e,t.substr(0,a-1),n)+l:o+l)):t==null?o:+t},ua=function(e,t,n){var s=vs(t[1]),r=(s?2:1)+(e<2?0:1),o=t[r],a,l;if(s&&(o.duration=t[1]),o.parent=n,e){for(a=o,l=n;l&&!("immediateRender"in a);)a=l.vars.defaults||{},l=Un(l.vars.inherit)&&l.parent;o.immediateRender=Un(a.immediateRender),e<2?o.runBackwards=1:o.startAt=t[r-1]}return new Qt(t[0],o,t[r+1])},js=function(e,t){return e||e===0?t(e):t},Na=function(e,t,n){return n<e?e:n>t?t:n},fn=function(e,t){return!nn(e)||!(t=y1.exec(e))?"":t[1]},L1=function(e,t,n){return js(n,function(s){return Na(e,t,s)})},Hf=[].slice,Xg=function(e,t){return e&&Yi(e)&&"length"in e&&(!t&&!e.length||e.length-1 in e&&Yi(e[0]))&&!e.nodeType&&e!==Pi},B1=function(e,t,n){return n===void 0&&(n=[]),e.forEach(function(s){var r;return nn(s)&&!t||Xg(s,1)?(r=n).push.apply(r,gi(s)):n.push(s)})||n},gi=function(e,t,n){return It&&!t&&It.selector?It.selector(e):nn(e)&&!n&&(Of||!Do())?Hf.call((t||Qh).querySelectorAll(e),0):pn(e)?B1(e,n):Xg(e)?Hf.call(e,0):e?[e]:[]},Vf=function(e){return e=gi(e)[0]||Ta("Invalid scope")||{},function(t){var n=e.current||e.nativeElement||e;return gi(t,n.querySelectorAll?n:n===e?Ta("Invalid scope")||Qh.createElement("div"):e)}},qg=function(e){return e.sort(function(){return .5-Math.random()})},Yg=function(e){if(Ht(e))return e;var t=Yi(e)?e:{each:e},n=yr(t.ease),s=t.from||0,r=parseFloat(t.base)||0,o={},a=s>0&&s<1,l=isNaN(s)||a,c=t.axis,u=s,f=s;return nn(s)?u=f={center:.5,edges:.5,end:1}[s]||0:!a&&l&&(u=s[0],f=s[1]),function(h,d,x){var p=(x||t).length,g=o[p],m,_,S,A,y,b,v,E,M;if(!g){if(M=t.grid==="auto"?0:(t.grid||[1,mi])[1],!M){for(v=-mi;v<(v=x[M++].getBoundingClientRect().left)&&M<p;);M<p&&M--}for(g=o[p]=[],m=l?Math.min(M,p)*u-.5:s%M,_=M===mi?0:l?p*f/M-.5:s/M|0,v=0,E=mi,b=0;b<p;b++)S=b%M-m,A=_-(b/M|0),g[b]=y=c?Math.abs(c==="y"?A:S):Rg(S*S+A*A),y>v&&(v=y),y<E&&(E=y);s==="random"&&qg(g),g.max=v-E,g.min=E,g.v=p=(parseFloat(t.amount)||parseFloat(t.each)*(M>p?p-1:c?c==="y"?p/M:M:Math.max(M,p/M))||0)*(s==="edges"?-1:1),g.b=p<0?r-p:r,g.u=fn(t.amount||t.each)||0,n=n&&p<0?ix(n):n}return p=(g[h]-g.min)/g.max||0,Ut(g.b+(n?n(p):p)*g.v)+g.u}},Gf=function(e){var t=Math.pow(10,((e+"").split(".")[1]||"").length);return function(n){var s=Ut(Math.round(parseFloat(n)/e)*e*t);return(s-s%1)/t+(vs(n)?0:fn(n))}},Qg=function(e,t){var n=pn(e),s,r;return!n&&Yi(e)&&(s=n=e.radius||mi,e.values?(e=gi(e.values),(r=!vs(e[0]))&&(s*=s)):e=Gf(e.increment)),js(t,n?Ht(e)?function(o){return r=e(o),Math.abs(r-o)<=s?r:o}:function(o){for(var a=parseFloat(r?o.x:o),l=parseFloat(r?o.y:0),c=mi,u=0,f=e.length,h,d;f--;)r?(h=e[f].x-a,d=e[f].y-l,h=h*h+d*d):h=Math.abs(e[f]-a),h<c&&(c=h,u=f);return u=!s||c<=s?e[u]:o,r||u===o||vs(o)?u:u+fn(o)}:Gf(e))},Kg=function(e,t,n,s){return js(pn(e)?!t:n===!0?!!(n=0):!s,function(){return pn(e)?e[~~(Math.random()*e.length)]:(n=n||1e-5)&&(s=n<1?Math.pow(10,(n+"").length-2):1)&&Math.floor(Math.round((e-n/2+Math.random()*(t-e+n*.99))/n)*n*s)/s})},U1=function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(s){return t.reduce(function(r,o){return o(r)},s)}},O1=function(e,t){return function(n){return e(parseFloat(n))+(t||fn(n))}},N1=function(e,t,n){return $g(e,t,0,1,n)},jg=function(e,t,n){return js(n,function(s){return e[~~t(s)]})},z1=function i(e,t,n){var s=t-e;return pn(e)?jg(e,i(0,e.length),t):js(n,function(r){return(s+(r-e)%s)%s+e})},k1=function i(e,t,n){var s=t-e,r=s*2;return pn(e)?jg(e,i(0,e.length-1),t):js(n,function(o){return o=(r+(o-e)%r)%r||0,e+(o>s?r-o:o)})},Ea=function(e){return e.replace(v1,function(t){var n=t.indexOf("[")+1,s=t.substring(n||7,n?t.indexOf("]"):t.length-1).split(S1);return Kg(n?s:+s[0],n?0:+s[1],+s[2]||1e-5)})},$g=function(e,t,n,s,r){var o=t-e,a=s-n;return js(r,function(l){return n+((l-e)/o*a||0)})},H1=function i(e,t,n,s){var r=isNaN(e+t)?0:function(d){return(1-d)*e+d*t};if(!r){var o=nn(e),a={},l,c,u,f,h;if(n===!0&&(s=1)&&(n=null),o)e={p:e},t={p:t};else if(pn(e)&&!pn(t)){for(u=[],f=e.length,h=f-2,c=1;c<f;c++)u.push(i(e[c-1],e[c]));f--,r=function(x){x*=f;var p=Math.min(h,~~x);return u[p](x-p)},n=t}else s||(e=wo(pn(e)?[]:{},e));if(!u){for(l in t)ed.call(a,e,l,"get",t[l]);r=function(x){return sd(x,a)||(o?e.p:e)}}}return js(n,r)},Sm=function(e,t,n){var s=e.labels,r=mi,o,a,l;for(o in s)a=s[o]-t,a<0==!!n&&a&&r>(a=Math.abs(a))&&(l=o,r=a);return l},$n=function(e,t,n){var s=e.vars,r=s[t],o=It,a=e._ctx,l,c,u;if(r)return l=s[t+"Params"],c=s.callbackScope||e,n&&ks.length&&$l(),a&&(It=a),u=l?r.apply(c,l):r.call(c),It=o,u},Ko=function(e){return Xs(e),e.scrollTrigger&&e.scrollTrigger.kill(!!on),e.progress()<1&&$n(e,"onInterrupt"),e},no,Zg=[],Jg=function(e){if(e)if(e=!e.name&&e.default||e,Yh()||e.headless){var t=e.name,n=Ht(e),s=t&&!n&&e.init?function(){this._props=[]}:e,r={init:Ca,render:sd,add:ed,kill:iw,modifier:nw,rawVars:0},o={targetTest:0,get:0,getSetter:id,aliases:{},register:0};if(Do(),e!==s){if(Yn[t])return;ni(s,ni(Zl(e,r),o)),wo(s.prototype,wo(r,Zl(e,o))),Yn[s.prop=t]=s,e.targetTest&&(Ll.push(s),jh[t]=1),t=(t==="css"?"CSS":t.charAt(0).toUpperCase()+t.substr(1))+"Plugin"}Bg(t,s),e.register&&e.register(kn,s,Nn)}else Zg.push(e)},At=255,jo={aqua:[0,At,At],lime:[0,At,0],silver:[192,192,192],black:[0,0,0],maroon:[128,0,0],teal:[0,128,128],blue:[0,0,At],navy:[0,0,128],white:[At,At,At],olive:[128,128,0],yellow:[At,At,0],orange:[At,165,0],gray:[128,128,128],purple:[128,0,128],green:[0,128,0],red:[At,0,0],pink:[At,192,203],cyan:[0,At,At],transparent:[At,At,At,0]},Eu=function(e,t,n){return e+=e<0?1:e>1?-1:0,(e*6<1?t+(n-t)*e*6:e<.5?n:e*3<2?t+(n-t)*(2/3-e)*6:t)*At+.5|0},ex=function(e,t,n){var s=e?vs(e)?[e>>16,e>>8&At,e&At]:0:jo.black,r,o,a,l,c,u,f,h,d,x;if(!s){if(e.substr(-1)===","&&(e=e.substr(0,e.length-1)),jo[e])s=jo[e];else if(e.charAt(0)==="#"){if(e.length<6&&(r=e.charAt(1),o=e.charAt(2),a=e.charAt(3),e="#"+r+r+o+o+a+a+(e.length===5?e.charAt(4)+e.charAt(4):"")),e.length===9)return s=parseInt(e.substr(1,6),16),[s>>16,s>>8&At,s&At,parseInt(e.substr(7),16)/255];e=parseInt(e.substr(1),16),s=[e>>16,e>>8&At,e&At]}else if(e.substr(0,3)==="hsl"){if(s=x=e.match(mm),!t)l=+s[0]%360/360,c=+s[1]/100,u=+s[2]/100,o=u<=.5?u*(c+1):u+c-u*c,r=u*2-o,s.length>3&&(s[3]*=1),s[0]=Eu(l+1/3,r,o),s[1]=Eu(l,r,o),s[2]=Eu(l-1/3,r,o);else if(~e.indexOf("="))return s=e.match(Dg),n&&s.length<4&&(s[3]=1),s}else s=e.match(mm)||jo.transparent;s=s.map(Number)}return t&&!x&&(r=s[0]/At,o=s[1]/At,a=s[2]/At,f=Math.max(r,o,a),h=Math.min(r,o,a),u=(f+h)/2,f===h?l=c=0:(d=f-h,c=u>.5?d/(2-f-h):d/(f+h),l=f===r?(o-a)/d+(o<a?6:0):f===o?(a-r)/d+2:(r-o)/d+4,l*=60),s[0]=~~(l+.5),s[1]=~~(c*100+.5),s[2]=~~(u*100+.5)),n&&s.length<4&&(s[3]=1),s},tx=function(e){var t=[],n=[],s=-1;return e.split(Hs).forEach(function(r){var o=r.match(to)||[];t.push.apply(t,o),n.push(s+=o.length+1)}),t.c=n,t},Am=function(e,t,n){var s="",r=(e+s).match(Hs),o=t?"hsla(":"rgba(",a=0,l,c,u,f;if(!r)return e;if(r=r.map(function(h){return(h=ex(h,t,1))&&o+(t?h[0]+","+h[1]+"%,"+h[2]+"%,"+h[3]:h.join(","))+")"}),n&&(u=tx(e),l=n.c,l.join(s)!==u.c.join(s)))for(c=e.replace(Hs,"1").split(to),f=c.length-1;a<f;a++)s+=c[a]+(~l.indexOf(a)?r.shift()||o+"0,0,0,0)":(u.length?u:r.length?r:n).shift());if(!c)for(c=e.split(Hs),f=c.length-1;a<f;a++)s+=c[a]+r[a];return s+c[f]},Hs=(function(){var i="(?:\\b(?:(?:rgb|rgba|hsl|hsla)\\(.+?\\))|\\B#(?:[0-9a-f]{3,4}){1,2}\\b",e;for(e in jo)i+="|"+e+"\\b";return new RegExp(i+")","gi")})(),V1=/hsl[a]?\(/,nx=function(e){var t=e.join(" "),n;if(Hs.lastIndex=0,Hs.test(t))return n=V1.test(t),e[1]=Am(e[1],n),e[0]=Am(e[0],n,tx(e[1])),!0},wa,Qn=(function(){var i=Date.now,e=500,t=33,n=i(),s=n,r=1e3/240,o=r,a=[],l,c,u,f,h,d,x=function p(g){var m=i()-s,_=g===!0,S,A,y,b;if((m>e||m<0)&&(n+=m-t),s+=m,y=s-n,S=y-o,(S>0||_)&&(b=++f.frame,h=y-f.time*1e3,f.time=y=y/1e3,o+=S+(S>=r?4:r-S),A=1),_||(l=c(p)),A)for(d=0;d<a.length;d++)a[d](y,h,b,g)};return f={time:0,frame:0,tick:function(){x(!0)},deltaRatio:function(g){return h/(1e3/(g||60))},wake:function(){Fg&&(!Of&&Yh()&&(Pi=Of=window,Qh=Pi.document||{},ti.gsap=kn,(Pi.gsapVersions||(Pi.gsapVersions=[])).push(kn.version),Lg(jl||Pi.GreenSockGlobals||!Pi.gsap&&Pi||{}),Zg.forEach(Jg)),u=typeof requestAnimationFrame<"u"&&requestAnimationFrame,l&&f.sleep(),c=u||function(g){return setTimeout(g,o-f.time*1e3+1|0)},wa=1,x(2))},sleep:function(){(u?cancelAnimationFrame:clearTimeout)(l),wa=0,c=Ca},lagSmoothing:function(g,m){e=g||1/0,t=Math.min(m||33,e)},fps:function(g){r=1e3/(g||240),o=f.time*1e3+r},add:function(g,m,_){var S=m?function(A,y,b,v){g(A,y,b,v),f.remove(S)}:g;return f.remove(g),a[_?"unshift":"push"](S),Do(),S},remove:function(g,m){~(m=a.indexOf(g))&&a.splice(m,1)&&d>=m&&d--},_listeners:a},f})(),Do=function(){return!wa&&Qn.wake()},st={},G1=/^[\d.\-M][\d.\-,\s]/,W1=/["']/g,X1=function(e){for(var t={},n=e.substr(1,e.length-3).split(":"),s=n[0],r=1,o=n.length,a,l,c;r<o;r++)l=n[r],a=r!==o-1?l.lastIndexOf(","):l.length,c=l.substr(0,a),t[s]=isNaN(c)?c.replace(W1,"").trim():+c,s=l.substr(a+1).trim();return t},q1=function(e){var t=e.indexOf("(")+1,n=e.indexOf(")"),s=e.indexOf("(",t);return e.substring(t,~s&&s<n?e.indexOf(")",n+1):n)},Y1=function(e){var t=(e+"").split("("),n=st[t[0]];return n&&t.length>1&&n.config?n.config.apply(null,~e.indexOf("{")?[X1(t[1])]:q1(e).split(",").map(zg)):st._CE&&G1.test(e)?st._CE("",e):n},ix=function(e){return function(t){return 1-e(1-t)}},sx=function i(e,t){for(var n=e._first,s;n;)n instanceof bn?i(n,t):n.vars.yoyoEase&&(!n._yoyo||!n._repeat)&&n._yoyo!==t&&(n.timeline?i(n.timeline,t):(s=n._ease,n._ease=n._yEase,n._yEase=s,n._yoyo=t)),n=n._next},yr=function(e,t){return e&&(Ht(e)?e:st[e]||Y1(e))||t},Cr=function(e,t,n,s){n===void 0&&(n=function(l){return 1-t(1-l)}),s===void 0&&(s=function(l){return l<.5?t(l*2)/2:1-t((1-l)*2)/2});var r={easeIn:t,easeOut:n,easeInOut:s},o;return On(e,function(a){st[a]=ti[a]=r,st[o=a.toLowerCase()]=n;for(var l in r)st[o+(l==="easeIn"?".in":l==="easeOut"?".out":".inOut")]=st[a+"."+l]=r[l]}),r},rx=function(e){return function(t){return t<.5?(1-e(1-t*2))/2:.5+e((t-.5)*2)/2}},wu=function i(e,t,n){var s=t>=1?t:1,r=(n||(e?.3:.45))/(t<1?t:1),o=r/Uf*(Math.asin(1/s)||0),a=function(u){return u===1?1:s*Math.pow(2,-10*u)*_1((u-o)*r)+1},l=e==="out"?a:e==="in"?function(c){return 1-a(1-c)}:rx(a);return r=Uf/r,l.config=function(c,u){return i(e,c,u)},l},Ru=function i(e,t){t===void 0&&(t=1.70158);var n=function(o){return o?--o*o*((t+1)*o+t)+1:0},s=e==="out"?n:e==="in"?function(r){return 1-n(1-r)}:rx(n);return s.config=function(r){return i(e,r)},s};On("Linear,Quad,Cubic,Quart,Quint,Strong",function(i,e){var t=e<5?e+1:e;Cr(i+",Power"+(t-1),e?function(n){return Math.pow(n,t)}:function(n){return n},function(n){return 1-Math.pow(1-n,t)},function(n){return n<.5?Math.pow(n*2,t)/2:1-Math.pow((1-n)*2,t)/2})});st.Linear.easeNone=st.none=st.Linear.easeIn;Cr("Elastic",wu("in"),wu("out"),wu());(function(i,e){var t=1/e,n=2*t,s=2.5*t,r=function(a){return a<t?i*a*a:a<n?i*Math.pow(a-1.5/e,2)+.75:a<s?i*(a-=2.25/e)*a+.9375:i*Math.pow(a-2.625/e,2)+.984375};Cr("Bounce",function(o){return 1-r(1-o)},r)})(7.5625,2.75);Cr("Expo",function(i){return Math.pow(2,10*(i-1))*i+i*i*i*i*i*i*(1-i)});Cr("Circ",function(i){return-(Rg(1-i*i)-1)});Cr("Sine",function(i){return i===1?1:-x1(i*m1)+1});Cr("Back",Ru("in"),Ru("out"),Ru());st.SteppedEase=st.steps=ti.SteppedEase={config:function(e,t){e===void 0&&(e=1);var n=1/e,s=e+(t?0:1),r=t?1:0,o=1-yt;return function(a){return((s*Na(0,o,a)|0)+r)*n}}};Eo.ease=st["quad.out"];On("onComplete,onUpdate,onStart,onRepeat,onReverseComplete,onInterrupt",function(i){return $h+=i+","+i+"Params,"});var ox=function(e,t){this.id=g1++,e._gsap=this,this.target=e,this.harness=t,this.get=t?t.get:Og,this.set=t?t.getSetter:id},Ra=(function(){function i(t){this.vars=t,this._delay=+t.delay||0,(this._repeat=t.repeat===1/0?-2:t.repeat||0)&&(this._rDelay=t.repeatDelay||0,this._yoyo=!!t.yoyo||!!t.yoyoEase),this._ts=1,Io(this,+t.duration,1,1),this.data=t.data,It&&(this._ctx=It,It.data.push(this)),wa||Qn.wake()}var e=i.prototype;return e.delay=function(n){return n||n===0?(this.parent&&this.parent.smoothChildTiming&&this.startTime(this._start+n-this._delay),this._delay=n,this):this._delay},e.duration=function(n){return arguments.length?this.totalDuration(this._repeat>0?n+(n+this._rDelay)*this._repeat:n):this.totalDuration()&&this._dur},e.totalDuration=function(n){return arguments.length?(this._dirty=0,Io(this,this._repeat<0?n:(n-this._repeat*this._rDelay)/(this._repeat+1))):this._tDur},e.totalTime=function(n,s){if(Do(),!arguments.length)return this._tTime;var r=this._dp;if(r&&r.smoothChildTiming&&this._ts){for(Ac(this,n),!r._dp||r.parent||Vg(r,this);r&&r.parent;)r.parent._time!==r._start+(r._ts>=0?r._tTime/r._ts:(r.totalDuration()-r._tTime)/-r._ts)&&r.totalTime(r._tTime,!0),r=r.parent;!this.parent&&this._dp.autoRemoveChildren&&(this._ts>0&&n<this._tDur||this._ts<0&&n>0||!this._tDur&&!n)&&Ui(this._dp,this,this._start-this._delay)}return(this._tTime!==n||!this._dur&&!s||this._initted&&Math.abs(this._zTime)===yt||!this._initted&&this._dur&&n||!n&&!this._initted&&(this.add||this._ptLookup))&&(this._ts||(this._pTime=n),Ng(this,n,s)),this},e.time=function(n,s){return arguments.length?this.totalTime(Math.min(this.totalDuration(),n+_m(this))%(this._dur+this._rDelay)||(n?this._dur:0),s):this._time},e.totalProgress=function(n,s){return arguments.length?this.totalTime(this.totalDuration()*n,s):this.totalDuration()?Math.min(1,this._tTime/this._tDur):this.rawTime()>=0&&this._initted?1:0},e.progress=function(n,s){return arguments.length?this.totalTime(this.duration()*(this._yoyo&&!(this.iteration()&1)?1-n:n)+_m(this),s):this.duration()?Math.min(1,this._time/this._dur):this.rawTime()>0?1:0},e.iteration=function(n,s){var r=this.duration()+this._rDelay;return arguments.length?this.totalTime(this._time+(n-1)*r,s):this._repeat?Ro(this._tTime,r)+1:1},e.timeScale=function(n,s){if(!arguments.length)return this._rts===-yt?0:this._rts;if(this._rts===n)return this;var r=this.parent&&this._ts?Jl(this.parent._time,this):this._tTime;return this._rts=+n||0,this._ts=this._ps||n===-yt?0:this._rts,this.totalTime(Na(-Math.abs(this._delay),this.totalDuration(),r),s!==!1),Sc(this),w1(this)},e.paused=function(n){return arguments.length?(this._ps!==n&&(this._ps=n,n?(this._pTime=this._tTime||Math.max(-this._delay,this.rawTime()),this._ts=this._act=0):(Do(),this._ts=this._rts,this.totalTime(this.parent&&!this.parent.smoothChildTiming?this.rawTime():this._tTime||this._pTime,this.progress()===1&&Math.abs(this._zTime)!==yt&&(this._tTime-=yt)))),this):this._ps},e.startTime=function(n){if(arguments.length){this._start=Ut(n);var s=this.parent||this._dp;return s&&(s._sort||!this.parent)&&Ui(s,this,this._start-this._delay),this}return this._start},e.endTime=function(n){return this._start+(Un(n)?this.totalDuration():this.duration())/Math.abs(this._ts||1)},e.rawTime=function(n){var s=this.parent||this._dp;return s?n&&(!this._ts||this._repeat&&this._time&&this.totalProgress()<1)?this._tTime%(this._dur+this._rDelay):this._ts?Jl(s.rawTime(n),this):this._tTime:this._tTime},e.revert=function(n){n===void 0&&(n=M1);var s=on;return on=n,Jh(this)&&(this.timeline&&this.timeline.revert(n),this.totalTime(-.01,n.suppressEvents)),this.data!=="nested"&&n.kill!==!1&&this.kill(),on=s,this},e.globalTime=function(n){for(var s=this,r=arguments.length?n:s.rawTime();s;)r=s._start+r/(Math.abs(s._ts)||1),s=s._dp;return!this.parent&&this._sat?this._sat.globalTime(n):r},e.repeat=function(n){return arguments.length?(this._repeat=n===1/0?-2:n,vm(this)):this._repeat===-2?1/0:this._repeat},e.repeatDelay=function(n){if(arguments.length){var s=this._time;return this._rDelay=n,vm(this),s?this.time(s):this}return this._rDelay},e.yoyo=function(n){return arguments.length?(this._yoyo=n,this):this._yoyo},e.seek=function(n,s){return this.totalTime(oi(this,n),Un(s))},e.restart=function(n,s){return this.play().totalTime(n?-this._delay:0,Un(s)),this._dur||(this._zTime=-yt),this},e.play=function(n,s){return n!=null&&this.seek(n,s),this.reversed(!1).paused(!1)},e.reverse=function(n,s){return n!=null&&this.seek(n||this.totalDuration(),s),this.reversed(!0).paused(!1)},e.pause=function(n,s){return n!=null&&this.seek(n,s),this.paused(!0)},e.resume=function(){return this.paused(!1)},e.reversed=function(n){return arguments.length?(!!n!==this.reversed()&&this.timeScale(-this._rts||(n?-yt:0)),this):this._rts<0},e.invalidate=function(){return this._initted=this._act=0,this._zTime=-yt,this},e.isActive=function(){var n=this.parent||this._dp,s=this._start,r;return!!(!n||this._ts&&this._initted&&n.isActive()&&(r=n.rawTime(!0))>=s&&r<this.endTime(!0)-yt)},e.eventCallback=function(n,s,r){var o=this.vars;return arguments.length>1?(s?(o[n]=s,r&&(o[n+"Params"]=r),n==="onUpdate"&&(this._onUpdate=s)):delete o[n],this):o[n]},e.then=function(n){var s=this,r=s._prom;return new Promise(function(o){var a=Ht(n)?n:kg,l=function(){var u=s.then;s.then=null,r&&r(),Ht(a)&&(a=a(s))&&(a.then||a===s)&&(s.then=u),o(a),s.then=u};s._initted&&s.totalProgress()===1&&s._ts>=0||!s._tTime&&s._ts<0?l():s._prom=l})},e.kill=function(){Ko(this)},i})();ni(Ra.prototype,{_time:0,_start:0,_end:0,_tTime:0,_tDur:0,_dirty:0,_repeat:0,_yoyo:!1,parent:null,_initted:!1,_rDelay:0,_ts:1,_dp:0,ratio:0,_zTime:-yt,_prom:0,_ps:!1,_rts:1});var bn=(function(i){wg(e,i);function e(n,s){var r;return n===void 0&&(n={}),r=i.call(this,n)||this,r.labels={},r.smoothChildTiming=!!n.smoothChildTiming,r.autoRemoveChildren=!!n.autoRemoveChildren,r._sort=Un(n.sortChildren),Ot&&Ui(n.parent||Ot,rs(r),s),n.reversed&&r.reverse(),n.paused&&r.paused(!0),n.scrollTrigger&&Gg(rs(r),n.scrollTrigger),r}var t=e.prototype;return t.to=function(s,r,o){return ua(0,arguments,this),this},t.from=function(s,r,o){return ua(1,arguments,this),this},t.fromTo=function(s,r,o,a){return ua(2,arguments,this),this},t.set=function(s,r,o){return r.duration=0,r.parent=this,ca(r).repeatDelay||(r.repeat=0),r.immediateRender=!!r.immediateRender,new Qt(s,r,oi(this,o),1),this},t.call=function(s,r,o){return Ui(this,Qt.delayedCall(0,s,r),o)},t.staggerTo=function(s,r,o,a,l,c,u){return o.duration=r,o.stagger=o.stagger||a,o.onComplete=c,o.onCompleteParams=u,o.parent=this,new Qt(s,o,oi(this,l)),this},t.staggerFrom=function(s,r,o,a,l,c,u){return o.runBackwards=1,ca(o).immediateRender=Un(o.immediateRender),this.staggerTo(s,r,o,a,l,c,u)},t.staggerFromTo=function(s,r,o,a,l,c,u,f){return a.startAt=o,ca(a).immediateRender=Un(a.immediateRender),this.staggerTo(s,r,a,l,c,u,f)},t.render=function(s,r,o){var a=this._time,l=this._dirty?this.totalDuration():this._tDur,c=this._dur,u=s<=0?0:Ut(s),f=this._zTime<0!=s<0&&(this._initted||!c),h,d,x,p,g,m,_,S,A,y,b,v;if(this!==Ot&&u>l&&s>=0&&(u=l),u!==this._tTime||o||f){if(a!==this._time&&c&&(u+=this._time-a,s+=this._time-a),h=u,A=this._start,S=this._ts,m=!S,f&&(c||(a=this._zTime),(s||!r)&&(this._zTime=s)),this._repeat){if(b=this._yoyo,g=c+this._rDelay,this._repeat<-1&&s<0)return this.totalTime(g*100+s,r,o);if(h=Ut(u%g),u===l?(p=this._repeat,h=c):(y=Ut(u/g),p=~~y,p&&p===y&&(h=c,p--),h>c&&(h=c)),y=Ro(this._tTime,g),!a&&this._tTime&&y!==p&&this._tTime-y*g-this._dur<=0&&(y=p),b&&p&1&&(h=c-h,v=1),p!==y&&!this._lock){var E=b&&y&1,M=E===(b&&p&1);if(p<y&&(E=!E),a=E?0:u%c?c:u,this._lock=1,this.render(a||(v?0:Ut(p*g)),r,!c)._lock=0,this._tTime=u,!r&&this.parent&&$n(this,"onRepeat"),this.vars.repeatRefresh&&!v&&(this.invalidate()._lock=1,y=p),a&&a!==this._time||m!==!this._ts||this.vars.onRepeat&&!this.parent&&!this._act)return this;if(c=this._dur,l=this._tDur,M&&(this._lock=2,a=E?c:-1e-4,this.render(a,!0),this.vars.repeatRefresh&&!v&&this.invalidate()),this._lock=0,!this._ts&&!m)return this;sx(this,v)}}if(this._hasPause&&!this._forcing&&this._lock<2&&(_=P1(this,Ut(a),Ut(h)),_&&(u-=h-(h=_._start))),this._tTime=u,this._time=h,this._act=!S,this._initted||(this._onUpdate=this.vars.onUpdate,this._initted=1,this._zTime=s,a=0),!a&&u&&c&&!r&&!y&&($n(this,"onStart"),this._tTime!==u))return this;if(h>=a&&s>=0)for(d=this._first;d;){if(x=d._next,(d._act||h>=d._start)&&d._ts&&_!==d){if(d.parent!==this)return this.render(s,r,o);if(d.render(d._ts>0?(h-d._start)*d._ts:(d._dirty?d.totalDuration():d._tDur)+(h-d._start)*d._ts,r,o),h!==this._time||!this._ts&&!m){_=0,x&&(u+=this._zTime=-yt);break}}d=x}else{d=this._last;for(var T=s<0?s:h;d;){if(x=d._prev,(d._act||T<=d._end)&&d._ts&&_!==d){if(d.parent!==this)return this.render(s,r,o);if(d.render(d._ts>0?(T-d._start)*d._ts:(d._dirty?d.totalDuration():d._tDur)+(T-d._start)*d._ts,r,o||on&&Jh(d)),h!==this._time||!this._ts&&!m){_=0,x&&(u+=this._zTime=T?-yt:yt);break}}d=x}}if(_&&!r&&(this.pause(),_.render(h>=a?0:-yt)._zTime=h>=a?1:-1,this._ts))return this._start=A,Sc(this),this.render(s,r,o);this._onUpdate&&!r&&$n(this,"onUpdate",!0),(u===l&&this._tTime>=this.totalDuration()||!u&&a)&&(A===this._start||Math.abs(S)!==Math.abs(this._ts))&&(this._lock||((s||!c)&&(u===l&&this._ts>0||!u&&this._ts<0)&&Xs(this,1),!r&&!(s<0&&!a)&&(u||a||!l)&&($n(this,u===l&&s>=0?"onComplete":"onReverseComplete",!0),this._prom&&!(u<l&&this.timeScale()>0)&&this._prom())))}return this},t.add=function(s,r){var o=this;if(vs(r)||(r=oi(this,r,s)),!(s instanceof Ra)){if(pn(s))return s.forEach(function(a){return o.add(a,r)}),this;if(nn(s))return this.addLabel(s,r);if(Ht(s))s=Qt.delayedCall(0,s);else return this}return this!==s?Ui(this,s,r):this},t.getChildren=function(s,r,o,a){s===void 0&&(s=!0),r===void 0&&(r=!0),o===void 0&&(o=!0),a===void 0&&(a=-mi);for(var l=[],c=this._first;c;)c._start>=a&&(c instanceof Qt?r&&l.push(c):(o&&l.push(c),s&&l.push.apply(l,c.getChildren(!0,r,o)))),c=c._next;return l},t.getById=function(s){for(var r=this.getChildren(1,1,1),o=r.length;o--;)if(r[o].vars.id===s)return r[o]},t.remove=function(s){return nn(s)?this.removeLabel(s):Ht(s)?this.killTweensOf(s):(s.parent===this&&vc(this,s),s===this._recent&&(this._recent=this._last),Ar(this))},t.totalTime=function(s,r){return arguments.length?(this._forcing=1,!this._dp&&this._ts&&(this._start=Ut(Qn.time-(this._ts>0?s/this._ts:(this.totalDuration()-s)/-this._ts))),i.prototype.totalTime.call(this,s,r),this._forcing=0,this):this._tTime},t.addLabel=function(s,r){return this.labels[s]=oi(this,r),this},t.removeLabel=function(s){return delete this.labels[s],this},t.addPause=function(s,r,o){var a=Qt.delayedCall(0,r||Ca,o);return a.data="isPause",this._hasPause=1,Ui(this,a,oi(this,s))},t.removePause=function(s){var r=this._first;for(s=oi(this,s);r;)r._start===s&&r.data==="isPause"&&Xs(r),r=r._next},t.killTweensOf=function(s,r,o){for(var a=this.getTweensOf(s,o),l=a.length;l--;)Ls!==a[l]&&a[l].kill(s,r);return this},t.getTweensOf=function(s,r){for(var o=[],a=gi(s),l=this._first,c=vs(r),u;l;)l instanceof Qt?T1(l._targets,a)&&(c?(!Ls||l._initted&&l._ts)&&l.globalTime(0)<=r&&l.globalTime(l.totalDuration())>r:!r||l.isActive())&&o.push(l):(u=l.getTweensOf(a,r)).length&&o.push.apply(o,u),l=l._next;return o},t.tweenTo=function(s,r){r=r||{};var o=this,a=oi(o,s),l=r,c=l.startAt,u=l.onStart,f=l.onStartParams,h=l.immediateRender,d,x=Qt.to(o,ni({ease:r.ease||"none",lazy:!1,immediateRender:!1,time:a,overwrite:"auto",duration:r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale())||yt,onStart:function(){if(o.pause(),!d){var g=r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale());x._dur!==g&&Io(x,g,0,1).render(x._time,!0,!0),d=1}u&&u.apply(x,f||[])}},r));return h?x.render(0):x},t.tweenFromTo=function(s,r,o){return this.tweenTo(r,ni({startAt:{time:oi(this,s)}},o))},t.recent=function(){return this._recent},t.nextLabel=function(s){return s===void 0&&(s=this._time),Sm(this,oi(this,s))},t.previousLabel=function(s){return s===void 0&&(s=this._time),Sm(this,oi(this,s),1)},t.currentLabel=function(s){return arguments.length?this.seek(s,!0):this.previousLabel(this._time+yt)},t.shiftChildren=function(s,r,o){o===void 0&&(o=0);var a=this._first,l=this.labels,c;for(s=Ut(s);a;)a._start>=o&&(a._start+=s,a._end+=s),a=a._next;if(r)for(c in l)l[c]>=o&&(l[c]+=s);return Ar(this)},t.invalidate=function(s){var r=this._first;for(this._lock=0;r;)r.invalidate(s),r=r._next;return i.prototype.invalidate.call(this,s)},t.clear=function(s){s===void 0&&(s=!0);for(var r=this._first,o;r;)o=r._next,this.remove(r),r=o;return this._dp&&(this._time=this._tTime=this._pTime=0),s&&(this.labels={}),Ar(this)},t.totalDuration=function(s){var r=0,o=this,a=o._last,l=mi,c,u,f;if(arguments.length)return o.timeScale((o._repeat<0?o.duration():o.totalDuration())/(o.reversed()?-s:s));if(o._dirty){for(f=o.parent;a;)c=a._prev,a._dirty&&a.totalDuration(),u=a._start,u>l&&o._sort&&a._ts&&!o._lock?(o._lock=1,Ui(o,a,u-a._delay,1)._lock=0):l=u,u<0&&a._ts&&(r-=u,(!f&&!o._dp||f&&f.smoothChildTiming)&&(o._start+=Ut(u/o._ts),o._time-=u,o._tTime-=u),o.shiftChildren(-u,!1,-1/0),l=0),a._end>r&&a._ts&&(r=a._end),a=c;Io(o,o===Ot&&o._time>r?o._time:r,1,1),o._dirty=0}return o._tDur},e.updateRoot=function(s){if(Ot._ts&&(Ng(Ot,Jl(s,Ot)),Ug=Qn.frame),Qn.frame>=gm){gm+=ei.autoSleep||120;var r=Ot._first;if((!r||!r._ts)&&ei.autoSleep&&Qn._listeners.length<2){for(;r&&!r._ts;)r=r._next;r||Qn.sleep()}}},e})(Ra);ni(bn.prototype,{_lock:0,_hasPause:0,_forcing:0});var Q1=function(e,t,n,s,r,o,a){var l=new Nn(this._pt,e,t,0,1,hx,null,r),c=0,u=0,f,h,d,x,p,g,m,_;for(l.b=n,l.e=s,n+="",s+="",(m=~s.indexOf("random("))&&(s=Ea(s)),o&&(_=[n,s],o(_,e,t),n=_[0],s=_[1]),h=n.match(Tu)||[];f=Tu.exec(s);)x=f[0],p=s.substring(c,f.index),d?d=(d+1)%5:p.substr(-5)==="rgba("&&(d=1),x!==h[u++]&&(g=parseFloat(h[u-1])||0,l._pt={_next:l._pt,p:p||u===1?p:",",s:g,c:x.charAt(1)==="="?po(g,x)-g:parseFloat(x)-g,m:d&&d<4?Math.round:0},c=Tu.lastIndex);return l.c=c<s.length?s.substring(c,s.length):"",l.fp=a,(Pg.test(s)||m)&&(l.e=0),this._pt=l,l},ed=function(e,t,n,s,r,o,a,l,c,u){Ht(s)&&(s=s(r||0,e,o));var f=e[t],h=n!=="get"?n:Ht(f)?c?e[t.indexOf("set")||!Ht(e["get"+t.substr(3)])?t:"get"+t.substr(3)](c):e[t]():f,d=Ht(f)?c?J1:ux:nd,x;if(nn(s)&&(~s.indexOf("random(")&&(s=Ea(s)),s.charAt(1)==="="&&(x=po(h,s)+(fn(h)||0),(x||x===0)&&(s=x))),!u||h!==s||Wf)return!isNaN(h*s)&&s!==""?(x=new Nn(this._pt,e,t,+h||0,s-(h||0),typeof f=="boolean"?tw:fx,0,d),c&&(x.fp=c),a&&x.modifier(a,this,e),this._pt=x):(!f&&!(t in e)&&Kh(t,s),Q1.call(this,e,t,h,s,d,l||ei.stringFilter,c))},K1=function(e,t,n,s,r){if(Ht(e)&&(e=fa(e,r,t,n,s)),!Yi(e)||e.style&&e.nodeType||pn(e)||Ig(e))return nn(e)?fa(e,r,t,n,s):e;var o={},a;for(a in e)o[a]=fa(e[a],r,t,n,s);return o},ax=function(e,t,n,s,r,o){var a,l,c,u;if(Yn[e]&&(a=new Yn[e]).init(r,a.rawVars?t[e]:K1(t[e],s,r,o,n),n,s,o)!==!1&&(n._pt=l=new Nn(n._pt,r,e,0,1,a.render,a,0,a.priority),n!==no))for(c=n._ptLookup[n._targets.indexOf(r)],u=a._props.length;u--;)c[a._props[u]]=l;return a},Ls,Wf,td=function i(e,t,n){var s=e.vars,r=s.ease,o=s.startAt,a=s.immediateRender,l=s.lazy,c=s.onUpdate,u=s.runBackwards,f=s.yoyoEase,h=s.keyframes,d=s.autoRevert,x=e._dur,p=e._startAt,g=e._targets,m=e.parent,_=m&&m.data==="nested"?m.vars.targets:g,S=e._overwrite==="auto"&&!Xh,A=e.timeline,y,b,v,E,M,T,I,P,B,N,G,V,q;if(A&&(!h||!r)&&(r="none"),e._ease=yr(r,Eo.ease),e._yEase=f?ix(yr(f===!0?r:f,Eo.ease)):0,f&&e._yoyo&&!e._repeat&&(f=e._yEase,e._yEase=e._ease,e._ease=f),e._from=!A&&!!s.runBackwards,!A||h&&!s.stagger){if(P=g[0]?Sr(g[0]).harness:0,V=P&&s[P.prop],y=Zl(s,jh),p&&(p._zTime<0&&p.progress(1),t<0&&u&&a&&!d?p.render(-1,!0):p.revert(u&&x?Fl:b1),p._lazy=0),o){if(Xs(e._startAt=Qt.set(g,ni({data:"isStart",overwrite:!1,parent:m,immediateRender:!0,lazy:!p&&Un(l),startAt:null,delay:0,onUpdate:c&&function(){return $n(e,"onUpdate")},stagger:0},o))),e._startAt._dp=0,e._startAt._sat=e,t<0&&(on||!a&&!d)&&e._startAt.revert(Fl),a&&x&&t<=0&&n<=0){t&&(e._zTime=t);return}}else if(u&&x&&!p){if(t&&(a=!1),v=ni({overwrite:!1,data:"isFromStart",lazy:a&&!p&&Un(l),immediateRender:a,stagger:0,parent:m},y),V&&(v[P.prop]=V),Xs(e._startAt=Qt.set(g,v)),e._startAt._dp=0,e._startAt._sat=e,t<0&&(on?e._startAt.revert(Fl):e._startAt.render(-1,!0)),e._zTime=t,!a)i(e._startAt,yt,yt);else if(!t)return}for(e._pt=e._ptCache=0,l=x&&Un(l)||l&&!x,b=0;b<g.length;b++){if(M=g[b],I=M._gsap||Zh(g)[b]._gsap,e._ptLookup[b]=N={},Nf[I.id]&&ks.length&&$l(),G=_===g?b:_.indexOf(M),P&&(B=new P).init(M,V||y,e,G,_)!==!1&&(e._pt=E=new Nn(e._pt,M,B.name,0,1,B.render,B,0,B.priority),B._props.forEach(function(X){N[X]=E}),B.priority&&(T=1)),!P||V)for(v in y)Yn[v]&&(B=ax(v,y,e,G,M,_))?B.priority&&(T=1):N[v]=E=ed.call(e,M,v,"get",y[v],G,_,0,s.stringFilter);e._op&&e._op[b]&&e.kill(M,e._op[b]),S&&e._pt&&(Ls=e,Ot.killTweensOf(M,N,e.globalTime(t)),q=!e.parent,Ls=0),e._pt&&l&&(Nf[I.id]=1)}T&&dx(e),e._onInit&&e._onInit(e)}e._onUpdate=c,e._initted=(!e._op||e._pt)&&!q,h&&t<=0&&A.render(mi,!0,!0)},j1=function(e,t,n,s,r,o,a,l){var c=(e._pt&&e._ptCache||(e._ptCache={}))[t],u,f,h,d;if(!c)for(c=e._ptCache[t]=[],h=e._ptLookup,d=e._targets.length;d--;){if(u=h[d][t],u&&u.d&&u.d._pt)for(u=u.d._pt;u&&u.p!==t&&u.fp!==t;)u=u._next;if(!u)return Wf=1,e.vars[t]="+=0",td(e,a),Wf=0,l?Ta(t+" not eligible for reset"):1;c.push(u)}for(d=c.length;d--;)f=c[d],u=f._pt||f,u.s=(s||s===0)&&!r?s:u.s+(s||0)+o*u.c,u.c=n-u.s,f.e&&(f.e=Vt(n)+fn(f.e)),f.b&&(f.b=u.s+fn(f.b))},$1=function(e,t){var n=e[0]?Sr(e[0]).harness:0,s=n&&n.aliases,r,o,a,l;if(!s)return t;r=wo({},t);for(o in s)if(o in r)for(l=s[o].split(","),a=l.length;a--;)r[l[a]]=r[o];return r},Z1=function(e,t,n,s){var r=t.ease||s||"power1.inOut",o,a;if(pn(t))a=n[e]||(n[e]=[]),t.forEach(function(l,c){return a.push({t:c/(t.length-1)*100,v:l,e:r})});else for(o in t)a=n[o]||(n[o]=[]),o==="ease"||a.push({t:parseFloat(e),v:t[o],e:r})},fa=function(e,t,n,s,r){return Ht(e)?e.call(t,n,s,r):nn(e)&&~e.indexOf("random(")?Ea(e):e},lx=$h+"repeat,repeatDelay,yoyo,repeatRefresh,yoyoEase,autoRevert",cx={};On(lx+",id,stagger,delay,duration,paused,scrollTrigger",function(i){return cx[i]=1});var Qt=(function(i){wg(e,i);function e(n,s,r,o){var a;typeof s=="number"&&(r.duration=s,s=r,r=null),a=i.call(this,o?s:ca(s))||this;var l=a.vars,c=l.duration,u=l.delay,f=l.immediateRender,h=l.stagger,d=l.overwrite,x=l.keyframes,p=l.defaults,g=l.scrollTrigger,m=l.yoyoEase,_=s.parent||Ot,S=(pn(n)||Ig(n)?vs(n[0]):"length"in s)?[n]:gi(n),A,y,b,v,E,M,T,I;if(a._targets=S.length?Zh(S):Ta("GSAP target "+n+" not found. https://gsap.com",!ei.nullTargetWarn)||[],a._ptLookup=[],a._overwrite=d,x||h||yl(c)||yl(u)){if(s=a.vars,A=a.timeline=new bn({data:"nested",defaults:p||{},targets:_&&_.data==="nested"?_.vars.targets:S}),A.kill(),A.parent=A._dp=rs(a),A._start=0,h||yl(c)||yl(u)){if(v=S.length,T=h&&Yg(h),Yi(h))for(E in h)~lx.indexOf(E)&&(I||(I={}),I[E]=h[E]);for(y=0;y<v;y++)b=Zl(s,cx),b.stagger=0,m&&(b.yoyoEase=m),I&&wo(b,I),M=S[y],b.duration=+fa(c,rs(a),y,M,S),b.delay=(+fa(u,rs(a),y,M,S)||0)-a._delay,!h&&v===1&&b.delay&&(a._delay=u=b.delay,a._start+=u,b.delay=0),A.to(M,b,T?T(y,M,S):0),A._ease=st.none;A.duration()?c=u=0:a.timeline=0}else if(x){ca(ni(A.vars.defaults,{ease:"none"})),A._ease=yr(x.ease||s.ease||"none");var P=0,B,N,G;if(pn(x))x.forEach(function(V){return A.to(S,V,">")}),A.duration();else{b={};for(E in x)E==="ease"||E==="easeEach"||Z1(E,x[E],b,x.easeEach);for(E in b)for(B=b[E].sort(function(V,q){return V.t-q.t}),P=0,y=0;y<B.length;y++)N=B[y],G={ease:N.e,duration:(N.t-(y?B[y-1].t:0))/100*c},G[E]=N.v,A.to(S,G,P),P+=G.duration;A.duration()<c&&A.to({},{duration:c-A.duration()})}}c||a.duration(c=A.duration())}else a.timeline=0;return d===!0&&!Xh&&(Ls=rs(a),Ot.killTweensOf(S),Ls=0),Ui(_,rs(a),r),s.reversed&&a.reverse(),s.paused&&a.paused(!0),(f||!c&&!x&&a._start===Ut(_._time)&&Un(f)&&R1(rs(a))&&_.data!=="nested")&&(a._tTime=-yt,a.render(Math.max(0,-u)||0)),g&&Gg(rs(a),g),a}var t=e.prototype;return t.render=function(s,r,o){var a=this._time,l=this._tDur,c=this._dur,u=s<0,f=s>l-yt&&!u?l:s<yt?0:s,h,d,x,p,g,m,_,S,A;if(!c)D1(this,s,r,o);else if(f!==this._tTime||!s||o||!this._initted&&this._tTime||this._startAt&&this._zTime<0!==u||this._lazy){if(h=f,S=this.timeline,this._repeat){if(p=c+this._rDelay,this._repeat<-1&&u)return this.totalTime(p*100+s,r,o);if(h=Ut(f%p),f===l?(x=this._repeat,h=c):(g=Ut(f/p),x=~~g,x&&x===g?(h=c,x--):h>c&&(h=c)),m=this._yoyo&&x&1,m&&(A=this._yEase,h=c-h),g=Ro(this._tTime,p),h===a&&!o&&this._initted&&x===g)return this._tTime=f,this;x!==g&&(S&&this._yEase&&sx(S,m),this.vars.repeatRefresh&&!m&&!this._lock&&h!==p&&this._initted&&(this._lock=o=1,this.render(Ut(p*x),!0).invalidate()._lock=0))}if(!this._initted){if(Wg(this,u?s:h,o,r,f))return this._tTime=0,this;if(a!==this._time&&!(o&&this.vars.repeatRefresh&&x!==g))return this;if(c!==this._dur)return this.render(s,r,o)}if(this._tTime=f,this._time=h,!this._act&&this._ts&&(this._act=1,this._lazy=0),this.ratio=_=(A||this._ease)(h/c),this._from&&(this.ratio=_=1-_),!a&&f&&!r&&!g&&($n(this,"onStart"),this._tTime!==f))return this;for(d=this._pt;d;)d.r(_,d.d),d=d._next;S&&S.render(s<0?s:S._dur*S._ease(h/this._dur),r,o)||this._startAt&&(this._zTime=s),this._onUpdate&&!r&&(u&&zf(this,s,r,o),$n(this,"onUpdate")),this._repeat&&x!==g&&this.vars.onRepeat&&!r&&this.parent&&$n(this,"onRepeat"),(f===this._tDur||!f)&&this._tTime===f&&(u&&!this._onUpdate&&zf(this,s,!0,!0),(s||!c)&&(f===this._tDur&&this._ts>0||!f&&this._ts<0)&&Xs(this,1),!r&&!(u&&!a)&&(f||a||m)&&($n(this,f===l?"onComplete":"onReverseComplete",!0),this._prom&&!(f<l&&this.timeScale()>0)&&this._prom()))}return this},t.targets=function(){return this._targets},t.invalidate=function(s){return(!s||!this.vars.runBackwards)&&(this._startAt=0),this._pt=this._op=this._onUpdate=this._lazy=this.ratio=0,this._ptLookup=[],this.timeline&&this.timeline.invalidate(s),i.prototype.invalidate.call(this,s)},t.resetTo=function(s,r,o,a,l){wa||Qn.wake(),this._ts||this.play();var c=Math.min(this._dur,(this._dp._time-this._start)*this._ts),u;return this._initted||td(this,c),u=this._ease(c/this._dur),j1(this,s,r,o,a,u,c,l)?this.resetTo(s,r,o,a,1):(Ac(this,0),this.parent||Hg(this._dp,this,"_first","_last",this._dp._sort?"_start":0),this.render(0))},t.kill=function(s,r){if(r===void 0&&(r="all"),!s&&(!r||r==="all"))return this._lazy=this._pt=0,this.parent?Ko(this):this.scrollTrigger&&this.scrollTrigger.kill(!!on),this;if(this.timeline){var o=this.timeline.totalDuration();return this.timeline.killTweensOf(s,r,Ls&&Ls.vars.overwrite!==!0)._first||Ko(this),this.parent&&o!==this.timeline.totalDuration()&&Io(this,this._dur*this.timeline._tDur/o,0,1),this}var a=this._targets,l=s?gi(s):a,c=this._ptLookup,u=this._pt,f,h,d,x,p,g,m;if((!r||r==="all")&&E1(a,l))return r==="all"&&(this._pt=0),Ko(this);for(f=this._op=this._op||[],r!=="all"&&(nn(r)&&(p={},On(r,function(_){return p[_]=1}),r=p),r=$1(a,r)),m=a.length;m--;)if(~l.indexOf(a[m])){h=c[m],r==="all"?(f[m]=r,x=h,d={}):(d=f[m]=f[m]||{},x=r);for(p in x)g=h&&h[p],g&&((!("kill"in g.d)||g.d.kill(p)===!0)&&vc(this,g,"_pt"),delete h[p]),d!=="all"&&(d[p]=1)}return this._initted&&!this._pt&&u&&Ko(this),this},e.to=function(s,r){return new e(s,r,arguments[2])},e.from=function(s,r){return ua(1,arguments)},e.delayedCall=function(s,r,o,a){return new e(r,0,{immediateRender:!1,lazy:!1,overwrite:!1,delay:s,onComplete:r,onReverseComplete:r,onCompleteParams:o,onReverseCompleteParams:o,callbackScope:a})},e.fromTo=function(s,r,o){return ua(2,arguments)},e.set=function(s,r){return r.duration=0,r.repeatDelay||(r.repeat=0),new e(s,r)},e.killTweensOf=function(s,r,o){return Ot.killTweensOf(s,r,o)},e})(Ra);ni(Qt.prototype,{_targets:[],_lazy:0,_startAt:0,_op:0,_onInit:0});On("staggerTo,staggerFrom,staggerFromTo",function(i){Qt[i]=function(){var e=new bn,t=Hf.call(arguments,0);return t.splice(i==="staggerFromTo"?5:4,0,0),e[i].apply(e,t)}});var nd=function(e,t,n){return e[t]=n},ux=function(e,t,n){return e[t](n)},J1=function(e,t,n,s){return e[t](s.fp,n)},ew=function(e,t,n){return e.setAttribute(t,n)},id=function(e,t){return Ht(e[t])?ux:qh(e[t])&&e.setAttribute?ew:nd},fx=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e6)/1e6,t)},tw=function(e,t){return t.set(t.t,t.p,!!(t.s+t.c*e),t)},hx=function(e,t){var n=t._pt,s="";if(!e&&t.b)s=t.b;else if(e===1&&t.e)s=t.e;else{for(;n;)s=n.p+(n.m?n.m(n.s+n.c*e):Math.round((n.s+n.c*e)*1e4)/1e4)+s,n=n._next;s+=t.c}t.set(t.t,t.p,s,t)},sd=function(e,t){for(var n=t._pt;n;)n.r(e,n.d),n=n._next},nw=function(e,t,n,s){for(var r=this._pt,o;r;)o=r._next,r.p===s&&r.modifier(e,t,n),r=o},iw=function(e){for(var t=this._pt,n,s;t;)s=t._next,t.p===e&&!t.op||t.op===e?vc(this,t,"_pt"):t.dep||(n=1),t=s;return!n},sw=function(e,t,n,s){s.mSet(e,t,s.m.call(s.tween,n,s.mt),s)},dx=function(e){for(var t=e._pt,n,s,r,o;t;){for(n=t._next,s=r;s&&s.pr>t.pr;)s=s._next;(t._prev=s?s._prev:o)?t._prev._next=t:r=t,(t._next=s)?s._prev=t:o=t,t=n}e._pt=r},Nn=(function(){function i(t,n,s,r,o,a,l,c,u){this.t=n,this.s=r,this.c=o,this.p=s,this.r=a||fx,this.d=l||this,this.set=c||nd,this.pr=u||0,this._next=t,t&&(t._prev=this)}var e=i.prototype;return e.modifier=function(n,s,r){this.mSet=this.mSet||this.set,this.set=sw,this.m=n,this.mt=r,this.tween=s},i})();On($h+"parent,duration,ease,delay,overwrite,runBackwards,startAt,yoyo,immediateRender,repeat,repeatDelay,data,paused,reversed,lazy,callbackScope,stringFilter,id,yoyoEase,stagger,inherit,repeatRefresh,keyframes,autoRevert,scrollTrigger",function(i){return jh[i]=1});ti.TweenMax=ti.TweenLite=Qt;ti.TimelineLite=ti.TimelineMax=bn;Ot=new bn({sortChildren:!1,defaults:Eo,autoRemoveChildren:!0,id:"root",smoothChildTiming:!0});ei.stringFilter=nx;var br=[],Bl={},rw=[],ym=0,ow=0,Iu=function(e){return(Bl[e]||rw).map(function(t){return t()})},Xf=function(){var e=Date.now(),t=[];e-ym>2&&(Iu("matchMediaInit"),br.forEach(function(n){var s=n.queries,r=n.conditions,o,a,l,c;for(a in s)o=Pi.matchMedia(s[a]).matches,o&&(l=1),o!==r[a]&&(r[a]=o,c=1);c&&(n.revert(),l&&t.push(n))}),Iu("matchMediaRevert"),t.forEach(function(n){return n.onMatch(n,function(s){return n.add(null,s)})}),ym=e,Iu("matchMedia"))},px=(function(){function i(t,n){this.selector=n&&Vf(n),this.data=[],this._r=[],this.isReverted=!1,this.id=ow++,t&&this.add(t)}var e=i.prototype;return e.add=function(n,s,r){Ht(n)&&(r=s,s=n,n=Ht);var o=this,a=function(){var c=It,u=o.selector,f;return c&&c!==o&&c.data.push(o),r&&(o.selector=Vf(r)),It=o,f=s.apply(o,arguments),Ht(f)&&o._r.push(f),It=c,o.selector=u,o.isReverted=!1,f};return o.last=a,n===Ht?a(o,function(l){return o.add(null,l)}):n?o[n]=a:a},e.ignore=function(n){var s=It;It=null,n(this),It=s},e.getTweens=function(){var n=[];return this.data.forEach(function(s){return s instanceof i?n.push.apply(n,s.getTweens()):s instanceof Qt&&!(s.parent&&s.parent.data==="nested")&&n.push(s)}),n},e.clear=function(){this._r.length=this.data.length=0},e.kill=function(n,s){var r=this;if(n?(function(){for(var a=r.getTweens(),l=r.data.length,c;l--;)c=r.data[l],c.data==="isFlip"&&(c.revert(),c.getChildren(!0,!0,!1).forEach(function(u){return a.splice(a.indexOf(u),1)}));for(a.map(function(u){return{g:u._dur||u._delay||u._sat&&!u._sat.vars.immediateRender?u.globalTime(0):-1/0,t:u}}).sort(function(u,f){return f.g-u.g||-1/0}).forEach(function(u){return u.t.revert(n)}),l=r.data.length;l--;)c=r.data[l],c instanceof bn?c.data!=="nested"&&(c.scrollTrigger&&c.scrollTrigger.revert(),c.kill()):!(c instanceof Qt)&&c.revert&&c.revert(n);r._r.forEach(function(u){return u(n,r)}),r.isReverted=!0})():this.data.forEach(function(a){return a.kill&&a.kill()}),this.clear(),s)for(var o=br.length;o--;)br[o].id===this.id&&br.splice(o,1)},e.revert=function(n){this.kill(n||{})},i})(),aw=(function(){function i(t){this.contexts=[],this.scope=t,It&&It.data.push(this)}var e=i.prototype;return e.add=function(n,s,r){Yi(n)||(n={matches:n});var o=new px(0,r||this.scope),a=o.conditions={},l,c,u;It&&!o.selector&&(o.selector=It.selector),this.contexts.push(o),s=o.add("onMatch",s),o.queries=n;for(c in n)c==="all"?u=1:(l=Pi.matchMedia(n[c]),l&&(br.indexOf(o)<0&&br.push(o),(a[c]=l.matches)&&(u=1),l.addListener?l.addListener(Xf):l.addEventListener("change",Xf)));return u&&s(o,function(f){return o.add(null,f)}),this},e.revert=function(n){this.kill(n||{})},e.kill=function(n){this.contexts.forEach(function(s){return s.kill(n,!0)})},i})(),ec={registerPlugin:function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];t.forEach(function(s){return Jg(s)})},timeline:function(e){return new bn(e)},getTweensOf:function(e,t){return Ot.getTweensOf(e,t)},getProperty:function(e,t,n,s){nn(e)&&(e=gi(e)[0]);var r=Sr(e||{}).get,o=n?kg:zg;return n==="native"&&(n=""),e&&(t?o((Yn[t]&&Yn[t].get||r)(e,t,n,s)):function(a,l,c){return o((Yn[a]&&Yn[a].get||r)(e,a,l,c))})},quickSetter:function(e,t,n){if(e=gi(e),e.length>1){var s=e.map(function(u){return kn.quickSetter(u,t,n)}),r=s.length;return function(u){for(var f=r;f--;)s[f](u)}}e=e[0]||{};var o=Yn[t],a=Sr(e),l=a.harness&&(a.harness.aliases||{})[t]||t,c=o?function(u){var f=new o;no._pt=0,f.init(e,n?u+n:u,no,0,[e]),f.render(1,f),no._pt&&sd(1,no)}:a.set(e,l);return o?c:function(u){return c(e,l,n?u+n:u,a,1)}},quickTo:function(e,t,n){var s,r=kn.to(e,ni((s={},s[t]="+=0.1",s.paused=!0,s.stagger=0,s),n||{})),o=function(l,c,u){return r.resetTo(t,l,c,u)};return o.tween=r,o},isTweening:function(e){return Ot.getTweensOf(e,!0).length>0},defaults:function(e){return e&&e.ease&&(e.ease=yr(e.ease,Eo.ease)),xm(Eo,e||{})},config:function(e){return xm(ei,e||{})},registerEffect:function(e){var t=e.name,n=e.effect,s=e.plugins,r=e.defaults,o=e.extendTimeline;(s||"").split(",").forEach(function(a){return a&&!Yn[a]&&!ti[a]&&Ta(t+" effect requires "+a+" plugin.")}),Cu[t]=function(a,l,c){return n(gi(a),ni(l||{},r),c)},o&&(bn.prototype[t]=function(a,l,c){return this.add(Cu[t](a,Yi(l)?l:(c=l)&&{},this),c)})},registerEase:function(e,t){st[e]=yr(t)},parseEase:function(e,t){return arguments.length?yr(e,t):st},getById:function(e){return Ot.getById(e)},exportRoot:function(e,t){e===void 0&&(e={});var n=new bn(e),s,r;for(n.smoothChildTiming=Un(e.smoothChildTiming),Ot.remove(n),n._dp=0,n._time=n._tTime=Ot._time,s=Ot._first;s;)r=s._next,(t||!(!s._dur&&s instanceof Qt&&s.vars.onComplete===s._targets[0]))&&Ui(n,s,s._start-s._delay),s=r;return Ui(Ot,n,0),n},context:function(e,t){return e?new px(e,t):It},matchMedia:function(e){return new aw(e)},matchMediaRefresh:function(){return br.forEach(function(e){var t=e.conditions,n,s;for(s in t)t[s]&&(t[s]=!1,n=1);n&&e.revert()})||Xf()},addEventListener:function(e,t){var n=Bl[e]||(Bl[e]=[]);~n.indexOf(t)||n.push(t)},removeEventListener:function(e,t){var n=Bl[e],s=n&&n.indexOf(t);s>=0&&n.splice(s,1)},utils:{wrap:z1,wrapYoyo:k1,distribute:Yg,random:Kg,snap:Qg,normalize:N1,getUnit:fn,clamp:L1,splitColor:ex,toArray:gi,selector:Vf,mapRange:$g,pipe:U1,unitize:O1,interpolate:H1,shuffle:qg},install:Lg,effects:Cu,ticker:Qn,updateRoot:bn.updateRoot,plugins:Yn,globalTimeline:Ot,core:{PropTween:Nn,globals:Bg,Tween:Qt,Timeline:bn,Animation:Ra,getCache:Sr,_removeLinkedListItem:vc,reverting:function(){return on},context:function(e){return e&&It&&(It.data.push(e),e._ctx=It),It},suppressOverwrites:function(e){return Xh=e}}};On("to,from,fromTo,delayedCall,set,killTweensOf",function(i){return ec[i]=Qt[i]});Qn.add(bn.updateRoot);no=ec.to({},{duration:0});var lw=function(e,t){for(var n=e._pt;n&&n.p!==t&&n.op!==t&&n.fp!==t;)n=n._next;return n},cw=function(e,t){var n=e._targets,s,r,o;for(s in t)for(r=n.length;r--;)o=e._ptLookup[r][s],o&&(o=o.d)&&(o._pt&&(o=lw(o,s)),o&&o.modifier&&o.modifier(t[s],e,n[r],s))},Du=function(e,t){return{name:e,headless:1,rawVars:1,init:function(s,r,o){o._onInit=function(a){var l,c;if(nn(r)&&(l={},On(r,function(u){return l[u]=1}),r=l),t){l={};for(c in r)l[c]=t(r[c]);r=l}cw(a,r)}}}},kn=ec.registerPlugin({name:"attr",init:function(e,t,n,s,r){var o,a,l;this.tween=n;for(o in t)l=e.getAttribute(o)||"",a=this.add(e,"setAttribute",(l||0)+"",t[o],s,r,0,0,o),a.op=o,a.b=l,this._props.push(o)},render:function(e,t){for(var n=t._pt;n;)on?n.set(n.t,n.p,n.b,n):n.r(e,n.d),n=n._next}},{name:"endArray",headless:1,init:function(e,t){for(var n=t.length;n--;)this.add(e,n,e[n]||0,t[n],0,0,0,0,0,1)}},Du("roundProps",Gf),Du("modifiers"),Du("snap",Qg))||ec;Qt.version=bn.version=kn.version="3.14.2";Fg=1;Yh()&&Do();st.Power0;st.Power1;st.Power2;st.Power3;st.Power4;st.Linear;st.Quad;st.Cubic;st.Quart;st.Quint;st.Strong;st.Elastic;st.Back;st.SteppedEase;st.Bounce;st.Sine;st.Expo;st.Circ;var bm,Bs,mo,rd,xr,Mm,od,uw=function(){return typeof window<"u"},Ss={},hr=180/Math.PI,go=Math.PI/180,Qr=Math.atan2,Tm=1e8,ad=/([A-Z])/g,fw=/(left|right|width|margin|padding|x)/i,hw=/[\s,\(]\S/,ki={autoAlpha:"opacity,visibility",scale:"scaleX,scaleY",alpha:"opacity"},qf=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},dw=function(e,t){return t.set(t.t,t.p,e===1?t.e:Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},pw=function(e,t){return t.set(t.t,t.p,e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},mw=function(e,t){return t.set(t.t,t.p,e===1?t.e:e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},gw=function(e,t){var n=t.s+t.c*e;t.set(t.t,t.p,~~(n+(n<0?-.5:.5))+t.u,t)},mx=function(e,t){return t.set(t.t,t.p,e?t.e:t.b,t)},gx=function(e,t){return t.set(t.t,t.p,e!==1?t.b:t.e,t)},xw=function(e,t,n){return e.style[t]=n},_w=function(e,t,n){return e.style.setProperty(t,n)},vw=function(e,t,n){return e._gsap[t]=n},Sw=function(e,t,n){return e._gsap.scaleX=e._gsap.scaleY=n},Aw=function(e,t,n,s,r){var o=e._gsap;o.scaleX=o.scaleY=n,o.renderTransform(r,o)},yw=function(e,t,n,s,r){var o=e._gsap;o[t]=n,o.renderTransform(r,o)},Nt="transform",zn=Nt+"Origin",bw=function i(e,t){var n=this,s=this.target,r=s.style,o=s._gsap;if(e in Ss&&r){if(this.tfm=this.tfm||{},e!=="transform")e=ki[e]||e,~e.indexOf(",")?e.split(",").forEach(function(a){return n.tfm[a]=cs(s,a)}):this.tfm[e]=o.x?o[e]:cs(s,e),e===zn&&(this.tfm.zOrigin=o.zOrigin);else return ki.transform.split(",").forEach(function(a){return i.call(n,a,t)});if(this.props.indexOf(Nt)>=0)return;o.svg&&(this.svgo=s.getAttribute("data-svg-origin"),this.props.push(zn,t,"")),e=Nt}(r||t)&&this.props.push(e,t,r[e])},xx=function(e){e.translate&&(e.removeProperty("translate"),e.removeProperty("scale"),e.removeProperty("rotate"))},Mw=function(){var e=this.props,t=this.target,n=t.style,s=t._gsap,r,o;for(r=0;r<e.length;r+=3)e[r+1]?e[r+1]===2?t[e[r]](e[r+2]):t[e[r]]=e[r+2]:e[r+2]?n[e[r]]=e[r+2]:n.removeProperty(e[r].substr(0,2)==="--"?e[r]:e[r].replace(ad,"-$1").toLowerCase());if(this.tfm){for(o in this.tfm)s[o]=this.tfm[o];s.svg&&(s.renderTransform(),t.setAttribute("data-svg-origin",this.svgo||"")),r=od(),(!r||!r.isStart)&&!n[Nt]&&(xx(n),s.zOrigin&&n[zn]&&(n[zn]+=" "+s.zOrigin+"px",s.zOrigin=0,s.renderTransform()),s.uncache=1)}},_x=function(e,t){var n={target:e,props:[],revert:Mw,save:bw};return e._gsap||kn.core.getCache(e),t&&e.style&&e.nodeType&&t.split(",").forEach(function(s){return n.save(s)}),n},vx,Yf=function(e,t){var n=Bs.createElementNS?Bs.createElementNS((t||"http://www.w3.org/1999/xhtml").replace(/^https/,"http"),e):Bs.createElement(e);return n&&n.style?n:Bs.createElement(e)},Zn=function i(e,t,n){var s=getComputedStyle(e);return s[t]||s.getPropertyValue(t.replace(ad,"-$1").toLowerCase())||s.getPropertyValue(t)||!n&&i(e,Po(t)||t,1)||""},Cm="O,Moz,ms,Ms,Webkit".split(","),Po=function(e,t,n){var s=t||xr,r=s.style,o=5;if(e in r&&!n)return e;for(e=e.charAt(0).toUpperCase()+e.substr(1);o--&&!(Cm[o]+e in r););return o<0?null:(o===3?"ms":o>=0?Cm[o]:"")+e},Qf=function(){uw()&&window.document&&(bm=window,Bs=bm.document,mo=Bs.documentElement,xr=Yf("div")||{style:{}},Yf("div"),Nt=Po(Nt),zn=Nt+"Origin",xr.style.cssText="border-width:0;line-height:0;position:absolute;padding:0",vx=!!Po("perspective"),od=kn.core.reverting,rd=1)},Em=function(e){var t=e.ownerSVGElement,n=Yf("svg",t&&t.getAttribute("xmlns")||"http://www.w3.org/2000/svg"),s=e.cloneNode(!0),r;s.style.display="block",n.appendChild(s),mo.appendChild(n);try{r=s.getBBox()}catch{}return n.removeChild(s),mo.removeChild(n),r},wm=function(e,t){for(var n=t.length;n--;)if(e.hasAttribute(t[n]))return e.getAttribute(t[n])},Sx=function(e){var t,n;try{t=e.getBBox()}catch{t=Em(e),n=1}return t&&(t.width||t.height)||n||(t=Em(e)),t&&!t.width&&!t.x&&!t.y?{x:+wm(e,["x","cx","x1"])||0,y:+wm(e,["y","cy","y1"])||0,width:0,height:0}:t},Ax=function(e){return!!(e.getCTM&&(!e.parentNode||e.ownerSVGElement)&&Sx(e))},qs=function(e,t){if(t){var n=e.style,s;t in Ss&&t!==zn&&(t=Nt),n.removeProperty?(s=t.substr(0,2),(s==="ms"||t.substr(0,6)==="webkit")&&(t="-"+t),n.removeProperty(s==="--"?t:t.replace(ad,"-$1").toLowerCase())):n.removeAttribute(t)}},Us=function(e,t,n,s,r,o){var a=new Nn(e._pt,t,n,0,1,o?gx:mx);return e._pt=a,a.b=s,a.e=r,e._props.push(n),a},Rm={deg:1,rad:1,turn:1},Tw={grid:1,flex:1},Ys=function i(e,t,n,s){var r=parseFloat(n)||0,o=(n+"").trim().substr((r+"").length)||"px",a=xr.style,l=fw.test(t),c=e.tagName.toLowerCase()==="svg",u=(c?"client":"offset")+(l?"Width":"Height"),f=100,h=s==="px",d=s==="%",x,p,g,m;if(s===o||!r||Rm[s]||Rm[o])return r;if(o!=="px"&&!h&&(r=i(e,t,n,"px")),m=e.getCTM&&Ax(e),(d||o==="%")&&(Ss[t]||~t.indexOf("adius")))return x=m?e.getBBox()[l?"width":"height"]:e[u],Vt(d?r/x*f:r/100*x);if(a[l?"width":"height"]=f+(h?o:s),p=s!=="rem"&&~t.indexOf("adius")||s==="em"&&e.appendChild&&!c?e:e.parentNode,m&&(p=(e.ownerSVGElement||{}).parentNode),(!p||p===Bs||!p.appendChild)&&(p=Bs.body),g=p._gsap,g&&d&&g.width&&l&&g.time===Qn.time&&!g.uncache)return Vt(r/g.width*f);if(d&&(t==="height"||t==="width")){var _=e.style[t];e.style[t]=f+s,x=e[u],_?e.style[t]=_:qs(e,t)}else(d||o==="%")&&!Tw[Zn(p,"display")]&&(a.position=Zn(e,"position")),p===e&&(a.position="static"),p.appendChild(xr),x=xr[u],p.removeChild(xr),a.position="absolute";return l&&d&&(g=Sr(p),g.time=Qn.time,g.width=p[u]),Vt(h?x*r/f:x&&r?f/x*r:0)},cs=function(e,t,n,s){var r;return rd||Qf(),t in ki&&t!=="transform"&&(t=ki[t],~t.indexOf(",")&&(t=t.split(",")[0])),Ss[t]&&t!=="transform"?(r=Da(e,s),r=t!=="transformOrigin"?r[t]:r.svg?r.origin:nc(Zn(e,zn))+" "+r.zOrigin+"px"):(r=e.style[t],(!r||r==="auto"||s||~(r+"").indexOf("calc("))&&(r=tc[t]&&tc[t](e,t,n)||Zn(e,t)||Og(e,t)||(t==="opacity"?1:0))),n&&!~(r+"").trim().indexOf(" ")?Ys(e,t,r,n)+n:r},Cw=function(e,t,n,s){if(!n||n==="none"){var r=Po(t,e,1),o=r&&Zn(e,r,1);o&&o!==n?(t=r,n=o):t==="borderColor"&&(n=Zn(e,"borderTopColor"))}var a=new Nn(this._pt,e.style,t,0,1,hx),l=0,c=0,u,f,h,d,x,p,g,m,_,S,A,y;if(a.b=n,a.e=s,n+="",s+="",s.substring(0,6)==="var(--"&&(s=Zn(e,s.substring(4,s.indexOf(")")))),s==="auto"&&(p=e.style[t],e.style[t]=s,s=Zn(e,t)||s,p?e.style[t]=p:qs(e,t)),u=[n,s],nx(u),n=u[0],s=u[1],h=n.match(to)||[],y=s.match(to)||[],y.length){for(;f=to.exec(s);)g=f[0],_=s.substring(l,f.index),x?x=(x+1)%5:(_.substr(-5)==="rgba("||_.substr(-5)==="hsla(")&&(x=1),g!==(p=h[c++]||"")&&(d=parseFloat(p)||0,A=p.substr((d+"").length),g.charAt(1)==="="&&(g=po(d,g)+A),m=parseFloat(g),S=g.substr((m+"").length),l=to.lastIndex-S.length,S||(S=S||ei.units[t]||A,l===s.length&&(s+=S,a.e+=S)),A!==S&&(d=Ys(e,t,p,S)||0),a._pt={_next:a._pt,p:_||c===1?_:",",s:d,c:m-d,m:x&&x<4||t==="zIndex"?Math.round:0});a.c=l<s.length?s.substring(l,s.length):""}else a.r=t==="display"&&s==="none"?gx:mx;return Pg.test(s)&&(a.e=0),this._pt=a,a},Im={top:"0%",bottom:"100%",left:"0%",right:"100%",center:"50%"},Ew=function(e){var t=e.split(" "),n=t[0],s=t[1]||"50%";return(n==="top"||n==="bottom"||s==="left"||s==="right")&&(e=n,n=s,s=e),t[0]=Im[n]||n,t[1]=Im[s]||s,t.join(" ")},ww=function(e,t){if(t.tween&&t.tween._time===t.tween._dur){var n=t.t,s=n.style,r=t.u,o=n._gsap,a,l,c;if(r==="all"||r===!0)s.cssText="",l=1;else for(r=r.split(","),c=r.length;--c>-1;)a=r[c],Ss[a]&&(l=1,a=a==="transformOrigin"?zn:Nt),qs(n,a);l&&(qs(n,Nt),o&&(o.svg&&n.removeAttribute("transform"),s.scale=s.rotate=s.translate="none",Da(n,1),o.uncache=1,xx(s)))}},tc={clearProps:function(e,t,n,s,r){if(r.data!=="isFromStart"){var o=e._pt=new Nn(e._pt,t,n,0,0,ww);return o.u=s,o.pr=-10,o.tween=r,e._props.push(n),1}}},Ia=[1,0,0,1,0,0],yx={},bx=function(e){return e==="matrix(1, 0, 0, 1, 0, 0)"||e==="none"||!e},Dm=function(e){var t=Zn(e,Nt);return bx(t)?Ia:t.substr(7).match(Dg).map(Vt)},ld=function(e,t){var n=e._gsap||Sr(e),s=e.style,r=Dm(e),o,a,l,c;return n.svg&&e.getAttribute("transform")?(l=e.transform.baseVal.consolidate().matrix,r=[l.a,l.b,l.c,l.d,l.e,l.f],r.join(",")==="1,0,0,1,0,0"?Ia:r):(r===Ia&&!e.offsetParent&&e!==mo&&!n.svg&&(l=s.display,s.display="block",o=e.parentNode,(!o||!e.offsetParent&&!e.getBoundingClientRect().width)&&(c=1,a=e.nextElementSibling,mo.appendChild(e)),r=Dm(e),l?s.display=l:qs(e,"display"),c&&(a?o.insertBefore(e,a):o?o.appendChild(e):mo.removeChild(e))),t&&r.length>6?[r[0],r[1],r[4],r[5],r[12],r[13]]:r)},Kf=function(e,t,n,s,r,o){var a=e._gsap,l=r||ld(e,!0),c=a.xOrigin||0,u=a.yOrigin||0,f=a.xOffset||0,h=a.yOffset||0,d=l[0],x=l[1],p=l[2],g=l[3],m=l[4],_=l[5],S=t.split(" "),A=parseFloat(S[0])||0,y=parseFloat(S[1])||0,b,v,E,M;n?l!==Ia&&(v=d*g-x*p)&&(E=A*(g/v)+y*(-p/v)+(p*_-g*m)/v,M=A*(-x/v)+y*(d/v)-(d*_-x*m)/v,A=E,y=M):(b=Sx(e),A=b.x+(~S[0].indexOf("%")?A/100*b.width:A),y=b.y+(~(S[1]||S[0]).indexOf("%")?y/100*b.height:y)),s||s!==!1&&a.smooth?(m=A-c,_=y-u,a.xOffset=f+(m*d+_*p)-m,a.yOffset=h+(m*x+_*g)-_):a.xOffset=a.yOffset=0,a.xOrigin=A,a.yOrigin=y,a.smooth=!!s,a.origin=t,a.originIsAbsolute=!!n,e.style[zn]="0px 0px",o&&(Us(o,a,"xOrigin",c,A),Us(o,a,"yOrigin",u,y),Us(o,a,"xOffset",f,a.xOffset),Us(o,a,"yOffset",h,a.yOffset)),e.setAttribute("data-svg-origin",A+" "+y)},Da=function(e,t){var n=e._gsap||new ox(e);if("x"in n&&!t&&!n.uncache)return n;var s=e.style,r=n.scaleX<0,o="px",a="deg",l=getComputedStyle(e),c=Zn(e,zn)||"0",u,f,h,d,x,p,g,m,_,S,A,y,b,v,E,M,T,I,P,B,N,G,V,q,X,ee,ce,be,Re,Fe,Oe,Ne;return u=f=h=p=g=m=_=S=A=0,d=x=1,n.svg=!!(e.getCTM&&Ax(e)),l.translate&&((l.translate!=="none"||l.scale!=="none"||l.rotate!=="none")&&(s[Nt]=(l.translate!=="none"?"translate3d("+(l.translate+" 0 0").split(" ").slice(0,3).join(", ")+") ":"")+(l.rotate!=="none"?"rotate("+l.rotate+") ":"")+(l.scale!=="none"?"scale("+l.scale.split(" ").join(",")+") ":"")+(l[Nt]!=="none"?l[Nt]:"")),s.scale=s.rotate=s.translate="none"),v=ld(e,n.svg),n.svg&&(n.uncache?(X=e.getBBox(),c=n.xOrigin-X.x+"px "+(n.yOrigin-X.y)+"px",q=""):q=!t&&e.getAttribute("data-svg-origin"),Kf(e,q||c,!!q||n.originIsAbsolute,n.smooth!==!1,v)),y=n.xOrigin||0,b=n.yOrigin||0,v!==Ia&&(I=v[0],P=v[1],B=v[2],N=v[3],u=G=v[4],f=V=v[5],v.length===6?(d=Math.sqrt(I*I+P*P),x=Math.sqrt(N*N+B*B),p=I||P?Qr(P,I)*hr:0,_=B||N?Qr(B,N)*hr+p:0,_&&(x*=Math.abs(Math.cos(_*go))),n.svg&&(u-=y-(y*I+b*B),f-=b-(y*P+b*N))):(Ne=v[6],Fe=v[7],ce=v[8],be=v[9],Re=v[10],Oe=v[11],u=v[12],f=v[13],h=v[14],E=Qr(Ne,Re),g=E*hr,E&&(M=Math.cos(-E),T=Math.sin(-E),q=G*M+ce*T,X=V*M+be*T,ee=Ne*M+Re*T,ce=G*-T+ce*M,be=V*-T+be*M,Re=Ne*-T+Re*M,Oe=Fe*-T+Oe*M,G=q,V=X,Ne=ee),E=Qr(-B,Re),m=E*hr,E&&(M=Math.cos(-E),T=Math.sin(-E),q=I*M-ce*T,X=P*M-be*T,ee=B*M-Re*T,Oe=N*T+Oe*M,I=q,P=X,B=ee),E=Qr(P,I),p=E*hr,E&&(M=Math.cos(E),T=Math.sin(E),q=I*M+P*T,X=G*M+V*T,P=P*M-I*T,V=V*M-G*T,I=q,G=X),g&&Math.abs(g)+Math.abs(p)>359.9&&(g=p=0,m=180-m),d=Vt(Math.sqrt(I*I+P*P+B*B)),x=Vt(Math.sqrt(V*V+Ne*Ne)),E=Qr(G,V),_=Math.abs(E)>2e-4?E*hr:0,A=Oe?1/(Oe<0?-Oe:Oe):0),n.svg&&(q=e.getAttribute("transform"),n.forceCSS=e.setAttribute("transform","")||!bx(Zn(e,Nt)),q&&e.setAttribute("transform",q))),Math.abs(_)>90&&Math.abs(_)<270&&(r?(d*=-1,_+=p<=0?180:-180,p+=p<=0?180:-180):(x*=-1,_+=_<=0?180:-180)),t=t||n.uncache,n.x=u-((n.xPercent=u&&(!t&&n.xPercent||(Math.round(e.offsetWidth/2)===Math.round(-u)?-50:0)))?e.offsetWidth*n.xPercent/100:0)+o,n.y=f-((n.yPercent=f&&(!t&&n.yPercent||(Math.round(e.offsetHeight/2)===Math.round(-f)?-50:0)))?e.offsetHeight*n.yPercent/100:0)+o,n.z=h+o,n.scaleX=Vt(d),n.scaleY=Vt(x),n.rotation=Vt(p)+a,n.rotationX=Vt(g)+a,n.rotationY=Vt(m)+a,n.skewX=_+a,n.skewY=S+a,n.transformPerspective=A+o,(n.zOrigin=parseFloat(c.split(" ")[2])||!t&&n.zOrigin||0)&&(s[zn]=nc(c)),n.xOffset=n.yOffset=0,n.force3D=ei.force3D,n.renderTransform=n.svg?Iw:vx?Mx:Rw,n.uncache=0,n},nc=function(e){return(e=e.split(" "))[0]+" "+e[1]},Pu=function(e,t,n){var s=fn(t);return Vt(parseFloat(t)+parseFloat(Ys(e,"x",n+"px",s)))+s},Rw=function(e,t){t.z="0px",t.rotationY=t.rotationX="0deg",t.force3D=0,Mx(e,t)},lr="0deg",Xo="0px",cr=") ",Mx=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.z,c=n.rotation,u=n.rotationY,f=n.rotationX,h=n.skewX,d=n.skewY,x=n.scaleX,p=n.scaleY,g=n.transformPerspective,m=n.force3D,_=n.target,S=n.zOrigin,A="",y=m==="auto"&&e&&e!==1||m===!0;if(S&&(f!==lr||u!==lr)){var b=parseFloat(u)*go,v=Math.sin(b),E=Math.cos(b),M;b=parseFloat(f)*go,M=Math.cos(b),o=Pu(_,o,v*M*-S),a=Pu(_,a,-Math.sin(b)*-S),l=Pu(_,l,E*M*-S+S)}g!==Xo&&(A+="perspective("+g+cr),(s||r)&&(A+="translate("+s+"%, "+r+"%) "),(y||o!==Xo||a!==Xo||l!==Xo)&&(A+=l!==Xo||y?"translate3d("+o+", "+a+", "+l+") ":"translate("+o+", "+a+cr),c!==lr&&(A+="rotate("+c+cr),u!==lr&&(A+="rotateY("+u+cr),f!==lr&&(A+="rotateX("+f+cr),(h!==lr||d!==lr)&&(A+="skew("+h+", "+d+cr),(x!==1||p!==1)&&(A+="scale("+x+", "+p+cr),_.style[Nt]=A||"translate(0, 0)"},Iw=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.rotation,c=n.skewX,u=n.skewY,f=n.scaleX,h=n.scaleY,d=n.target,x=n.xOrigin,p=n.yOrigin,g=n.xOffset,m=n.yOffset,_=n.forceCSS,S=parseFloat(o),A=parseFloat(a),y,b,v,E,M;l=parseFloat(l),c=parseFloat(c),u=parseFloat(u),u&&(u=parseFloat(u),c+=u,l+=u),l||c?(l*=go,c*=go,y=Math.cos(l)*f,b=Math.sin(l)*f,v=Math.sin(l-c)*-h,E=Math.cos(l-c)*h,c&&(u*=go,M=Math.tan(c-u),M=Math.sqrt(1+M*M),v*=M,E*=M,u&&(M=Math.tan(u),M=Math.sqrt(1+M*M),y*=M,b*=M)),y=Vt(y),b=Vt(b),v=Vt(v),E=Vt(E)):(y=f,E=h,b=v=0),(S&&!~(o+"").indexOf("px")||A&&!~(a+"").indexOf("px"))&&(S=Ys(d,"x",o,"px"),A=Ys(d,"y",a,"px")),(x||p||g||m)&&(S=Vt(S+x-(x*y+p*v)+g),A=Vt(A+p-(x*b+p*E)+m)),(s||r)&&(M=d.getBBox(),S=Vt(S+s/100*M.width),A=Vt(A+r/100*M.height)),M="matrix("+y+","+b+","+v+","+E+","+S+","+A+")",d.setAttribute("transform",M),_&&(d.style[Nt]=M)},Dw=function(e,t,n,s,r){var o=360,a=nn(r),l=parseFloat(r)*(a&&~r.indexOf("rad")?hr:1),c=l-s,u=s+c+"deg",f,h;return a&&(f=r.split("_")[1],f==="short"&&(c%=o,c!==c%(o/2)&&(c+=c<0?o:-o)),f==="cw"&&c<0?c=(c+o*Tm)%o-~~(c/o)*o:f==="ccw"&&c>0&&(c=(c-o*Tm)%o-~~(c/o)*o)),e._pt=h=new Nn(e._pt,t,n,s,c,dw),h.e=u,h.u="deg",e._props.push(n),h},Pm=function(e,t){for(var n in t)e[n]=t[n];return e},Pw=function(e,t,n){var s=Pm({},n._gsap),r="perspective,force3D,transformOrigin,svgOrigin",o=n.style,a,l,c,u,f,h,d,x;s.svg?(c=n.getAttribute("transform"),n.setAttribute("transform",""),o[Nt]=t,a=Da(n,1),qs(n,Nt),n.setAttribute("transform",c)):(c=getComputedStyle(n)[Nt],o[Nt]=t,a=Da(n,1),o[Nt]=c);for(l in Ss)c=s[l],u=a[l],c!==u&&r.indexOf(l)<0&&(d=fn(c),x=fn(u),f=d!==x?Ys(n,l,c,x):parseFloat(c),h=parseFloat(u),e._pt=new Nn(e._pt,a,l,f,h-f,qf),e._pt.u=x||0,e._props.push(l));Pm(a,s)};On("padding,margin,Width,Radius",function(i,e){var t="Top",n="Right",s="Bottom",r="Left",o=(e<3?[t,n,s,r]:[t+r,t+n,s+n,s+r]).map(function(a){return e<2?i+a:"border"+a+i});tc[e>1?"border"+i:i]=function(a,l,c,u,f){var h,d;if(arguments.length<4)return h=o.map(function(x){return cs(a,x,c)}),d=h.join(" "),d.split(h[0]).length===5?h[0]:d;h=(u+"").split(" "),d={},o.forEach(function(x,p){return d[x]=h[p]=h[p]||h[(p-1)/2|0]}),a.init(l,d,f)}});var Tx={name:"css",register:Qf,targetTest:function(e){return e.style&&e.nodeType},init:function(e,t,n,s,r){var o=this._props,a=e.style,l=n.vars.startAt,c,u,f,h,d,x,p,g,m,_,S,A,y,b,v,E,M;rd||Qf(),this.styles=this.styles||_x(e),E=this.styles.props,this.tween=n;for(p in t)if(p!=="autoRound"&&(u=t[p],!(Yn[p]&&ax(p,t,n,s,e,r)))){if(d=typeof u,x=tc[p],d==="function"&&(u=u.call(n,s,e,r),d=typeof u),d==="string"&&~u.indexOf("random(")&&(u=Ea(u)),x)x(this,e,p,u,n)&&(v=1);else if(p.substr(0,2)==="--")c=(getComputedStyle(e).getPropertyValue(p)+"").trim(),u+="",Hs.lastIndex=0,Hs.test(c)||(g=fn(c),m=fn(u),m?g!==m&&(c=Ys(e,p,c,m)+m):g&&(u+=g)),this.add(a,"setProperty",c,u,s,r,0,0,p),o.push(p),E.push(p,0,a[p]);else if(d!=="undefined"){if(l&&p in l?(c=typeof l[p]=="function"?l[p].call(n,s,e,r):l[p],nn(c)&&~c.indexOf("random(")&&(c=Ea(c)),fn(c+"")||c==="auto"||(c+=ei.units[p]||fn(cs(e,p))||""),(c+"").charAt(1)==="="&&(c=cs(e,p))):c=cs(e,p),h=parseFloat(c),_=d==="string"&&u.charAt(1)==="="&&u.substr(0,2),_&&(u=u.substr(2)),f=parseFloat(u),p in ki&&(p==="autoAlpha"&&(h===1&&cs(e,"visibility")==="hidden"&&f&&(h=0),E.push("visibility",0,a.visibility),Us(this,a,"visibility",h?"inherit":"hidden",f?"inherit":"hidden",!f)),p!=="scale"&&p!=="transform"&&(p=ki[p],~p.indexOf(",")&&(p=p.split(",")[0]))),S=p in Ss,S){if(this.styles.save(p),M=u,d==="string"&&u.substring(0,6)==="var(--"){if(u=Zn(e,u.substring(4,u.indexOf(")"))),u.substring(0,5)==="calc("){var T=e.style.perspective;e.style.perspective=u,u=Zn(e,"perspective"),T?e.style.perspective=T:qs(e,"perspective")}f=parseFloat(u)}if(A||(y=e._gsap,y.renderTransform&&!t.parseTransform||Da(e,t.parseTransform),b=t.smoothOrigin!==!1&&y.smooth,A=this._pt=new Nn(this._pt,a,Nt,0,1,y.renderTransform,y,0,-1),A.dep=1),p==="scale")this._pt=new Nn(this._pt,y,"scaleY",y.scaleY,(_?po(y.scaleY,_+f):f)-y.scaleY||0,qf),this._pt.u=0,o.push("scaleY",p),p+="X";else if(p==="transformOrigin"){E.push(zn,0,a[zn]),u=Ew(u),y.svg?Kf(e,u,0,b,0,this):(m=parseFloat(u.split(" ")[2])||0,m!==y.zOrigin&&Us(this,y,"zOrigin",y.zOrigin,m),Us(this,a,p,nc(c),nc(u)));continue}else if(p==="svgOrigin"){Kf(e,u,1,b,0,this);continue}else if(p in yx){Dw(this,y,p,h,_?po(h,_+u):u);continue}else if(p==="smoothOrigin"){Us(this,y,"smooth",y.smooth,u);continue}else if(p==="force3D"){y[p]=u;continue}else if(p==="transform"){Pw(this,u,e);continue}}else p in a||(p=Po(p)||p);if(S||(f||f===0)&&(h||h===0)&&!hw.test(u)&&p in a)g=(c+"").substr((h+"").length),f||(f=0),m=fn(u)||(p in ei.units?ei.units[p]:g),g!==m&&(h=Ys(e,p,c,m)),this._pt=new Nn(this._pt,S?y:a,p,h,(_?po(h,_+f):f)-h,!S&&(m==="px"||p==="zIndex")&&t.autoRound!==!1?gw:qf),this._pt.u=m||0,S&&M!==u?(this._pt.b=c,this._pt.e=M,this._pt.r=mw):g!==m&&m!=="%"&&(this._pt.b=c,this._pt.r=pw);else if(p in a)Cw.call(this,e,p,c,_?_+u:u);else if(p in e)this.add(e,p,c||e[p],_?_+u:u,s,r);else if(p!=="parseTransform"){Kh(p,u);continue}S||(p in a?E.push(p,0,a[p]):typeof e[p]=="function"?E.push(p,2,e[p]()):E.push(p,1,c||e[p])),o.push(p)}}v&&dx(this)},render:function(e,t){if(t.tween._time||!od())for(var n=t._pt;n;)n.r(e,n.d),n=n._next;else t.styles.revert()},get:cs,aliases:ki,getSetter:function(e,t,n){var s=ki[t];return s&&s.indexOf(",")<0&&(t=s),t in Ss&&t!==zn&&(e._gsap.x||cs(e,"x"))?n&&Mm===n?t==="scale"?Sw:vw:(Mm=n||{})&&(t==="scale"?Aw:yw):e.style&&!qh(e.style[t])?xw:~t.indexOf("-")?_w:id(e,t)},core:{_removeProperty:qs,_getMatrix:ld}};kn.utils.checkPrefix=Po;kn.core.getStyleSaver=_x;(function(i,e,t,n){var s=On(i+","+e+","+t,function(r){Ss[r]=1});On(e,function(r){ei.units[r]="deg",yx[r]=1}),ki[s[13]]=i+","+e,On(n,function(r){var o=r.split(":");ki[o[1]]=s[o[0]]})})("x,y,z,scale,scaleX,scaleY,xPercent,yPercent","rotation,rotationX,rotationY,skewX,skewY","transform,transformOrigin,svgOrigin,force3D,smoothOrigin,transformPerspective","0:translateX,1:translateY,2:translateZ,8:rotate,8:rotationZ,8:rotateZ,9:rotateX,10:rotateY");On("x,y,z,top,right,bottom,left,width,height,fontSize,padding,margin,perspective",function(i){ei.units[i]="px"});kn.registerPlugin(Tx);var Jr=kn.registerPlugin(Tx)||kn;Jr.core.Tween;const Fw=(i,e)=>{const t=i.__vccOpts||i;for(const[n,s]of e)t[n]=s;return t},Lw={class:"top-hud"},Bw={class:"top-actions"},Uw={key:0,class:"fps-counter"},Ow={key:0,class:"loading-overlay"},Nw={key:1,class:"error-overlay"},zw={class:"error-card"},kw={class:"error-msg"},Hw=["min","max"],Vw={class:"focal-row"},Gw=["min","max"],Ww={class:"focal-row"},Xw={class:"focal-row"},qw={class:"camera-track-header"},Yw={class:"camera-track-copy"},Qw=["onClick"],Kw=["src"],jw={key:1,class:"camera-tag-overlay"},$w={class:"camera-tag-text"},Zw={key:2},Jw=["src"],e3={key:0,class:"ref-info"},t3={class:"info-tag info-tag--accent"},n3={key:1,class:"ref-info"},i3={class:"info-tag"},s3={class:"info-tag"},r3={class:"info-tag"},Kr=380,Fm=.065,Lm=.0022,Bm=.08,o3=1,a3={__name:"GaussianViewer",setup(i){const e=Xt(null),t=Xt(!1),n=Xt(!1),s=Xt(!1),r={FREE:"free",ORBIT:"orbit"},o=Xt(r.FREE),a=Xt([]),l=Xt(""),c=Xt(""),u=Xt(""),f=Xt({}),h=Xt({x:0,y:0,z:0}),d=Xt({x:0,y:0,z:0}),x=Xt(""),p=Xt(0),g=Xt(!1),m=Xt(0),_=Xt(0),S=Xt(null),A=Yo(()=>o.value===r.ORBIT),y=Yo(()=>{if(!l.value.trim()){const z=a.value.filter(he=>he.tag);return z.length>0?z:a.value.slice(0,60)}const H=l.value.trim().toLowerCase();return a.value.filter(z=>z.tag&&z.tag.toLowerCase().includes(H))}),b=()=>{y.value.length>0?Ve(y.value[0]):alert("场景中没有找到符合该描述的视角哦~")};let v,E;const M=Xt({x:0,y:0}),T=(H,z)=>!H||!z?null:2*Math.atan(z/2/H)*(180/Math.PI),I=(H,z)=>{if(!H||!z)return null;const he=H*Math.PI/180/2;return he<=0?null:z/2/Math.tan(he)},P=()=>{if(!v||!v.camera)return;const H=f.value.h||e.value?.clientHeight||window.innerHeight;if(m.value=Number(v.camera.fov||0),H&&m.value>0&&m.value<179){const z=I(m.value,H);_.value=z?Number(z.toFixed(1)):0}},B=(H,z={})=>{if(!v||!v.camera)return;const he=f.value.h||e.value?.clientHeight||window.innerHeight;if(!he||!H)return;const Me=T(H,he);if(!Me||!Number.isFinite(Me))return;const O=v.camera,ve=z.duration??0;if(ve>0)Jr.to(O,{fov:Me,duration:ve,ease:z.ease||"power2.out",onUpdate:()=>{O.updateProjectionMatrix();try{v.update(),v.render()}catch{}P()}});else{O.fov=Me,O.updateProjectionMatrix();try{v.update(),v.render()}catch{}P()}},N=H=>Number.isFinite(H)?Math.min(X.value,Math.max(q.value,H)):null,G=()=>{const H=Number(S.value||_.value||f.value.fl_y||Kr);return N(H)},V=H=>{if(!v||!v.camera||!Number.isFinite(H)||H<=0)return;const z=G();if(!z)return;const he=N(z*H);he&&(S.value=Number(he.toFixed(1)),B(he))},q=Yo(()=>{const H=Number(f.value.fl_y||0);return H>0?Math.max(50,Math.floor(H*.4)):50}),X=Yo(()=>{const H=Number(f.value.fl_y||0);return H>0?Math.max(500,Math.ceil(H*2.5)):3e3}),ee=()=>{g.value=!g.value,g.value&&!S.value&&(S.value=Number((_.value||f.value.fl_y||Kr).toFixed(1)))},ce=()=>{const H=Number(S.value);!Number.isFinite(H)||H<=0||B(H)},be=()=>{const H=Number(f.value.fl_y||0);H&&(S.value=Number(H.toFixed(1)),B(H,{duration:.5,ease:"power2.inOut"}))},Re=()=>{if(!v||!v.camera)return;const H=new Ei().setFromQuaternion(v.camera.quaternion,"YXZ");h.value={x:(H.x*180/Math.PI).toFixed(1),y:(H.y*180/Math.PI).toFixed(1),z:(H.z*180/Math.PI).toFixed(1)},P()},Fe=()=>xe.uCenter.value.clone(),Oe=()=>{if(!v||!v.camera)return new F(0,0,0);const H=Fe(),z=v.camera.position.distanceTo(H),he=new F(0,0,-1).applyQuaternion(v.camera.quaternion),Me=Number.isFinite(z)&&z>0?z*.6:3;return H.distanceTo(v.camera.position)>.001?H:v.camera.position.clone().add(he.multiplyScalar(Me))},Ne=(H=null)=>{if(!v||!v.controls||!A.value)return;const z=H?H.clone():Oe();v.controls.target.copy(z),v.controls.update()},J={FLY_IN:0,DIFFUSION:1,COLORING:2,FINISHED:3},ne={isLoaded:!1,lastFrameTime:0,phase:J.FLY_IN,flyDuration:1.5,diffusionDuration:1,colorDuration:4},xe={uTime:{value:0},uCenter:{value:new F(0,0,0)},uGeoRadius:{value:0},uColorRadius:{value:0},uMaxRadius:{value:50},uParticleProgress:{value:0}},Be=H=>{if(!v)return;const z=H.getSplatCount();H.updateMatrixWorld();let he=1/0,Me=1/0,O=1/0,ve=-1/0,ge=-1/0,Se=-1/0;const de=new F,le=Math.max(1,Math.floor(z/1e3));for(let ii=0;ii<z;ii+=le)H.getSplatCenter(ii,de),de.applyMatrix4(H.matrixWorld),de.x<he&&(he=de.x),de.x>ve&&(ve=de.x),de.y<Me&&(Me=de.y),de.y>ge&&(ge=de.y),de.z<O&&(O=de.z),de.z>Se&&(Se=de.z);const Ce=(he+ve)/2,ze=(Me+ge)/2,it=(O+Se)/2,Ge=Math.max(ve-he,ge-Me,Se-O);xe.uCenter.value.set(Ce,ze,it),xe.uMaxRadius.value=Ge*.7;let vt=6e4;z<4e4?vt=z:z>1e6&&(vt=4e5);const Ct=Math.ceil(z/vt);let wi=Ge/200*window.devicePixelRatio;wi<.5&&(wi=.5);const za=Ge*1;console.log(`[Adaptive] MaxDim: ${Ge.toFixed(2)}, Particles: ~${Math.floor(z/Ct)}, Size: ${wi.toFixed(2)}`);const $s=new En,Ri=[],Er=[],Bo=[];for(let ii=0;ii<z;ii+=Ct){H.getSplatCenter(ii,de),de.applyMatrix4(H.matrixWorld),Er.push(de.x,de.y,de.z);const wr=za+Math.random()*(Ge*.5),Zs=Math.random()*Math.PI*2,Rr=Math.acos(2*Math.random()-1),Ha=Ce+wr*Math.sin(Rr)*Math.cos(Zs),yc=ze+wr*Math.sin(Rr)*Math.sin(Zs),bc=it+wr*Math.cos(Rr);Ri.push(Ha,yc,bc),Bo.push(Math.random())}$s.setAttribute("position",new dn(Ri,3)),$s.setAttribute("aTarget",new dn(Er,3)),$s.setAttribute("aRandom",new dn(Bo,1));const ka=new Cn({uniforms:{uProgress:xe.uParticleProgress,uSize:{value:wi},uColor:{value:new rt(.6,.6,.6)}},vertexShader:`
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
    `,transparent:!0,opacity:1,depthTest:!0,depthWrite:!1});E=new EA($s,ka),E.frustumCulled=!1,v.threeScene.add(E)},Te=H=>{if(!H||!H.material)return;const z=H.material;z.uniforms=z.uniforms||{},z.uniforms.uGeoRadius=xe.uGeoRadius,z.uniforms.uColorRadius=xe.uColorRadius,z.uniforms.uMaxRadius=xe.uMaxRadius,z.uniforms.uCenter=xe.uCenter,z.vertexShader=`varying vec3 vWorldPosition;
`+z.vertexShader;const he=z.vertexShader.lastIndexOf("}");if(he!==-1){const ve=`vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
`;z.vertexShader=z.vertexShader.substring(0,he)+ve+"}"}const Me=`
    uniform float uGeoRadius;
    uniform float uColorRadius;
    uniform float uMaxRadius;
    uniform vec3 uCenter;
    varying vec3 vWorldPosition;
  `;z.fragmentShader=Me+z.fragmentShader;const O=z.fragmentShader.lastIndexOf("}");if(O!==-1){const ve=z.fragmentShader.substring(0,O),ge=`
      float distFromCenter = distance(vWorldPosition, uCenter);
      
      if (distFromCenter > uGeoRadius) {
          discard;
      }
      if (distFromCenter > uColorRadius) {
          if (gl_FragColor.a < 0.8) discard; 
          gl_FragColor.a = 1.0; 
          gl_FragColor.rgb = vec3(0.6, 0.6, 0.6);
      } 
    `;z.fragmentShader=ve+ge+"}"}z.needsUpdate=!0},Ve=H=>{if(!v||!v.camera)return;const z=v.camera,he=v.getSplatMesh();c.value=H.image_url,u.value=H.tag||"";const Me=new Ye().fromArray(H.matrix),O=new Ye;he?(he.updateMatrixWorld(),O.copy(he.matrixWorld).multiply(Me)):O.copy(Me);const ve=new F,ge=new Mt,Se=new F;O.decompose(ve,ge,Se);const de=H.fl_y||f.value.fl_y,le=H.h||f.value.h;de&&le&&(f.value.h=le,S.value=Number(de.toFixed(1)),B(de,{duration:1.5,ease:"power3.inOut"})),z.near>.001&&(z.near=.001,z.updateProjectionMatrix());const Ce=new F(0,0,-1).applyQuaternion(ge),ze=ve.clone().add(Ce.multiplyScalar(5)),it=A.value?Fe():ze.clone();t.value=!1,v.controls&&(v.controls.enabled=!1);const Ge=z.position.clone(),vt=z.quaternion.clone(),Ct={t:0};Jr.killTweensOf(z.position),Jr.killTweensOf(z.quaternion),Jr.killTweensOf(Ct),Jr.to(Ct,{t:1,duration:1.5,ease:"power3.inOut",onUpdate:()=>{z.position.lerpVectors(Ge,ve,Ct.t),z.quaternion.slerpQuaternions(vt,ge,Ct.t)},onComplete:()=>{const wi=new Ei().setFromQuaternion(z.quaternion,"YXZ");d.value={x:(wi.x*180/Math.PI).toFixed(1),y:(wi.y*180/Math.PI).toFixed(1),z:(wi.z*180/Math.PI).toFixed(1)},M.value={x:0,y:0},Re(),Ne(it),v.controls&&(v.controls.enabled=!0)}})},L=()=>{const H=/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);return{rootElement:e.value,cameraUp:[0,1,0],initialCameraPosition:[0,0,5],initialCameraLookAt:[0,0,0],useBuiltInControls:!1,gpuAcceleratedSort:!1,webXRMode:fr.None,sharedMemoryForWorkers:!1,antialiased:!H}};let U="/models/scene_auto_sync.ply",Y="/models/webgl_poses_with_tags.json",w=!1;const oe=()=>{const H=new URLSearchParams(window.location.search),z=H.get("payload");if(z)try{const ge=JSON.parse(decodeURIComponent(z));return{ply:ge.ply||null,poses:ge.poses||null,matrix:ge.matrix||null}}catch(ge){console.warn("[Viewer] 无法解析 payload 查询参数:",ge)}const he=H.get("ply"),Me=H.get("poses"),O=H.get("matrix");let ve=null;if(O)try{ve=JSON.parse(decodeURIComponent(O))}catch(ge){console.warn("[Viewer] 无法解析 matrix 查询参数:",ge)}return he||Me||ve?{ply:he||null,poses:Me||null,matrix:ve}:null},re=async(H,z,he)=>{if(!n.value){n.value=!0,H&&(U=H),z&&(Y=z);try{v&&(v.renderer.setAnimationLoop(null),v.dispose&&await v.dispose(),v=null),e.value&&(e.value.innerHTML=""),ne.isLoaded=!1,ne.phase=J.FLY_IN,xe.uParticleProgress.value=0,xe.uGeoRadius.value=0,xe.uColorRadius.value=0;const Me=L();v=new eo(Me),window.viewer=v,S.value=Kr,console.log(`[Viewer] 加载模型: ${U}`),await v.addSplatScene(U,{showLoadingUI:!0,progressiveLoad:!1,rotation:[0,0,0,1]}),n.value=!1,window.BrainDanceChannel&&window.BrainDanceChannel.postMessage(JSON.stringify({status:"success",msg:"模型加载完成"})),console.log(`[Viewer] 加载位姿: ${Y}`),fetch(Y).then(le=>le.json()).then(le=>{le.frames?(f.value={w:le.w,h:le.h,fl_x:le.fl_x,fl_y:le.fl_y},S.value=Number((le.fl_y||0).toFixed(1)),a.value=le.frames.map(Ce=>{let ze=Ce.image_url;if(ze&&!ze.startsWith("http")&&Y.startsWith("http")){const it=Y.substring(0,Y.lastIndexOf("/"));let Ge=ze;const vt=Ge.indexOf("images/");vt!==-1?Ge=Ge.substring(vt):Ge.startsWith("/models/")?Ge=Ge.substring(8):Ge.startsWith("/")&&(Ge=Ge.substring(1)),ze=`${it}/${Ge}`}return{id:Ce.id,matrix:Ce.matrix,image_url:ze,tag:Ce.tag,fl_x:Ce.fl_x,fl_y:Ce.fl_y,w:Ce.w||le.w,h:Ce.h||le.h}}),f.value.fl_y&&f.value.h?B(f.value.fl_y):B(Kr)):(a.value=le,B(Kr))}).catch(le=>{console.error("加载位姿失败:",le),B(Kr)});const O=v.getSplatMesh();O.visible=!1,setTimeout(()=>{O&&(Be(O),Te(O),he&&setTimeout(()=>{Ve({matrix:he})},50),ne.lastFrameTime=Date.now(),ne.startTime=Date.now(),ne.isLoaded=!0)},200);let ve=performance.now();const ge=1e3/120;let Se=0,de=performance.now();v.renderer.setAnimationLoop(()=>{const le=performance.now(),Ce=le-ve;if(Ce<ge||(ve=le-Ce%ge,v.update(),v.render(),Se++,le-de>=1e3&&(p.value=Se,Se=0,de=le),!ne.isLoaded||ne.phase===J.FINISHED))return;const ze=Date.now(),it=(ze-ne.lastFrameTime)/1e3||.016;if(ne.lastFrameTime=ze,ne.phase===J.FLY_IN){const Ge=1/ne.flyDuration;let vt=xe.uParticleProgress.value+it*Ge;if(vt>=1.2){vt=1.2;const Ct=v.getSplatMesh();Ct&&(Ct.visible=!0),ne.phase=J.DIFFUSION,ne.diffuseTime=0}xe.uParticleProgress.value=vt}else if(ne.phase===J.DIFFUSION){ne.diffuseTime+=it;const Ge=Math.min(ne.diffuseTime/ne.diffusionDuration,1),vt=xe.uMaxRadius.value;xe.uGeoRadius.value=Ge*(vt*1.5),E&&E.material&&(E.material.opacity=1-Ge),Ge>=1&&(E&&(E.visible=!1),xe.uGeoRadius.value=99999,ne.phase=J.COLORING,ne.colorStartTime=ze)}else if(ne.phase===J.COLORING){const Ge=(ze-ne.colorStartTime)/1e3,vt=xe.uMaxRadius.value,Ct=Ge/ne.colorDuration;xe.uColorRadius.value=Ct*(vt*1.5),Ct>=1&&(ne.phase=J.FINISHED,xe.uColorRadius.value=99999)}}),ie()}catch(Me){console.error("error:",Me),x.value=Me&&(Me.message||String(Me))||"模型加载失败，请检查模型 URL 是否正确可访问"}finally{n.value=!1}}},pe=()=>{!v||!v.controls||(v.controls.dispose(),v.controls=null)},se=()=>{v&&pe()},me=()=>{if(!v)return;pe();const H=new n1(v.camera,v.renderer.domElement);H.enableDamping=!0,H.dampingFactor=.08,H.screenSpacePanning=!0,H.enablePan=!0,H.rotateSpeed=.8,H.zoomSpeed=.9,H.target.copy(Oe()),H.update(),v.controls=H},ie=()=>{v&&(A.value?me():se())},Ae=H=>{H!==r.FREE&&H!==r.ORBIT||o.value!==H&&(o.value=H,ie(),A.value&&Ne())},R=()=>{const H=window.location.hostname==="localhost"||window.location.hostname==="127.0.0.1",z=window.location.protocol==="https:";s.value=H||z},C=Xt(!1),W={x:0,y:0},$={active:!1,distance:0},fe=(H,z)=>{const he=H.clientX-z.clientX,Me=H.clientY-z.clientY;return Math.hypot(he,Me)},Z=H=>{A.value||(C.value=!0,$.active=!1,W.x=H.clientX,W.y=H.clientY)},Ie=H=>{if(A.value||!C.value||!v||!v.camera)return;const z=H.clientX-W.x,Me=(H.clientY-W.y)*Fm;v.camera.rotateX(Me*Math.PI/180),v.camera.translateX(-z*Lm),v.camera.updateProjectionMatrix(),Re(),W.x=H.clientX,W.y=H.clientY},ye=()=>{A.value||(C.value=!1,$.active=!1)},Ue=H=>{if(!v||!v.camera||A.value)return;const z=H.deltaY<0?1+Bm:1/(1+Bm);V(z)},k=H=>{if(!A.value){if(H.touches.length>=2){C.value=!1,$.active=!0,$.distance=fe(H.touches[0],H.touches[1]);return}$.active=!1,H.touches.length===1&&(C.value=!0,W.x=H.touches[0].clientX,W.y=H.touches[0].clientY)}},te=H=>{if(A.value||!v||!v.camera||H.touches.length===0)return;if(H.touches.length>=2){const O=fe(H.touches[0],H.touches[1]);if($.active&&$.distance>0&&O>0){const ve=O/$.distance;V(1+(ve-1)*o3)}$.active=!0,$.distance=O,C.value=!1;return}if(!C.value)return;const z=H.touches[0].clientX-W.x,Me=(H.touches[0].clientY-W.y)*Fm;M.value.x+=Me,v.camera.rotateX(Me*Math.PI/180),v.camera.translateX(-z*Lm),v.camera.updateProjectionMatrix(),Re(),W.x=H.touches[0].clientX,W.y=H.touches[0].clientY},_e=H=>{if(!A.value){if(H.touches.length>=2){$.active=!0,$.distance=fe(H.touches[0],H.touches[1]),C.value=!1;return}$.active=!1,$.distance=0,C.value=!1,H.touches.length===1&&(W.x=H.touches[0].clientX,W.y=H.touches[0].clientY,C.value=!0)}};return x0(()=>{if(e.value){if(R(),window.loadModelFromFlutter=H=>{console.log("[Flutter->WebGL] 收到加载请求:",H),typeof H=="string"?re(H,null,null):typeof H=="object"&&H!==null?re(H.ply||null,H.poses||null,H.matrix||null):re(null,null,null)},window.BrainDanceChannel)window.BrainDanceChannel.postMessage(JSON.stringify({status:"ready"}));else{const H=oe();H&&!w?(w=!0,re(H.ply,H.poses,H.matrix)):re(null,null)}window.addEventListener("mousedown",Z),window.addEventListener("mousemove",Ie),window.addEventListener("mouseup",ye)}}),_0(async()=>{window.removeEventListener("mousedown",Z),window.removeEventListener("mousemove",Ie),window.removeEventListener("mouseup",ye),v&&(v.renderer.setAnimationLoop(null),await v.dispose())}),(H,z)=>(cn(),_n("div",{class:"app-container",onMousedown:Z,onMousemove:Ie,onMouseup:ye,onWheel:Lt(Ue,["prevent"]),onMouseleave:ye,onTouchstart:k,onTouchmove:Lt(te,["prevent"]),onTouchend:_e,onTouchcancel:_e},[je("div",{ref_key:"containerRef",ref:e,class:"viewer-container"},null,512),z[37]||(z[37]=je("div",{class:"viewer-vignette"},null,-1)),je("div",Lw,[je("div",{class:"search-panel archive-card",onMousedown:z[1]||(z[1]=Lt(()=>{},["stop"])),onTouchstart:z[2]||(z[2]=Lt(()=>{},["stop"])),onTouchmove:z[3]||(z[3]=Lt(()=>{},["stop"])),onTouchend:z[4]||(z[4]=Lt(()=>{},["stop"]))},[Ic(je("input",{type:"text","onUpdate:modelValue":z[0]||(z[0]=he=>l.value=he),onKeyup:$v(b,["enter"]),placeholder:"例如：门口、桌面左侧、正面特写",class:"search-input"},null,544),[[Nc,l.value]]),je("button",{onClick:b,class:"archive-btn archive-btn--solid search-btn"},"检索视角")],32),je("div",Bw,[je("div",{class:"view-mode-switch archive-card",onMousedown:z[7]||(z[7]=Lt(()=>{},["stop"])),onTouchstart:z[8]||(z[8]=Lt(()=>{},["stop"])),onTouchmove:z[9]||(z[9]=Lt(()=>{},["stop"])),onTouchend:z[10]||(z[10]=Lt(()=>{},["stop"]))},[je("button",{class:ro(["mode-chip",{active:o.value===r.FREE}]),onClick:z[5]||(z[5]=he=>Ae(r.FREE))}," 自由模式 ",2),je("button",{class:ro(["mode-chip",{active:o.value===r.ORBIT}]),onClick:z[6]||(z[6]=he=>Ae(r.ORBIT))}," Orbit 模式 ",2)],32),je("button",{class:"archive-btn archive-btn--ghost focal-settings-toggle",onClick:ee,onMousedown:z[11]||(z[11]=Lt(()=>{},["stop"])),onTouchstart:z[12]||(z[12]=Lt(()=>{},["stop"])),onTouchend:z[13]||(z[13]=Lt(()=>{},["stop"]))},Xn(g.value?"收起焦距":"焦距设置"),33),p.value>0?(cn(),_n("div",Uw,"FPS "+Xn(p.value),1)):vi("",!0)])]),n.value?(cn(),_n("div",Ow,[...z[27]||(z[27]=[je("div",{class:"loading-card"},[je("div",{class:"loading-dot"}),je("div",{class:"loading-title"},"场景正在展开"),je("div",{class:"loading-copy"},"模型与参考镜头正在同步到工作台。")],-1)])])):vi("",!0),x.value?(cn(),_n("div",Nw,[je("div",zw,[z[28]||(z[28]=je("div",{class:"eyebrow"},"Load Failed",-1)),z[29]||(z[29]=je("div",{class:"error-title"},"模型未能正常打开",-1)),je("div",kw,Xn(x.value),1),je("button",{class:"archive-btn archive-btn--solid",onClick:z[14]||(z[14]=he=>re(Nu(U),Nu(Y),null))}," 重新载入 ")])])):vi("",!0),vi("",!0),g.value?(cn(),_n("div",{key:3,class:"focal-settings-panel",onMousedown:z[17]||(z[17]=Lt(()=>{},["stop"])),onTouchstart:z[18]||(z[18]=Lt(()=>{},["stop"])),onTouchmove:z[19]||(z[19]=Lt(()=>{},["stop"])),onTouchend:z[20]||(z[20]=Lt(()=>{},["stop"])),onTouchcancel:z[21]||(z[21]=Lt(()=>{},["stop"]))},[z[31]||(z[31]=je("div",{class:"eyebrow"},"Lens Control",-1)),z[32]||(z[32]=je("div",{class:"focal-title"},"镜头焦距",-1)),Ic(je("input",{type:"range","onUpdate:modelValue":z[15]||(z[15]=he=>S.value=he),min:q.value,max:X.value,step:"1",onInput:ce},null,40,Hw),[[Nc,S.value,void 0,{number:!0}]]),je("div",Vw,[Ic(je("input",{class:"focal-number-input",type:"number","onUpdate:modelValue":z[16]||(z[16]=he=>S.value=he),min:q.value,max:X.value,step:"1",onChange:ce},null,40,Gw),[[Nc,S.value,void 0,{number:!0}]]),z[30]||(z[30]=je("span",null,"px",-1))]),je("div",Ww,[je("span",null,"当前 FOV: "+Xn(m.value.toFixed(1))+"°",1)]),je("div",Xw,[je("span",null,"当前焦距: "+Xn(_.value.toFixed(1))+" px",1)]),je("button",{class:"archive-btn archive-btn--solid focal-reset-btn",onClick:be},"恢复拍摄焦距")],32)):vi("",!0),y.value.length>0?(cn(),_n("div",{key:4,class:"camera-track",onMousedown:z[22]||(z[22]=Lt(()=>{},["stop"])),onTouchstart:z[23]||(z[23]=Lt(()=>{},["stop"])),onTouchmove:z[24]||(z[24]=Lt(()=>{},["stop"])),onTouchend:z[25]||(z[25]=Lt(()=>{},["stop"]))},[je("div",qw,[z[33]||(z[33]=je("div",{class:"eyebrow"},"Shot Strip",-1)),je("div",Yw,Xn(l.value?"按当前检索结果排序":"优先显示已打标签镜头"),1)]),(cn(!0),_n(Fi,null,z_(y.value,(he,Me)=>(cn(),_n("div",{key:he.id,class:ro(["camera-btn",{active:c.value===he.image_url}]),onClick:Lt(O=>Ve(he),["stop"])},[he.image_url?(cn(),_n("img",{key:0,src:he.image_url,class:"btn-thumb"},null,8,Kw)):vi("",!0),he.tag?(cn(),_n("div",jw,[je("div",$w,Xn(he.tag),1)])):he.image_url?vi("",!0):(cn(),_n("span",Zw,"未命名视角"))],10,Qw))),128))],32)):vi("",!0),c.value?(cn(),_n("div",{key:5,class:"reference-overlay",onClick:z[26]||(z[26]=he=>{c.value="",u.value=""})},[z[34]||(z[34]=je("div",{class:"eyebrow"},"Reference Still",-1)),z[35]||(z[35]=je("div",{class:"ref-title"},"参考原图",-1)),je("img",{src:c.value,class:"ref-img"},null,8,Jw),u.value?(cn(),_n("div",e3,[je("span",t3,Xn(u.value),1)])):vi("",!0),f.value.fl_y?(cn(),_n("div",n3,[je("span",i3,"焦距: "+Xn(f.value.fl_y.toFixed(1))+" px",1),je("span",s3,"FOV: "+Xn((2*Math.atan(f.value.h/(2*f.value.fl_y))*(180/Math.PI)).toFixed(1))+"°",1),je("span",r3,"分辨率: "+Xn(f.value.w)+"x"+Xn(f.value.h),1)])):vi("",!0),z[36]||(z[36]=je("div",{class:"ref-hint"},"点击关闭对比",-1))])):vi("",!0)],32))}},l3=Fw(a3,[["__scopeId","data-v-b649d438"]]),c3={__name:"App",setup(i){return(e,t)=>(cn(),_n("main",null,[Vi(l3)]))}};eS(c3).mount("#app");
