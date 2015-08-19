<?php
namespace library\dom\elements\components;
use library\mediator\nodeElement;
use library\dom\object;

/**
 * Description of text
 *
 * @author durso
 */
class text extends object{
    public function __construct($value) {
        parent::__construct();
        $this->html = $value;
    }
    public function hasId(){
        return false;
    }
    public function getValue(){
        return $this->html;
    }
    public function setValue($value){
        $this->html = $value;
        if($this->isRendered){
            $this->updateJS('changeText', $value,$this->siblingsIndex());
        }
    }
    public function removeValue(){
        $this->html = "";
        if($this->isRendered){
            $this->updateJS('removeText', $this->html,$this->siblingsIndex());
        }
    }
    
    public function append($value){
        $this->html .= $value;
        if($this->isRendered){
            $this->updateJS('appendText', $value,$this->siblingsIndex());
        }
    }
    public function getUid(){
        $uid = nodeElement::getParent($this)->getUid();
        if($uid === false){
            throw new \Exception("Text node '".$this->html."' has no parent");
        }
        return $uid;
    }
    
    public function is($arg){
        return false;
    }


}
