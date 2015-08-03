<?php
namespace library\dom\elements\components;
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
            $this->updateJS('changeText', $value);
        }
    }
    public function getUid(){
        return $this->node->getParent()->getValue()->getUid();
    }
    public function is($arg){
        return false;
    }

}
